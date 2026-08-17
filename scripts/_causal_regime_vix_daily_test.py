"""Two changes at once, both motivated by the "Filtering vs. Smoothing"
detection-lag literature found today:

1. Regime input is log(VIX LEVEL) on DAILY data, not ACWI/AGG returns --
   VIX is a leading, already-mean-reverting volatility measure; feeding its
   level (not its own day-to-day change) is the standard treatment. Fit
   uses "panel" mode (single series, gets LazyStats' built-in
   standardization) refit causally at every fold from data <= that date.

2. Rebalance-on-regime-change instead of calendar-driven: the tree's own
   monthly fold schedule is kept (re-estimating the whole hierarchy is the
   expensive part), but a fold only triggers a NEW estimation + rebalance
   when the causal VIX regime state actually differs from the state at the
   last rebalance. Otherwise the ledger just keeps drifting on the
   previously-set weights -- no needless monthly churn chasing a signal
   that hasn't moved.
"""
from __future__ import annotations
import json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [
    str(ROOT / "src"), str(ROOT / "project"),
    str(ROOT.parent / "LazyStats" / "src"), str(ROOT.parent / "market-data-hub"),
]
import tree_studio  # noqa: E402
from lazystats.regimes.core import MSRegimeEngine  # noqa: E402
from market_data_hub import extract  # noqa: E402
from lazyportfolio.calendar import _annualization_factor  # noqa: E402
from lazyportfolio.models import BacktestSpec  # noqa: E402
from lazyportfolio.walk_forward import prepare_walk_forward_inputs  # noqa: E402
from lazyportfolio.v2.backtest import HierarchicalV2Backtester, _V2Ledger  # noqa: E402
from lazyportfolio.v2.hierarchy import HierarchicalV2Estimator  # noqa: E402
from lazyportfolio.v2.model import V2Model  # noqa: E402
from lazyportfolio.v2 import store  # noqa: E402
from pruning_runner import significance_report, sharpe_significance_report  # noqa: E402

BURN_IN_YEARS = 2.0


def load_vix_log_level(start: str, end: str) -> pd.Series:
    df, _meta = extract.extract_series(["^VIX"], transform="level", frequency="D", start=start, end=end)
    return np.log(df["^VIX"]).rename("V")


def fit_vix_regime_causal(vix_log: pd.Series, as_of) -> tuple[float, str]:
    """Fit fresh on VIX log-level data <= as_of only. Returns (P(high vol)
    today, today's argmax state label) -- used both for the blend weight
    and to decide whether a regime change happened since the last trigger.
    """
    window = vix_log.loc[:as_of].to_frame("V")
    engine = MSRegimeEngine(S_max=2, S_min=2, criterion="bic", n_starts=20, random_state=7, standardize=True)
    run = engine.fit(window, model="panel")
    p_high_today = float(run.panel["P_V_HV"].iloc[-1])
    state_today = str(int(run.panel["V_state"].iloc[-1]))
    return p_high_today, state_today


def main(max_folds: int | None = None, workers: int = 4) -> None:
    tree_name = "Global Multi-Asset"
    config = store.read_model(tree_name)
    model, dataset = tree_studio._v2_inputs(config)
    bt, mode = config["backtest"], tree_studio._v2_mode(config)
    train_size, freq = int(bt.get("train_size") or 104), str(bt.get("estimation_frequency") or "W")
    instruments = list(dict.fromkeys(
        [*model.root.terminal_instruments(), *(n.proxy for n in model.root.walk() if n.proxy), *model.benchmark.weights]
    ))
    ppy = _annualization_factor(freq)

    print("Running proper walk-forward baseline (reused as STATIC_FINAL everywhere else today)...")
    _, _, reference_report = tree_studio._run_full_backtest(
        config, capture_audit_series=False, max_workers=4, expanding=False
    )
    baseline_curve = reference_report.curves["FINAL"]
    burn_in_cutoff = reference_report.folds[0].holding_start + pd.DateOffset(years=BURN_IN_YEARS)

    vix_log = load_vix_log_level(
        str((dataset.returns.index.min() - pd.DateOffset(years=8)).date()),
        str(dataset.returns.index.max().date()),
    )

    valuation, estimation, schedule = prepare_walk_forward_inputs(
        dataset.returns, instruments,
        BacktestSpec(id="causal-vix-regime", train_size=train_size, rebalance_frequency=str(bt.get("rebalance_frequency") or "M")),
        freq,
    )
    specs = []
    for i, signal in enumerate(schedule[:-1]):
        available = estimation.loc[estimation.index <= signal]
        train = available.tail(train_size)
        holding = valuation.loc[(valuation.index > signal) & (valuation.index <= schedule[i + 1])]
        if len(train) >= train_size and not holding.empty:
            specs.append((signal, train, holding))
    if max_folds is not None:
        specs = specs[-(max_folds + 1):]

    # Phase 1: cheap, parallel -- today's regime state at every post-burn-in
    # fold, without touching the tree at all.
    post_burn_in = [(i, spec) for i, spec in enumerate(specs) if spec[0] >= burn_in_cutoff]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        regime_by_index = dict(zip(
            (i for i, _ in post_burn_in),
            pool.map(lambda spec: fit_vix_regime_causal(vix_log, spec[0]), (spec for _, spec in post_burn_in)),
        ))

    # Phase 2: sequential, cheap -- which folds actually trigger a rebalance.
    triggers: list[int] = []
    last_state = None
    for i, _ in post_burn_in:
        _, state_today = regime_by_index[i]
        if state_today != last_state:
            triggers.append(i)
            last_state = state_today
    print(f"{len(triggers)}/{len(post_burn_in)} post-burn-in folds trigger a rebalance (regime actually changed)")

    # Phase 3: expensive, parallel -- only for triggered folds. Rebuild the
    # daily VIX state sequence once per trigger (already computed by
    # fit_vix_regime_causal's own fit -- refit here is unavoidable since we
    # only kept the scalar (p, state) from phase 1, not the full series).
    def _prepare(i):
        signal = specs[i][0]
        window = vix_log.loc[:signal].to_frame("V")
        engine = MSRegimeEngine(S_max=2, S_min=2, criterion="bic", n_starts=20, random_state=7, standardize=True)
        run = engine.fit(window, model="panel")
        daily_state = run.panel["V_state"].astype(int).astype(str)
        p_high = float(run.panel["P_V_HV"].iloc[-1])
        row_state = daily_state.reindex(estimation.index, method="ffill")
        causal_estimation = estimation.loc[:signal]
        causal_row_state = row_state.loc[:signal]
        low_rows = causal_estimation[causal_row_state == "0"]
        high_rows = causal_estimation[causal_row_state == "1"]
        estimator = HierarchicalV2Estimator()
        weights_low = dict(estimator.estimate(V2Model.from_config(config), low_rows, mode=mode, periods_per_year=ppy).terminal_weights) if len(low_rows) >= 3 else {}
        weights_high = dict(estimator.estimate(V2Model.from_config(config), high_rows, mode=mode, periods_per_year=ppy).terminal_weights) if len(high_rows) >= 3 else {}
        names = set(weights_low) | set(weights_high)
        target = {n: p_high * weights_high.get(n, 0.0) + (1 - p_high) * weights_low.get(n, 0.0) for n in names}
        print(f"[{signal.date()}] TRIGGER p_high_vol={p_high:.3f} low_rows={len(low_rows)} high_rows={len(high_rows)}", flush=True)
        return target

    with ThreadPoolExecutor(max_workers=workers) as pool:
        target_by_trigger = dict(zip(triggers, pool.map(_prepare, triggers)))

    # Phase 4: sequential ledger -- rebalance only on triggers (or during
    # burn-in, where the unpruned/unblended tree's own decision is used
    # every fold same as the other scripts today), drift otherwise.
    signal_to_fold_target = {f.signal: f.targets["FINAL"] for f in reference_report.folds}
    ledger = _V2Ledger(float(bt.get("transaction_cost_bps") or 0))
    points = []
    for i, (signal, train, holding) in enumerate(specs):
        if i in target_by_trigger:
            cost = ledger.rebalance(target_by_trigger[i])
        elif signal < burn_in_cutoff:
            cost = ledger.rebalance(dict(signal_to_fold_target[signal]))
        else:
            cost = 0.0  # drift: no rebalance call, ledger keeps previous weights
        for first, (day, row) in enumerate(holding.iterrows()):
            points.append((day, ledger.step(row) - (cost if first == 0 else 0)))

    report_start = specs[0][0]
    curves = {
        "BASELINE": baseline_curve.loc[report_start:],
        "REGIME_VIX_EVENT": pd.Series([x[1] for x in points], index=pd.DatetimeIndex([x[0] for x in points])),
    }
    common_index = curves["BASELINE"].index.intersection(curves["REGIME_VIX_EVENT"].index)
    curves = {k: v.loc[common_index] for k, v in curves.items()}
    metrics = {a: HierarchicalV2Backtester._metrics(c) for a, c in curves.items()}
    significance = significance_report(curves, [("REGIME_VIX_EVENT", "BASELINE")], resample_frequency="M")
    sharpe_significance = sharpe_significance_report(curves, [("REGIME_VIX_EVENT", "BASELINE")], resample_frequency="M")
    print(json.dumps({
        "metrics": metrics, "significance": significance, "sharpe_significance": sharpe_significance,
        "folds": len(specs), "triggers": len(triggers),
    }, default=str, indent=2))


if __name__ == "__main__":
    max_folds = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(max_folds)
