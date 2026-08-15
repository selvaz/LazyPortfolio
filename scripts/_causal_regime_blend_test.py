"""Properly causal version of the regime-blend smoke test: at every
post-burn-in rebalance, the HMM is refit from scratch using only ACWI/AGG
data available up to that date (no future information anywhere), the two
regime-conditional tree portfolios are re-estimated on only that same
causal data, and the final weights are a probability-weighted blend --
never a hard switch. Refitting is cheap enough (a few seconds) to do at
every fold; no checkpoint/extend scheme is needed.
"""
from __future__ import annotations
import json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "project"), str(ROOT.parent / "LazyStats" / "src")]
import tree_studio  # noqa: E402
from lazystats.regimes.core import MSRegimeEngine  # noqa: E402
from lazyportfolio.calendar import _annualization_factor  # noqa: E402
from lazyportfolio.models import BacktestSpec  # noqa: E402
from lazyportfolio.walk_forward import prepare_walk_forward_inputs  # noqa: E402
from lazyportfolio.v2.backtest import HierarchicalV2Backtester, _V2Ledger  # noqa: E402
from lazyportfolio.v2.hierarchy import HierarchicalV2Estimator  # noqa: E402
from lazyportfolio.v2.model import V2Model  # noqa: E402
from lazyportfolio.v2 import store  # noqa: E402
from pruning_runner import significance_report, sharpe_significance_report  # noqa: E402

BURN_IN_YEARS = 2.0


def fit_regime_causal(acwi_agg_monthly: pd.DataFrame, as_of) -> tuple[float, pd.Series]:
    """Fit fresh on data <= as_of only, joint ACWI+AGG (the variant that
    reached Sharpe 0.792 vs baseline's 0.741, before the equity-only variant
    tested worse). Returns (P(high vol) today, monthly state-membership
    Series indexed like the input).
    """
    window = acwi_agg_monthly.loc[:as_of]
    engine = MSRegimeEngine(S_max=2, S_min=2, criterion="bic", n_starts=20, random_state=7)
    run = engine.fit(window, model="joint_full")
    p_high_today = float(run.panel["P_A_HV"].iloc[-1])
    state = run.panel["A_state"]
    return p_high_today, state


def _prepare_fold(config, mode, ppy, acwi_agg_monthly, estimation, signal):
    """One post-burn-in fold's target weights + decision info. Independent
    of every other fold (no ledger state involved), so safe to run in
    parallel threads.
    """
    estimator = HierarchicalV2Estimator()
    p_high, monthly_state = fit_regime_causal(acwi_agg_monthly, signal)
    month_of = estimation.index.to_period("M")
    state_by_month = monthly_state.copy()
    state_by_month.index = state_by_month.index.to_period("M")
    row_state = pd.Series(month_of, index=estimation.index).map(state_by_month)
    causal_estimation = estimation.loc[:signal]
    causal_row_state = row_state.loc[:signal]
    low_rows = causal_estimation[causal_row_state == 0]
    high_rows = causal_estimation[causal_row_state == 1]
    weights_low = dict(estimator.estimate(V2Model.from_config(config), low_rows, mode=mode, periods_per_year=ppy).terminal_weights) if len(low_rows) >= 3 else {}
    weights_high = dict(estimator.estimate(V2Model.from_config(config), high_rows, mode=mode, periods_per_year=ppy).terminal_weights) if len(high_rows) >= 3 else {}
    names = set(weights_low) | set(weights_high)
    target = {n: p_high * weights_high.get(n, 0.0) + (1 - p_high) * weights_low.get(n, 0.0) for n in names}
    decision = {
        "signal": str(signal.date()), "burn_in": False, "p_high_vol": p_high,
        "low_rows": len(low_rows), "high_rows": len(high_rows),
    }
    print(f"[{signal.date()}] p_high_vol={p_high:.3f} low_rows={len(low_rows)} high_rows={len(high_rows)}", flush=True)
    return target, decision


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

    acwi_agg_daily = dataset.returns[["ticker:ACWI", "ticker:AGG"]].rename(columns={"ticker:ACWI": "A", "ticker:AGG": "B"})
    acwi_agg_monthly = (1 + acwi_agg_daily).resample("ME").prod() - 1

    print("Running proper walk-forward baseline (reused as STATIC_FINAL everywhere else today)...")
    _, _, reference_report = tree_studio._run_full_backtest(
        config, capture_audit_series=False, max_workers=4, expanding=False
    )
    baseline_curve = reference_report.curves["FINAL"]
    burn_in_cutoff = reference_report.folds[0].holding_start + pd.DateOffset(years=BURN_IN_YEARS)

    valuation, estimation, schedule = prepare_walk_forward_inputs(
        dataset.returns, instruments,
        BacktestSpec(id="causal-regime-blend", train_size=train_size, rebalance_frequency=str(bt.get("rebalance_frequency") or "M")),
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

    signal_to_fold_target = {f.signal: f.targets["FINAL"] for f in reference_report.folds}
    post_burn_in = [(i, spec) for i, spec in enumerate(specs) if spec[0] >= burn_in_cutoff]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        prepared_by_index = dict(zip(
            (i for i, _ in post_burn_in),
            pool.map(
                lambda spec: _prepare_fold(config, mode, ppy, acwi_agg_monthly, estimation, spec[0]),
                (spec for _, spec in post_burn_in),
            ),
        ))

    ledger = _V2Ledger(float(bt.get("transaction_cost_bps") or 0))
    points = []
    decisions = []
    for i, (signal, train, holding) in enumerate(specs):
        if i in prepared_by_index:
            target, decision = prepared_by_index[i]
        else:
            target = dict(signal_to_fold_target[signal])
            decision = {"signal": str(signal.date()), "burn_in": True}
        decisions.append(decision)
        cost = ledger.rebalance(target)
        for first, (day, row) in enumerate(holding.iterrows()):
            points.append((day, ledger.step(row) - (cost if first == 0 else 0)))

    report_start, report_end = specs[0][0], specs[-1][0]
    curves = {
        "BASELINE": baseline_curve.loc[report_start:],
        "REGIME_BLEND_CAUSAL": pd.Series([x[1] for x in points], index=pd.DatetimeIndex([x[0] for x in points])),
    }
    common_index = curves["BASELINE"].index.intersection(curves["REGIME_BLEND_CAUSAL"].index)
    curves = {k: v.loc[common_index] for k, v in curves.items()}
    metrics = {a: HierarchicalV2Backtester._metrics(c) for a, c in curves.items()}
    significance = significance_report(curves, [("REGIME_BLEND_CAUSAL", "BASELINE")], resample_frequency="M")
    sharpe_significance = sharpe_significance_report(curves, [("REGIME_BLEND_CAUSAL", "BASELINE")], resample_frequency="M")
    print(json.dumps({
        "metrics": metrics, "significance": significance, "sharpe_significance": sharpe_significance,
        "folds": len(specs),
    }, default=str, indent=2))


if __name__ == "__main__":
    max_folds = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(max_folds)
