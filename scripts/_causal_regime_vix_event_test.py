"""Genuinely event-driven version: the tree rebalances exactly when the
causal VIX regime changes, not at the next fixed monthly checkpoint.

Two-tier scheme, because a fresh EM fit is too expensive to run on every
single day but a daily-resolution signal is exactly what's needed:

1. Periodic EM refits (quarterly), each using only data up to that
   checkpoint -- expensive, infrequent.
2. Between checkpoints, every single day is scored with
   ``infer_with_params`` using the last checkpoint's FIXED parameters --
   cheap (no EM), so daily resolution is affordable. Critically, each
   day's call uses a window ending exactly at that day (never a longer
   batch): ``infer_with_params`` returns hmmlearn's *smoothed*
   forward-backward posterior, which would leak later days backward onto
   earlier ones inside a single batch call -- confirmed by reading
   ``_result_from_model`` in LazyStats directly. A window that stops
   exactly at the day being queried has no later data to leak from,
   regardless of smoothing, which is what keeps this causal.

The resulting stitched daily state sequence is scanned for actual
transition dates; those (irregular) dates -- not a calendar -- are where
the tree re-estimates and rebalances. Between transitions the ledger just
drifts on the last computed weights.
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
from lazystats.regimes.core import HMMParams, MSRegimeEngine, infer_with_params  # noqa: E402
from market_data_hub import extract  # noqa: E402
from lazyportfolio.calendar import _annualization_factor  # noqa: E402
from lazyportfolio.v2.backtest import HierarchicalV2Backtester, _V2Ledger  # noqa: E402
from lazyportfolio.v2.hierarchy import HierarchicalV2Estimator  # noqa: E402
from lazyportfolio.v2.model import V2Model  # noqa: E402
from lazyportfolio.v2 import store  # noqa: E402
from pruning_runner import significance_report, sharpe_significance_report  # noqa: E402

BURN_IN_YEARS = 2.0
REFIT_EVERY = pd.DateOffset(months=3)


def load_vix_log_level(start: str, end: str) -> pd.Series:
    df, _meta = extract.extract_series(["^VIX"], transform="level", frequency="D", start=start, end=end)
    return np.log(df["^VIX"]).rename("V")


def build_daily_causal_states(vix_log: pd.Series, eval_start, eval_end) -> pd.Series:
    """One EM fit per checkpoint (quarterly, using data <= checkpoint only),
    then one cheap infer_with_params call per day since the last checkpoint
    (window always ending exactly at that day -- see module docstring for
    why that specific shape is what keeps this causal despite gamma_ being
    a smoothed posterior in general).
    """
    checkpoints = pd.date_range(eval_start, eval_end, freq=REFIT_EVERY)
    states = {}
    for checkpoint in checkpoints:
        fit_window = vix_log.loc[:checkpoint].to_frame("V")
        if len(fit_window) < 30:
            continue
        engine = MSRegimeEngine(S_max=2, S_min=2, criterion="bic", n_starts=20, random_state=7, standardize=True)
        run = engine.fit(fit_window, model="panel")
        params = HMMParams.from_dict(run.meta["V"])
        next_checkpoint = checkpoint + REFIT_EVERY
        score_days = vix_log.loc[checkpoint:min(next_checkpoint, eval_end)].index
        for day in score_days:
            window = vix_log.loc[checkpoint:day].to_numpy().reshape(-1, 1)
            res = infer_with_params(window, params)
            states[day] = (float(res.gamma_[-1, 1]), int(res.viterbi_path_[-1]))
        print(f"checkpoint {checkpoint.date()}: fit on {len(fit_window)} rows, scored {len(score_days)} days", flush=True)
    return pd.Series({d: s for d, (p, s) in states.items()}), pd.Series({d: p for d, (p, s) in states.items()})


def _prepare_trigger(config, mode, ppy, estimation, row_state, trigger_date, p_high):
    estimator = HierarchicalV2Estimator()
    causal_estimation = estimation.loc[:trigger_date]
    causal_row_state = row_state.reindex(causal_estimation.index, method="ffill")
    low_rows = causal_estimation[causal_row_state == 0]
    high_rows = causal_estimation[causal_row_state == 1]
    weights_low = dict(estimator.estimate(V2Model.from_config(config), low_rows, mode=mode, periods_per_year=ppy).terminal_weights) if len(low_rows) >= 3 else {}
    weights_high = dict(estimator.estimate(V2Model.from_config(config), high_rows, mode=mode, periods_per_year=ppy).terminal_weights) if len(high_rows) >= 3 else {}
    names = set(weights_low) | set(weights_high)
    target = {n: p_high * weights_high.get(n, 0.0) + (1 - p_high) * weights_low.get(n, 0.0) for n in names}
    print(f"[{trigger_date.date()}] TRIGGER p_high_vol={p_high:.3f} low_rows={len(low_rows)} high_rows={len(high_rows)}", flush=True)
    return target


def main(workers: int = 4) -> None:
    tree_name = "Global Multi-Asset"
    config = store.read_model(tree_name)
    model, dataset = tree_studio._v2_inputs(config)
    bt, mode = config["backtest"], tree_studio._v2_mode(config)
    freq = str(bt.get("estimation_frequency") or "W")
    ppy = _annualization_factor(freq)
    estimation = (1 + dataset.returns).resample(freq).prod() - 1

    print("Running proper walk-forward baseline (reused as STATIC_FINAL everywhere else today)...")
    _, _, reference_report = tree_studio._run_full_backtest(
        config, capture_audit_series=False, max_workers=4, expanding=False
    )
    baseline_curve = reference_report.curves["FINAL"]
    burn_in_cutoff = reference_report.folds[0].holding_start + pd.DateOffset(years=BURN_IN_YEARS)
    eval_end = dataset.returns.index.max()

    vix_log = load_vix_log_level(
        str((dataset.returns.index.min() - pd.DateOffset(years=8)).date()), str(eval_end.date())
    )

    print("Building daily causal regime state sequence (quarterly refits + daily filtered inference)...")
    daily_state, daily_p_high = build_daily_causal_states(vix_log, burn_in_cutoff, eval_end)
    daily_state = daily_state.reindex(dataset.returns.loc[burn_in_cutoff:eval_end].index, method="ffill")
    daily_p_high = daily_p_high.reindex(dataset.returns.loc[burn_in_cutoff:eval_end].index, method="ffill")

    transitions = daily_state[daily_state != daily_state.shift(1)].index
    print(f"{len(transitions)} regime transitions (rebalance triggers) in the evaluation window", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        targets = list(pool.map(
            lambda d: _prepare_trigger(config, mode, ppy, estimation, daily_state, d, float(daily_p_high.loc[d])),
            transitions,
        ))
    trigger_target = dict(zip(transitions, targets))

    ledger = _V2Ledger(float(bt.get("transaction_cost_bps") or 0))
    points = []
    pre_burn_in = dataset.returns.loc[dataset.returns.index < burn_in_cutoff]
    burn_in_target = None
    for f in reference_report.folds:
        if f.holding_start < burn_in_cutoff:
            burn_in_target = f.targets["FINAL"]
    for first, (day, row) in enumerate(pre_burn_in.iterrows()):
        cost = ledger.rebalance(dict(burn_in_target)) if first == 0 and burn_in_target is not None else 0.0
        points.append((day, ledger.step(row) - cost))
    for day, row in dataset.returns.loc[burn_in_cutoff:eval_end].iterrows():
        cost = ledger.rebalance(trigger_target[day]) if day in trigger_target else 0.0
        points.append((day, ledger.step(row) - cost))

    report_start = dataset.returns.index[0]
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
        "transitions": len(transitions),
    }, default=str, indent=2))


if __name__ == "__main__":
    main()
