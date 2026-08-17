"""Quick, deliberately leaky smoke test for a different regime approach:
estimate two STATIC regime-conditional portfolios (using the whole history
split by regime, no walk-forward), then simulate a monthly probability-
weighted blend between them. Same one-HMM-fit leak as the view test, but no
per-node dilution: the blend acts directly on final weights.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "project")]
import tree_studio  # noqa: E402
from lazyportfolio.calendar import _annualization_factor  # noqa: E402
from lazyportfolio.v2.backtest import HierarchicalV2Backtester, _V2Ledger  # noqa: E402
from lazyportfolio.v2.hierarchy import HierarchicalV2Estimator  # noqa: E402
from lazyportfolio.v2.model import V2Model  # noqa: E402
from lazyportfolio.v2 import store  # noqa: E402
from pruning_runner import significance_report  # noqa: E402

REGIME_FILE = Path(r"C:\Users\ADMINI~1\AppData\Local\Temp\claude\C--Users-Administrator-Documents-GitHub\ff407865-9171-4d5b-9bf7-28afdb9d80b4\scratchpad\regime_probs.json")


def load_regime_prob() -> pd.Series:
    payload = json.loads(REGIME_FILE.read_text())
    return pd.Series(payload["prob_high_vol"], index=pd.to_datetime(payload["index"]))


def main() -> None:
    tree_name = "Global Multi-Asset"
    config = store.read_model(tree_name)
    model, dataset = tree_studio._v2_inputs(config)
    bt, mode = config["backtest"], tree_studio._v2_mode(config)
    freq = str(bt.get("estimation_frequency") or "W")
    ppy = _annualization_factor(freq)

    estimation = (1 + dataset.returns).resample(freq).prod() - 1
    prob_high_monthly = load_regime_prob().resample("ME").last().reindex(estimation.resample("ME").last().index, method="ffill")

    # Tag each estimation-frequency row by the regime prevailing that month.
    month_of = estimation.index.to_period("M")
    prob_by_month = prob_high_monthly.copy()
    prob_by_month.index = prob_by_month.index.to_period("M")
    row_prob_high = pd.Series(month_of, index=estimation.index).map(prob_by_month)
    high_vol_rows = estimation[row_prob_high > 0.5]
    low_vol_rows = estimation[row_prob_high <= 0.5]
    print(f"low_vol training rows: {len(low_vol_rows)}, high_vol training rows: {len(high_vol_rows)}")

    estimator = HierarchicalV2Estimator()
    weights_low = dict(estimator.estimate(V2Model.from_config(config), low_vol_rows, mode=mode, periods_per_year=ppy).terminal_weights)
    weights_high = dict(estimator.estimate(V2Model.from_config(config), high_vol_rows, mode=mode, periods_per_year=ppy).terminal_weights)
    print("weights_low_vol:", json.dumps(weights_low, indent=2))
    print("weights_high_vol:", json.dumps(weights_high, indent=2))

    # Proper baseline: the SAME walk-forward reference run used as
    # STATIC_FINAL everywhere else today -- causal, rebalanced every fold,
    # never fit on data after the date it's applied to. Only the regime
    # blend keeps its intentional leak (one HMM fit + two static regime
    # portfolios estimated on the whole history) -- that's the thing being
    # smoke-tested, not the baseline.
    print("Running proper walk-forward baseline (this is the slow part)...")
    _, _, reference_report = tree_studio._run_full_backtest(
        config, capture_audit_series=False, max_workers=4, expanding=False
    )
    baseline_curve = reference_report.curves["FINAL"]
    report_start, report_end = baseline_curve.index.min(), baseline_curve.index.max()

    daily_prob_high = load_regime_prob().reindex(dataset.returns.index, method="ffill").bfill()
    names = set(weights_low) | set(weights_high)
    ledger = _V2Ledger(float(bt.get("transaction_cost_bps") or 0))
    points = []
    last_month = None
    for day, row in dataset.returns.loc[report_start:report_end].iterrows():
        if last_month is None or day.month != last_month:
            p = float(daily_prob_high.loc[day])
            blended = {name: p * weights_high.get(name, 0.0) + (1 - p) * weights_low.get(name, 0.0) for name in names}
            ledger.rebalance(blended)
            last_month = day.month
        points.append((day, ledger.step(row)))

    curves = {
        "BASELINE": baseline_curve,
        "REGIME_BLEND": pd.Series([x[1] for x in points], index=pd.DatetimeIndex([x[0] for x in points])),
    }
    metrics = {a: HierarchicalV2Backtester._metrics(c) for a, c in curves.items()}
    significance = significance_report(curves, [("REGIME_BLEND", "BASELINE")], resample_frequency="M")
    print(json.dumps({"metrics": metrics, "significance": significance}, default=str, indent=2))


if __name__ == "__main__":
    main()
