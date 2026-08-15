"""Same smoke test as _quick_regime_blend_test.py, generalized to N regime
states: estimate one STATIC portfolio per state (using the whole history,
rows assigned by argmax state), then blend all N by each date's state
probabilities. Proper walk-forward baseline (same FINAL curve used
everywhere else today), only the regime construction stays intentionally
leaky (whole-history HMM fit, whole-history static per-state portfolios).
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

REGIME_FILE = Path(r"C:\Users\ADMINI~1\AppData\Local\Temp\claude\C--Users-Administrator-Documents-GitHub\ff407865-9171-4d5b-9bf7-28afdb9d80b4\scratchpad\regime_probs_3state.json")
STATE_COLUMNS = {"low": "p_low", "mid": "p_mid", "high": "p_high"}


def load_regime_probs() -> pd.DataFrame:
    payload = json.loads(REGIME_FILE.read_text())
    index = pd.to_datetime(payload["index"])
    return pd.DataFrame({state: payload[col] for state, col in STATE_COLUMNS.items()}, index=index)


def main() -> None:
    tree_name = "Global Multi-Asset"
    config = store.read_model(tree_name)
    model, dataset = tree_studio._v2_inputs(config)
    bt, mode = config["backtest"], tree_studio._v2_mode(config)
    freq = str(bt.get("estimation_frequency") or "W")
    ppy = _annualization_factor(freq)

    probs = load_regime_probs()
    estimation = (1 + dataset.returns).resample(freq).prod() - 1
    probs_monthly = probs.resample("ME").last()
    month_of = estimation.index.to_period("M")
    probs_by_month = probs_monthly.copy()
    probs_by_month.index = probs_by_month.index.to_period("M")
    row_state = probs_by_month.idxmax(axis=1).reindex(month_of).set_axis(estimation.index)

    estimator = HierarchicalV2Estimator()
    weights_by_state: dict[str, dict[str, float]] = {}
    for state in STATE_COLUMNS:
        rows = estimation[row_state == state]
        print(f"{state} training rows: {len(rows)}")
        weights_by_state[state] = dict(
            estimator.estimate(V2Model.from_config(config), rows, mode=mode, periods_per_year=ppy).terminal_weights
        )
        print(f"weights_{state}:", json.dumps(weights_by_state[state], indent=2))

    print("Running proper walk-forward baseline (this is the slow part)...")
    _, _, reference_report = tree_studio._run_full_backtest(
        config, capture_audit_series=False, max_workers=4, expanding=False
    )
    baseline_curve = reference_report.curves["FINAL"]
    report_start, report_end = baseline_curve.index.min(), baseline_curve.index.max()

    daily_probs = probs.reindex(dataset.returns.index, method="ffill").bfill()
    names = set().union(*(w.keys() for w in weights_by_state.values()))
    ledger = _V2Ledger(float(bt.get("transaction_cost_bps") or 0))
    points = []
    last_month = None
    for day, row in dataset.returns.loc[report_start:report_end].iterrows():
        if last_month is None or day.month != last_month:
            p = daily_probs.loc[day]
            blended = {
                name: sum(float(p[state]) * weights_by_state[state].get(name, 0.0) for state in STATE_COLUMNS)
                for name in names
            }
            ledger.rebalance(blended)
            last_month = day.month
        points.append((day, ledger.step(row)))

    curves = {
        "BASELINE": baseline_curve,
        "REGIME_BLEND_3STATE": pd.Series([x[1] for x in points], index=pd.DatetimeIndex([x[0] for x in points])),
    }
    metrics = {a: HierarchicalV2Backtester._metrics(c) for a, c in curves.items()}
    significance = significance_report(curves, [("REGIME_BLEND_3STATE", "BASELINE")], resample_frequency="M")
    print(json.dumps({"metrics": metrics, "significance": significance}, default=str, indent=2))


if __name__ == "__main__":
    main()
