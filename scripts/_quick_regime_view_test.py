"""One-off, deliberately leaky smoke test: is a regime-conditioned BL view on
Government Duration (TLT vs SHY tilt) worth building the full causal version
of? The HMM is fit ONCE on the whole history (future info leaks into the
regime characterization) and reused at every fold -- only the tree's own
walk-forward re-estimation is real. If even this best-case, unfair-advantage
version shows nothing, the properly causal version won't either.
"""
from __future__ import annotations
import json, sys
from copy import deepcopy
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "project")]
import tree_studio  # noqa: E402
from lazyportfolio.calendar import _annualization_factor  # noqa: E402
from lazyportfolio.models import BacktestSpec  # noqa: E402
from lazyportfolio.walk_forward import prepare_walk_forward_inputs  # noqa: E402
from lazyportfolio.v2.backtest import HierarchicalV2Backtester, _V2Ledger  # noqa: E402
from lazyportfolio.v2.hierarchy import HierarchicalV2Estimator  # noqa: E402
from lazyportfolio.v2.model import V2Model  # noqa: E402
from lazyportfolio.v2 import store  # noqa: E402
from pruning_runner import significance_report  # noqa: E402

REGIME_FILE = Path(r"C:\Users\ADMINI~1\AppData\Local\Temp\claude\C--Users-Administrator-Documents-GitHub\ff407865-9171-4d5b-9bf7-28afdb9d80b4\scratchpad\regime_probs.json")


def load_regime_prob() -> pd.Series:
    payload = json.loads(REGIME_FILE.read_text())
    return pd.Series(payload["prob_high_vol"], index=pd.to_datetime(payload["index"]))


def conditional_means(prob_high: pd.Series, returns: pd.Series) -> tuple[float, float]:
    """Probability-weighted low/high-vol conditional means (monthly, simple)."""
    aligned = pd.concat([prob_high, returns], axis=1, join="inner").dropna()
    p, r = aligned.iloc[:, 0], aligned.iloc[:, 1]
    mean_high = float((p * r).sum() / p.sum())
    mean_low = float(((1 - p) * r).sum() / (1 - p).sum())
    return mean_low, mean_high


def q_series(prob_high: pd.Series, mean_low: float, mean_high: float) -> pd.Series:
    """Annualized regime-blended expected return per date, monthly * 12."""
    return (prob_high * mean_high + (1 - prob_high) * mean_low) * 12.0


def views_for_date(q_defensive: pd.Series, q_cyclical: pd.Series, date, confidence: float = 0.5) -> list[dict]:
    idx = q_defensive.index[q_defensive.index <= date]
    as_of = idx[-1] if len(idx) else q_defensive.index[0]
    return [
        {"instruments": {"XLP": 1.0}, "expected_return": float(q_defensive.loc[as_of]), "confidence": confidence, "source": "regime:HMM"},
        {"instruments": {"XLY": 1.0}, "expected_return": float(q_cyclical.loc[as_of]), "confidence": confidence, "source": "regime:HMM"},
    ]


def main() -> None:
    tree_name = "Global Multi-Asset"
    base_config = store.read_model(tree_name)

    prob_high = load_regime_prob()
    all_returns = tree_studio._v2_inputs(base_config)[1].returns
    xlp_returns = all_returns["ticker:XLP"]
    xly_returns = all_returns["ticker:XLY"]
    xlp_monthly = (1 + xlp_returns).resample("ME").prod() - 1
    xly_monthly = (1 + xly_returns).resample("ME").prod() - 1
    xlp_low, xlp_high = conditional_means(prob_high, xlp_monthly)
    xly_low, xly_high = conditional_means(prob_high, xly_monthly)
    print(f"XLP (defensive) conditional monthly mean: low_vol={xlp_low:.4%} high_vol={xlp_high:.4%}")
    print(f"XLY (cyclical) conditional monthly mean: low_vol={xly_low:.4%} high_vol={xly_high:.4%}")
    q_defensive = q_series(prob_high, xlp_low, xlp_high)
    q_cyclical = q_series(prob_high, xly_low, xly_high)

    model, dataset = tree_studio._v2_inputs(base_config)
    bt, mode = base_config["backtest"], tree_studio._v2_mode(base_config)
    train_size, freq = int(bt.get("train_size") or 104), str(bt.get("estimation_frequency") or "W")
    instruments = list(dict.fromkeys(
        [*model.root.terminal_instruments(), *(n.proxy for n in model.root.walk() if n.proxy), *model.benchmark.weights]
    ))
    ppy = _annualization_factor(freq)
    valuation, estimation, schedule = prepare_walk_forward_inputs(
        dataset.returns, instruments,
        BacktestSpec(id="regime-view-quicktest", train_size=train_size, rebalance_frequency=str(bt.get("rebalance_frequency") or "M")),
        freq,
    )
    specs = []
    for i, signal in enumerate(schedule[:-1]):
        available = estimation.loc[estimation.index <= signal]
        train = available.tail(train_size)
        holding = valuation.loc[(valuation.index > signal) & (valuation.index <= schedule[i + 1])]
        if len(train) >= train_size and not holding.empty:
            specs.append((signal, train, holding))

    estimator = HierarchicalV2Estimator()
    arms = ("BASELINE", "REGIME_VIEW")
    ledgers = {a: _V2Ledger(float(bt.get("transaction_cost_bps") or 0)) for a in arms}
    points = {a: [] for a in arms}
    for signal, train, holding in specs:
        baseline_estimate = estimator.estimate(V2Model.from_config(base_config), train, mode=mode, periods_per_year=ppy)
        viewed_config = deepcopy(base_config)
        target_node = next(n for n in viewed_config["nodes"] if n.get("name") == "US Equity")
        target_node.setdefault("constraints", {})["views"] = views_for_date(q_defensive, q_cyclical, signal)
        viewed_estimate = estimator.estimate(V2Model.from_config(viewed_config), train, mode=mode, periods_per_year=ppy)
        targets = {"BASELINE": dict(baseline_estimate.terminal_weights), "REGIME_VIEW": dict(viewed_estimate.terminal_weights)}
        for arm, target in targets.items():
            cost = ledgers[arm].rebalance(target)
            for first, (day, row) in enumerate(holding.iterrows()):
                points[arm].append((day, ledgers[arm].step(row) - (cost if first == 0 else 0)))

    curves = {a: pd.Series([x[1] for x in points[a]], index=pd.DatetimeIndex([x[0] for x in points[a]])) for a in arms}
    metrics = {a: HierarchicalV2Backtester._metrics(c) for a, c in curves.items()}
    significance = significance_report(curves, [("REGIME_VIEW", "BASELINE")], resample_frequency=bt.get("rebalance_frequency"))
    print(json.dumps({"metrics": metrics, "significance": significance, "folds": len(specs)}, default=str, indent=2))


if __name__ == "__main__":
    main()
