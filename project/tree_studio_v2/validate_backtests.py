"""External gate 3: run and independently reconcile V2 walk-forward backtests."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

from lazyportfolio.backend import MarketDataHubOptimizationBackend  # noqa: E402
from lazyportfolio.hierarchical_v2 import (  # noqa: E402
    HierarchicalV2Backtester,
    V2Model,
)

MODEL_PATH = (
    ROOT
    / "reports"
    / "tree_studio"
    / "models"
    / "Global allocation_3pct_TEV_father.json"
)
OUTPUT_DIR = ROOT / "reports" / "tree_studio" / "v2"


def load_daily(config: dict, model: V2Model, end: str) -> pd.DataFrame:
    instruments = list(
        dict.fromkeys(
            [
                *model.root.terminal_instruments(),
                *(node.proxy for node in model.root.walk() if node.proxy),
                *model.benchmark.weights,
            ]
        )
    )
    dataset = MarketDataHubOptimizationBackend().load_returns(
        instruments,
        start=str((config.get("data") or {}).get("start") or ""),
        end=end,
    )
    return dataset.returns.loc[:, instruments].dropna(how="any")


def independent_final_curve(daily: pd.DataFrame, folds: list) -> pd.Series:
    """Second ledger implementation used only to reconcile FINAL returns."""
    weights: dict[str, float] = {}
    points: list[tuple[pd.Timestamp, float]] = []
    for fold in folds:
        weights = dict(fold.targets["FINAL"])
        holding = daily.loc[
            (daily.index >= fold.holding_start) & (daily.index <= fold.holding_end)
        ]
        for day, row in holding.iterrows():
            value = sum(weight * float(row[name]) for name, weight in weights.items())
            denominator = 1.0 + value
            weights = {
                name: weight * (1.0 + float(row[name])) / denominator
                for name, weight in weights.items()
            }
            points.append((day, value))
    return pd.Series(
        [value for _, value in points],
        index=pd.DatetimeIndex([day for day, _ in points]),
        dtype=float,
    )


def validate_report(daily: pd.DataFrame, report) -> None:
    observations = {int(metrics["n_obs"]) for metrics in report.metrics.values()}
    if len(observations) != 1:
        raise AssertionError(f"{report.mode}: arms have different OOS observation counts")
    independent = independent_final_curve(daily, report.folds)
    actual = report.curves["FINAL"]
    if not independent.index.equals(actual.index):
        raise AssertionError(f"{report.mode}: FINAL curve dates differ from fold ledger")
    if float((independent - actual).abs().max()) > 1e-14:
        raise AssertionError(f"{report.mode}: FINAL returns differ from target-weight ledger")
    for fold in report.folds:
        for node, audit in fold.audits.items():
            if audit.target_status == "matched":
                if abs(audit.actual_volatility - audit.target_volatility) > 5e-5:
                    raise AssertionError(f"{report.mode}/{fold.signal}/{node}: target mismatch")
            if audit.volatility_cap is not None:
                if audit.actual_volatility > audit.volatility_cap + 5e-5:
                    raise AssertionError(f"{report.mode}/{fold.signal}/{node}: cap violated")
            if audit.tracking_error_limit is not None:
                if audit.actual_tracking_error is None:
                    raise AssertionError(f"{report.mode}/{fold.signal}/{node}: TEV missing")
                if (
                    audit.tracking_error_status == "within_limit"
                    and audit.actual_tracking_error > audit.tracking_error_limit + 5e-5
                ):
                    raise AssertionError(f"{report.mode}/{fold.signal}/{node}: TEV violated")
            if min(audit.minimum_slack.values(), default=0.0) < -2e-6:
                raise AssertionError(f"{report.mode}/{fold.signal}/{node}: minimum violated")
            if min(audit.maximum_slack.values(), default=0.0) < -2e-6:
                raise AssertionError(f"{report.mode}/{fold.signal}/{node}: maximum violated")
    if report.mode == "forward_backward":
        if "B0_SYNTH" not in report.curves:
            raise AssertionError("forward_backward: diagnostic B0_SYNTH arm is missing")
        for fold in report.folds:
            if "B0_SYNTH" not in fold.targets:
                raise AssertionError("forward_backward: fold B0_SYNTH target is missing")
    sectors = {
        "ticker:XLB",
        "ticker:XLE",
        "ticker:XLF",
        "ticker:XLI",
        "ticker:XLP",
        "ticker:XLU",
        "ticker:XLV",
        "ticker:XLY",
    }
    for fold in report.folds:
        equity = fold.targets["LOCAL:Equity"]
        if sectors.intersection(equity):
            raise AssertionError(f"{report.mode}: Equity local weights contain sectors")
        expected = "ticker:SPY_SYNTH" if report.mode == "forward_backward" else "ticker:SPY"
        if expected not in equity:
            raise AssertionError(f"{report.mode}: Equity local weights omit {expected}")


def snapshot(report) -> dict:
    return {
        "mode": report.mode,
        "metrics": report.metrics,
        "transaction_cost_paid": report.transaction_cost_paid,
        "folds": [asdict(fold) for fold in report.folds],
        "curves": {
            arm: [
                {"date": str(day.date()), "return": float(value)}
                for day, value in curve.items()
            ]
            for arm, curve in report.curves.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--end", default="2022-12-31")
    args = parser.parse_args()
    config = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    model = V2Model.from_config(config)
    daily = load_daily(config, model, args.end)
    backtest = config["backtest"]
    engine = HierarchicalV2Backtester()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for mode in ("flat", "forward", "forward_backward"):
        report = engine.run(
            model,
            daily,
            mode=mode,
            train_size=int(backtest.get("train_size") or 104),
            estimation_frequency=str(backtest.get("estimation_frequency") or "W"),
            rebalance_frequency=str(backtest.get("rebalance_frequency") or "M"),
            transaction_cost_bps=float(backtest.get("transaction_cost_bps") or 0),
        )
        validate_report(daily, report)
        output = OUTPUT_DIR / f"backtest_{mode}.json"
        output.write_text(
            json.dumps(snapshot(report), indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        final = report.metrics["FINAL"]
        print(
            f"PASS {mode}: {len(report.folds)} folds, "
            f"CAGR={final['cagr']:.4%}, Sharpe={final['annualized_sharpe']:.4f}"
        )


if __name__ == "__main__":
    main()
