"""External gate 2: validate all non-iterative V2 estimates on the real example."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lazyportfolio.backend import MarketDataHubOptimizationBackend  # noqa: E402
from lazyportfolio.hierarchical_v2 import (  # noqa: E402
    HierarchicalV2Estimator,
    V2Estimate,
    V2Model,
    V2Node,
)

MODEL_PATH = (
    ROOT
    / "reports"
    / "tree_studio"
    / "models"
    / "Global allocation_3pct_TEV_father.json"
)
OUTPUT_PATH = ROOT / "reports" / "tree_studio" / "v2" / "point_estimates.json"


def load_window(config: dict, model: V2Model) -> pd.DataFrame:
    instruments = list(
        dict.fromkeys(
            [
                *model.root.terminal_instruments(),
                *(node.proxy for node in model.root.walk() if node.proxy),
                *model.benchmark.weights,
            ]
        )
    )
    data = config.get("data") or {}
    dataset = MarketDataHubOptimizationBackend().load_returns(
        instruments,
        start=str(data.get("start") or ""),
        end=str(data.get("end") or ""),
    )
    daily = dataset.returns.loc[:, instruments].dropna(how="any")
    weekly = daily.resample("W-FRI").apply(lambda values: (1.0 + values).prod() - 1.0)
    train_size = int(config["backtest"].get("train_size") or 104)
    if len(weekly) < train_size:
        raise AssertionError("real example does not have enough complete weekly observations")
    return weekly.tail(train_size)


def independently_compose(node: V2Node, estimate: V2Estimate) -> dict[str, float]:
    result = estimate.node_results[node.name]
    terminal = {instrument: result.local_weights[instrument] for instrument in node.instruments}
    for child in node.children:
        synthetic = f"{child.proxy}_SYNTH"
        column = synthetic if synthetic in result.local_weights else child.proxy
        if column is None:
            raise AssertionError(f"{child.name}: missing child component")
        child_weights = independently_compose(child, estimate)
        for instrument, weight in child_weights.items():
            terminal[instrument] = (
                terminal.get(instrument, 0.0) + result.local_weights[column] * weight
            )
    return {key: value for key, value in terminal.items() if abs(value) > 1e-12}


def assert_weights_equal(left: dict[str, float], right: dict[str, float], label: str) -> None:
    if left.keys() != right.keys():
        raise AssertionError(f"{label}: terminal keys differ")
    delta = max((abs(left[key] - right[key]) for key in left), default=0.0)
    if delta > 2e-8:
        raise AssertionError(f"{label}: maximum composition delta {delta}")


def validate_estimate(model: V2Model, train: pd.DataFrame, estimate: V2Estimate) -> None:
    if abs(sum(estimate.terminal_weights.values()) - 1.0) > 2e-8:
        raise AssertionError(f"{estimate.mode}: final weights do not sum to one")
    for name, result in estimate.node_results.items():
        if abs(sum(result.local_weights.values()) - 1.0) > 2e-8:
            raise AssertionError(f"{estimate.mode}/{name}: local weights do not sum to one")
        audit = result.audit
        if audit.target_status == "matched":
            if abs(audit.actual_volatility - audit.target_volatility) > 5e-5:
                raise AssertionError(f"{estimate.mode}/{name}: target volatility mismatch")
        if audit.volatility_cap is not None:
            if audit.actual_volatility > audit.volatility_cap + 5e-5:
                raise AssertionError(f"{estimate.mode}/{name}: volatility cap violated")
        if audit.tracking_error_limit is not None:
            if audit.actual_tracking_error is None:
                raise AssertionError(f"{estimate.mode}/{name}: TEV was not measured")
            if (
                audit.tracking_error_status == "within_limit"
                and audit.actual_tracking_error > audit.tracking_error_limit + 5e-5
            ):
                raise AssertionError(f"{estimate.mode}/{name}: TEV limit violated")
        if min(audit.minimum_slack.values(), default=0.0) < -2e-6:
            raise AssertionError(f"{estimate.mode}/{name}: minimum weight violated")
        if min(audit.maximum_slack.values(), default=0.0) < -2e-6:
            raise AssertionError(f"{estimate.mode}/{name}: maximum weight violated")
        reconstructed = train.loc[:, list(result.terminal_weights)].mul(
            result.terminal_weights, axis="columns"
        ).sum(axis="columns")
        if float((reconstructed - result.synthetic_returns).abs().max()) > 1e-12:
            raise AssertionError(f"{estimate.mode}/{name}: synthetic series mismatch")

    if estimate.mode != "flat":
        expected = independently_compose(model.root, estimate)
        assert_weights_equal(expected, estimate.terminal_weights, estimate.mode)
    if estimate.mode == "forward":
        if "ticker:SPY" not in estimate.node_results["Equity"].local_weights:
            raise AssertionError("Forward Equity solve does not contain SPY")
    if estimate.mode == "forward_backward":
        equity = estimate.node_results["Equity"].local_weights
        root = estimate.node_results["Global allocation"].local_weights
        if "ticker:SPY_SYNTH" not in equity:
            raise AssertionError("Backward Equity solve does not contain SPY_SYNTH")
        for required in ("ticker:ACWI_SYNTH", "ticker:AGG_SYNTH", "ticker:DBC_SYNTH"):
            if required not in root:
                raise AssertionError(f"Backward root solve does not contain {required}")
        b0_raw = train.loc[:, list(model.benchmark.weights)].mul(
            model.benchmark.weights, axis="columns"
        ).sum(axis="columns")
        root_result = estimate.node_results[model.root.name]
        expected_target = float(b0_raw.std(ddof=1) * (52.0**0.5))
        if abs((root_result.audit.target_volatility or 0.0) - expected_target) > 1e-10:
            raise AssertionError("Backward root target is not anchored to raw B0")
        expected_tev = float(
            (root_result.synthetic_returns - b0_raw).std(ddof=1) * (52.0**0.5)
        )
        if abs((root_result.audit.actual_tracking_error or 0.0) - expected_tev) > 1e-10:
            raise AssertionError("Backward root TEV is not measured against raw B0")
        if not estimate.synthetic_benchmark_weights:
            raise AssertionError("Backward estimate does not expose diagnostic B0_SYNTH")
        if abs(sum(estimate.synthetic_benchmark_weights.values()) - 1.0) > 2e-8:
            raise AssertionError("B0_SYNTH terminal weights do not sum to one")


def serialise(estimate: V2Estimate) -> dict:
    return {
        "mode": estimate.mode,
        "terminal_weights": estimate.terminal_weights,
        "synthetic_benchmark_weights": estimate.synthetic_benchmark_weights,
        "nodes": {
            name: {
                "local_weights": result.local_weights,
                "terminal_weights": result.terminal_weights,
                "audit": asdict(result.audit),
            }
            for name, result in estimate.node_results.items()
        },
    }


def main() -> None:
    config = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    model = V2Model.from_config(config)
    train = load_window(config, model)
    estimator = HierarchicalV2Estimator()
    snapshots = {}
    for mode in ("flat", "forward", "forward_backward"):
        estimate = estimator.estimate(model, train, mode=mode, periods_per_year=52.0)
        validate_estimate(model, train, estimate)
        snapshots[mode] = serialise(estimate)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(snapshots, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for mode, body in snapshots.items():
        active = sum(weight > 1e-6 for weight in body["terminal_weights"].values())
        print(f"PASS {mode}: {active} active terminal series")

    no_father_config = deepcopy(config)
    for raw_node in no_father_config["nodes"]:
        proxy = raw_node.get("proxy")
        if proxy:
            raw_node["instruments"] = [
                instrument
                for instrument in raw_node.get("instruments") or []
                if str(instrument).split(":")[-1].upper() != str(proxy).split(":")[-1].upper()
            ]
    no_father_model = V2Model.from_config(no_father_config)
    for mode in ("forward", "forward_backward"):
        estimate = estimator.estimate(
            no_father_model,
            train,
            mode=mode,
            periods_per_year=52.0,
        )
        validate_estimate(no_father_model, train, estimate)
        for node in no_father_model.root.walk():
            if node.proxy and node.proxy in estimate.node_results[node.name].local_weights:
                raise AssertionError(f"{mode}/{node.name}: father was reinserted")
        print(f"PASS {mode}: father references absent from candidate sleeves")
    print(f"Snapshot: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
