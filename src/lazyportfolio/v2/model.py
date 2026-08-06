"""Validated parser for the canonical hierarchical optimizer V2 model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lazyportfolio.v2.contracts import (
    V2Benchmark,
    V2Constraints,
    V2Node,
    V2View,
    ticker,
)
from lazyportfolio.v2.validation import normalize_config


@dataclass(frozen=True)
class V2Model:
    root: V2Node
    benchmark: V2Benchmark
    reference_currency: str

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> V2Model:
        normalized = normalize_config(config)
        raw_nodes = {str(item["id"]): item for item in normalized["nodes"]}

        def build(node_id: str) -> V2Node:
            raw = raw_nodes[node_id]
            constraints = raw["constraints"]
            target = constraints.get("vol_target")
            max_volatility = constraints.get("max_volatility")
            views = tuple(
                V2View(
                    instruments={
                        ticker(key): float(value)
                        for key, value in item["instruments"].items()
                    },
                    expected_return=float(item["expected_return"]),
                    confidence=float(item["confidence"]),
                    source=str(item.get("source") or "manual"),
                )
                for item in constraints.get("views") or []
            )
            return V2Node(
                id=node_id,
                name=str(raw.get("name") or node_id),
                instruments=[ticker(value) for value in raw.get("instruments") or []],
                children=[build(str(child)) for child in raw.get("children") or []],
                proxy=ticker(raw["proxy"]) if raw.get("proxy") else None,
                objective=str((raw.get("goal") or {}).get("objective") or "min_risk"),
                constraints=V2Constraints(
                    min_weights={
                        ticker(key): float(value)
                        for key, value in constraints["min_weights"].items()
                    },
                    max_weights={
                        ticker(key): float(value)
                        for key, value in constraints["max_weights"].items()
                    },
                    per_asset_cap=(
                        float(constraints["per_asset_cap"])
                        if constraints.get("per_asset_cap") not in (None, "")
                        else None
                    ),
                    volatility_reference=str(
                        constraints.get("volatility_reference")
                        or ("manual" if target not in (None, "") else "none")
                    ),
                    volatility_target=(
                        float(target) if target not in (None, "") else None
                    ),
                    max_volatility_reference=str(
                        constraints.get("max_volatility_reference")
                        or (
                            "manual"
                            if max_volatility not in (None, "")
                            else "none"
                        )
                    ),
                    max_volatility=(
                        float(max_volatility)
                        if max_volatility not in (None, "")
                        else None
                    ),
                    max_tracking_error=(
                        float(constraints["max_tracking_error"])
                        if constraints.get("max_tracking_error") not in (None, "")
                        else None
                    ),
                    tracking_error_reference=str(
                        constraints.get("tracking_error_reference") or "declared"
                    ),
                    mean_estimator=str(
                        constraints.get("mean_estimator") or "auto"
                    ),
                    views=views,
                    view_tau=(
                        float(constraints["view_tau"])
                        if constraints.get("view_tau") not in (None, "")
                        else 0.05
                    ),
                    risk_aversion=(
                        float(constraints["risk_aversion"])
                        if constraints.get("risk_aversion") not in (None, "")
                        else None
                    ),
                    risk_free_rate=(
                        float(constraints["risk_free_rate"])
                        if constraints.get("risk_free_rate") not in (None, "")
                        else None
                    ),
                    covariance_estimator=str(
                        constraints.get("covariance_estimator") or "shrunk_fixed"
                    ),
                    view_covariance_policy=str(
                        constraints.get("view_covariance_policy") or "prior_risk"
                    ),
                    volatility_target_mode=str(
                        constraints.get("volatility_target_mode") or "exact"
                    ),
                    cash_enabled=bool(constraints.get("cash_enabled", False)),
                    max_leverage=float(constraints.get("max_leverage", 1.0)),
                    borrow_spread_bps=float(
                        constraints.get("borrow_spread_bps", 0.0)
                    ),
                    cash_enabled_source=str(
                        constraints.get("cash_enabled_source") or "default"
                    ),
                    max_leverage_source=str(
                        constraints.get("max_leverage_source") or "default"
                    ),
                    borrow_spread_bps_source=str(
                        constraints.get("borrow_spread_bps_source") or "default"
                    ),
                    mean_reference_kind=str(
                        constraints.get("mean_reference_kind") or "none"
                    ),
                    mean_reference_weights=(
                        {
                            ticker(key): float(value)
                            for key, value in constraints["mean_reference_weights"].items()
                        }
                        if constraints.get("mean_reference_weights")
                        else None
                    ),
                    tracking_error_policy=str(
                        constraints.get("tracking_error_policy") or "hard_fail"
                    ),
                    volatility_target_policy=str(
                        constraints.get("volatility_target_policy") or "hard_fail"
                    ),
                    volatility_cap_policy=str(
                        constraints.get("volatility_cap_policy") or "hard_fail"
                    ),
                ),
            )

        benchmark_raw = normalized["backtest"]["benchmark"]
        return cls(
            root=build(str(normalized["root_id"])),
            benchmark=V2Benchmark(
                name=str(
                    benchmark_raw.get("name")
                    or benchmark_raw.get("id")
                    or "B0"
                ),
                weights={
                    ticker(key): float(value)
                    for key, value in benchmark_raw["weights"].items()
                },
            ),
            reference_currency=str(normalized["currency"]),
        )


__all__ = ["V2Model"]
