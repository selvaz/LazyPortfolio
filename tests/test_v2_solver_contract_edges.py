from __future__ import annotations

import math

import pandas as pd
import pytest

from lazyportfolio import (
    HierarchicalV2Estimator,
    V2Benchmark,
    V2Constraints,
    V2LocalOptimizer,
    V2Model,
    V2Node,
    V2OptimizationError,
)


def _returns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker:A": [0.01, 0.02, -0.01, 0.015],
            "ticker:B": [0.005, -0.002, 0.008, 0.004],
        }
    )


def _solve(constraints: V2Constraints, objective: str = "max_return") -> None:
    V2LocalOptimizer().solve(
        _returns(),
        objective=objective,
        constraints=constraints,
        periods_per_year=12.0,
        target_reference_series=None,
        cap_reference_series=None,
        tracking_reference_series=None,
        reference_weights=None,
    )


@pytest.mark.parametrize(
    ("constraints", "objective", "message"),
    [
        (
            V2Constraints(covariance_estimator="bad"),
            "max_return",
            "unsupported covariance_estimator",
        ),
        (
            V2Constraints(view_covariance_policy="bad"),
            "max_return",
            "unsupported view_covariance_policy",
        ),
        (
            V2Constraints(view_tau=0.0),
            "max_return",
            "view_tau must be positive",
        ),
        (
            V2Constraints(cash_enabled=True),
            "hrp",
            "does not support cash or leverage",
        ),
        (
            V2Constraints(cash_enabled=True, max_leverage=math.nan),
            "max_return",
            "max_leverage must be finite",
        ),
        (
            V2Constraints(cash_enabled=True, borrow_spread_bps=-1.0),
            "max_return",
            "borrow_spread_bps must be finite",
        ),
        (
            V2Constraints(borrow_spread_bps=10.0),
            "max_return",
            "requires cash_enabled",
        ),
        (
            V2Constraints(volatility_target_mode="bad"),
            "max_return",
            "unsupported volatility_target_mode",
        ),
    ],
)
def test_direct_solver_rejects_invalid_economic_contracts(
    constraints: V2Constraints,
    objective: str,
    message: str,
) -> None:
    with pytest.raises(V2OptimizationError, match=message):
        _solve(constraints, objective)


def test_estimator_rejects_unknown_hierarchy_mode() -> None:
    root = V2Node(
        id="root",
        name="Root",
        instruments=["ticker:A", "ticker:B"],
        children=[],
        proxy=None,
        objective="max_return",
        constraints=V2Constraints(mean_estimator="empirical"),
    )
    model = V2Model(
        root=root,
        benchmark=V2Benchmark(
            name="B0",
            weights={"ticker:A": 0.5, "ticker:B": 0.5},
        ),
        reference_currency="USD",
    )
    with pytest.raises(V2OptimizationError, match="unsupported hierarchy mode"):
        HierarchicalV2Estimator().estimate(
            model,
            _returns(),
            mode="unknown",  # type: ignore[arg-type]
            periods_per_year=12.0,
        )
