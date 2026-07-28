from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lazyportfolio import (
    CASH_BORROW,
    V2Constraints,
    V2LocalOptimizer,
    V2OptimizationError,
)


def _returns() -> pd.DataFrame:
    rng = np.random.default_rng(122)
    return pd.DataFrame(
        {
            "ticker:A": rng.normal(0.01, 0.008, 180),
            "ticker:B": rng.normal(0.008, 0.007, 180),
        }
    )


def _solve(constraints: V2Constraints) -> tuple[dict[str, float], object]:
    return V2LocalOptimizer().solve(
        _returns(),
        objective="max_return",
        constraints=constraints,
        periods_per_year=12.0,
        target_reference_series=None,
        cap_reference_series=None,
        tracking_reference_series=None,
        reference_weights=None,
        risk_free_rate=0.02,
    )


def test_borrowing_is_tried_when_lending_regime_is_infeasible() -> None:
    weights, audit = _solve(
        V2Constraints(
            cash_enabled=True,
            max_leverage=1.3,
            mean_estimator="empirical",
            min_weights={"ticker:A": 0.6, "ticker:B": 0.6},
        )
    )
    assert weights[CASH_BORROW] <= -0.2 + 2e-6
    assert audit.financing_regime == "cash_borrowing"
    assert audit.risky_gross_exposure >= 1.2 - 2e-6


def test_combined_error_reports_each_infeasible_regime() -> None:
    with pytest.raises(V2OptimizationError, match="lend:.*borrow:"):
        _solve(
            V2Constraints(
                cash_enabled=True,
                max_leverage=1.2,
                mean_estimator="empirical",
                min_weights={"ticker:A": 0.75, "ticker:B": 0.75},
            )
        )


def test_solver_rejects_invalid_direct_risky_bounds() -> None:
    with pytest.raises(V2OptimizationError, match="risky-asset bounds"):
        _solve(
            V2Constraints(
                cash_enabled=True,
                mean_estimator="empirical",
                min_weights={"ticker:A": -0.1},
            )
        )
