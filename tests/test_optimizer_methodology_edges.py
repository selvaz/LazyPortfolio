from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lazyportfolio import V2Constraints, V2LocalOptimizer


def test_exact_target_above_max_return_volatility_can_reduce_return() -> None:
    periods = 120
    phase = np.linspace(0.0, 8.0 * np.pi, periods)
    frame = pd.DataFrame(
        {
            "ticker:QUALITY": 0.010 + 0.001 * np.sin(phase),
            "ticker:VOLATILE": -0.004 + 0.045 * np.cos(phase),
        }
    )
    optimizer = V2LocalOptimizer()
    unconstrained_weights, unconstrained = optimizer.solve(
        frame,
        objective="max_return",
        constraints=V2Constraints(mean_estimator="empirical"),
        periods_per_year=12.0,
        target_reference_series=None,
        cap_reference_series=None,
        tracking_reference_series=None,
        reference_weights=None,
    )
    target = max(unconstrained.actual_volatility * 4.0, 0.08)
    targeted_weights, targeted = optimizer.solve(
        frame,
        objective="max_return",
        constraints=V2Constraints(
            mean_estimator="empirical",
            volatility_reference="manual",
            volatility_target=target,
            volatility_target_mode="exact",
        ),
        periods_per_year=12.0,
        target_reference_series=None,
        cap_reference_series=None,
        tracking_reference_series=None,
        reference_weights=None,
    )
    assert unconstrained_weights["ticker:QUALITY"] == pytest.approx(1.0, abs=1e-5)
    assert targeted_weights["ticker:VOLATILE"] > 0.0
    assert targeted.expected_return_annualized < unconstrained.expected_return_annualized
    assert targeted.volatility_target_mode == "exact"
    assert targeted.global_optimality_claim is False


def test_hrp_is_finite_on_near_singular_returns() -> None:
    rng = np.random.default_rng(91)
    common = rng.normal(0.0004, 0.01, size=240)
    frame = pd.DataFrame(
        {
            f"ticker:A{index}": common + rng.normal(0.0, 1e-6, size=len(common))
            for index in range(5)
        }
    )
    weights, audit = V2LocalOptimizer().solve(
        frame,
        objective="hrp",
        constraints=V2Constraints(
            min_weights={"ticker:A0": 0.05},
            max_weights={"ticker:A4": 0.50},
        ),
        periods_per_year=52.0,
        target_reference_series=None,
        cap_reference_series=None,
        tracking_reference_series=None,
        reference_weights=None,
    )
    vector = np.array(list(weights.values()), dtype=float)
    assert np.all(np.isfinite(vector))
    assert vector.sum() == pytest.approx(1.0)
    assert weights["ticker:A0"] >= 0.05 - 2e-6
    assert weights["ticker:A4"] <= 0.50 + 2e-6
    assert audit.solver_strategy == "skfolio_hrp_ward_pearson"
