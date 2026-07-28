from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from lazyportfolio import (
    HierarchicalV2Backtester,
    V2Constraints,
    V2LocalOptimizer,
    V2Model,
)
from lazyportfolio.hierarchical_v2 import V2OptimizationError, V2View


def _config(*, objective: str = "min_risk", constraints: dict | None = None) -> dict:
    return {
        "root_id": "root",
        "nodes": [
            {
                "id": "root",
                "name": "Root",
                "children": [],
                "instruments": ["AAA", "BBB"],
                "goal": {"objective": objective},
                "constraints": constraints or {},
            }
        ],
        "backtest": {
            "benchmark": {
                "name": "B0",
                "weights": {"AAA": 0.6, "BBB": 0.4},
            }
        },
    }


def _returns(columns: tuple[str, ...] = ("ticker:AAA", "ticker:BBB")) -> pd.DataFrame:
    rng = np.random.default_rng(17)
    base = rng.normal(0.001, 0.015, size=(160, len(columns)))
    common = rng.normal(0.0, 0.006, size=(160, 1))
    return pd.DataFrame(base + common, columns=list(columns))


def test_config_rejects_non_finite_and_invalid_controls() -> None:
    with pytest.raises(ValueError, match="view_tau must be positive"):
        V2Model.from_config(_config(constraints={"view_tau": 0}))
    with pytest.raises(ValueError, match="risk_aversion must be positive"):
        V2Model.from_config(_config(constraints={"risk_aversion": 0}))
    with pytest.raises(ValueError, match="max_leverage must be at least"):
        V2Model.from_config(_config(constraints={"max_leverage": 0.9}))
    with pytest.raises(ValueError, match="unsupported covariance_estimator"):
        V2Model.from_config(
            _config(constraints={"covariance_estimator": "not-an-estimator"})
        )


def test_identity_leverage_preserves_fully_invested_contract() -> None:
    model = V2Model.from_config(_config(constraints={"max_leverage": "1"}))
    assert model.root.constraints.covariance_estimator == "shrunk_fixed"
    assert model.root.constraints.cash_enabled is False
    assert model.root.constraints.max_leverage == pytest.approx(1.0)


def test_cap_mode_is_normalized_to_convex_max_volatility_path() -> None:
    model = V2Model.from_config(
        _config(
            constraints={
                "volatility_target_mode": "cap",
                "volatility_reference": "manual",
                "vol_target": 0.12,
            }
        )
    )
    constraints = model.root.constraints
    assert constraints.volatility_target is None
    assert constraints.max_volatility == pytest.approx(0.12)
    assert constraints.max_volatility_reference == "manual"
    assert constraints.volatility_target_mode == "cap"


def test_max_utility_auto_does_not_resolve_to_equilibrium() -> None:
    frame = _returns()
    optimizer = V2LocalOptimizer()
    _, audit = optimizer.solve(
        frame,
        objective="max_utility",
        constraints=V2Constraints(mean_estimator="auto"),
        periods_per_year=52.0,
        target_reference_series=None,
        cap_reference_series=None,
        tracking_reference_series=None,
        reference_weights={"ticker:AAA": 0.75, "ticker:BBB": 0.25},
        risk_aversion=2.0,
        risk_free_rate=0.02,
    )
    assert audit.configured_mean_estimator == "auto"
    assert audit.resolved_mean_estimator == "bayes_stein"
    assert (
        audit.mean_resolution_reason
        == "auto_for_max_utility_avoids_reference_reconstruction"
    )


def test_explicit_equilibrium_remains_available_for_max_utility() -> None:
    frame = _returns()
    optimizer = V2LocalOptimizer()
    _, audit = optimizer.solve(
        frame,
        objective="max_utility",
        constraints=V2Constraints(mean_estimator="equilibrium"),
        periods_per_year=52.0,
        target_reference_series=None,
        cap_reference_series=None,
        tracking_reference_series=None,
        reference_weights={"ticker:AAA": 0.75, "ticker:BBB": 0.25},
        risk_aversion=2.0,
        risk_free_rate=0.02,
    )
    assert audit.resolved_mean_estimator == "equilibrium"
    assert audit.mean_resolution_reason == "explicit_estimator"


def test_default_views_do_not_redefine_risk_covariance() -> None:
    frame = _returns()
    optimizer = V2LocalOptimizer()
    common = {"mean_estimator": "empirical"}
    baseline_weights, baseline = optimizer.solve(
        frame,
        objective="min_risk",
        constraints=V2Constraints(**common),
        periods_per_year=52.0,
        target_reference_series=None,
        cap_reference_series=None,
        tracking_reference_series=None,
        reference_weights=None,
    )
    viewed_weights, viewed = optimizer.solve(
        frame,
        objective="min_risk",
        constraints=V2Constraints(
            **common,
            views=(
                V2View(
                    instruments={"ticker:AAA": 1.0, "ticker:BBB": -1.0},
                    expected_return=0.08,
                    confidence=0.7,
                ),
            ),
        ),
        periods_per_year=52.0,
        target_reference_series=None,
        cap_reference_series=None,
        tracking_reference_series=None,
        reference_weights=None,
    )
    assert viewed_weights == pytest.approx(baseline_weights)
    assert viewed.actual_volatility == pytest.approx(
        baseline.actual_volatility, rel=0, abs=1e-12
    )
    assert viewed.risk_covariance_role == "prior"
    assert viewed.objective_covariance_role == "prior"
    assert viewed.views_applied == 1


def test_covariance_estimator_provenance_is_audited() -> None:
    optimizer = V2LocalOptimizer()
    _, audit = optimizer.solve(
        _returns(),
        objective="min_risk",
        constraints=V2Constraints(covariance_estimator="ledoit_wolf"),
        periods_per_year=52.0,
        target_reference_series=None,
        cap_reference_series=None,
        tracking_reference_series=None,
        reference_weights=None,
    )
    assert audit.covariance_estimator == "ledoit_wolf"
    assert audit.covariance_estimator_class == "LedoitWolf"


def test_backtest_metrics_use_configured_risk_free_rate() -> None:
    returns = pd.Series([0.001] * 260, dtype=float)
    zero = HierarchicalV2Backtester._metrics(returns, 0.0)
    positive = HierarchicalV2Backtester._metrics(returns, 0.10)
    assert zero["risk_free_rate"] == 0.0
    assert positive["risk_free_rate"] == pytest.approx(0.10)
    assert positive["annualized_excess_return"] < zero["annualized_excess_return"]


def test_hrp_identity_and_post_fit_audit_are_explicit() -> None:
    frame = _returns(("ticker:A", "ticker:B", "ticker:C", "ticker:D"))
    weights, audit = V2LocalOptimizer().solve(
        frame,
        objective="hrp",
        constraints=V2Constraints(),
        periods_per_year=52.0,
        target_reference_series=None,
        cap_reference_series=None,
        tracking_reference_series=None,
        reference_weights=None,
    )
    vector = np.array(list(weights.values()), dtype=float)
    assert np.all(np.isfinite(vector))
    assert vector.sum() == pytest.approx(1.0)
    assert audit.hrp_distance_metric == "pearson"
    assert audit.hrp_linkage_method == "ward"
    assert audit.hrp_risk_measure == "variance"
    assert audit.solver_strategy == "skfolio_hrp_ward_pearson"


def test_original_config_is_not_mutated_during_normalization() -> None:
    config = _config(
        constraints={
            "max_leverage": "1",
            "volatility_target_mode": "cap",
            "volatility_reference": "manual",
            "vol_target": 0.10,
        }
    )
    original = deepcopy(config)
    V2Model.from_config(config)
    assert config == original


def test_hrp_rejects_covariance_variant_it_does_not_implement() -> None:
    with pytest.raises(V2OptimizationError, match="supports covariance_estimator"):
        V2LocalOptimizer().solve(
            _returns(("ticker:A", "ticker:B", "ticker:C", "ticker:D")),
            objective="hrp",
            constraints=V2Constraints(covariance_estimator="ledoit_wolf"),
            periods_per_year=52.0,
            target_reference_series=None,
            cap_reference_series=None,
            tracking_reference_series=None,
            reference_weights=None,
        )
