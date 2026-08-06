from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from lazyportfolio import (
    CASH_BORROW,
    CASH_LEND,
    HierarchicalV2Backtester,
    HierarchicalV2Estimator,
    V2Constraints,
    V2LocalOptimizer,
    V2Model,
    V2Node,
    V2OptimizationError,
)
from lazyportfolio.scientific_study import (
    baseline_allocations,
    paired_block_bootstrap,
)
from lazyportfolio.v2.contracts import (
    V2View,
    audit_from_base,
    constraints_from_base,
)
from lazyportfolio.v2.moments import apply_views, estimate_moments
from lazyportfolio.v2.validation import (
    boolean,
    finite_float,
    normalize_config,
    optional_number,
    setting_source,
)


def _config(constraints: object | None = None) -> dict[str, object]:
    return {
        "root_id": "root",
        "currency": "USD",
        "nodes": [
            {
                "id": "root",
                "name": "Root",
                "children": [],
                "instruments": ["A", "B"],
                "goal": {"objective": "max_return"},
                "constraints": {} if constraints is None else constraints,
            }
        ],
        "backtest": {
            "benchmark": {"name": "B0", "weights": {"A": 0.5, "B": 0.5}}
        },
    }


def _returns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker:A": [0.01, 0.02, -0.01, 0.015],
            "ticker:B": [0.005, -0.002, 0.008, 0.004],
        }
    )


def test_numeric_and_boolean_helpers_fail_loudly() -> None:
    with pytest.raises(ValueError, match="finite number"):
        finite_float(object(), "x")
    with pytest.raises(ValueError, match="must be finite"):
        finite_float(float("inf"), "x")
    assert optional_number({}, "x", "x") is None
    assert setting_source(1.0, 2.0) == "node"
    assert setting_source(None, 2.0) == "root"
    assert setting_source(None, None) == "hard_default"
    assert boolean(True, "x") is True
    assert boolean("yes", "x") is True
    assert boolean("off", "x") is False
    with pytest.raises(ValueError, match="must be boolean"):
        boolean("maybe", "x")


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"nodes": None}, "nodes must be a list"),
        ({"nodes": ["bad"]}, "each node must be an object"),
        (_config(["bad"]), "constraints must be an object"),
        (_config({"max_turnover": 0.2}), "unsupported until"),
        (_config({"view_covariance_policy": "bad"}), "unsupported view_covariance_policy"),
        (_config({"volatility_target_mode": "bad"}), "unsupported volatility_target_mode"),
        (_config({"risk_free_rate": "nan"}), "must be finite"),
        (_config({"risk_aversion": 0}), "must be positive"),
        (_config({"per_asset_cap": -0.1}), "cannot be negative"),
        (_config({"per_asset_cap": 1.1}), "must be in"),
        (_config({"min_weights": ["bad"]}), "min_weights must be an object"),
        (_config({"max_weights": {"A": 1.2}}), r"must be in \[0, 1\]"),
        (
            _config({"min_weights": {"A": 0.8}, "max_weights": {"A": 0.2}}),
            "exceeds max_weights",
        ),
        (_config({"views": "bad"}), "views must be a list"),
        (_config({"views": ["bad"]}), "view 0 must be an object"),
        (
            _config(
                {
                    "views": [
                        {
                            "instruments": {"A": 1.0},
                            "expected_return": 0.1,
                            "confidence": 0.0,
                        }
                    ]
                }
            ),
            "confidence must be in",
        ),
        (
            _config(
                {
                    "views": [
                        {
                            "instruments": {},
                            "expected_return": 0.1,
                            "confidence": 0.5,
                        }
                    ]
                }
            ),
            "requires instrument coefficients",
        ),
        (
            _config(
                {
                    "volatility_target_mode": "cap",
                    "vol_target": 0.1,
                    "max_volatility": 0.2,
                }
            ),
            "cannot declare both",
        ),
    ],
)
def test_config_validation_errors(config: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_config(config)


def test_config_root_data_objective_and_financing_guards() -> None:
    missing = _config()
    missing["root_id"] = "missing"
    with pytest.raises(ValueError, match="root node is missing"):
        normalize_config(missing)

    malformed_data = _config()
    malformed_data["data"] = []
    with pytest.raises(ValueError, match="data must be an object"):
        normalize_config(malformed_data)

    invalid_objective = _config()
    invalid_objective["nodes"][0]["goal"] = {"objective": "bad"}
    with pytest.raises(ValueError, match="unsupported objective"):
        normalize_config(invalid_objective)

    negative_spread = _config({"max_leverage": 1.2})
    negative_spread["data"] = {"borrow_spread_bps": -1}
    with pytest.raises(ValueError, match="cannot be negative"):
        normalize_config(negative_spread)

    child_financing = _config()
    child_financing["nodes"] = [
        {
            "id": "root",
            "name": "Root",
            "children": ["child"],
            "instruments": ["A"],
            "goal": {"objective": "max_return"},
            "constraints": {},
        },
        {
            "id": "child",
            "name": "Child",
            "children": [],
            "instruments": ["B"],
            "proxy": "B",
            "goal": {"objective": "max_return"},
            "constraints": {"cash_enabled": True},
        },
    ]
    normalized = normalize_config(child_financing)
    child_constraints = normalized["nodes"][1]["constraints"]
    assert child_constraints["cash_enabled"] is True
    assert child_constraints["max_leverage"] == pytest.approx(1.0)


def test_config_migration_preserves_cash_leverage_and_risk_free() -> None:
    config = _config(
        {
            "allow_cash": "yes",
            "max_leverage": "1.4",
            "volatility_target_mode": "at_most",
            "volatility_reference": "manual",
            "vol_target": 0.1,
            "min_weights": {"A": "", "B": 0.1},
        }
    )
    config["data"] = {"risk_free_annual": 0.02, "borrow_spread_bps": 75}
    normalized = normalize_config(config)
    constraints = normalized["nodes"][0]["constraints"]
    assert constraints["cash_enabled"] is True
    assert constraints["max_leverage"] == pytest.approx(1.4)
    assert constraints["borrow_spread_bps"] == pytest.approx(75.0)
    assert constraints["max_volatility"] == pytest.approx(0.1)
    assert constraints["risk_free_rate"] == pytest.approx(0.02)
    assert constraints["min_weights"] == {"B": 0.1}


def test_conflicting_risk_free_rates_fail() -> None:
    config = _config({"risk_free_rate": 0.03})
    config["data"] = {"risk_free_annual": 0.02}
    with pytest.raises(ValueError, match="conflicting risk-free"):
        normalize_config(config)


def test_contract_promotion_helpers_preserve_fields() -> None:
    base_constraints = V2Constraints(mean_estimator="empirical")
    promoted = constraints_from_base(base_constraints, max_leverage=1.2)
    assert promoted.mean_estimator == "empirical"
    assert promoted.max_leverage == pytest.approx(1.2)

    audit = V2LocalOptimizer().solve(
        _returns(),
        objective="min_risk",
        constraints=V2Constraints(mean_estimator="empirical"),
        periods_per_year=12.0,
        target_reference_series=None,
        cap_reference_series=None,
        tracking_reference_series=None,
        reference_weights=None,
    )[1]
    promoted_audit = audit_from_base(audit, solver_strategy="test")
    assert promoted_audit.actual_volatility == pytest.approx(audit.actual_volatility)
    assert promoted_audit.solver_strategy == "test"


def test_moment_financing_and_view_guards() -> None:
    frame = _returns()
    frame[CASH_LEND] = 0.001
    frame[CASH_BORROW] = 0.002
    with pytest.raises(V2OptimizationError, match="exactly one cash"):
        estimate_moments(
            frame,
            list(frame.columns),
            None,
            "empirical",
            1.0,
            0.0,
            "shrunk_fixed",
        )
    with pytest.raises(V2OptimizationError, match="equilibrium"):
        estimate_moments(
            frame.loc[:, ["ticker:A", CASH_LEND]],
            ["ticker:A", CASH_LEND],
            None,
            "equilibrium",
            1.0,
            0.0,
            "shrunk_fixed",
        )
    with pytest.raises(V2OptimizationError, match="at least one risky"):
        estimate_moments(
            frame.loc[:, [CASH_LEND]],
            [CASH_LEND],
            None,
            "empirical",
            1.0,
            0.0,
            "shrunk_fixed",
        )
    with pytest.raises(ValueError, match="unsupported covariance"):
        estimate_moments(
            _returns(),
            list(_returns().columns),
            None,
            "empirical",
            1.0,
            0.0,
            "bad",
        )
    with pytest.raises(V2OptimizationError, match="unsupported view covariance"):
        apply_views(
            np.eye(2),
            np.zeros(2),
            ["ticker:A", "ticker:B"],
            (
                type(
                    "View",
                    (),
                    {
                        "instruments": {"ticker:A": 1.0},
                        "expected_return": 0.1,
                        "confidence": 0.5,
                        "source": "test",
                    },
                )(),
            ),
            0.05,
            12.0,
            "bad",
        )

    prior = np.array([[1.0, 0.2], [0.2, 1.0]], dtype=float)
    posterior, posterior_means, details = apply_views(
        prior,
        np.array([0.01, 0.02], dtype=float),
        ["ticker:A", "ticker:B"],
        (
            V2View(
                instruments={"ticker:A": 1.0},
                expected_return=0.24,
                confidence=0.5,
            ),
        ),
        0.05,
        12.0,
        "posterior_all",
    )
    assert not np.array_equal(posterior, prior)
    assert posterior_means[0] != pytest.approx(0.01)
    assert len(details) == 1


def test_financing_composition_requires_available_cash_series() -> None:
    node = V2Node(
        id="root",
        name="Root",
        instruments=["ticker:A"],
        children=[],
        proxy=None,
        objective="max_return",
        constraints=V2Constraints(cash_enabled=True),
    )
    audit = V2LocalOptimizer().solve(
        _returns(),
        objective="min_risk",
        constraints=V2Constraints(),
        periods_per_year=12.0,
        target_reference_series=None,
        cap_reference_series=None,
        tracking_reference_series=None,
        reference_weights=None,
    )[1]
    with pytest.raises(V2OptimizationError, match="missing returns"):
        HierarchicalV2Estimator._compose(
            node,
            _returns(),
            {"ticker:A": 0.5, CASH_LEND: 0.5},
            audit,
            {},
            {},
        )


def test_scientific_and_non_financed_paths_remain_available() -> None:
    model = V2Model.from_config(_config())
    with pytest.raises(V2OptimizationError, match="at least three"):
        baseline_allocations(
            model,
            _returns().iloc[:2],
            risk_aversion=1.0,
            risk_free_rate=0.0,
        )
    with pytest.raises(ValueError, match="at least two"):
        paired_block_bootstrap(
            np.array([[0.1, 0.2]]),
            samples=100,
            block_size=1,
            random_seed=1,
        )

    estimate = HierarchicalV2Estimator().estimate(
        model,
        _returns(),
        mode="flat",
        periods_per_year=12.0,
    )
    assert CASH_LEND not in estimate.terminal_weights

    daily = pd.concat([_returns()] * 40, ignore_index=True)
    daily.index = pd.bdate_range("2025-01-02", periods=len(daily))
    report = HierarchicalV2Backtester().run(
        model,
        daily,
        mode="flat",
        train_size=20,
        estimation_frequency="D",
        rebalance_frequency="M",
        include_partial_last_period=True,
    )
    assert "FINAL" in report.curves


def test_invalid_direct_financing_bounds_fail_through_solver() -> None:
    with pytest.raises(V2OptimizationError, match="risky-asset bounds"):
        V2LocalOptimizer().solve(
            _returns(),
            objective="max_return",
            constraints=V2Constraints(
                cash_enabled=True,
                mean_estimator="empirical",
                min_weights={"ticker:A": -0.1},
            ),
            periods_per_year=12.0,
            target_reference_series=None,
            cap_reference_series=None,
            tracking_reference_series=None,
            reference_weights=None,
        )

    with pytest.raises(V2OptimizationError, match="cannot satisfy"):
        V2LocalOptimizer().solve(
            _returns(),
            objective="max_return",
            constraints=replace(
                V2Constraints(cash_enabled=True, mean_estimator="empirical"),
                min_weights={"ticker:A": 0.8, "ticker:B": 0.8},
            ),
            periods_per_year=12.0,
            target_reference_series=None,
            cap_reference_series=None,
            tracking_reference_series=None,
            reference_weights=None,
        )
