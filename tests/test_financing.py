from __future__ import annotations

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
    V2OptimizationError,
)


def _returns(mean: float, volatility: float = 0.001, periods: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(41)
    return pd.DataFrame(
        {
            "ticker:A": rng.normal(mean, volatility, periods),
            "ticker:B": rng.normal(mean * 0.8, volatility, periods),
        }
    )


def _model(constraints: dict[str, object]) -> V2Model:
    return V2Model.from_config(
        {
            "root_id": "root",
            "nodes": [
                {
                    "id": "root",
                    "name": "Root",
                    "children": [],
                    "instruments": ["A", "B"],
                    "goal": {"objective": "max_return"},
                    "constraints": constraints,
                }
            ],
            "backtest": {
                "benchmark": {
                    "name": "B0",
                    "weights": {"A": 0.5, "B": 0.5},
                }
            },
        }
    )


def test_positive_cash_earns_risk_free_rate() -> None:
    optimizer = V2LocalOptimizer()
    weights, audit = optimizer.solve(
        _returns(mean=-0.002),
        objective="max_return",
        constraints=V2Constraints(
            cash_enabled=True,
            max_leverage=1.0,
            mean_estimator="empirical",
        ),
        periods_per_year=12.0,
        target_reference_series=None,
        cap_reference_series=None,
        tracking_reference_series=None,
        reference_weights=None,
        risk_free_rate=0.06,
    )
    assert weights[CASH_LEND] == pytest.approx(1.0, abs=2e-5)
    assert audit.cash_weight == pytest.approx(1.0, abs=2e-5)
    assert audit.risky_gross_exposure == pytest.approx(0.0, abs=2e-5)
    assert audit.expected_return_annualized == pytest.approx(0.06, abs=2e-5)
    assert audit.financing_regime == "cash_lending"


def test_leverage_is_negative_cash_and_respects_limit() -> None:
    optimizer = V2LocalOptimizer()
    weights, audit = optimizer.solve(
        _returns(mean=0.02),
        objective="max_return",
        constraints=V2Constraints(
            cash_enabled=True,
            max_leverage=1.5,
            borrow_spread_bps=100.0,
            mean_estimator="empirical",
        ),
        periods_per_year=12.0,
        target_reference_series=None,
        cap_reference_series=None,
        tracking_reference_series=None,
        reference_weights=None,
        risk_free_rate=0.03,
    )
    assert weights[CASH_BORROW] == pytest.approx(-0.5, abs=2e-5)
    assert audit.cash_weight == pytest.approx(-0.5, abs=2e-5)
    assert audit.risky_gross_exposure == pytest.approx(1.5, abs=2e-5)
    assert audit.max_leverage == pytest.approx(1.5)
    assert audit.cash_borrowing_rate == pytest.approx(0.04)
    assert audit.financing_regime == "cash_borrowing"


def test_borrowing_spread_reduces_leveraged_expected_return() -> None:
    common = dict(
        frame=_returns(mean=0.012),
        objective="max_return",
        periods_per_year=12.0,
        target_reference_series=None,
        cap_reference_series=None,
        tracking_reference_series=None,
        reference_weights=None,
        risk_free_rate=0.03,
    )
    _, free = V2LocalOptimizer().solve(
        constraints=V2Constraints(
            cash_enabled=True,
            max_leverage=1.5,
            borrow_spread_bps=0.0,
            mean_estimator="empirical",
        ),
        **common,
    )
    _, costly = V2LocalOptimizer().solve(
        constraints=V2Constraints(
            cash_enabled=True,
            max_leverage=1.5,
            borrow_spread_bps=500.0,
            mean_estimator="empirical",
        ),
        **common,
    )
    assert costly.expected_return_annualized < free.expected_return_annualized


def test_config_enables_financing_when_max_leverage_exceeds_one() -> None:
    model = _model(
        {
            "max_leverage": "1.4",
            "risk_free_rate": "0.025",
            "mean_estimator": "empirical",
        }
    )
    assert model.root.constraints.cash_enabled is True
    assert model.root.constraints.max_leverage == pytest.approx(1.4)


def test_identity_leverage_spread_requires_explicit_cash_permission() -> None:
    with pytest.raises(ValueError, match="requires cash_enabled"):
        _model({"max_leverage": 1.0, "borrow_spread_bps": 25.0})

    model = _model(
        {
            "cash_enabled": True,
            "max_leverage": 1.0,
            "borrow_spread_bps": 25.0,
            "mean_estimator": "empirical",
        }
    )
    constraints = model.root.constraints
    assert constraints.cash_enabled is True
    assert constraints.max_leverage == pytest.approx(1.0)
    assert constraints.borrow_spread_bps == pytest.approx(25.0)


def test_global_borrow_spread_is_migrated_to_root() -> None:
    config = {
        "root_id": "root",
        "nodes": [
            {
                "id": "root",
                "name": "Root",
                "children": [],
                "instruments": ["A", "B"],
                "goal": {"objective": "max_return"},
                "constraints": {"max_leverage": 1.2},
            }
        ],
        "backtest": {
            "benchmark": {"name": "B0", "weights": {"A": 0.5, "B": 0.5}}
        },
        "data": {"borrow_spread_bps": 75},
    }
    model = V2Model.from_config(config)
    assert model.root.constraints.borrow_spread_bps == pytest.approx(75.0)


def test_global_spread_can_fund_child_without_activating_root() -> None:
    config = {
        "root_id": "root",
        "nodes": [
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
                "constraints": {"max_leverage": 1.2},
            },
        ],
        "backtest": {
            "benchmark": {"name": "B0", "weights": {"A": 0.5, "B": 0.5}}
        },
        "data": {"borrow_spread_bps": 25},
    }
    model = V2Model.from_config(config)
    root = model.root.constraints
    child = model.root.children[0].constraints
    assert root.cash_enabled is False
    assert root.borrow_spread_bps == pytest.approx(0.0)
    assert root.borrow_spread_bps_source == "default"
    assert child.borrow_spread_bps == pytest.approx(25.0)
    assert child.borrow_spread_bps_source == "root"


def test_non_root_financing_is_accepted() -> None:
    config = {
        "root_id": "root",
        "nodes": [
            {
                "id": "root",
                "name": "Root",
                "children": ["child"],
                "instruments": ["A"],
                "goal": {"objective": "max_return"},
                "constraints": {"borrow_spread_bps": 80, "cash_enabled": True},
            },
            {
                "id": "child",
                "name": "Child",
                "children": [],
                "instruments": ["B"],
                "proxy": "B",
                "goal": {"objective": "max_return"},
                "constraints": {"max_leverage": 1.2},
            },
        ],
        "backtest": {
            "benchmark": {"name": "B0", "weights": {"A": 0.5, "B": 0.5}}
        },
    }
    model = V2Model.from_config(config)
    child = model.root.children[0].constraints
    assert child.cash_enabled is True
    assert child.max_leverage == pytest.approx(1.2)
    assert child.borrow_spread_bps == pytest.approx(80.0)
    assert child.borrow_spread_bps_source == "root"


def test_hrp_rejects_cash_and_leverage() -> None:
    with pytest.raises(V2OptimizationError, match="does not support cash or leverage"):
        V2LocalOptimizer().solve(
            _returns(mean=0.005),
            objective="hrp",
            constraints=V2Constraints(cash_enabled=True),
            periods_per_year=12.0,
            target_reference_series=None,
            cap_reference_series=None,
            tracking_reference_series=None,
            reference_weights=None,
        )


def test_estimator_composes_cash_as_terminal_instrument() -> None:
    model = _model(
        {
            "cash_enabled": True,
            "max_leverage": 1.0,
            "risk_free_rate": 0.06,
            "mean_estimator": "empirical",
        }
    )
    estimate = HierarchicalV2Estimator().estimate(
        model,
        _returns(mean=-0.002),
        mode="flat",
        periods_per_year=12.0,
    )
    assert estimate.terminal_weights[CASH_LEND] == pytest.approx(1.0, abs=2e-5)
    assert (
        estimate.node_results[
            "Global flat terminal allocation"
        ].synthetic_returns.mean()
        == pytest.approx(0.005)
    )


def test_backtest_cash_curve_is_remunerated() -> None:
    model = _model(
        {
            "cash_enabled": True,
            "max_leverage": 1.0,
            "risk_free_rate": 0.0504,
            "mean_estimator": "empirical",
        }
    )
    daily = _returns(mean=-0.002, periods=180)
    daily.index = pd.bdate_range("2025-01-02", periods=len(daily))
    report = HierarchicalV2Backtester().run(
        model,
        daily,
        mode="flat",
        train_size=40,
        estimation_frequency="D",
        rebalance_frequency="M",
        include_partial_last_period=True,
    )
    final = report.curves["FINAL"]
    assert np.median(final.to_numpy(dtype=float)) == pytest.approx(0.0504 / 252.0)
    assert report.metrics["FINAL"]["risk_free_rate"] == pytest.approx(0.0504)


def _single_asset_returns() -> pd.DataFrame:
    values = np.tile(np.array([-0.02, -0.01, 0.01, 0.02], dtype=float), 45)
    return pd.DataFrame({"ticker:A": values + 0.004})


def test_father_target_becomes_feasible_through_positive_cash() -> None:
    frame = _single_asset_returns()
    father = frame["ticker:A"] * 0.5
    weights, audit = V2LocalOptimizer().solve(
        frame,
        objective="max_return",
        constraints=V2Constraints(
            cash_enabled=True,
            mean_estimator="empirical",
            volatility_reference="father_proxy",
        ),
        periods_per_year=12.0,
        target_reference_series=father,
        cap_reference_series=None,
        tracking_reference_series=None,
        reference_weights=None,
        risk_free_rate=0.0,
    )
    assert weights["ticker:A"] == pytest.approx(0.5, abs=2e-3)
    assert weights[CASH_LEND] == pytest.approx(0.5, abs=2e-3)
    assert audit.target_status == "matched"


def test_father_target_becomes_feasible_through_local_leverage() -> None:
    frame = _single_asset_returns()
    frame["ticker:B"] = frame["ticker:A"]
    father = frame.mean(axis="columns") * 1.3
    weights, audit = V2LocalOptimizer().solve(
        frame,
        objective="max_return",
        constraints=V2Constraints(
            max_leverage=1.5,
            mean_estimator="empirical",
            volatility_reference="father_proxy",
        ),
        periods_per_year=12.0,
        target_reference_series=father,
        cap_reference_series=None,
        tracking_reference_series=None,
        reference_weights=None,
        risk_free_rate=0.0,
    )
    risky_gross = sum(
        weight
        for name, weight in weights.items()
        if name not in {CASH_LEND, CASH_BORROW}
    )
    assert 1.0 < risky_gross <= 1.5 + 2e-6
    assert weights[CASH_BORROW] == pytest.approx(1.0 - risky_gross, abs=2e-6)
    assert audit.risky_gross_exposure == pytest.approx(risky_gross, abs=2e-6)
    assert audit.target_status == "matched"


def test_cash_allowed_but_not_used_when_risky_return_is_better() -> None:
    weights, _ = V2LocalOptimizer().solve(
        _returns(mean=0.02),
        objective="max_return",
        constraints=V2Constraints(cash_enabled=True, mean_estimator="empirical"),
        periods_per_year=12.0,
        target_reference_series=None,
        cap_reference_series=None,
        tracking_reference_series=None,
        reference_weights=None,
        risk_free_rate=0.01,
    )
    assert weights[CASH_LEND] == pytest.approx(0.0, abs=2e-5)
    assert sum(weight for name, weight in weights.items() if name != CASH_LEND) == pytest.approx(
        1.0, abs=2e-5
    )


def test_leverage_allowed_but_not_used_when_borrowing_is_uneconomic() -> None:
    weights, _ = V2LocalOptimizer().solve(
        _returns(mean=0.01),
        objective="max_return",
        constraints=V2Constraints(
            max_leverage=1.5,
            borrow_spread_bps=5000.0,
            mean_estimator="empirical",
        ),
        periods_per_year=12.0,
        target_reference_series=None,
        cap_reference_series=None,
        tracking_reference_series=None,
        reference_weights=None,
        risk_free_rate=0.01,
    )
    assert sum(
        weight
        for name, weight in weights.items()
        if name not in {CASH_LEND, CASH_BORROW}
    ) == pytest.approx(1.0, abs=2e-5)
    assert abs(weights.get(CASH_LEND, 0.0)) <= 2e-5
    assert abs(weights.get(CASH_BORROW, 0.0)) <= 2e-5
