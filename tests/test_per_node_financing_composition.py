from __future__ import annotations

from dataclasses import replace
from typing import Any

import pandas as pd
import pytest

from lazyportfolio.v2.contracts import V2Audit
from lazyportfolio.v2.hierarchy import HierarchicalV2Estimator
from lazyportfolio.v2.model import V2Model
from lazyportfolio.v2.moments import CASH_BORROW, CASH_LEND, financing_instrument


def _audit() -> V2Audit:
    return V2Audit(
        target_reference="none",
        target_volatility=None,
        actual_volatility=0.1,
        cap_reference="none",
        volatility_cap=None,
        tracking_error_limit=None,
        actual_tracking_error=None,
        minimum_slack={},
        maximum_slack={},
        sum_weights=1.0,
        solver_message="ok",
        target_status="not_requested",
        tracking_error_status="not_requested",
        configured_objective="max_return",
        effective_objective="max_return",
        expected_return_annualized=0.1,
        objective_value=0.1,
        soft_constraint_violation=0.0,
        configured_mean_estimator="empirical",
        resolved_mean_estimator="empirical",
        views_applied=0,
        view_details=(),
        risk_aversion=1.0,
        risk_free_rate=0.0,
    )


class _ScriptedOptimizer:
    def __init__(self, weights: list[dict[str, float]]) -> None:
        self.weights = list(weights)

    def solve(self, frame: Any, **kwargs: Any) -> tuple[dict[str, float], V2Audit]:
        del frame, kwargs
        weights = self.weights.pop(0)
        cash = next((name for name in (CASH_LEND, CASH_BORROW) if name in weights), "")
        cash_weight = weights.get(cash, 0.0) if cash else 0.0
        risky = sum(
            value
            for name, value in weights.items()
            if name not in {CASH_LEND, CASH_BORROW}
        )
        regime = "fully_invested"
        if cash == CASH_LEND and cash_weight > 0.0:
            regime = "cash_lending"
        elif cash == CASH_BORROW and cash_weight < 0.0:
            regime = "cash_borrowing"
        return weights, replace(
            _audit(),
            cash_instrument=cash,
            cash_weight=cash_weight,
            risky_gross_exposure=risky,
            financing_regime=regime,
        )


def _config(*, three_levels: bool = False) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = [
        {
            "id": "root",
            "name": "Root",
            "children": ["middle" if three_levels else "child"],
            "instruments": ["A"],
            "goal": {"objective": "max_return"},
            "constraints": {
                "cash_enabled": True,
                "risk_free_rate": 0.03,
                "borrow_spread_bps": 50,
            },
        }
    ]
    if three_levels:
        nodes.extend(
            [
                {
                    "id": "middle",
                    "name": "Middle",
                    "children": ["leaf"],
                    "instruments": ["M"],
                    "proxy": "PM",
                    "goal": {"objective": "max_return"},
                    "constraints": {"cash_enabled": True},
                },
                {
                    "id": "leaf",
                    "name": "Leaf",
                    "children": [],
                    "instruments": ["B"],
                    "proxy": "PL",
                    "goal": {"objective": "max_return"},
                    "constraints": {"max_leverage": 1.5, "borrow_spread_bps": 100},
                },
            ]
        )
    else:
        nodes.append(
            {
                "id": "child",
                "name": "Child",
                "children": [],
                "instruments": ["B"],
                "proxy": "P",
                "goal": {"objective": "max_return"},
                "constraints": {
                    "max_leverage": 1.5,
                    "risk_free_rate": 0.04,
                    "borrow_spread_bps": 100,
                },
            }
        )
    return {
        "root_id": "root",
        "currency": "USD",
        "nodes": nodes,
        "backtest": {"benchmark": {"name": "B0", "weights": {"A": 0.5, "B": 0.5}}},
    }


def _returns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker:A": [0.01, 0.02, 0.0, 0.01],
            "ticker:M": [0.012, 0.011, 0.013, 0.010],
            "ticker:B": [0.02, 0.01, 0.03, 0.02],
            "ticker:P": [0.015, 0.012, 0.018, 0.014],
            "ticker:PM": [0.011, 0.014, 0.012, 0.013],
            "ticker:PL": [0.018, 0.012, 0.022, 0.017],
        }
    )


def test_root_and_child_financing_are_scaled_without_collision() -> None:
    model = V2Model.from_config(_config())
    estimate = HierarchicalV2Estimator(
        _ScriptedOptimizer(
            [
                {"ticker:A": 0.5, "ticker:P": 0.4, CASH_LEND: 0.1},
                {"ticker:B": 1.5, CASH_BORROW: -0.5},
            ]
        )
    ).estimate(model, _returns(), mode="forward", periods_per_year=12.0)
    child_borrow = financing_instrument(CASH_BORROW, "child", is_root=False)
    assert estimate.terminal_weights == pytest.approx(
        {"ticker:A": 0.5, "ticker:B": 0.6, CASH_LEND: 0.1, child_borrow: -0.2}
    )
    audit = estimate.node_results["Child"].audit
    assert audit.parent_weight == pytest.approx(0.4)
    assert audit.global_risky_gross_exposure == pytest.approx(0.6)
    assert audit.global_cash_weight == pytest.approx(-0.2)
    assert audit.portfolio_risky_gross_exposure == pytest.approx(1.1)
    assert audit.portfolio_cash_weight == pytest.approx(-0.1)


def test_leaf_cash_uses_local_rate_and_remains_internal() -> None:
    config = _config()
    config["nodes"][1]["constraints"] = {"cash_enabled": True, "risk_free_rate": 0.04}
    model = V2Model.from_config(config)
    estimate = HierarchicalV2Estimator(
        _ScriptedOptimizer(
            [{"ticker:A": 0.5, "ticker:P": 0.5}, {"ticker:B": 0.75, CASH_LEND: 0.25}]
        )
    ).estimate(model, _returns(), mode="forward", periods_per_year=12.0)
    child_lend = financing_instrument(CASH_LEND, "child", is_root=False)
    assert estimate.terminal_weights == pytest.approx(
        {"ticker:A": 0.5, "ticker:B": 0.375, child_lend: 0.125}
    )
    expected = _returns()["ticker:B"] * 0.75 + 0.25 * (0.04 / 12.0)
    assert estimate.node_results["Child"].synthetic_returns.equals(expected)


def test_intermediate_cash_and_leverage_are_internal_and_scaled() -> None:
    model = V2Model.from_config(_config(three_levels=True))
    estimate = HierarchicalV2Estimator(
        _ScriptedOptimizer(
            [
                {"ticker:A": 0.5, "ticker:PM": 0.5},
                {"ticker:M": 0.4, "ticker:PL": 0.8, CASH_BORROW: -0.2},
                {"ticker:B": 1.0},
            ]
        )
    ).estimate(model, _returns(), mode="forward", periods_per_year=12.0)
    middle_borrow = financing_instrument(CASH_BORROW, "middle", is_root=False)
    assert estimate.terminal_weights == pytest.approx(
        {"ticker:A": 0.5, "ticker:M": 0.2, "ticker:B": 0.4, middle_borrow: -0.1}
    )
    audit = estimate.node_results["Middle"].audit
    assert audit.risky_gross_exposure == pytest.approx(1.2)
    assert audit.global_risky_gross_exposure == pytest.approx(0.6)
    assert audit.global_cash_weight == pytest.approx(-0.1)


def test_forward_backward_scales_nested_financing() -> None:
    model = V2Model.from_config(_config(three_levels=True))
    estimate = HierarchicalV2Estimator(
        _ScriptedOptimizer(
            [
                {"ticker:A": 0.5, "ticker:PM": 0.5},
                {"ticker:M": 0.4, "ticker:PL": 0.6},
                {"ticker:B": 1.2, CASH_BORROW: -0.2},
                {"ticker:M": 0.3, "ticker:PL_SYNTH": 0.6, CASH_LEND: 0.1},
                {"ticker:A": 0.4, "ticker:PM_SYNTH": 0.6},
            ]
        )
    ).estimate(model, _returns(), mode="forward_backward", periods_per_year=12.0)
    middle_cash = financing_instrument(CASH_LEND, "middle", is_root=False)
    leaf_borrow = financing_instrument(CASH_BORROW, "leaf", is_root=False)
    assert estimate.terminal_weights == pytest.approx(
        {
            "ticker:A": 0.4,
            "ticker:M": 0.18,
            "ticker:B": 0.432,
            middle_cash: 0.06,
            leaf_borrow: -0.072,
        }
    )
    leaf = estimate.node_results["Leaf"].audit
    assert leaf.global_node_weight == pytest.approx(0.36)
    assert leaf.global_cash_weight == pytest.approx(-0.072)
    assert (
        estimate.forward_node_results["Leaf"].audit.global_cash_weight
        == pytest.approx(-0.06)
    )


def test_flat_mode_remains_one_terminal_solve() -> None:
    config = _config()
    config["nodes"][0]["constraints"] = {}
    config["nodes"][1]["constraints"] = {}
    model = V2Model.from_config(config)
    estimate = HierarchicalV2Estimator(
        _ScriptedOptimizer(
            [
                {"ticker:A": 0.5, "ticker:P": 0.5},
                {"ticker:B": 1.0},
                {"ticker:A": 0.25, "ticker:B": 0.75},
            ]
        )
    ).estimate(model, _returns(), mode="flat", periods_per_year=12.0)
    assert estimate.terminal_weights == pytest.approx(
        {"ticker:A": 0.25, "ticker:B": 0.75}
    )
