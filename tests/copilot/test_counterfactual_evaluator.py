"""docs/node-copilot-operational-plan.md §13 Fase 2 exit criteria for the
counterfactual evaluator: baseline and variant share the same dataset,
empty proposed_views reproduce the baseline exactly, and deltas/turnover
are internally consistent."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd

from lazyportfolio.backend import OptimizationDataset
from lazyportfolio.copilot.contracts import ProposedView
from lazyportfolio.copilot.counterfactual import evaluate_view_counterfactual


def _config() -> dict[str, Any]:
    return {
        "root_id": "root",
        "currency": "USD",
        "nodes": [
            {
                "id": "root",
                "name": "Root",
                "children": ["equity"],
                "instruments": ["ticker:BOND"],
                "goal": {"objective": "min_risk"},
                "constraints": {},
            },
            {
                "id": "equity",
                "name": "Equity",
                "children": [],
                "instruments": ["ticker:CHILD_A", "ticker:CHILD_B"],
                "proxy": "ticker:EQUITY",
                # max_ratio (not min_risk/hrp): it actually consumes the
                # expected-return vector, so a Black-Litterman view can move
                # its solved weights. min_risk/hrp ignore expected returns
                # entirely regardless of view_covariance_policy -- see
                # lazyportfolio.copilot.node_universe's "no_effect_on_weights"
                # rule, which this config would otherwise trip.
                "goal": {"objective": "max_ratio"},
                "constraints": {},
            },
        ],
        "backtest": {
            "benchmark": {
                "name": "B0",
                "weights": {"ticker:EQUITY": 0.6, "ticker:BOND": 0.4},
            }
        },
    }


def _dataset() -> OptimizationDataset:
    rng = np.random.default_rng(20260809)
    index = pd.bdate_range("2020-01-01", periods=300)
    returns = pd.DataFrame(
        {
            "ticker:CHILD_A": rng.normal(0.0005, 0.01, len(index)),
            "ticker:CHILD_B": rng.normal(0.0003, 0.008, len(index)),
            "ticker:BOND": rng.normal(0.0002, 0.004, len(index)),
        },
        index=index,
    )
    returns["ticker:EQUITY"] = 0.5 * returns["ticker:CHILD_A"] + 0.5 * returns["ticker:CHILD_B"]
    return OptimizationDataset(returns=returns, metadata={})


def _view() -> ProposedView:
    return ProposedView(
        instruments={"ticker:CHILD_A": 1.0, "ticker:CHILD_B": -1.0},
        expected_return=0.03,
        confidence=0.6,
        rationale="test view",
    )


def test_empty_proposed_views_reproduce_the_baseline_exactly() -> None:
    """§11: baseline and variant share the exact same dataset -- with no
    view difference at all, they must solve to the identical weights."""

    result = evaluate_view_counterfactual(
        _config(),
        "equity",
        [],
        _dataset(),
        mode="forward_backward",
        periods_per_year=252.0,
    )
    assert result.delta["terminal_weights"]
    for delta in result.delta["terminal_weights"].values():
        assert delta == 0.0
    assert result.turnover_one_way == 0.0


def test_a_real_view_produces_a_nonzero_delta_and_turnover() -> None:
    result = evaluate_view_counterfactual(
        _config(),
        "equity",
        [_view()],
        _dataset(),
        mode="forward_backward",
        periods_per_year=252.0,
    )
    assert result.turnover_one_way is not None
    assert result.turnover_one_way > 0.0
    assert any(delta != 0.0 for delta in result.delta["terminal_weights"].values())


def test_turnover_matches_the_half_sum_of_absolute_terminal_deltas() -> None:
    result = evaluate_view_counterfactual(
        _config(),
        "equity",
        [_view()],
        _dataset(),
        mode="forward_backward",
        periods_per_year=252.0,
    )
    expected = 0.5 * sum(abs(v) for v in result.delta["terminal_weights"].values())
    assert result.turnover_one_way == expected


def test_baseline_and_variant_carry_terminal_weights_and_node_audit() -> None:
    result = evaluate_view_counterfactual(
        _config(),
        "equity",
        [_view()],
        _dataset(),
        mode="forward_backward",
        periods_per_year=252.0,
    )
    assert "terminal_weights" in result.baseline
    assert "terminal_weights" in result.variant
    assert result.baseline["node_audit"] is not None
    assert result.variant["node_audit"] is not None


def test_solver_versions_are_recorded() -> None:
    result = evaluate_view_counterfactual(
        _config(),
        "equity",
        [_view()],
        _dataset(),
        mode="forward_backward",
        periods_per_year=252.0,
    )
    assert result.solver_versions.get("solver_strategy")


def test_seed_is_recorded_when_given() -> None:
    result = evaluate_view_counterfactual(
        _config(),
        "equity",
        [_view()],
        _dataset(),
        mode="forward_backward",
        periods_per_year=252.0,
        seed=7,
    )
    assert result.seed == 7


def test_never_mutates_the_caller_supplied_base_config() -> None:
    config = _config()
    before = copy.deepcopy(config)
    evaluate_view_counterfactual(
        config,
        "equity",
        [_view()],
        _dataset(),
        mode="forward_backward",
        periods_per_year=252.0,
    )
    assert config == before
