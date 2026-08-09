"""V2ProblemClass.classify: a pure function of (objective, constraints).

Assumes constraints have already been through
V2LocalOptimizer._normalize_direct_constraints (a "cap"/"at_most"
volatility_target_mode is already folded into max_volatility by the time
classify() runs) -- see solver.py's actual call site for where that's true.
"""

from __future__ import annotations

import pytest

from lazyportfolio.v2.contracts import V2Constraints
from lazyportfolio.v2.problem_class import classify


def test_plain_min_risk_has_no_volatility_mode() -> None:
    result = classify("min_risk", V2Constraints())
    assert result.volatility_mode == "none"
    assert not result.has_tev
    assert not result.has_financing
    assert not result.has_views


def test_hrp_always_reports_no_volatility_mode_even_if_constraints_declare_one() -> None:
    """hrp's own constraints validation (solver.py's _solve_hrp) already
    rejects a volatility_reference for hrp -- classify() defensively treats
    the objective itself as authoritative, not just the constraint fields."""
    result = classify("hrp", V2Constraints(volatility_target=0.1))
    assert result.volatility_mode == "none"


def test_exact_volatility_target_is_classified_exact() -> None:
    result = classify("max_return", V2Constraints(volatility_target=0.12))
    assert result.volatility_mode == "exact"


def test_volatility_cap_is_classified_capped() -> None:
    result = classify("max_return", V2Constraints(max_volatility=0.15))
    assert result.volatility_mode == "capped"


def test_both_target_and_cap_set_raises_never_silently_picks_one() -> None:
    """Should never occur past _normalize_direct_constraints -- but if it
    does (a caller bug upstream), fail loudly rather than guess."""
    with pytest.raises(ValueError, match="cannot both be set"):
        classify("max_return", V2Constraints(volatility_target=0.1, max_volatility=0.2))


def test_tracking_error_is_flagged() -> None:
    result = classify("max_return", V2Constraints(max_tracking_error=0.05))
    assert result.has_tev


def test_financing_is_flagged() -> None:
    result = classify("max_return", V2Constraints(cash_enabled=True))
    assert result.has_financing


def test_views_are_flagged_with_their_covariance_policy() -> None:
    from lazyportfolio.v2.contracts import V2View

    views = (V2View(instruments={"ticker:A": 1.0}, expected_return=0.05, confidence=0.5),)
    result = classify(
        "max_utility", V2Constraints(views=views, view_covariance_policy="posterior_all")
    )
    assert result.has_views
    assert result.view_covariance_policy == "posterior_all"


def test_no_views_reports_false_regardless_of_policy_default() -> None:
    result = classify("min_risk", V2Constraints())
    assert not result.has_views
    assert result.view_covariance_policy == "prior_risk"


@pytest.mark.parametrize("objective", ["min_risk", "max_return", "max_ratio", "max_utility"])
def test_label_is_stable_and_readable_for_every_non_hrp_objective(objective: str) -> None:
    label = classify(objective, V2Constraints()).label
    assert label == f"{objective}|vol=none"


def test_label_includes_every_active_dimension() -> None:
    from lazyportfolio.v2.contracts import V2View

    views = (V2View(instruments={"ticker:A": 1.0}, expected_return=0.05, confidence=0.5),)
    result = classify(
        "max_return",
        V2Constraints(
            max_volatility=0.15,
            max_tracking_error=0.05,
            cash_enabled=True,
            views=views,
            view_covariance_policy="posterior_all",
        ),
    )
    assert result.label == "max_return|vol=capped|tev|financing|views=posterior_all"
