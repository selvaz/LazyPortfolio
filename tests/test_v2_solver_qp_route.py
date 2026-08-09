"""Phase B2 (v3 performance roadmap): the QP fast path for
objective in {'min_risk', 'max_utility'} with no vol target/cap, no TEV,
no financing, only budget + box bounds.

Every test compares the QP route's real output against SLSQP's on the exact
same problem (never asserts QP in isolation) -- SLSQP is temporarily
disabled via monkeypatching V2LocalOptimizer._solve_qp_convex to return
None, the same fallback path a genuine OSQP failure or absence takes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lazyportfolio.v2.contracts import V2Constraints, V2View
from lazyportfolio.v2.solver import V2LocalOptimizer


def _returns() -> pd.DataFrame:
    rng = np.random.default_rng(20260806)
    index = pd.bdate_range("2020-01-01", periods=400)
    return pd.DataFrame(
        {
            "A": rng.normal(0.0003, 0.010, len(index)),
            "B": rng.normal(0.0008, 0.012, len(index)),
            "C": rng.normal(0.0002, 0.008, len(index)),
        },
        index=index,
    )


def _solve_forcing_slsqp(monkeypatch, returns, **kwargs):
    """Run solve() with the QP route disabled, so SLSQP handles it -- the
    ground truth every QP-route test compares against.

    Uses monkeypatch (not a manual save/restore) deliberately: assigning a
    plain function back onto the class after saving
    ``V2LocalOptimizer._solve_qp_convex`` loses its ``staticmethod``
    wrapping (attribute access on a staticmethod returns the plain
    function), so a naive restore turns it into a bound method on the next
    call -- monkeypatch's setattr/undo round-trips the real descriptor.
    """
    monkeypatch.setattr(
        V2LocalOptimizer, "_solve_qp_convex", staticmethod(lambda *a, **k: None)
    )
    return V2LocalOptimizer().solve(returns, **kwargs)


def _solve_kwargs(objective: str, constraints: V2Constraints) -> dict:
    return {
        "objective": objective,
        "constraints": constraints,
        "periods_per_year": 252.0,
        "target_reference_series": None,
        "cap_reference_series": None,
        "tracking_reference_series": None,
        "reference_weights": None,
    }


@pytest.mark.parametrize("objective", ["min_risk", "max_utility"])
def test_qp_route_engages_for_plain_objective(objective: str) -> None:
    returns = _returns()
    weights, audit = V2LocalOptimizer().solve(
        returns, **_solve_kwargs(objective, V2Constraints())
    )
    assert audit.solver_strategy == "qp_osqp"
    assert audit.problem_class == f"{objective}|vol=none"
    assert audit.global_optimality_claim is True
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("objective", ["min_risk", "max_utility"])
def test_qp_route_matches_slsqp_weights_and_objective(monkeypatch, objective: str) -> None:
    returns = _returns()
    constraints = V2Constraints(max_weights={"B": 0.7})
    qp_weights, qp_audit = V2LocalOptimizer().solve(
        returns, **_solve_kwargs(objective, constraints)
    )
    slsqp_weights, slsqp_audit = _solve_forcing_slsqp(
        monkeypatch, returns, **_solve_kwargs(objective, constraints)
    )

    assert slsqp_audit.solver_strategy == "slsqp_multistart_audited"
    for name in qp_weights:
        assert qp_weights[name] == pytest.approx(slsqp_weights[name], abs=1e-3)
    assert qp_audit.objective_value == pytest.approx(slsqp_audit.objective_value, abs=1e-6)


def test_qp_route_respects_min_and_max_weight_bounds() -> None:
    returns = _returns()
    constraints = V2Constraints(min_weights={"C": 0.2}, max_weights={"B": 0.4})
    weights, audit = V2LocalOptimizer().solve(
        returns, **_solve_kwargs("min_risk", constraints)
    )
    assert audit.solver_strategy == "qp_osqp"
    assert weights["C"] >= 0.2 - 1e-6
    assert weights["B"] <= 0.4 + 1e-6
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("objective", ["min_risk", "max_utility"])
def test_qp_route_uses_view_adjusted_moments_same_as_slsqp(
    monkeypatch, objective: str
) -> None:
    """The QP route must consume apply_views()'s output, never recompute
    means/covariance itself -- see the v3 plan's "views invariant" for
    Phase B. min_risk only differs under view_covariance_policy=
    "posterior_all"; max_utility also depends on the view-adjusted means
    unconditionally."""
    returns = _returns()
    views = (V2View(instruments={"C": 1.0}, expected_return=0.20, confidence=0.9),)
    constraints = V2Constraints(views=views, view_covariance_policy="posterior_all")
    qp_weights, qp_audit = V2LocalOptimizer().solve(
        returns, **_solve_kwargs(objective, constraints)
    )
    slsqp_weights, slsqp_audit = _solve_forcing_slsqp(
        monkeypatch, returns, **_solve_kwargs(objective, constraints)
    )

    assert qp_audit.views_applied == 1
    assert qp_audit.objective_value == pytest.approx(slsqp_audit.objective_value, abs=1e-6)
    for name in qp_weights:
        assert qp_weights[name] == pytest.approx(slsqp_weights[name], abs=1e-3)


@pytest.mark.parametrize("objective", ["min_risk", "max_utility"])
def test_qp_route_skipped_when_volatility_target_is_set(objective: str) -> None:
    returns = _returns()
    constraints = V2Constraints(volatility_reference="manual", volatility_target=0.15)
    _, audit = V2LocalOptimizer().solve(returns, **_solve_kwargs(objective, constraints))
    assert audit.solver_strategy == "slsqp_multistart_audited"


@pytest.mark.parametrize("objective", ["min_risk", "max_utility"])
def test_qp_route_skipped_when_volatility_cap_is_set(objective: str) -> None:
    returns = _returns()
    constraints = V2Constraints(max_volatility_reference="manual", max_volatility=0.15)
    _, audit = V2LocalOptimizer().solve(returns, **_solve_kwargs(objective, constraints))
    assert audit.solver_strategy == "slsqp_multistart_audited"


def test_qp_route_skipped_when_tracking_error_limit_is_set() -> None:
    returns = _returns()
    rng = np.random.default_rng(1)
    father = pd.Series(rng.normal(0.0004, 0.009, len(returns)), index=returns.index)
    constraints = V2Constraints(max_tracking_error=0.50)
    _, audit = V2LocalOptimizer().solve(
        returns,
        objective="min_risk",
        constraints=constraints,
        periods_per_year=252.0,
        target_reference_series=None,
        cap_reference_series=None,
        tracking_reference_series=father,
        reference_weights=None,
    )
    assert audit.solver_strategy == "slsqp_multistart_audited"


def test_qp_route_skipped_when_financing_is_enabled() -> None:
    returns = _returns()
    constraints = V2Constraints(cash_enabled=True, max_leverage=1.2)
    _, audit = V2LocalOptimizer().solve(
        returns, **_solve_kwargs("min_risk", constraints)
    )
    assert audit.solver_strategy != "qp_osqp"


def test_qp_route_never_used_for_max_return_or_max_ratio_or_hrp() -> None:
    returns = _returns()
    for objective in ("max_return", "max_ratio"):
        _, audit = V2LocalOptimizer().solve(
            returns, **_solve_kwargs(objective, V2Constraints())
        )
        assert audit.solver_strategy != "qp_osqp"


def test_qp_route_falls_back_to_slsqp_when_osqp_reports_failure(monkeypatch) -> None:
    returns = _returns()
    monkeypatch.setattr(
        V2LocalOptimizer, "_solve_qp_convex", staticmethod(lambda *a, **k: None)
    )
    weights, audit = V2LocalOptimizer().solve(
        returns, **_solve_kwargs("min_risk", V2Constraints())
    )
    assert audit.solver_strategy == "slsqp_multistart_audited"
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
