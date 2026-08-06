"""Phase B1 (v3 performance roadmap): the LP fast path for objective='max_return'
with no vol target/cap, no TEV, no financing, only budget + box bounds.

Every test compares the LP route's real output against SLSQP's on the exact
same problem (never asserts LP in isolation) -- SLSQP is temporarily
disabled via monkeypatching V2LocalOptimizer._solve_lp_max_return to
return None, the same fallback path a genuine HiGHS failure takes.
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
    """Run solve() with the LP route disabled, so SLSQP handles it -- the
    ground truth every LP-route test compares against.

    Uses monkeypatch (not a manual save/restore) deliberately: assigning a
    plain function back onto the class after saving
    ``V2LocalOptimizer._solve_lp_max_return`` loses its ``staticmethod``
    wrapping (attribute access on a staticmethod returns the plain
    function), so a naive restore turns it into a bound method on the next
    call -- monkeypatch's setattr/undo round-trips the real descriptor.
    """
    monkeypatch.setattr(
        V2LocalOptimizer, "_solve_lp_max_return", staticmethod(lambda *a, **k: None)
    )
    return V2LocalOptimizer().solve(returns, **kwargs)


def _solve_kwargs(constraints: V2Constraints) -> dict:
    return {
        "objective": "max_return",
        "constraints": constraints,
        "periods_per_year": 252.0,
        "target_reference_series": None,
        "cap_reference_series": None,
        "tracking_reference_series": None,
        "reference_weights": None,
    }


def test_lp_route_engages_for_plain_max_return() -> None:
    returns = _returns()
    weights, audit = V2LocalOptimizer().solve(returns, **_solve_kwargs(V2Constraints()))
    assert audit.solver_strategy == "lp_highs"
    assert audit.problem_class == "max_return|vol=none"
    assert audit.global_optimality_claim is True
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-8)


def test_lp_route_matches_slsqp_weights_and_expected_return(monkeypatch) -> None:
    returns = _returns()
    constraints = V2Constraints(max_weights={"B": 0.7})
    lp_weights, lp_audit = V2LocalOptimizer().solve(returns, **_solve_kwargs(constraints))
    slsqp_weights, slsqp_audit = _solve_forcing_slsqp(
        monkeypatch, returns, **_solve_kwargs(constraints)
    )

    assert slsqp_audit.solver_strategy == "slsqp_multistart_audited"
    for name in lp_weights:
        assert lp_weights[name] == pytest.approx(slsqp_weights[name], abs=1e-4)
    assert lp_audit.expected_return_annualized == pytest.approx(
        slsqp_audit.expected_return_annualized, abs=1e-6
    )


def test_lp_route_respects_min_and_max_weight_bounds() -> None:
    returns = _returns()
    constraints = V2Constraints(min_weights={"C": 0.2}, max_weights={"B": 0.4})
    weights, audit = V2LocalOptimizer().solve(returns, **_solve_kwargs(constraints))
    assert audit.solver_strategy == "lp_highs"
    assert weights["C"] >= 0.2 - 1e-8
    assert weights["B"] <= 0.4 + 1e-8
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-8)


def test_lp_route_uses_view_adjusted_means_same_as_slsqp(monkeypatch) -> None:
    """The LP route must consume apply_views()'s output, never recompute
    means itself -- see the v3 plan's "views invariant" for Phase B."""
    returns = _returns()
    views = (V2View(instruments={"C": 1.0}, expected_return=0.20, confidence=0.9),)
    constraints = V2Constraints(views=views)
    lp_weights, lp_audit = V2LocalOptimizer().solve(returns, **_solve_kwargs(constraints))
    slsqp_weights, slsqp_audit = _solve_forcing_slsqp(
        monkeypatch, returns, **_solve_kwargs(constraints)
    )

    assert lp_audit.views_applied == 1
    assert lp_audit.expected_return_annualized == pytest.approx(
        slsqp_audit.expected_return_annualized, abs=1e-6
    )
    for name in lp_weights:
        assert lp_weights[name] == pytest.approx(slsqp_weights[name], abs=1e-4)


def test_lp_route_skipped_when_volatility_target_is_set() -> None:
    returns = _returns()
    constraints = V2Constraints(volatility_reference="manual", volatility_target=0.15)
    _, audit = V2LocalOptimizer().solve(returns, **_solve_kwargs(constraints))
    assert audit.solver_strategy == "slsqp_multistart_audited"


def test_lp_route_skipped_when_volatility_cap_is_set() -> None:
    returns = _returns()
    constraints = V2Constraints(max_volatility_reference="manual", max_volatility=0.15)
    _, audit = V2LocalOptimizer().solve(returns, **_solve_kwargs(constraints))
    assert audit.solver_strategy == "slsqp_multistart_audited"


def test_lp_route_skipped_when_tracking_error_limit_is_set() -> None:
    returns = _returns()
    rng = np.random.default_rng(1)
    father = pd.Series(rng.normal(0.0004, 0.009, len(returns)), index=returns.index)
    # A generous limit -- this test only checks that the LP route is
    # skipped once a TEV constraint exists, not TEV feasibility mechanics.
    constraints = V2Constraints(max_tracking_error=0.50)
    _, audit = V2LocalOptimizer().solve(
        returns,
        objective="max_return",
        constraints=constraints,
        periods_per_year=252.0,
        target_reference_series=None,
        cap_reference_series=None,
        tracking_reference_series=father,
        reference_weights=None,
    )
    assert audit.solver_strategy == "slsqp_multistart_audited"


def test_lp_route_skipped_when_financing_is_enabled() -> None:
    returns = _returns()
    constraints = V2Constraints(cash_enabled=True, max_leverage=1.2)
    _, audit = V2LocalOptimizer().solve(returns, **_solve_kwargs(constraints))
    assert audit.solver_strategy != "lp_highs"


def test_lp_route_never_used_for_min_risk_or_max_ratio_or_max_utility() -> None:
    returns = _returns()
    for objective in ("min_risk", "max_ratio", "max_utility"):
        constraints = V2Constraints()
        kwargs = _solve_kwargs(constraints)
        kwargs["objective"] = objective
        _, audit = V2LocalOptimizer().solve(returns, **kwargs)
        assert audit.solver_strategy != "lp_highs"


def test_lp_route_falls_back_to_slsqp_when_highs_reports_failure(monkeypatch) -> None:
    returns = _returns()
    monkeypatch.setattr(
        V2LocalOptimizer, "_solve_lp_max_return", staticmethod(lambda *a, **k: None)
    )
    weights, audit = V2LocalOptimizer().solve(returns, **_solve_kwargs(V2Constraints()))
    assert audit.solver_strategy == "slsqp_multistart_audited"
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-8)
