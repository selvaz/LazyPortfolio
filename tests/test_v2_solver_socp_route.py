"""Phase C (v3 performance roadmap, merged C1+C2): the Clarabel SOCP route.

C1 is the standalone convex fast path for objective='max_return' with a
volatility *cap* and/or TEV cap (never an exact equality target), no
financing -- an exact SOCP, so it can honestly claim global optimality like
the LP/QP routes (B1/B2). C2 is the hybrid warm start for the (much more
common on real trees) *exact*-target case: the cap-relaxed SOCP solution
either binds exactly at the target (accepted directly) or seeds the existing
frontier/multi-start SLSQP search instead of a blind boundary start.

Every test compares the SOCP-assisted result against SLSQP's own result on
the exact same problem (never asserts SOCP output in isolation) -- both
routes share one Clarabel entry point, `V2LocalOptimizer._solve_socp_cap_weights`,
so disabling it (monkeypatched to return None) forces the fully-SLSQP
fallback for both C1 and C2, the same path a genuine cvxpy/Clarabel absence
or failure takes.
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


def _father_series(returns: pd.DataFrame, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(
        rng.normal(0.0004, 0.009, len(returns)), index=returns.index, name="FATHER"
    )


def _solve_forcing_slsqp(monkeypatch, returns, **kwargs):
    """Run solve() with the shared SOCP entry point disabled, so SLSQP alone
    handles it (C1 falls all the way through; C2 loses its warm start but
    still reaches the same eventual multi-start search) -- the ground truth
    every SOCP-route test compares against.

    Uses monkeypatch (not a manual save/restore) deliberately -- see
    test_v2_solver_qp_route.py's `_solve_forcing_slsqp` for why a naive
    save/restore of a staticmethod is unsafe.
    """
    monkeypatch.setattr(
        V2LocalOptimizer, "_solve_socp_cap_weights", staticmethod(lambda *a, **k: None)
    )
    return V2LocalOptimizer().solve(returns, **kwargs)


def _solve_kwargs(
    objective: str,
    constraints: V2Constraints,
    *,
    target_reference_series: pd.Series | None = None,
    cap_reference_series: pd.Series | None = None,
    tracking_reference_series: pd.Series | None = None,
) -> dict:
    return {
        "objective": objective,
        "constraints": constraints,
        "periods_per_year": 252.0,
        "target_reference_series": target_reference_series,
        "cap_reference_series": cap_reference_series,
        "tracking_reference_series": tracking_reference_series,
        "reference_weights": None,
    }


# --- C1: standalone cap/TEV route -------------------------------------------


def test_socp_cap_route_engages_for_max_return_with_volatility_cap() -> None:
    returns = _returns()
    constraints = V2Constraints(max_volatility_reference="manual", max_volatility=0.12)
    weights, audit = V2LocalOptimizer().solve(
        returns, **_solve_kwargs("max_return", constraints)
    )
    assert audit.solver_strategy == "socp_clarabel"
    assert audit.global_optimality_claim is True
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
    assert audit.actual_volatility <= (audit.volatility_cap or 0.0) + 1e-6


def test_socp_cap_route_engages_for_max_return_with_tev_limit() -> None:
    returns = _returns()
    father = _father_series(returns)
    constraints = V2Constraints(max_tracking_error=0.50)
    weights, audit = V2LocalOptimizer().solve(
        returns,
        **_solve_kwargs(
            "max_return", constraints, tracking_reference_series=father
        ),
    )
    assert audit.solver_strategy == "socp_clarabel"
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
    assert audit.actual_tracking_error <= constraints.max_tracking_error + 1e-6


@pytest.mark.parametrize("limit_kind", ["volatility_cap", "tev"])
def test_socp_cap_route_matches_slsqp(monkeypatch, limit_kind: str) -> None:
    returns = _returns()
    father = _father_series(returns)
    if limit_kind == "volatility_cap":
        constraints = V2Constraints(max_volatility_reference="manual", max_volatility=0.12)
        kwargs = _solve_kwargs("max_return", constraints)
    else:
        constraints = V2Constraints(max_tracking_error=0.50)
        kwargs = _solve_kwargs(
            "max_return", constraints, tracking_reference_series=father
        )

    socp_weights, socp_audit = V2LocalOptimizer().solve(returns, **kwargs)
    slsqp_weights, slsqp_audit = _solve_forcing_slsqp(monkeypatch, returns, **kwargs)

    assert slsqp_audit.solver_strategy == "slsqp_multistart_audited"
    assert socp_audit.expected_return_annualized == pytest.approx(
        slsqp_audit.expected_return_annualized, abs=1e-5
    )
    for name in socp_weights:
        assert socp_weights[name] == pytest.approx(slsqp_weights[name], abs=5e-3)


def test_socp_cap_route_uses_view_adjusted_moments_same_as_slsqp(monkeypatch) -> None:
    """See the v3 plan's "views invariant": every route (LP/QP/SOCP) must
    consume apply_views()'s output, never recompute means/covariance."""
    returns = _returns()
    views = (V2View(instruments={"C": 1.0}, expected_return=0.20, confidence=0.9),)
    constraints = V2Constraints(
        max_volatility_reference="manual", max_volatility=0.12, views=views
    )
    kwargs = _solve_kwargs("max_return", constraints)
    socp_weights, socp_audit = V2LocalOptimizer().solve(returns, **kwargs)
    slsqp_weights, slsqp_audit = _solve_forcing_slsqp(monkeypatch, returns, **kwargs)

    assert socp_audit.views_applied == 1
    assert socp_audit.expected_return_annualized == pytest.approx(
        slsqp_audit.expected_return_annualized, abs=1e-5
    )
    for name in socp_weights:
        assert socp_weights[name] == pytest.approx(slsqp_weights[name], abs=5e-3)


def test_socp_cap_route_skipped_when_target_is_set() -> None:
    # An exact equality target is C2's (hybrid warm-start) problem, not C1's
    # -- the final solve still goes through SLSQP either way (only
    # `warm_started` reflects the SOCP assist), so solver_strategy must
    # never become "socp_clarabel" for a target-mode problem.
    returns = _returns()
    father = _father_series(returns)
    constraints = V2Constraints(volatility_reference="manual", volatility_target=0.12)
    _, audit = V2LocalOptimizer().solve(
        returns, **_solve_kwargs("max_return", constraints, target_reference_series=father)
    )
    assert audit.solver_strategy == "slsqp_multistart_audited"


def test_socp_cap_route_never_used_for_min_risk_or_max_utility_or_max_ratio() -> None:
    returns = _returns()
    constraints = V2Constraints(max_volatility_reference="manual", max_volatility=0.12)
    for objective in ("min_risk", "max_utility", "max_ratio"):
        _, audit = V2LocalOptimizer().solve(
            returns, **_solve_kwargs(objective, constraints)
        )
        assert audit.solver_strategy != "socp_clarabel"


def test_socp_cap_route_falls_back_to_slsqp_when_clarabel_unavailable(monkeypatch) -> None:
    returns = _returns()
    constraints = V2Constraints(max_volatility_reference="manual", max_volatility=0.12)
    weights, audit = _solve_forcing_slsqp(
        monkeypatch, returns, **_solve_kwargs("max_return", constraints)
    )
    assert audit.solver_strategy == "slsqp_multistart_audited"
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)


# --- C2: hybrid exact-target warm start --------------------------------------


def test_hybrid_target_route_matches_slsqp(monkeypatch) -> None:
    returns = _returns()
    father = _father_series(returns)
    constraints = V2Constraints(volatility_reference="manual", volatility_target=0.11)
    kwargs = _solve_kwargs("max_return", constraints, target_reference_series=father)

    hybrid_weights, hybrid_audit = V2LocalOptimizer().solve(returns, **kwargs)
    slsqp_weights, slsqp_audit = _solve_forcing_slsqp(monkeypatch, returns, **kwargs)

    assert hybrid_audit.solver_strategy == "slsqp_multistart_audited"
    assert hybrid_audit.target_status == slsqp_audit.target_status
    assert hybrid_audit.expected_return_annualized == pytest.approx(
        slsqp_audit.expected_return_annualized, abs=1e-5
    )
    for name in hybrid_weights:
        assert hybrid_weights[name] == pytest.approx(slsqp_weights[name], abs=1e-3)


def test_hybrid_target_route_sets_warm_started_flag(monkeypatch) -> None:
    returns = _returns()
    father = _father_series(returns)
    constraints = V2Constraints(volatility_reference="manual", volatility_target=0.11)
    kwargs = _solve_kwargs("max_return", constraints, target_reference_series=father)

    _, warm_audit = V2LocalOptimizer().solve(returns, **kwargs)
    _, cold_audit = _solve_forcing_slsqp(monkeypatch, returns, **kwargs)

    assert warm_audit.warm_started is True
    assert cold_audit.warm_started is False


def test_hybrid_target_route_matches_slsqp_for_investable_proxy(monkeypatch) -> None:
    """Phase B.5's regression case (see test_v2_solver_risk_measure_consistency.py
    for the dedicated coverage), exercised through the SOCP-assisted path
    specifically: when the father proxy is also a candidate instrument and
    every other instrument is far too volatile to blend in without moving
    realized volatility measurably off target, the solve must still land on
    "matched" -- and the SOCP-warm-started result must agree with the
    SLSQP-only fallback, not just each reach "matched" independently.
    """
    rng = np.random.default_rng(7)
    index = pd.bdate_range("2020-01-01", periods=500)
    returns = pd.DataFrame(
        {
            "GLD": rng.normal(0.0003, 0.010, len(index)),
            "SLV": rng.normal(0.0002, 0.05, len(index)),
            "PPLT": rng.normal(0.0001, 0.06, len(index)),
            "PALL": rng.normal(0.0004, 0.07, len(index)),
        },
        index=index,
    )
    father = returns["GLD"]
    constraints = V2Constraints(
        volatility_reference="father_proxy", volatility_target_policy="hard_fail"
    )
    kwargs = _solve_kwargs("max_return", constraints, target_reference_series=father)

    warm_weights, warm_audit = V2LocalOptimizer().solve(returns, **kwargs)
    cold_weights, cold_audit = _solve_forcing_slsqp(monkeypatch, returns, **kwargs)

    assert warm_audit.target_status == cold_audit.target_status == "matched"
    assert warm_audit.warm_started is True
    assert cold_audit.warm_started is False
    for name in warm_weights:
        assert warm_weights[name] == pytest.approx(cold_weights[name], abs=1e-3)


def test_hybrid_target_route_falls_back_cleanly_when_clarabel_unavailable(
    monkeypatch,
) -> None:
    returns = _returns()
    father = _father_series(returns)
    constraints = V2Constraints(volatility_reference="manual", volatility_target=0.11)
    weights, audit = _solve_forcing_slsqp(
        monkeypatch,
        returns,
        **_solve_kwargs("max_return", constraints, target_reference_series=father),
    )
    assert audit.solver_strategy == "slsqp_multistart_audited"
    assert audit.warm_started is False
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
