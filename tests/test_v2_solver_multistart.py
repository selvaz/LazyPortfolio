"""Regression tests for randomized multi-start restarts and their diagnostics.

`max_ratio` and an exact volatility target are non-convex problems (Sharpe
ratio, and an equality constraint on a quadratic form); SLSQP has no
global-optimality guarantee there. `V2LocalOptimizer` mitigates this with a
mix of structured and randomized starts and reports a dispersion diagnostic
(`V2Audit.restart_objective_spread`) instead of silently claiming a single
answer is the global optimum.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lazyportfolio import V2Constraints, V2LocalOptimizer


def _returns() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 60
    return pd.DataFrame(
        {
            "ticker:A": rng.normal(0.0008, 0.02, n),
            "ticker:B": rng.normal(0.0004, 0.01, n),
            "ticker:C": rng.normal(0.0006, 0.015, n),
            "ticker:D": rng.normal(0.0002, 0.008, n),
        }
    )


def _solve(objective: str, constraints: V2Constraints) -> tuple[dict[str, float], object]:
    return V2LocalOptimizer().solve(
        _returns(),
        objective=objective,
        constraints=constraints,
        periods_per_year=252.0,
        target_reference_series=None,
        cap_reference_series=None,
        tracking_reference_series=None,
        reference_weights=None,
    )


class TestRandomizedStarts:
    def test_produces_feasible_points_within_bounds(self) -> None:
        lower = np.array([0.0, 0.1, 0.0, 0.05])
        upper = np.array([0.6, 0.5, 0.4, 0.3])
        starts = V2LocalOptimizer._randomized_starts(lower, upper, count=8, seed=1_337)
        assert len(starts) == 8
        for candidate in starts:
            assert abs(float(candidate.sum()) - 1.0) <= 1e-8
            assert np.all(candidate >= lower - 1e-9)
            assert np.all(candidate <= upper + 1e-9)

    def test_is_deterministic_for_the_same_seed(self) -> None:
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([1.0, 1.0, 1.0])
        first = V2LocalOptimizer._randomized_starts(lower, upper, count=5, seed=7)
        second = V2LocalOptimizer._randomized_starts(lower, upper, count=5, seed=7)
        for left, right in zip(first, second, strict=True):
            assert np.allclose(left, right)

    def test_different_seeds_explore_different_points(self) -> None:
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([1.0, 1.0, 1.0])
        first = V2LocalOptimizer._randomized_starts(lower, upper, count=5, seed=1)
        second = V2LocalOptimizer._randomized_starts(lower, upper, count=5, seed=2)
        assert any(
            not np.allclose(left, right) for left, right in zip(first, second, strict=True)
        )


class TestMultiStartDiagnosticsEndToEnd:
    @pytest.mark.parametrize("objective", ["max_ratio", "max_utility", "min_risk"])
    def test_restart_candidate_count_includes_randomized_starts(
        self, monkeypatch, objective: str
    ) -> None:
        # max_utility/min_risk with no volatility target/cap/TEV now qualify
        # for the v3 roadmap's exact QP fast path (Phase B2), which reports
        # restart_candidate_count=1 by design -- an exact convex solve needs
        # no restarts. This test is about SLSQP's OWN multi-start diagnostic
        # wiring, not about which route the classifier picks, so force the
        # SLSQP path the same way the QP route's own tests do (monkeypatch,
        # not manual save/restore -- see tests/test_v2_solver_qp_route.py for
        # why a naive restore loses the staticmethod wrapping).
        monkeypatch.setattr(
            V2LocalOptimizer, "_solve_qp_convex", staticmethod(lambda *a, **k: None)
        )
        _, audit = _solve(objective, V2Constraints(mean_estimator="empirical"))
        # 1 structured start + up to 2 boundary starts + 8 randomized starts;
        # randomized starts must actually have been fed into the search, not
        # merely generated and discarded.
        assert audit.restart_candidate_count >= 1 + V2LocalOptimizer._RANDOM_RESTART_COUNT
        assert audit.restart_objective_spread >= 0.0
        assert np.isfinite(audit.restart_objective_spread)

    def test_exact_volatility_target_also_gets_randomized_starts(self) -> None:
        _, audit = _solve(
            "max_return",
            V2Constraints(
                mean_estimator="empirical",
                volatility_target=0.15,
                volatility_reference="manual",
            ),
        )
        assert audit.restart_candidate_count >= V2LocalOptimizer._RANDOM_RESTART_COUNT
        assert np.isfinite(audit.restart_objective_spread)

    def test_hrp_does_not_fabricate_a_restart_diagnostic(self) -> None:
        _, audit = _solve("hrp", V2Constraints())
        assert audit.restart_candidate_count == 0
        assert audit.restart_objective_spread == 0.0

    def test_solve_is_reproducible_across_runs(self) -> None:
        weights_a, audit_a = _solve("max_ratio", V2Constraints(mean_estimator="empirical"))
        weights_b, audit_b = _solve("max_ratio", V2Constraints(mean_estimator="empirical"))
        assert weights_a.keys() == weights_b.keys()
        for name in weights_a:
            assert weights_a[name] == pytest.approx(weights_b[name], abs=1e-9)
        assert audit_a.restart_objective_spread == pytest.approx(
            audit_b.restart_objective_spread, abs=1e-9
        )
