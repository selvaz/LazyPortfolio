"""Standalone local optimizer for the hierarchical V2 engine."""

from __future__ import annotations

import time
from dataclasses import replace
from math import isfinite, sqrt
from typing import Any, Literal

import numpy as np

from lazyportfolio.v2.contracts import (
    RECOGNIZED_OBJECTIVES,
    V2Audit,
    V2Constraints,
    V2OptimizationError,
    V2View,
)
from lazyportfolio.v2.moments import (
    CASH_BORROW,
    CASH_LEND,
    CASH_NAMES,
    apply_views,
    black_litterman_posterior,
    estimate_moments,
)
from lazyportfolio.v2.problem_class import classify
from lazyportfolio.v2.validation import validate_economic_settings

FinancingRegime = Literal["none", "lend", "borrow"]


class V2LocalOptimizer:
    """Deterministic audited SLSQP/HRP solver with explicit financing regimes."""

    _SHRUNK_MU_METHODS = {"bayes_stein", "james_stein", "bodnar_okhrin"}
    _MEAN_ESTIMATORS = _SHRUNK_MU_METHODS | {
        "auto",
        "equilibrium",
        "empirical",
    }
    #: `max_ratio` and an exact volatility target are non-convex problems
    #: (Sharpe ratio and an equality constraint on a quadratic form); SLSQP
    #: has no global-optimality guarantee there. A fixed, deterministic seed
    #: keeps every solve reproducible (same bounds -> same random starts)
    #: while still diversifying the search beyond the structured starts.
    _RANDOM_RESTART_COUNT = 8
    _RANDOM_RESTART_SEED = 1_337

    def solve(
        self,
        frame: Any,
        *,
        objective: str,
        constraints: V2Constraints,
        periods_per_year: float,
        target_reference_series: Any | None,
        cap_reference_series: Any | None,
        tracking_reference_series: Any | None,
        reference_weights: dict[str, float] | None,
        risk_aversion: float = 1.0,
        risk_free_rate: float = 0.0,
        bound_aliases: dict[str, str] | None = None,
    ) -> tuple[dict[str, float], V2Audit]:
        validate_economic_settings(risk_aversion, risk_free_rate)
        constraints = self._normalize_direct_constraints(constraints)
        if constraints.covariance_estimator not in {
            "shrunk_fixed",
            "ledoit_wolf",
        }:
            raise V2OptimizationError(
                "unsupported covariance_estimator "
                f"{constraints.covariance_estimator!r}"
            )
        if constraints.view_covariance_policy not in {
            "prior_risk",
            "posterior_all",
        }:
            raise V2OptimizationError(
                "unsupported view_covariance_policy "
                f"{constraints.view_covariance_policy!r}"
            )
        if not isfinite(constraints.view_tau) or constraints.view_tau <= 0.0:
            raise V2OptimizationError("view_tau must be positive and finite")

        financing_active = (
            constraints.cash_enabled
            or constraints.max_leverage > 1.0
            or constraints.borrow_spread_bps > 0.0
        )
        if not financing_active:
            return self._solve_one(
                frame,
                objective=objective,
                constraints=constraints,
                periods_per_year=periods_per_year,
                target_reference_series=target_reference_series,
                cap_reference_series=cap_reference_series,
                tracking_reference_series=tracking_reference_series,
                reference_weights=reference_weights,
                risk_aversion=risk_aversion,
                risk_free_rate=risk_free_rate,
                bound_aliases=bound_aliases,
                regime="none",
            )

        if objective == "hrp":
            raise V2OptimizationError(
                "objective='hrp' does not support cash or leverage"
            )
        if not isfinite(constraints.max_leverage) or constraints.max_leverage < 1.0:
            raise V2OptimizationError(
                "max_leverage must be finite and at least 1.0"
            )
        if (
            not isfinite(constraints.borrow_spread_bps)
            or constraints.borrow_spread_bps < 0.0
        ):
            raise V2OptimizationError(
                "borrow_spread_bps must be finite and non-negative"
            )
        if (
            constraints.borrow_spread_bps > 0.0
            and not constraints.cash_enabled
            and constraints.max_leverage <= 1.0
        ):
            raise V2OptimizationError(
                "borrow_spread_bps requires cash_enabled or max_leverage > 1"
            )
        for view in constraints.views:
            if any(name in CASH_NAMES for name in view.instruments):
                raise V2OptimizationError(
                    "Black-Litterman views cannot target financing instruments"
                )

        configured_mean = constraints.mean_estimator
        effective_constraints = constraints
        if configured_mean == "auto":
            effective_constraints = replace(
                constraints,
                mean_estimator="bayes_stein",
            )
        candidates: list[tuple[dict[str, float], V2Audit]] = []
        failures: list[str] = []

        def run_regime(
            cash_name: str,
            annual_rate: float,
            regime: Literal["lend", "borrow"],
        ) -> None:
            augmented = frame.copy()
            augmented[cash_name] = annual_rate / periods_per_year
            try:
                weights, audit = self._solve_one(
                    augmented,
                    objective=objective,
                    constraints=effective_constraints,
                    periods_per_year=periods_per_year,
                    target_reference_series=target_reference_series,
                    cap_reference_series=cap_reference_series,
                    tracking_reference_series=tracking_reference_series,
                    reference_weights=reference_weights,
                    risk_aversion=risk_aversion,
                    risk_free_rate=risk_free_rate,
                    bound_aliases=bound_aliases,
                    regime=regime,
                    configured_mean_override=configured_mean,
                    mean_reason_override=(
                        "auto_for_cash_financing_uses_bayes_stein"
                        if configured_mean == "auto"
                        else "explicit_estimator"
                    ),
                )
            except V2OptimizationError as exc:
                failures.append(f"{regime}: {exc}")
                return

            cash_weight = float(weights.get(cash_name, 0.0))
            risky_gross = float(
                sum(
                    weight
                    for instrument, weight in weights.items()
                    if instrument not in CASH_NAMES
                )
            )
            candidates.append(
                (
                    weights,
                    replace(
                        audit,
                        cash_enabled=True,
                        cash_instrument=cash_name,
                        cash_weight=cash_weight,
                        risky_gross_exposure=risky_gross,
                        max_leverage=constraints.max_leverage,
                        cash_lending_rate=risk_free_rate,
                        cash_borrowing_rate=(
                            risk_free_rate
                            + constraints.borrow_spread_bps / 10_000.0
                        ),
                        borrow_spread_bps=constraints.borrow_spread_bps,
                        financing_regime=(
                            "cash_lending"
                            if regime == "lend"
                            else "cash_borrowing"
                        ),
                    ),
                )
            )

        run_regime(CASH_LEND, risk_free_rate, "lend")
        if constraints.max_leverage > 1.0:
            run_regime(
                CASH_BORROW,
                risk_free_rate + constraints.borrow_spread_bps / 10_000.0,
                "borrow",
            )
        if not candidates:
            detail = "; ".join(failures) or "no financing regime was evaluated"
            raise V2OptimizationError(
                f"no audited financing regime is feasible ({detail})"
            )

        def economic_score(item: tuple[dict[str, float], V2Audit]) -> float:
            audit = item[1]
            if audit.effective_objective == "min_risk":
                return -float(audit.objective_value)
            return float(audit.objective_value)

        return max(candidates, key=economic_score)

    @staticmethod
    def _normalize_direct_constraints(constraints: V2Constraints) -> V2Constraints:
        mode = constraints.volatility_target_mode
        if mode == "at_most":
            mode = "cap"
        if mode not in {"exact", "cap"}:
            raise V2OptimizationError(
                f"unsupported volatility_target_mode {mode!r}"
            )
        if mode == "cap" and constraints.volatility_target is not None:
            if constraints.max_volatility is not None:
                raise V2OptimizationError(
                    "cap mode cannot declare both volatility_target and max_volatility"
                )
            return replace(
                constraints,
                volatility_target=None,
                volatility_reference="none",
                max_volatility=constraints.volatility_target,
                max_volatility_reference=constraints.volatility_reference,
                volatility_target_mode="cap",
            )
        if mode != constraints.volatility_target_mode:
            return replace(constraints, volatility_target_mode=mode)
        return constraints

    def _solve_one(
        self,
        frame: Any,
        *,
        objective: str,
        constraints: V2Constraints,
        periods_per_year: float,
        target_reference_series: Any | None,
        cap_reference_series: Any | None,
        tracking_reference_series: Any | None,
        reference_weights: dict[str, float] | None,
        risk_aversion: float,
        risk_free_rate: float,
        bound_aliases: dict[str, str] | None,
        regime: FinancingRegime,
        configured_mean_override: str | None = None,
        mean_reason_override: str | None = None,
    ) -> tuple[dict[str, float], V2Audit]:
        configured_mean = configured_mean_override or constraints.mean_estimator
        effective_constraints = constraints
        if objective == "max_utility" and configured_mean == "auto":
            effective_constraints = replace(
                constraints,
                mean_estimator="bayes_stein",
            )
            mean_reason = "auto_for_max_utility_avoids_reference_reconstruction"
        elif configured_mean == "auto":
            mean_reason = "auto_resolved_from_reference_availability"
        else:
            mean_reason = "explicit_estimator"
        if mean_reason_override is not None:
            mean_reason = mean_reason_override

        weights, audit = self._solve_core(
            frame,
            objective=objective,
            constraints=effective_constraints,
            periods_per_year=periods_per_year,
            target_reference_series=target_reference_series,
            cap_reference_series=cap_reference_series,
            tracking_reference_series=tracking_reference_series,
            reference_weights=reference_weights,
            risk_aversion=risk_aversion,
            risk_free_rate=risk_free_rate,
            bound_aliases=bound_aliases,
            regime=regime,
        )
        return weights, replace(
            audit,
            configured_mean_estimator=configured_mean,
            mean_resolution_reason=mean_reason,
        )

    def _solve_core(
        self,
        frame: Any,
        *,
        objective: str,
        constraints: V2Constraints,
        periods_per_year: float,
        target_reference_series: Any | None,
        cap_reference_series: Any | None,
        tracking_reference_series: Any | None,
        reference_weights: dict[str, float] | None,
        risk_aversion: float,
        risk_free_rate: float,
        bound_aliases: dict[str, str] | None,
        regime: FinancingRegime,
    ) -> tuple[dict[str, float], V2Audit]:
        from scipy.optimize import minimize

        if objective not in RECOGNIZED_OBJECTIVES:
            raise V2OptimizationError(
                f"unsupported objective {objective!r}; must be one of "
                f"{sorted(RECOGNIZED_OBJECTIVES)}"
            )
        problem_class = classify(objective, constraints)
        clean = frame.dropna(how="any")
        if len(clean) < 3:
            raise V2OptimizationError(
                "local solve requires at least three complete observations"
            )
        names = list(clean.columns)
        aliases = bound_aliases or {}
        if objective == "hrp":
            return self._solve_hrp(
                clean,
                names,
                constraints,
                periods_per_year,
                aliases,
                risk_aversion,
                risk_free_rate,
            )
        values = clean.to_numpy(dtype=float)
        covariance, means, resolved_mean_estimator = estimate_moments(
            clean,
            names,
            reference_weights,
            constraints.mean_estimator,
            risk_aversion,
            risk_free_rate / periods_per_year,
            constraints.covariance_estimator,
        )
        covariance, means, view_details = apply_views(
            covariance,
            means,
            names,
            constraints.views,
            constraints.view_tau,
            periods_per_year,
            constraints.view_covariance_policy,
        )
        annualizer = sqrt(periods_per_year)
        risk_free_periodic = risk_free_rate / periods_per_year
        excess_means = means - risk_free_periodic
        lower, upper = self._bounds_for_regime(
            names,
            constraints,
            aliases,
            regime,
        )

        def aligned(series: Any | None, label: str) -> np.ndarray | None:
            if series is None:
                return None
            result = series.reindex(clean.index).to_numpy(dtype=float)
            if np.isnan(result).any():
                raise V2OptimizationError(
                    f"{label} reference series is incomplete in the estimation window"
                )
            return np.asarray(result, dtype=float)

        tracking_reference = aligned(tracking_reference_series, "TEV")

        def volatility(weights: np.ndarray) -> float:
            return float(sqrt(max(float(weights @ covariance @ weights), 0.0)))

        def realized_volatility(weights: np.ndarray) -> float:
            # Target/cap/TEV feasibility must be measured with the same risk
            # measure the target/cap itself is computed from (the reference
            # series' own historical std, see `_resolve_volatility`), never
            # the shrunk-covariance-implied figure `volatility()` returns:
            # shrinkage changes a single-asset "portfolio's" implied variance
            # relative to that asset's own raw historical variance, so a
            # 100%-proxy allocation would otherwise fail its own target by
            # construction. `volatility()` remains the risk measure driving
            # weight estimation (SLSQP loss for min_risk/max_ratio/
            # max_utility) -- only target/cap/TEV feasibility switches.
            return float(np.std(values @ weights, ddof=1))

        def tracking_error(weights: np.ndarray) -> float:
            if tracking_reference is None:
                return 0.0
            active = values @ weights - tracking_reference
            return float(np.std(active, ddof=1))

        target_annual = self._resolve_volatility(
            constraints.volatility_reference,
            constraints.volatility_target,
            target_reference_series,
            periods_per_year,
        )
        target_periodic = (
            target_annual / annualizer if target_annual is not None else None
        )
        cap_annual = self._resolve_volatility(
            constraints.max_volatility_reference,
            constraints.max_volatility,
            cap_reference_series,
            periods_per_year,
        )
        cap_periodic = cap_annual / annualizer if cap_annual is not None else None
        tev_periodic = (
            constraints.max_tracking_error / annualizer
            if constraints.max_tracking_error is not None
            else None
        )
        # Gate every fast path on the *resolved* target/cap/TEV values, not
        # problem_class.volatility_mode / has_tev: those only look at
        # constraints.volatility_target/max_volatility, which stay None for
        # a father_proxy/benchmark-referenced target (the common case for
        # non-root nodes) even though target_periodic above is then
        # non-None. Gating on the raw classifier fields let a fast path fire
        # on vol-constrained nodes and silently ignore the constraint.
        linear_constraints_only = (
            regime == "none"
            and target_periodic is None
            and cap_periodic is None
            and tev_periodic is None
        )
        # Deterministic seed for the common case where a father/benchmark
        # reference series *is* one of this node's own candidate instruments
        # (e.g. a Precious Metals node holding GLD, SLV, PPLT, PALL with
        # father proxy GLD): 100% on that instrument is provably feasible
        # and provably matches its own reference by construction. Hand SLSQP
        # that witness directly via the reference series' own `.name`
        # (set by pandas when `_risk_reference` slices `returns[node.proxy]`
        # -- a multi-weight benchmark series has no single `.name` and is
        # correctly skipped here) rather than relying on multi-start to
        # rediscover it by luck.
        proxy_seed_names: set[str] = set()
        for reference_series in (
            target_reference_series,
            cap_reference_series,
            tracking_reference_series,
        ):
            candidate_name = getattr(reference_series, "name", None)
            if isinstance(candidate_name, str) and candidate_name in names:
                proxy_seed_names.add(candidate_name)
        proxy_seeds: list[np.ndarray] = []
        for proxy_name in sorted(proxy_seed_names):
            seed = np.zeros(len(names))
            seed[names.index(proxy_name)] = 1.0
            if np.all(seed >= lower - 1e-9) and np.all(seed <= upper + 1e-9):
                proxy_seeds.append(seed)
        if linear_constraints_only and problem_class.objective == "max_return":
            lp_result = self._solve_lp_max_return(
                names,
                means,
                lower,
                upper,
                covariance,
                periods_per_year,
                risk_aversion,
                risk_free_rate,
                constraints,
                resolved_mean_estimator,
                view_details,
                problem_class.label,
            )
            if lp_result is not None:
                return lp_result
            # HiGHS failing (rare -- a well-posed LP with a bounded feasible
            # region should always succeed) falls through to the audited
            # SLSQP path below rather than raising, so an LP hiccup never
            # aborts a solve the existing multi-start search could handle.
        elif linear_constraints_only and problem_class.objective in (
            "min_risk",
            "max_utility",
        ):
            qp_result = self._solve_qp_convex(
                objective,
                names,
                means,
                excess_means,
                lower,
                upper,
                covariance,
                periods_per_year,
                risk_aversion,
                risk_free_rate,
                constraints,
                resolved_mean_estimator,
                view_details,
                problem_class.label,
            )
            if qp_result is not None:
                return qp_result
            # OSQP missing or failing falls through to the audited SLSQP
            # path below, exactly like the LP route's own fallback.
        elif (
            regime == "none"
            and problem_class.objective == "max_return"
            and target_periodic is None
            and (cap_periodic is not None or tev_periodic is not None)
        ):
            # v3 performance roadmap Phase C1: max_return with a volatility
            # *cap* (never an exact equality target -- that's the hybrid
            # warm-start below, C2) and/or a TEV cap, no financing, is a
            # convex SOCP. Building block only: live inspection of the 3
            # real benchmark trees found 0 nodes using cap mode (every
            # father/benchmark-relative target defaults to exact), so this
            # route alone has no measured real-tree benefit -- see Phase C
            # in the roadmap doc for why C1+C2 were merged.
            socp_result = self._solve_socp_max_return_cap(
                names,
                means,
                lower,
                upper,
                values,
                periods_per_year,
                risk_aversion,
                risk_free_rate,
                constraints,
                cap_periodic,
                cap_annual,
                tev_periodic,
                tracking_reference,
                resolved_mean_estimator,
                view_details,
                problem_class.label,
            )
            if socp_result is not None:
                return socp_result
            # Clarabel missing/failing/infeasible falls through to the
            # audited SLSQP path below, same fallback contract as LP/QP.
        start = self._initial_weights(names, lower, upper, reference_weights)
        hard_constraints: list[dict[str, Any]] = [
            {"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)}
        ]
        if cap_periodic is not None:
            hard_constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda weights: cap_periodic - realized_volatility(weights),
                }
            )
        base_constraints = list(hard_constraints)
        if tev_periodic is not None:
            if tracking_reference is None:
                raise V2OptimizationError(
                    "TEV limit requires an explicit reference series"
                )
            base_constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda weights: tev_periodic - tracking_error(weights),
                }
            )
        scipy_constraints = list(base_constraints)
        if target_periodic is not None:
            scipy_constraints.append(
                {
                    "type": "eq",
                    "fun": lambda weights: (
                        realized_volatility(weights) - target_periodic
                    ),
                }
            )

        def loss(weights: np.ndarray) -> float:
            if target_periodic is not None or objective == "max_return":
                return -float(excess_means @ weights) * 1_000.0
            if objective == "max_ratio":
                return -float(excess_means @ weights) / max(
                    volatility(weights),
                    1e-12,
                )
            if objective == "max_utility":
                return -(
                    float(excess_means @ weights)
                    - (risk_aversion / 2.0)
                    * float(weights @ covariance @ weights)
                )
            return volatility(weights) ** 2

        boundary_starts = self._bounded_starts(names, lower, upper)
        randomized_starts = self._randomized_starts(
            lower,
            upper,
            self._RANDOM_RESTART_COUNT,
            self._RANDOM_RESTART_SEED,
        )
        candidates = [start, *proxy_seeds]
        warm_started = False
        if target_periodic is not None:
            # v3 performance roadmap Phase C2 (hybrid exact-target route):
            # solve the convex cap-relaxed problem (target as an upper
            # bound, not an equality) exactly via Clarabel before falling
            # back to the expensive multi-SLSQP-call search below. This is
            # the route that actually matters for real trees -- every
            # father/benchmark-relative target in the 3 benchmark trees
            # defaults to volatility_target_mode="exact" (see Phase C.1's
            # 0%-real-coverage finding), so this is where a real wall-clock
            # win has to come from, unlike the B1/B2/C1 fast paths.
            #
            # Only the "binding" case is used: if the cap-relaxed optimum
            # lands exactly at the target, an active inequality constraint
            # at that value already satisfies the equality-target problem,
            # and Clarabel is an exact convex solver, so no further search
            # is needed -- skip the expensive cap_results loop entirely
            # (verified live: ~10-25ms faster per solve on a 3-asset
            # fixture once measured correctly with the process's one-time
            # sklearn/cvxpy import cost warmed up first -- an earlier,
            # unwarmed measurement wrongly attributed several *seconds* of
            # cold-import overhead to this code path). Feeding a
            # non-binding SOCP point into `_frontier_starts` as an extra
            # seed was considered too, but not implemented: `candidates`
            # always gets `_RANDOM_RESTART_COUNT` randomized starts appended
            # regardless of this branch (see below), so a non-binding SOCP
            # seed only adds 2 more `_frontier_starts` sub-solves without a
            # verified payoff -- left for a later pass once real-tree
            # benchmarking (Phase A1's harness) shows it's worth the risk.
            socp_candidate = None
            if regime == "none":
                socp_weights = self._solve_socp_cap_weights(
                    names, means, lower, upper, values,
                    target_periodic, tracking_reference, tev_periodic,
                )
                if socp_weights is not None and self._is_feasible(
                    socp_weights, lower, upper, realized_volatility,
                    None, target_periodic, tracking_error, tev_periodic,
                ):
                    socp_candidate = socp_weights
            if socp_candidate is not None and abs(
                realized_volatility(socp_candidate) - target_periodic
            ) <= 2e-6:
                candidates = [socp_candidate]
                warm_started = True
            else:
                target_cap_constraints = [
                    *base_constraints,
                    {
                        "type": "ineq",
                        "fun": lambda weights: (
                            target_periodic - realized_volatility(weights)
                        ),
                    },
                ]
                cap_results = []
                for cap_start in [start, *proxy_seeds, *boundary_starts[:2]]:
                    cap_result = minimize(
                        loss,
                        cap_start,
                        method="SLSQP",
                        bounds=list(zip(lower, upper, strict=True)),
                        constraints=target_cap_constraints,
                        options={"ftol": 1e-12, "maxiter": 2_000, "disp": False},
                    )
                    if cap_result.success and self._is_feasible(
                        cap_result.x,
                        lower,
                        upper,
                        realized_volatility,
                        None,
                        target_periodic,
                        tracking_error,
                        tev_periodic,
                    ):
                        cap_results.append(cap_result)
                binding = [
                    result
                    for result in cap_results
                    if abs(realized_volatility(result.x) - target_periodic) <= 2e-6
                ]
                if binding:
                    candidates = [min(binding, key=lambda item: float(item.fun)).x]
                else:
                    candidates.extend(
                        self._frontier_starts(
                            [start, *proxy_seeds, *boundary_starts[:2]],
                            means,
                            lower,
                            upper,
                            base_constraints,
                            realized_volatility,
                            target_periodic,
                            tracking_error,
                            tev_periodic,
                            cap_periodic,
                        )
                    )
        else:
            candidates.extend(boundary_starts[:2])
        candidates.extend(randomized_starts)

        solve_started = time.perf_counter()
        accepted: list[Any] = []
        for candidate in candidates:
            result = minimize(
                loss,
                candidate,
                method="SLSQP",
                bounds=list(zip(lower, upper, strict=True)),
                constraints=scipy_constraints,
                options={"ftol": 1e-12, "maxiter": 2_000, "disp": False},
            )
            if result.success and self._is_feasible(
                result.x,
                lower,
                upper,
                realized_volatility,
                target_periodic,
                cap_periodic,
                tracking_error,
                tev_periodic,
            ):
                accepted.append(result)
        projected = False
        stage_results: list[dict[str, Any]] = []
        if accepted:
            result = min(accepted, key=lambda item: float(item.fun))
        elif target_periodic is not None or tev_periodic is not None:
            result = self._lexicographic_fallback(
                [start, *proxy_seeds, *boundary_starts, *randomized_starts],
                loss,
                lower,
                upper,
                hard_constraints,
                realized_volatility,
                target_periodic,
                tracking_error,
                tev_periodic,
                constraints.tracking_error_policy,
                constraints.volatility_target_policy,
                stage_results,
            )
            projected = True
        else:
            raise V2OptimizationError(
                "local optimiser found no audited feasible solution"
            )
        solve_seconds = time.perf_counter() - solve_started

        restart_candidate_count = len(candidates)
        restart_objective_spread = 0.0
        if accepted and len(accepted) >= 2:
            restart_values = sorted(float(item.fun) for item in accepted)
            restart_objective_spread = abs(restart_values[1] - restart_values[0])

        weights = np.asarray(result.x, dtype=float)
        weights[np.abs(weights) < 1e-10] = 0.0
        weights /= weights.sum()
        self._audit_hard_constraints(
            weights, lower, upper, cap_periodic, realized_volatility
        )
        # `actual_vol` switches to the realized measure only when a
        # target/cap was actually enforced above, so the reported
        # actual_volatility never contradicts what the constraint checked
        # (e.g. exceeding volatility_cap just because a *different* risk
        # measure was used to report vs. enforce it) -- see
        # `realized_volatility`'s docstring. When no target/cap is active,
        # `actual_vol` must stay the shrunk-covariance figure: it is what
        # `loss()` actually optimized for min_risk/max_utility, what the QP
        # route (`_solve_qp_convex`) reports for the same unconstrained
        # problems, and what max_ratio's Sharpe denominator used internally
        # -- switching it there would make objective_value self-inconsistent
        # with actual_volatility, and inconsistent across routes/solves of
        # the identical unconstrained problem, with no target/cap bug to fix.
        actual_vol = (
            realized_volatility(weights)
            if target_periodic is not None or cap_periodic is not None
            else volatility(weights)
        ) * annualizer
        actual_tev = (
            tracking_error(weights) * annualizer
            if tracking_reference is not None
            else None
        )
        expected_return_annualized = float(means @ weights) * periods_per_year
        effective_objective = (
            "max_return_at_target"
            if target_periodic is not None
            else objective
        )
        if effective_objective in {"max_return", "max_return_at_target"}:
            objective_value = expected_return_annualized
        elif effective_objective == "max_ratio":
            objective_value = (
                expected_return_annualized - risk_free_rate
            ) / max(actual_vol, 1e-12)
        elif effective_objective == "max_utility":
            objective_value = (
                expected_return_annualized
                - risk_free_rate
                - (risk_aversion / 2.0) * actual_vol**2
            )
        else:
            objective_value = actual_vol
        soft_violation = 0.0
        if target_annual is not None:
            soft_violation += (
                (actual_vol - target_annual) / max(abs(target_annual), 1e-8)
            ) ** 2
        if constraints.max_tracking_error is not None and actual_tev is not None:
            soft_violation += (
                max(actual_tev - constraints.max_tracking_error, 0.0)
                / max(abs(constraints.max_tracking_error), 1e-8)
            ) ** 2

        posterior_all = (
            constraints.view_covariance_policy == "posterior_all"
            and bool(constraints.views)
        )
        audit = V2Audit(
            target_reference=constraints.volatility_reference,
            target_volatility=target_annual,
            actual_volatility=actual_vol,
            cap_reference=constraints.max_volatility_reference,
            volatility_cap=cap_annual,
            tracking_error_limit=constraints.max_tracking_error,
            actual_tracking_error=actual_tev,
            minimum_slack={
                name: float(weights[index] - lower[index])
                for index, name in enumerate(names)
            },
            maximum_slack={
                name: float(upper[index] - weights[index])
                for index, name in enumerate(names)
            },
            sum_weights=float(weights.sum()),
            solver_message=(
                f"nearest feasible projection: {result.message}"
                if projected
                else str(result.message)
            ),
            target_status=(
                "not_requested"
                if target_periodic is None
                else (
                    "matched"
                    if abs(realized_volatility(weights) - target_periodic) <= 2e-6
                    else "nearest_feasible"
                )
            ),
            tracking_error_status=(
                "not_requested"
                if tev_periodic is None
                else (
                    "within_limit"
                    if tracking_error(weights) <= tev_periodic + 2e-6
                    else "nearest_feasible"
                )
            ),
            configured_objective=objective,
            effective_objective=effective_objective,
            expected_return_annualized=expected_return_annualized,
            objective_value=objective_value,
            soft_constraint_violation=soft_violation,
            configured_mean_estimator=constraints.mean_estimator,
            resolved_mean_estimator=resolved_mean_estimator,
            views_applied=len(view_details),
            view_details=view_details,
            risk_aversion=risk_aversion,
            risk_free_rate=risk_free_rate,
            covariance_estimator=constraints.covariance_estimator,
            covariance_estimator_class={
                "shrunk_fixed": "ShrunkCovariance",
                "ledoit_wolf": "LedoitWolf",
            }[constraints.covariance_estimator],
            risk_covariance_role="posterior" if posterior_all else "prior",
            objective_covariance_role="posterior" if posterior_all else "prior",
            view_covariance_policy=constraints.view_covariance_policy,
            volatility_target_mode=constraints.volatility_target_mode,
            global_optimality_claim=False,
            solver_strategy="slsqp_multistart_audited",
            constraint_stage_results=tuple(stage_results),
            restart_candidate_count=restart_candidate_count,
            restart_objective_spread=restart_objective_spread,
            problem_class=problem_class.label,
            solver_status="nearest_feasible_fallback" if projected else "ok",
            solve_seconds=solve_seconds,
            warm_started=warm_started,
            fallback_reason=(
                "no multi-start candidate satisfied hard constraints; used the "
                "lexicographic nearest-feasible projection"
                if projected
                else ""
            ),
        )
        return dict(zip(names, weights, strict=True)), audit

    @staticmethod
    def _solve_lp_max_return(
        names: list[str],
        means: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        covariance: np.ndarray,
        periods_per_year: float,
        risk_aversion: float,
        risk_free_rate: float,
        constraints: V2Constraints,
        resolved_mean_estimator: str,
        view_details: tuple[dict[str, Any], ...],
        problem_class_label: str,
    ) -> tuple[dict[str, float], V2Audit] | None:
        """Exact LP fast path: ``objective='max_return'``, no vol target/cap,
        no TEV, no financing, only budget + box bounds -- v3 performance
        roadmap Phase B (docs/hierarchical-optimizer-performance-plan.md).
        A linear objective over linear equality/bound constraints is a
        well-posed LP; HiGHS (``scipy.optimize.linprog``) is an exact global
        solver for it, unlike SLSQP's non-convex multi-start heuristic, so
        this route can honestly claim global optimality.

        Returns ``None`` (never raises) if HiGHS doesn't report success, so
        the caller falls back to the audited SLSQP path rather than
        silently accept a non-optimal result.
        """
        from scipy.optimize import linprog

        started = time.perf_counter()
        result = linprog(
            c=-means,
            A_eq=[np.ones(len(names))],
            b_eq=[1.0],
            bounds=list(zip(lower, upper, strict=True)),
            method="highs",
        )
        solve_seconds = time.perf_counter() - started
        if not result.success:
            return None

        weights = np.asarray(result.x, dtype=float)
        weights[np.abs(weights) < 1e-10] = 0.0
        total = weights.sum()
        if total <= 0.0:
            return None
        weights /= total
        annualizer = sqrt(periods_per_year)
        actual_vol = (
            float(sqrt(max(float(weights @ covariance @ weights), 0.0))) * annualizer
        )
        expected_return_annualized = float(means @ weights) * periods_per_year
        posterior_all = (
            constraints.view_covariance_policy == "posterior_all" and bool(constraints.views)
        )
        audit = V2Audit(
            target_reference="none",
            target_volatility=None,
            actual_volatility=actual_vol,
            cap_reference="none",
            volatility_cap=None,
            tracking_error_limit=None,
            actual_tracking_error=None,
            minimum_slack={
                name: float(weights[index] - lower[index]) for index, name in enumerate(names)
            },
            maximum_slack={
                name: float(upper[index] - weights[index]) for index, name in enumerate(names)
            },
            sum_weights=float(weights.sum()),
            solver_message=str(result.message),
            target_status="not_requested",
            tracking_error_status="not_requested",
            configured_objective="max_return",
            effective_objective="max_return",
            expected_return_annualized=expected_return_annualized,
            objective_value=expected_return_annualized,
            soft_constraint_violation=0.0,
            configured_mean_estimator=constraints.mean_estimator,
            resolved_mean_estimator=resolved_mean_estimator,
            views_applied=len(view_details),
            view_details=view_details,
            risk_aversion=risk_aversion,
            risk_free_rate=risk_free_rate,
            covariance_estimator=constraints.covariance_estimator,
            covariance_estimator_class={
                "shrunk_fixed": "ShrunkCovariance",
                "ledoit_wolf": "LedoitWolf",
            }[constraints.covariance_estimator],
            risk_covariance_role="posterior" if posterior_all else "prior",
            objective_covariance_role="posterior" if posterior_all else "prior",
            view_covariance_policy=constraints.view_covariance_policy,
            volatility_target_mode=constraints.volatility_target_mode,
            global_optimality_claim=True,
            solver_strategy="lp_highs",
            restart_candidate_count=1,
            restart_objective_spread=0.0,
            problem_class=problem_class_label,
            solver_status="ok",
            solve_seconds=solve_seconds,
            warm_started=False,
            fallback_reason="",
        )
        return dict(zip(names, weights, strict=True)), audit

    @staticmethod
    def _solve_qp_convex(
        objective: str,
        names: list[str],
        means: np.ndarray,
        excess_means: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        covariance: np.ndarray,
        periods_per_year: float,
        risk_aversion: float,
        risk_free_rate: float,
        constraints: V2Constraints,
        resolved_mean_estimator: str,
        view_details: tuple[dict[str, Any], ...],
        problem_class_label: str,
    ) -> tuple[dict[str, float], V2Audit] | None:
        """Exact QP fast path: ``objective`` in ``{'min_risk', 'max_utility'}``,
        no vol target/cap, no TEV, no financing, only budget + box bounds --
        v3 performance roadmap Phase B (docs/hierarchical-optimizer-performance-plan.md).
        Both objectives are convex quadratics over linear equality/bound
        constraints, a well-posed QP; OSQP is an exact convex solver for it,
        unlike SLSQP's non-convex multi-start heuristic, so this route can
        honestly claim global optimality.

        Mirrors the periodic-scale objective SLSQP's ``loss()`` minimizes for
        these two objectives exactly (same sign convention, same
        ``covariance``/``excess_means`` -- already view-adjusted by
        ``apply_views()`` upstream, never recomputed here):
        ``min_risk`` minimizes ``w'Sigma w``; ``max_utility`` minimizes
        ``(risk_aversion/2) w'Sigma w - excess_means'w``.

        Returns ``None`` (never raises) if OSQP isn't installed or doesn't
        report success, so the caller falls back to the audited SLSQP path
        rather than silently accept a non-optimal result.
        """
        try:
            import osqp
        except ImportError:  # pragma: no cover - optional fast path
            return None
        import scipy.sparse as sp

        count = len(names)
        # OSQP minimizes 0.5 x'Px + q'x, so P must equal 2x the target
        # quadratic form's own matrix (min_risk's w'Sigma w has none of its
        # own leading factor; max_utility's loss carries risk_aversion/2).
        # p_full already includes that factor of 2 -- average(p_full,
        # p_full.T) only defensively symmetrizes a not-quite-symmetric
        # covariance, it must NOT add another factor of 2 on top (summing
        # instead of averaging silently doubled max_utility's effective
        # risk_aversion, verified live against an independent multi-start
        # SLSQP cross-check: OSQP converged to a real, different, and worse
        # KKT point until this was fixed).
        if objective == "min_risk":
            p_full = 2.0 * covariance
            linear = np.zeros(count)
        else:
            p_full = risk_aversion * covariance
            linear = -excess_means
        p_matrix = sp.csc_matrix(0.5 * (p_full + p_full.T))
        a_matrix = sp.csc_matrix(np.vstack([np.ones((1, count)), np.eye(count)]))
        constraint_lower = np.concatenate([[1.0], lower])
        constraint_upper = np.concatenate([[1.0], upper])

        started = time.perf_counter()
        solver = osqp.OSQP()
        solver.setup(
            P=p_matrix,
            q=linear,
            A=a_matrix,
            l=constraint_lower,
            u=constraint_upper,
            verbose=False,
            # OSQP's default eps_abs/eps_rel (1e-3) are tuned for
            # O(1)-scale problems; a periodic covariance's entries are
            # O(1e-4) to O(1e-5), so the default tolerance lets OSQP report
            # "solved" tens of iterations before it reaches the true
            # optimum (verified live: default settings left ~30% excess
            # variance on a real fixture). Tighten to the estimator's own
            # precision and polish the active-set solution so this route's
            # global-optimality claim actually holds.
            eps_abs=1e-10,
            eps_rel=1e-10,
            max_iter=20_000,
            polishing=True,
        )
        # raise_error=False: this route already inspects result.info.status
        # itself and returns None (falling back to SLSQP) on anything but
        # "solved" -- never let OSQP raise instead of returning a status.
        result = solver.solve(raise_error=False)
        solve_seconds = time.perf_counter() - started
        if result.info.status != "solved":
            return None

        weights = np.asarray(result.x, dtype=float)
        weights[np.abs(weights) < 1e-10] = 0.0
        total = weights.sum()
        if total <= 0.0:
            return None
        weights /= total
        annualizer = sqrt(periods_per_year)
        actual_vol = (
            float(sqrt(max(float(weights @ covariance @ weights), 0.0))) * annualizer
        )
        expected_return_annualized = float(means @ weights) * periods_per_year
        if objective == "max_utility":
            objective_value = (
                expected_return_annualized
                - risk_free_rate
                - (risk_aversion / 2.0) * actual_vol**2
            )
        else:
            objective_value = actual_vol
        posterior_all = (
            constraints.view_covariance_policy == "posterior_all" and bool(constraints.views)
        )
        audit = V2Audit(
            target_reference="none",
            target_volatility=None,
            actual_volatility=actual_vol,
            cap_reference="none",
            volatility_cap=None,
            tracking_error_limit=None,
            actual_tracking_error=None,
            minimum_slack={
                name: float(weights[index] - lower[index]) for index, name in enumerate(names)
            },
            maximum_slack={
                name: float(upper[index] - weights[index]) for index, name in enumerate(names)
            },
            sum_weights=float(weights.sum()),
            solver_message=str(result.info.status),
            target_status="not_requested",
            tracking_error_status="not_requested",
            configured_objective=objective,
            effective_objective=objective,
            expected_return_annualized=expected_return_annualized,
            objective_value=objective_value,
            soft_constraint_violation=0.0,
            configured_mean_estimator=constraints.mean_estimator,
            resolved_mean_estimator=resolved_mean_estimator,
            views_applied=len(view_details),
            view_details=view_details,
            risk_aversion=risk_aversion,
            risk_free_rate=risk_free_rate,
            covariance_estimator=constraints.covariance_estimator,
            covariance_estimator_class={
                "shrunk_fixed": "ShrunkCovariance",
                "ledoit_wolf": "LedoitWolf",
            }[constraints.covariance_estimator],
            risk_covariance_role="posterior" if posterior_all else "prior",
            objective_covariance_role="posterior" if posterior_all else "prior",
            view_covariance_policy=constraints.view_covariance_policy,
            volatility_target_mode=constraints.volatility_target_mode,
            global_optimality_claim=True,
            solver_strategy="qp_osqp",
            restart_candidate_count=1,
            restart_objective_spread=0.0,
            problem_class=problem_class_label,
            solver_status="ok",
            solve_seconds=solve_seconds,
            warm_started=False,
            fallback_reason="",
        )
        return dict(zip(names, weights, strict=True)), audit

    @staticmethod
    def _solve_socp_cap_weights(
        names: list[str],
        means: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        values: np.ndarray,
        cap_periodic: float | None,
        tracking_reference: np.ndarray | None,
        tev_periodic: float | None,
    ) -> np.ndarray | None:
        """Exact convex core shared by the Phase C SOCP cap route (C1) and
        the hybrid exact-target warm start (C2, in ``_solve_core``):
        maximize ``means'w`` subject to budget + box bounds, an optional
        realized-volatility cap, and an optional realized-TEV cap, via
        Clarabel.

        Both risk constraints are built as the L2 norm of an affine
        expression in centered returns -- exactly ``realized_volatility``/
        ``tracking_error``'s own formula (``std(x) == ||demean(x)||_2 /
        sqrt(T-1)`` for ddof=1), not a covariance approximation, so this
        route's constraint is numerically identical to what SLSQP enforces
        elsewhere in ``_solve_core``, never a looser or tighter proxy for
        it (the whole point of Phase B.5's risk-measure-consistency fix).

        Returns ``None`` (never raises) on missing cvxpy/clarabel, solver
        failure, or infeasibility, so callers always have an audited SLSQP
        fallback.
        """
        try:
            import cvxpy as cp
        except ImportError:  # pragma: no cover - optional fast path
            return None

        count = len(names)
        scale = sqrt(max(len(values) - 1, 1))
        centered = values - values.mean(axis=0)
        # cvxpy's installed type stubs don't fully cover cp.sum/cp.norm/
        # Problem.solve -- this whole block is a lazily-imported, optional
        # fast path (never a required import), so the noise isn't worth
        # chasing; every value crossing back out of this function is
        # re-annotated as a plain np.ndarray below.
        w = cp.Variable(count)
        socp_constraints = [cp.sum(w) == 1.0, w >= lower, w <= upper]  # type: ignore[attr-defined]
        if cap_periodic is not None:
            socp_constraints.append(
                cp.norm(centered @ w, 2) <= cap_periodic * scale  # type: ignore[attr-defined]
            )
        if tev_periodic is not None:
            if tracking_reference is None:
                return None
            active_reference_centered = tracking_reference - tracking_reference.mean()
            socp_constraints.append(
                cp.norm(centered @ w - active_reference_centered, 2)  # type: ignore[attr-defined]
                <= tev_periodic * scale
            )
        problem = cp.Problem(cp.Maximize(means @ w), socp_constraints)
        try:
            problem.solve(solver=cp.CLARABEL)  # type: ignore[no-untyped-call]
        except cp.error.SolverError:
            return None
        if w.value is None or problem.status not in (
            cp.OPTIMAL,
            cp.OPTIMAL_INACCURATE,
        ):
            return None
        weights = np.asarray(w.value, dtype=float)
        weights[np.abs(weights) < 1e-10] = 0.0
        total = weights.sum()
        if total <= 0.0:
            return None
        weights /= total
        return weights

    @staticmethod
    def _solve_socp_max_return_cap(
        names: list[str],
        means: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        values: np.ndarray,
        periods_per_year: float,
        risk_aversion: float,
        risk_free_rate: float,
        constraints: V2Constraints,
        cap_periodic: float | None,
        cap_annual: float | None,
        tev_periodic: float | None,
        tracking_reference: np.ndarray | None,
        resolved_mean_estimator: str,
        view_details: tuple[dict[str, Any], ...],
        problem_class_label: str,
    ) -> tuple[dict[str, float], V2Audit] | None:
        """Exact SOCP fast path (v3 performance roadmap Phase C1):
        ``objective='max_return'`` with a volatility *cap* and/or TEV cap
        (never an exact equality target -- that's the hybrid warm start in
        ``_solve_core``, C2), no financing. Clarabel is an exact convex
        solver for this SOCP, so this route can honestly claim global
        optimality, same rationale as the LP/QP routes (B1/B2).
        """
        started = time.perf_counter()
        weights = V2LocalOptimizer._solve_socp_cap_weights(
            names, means, lower, upper, values, cap_periodic,
            tracking_reference, tev_periodic,
        )
        solve_seconds = time.perf_counter() - started
        if weights is None:
            return None

        annualizer = sqrt(periods_per_year)
        realized_vol = float(np.std(values @ weights, ddof=1)) * annualizer
        # Defensive: never report a route as having satisfied a constraint
        # it actually violated (e.g. a near-boundary numerical slip).
        if cap_annual is not None and realized_vol > cap_annual + 1e-6:
            return None
        expected_return_annualized = float(means @ weights) * periods_per_year
        actual_tev = None
        tracking_error_status = "not_requested"
        if tracking_reference is not None:
            active = values @ weights - tracking_reference
            actual_tev = float(np.std(active, ddof=1)) * annualizer
            if constraints.max_tracking_error is not None:
                if actual_tev > constraints.max_tracking_error + 1e-6:
                    return None
                tracking_error_status = "within_limit"
        posterior_all = (
            constraints.view_covariance_policy == "posterior_all"
            and bool(constraints.views)
        )
        audit = V2Audit(
            target_reference="none",
            target_volatility=None,
            actual_volatility=realized_vol,
            cap_reference=constraints.max_volatility_reference,
            volatility_cap=cap_annual,
            tracking_error_limit=constraints.max_tracking_error,
            actual_tracking_error=actual_tev,
            minimum_slack={
                name: float(weights[index] - lower[index]) for index, name in enumerate(names)
            },
            maximum_slack={
                name: float(upper[index] - weights[index]) for index, name in enumerate(names)
            },
            sum_weights=float(weights.sum()),
            solver_message="clarabel SOCP: optimal",
            target_status="not_requested",
            tracking_error_status=tracking_error_status,
            configured_objective="max_return",
            effective_objective="max_return",
            expected_return_annualized=expected_return_annualized,
            objective_value=expected_return_annualized,
            soft_constraint_violation=0.0,
            configured_mean_estimator=constraints.mean_estimator,
            resolved_mean_estimator=resolved_mean_estimator,
            views_applied=len(view_details),
            view_details=view_details,
            risk_aversion=risk_aversion,
            risk_free_rate=risk_free_rate,
            covariance_estimator=constraints.covariance_estimator,
            covariance_estimator_class={
                "shrunk_fixed": "ShrunkCovariance",
                "ledoit_wolf": "LedoitWolf",
            }[constraints.covariance_estimator],
            risk_covariance_role="posterior" if posterior_all else "prior",
            objective_covariance_role="posterior" if posterior_all else "prior",
            view_covariance_policy=constraints.view_covariance_policy,
            volatility_target_mode=constraints.volatility_target_mode,
            global_optimality_claim=True,
            solver_strategy="socp_clarabel",
            restart_candidate_count=1,
            restart_objective_spread=0.0,
            problem_class=problem_class_label,
            solver_status="ok",
            solve_seconds=solve_seconds,
            warm_started=False,
            fallback_reason="",
        )
        return dict(zip(names, weights, strict=True)), audit

    @classmethod
    def _solve_hrp(
        cls,
        clean: Any,
        names: list[str],
        constraints: V2Constraints,
        periods_per_year: float,
        aliases: dict[str, str],
        risk_aversion: float,
        risk_free_rate: float,
    ) -> tuple[dict[str, float], V2Audit]:
        if constraints.covariance_estimator != "shrunk_fixed":
            raise V2OptimizationError(
                "objective='hrp' currently supports "
                "covariance_estimator='shrunk_fixed' only"
            )
        if (
            constraints.volatility_reference != "none"
            or constraints.max_volatility_reference != "none"
        ):
            raise V2OptimizationError(
                "objective='hrp' does not support a volatility target or cap"
            )
        if constraints.max_tracking_error is not None:
            raise V2OptimizationError(
                "objective='hrp' does not support a tracking-error limit"
            )
        if constraints.mean_estimator != "auto":
            raise V2OptimizationError(
                "objective='hrp' does not use an expected-return estimator; "
                "leave mean_estimator at 'auto'"
            )
        if constraints.views:
            raise V2OptimizationError("objective='hrp' does not use views")

        from skfolio.moments import ShrunkCovariance
        from skfolio.optimization import HierarchicalRiskParity
        from skfolio.prior import EmpiricalPrior

        lower, upper = cls._bounds_for_regime(
            names,
            constraints,
            aliases,
            "none",
        )
        estimator = HierarchicalRiskParity(
            prior_estimator=EmpiricalPrior(
                covariance_estimator=ShrunkCovariance()
            ),
            min_weights=dict(zip(names, lower, strict=True)),
            max_weights=dict(zip(names, upper, strict=True)),
        )
        solve_started = time.perf_counter()
        estimator.fit(clean)
        solve_seconds = time.perf_counter() - solve_started
        weights = np.asarray(estimator.weights_, dtype=float)
        if not np.all(np.isfinite(weights)):
            raise V2OptimizationError("HRP returned non-finite weights")
        if abs(float(weights.sum()) - 1.0) > 2e-6:
            raise V2OptimizationError("HRP weights are not fully invested")
        if np.any(weights < lower - 2e-6) or np.any(weights > upper + 2e-6):
            raise V2OptimizationError("HRP weights violate declared bounds")

        covariance, means, _ = estimate_moments(
            clean,
            names,
            None,
            "auto",
            risk_aversion,
            risk_free_rate / periods_per_year,
            "shrunk_fixed",
        )
        annualizer = sqrt(periods_per_year)
        actual_vol = (
            float(sqrt(max(float(weights @ covariance @ weights), 0.0)))
            * annualizer
        )
        expected_return_annualized = float(means @ weights) * periods_per_year
        audit = V2Audit(
            target_reference="none",
            target_volatility=None,
            actual_volatility=actual_vol,
            cap_reference="none",
            volatility_cap=None,
            tracking_error_limit=None,
            actual_tracking_error=None,
            minimum_slack={
                name: float(weights[index] - lower[index])
                for index, name in enumerate(names)
            },
            maximum_slack={
                name: float(upper[index] - weights[index])
                for index, name in enumerate(names)
            },
            sum_weights=float(weights.sum()),
            solver_message="hrp",
            target_status="not_requested",
            tracking_error_status="not_requested",
            configured_objective="hrp",
            effective_objective="hrp",
            expected_return_annualized=expected_return_annualized,
            objective_value=actual_vol,
            soft_constraint_violation=0.0,
            configured_mean_estimator=constraints.mean_estimator,
            resolved_mean_estimator="not_applicable",
            views_applied=0,
            view_details=(),
            risk_aversion=risk_aversion,
            risk_free_rate=risk_free_rate,
            covariance_estimator="shrunk_fixed",
            covariance_estimator_class="ShrunkCovariance",
            risk_covariance_role="prior",
            objective_covariance_role="prior",
            view_covariance_policy="prior_risk",
            mean_resolution_reason="not_applicable",
            volatility_target_mode=constraints.volatility_target_mode,
            global_optimality_claim=False,
            solver_strategy="skfolio_hrp_ward_pearson",
            hrp_distance_metric="pearson",
            hrp_linkage_method="ward",
            hrp_risk_measure="variance",
            problem_class=classify("hrp", constraints).label,
            solver_status="ok",
            solve_seconds=solve_seconds,
            warm_started=False,
            fallback_reason="",
        )
        return dict(zip(names, weights, strict=True)), audit

    @staticmethod
    def _bounds(
        names: list[str],
        constraints: V2Constraints,
        aliases: dict[str, str],
    ) -> tuple[np.ndarray, np.ndarray]:
        cash = [name for name in names if name in CASH_NAMES]
        if len(cash) > 1:
            raise V2OptimizationError(
                "a financing solve must contain exactly one cash instrument"
            )
        regime: FinancingRegime = "none"
        if cash == [CASH_LEND]:
            regime = "lend"
        elif cash == [CASH_BORROW]:
            regime = "borrow"
        return V2LocalOptimizer._bounds_for_regime(
            names,
            constraints,
            aliases,
            regime,
        )

    @staticmethod
    def _bounds_for_regime(
        names: list[str],
        constraints: V2Constraints,
        aliases: dict[str, str],
        regime: FinancingRegime,
    ) -> tuple[np.ndarray, np.ndarray]:
        cash_positions = [
            index for index, name in enumerate(names) if name in CASH_NAMES
        ]
        if not cash_positions:
            lower = np.array(
                [
                    constraints.min_weights.get(aliases.get(name, name), 0.0)
                    for name in names
                ],
                dtype=float,
            )
            upper = np.array(
                [
                    min(
                        constraints.max_weights.get(
                            aliases.get(name, name),
                            1.0,
                        ),
                        constraints.per_asset_cap
                        if constraints.per_asset_cap is not None
                        else 1.0,
                    )
                    for name in names
                ],
                dtype=float,
            )
            if (
                np.any(lower < 0.0)
                or np.any(upper > 1.0)
                or np.any(lower > upper)
            ):
                raise V2OptimizationError(
                    "invalid local min/max allocation bounds"
                )
            if lower.sum() > 1.0 + 1e-10 or upper.sum() < 1.0 - 1e-10:
                raise V2OptimizationError(
                    "local min/max allocation bounds cannot sum to one"
                )
            return lower, upper
        if len(cash_positions) != 1:
            raise V2OptimizationError(
                "a financing solve must contain exactly one cash instrument"
            )

        lower = np.zeros(len(names), dtype=float)
        upper = np.ones(len(names), dtype=float)
        risky = np.ones(len(names), dtype=bool)
        for index, name in enumerate(names):
            if name == CASH_LEND:
                if regime != "lend":
                    raise V2OptimizationError("cash lending regime mismatch")
                lower[index], upper[index] = 0.0, 1.0
                risky[index] = False
                continue
            if name == CASH_BORROW:
                if regime != "borrow":
                    raise V2OptimizationError("cash borrowing regime mismatch")
                lower[index] = 1.0 - constraints.max_leverage
                upper[index] = 0.0
                risky[index] = False
                continue
            alias = aliases.get(name, name)
            lower[index] = constraints.min_weights.get(alias, 0.0)
            upper[index] = min(
                constraints.max_weights.get(alias, 1.0),
                constraints.per_asset_cap
                if constraints.per_asset_cap is not None
                else 1.0,
            )
        if np.any(lower[risky] < -1e-12) or np.any(
            upper[risky] > 1.0 + 1e-12
        ):
            raise V2OptimizationError(
                "risky-asset bounds must remain in [0, 1] under financing"
            )
        if np.any(lower > upper):
            raise V2OptimizationError(
                "invalid local cash/risky allocation bounds"
            )
        if lower.sum() > 1.0 + 1e-10 or upper.sum() < 1.0 - 1e-10:
            raise V2OptimizationError(
                "cash/risky allocation bounds cannot satisfy net investment of one"
            )
        return lower, upper

    @classmethod
    def _estimate_moments(
        cls,
        clean: Any,
        names: list[str],
        reference_weights: dict[str, float] | None,
        mean_estimator: str,
        risk_aversion: float = 1.0,
        risk_free_periodic: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        del cls
        return estimate_moments(
            clean,
            names,
            reference_weights,
            mean_estimator,
            risk_aversion,
            risk_free_periodic,
            "shrunk_fixed",
        )

    @staticmethod
    def _apply_views(
        covariance: np.ndarray,
        means: np.ndarray,
        names: list[str],
        views: tuple[V2View, ...],
        tau: float,
        periods_per_year: float,
    ) -> tuple[np.ndarray, np.ndarray, tuple[dict[str, Any], ...]]:
        return black_litterman_posterior(
            covariance,
            means,
            names,
            views,
            tau,
            periods_per_year,
        )

    @staticmethod
    def _minimize_metric(
        metric: Any,
        starts: list[np.ndarray],
        lower: np.ndarray,
        upper: np.ndarray,
        constraints: list[dict[str, Any]],
    ) -> tuple[float, np.ndarray] | tuple[None, None]:
        """Minimize an arbitrary scalar metric subject to hard constraints.

        Used to determine the minimum achievable value of one relaxable
        constraint (TEV excess, volatility deviation) in isolation, before
        deciding — per that constraint's own policy — whether to fail or to
        pin the next stage to this minimum.
        """

        from scipy.optimize import minimize

        bounds = list(zip(lower, upper, strict=True))
        best_value: float | None = None
        best_weights: np.ndarray | None = None
        for start in starts:
            result = minimize(
                metric,
                start,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-14, "maxiter": 3_000, "disp": False},
            )
            if not result.success:
                continue
            weights = np.asarray(result.x, dtype=float)
            if abs(float(weights.sum()) - 1.0) > 2e-6:
                continue
            if np.any(weights < lower - 2e-6) or np.any(weights > upper + 2e-6):
                continue
            value = float(metric(weights))
            if best_value is None or value < best_value:
                best_value = value
                best_weights = weights
        if best_value is None or best_weights is None:
            return None, None
        return best_value, best_weights

    @classmethod
    def _lexicographic_fallback(
        cls,
        starts: list[np.ndarray],
        economic_loss: Any,
        lower: np.ndarray,
        upper: np.ndarray,
        hard_constraints: list[dict[str, Any]],
        volatility: Any,
        target: float | None,
        tracking_error: Any,
        tev_limit: float | None,
        tracking_error_policy: str,
        volatility_target_policy: str,
        stage_results: list[dict[str, Any]],
    ) -> Any:
        """Stage A (TEV) -> Stage B (volatility) -> Stage C (objective).

        Every stage is solved subject to the hard constraints (budget,
        volatility cap — never relaxed here) plus whatever bound the earlier
        stages fixed. A relaxable constraint whose own policy is
        ``hard_fail`` raises the moment it is found infeasible in isolation;
        it never reaches a nearest-feasible projection. ``nearest_feasible``
        pins the following stages to the minimum achievable excess/deviation
        — never a value smaller than what is fed forward, and never
        reoptimized away by a later stage.
        """

        from scipy.optimize import OptimizeResult, minimize

        bounds = list(zip(lower, upper, strict=True))
        stage_constraints = list(hard_constraints)
        tev_constraint_obj: dict[str, Any] | None = None
        vol_constraint_obj: dict[str, Any] | None = None
        # Witness points a stage's own projection already proved feasible for
        # its resolved bound -- seeding stage C's SLSQP restarts with these
        # (on top of the generic `starts`) fixes the common case where a
        # feasible point provably exists (this one) but none of the generic
        # starts happens to be near it, so every restart fails to converge
        # even though the constraint set itself is not empty.
        tev_weights: np.ndarray | None = None
        dev_weights: np.ndarray | None = None

        if tev_limit is not None:
            tev_min, tev_weights = cls._minimize_metric(
                tracking_error, starts, lower, upper, hard_constraints
            )
            if tev_min is None:
                raise V2OptimizationError(
                    "hard constraints are infeasible; TEV projection is unavailable"
                )
            if tev_min <= tev_limit + 2e-6:
                tev_bound = tev_limit
                stage_results.append(
                    {
                        "stage": "tracking_error",
                        "policy": tracking_error_policy,
                        "requested": tev_limit,
                        "achieved": tev_min,
                        "status": "within_limit",
                    }
                )
            elif tracking_error_policy == "hard_fail":
                raise V2OptimizationError(
                    "tracking-error limit is infeasible under hard constraints "
                    f"(minimum achievable {tev_min:.6f} exceeds limit {tev_limit:.6f}); "
                    "tracking_error_policy='hard_fail' does not allow a "
                    "nearest-feasible projection"
                )
            else:
                tev_bound = tev_min + 1e-9
                stage_results.append(
                    {
                        "stage": "tracking_error",
                        "policy": tracking_error_policy,
                        "requested": tev_limit,
                        "achieved": tev_min,
                        "status": "nearest_feasible",
                    }
                )
            tev_constraint_obj = {
                "type": "ineq",
                "fun": lambda weights, bound=tev_bound: (
                    bound - tracking_error(weights)
                ),
            }
            stage_constraints.append(tev_constraint_obj)

        if target is not None:

            def deviation(weights: np.ndarray, target: float = target) -> float:
                return float(abs(volatility(weights) - target))

            dev_min, dev_weights = cls._minimize_metric(
                deviation, starts, lower, upper, stage_constraints
            )
            if dev_min is None or dev_weights is None:
                raise V2OptimizationError(
                    "hard constraints (including any resolved TEV bound) are "
                    "infeasible; volatility target projection is unavailable"
                )
            if dev_min <= 2e-6:
                stage_results.append(
                    {
                        "stage": "volatility",
                        "policy": volatility_target_policy,
                        "requested": target,
                        "achieved": float(volatility(dev_weights)),
                        "status": "matched",
                    }
                )
                vol_constraint_obj = {
                    "type": "eq",
                    "fun": lambda weights, target=target: (
                        volatility(weights) - target
                    ),
                }
                stage_constraints.append(vol_constraint_obj)
            elif volatility_target_policy == "hard_fail":
                raise V2OptimizationError(
                    "volatility target is infeasible under hard constraints "
                    f"(minimum achievable deviation {dev_min:.6f}); "
                    "volatility_target_policy='hard_fail' does not allow a "
                    "nearest-feasible projection"
                )
            else:
                achieved_vol = float(volatility(dev_weights))
                band = max(dev_min, 1e-9) * 1e-6 + 1e-9
                stage_results.append(
                    {
                        "stage": "volatility",
                        "policy": volatility_target_policy,
                        "requested": target,
                        "achieved": achieved_vol,
                        "status": "nearest_feasible",
                    }
                )
                vol_constraint_obj = {
                    "type": "ineq",
                    "fun": lambda weights, achieved=achieved_vol, band=band: (
                        band - abs(volatility(weights) - achieved)
                    ),
                }
                stage_constraints.append(vol_constraint_obj)

        def _solve_objective(
            constraints: list[dict[str, Any]], extra_starts: list[np.ndarray] | None = None
        ) -> Any | None:
            refined: list[Any] = []
            for start in [*starts, *(extra_starts or [])]:
                candidate = minimize(
                    economic_loss,
                    start,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints,
                    options={"ftol": 1e-12, "maxiter": 3_000, "disp": False},
                )
                if candidate.success:
                    weights = np.asarray(candidate.x, dtype=float)
                    if (
                        abs(float(weights.sum()) - 1.0) <= 2e-6
                        and not np.any(weights < lower - 2e-6)
                        and not np.any(weights > upper + 2e-6)
                    ):
                        refined.append(candidate)
            return min(refined, key=lambda item: float(item.fun)) if refined else None

        # dev_weights (stage B's own witness) already satisfies stage_constraints
        # in full -- it was found *inside* the TEV bound (if any) and its own
        # volatility deviation from the target is exactly what the vol band is
        # centered on. tev_weights (stage A's witness) only satisfies the TEV
        # bound in isolation, so it is not a safe seed for the joint set.
        best = _solve_objective(
            stage_constraints, extra_starts=[dev_weights] if dev_weights is not None else None
        )
        if best is not None:
            stage_results.append({"stage": "objective", "status": "optimized"})
            return best

        # The TEV and volatility bands were resolved independently (stage A,
        # then stage B inside stage A's own bound) -- each is feasible in
        # isolation, but pinning BOTH simultaneously for the objective solve
        # (stage C) can still be jointly infeasible. Only ever relax here
        # when the caller already opted into "nearest feasible" for BOTH --
        # a "within_limit"/"matched" bound was exactly achievable on its own
        # merit and must not be silently loosened. TEV keeps priority over
        # volatility, per the configured stage order (A before B).
        if (
            tev_constraint_obj is not None
            and vol_constraint_obj is not None
            and tracking_error_policy == "nearest_feasible"
            and volatility_target_policy == "nearest_feasible"
        ):
            tev_only = [*hard_constraints, tev_constraint_obj]
            best = _solve_objective(
                tev_only, extra_starts=[tev_weights] if tev_weights is not None else None
            )
            if best is not None:
                stage_results.append(
                    {
                        "stage": "objective",
                        "status": "optimized_tev_priority",
                        "note": (
                            "joint TEV+volatility bound was infeasible for the "
                            "objective; volatility band relaxed, TEV band kept"
                        ),
                    }
                )
                return best

            vol_only = [*hard_constraints, vol_constraint_obj]
            best = _solve_objective(
                vol_only, extra_starts=[dev_weights] if dev_weights is not None else None
            )
            if best is not None:
                stage_results.append(
                    {
                        "stage": "objective",
                        "status": "optimized_volatility_priority",
                        "note": (
                            "joint TEV+volatility bound was infeasible for the "
                            "objective; TEV band relaxed, volatility band kept"
                        ),
                    }
                )
                return best

        # A nearest-feasible volatility witness is already a valid portfolio:
        # Stage B found it under every hard constraint (and under the fixed
        # TEV bound, when one exists).  SLSQP can nevertheless fail to make
        # further economic progress inside the intentionally tiny band around
        # that witness.  That numerical failure must not abort a walk-forward
        # fold after the user explicitly chose best-effort target handling.
        # Return the proven feasible witness, preserving the closest-volatility
        # result and making the lack of Stage-C improvement auditable.
        if volatility_target_policy == "nearest_feasible" and dev_weights is not None:
            stage_results.append(
                {
                    "stage": "objective",
                    "status": "nearest_feasible_witness",
                    "note": (
                        "economic refinement failed inside the resolved "
                        "volatility band; returned the Stage-B feasible witness"
                    ),
                }
            )
            return OptimizeResult(
                x=np.asarray(dev_weights, dtype=float),
                success=True,
                message="nearest-feasible volatility witness",
            )

        raise V2OptimizationError(
            "the economic objective could not be optimized subject to the "
            "resolved TEV/volatility stage bounds"
            + (
                ", even after relaxing to either bound alone"
                if tracking_error_policy == "nearest_feasible"
                and volatility_target_policy == "nearest_feasible"
                else ""
            )
        )

    @staticmethod
    def _resolve_volatility(
        reference: str,
        manual_value: float | None,
        reference_series: Any | None,
        periods_per_year: float,
    ) -> float | None:
        if reference in {"benchmark", "father_proxy", "forward_root_reference"}:
            if reference_series is None:
                raise V2OptimizationError(
                    "relative volatility target requires a reference series"
                )
            return float(reference_series.std(ddof=1)) * sqrt(periods_per_year)
        return manual_value

    @staticmethod
    def _initial_weights(
        names: list[str],
        lower: np.ndarray,
        upper: np.ndarray,
        reference_weights: dict[str, float] | None,
    ) -> np.ndarray:
        if reference_weights:
            candidate = np.array(
                [reference_weights.get(name, 0.0) for name in names],
                dtype=float,
            )
            if (
                abs(candidate.sum() - 1.0) <= 1e-8
                and np.all(candidate >= lower)
                and np.all(candidate <= upper)
            ):
                return candidate
        candidate = lower.copy()
        room = upper - candidate
        remaining = 1.0 - candidate.sum()
        while remaining > 1e-12:
            active = room > 1e-12
            if not active.any():
                raise V2OptimizationError(
                    "cannot construct a bounded fully-invested start"
                )
            addition = min(
                remaining / active.sum(),
                float(room[active].min()),
            )
            candidate[active] += addition
            room[active] -= addition
            remaining -= addition * active.sum()
        return candidate

    @staticmethod
    def _bounded_starts(
        names: list[str],
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> list[np.ndarray]:
        starts: list[np.ndarray] = []
        for index in range(len(names)):
            candidate = lower.copy()
            candidate[index] = min(
                upper[index],
                candidate[index] + 1.0 - candidate.sum(),
            )
            remaining = 1.0 - candidate.sum()
            for other in range(len(names)):
                addition = min(remaining, upper[other] - candidate[other])
                candidate[other] += addition
                remaining -= addition
            if remaining <= 1e-10:
                starts.append(candidate)
        return starts

    @classmethod
    def _randomized_starts(
        cls,
        lower: np.ndarray,
        upper: np.ndarray,
        count: int,
        seed: int,
    ) -> list[np.ndarray]:
        """Randomized feasible points of the box-constrained simplex.

        `max_ratio` and an exact volatility target are non-convex problems;
        the structured starts (``_initial_weights``/``_bounded_starts``) are
        deterministic and can repeatedly steer SLSQP into the same local
        optimum. This adds diversity via a random stick-breaking fill: for
        each sample, visit assets in a random order and give each one a
        random fraction of the still-unallocated budget, capped by its own
        room — always feasible by construction (no separate projection step
        needed, unlike a plain Dirichlet draw over an unconstrained simplex).
        Seeded deterministically so a solve stays reproducible for the same
        declared bounds.
        """

        rng = np.random.default_rng(seed)
        n = len(lower)
        starts: list[np.ndarray] = []
        for _ in range(count):
            order = rng.permutation(n)
            candidate = lower.copy()
            remaining = 1.0 - float(candidate.sum())
            for position, index in enumerate(order):
                if remaining <= 1e-12:
                    break
                room = upper[index] - candidate[index]
                if room <= 1e-12:
                    continue
                is_last = position == len(order) - 1
                addition = (
                    min(room, remaining)
                    if is_last
                    else min(room, remaining * rng.uniform(0.0, 1.0))
                )
                candidate[index] += addition
                remaining -= addition
            if remaining <= 1e-9:
                starts.append(candidate)
        return starts

    @classmethod
    def _frontier_starts(
        cls,
        starts: list[np.ndarray],
        means: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        constraints: list[dict[str, Any]],
        volatility: Any,
        target: float,
        tracking_error: Any,
        tev_limit: float | None,
        cap: float | None,
    ) -> list[np.ndarray]:
        from scipy.optimize import brentq, minimize

        endpoints: list[np.ndarray] = []
        objectives = (
            lambda weights: volatility(weights) ** 2,
            lambda weights: -float(means @ weights) * 1_000.0,
        )
        for objective in objectives:
            for start in starts:
                result = minimize(
                    objective,
                    start,
                    method="SLSQP",
                    bounds=list(zip(lower, upper, strict=True)),
                    constraints=constraints,
                    options={"ftol": 1e-12, "maxiter": 2_000, "disp": False},
                )
                if result.success and cls._is_feasible(
                    result.x,
                    lower,
                    upper,
                    volatility,
                    None,
                    cap,
                    tracking_error,
                    tev_limit,
                ):
                    endpoints.append(np.asarray(result.x, dtype=float))
        feasible_starts: list[np.ndarray] = []
        for left_index, left in enumerate(endpoints):
            for right in endpoints[left_index + 1 :]:
                left_gap = volatility(left) - target
                right_gap = volatility(right) - target
                if abs(left_gap) <= 2e-6:
                    feasible_starts.append(left)
                    continue
                if left_gap * right_gap > 0.0:
                    continue
                alpha = brentq(
                    lambda value, left=left, right=right: (
                        volatility(value * left + (1.0 - value) * right)
                        - target
                    ),
                    0.0,
                    1.0,
                )
                candidate = alpha * left + (1.0 - alpha) * right
                if cls._is_feasible(
                    candidate,
                    lower,
                    upper,
                    volatility,
                    target,
                    cap,
                    tracking_error,
                    tev_limit,
                ):
                    feasible_starts.append(candidate)
        return feasible_starts

    @staticmethod
    def _is_feasible(
        weights: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        volatility: Any,
        target: float | None,
        cap: float | None,
        tracking_error: Any,
        tev_limit: float | None,
    ) -> bool:
        if abs(float(weights.sum()) - 1.0) > 2e-6:
            return False
        if np.any(weights < lower - 2e-6) or np.any(weights > upper + 2e-6):
            return False
        if target is not None and abs(volatility(weights) - target) > 2e-6:
            return False
        if cap is not None and volatility(weights) > cap + 2e-6:
            return False
        return tev_limit is None or tracking_error(weights) <= tev_limit + 2e-6

    @staticmethod
    def _audit_hard_constraints(
        weights: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        cap_periodic: float | None,
        volatility: Any,
    ) -> None:
        """Re-check the always-hard constraints on the final, zeroed-and-
        renormalized weight vector.

        The solver's numerical output is validated against these same
        constraints *before* the tiny-weight zeroing and sum-renormalization
        step above, but that canonicalization is itself a (small) further
        perturbation. Hard constraints (bounds, volatility cap) must never be
        silently violated by it; raise loudly instead of returning a result
        that quietly no longer satisfies what was declared hard.
        """

        if np.any(weights < lower - 1e-6) or np.any(weights > upper + 1e-6):
            raise V2OptimizationError(
                "canonicalized weights violate declared bounds after "
                "zeroing/renormalization; this indicates a numerical "
                "inconsistency, not an audited result"
            )
        if cap_periodic is not None and volatility(weights) > cap_periodic + 1e-6:
            raise V2OptimizationError(
                "canonicalized weights violate the hard volatility cap after "
                "zeroing/renormalization; this indicates a numerical "
                "inconsistency, not an audited result"
            )


__all__ = ["CASH_BORROW", "CASH_LEND", "V2LocalOptimizer"]
