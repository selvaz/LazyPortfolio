"""Moment estimation and Black-Litterman policy for optimizer V2."""

from __future__ import annotations

from typing import Any

import numpy as np

from lazyportfolio.v2.contracts import V2OptimizationError, V2View

CASH_LEND = "cash:RF"
CASH_BORROW = "cash:BORROW"
CASH_NAMES = frozenset({CASH_LEND, CASH_BORROW})
FINANCING_SEPARATOR = "@"
SHRUNK_MU_METHODS = {"bayes_stein", "james_stein", "bodnar_okhrin"}
MEAN_ESTIMATORS = SHRUNK_MU_METHODS | {"auto", "equilibrium", "empirical"}


def financing_instrument(base_name: str, node_id: str, *, is_root: bool) -> str:
    """Return a collision-free ledger name for one node's financing position."""

    if base_name not in CASH_NAMES:
        raise ValueError(f"unsupported financing instrument {base_name!r}")
    return base_name if is_root else f"{base_name}{FINANCING_SEPARATOR}{node_id}"


def is_financing_instrument(name: str) -> bool:
    return name in CASH_NAMES or any(
        name.startswith(f"{base}{FINANCING_SEPARATOR}") for base in CASH_NAMES
    )


def financing_base(name: str) -> str | None:
    if name in CASH_NAMES:
        return name
    for base in CASH_NAMES:
        if name.startswith(f"{base}{FINANCING_SEPARATOR}"):
            return base
    return None


def covariance_estimator(name: str) -> Any:
    if name == "shrunk_fixed":
        from skfolio.moments import ShrunkCovariance

        return ShrunkCovariance()
    if name == "ledoit_wolf":
        from skfolio.moments import LedoitWolf

        return LedoitWolf()
    raise ValueError(f"unsupported covariance estimator {name!r}")


def estimate_moments(
    clean: Any,
    names: list[str],
    reference_weights: dict[str, float] | None,
    mean_estimator: str,
    risk_aversion: float,
    risk_free_periodic: float,
    covariance_name: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Estimate one covariance and mean vector on one complete matrix."""

    cash_names = [name for name in names if name in CASH_NAMES]
    if cash_names:
        if len(cash_names) != 1:
            raise V2OptimizationError(
                "a financing solve must contain exactly one cash instrument"
            )
        if mean_estimator == "equilibrium":
            raise V2OptimizationError(
                "mean_estimator='equilibrium' is unsupported when cash financing "
                "is active; use an explicit statistical mean estimator"
            )
        cash_name = cash_names[0]
        risky_names = [name for name in names if name != cash_name]
        if not risky_names:
            raise V2OptimizationError("cash financing requires at least one risky asset")
        risky_reference = (
            {
                name: weight
                for name, weight in (reference_weights or {}).items()
                if name in risky_names
            }
            or None
        )
        covariance, means, resolved = estimate_moments(
            clean.loc[:, risky_names],
            risky_names,
            risky_reference,
            mean_estimator,
            risk_aversion,
            risk_free_periodic,
            covariance_name,
        )
        full_covariance = np.zeros((len(names), len(names)), dtype=float)
        full_means = np.zeros(len(names), dtype=float)
        risky_index = {name: index for index, name in enumerate(risky_names)}
        for row, row_name in enumerate(names):
            if row_name == cash_name:
                full_means[row] = float(clean[cash_name].iloc[0])
                continue
            source_row = risky_index[row_name]
            full_means[row] = means[source_row]
            for column, column_name in enumerate(names):
                if column_name == cash_name:
                    continue
                full_covariance[row, column] = covariance[
                    source_row, risky_index[column_name]
                ]
        return full_covariance, full_means, resolved

    if mean_estimator not in MEAN_ESTIMATORS:
        raise V2OptimizationError(f"unsupported mean_estimator {mean_estimator!r}")

    from skfolio.moments import EmpiricalMu, EquilibriumMu, ShrunkMu, ShrunkMuMethods

    covariance = np.atleast_2d(
        covariance_estimator(covariance_name).fit(clean).covariance_
    )
    reference = np.array(
        [(reference_weights or {}).get(name, 0.0) for name in names],
        dtype=float,
    )
    has_full_reference = bool(reference_weights) and abs(float(reference.sum()) - 1.0) <= 1e-6
    resolved = mean_estimator
    if resolved == "auto":
        resolved = "equilibrium" if has_full_reference else "bayes_stein"
    if resolved == "equilibrium":
        if not has_full_reference:
            raise V2OptimizationError(
                "mean_estimator='equilibrium' requires a full, fully-invested "
                "benchmark/father reference for this node"
            )
        estimator = EquilibriumMu(
            risk_aversion=risk_aversion,
            weights=reference,
            covariance_estimator=covariance_estimator(covariance_name),
        )
        means = np.asarray(estimator.fit(clean).mu_, dtype=float) + risk_free_periodic
        return covariance, means, resolved
    if resolved == "empirical":
        estimator = EmpiricalMu()
    else:
        estimator = ShrunkMu(
            covariance_estimator=covariance_estimator(covariance_name),
            method=ShrunkMuMethods(resolved),
        )
    means = np.asarray(estimator.fit(clean).mu_, dtype=float)
    return covariance, means, resolved


def black_litterman_posterior(
    covariance: np.ndarray,
    means: np.ndarray,
    names: list[str],
    views: tuple[V2View, ...],
    tau: float,
    periods_per_year: float,
) -> tuple[np.ndarray, np.ndarray, tuple[dict[str, Any], ...]]:
    """Fuse typed views using Idzorek confidence-scaled uncertainty."""

    if not views:
        return covariance, means, ()
    if tau <= 0.0:
        raise V2OptimizationError("view_tau must be positive")
    index = {name: position for position, name in enumerate(names)}
    pick = np.zeros((len(views), len(names)), dtype=float)
    q = np.zeros(len(views), dtype=float)
    omega_diagonal = np.zeros(len(views), dtype=float)
    for row, view in enumerate(views):
        if not 0.0 < view.confidence <= 1.0:
            raise V2OptimizationError(
                f"view confidence must be in (0, 1], got {view.confidence!r}"
            )
        for instrument, coefficient in view.instruments.items():
            if instrument not in index:
                raise V2OptimizationError(
                    f"view references {instrument!r}, which is not part of "
                    "this node's solved universe"
                )
            pick[row, index[instrument]] = coefficient
        if not np.any(pick[row]):
            raise V2OptimizationError("view declares no instrument coefficients")
        q[row] = view.expected_return / periods_per_year
        alpha = (1.0 - view.confidence) / view.confidence
        prior_view_variance = float(pick[row] @ covariance @ pick[row])
        omega_diagonal[row] = tau * alpha * prior_view_variance

    tau_sigma_pick = tau * covariance @ pick.T
    system = pick @ tau_sigma_pick + np.diag(omega_diagonal)
    try:
        mu_solution = np.linalg.solve(system, q - pick @ means)
        cov_solution = np.linalg.solve(system, tau_sigma_pick.T)
    except np.linalg.LinAlgError as exc:
        raise V2OptimizationError(
            "view system is singular; check for duplicate or contradictory views"
        ) from exc
    posterior_means = means + tau_sigma_pick @ mu_solution
    posterior_covariance = covariance + (
        tau * covariance - tau_sigma_pick @ cov_solution
    )
    details = tuple(
        {
            "instruments": dict(view.instruments),
            "expected_return_annualized": view.expected_return,
            "confidence": view.confidence,
            "source": view.source,
            "prior_view_return_annualized": (
                float(pick[row] @ means) * periods_per_year
            ),
            "posterior_view_return_annualized": (
                float(pick[row] @ posterior_means) * periods_per_year
            ),
        }
        for row, view in enumerate(views)
    )
    return posterior_covariance, posterior_means, details


def apply_views(
    covariance: np.ndarray,
    means: np.ndarray,
    names: list[str],
    views: tuple[V2View, ...],
    tau: float,
    periods_per_year: float,
    policy: str,
) -> tuple[np.ndarray, np.ndarray, tuple[dict[str, Any], ...]]:
    """Apply views while declaring whether posterior covariance controls risk."""

    if not views:
        return covariance, means, ()
    if any(any(name in CASH_NAMES for name in view.instruments) for view in views):
        raise V2OptimizationError(
            "Black-Litterman views cannot target financing instruments"
        )
    risky_indexes = [index for index, name in enumerate(names) if name not in CASH_NAMES]
    risky_names = [names[index] for index in risky_indexes]
    posterior_covariance, posterior_means, details = black_litterman_posterior(
        covariance[np.ix_(risky_indexes, risky_indexes)],
        means[risky_indexes],
        risky_names,
        views,
        tau,
        periods_per_year,
    )
    full_means = means.copy()
    full_means[risky_indexes] = posterior_means
    if policy == "prior_risk":
        return covariance, full_means, details
    if policy != "posterior_all":
        raise V2OptimizationError(f"unsupported view covariance policy {policy!r}")
    full_covariance = covariance.copy()
    full_covariance[np.ix_(risky_indexes, risky_indexes)] = posterior_covariance
    return full_covariance, full_means, details
