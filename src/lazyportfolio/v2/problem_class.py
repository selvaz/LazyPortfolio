"""Pure classification of a local-solve problem: (objective, constraints) -> shape.

Part of the v3 performance roadmap's solver router
(docs/hierarchical-optimizer-performance-plan.md): later phases dispatch a
classified problem to a fast exact route (LP/QP/SOCP)
when it qualifies, falling back to today's audited multi-start SLSQP
otherwise. This module only classifies -- it never solves, never touches
`means`/`covariance`, and has no side effects, so introducing it changes
nothing about how any node solves today.

Black-Litterman views are resolved (`apply_views`, adjusting `means` and,
under `view_covariance_policy="posterior_all"`, `covariance` too) *before*
any route -- including a future fast-path route -- ever sees the problem.
`has_views`/`view_covariance_policy` are carried here purely for audit
visibility and for routing decisions that must stay conservative around a
`posterior_all` node (e.g. a `min_risk` fast path would need the
view-adjusted covariance, not raw); classification itself never applies or
re-derives a view.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from lazyportfolio.v2.contracts import V2Constraints

VolatilityMode = Literal["none", "capped", "exact"]


@dataclass(frozen=True)
class V2ProblemClass:
    """A local-solve problem's shape, independent of the data it will run on."""

    objective: str
    volatility_mode: VolatilityMode
    has_tev: bool
    has_financing: bool
    has_views: bool
    view_covariance_policy: str

    @property
    def label(self) -> str:
        """Compact, human-readable summary for audit/diagnostic display."""
        parts = [self.objective, f"vol={self.volatility_mode}"]
        if self.has_tev:
            parts.append("tev")
        if self.has_financing:
            parts.append("financing")
        if self.has_views:
            parts.append(f"views={self.view_covariance_policy}")
        return "|".join(parts)


def classify(objective: str, constraints: V2Constraints) -> V2ProblemClass:
    """Classify one node's local-solve problem.

    Assumes ``objective`` has already been validated against
    ``RECOGNIZED_OBJECTIVES`` and ``constraints`` has already gone through
    ``V2LocalOptimizer._normalize_direct_constraints`` -- by the time either
    solver path calls this, a ``volatility_target_mode`` of ``"cap"``/
    ``"at_most"`` has already been folded into ``max_volatility``, so
    ``volatility_target`` being set here always means an *exact* target.
    """
    if constraints.volatility_target is not None and constraints.max_volatility is not None:
        raise ValueError(
            "invalid constraints: volatility_target and max_volatility cannot "
            "both be set (should already have been normalized upstream)"
        )
    volatility_mode: VolatilityMode
    if objective == "hrp":
        volatility_mode = "none"
    elif constraints.volatility_target is not None:
        volatility_mode = "exact"
    elif constraints.max_volatility is not None:
        volatility_mode = "capped"
    else:
        volatility_mode = "none"
    return V2ProblemClass(
        objective=objective,
        volatility_mode=volatility_mode,
        has_tev=constraints.max_tracking_error is not None,
        has_financing=constraints.cash_enabled,
        has_views=bool(constraints.views),
        view_covariance_policy=constraints.view_covariance_policy,
    )


__all__ = ["V2ProblemClass", "VolatilityMode", "classify"]
