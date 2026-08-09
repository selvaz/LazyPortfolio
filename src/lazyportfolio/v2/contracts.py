"""Canonical contracts for the hierarchical optimizer V2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Mode = Literal["flat", "forward", "forward_backward"]
RECOGNIZED_OBJECTIVES = {
    "min_risk",
    "max_return",
    "max_ratio",
    "max_utility",
    "hrp",
}

#: Canonical vocabulary for a reference-kind field (``volatility_reference``,
#: ``max_volatility_reference``, ``tracking_error_reference``,
#: ``mean_reference_kind``). ``"forward_root_reference"`` names the frozen root
#: synthetic series computed once during the Forward diagnostic pass — it is
#: never the current (possibly still-being-solved) parent or root. It replaces
#: the older, ambiguous ``"root"`` label.
ReferencePolicy = Literal[
    "none",
    "manual",
    "declared",
    "forward_root_reference",
    "benchmark",
    "father_proxy",
    "local_weights",
]

#: Reserved for a future iterative-hierarchy mode (parent references its own
#: currently-being-solved synthetic series). Not implemented; any reference
#: field carrying one of these values must be rejected in ``flat``,
#: ``forward`` and ``forward_backward`` — never silently accepted or
#: coerced to a supported policy.
RESERVED_ITERATIVE_REFERENCES = frozenset(
    {"current_parent_synthetic", "current_root_synthetic"}
)

#: Fallback policy for a relaxable constraint: fail the solve outright, or
#: report the nearest feasible point under the lexicographic TEV-then-
#: volatility ordering (see ``V2LocalOptimizer``). ``hard_fail`` is the
#: default for every constraint; only TEV and the exact volatility target may
#: be configured as ``nearest_feasible``. The volatility *cap* must never be
#: configured to anything other than ``hard_fail`` — it is never relaxed by
#: TEV/target infeasibility.
ConstraintPolicy = Literal["hard_fail", "nearest_feasible"]


def ticker(value: Any) -> str:
    label = str(value).strip()
    key = label.split(":", 1)[1] if ":" in label else label
    return f"ticker:{key.upper()}"


def effective_setting(
    node_value: float | None,
    root_value: float | None,
    default: float,
) -> float:
    if node_value is not None:
        return node_value
    if root_value is not None:
        return root_value
    return default


@dataclass(frozen=True)
class V2View:
    instruments: dict[str, float]
    expected_return: float
    confidence: float
    source: str = "manual"


@dataclass(frozen=True)
class V2Constraints:
    """Complete local optimization contract.

    Every node may optionally lend through positive local cash or borrow through
    negative local cash. Risky assets remain long-only and the local net budget is
    always ``sum(risky weights) + cash = 1``.
    """

    min_weights: dict[str, float] = field(default_factory=dict)
    max_weights: dict[str, float] = field(default_factory=dict)
    per_asset_cap: float | None = None
    volatility_reference: str = "none"
    volatility_target: float | None = None
    max_volatility_reference: str = "none"
    max_volatility: float | None = None
    max_tracking_error: float | None = None
    tracking_error_reference: str = "declared"
    mean_estimator: str = "auto"
    views: tuple[V2View, ...] = ()
    view_tau: float = 0.05
    risk_aversion: float | None = None
    risk_free_rate: float | None = None
    covariance_estimator: str = "shrunk_fixed"
    view_covariance_policy: str = "prior_risk"
    volatility_target_mode: str = "exact"
    cash_enabled: bool = False
    max_leverage: float = 1.0
    borrow_spread_bps: float = 0.0
    cash_enabled_source: str = "default"
    max_leverage_source: str = "default"
    borrow_spread_bps_source: str = "default"
    # Independent mean-reference axis (Bayes/equilibrium prior weights), kept
    # deliberately separate from volatility_reference/max_volatility_reference/
    # tracking_error_reference: those are risk references (TEV, vol target,
    # vol cap); this is the strategic-weight reference for equilibrium
    # expected-return construction. Configuring a risk reference must never
    # implicitly select a mean reference, and vice versa.
    mean_reference_kind: str = "none"
    mean_reference_weights: dict[str, float] | None = None
    # Constraint fallback policy. Hard by default everywhere; only TEV and an
    # exact volatility target may be relaxed to "nearest_feasible" (staged
    # lexicographically: TEV excess first, then volatility deviation, then the
    # economic objective — see V2LocalOptimizer). volatility_cap_policy must
    # stay "hard_fail": a volatility cap is never relaxed for TEV/target
    # infeasibility.
    tracking_error_policy: str = "hard_fail"
    volatility_target_policy: str = "hard_fail"
    volatility_cap_policy: str = "hard_fail"


@dataclass
class V2Node:
    id: str
    name: str
    instruments: list[str]
    children: list[V2Node]
    proxy: str | None
    objective: str
    constraints: V2Constraints

    def walk(self) -> list[V2Node]:
        nodes = [self]
        for child in self.children:
            nodes.extend(child.walk())
        return nodes

    def terminal_instruments(self) -> list[str]:
        result = list(self.instruments)
        for child in self.children:
            result.extend(child.terminal_instruments())
        return list(dict.fromkeys(result))


@dataclass(frozen=True)
class V2Benchmark:
    name: str
    weights: dict[str, float]


@dataclass(frozen=True)
class V2Component:
    """A stable economic-component identity, distinct from any series.

    ``id`` never changes across Forward/Backward passes and never implies
    which series (raw proxy vs synthetic) is currently in use for a given
    role — that is decided per-role by the resolvers that build a
    ``V2SolveContext``, not by inspecting the id or a naming convention like
    a ``_SYNTH`` suffix.
    """

    id: str
    kind: Literal["direct", "child"]
    raw_series_key: str
    child_id: str | None = None


@dataclass
class V2SolveContext:
    """Per-node-solve identity/series bookkeeping for one local frame.

    Every solver-facing column name is mapped back to exactly one
    ``V2Component``, and every component exposes its raw proxy series, its
    synthetic series (when one exists), and whichever series is currently
    the *candidate* for this pass (raw in Forward, synthetic in Backward for
    a child component; always raw for a direct instrument or for any risk/
    benchmark reference). Risk references and mean references are resolved
    independently against this context — never by sharing one fallback
    value between them.
    """

    pass_kind: Literal["forward", "backward", "flat"]
    components: dict[str, V2Component] = field(default_factory=dict)
    candidate_series_by_component: dict[str, Any] = field(default_factory=dict)
    raw_proxy_series_by_component: dict[str, Any] = field(default_factory=dict)
    synthetic_series_by_component: dict[str, Any] = field(default_factory=dict)
    component_to_solver_column: dict[str, str] = field(default_factory=dict)
    solver_column_to_component: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class V2Audit:
    target_reference: str
    target_volatility: float | None
    actual_volatility: float
    cap_reference: str
    volatility_cap: float | None
    tracking_error_limit: float | None
    actual_tracking_error: float | None
    minimum_slack: dict[str, float]
    maximum_slack: dict[str, float]
    sum_weights: float
    solver_message: str
    target_status: str
    tracking_error_status: str
    configured_objective: str
    effective_objective: str
    expected_return_annualized: float
    objective_value: float
    soft_constraint_violation: float
    configured_mean_estimator: str
    resolved_mean_estimator: str
    views_applied: int
    view_details: tuple[dict[str, Any], ...]
    risk_aversion: float
    risk_free_rate: float
    covariance_estimator: str = "shrunk_fixed"
    covariance_estimator_class: str = "ShrunkCovariance"
    risk_covariance_role: str = "prior"
    objective_covariance_role: str = "prior"
    view_covariance_policy: str = "prior_risk"
    mean_resolution_reason: str = ""
    risk_free_rate_source: str = "hard_default"
    risk_aversion_source: str = "hard_default"
    volatility_target_mode: str = "exact"
    global_optimality_claim: bool = False
    solver_strategy: str = "slsqp_multistart_audited"
    #: Number of starting points (structured + randomized) fed into the
    #: multi-start SLSQP search for this solve. Diagnostic only — SLSQP has
    #: no global-optimality guarantee on the non-convex `max_ratio` and
    #: exact-volatility-target problems, so this records how much of a
    #: multi-start search actually happened.
    restart_candidate_count: int = 0
    #: Gap between the best and second-best accepted restart's loss value.
    #: Only meaningful when `solver_message` is not a "nearest feasible
    #: projection" (the lexicographic fallback path) and the objective is
    #: not `hrp` (no restart search there) — left at 0.0 otherwise, not
    #: fabricated. A large spread is a signal the accepted restarts disagree
    #: on where the optimum is, i.e. the search is likely multimodal and the
    #: reported result should not be read as a global optimum.
    restart_objective_spread: float = 0.0
    hrp_distance_metric: str = ""
    hrp_linkage_method: str = ""
    hrp_risk_measure: str = ""
    cash_enabled: bool = False
    cash_instrument: str = ""
    cash_weight: float = 0.0
    risky_gross_exposure: float = 1.0
    max_leverage: float = 1.0
    cash_lending_rate: float = 0.0
    cash_borrowing_rate: float = 0.0
    borrow_spread_bps: float = 0.0
    financing_regime: str = "fully_invested"
    cash_enabled_source: str = "default"
    max_leverage_source: str = "default"
    borrow_spread_bps_source: str = "default"
    parent_weight: float = 1.0
    global_node_weight: float = 1.0
    global_risky_gross_exposure: float = 1.0
    global_cash_weight: float = 0.0
    portfolio_risky_gross_exposure: float = 1.0
    portfolio_cash_weight: float = 0.0
    portfolio_net_exposure: float = 1.0
    # Component identity / pass / reference-source provenance (clean-engine
    # follow-up). Defaulted so every existing call site keeps working;
    # populated once the hierarchy resolver (Phase 3) and solver (Phase 4)
    # are wired to a V2SolveContext.
    component_id: str = ""
    pass_kind: str = ""
    candidate_frame_composition: dict[str, str] = field(default_factory=dict)
    mean_reference_source: str = "none"
    risk_reference_source: str = "none"
    constraint_stage_results: tuple[dict[str, Any], ...] = ()
    #: v3 performance-roadmap fields (see lazyportfolio.v2.problem_class).
    #: Diagnostic today (every route still resolves to solver_strategy's two
    #: existing values); populated meaningfully once Phase B+ routes exist.
    problem_class: str = ""
    solver_status: str = "ok"
    solve_seconds: float = 0.0
    warm_started: bool = False
    fallback_reason: str = ""


@dataclass
class V2NodeResult:
    local_weights: dict[str, float]
    terminal_weights: dict[str, float]
    synthetic_returns: Any
    audit: V2Audit


@dataclass
class V2Estimate:
    mode: Mode
    terminal_weights: dict[str, float]
    node_results: dict[str, V2NodeResult]
    forward_node_results: dict[str, V2NodeResult] = field(default_factory=dict)
    synthetic_benchmark_weights: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class V2Fold:
    signal: Any
    training_start: Any
    training_end: Any
    holding_start: Any
    holding_end: Any
    targets: dict[str, dict[str, float]]
    audits: dict[str, V2Audit]
    forward_audits: dict[str, V2Audit] = field(default_factory=dict)
    candidate_series: dict[str, list[str]] = field(default_factory=dict)
    estimation_series: dict[str, Any] = field(default_factory=dict)


@dataclass
class V2BacktestReport:
    mode: Mode
    folds: list[V2Fold]
    curves: dict[str, Any]
    metrics: dict[str, dict[str, float | int]]
    transaction_cost_paid: dict[str, float]


class V2OptimizationError(RuntimeError):
    """The declared local problem is infeasible or failed its independent audit."""


def constraints_from_base(
    constraints: V2Constraints,
    **updates: Any,
) -> V2Constraints:
    """Compatibility helper retained for callers migrating from the patch layer."""

    from dataclasses import replace

    return replace(constraints, **updates)


def audit_from_base(audit: V2Audit, **updates: Any) -> V2Audit:
    """Compatibility helper retained for callers migrating from the patch layer."""

    from dataclasses import replace

    return replace(audit, **updates)
