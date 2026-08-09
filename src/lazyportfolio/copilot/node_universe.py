"""Node-scoped universe resolution and view validation (docs/node-copilot-operational-plan.md §6.1).

Reuses ``V2Model``/``V2Node`` identity as already resolved by
``V2Model.from_config`` -- ``allowed_view_instruments`` is built from a
node's own direct instruments plus each child's ``proxy`` (never a child's
internal terminal tickers), the same candidate-column semantics the solver
itself uses (``V2Component``/``V2SolveContext`` in ``lazyportfolio.v2.hierarchy``),
not a second, independently-invented notion of "what this node can see".
"""

from __future__ import annotations

import copy
import math
from typing import Any
from uuid import UUID

from lazyportfolio.copilot.contracts import (
    NodeComponent,
    NodeContext,
    ProposedView,
    SnapshotDescriptor,
    ValidationIssue,
    ValidationResult,
)
from lazyportfolio.v2.contracts import Mode, V2Node, ticker
from lazyportfolio.v2.model import V2Model
from lazyportfolio.v2.validation import _is_financing_view_label


class NodeNotFoundError(ValueError):
    """``node_id`` does not exist in the resolved ``V2Model``."""


def find_node(model: V2Model, node_id: str) -> V2Node:
    """The :class:`V2Node` with this ``id``, or raise :class:`NodeNotFoundError`.

    Public: shared by :mod:`lazyportfolio.copilot.approval_service` and
    :mod:`lazyportfolio.copilot.counterfactual` too, both of which need the
    node's own ``.name`` -- ``V2Estimate.node_results`` is keyed by node
    *name*, not ``id`` (see ``lazyportfolio.v2.validation._validate_tree_structure``'s
    docstring) -- so "look up a node by id" is not this module's private
    concern alone.
    """

    for node in model.root.walk():
        if node.id == node_id:
            return node
    raise NodeNotFoundError(node_id)


def _find_parent(model: V2Model, node_id: str) -> V2Node | None:
    for node in model.root.walk():
        if any(child.id == node_id for child in node.children):
            return node
    return None


def resolve_node_context(
    config: dict[str, Any],
    node_id: str,
    *,
    mode: Mode,
    tree_id: UUID,
    revision_id: UUID,
    snapshot: SnapshotDescriptor | None = None,
) -> NodeContext:
    """Build the canonical :class:`NodeContext` for one node of ``config``.

    ``tree_id``/``revision_id`` identify the caller's already-known
    revision (from :mod:`lazyportfolio.copilot.repository`) -- a raw V2
    config dict carries no identity of its own, so this function cannot
    derive them; it only resolves what depends on the config's *content*.
    """

    model = V2Model.from_config(config)
    node = find_node(model, node_id)
    parent = _find_parent(model, node_id)

    solved_components = [
        NodeComponent(
            component_id=instrument,
            kind="direct",
            label=instrument,
            candidate_instrument=instrument,
            child_node_id=None,
        )
        for instrument in node.instruments
    ] + [
        NodeComponent(
            component_id=child.id,
            kind="child",
            label=child.name,
            candidate_instrument=child.proxy or "",
            child_node_id=child.id,
        )
        for child in node.children
    ]
    allowed_view_instruments = [
        *node.instruments,
        *(child.proxy for child in node.children if child.proxy),
    ]

    return NodeContext(
        schema_version="1.0",
        tree_id=tree_id,
        revision_id=revision_id,
        node_id=node.id,
        node_name=node.name,
        objective=node.objective,
        mode=mode,
        solved_components=solved_components,
        allowed_view_instruments=allowed_view_instruments,
        direct_instruments=list(node.instruments),
        child_node_ids=[child.id for child in node.children],
        parent_node_id=parent.id if parent is not None else None,
        parent_candidate_instrument=node.proxy,
        constraints={
            "view_tau": node.constraints.view_tau,
            "view_covariance_policy": node.constraints.view_covariance_policy,
            "mean_estimator": node.constraints.mean_estimator,
        },
        current_views=[
            {
                "instruments": dict(view.instruments),
                "expected_return": view.expected_return,
                "confidence": view.confidence,
                "source": view.source,
            }
            for view in node.constraints.views
        ],
        snapshot=snapshot,
        recent_run=None,
    )


def validate_view_set(
    config: dict[str, Any],
    node_id: str,
    views: list[ProposedView],
    *,
    mode: Mode,
) -> ValidationResult:
    """Validate a candidate view set against ``node_id``'s resolved universe (§6.1).

    ``mode`` is accepted for signature symmetry with :func:`resolve_node_context`
    and forward-compatibility (a future mode-dependent rule); it is unused
    today because the universe a node's views may target does not depend on
    which pass (flat/forward/forward_backward) produced the current
    estimate, only on the node's own children/instruments.
    """

    del mode  # see docstring
    model = V2Model.from_config(config)
    node = find_node(model, node_id)
    allowed = {*node.instruments, *(child.proxy for child in node.children if child.proxy)}

    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    if not views:
        return ValidationResult(valid=True, errors=[], warnings=[])

    signatures: list[tuple[tuple[str, float], ...]] = []
    for index, view in enumerate(views):
        path = f"views[{index}]"
        finite_nonzero = False
        for raw_instrument, coefficient in view.instruments.items():
            # Financing labels ("cash:rf...", "cash:borrow...") must be checked
            # on the RAW key: ticker() collapses everything after the first
            # ":" into "ticker:<REST>", which would turn "cash:rf" into
            # "ticker:RF" and silently defeat this check -- the same reason
            # lazyportfolio.v2.validation.normalize_config checks the raw
            # view instrument key before any node in the tree ever sees a
            # ticker()-normalized one.
            if _is_financing_view_label(raw_instrument):
                errors.append(
                    ValidationIssue(
                        code="financing_instrument_forbidden",
                        message=f"{raw_instrument!r} is a financing instrument; "
                        "Black-Litterman views cannot target it",
                        path=path,
                    )
                )
                continue
            instrument = ticker(raw_instrument)
            if instrument not in allowed:
                errors.append(
                    ValidationIssue(
                        code="instrument_outside_universe",
                        message=f"{instrument!r} is not in node {node_id!r}'s "
                        "allowed_view_instruments",
                        path=path,
                    )
                )
                continue
            if not math.isfinite(coefficient):
                errors.append(
                    ValidationIssue(
                        code="non_finite_coefficient",
                        message=f"coefficient for {instrument!r} is not finite",
                        path=path,
                    )
                )
                continue
            if coefficient != 0.0:
                finite_nonzero = True
        if not finite_nonzero:
            errors.append(
                ValidationIssue(
                    code="all_coefficients_zero",
                    message="every coefficient in this view is zero or invalid",
                    path=path,
                )
            )
        if abs(view.expected_return) > 1.0:
            warnings.append(
                ValidationIssue(
                    code="extreme_expected_return",
                    message=f"expected_return={view.expected_return!r} is an extreme "
                    "(>100%) annualized value -- verify scale before approving",
                    path=path,
                )
            )
        signature = tuple(
            sorted((ticker(k), float(v)) for k, v in view.instruments.items())
        )
        signatures.append(signature)

    for i, sig_a in enumerate(signatures):
        for j in range(i + 1, len(signatures)):
            sig_b = signatures[j]
            if sig_a == sig_b:
                errors.append(
                    ValidationIssue(
                        code="duplicate_view",
                        message=f"views[{i}] and views[{j}] are identical picks",
                        path=None,
                    )
                )
            elif {k for k, _ in sig_a} == {k for k, _ in sig_b} and all(
                a_v == -b_v for (a_k, a_v), (_, b_v) in zip(sig_a, sig_b, strict=True)
            ):
                errors.append(
                    ValidationIssue(
                        code="opposite_pick_same_horizon",
                        message=f"views[{i}] and views[{j}] are exact opposite picks "
                        "on the same instruments",
                        path=None,
                    )
                )

    no_effect_on_weights = (
        node.objective in ("min_risk", "hrp")
        and node.constraints.view_covariance_policy == "prior_risk"
    )
    if no_effect_on_weights:
        warnings.append(
            ValidationIssue(
                code="no_effect_on_weights",
                message=f"node {node_id!r} objective={node.objective!r} with "
                "view_covariance_policy='prior_risk' means views do not change the "
                "solved weights -- do not move this proposal to pending_approval",
                path=None,
            )
        )

    return ValidationResult(valid=not errors, errors=errors, warnings=warnings)


def apply_views_to_config(
    config: dict[str, Any], node_id: str, views: list[ProposedView]
) -> dict[str, Any]:
    """Return a deep copy of ``config`` with ``node_id``'s ``constraints.views``
    replaced by ``views`` -- nothing else in the tree changes.

    Shared by :mod:`lazyportfolio.copilot.approval_service` (applying an
    approved proposal) and :mod:`lazyportfolio.copilot.counterfactual`
    (building the variant to solve) so both stay byte-for-byte in agreement
    on what "apply this node's views" means, instead of each carrying its
    own copy of the same patch logic.
    """

    new_config = copy.deepcopy(config)
    for node in new_config.get("nodes", []):
        if str(node.get("id")) == node_id:
            constraints = node.setdefault("constraints", {})
            constraints["views"] = [
                {
                    "instruments": {ticker(k): float(v) for k, v in view.instruments.items()},
                    "expected_return": view.expected_return,
                    "confidence": view.confidence,
                    "source": view.source,
                }
                for view in views
            ]
            return new_config
    raise NodeNotFoundError(node_id)


__all__ = [
    "NodeNotFoundError",
    "apply_views_to_config",
    "find_node",
    "resolve_node_context",
    "validate_view_set",
]
