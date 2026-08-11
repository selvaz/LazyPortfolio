"""Create a batch of proposals through the public advisor contract.

This module deliberately knows nothing about the process that selected the
nodes or produced their views.  A scheduler, research workflow, or agent can
therefore use the same validation, persistence, and approval state machine as
the interactive advisor without introducing a second proposal system.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from project.advisor import services

if TYPE_CHECKING:
    from lazyportfolio import OptimizationDataBackend
    from lazyportfolio.advisor.contracts import ChangeProposal


@dataclass(frozen=True)
class BatchProposalResult:
    """Outcome of one batch; invalid nodes do not block valid siblings."""

    batch_id: UUID
    proposals: list[ChangeProposal] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)


def _required_text(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


def run_proposal_batch(
    tree_id: str,
    node_views: dict[str, list[dict[str, Any]]],
    *,
    producer_id: str,
    rationale: str,
    model: str,
    batch_id: UUID | None = None,
    backend: OptimizationDataBackend | None = None,
    db_path: str | os.PathLike[str] | None = None,
) -> BatchProposalResult:
    """Create one pending proposal per node, sharing a single batch id.

    Producer identity and provenance are mandatory caller inputs.  This keeps
    the reusable public function independent from every private workflow.
    Expected per-node validation errors are collected so one bad recommendation
    cannot discard the valid proposals in the same batch.
    """

    resolved_producer_id = _required_text(producer_id, "producer_id")
    resolved_rationale = _required_text(rationale, "rationale")
    resolved_model = _required_text(model, "model")
    resolved_batch_id = batch_id if batch_id is not None else uuid4()
    proposals: list[ChangeProposal] = []
    errors: dict[str, str] = {}

    for node_id, views in node_views.items():
        try:
            proposal = services.create_proposal(
                tree_id,
                node_id,
                views,
                caller_id=resolved_producer_id,
                rationale=resolved_rationale,
                producer_kind="scheduled_batch",
                producer_id=resolved_producer_id,
                model=resolved_model,
                batch_id=resolved_batch_id,
                backend=backend,
                db_path=db_path,
            )
        except (ValueError, services.TreeNotFound) as exc:
            errors[node_id] = str(exc)
            continue
        proposals.append(proposal)

    return BatchProposalResult(
        batch_id=resolved_batch_id,
        proposals=proposals,
        errors=errors,
    )


__all__ = ["BatchProposalResult", "run_proposal_batch"]
