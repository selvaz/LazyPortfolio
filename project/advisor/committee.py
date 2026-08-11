"""Fase 6 -- the Investment Committee as a second ``ChangeProposal`` producer.

docs/node-advisor-operational-plan.md §13 Fase 6: "un job schedulato
(LazyPulse) chiama gli stessi NodeContextService/ProposalService di Fase 3
con ModelProvenance.producer_kind='scheduled_batch',
producer_id='investment-committee', popolando batch_id per raggruppare le
proposte multi-nodo di una singola run [...] Nessuna nuova tabella, nessuna
nuova state machine -- è il punto reso possibile dai vincoli
producer-agnostic fissati in Fase 0/§3.4."

**Deliberate scope of this module (structural proof, not real committee
reasoning):** the hard, novel part of an Investment Committee -- deciding
*which* nodes to touch and *what* views to propose from macro/market
analysis -- is out of scope here. That is a substantial standalone feature
(multi-specialist LLM synthesis, real cost, its own design review), not a
plumbing exercise. What this module proves is narrower and matches the
plan's actual exit criterion: that a second, non-interactive producer can
create valid ``pending_approval`` proposals across multiple nodes in one
batch, through the *identical* validation/hash/state-machine pipeline the
interactive Node Advisor uses, with zero schema or state-machine changes.
``node_views`` is caller-supplied (a fixture, a config file, or -- later --
real committee output) rather than computed here; nothing about that
input's origin is this module's concern.

Not wired to an actual scheduler (LazyPulse cron) yet: scheduling a job
that has no real decision logic behind it has no operational value. Wiring
a real trigger belongs with whatever eventually supplies ``node_views``.
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

#: The one producer identity every committee-originated proposal shares --
#: distinct from any interactive_chat producer_id (§3.4 point 3).
PRODUCER_ID = "investment-committee"
_NO_LLM_MODEL = "none (Fase 6 structural proof, no committee reasoning yet)"


@dataclass(frozen=True)
class CommitteeBatchResult:
    """One committee run's outcome. Partial success is normal: a batch
    covering several nodes should not let one node's invalid views block
    proposals for the others -- each node's ``create_proposal`` call is
    independent, matching how a human would treat a multi-node committee
    memo (some recommendations are actionable, some get sent back)."""

    batch_id: UUID
    proposals: list[ChangeProposal] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)


def run_committee_batch(
    tree_id: str,
    node_views: dict[str, list[dict[str, Any]]],
    *,
    rationale: str = "Investment Committee batch proposal.",
    batch_id: UUID | None = None,
    backend: OptimizationDataBackend | None = None,
    db_path: str | os.PathLike[str] | None = None,
) -> CommitteeBatchResult:
    """Create one ``pending_approval`` proposal per ``(node_id, views)``
    pair in ``node_views``, all sharing one ``batch_id`` -- the Fase 6
    entry point. Every proposal goes through
    :func:`advisor.services.create_proposal`, unchanged: same validation,
    same counterfactual, same content hash, same state machine as the
    interactive Node Advisor and Fase 3's fixture path use.

    A node whose views fail validation is recorded in
    :attr:`CommitteeBatchResult.errors` and does not stop the rest of the
    batch. Returns after every node in ``node_views`` has been attempted
    once -- never partially retries.
    """

    resolved_batch_id = batch_id if batch_id is not None else uuid4()
    proposals: list[ChangeProposal] = []
    errors: dict[str, str] = {}

    for node_id, views in node_views.items():
        try:
            proposal = services.create_proposal(
                tree_id,
                node_id,
                views,
                caller_id=PRODUCER_ID,
                rationale=rationale,
                producer_kind="scheduled_batch",
                producer_id=PRODUCER_ID,
                model=_NO_LLM_MODEL,
                batch_id=resolved_batch_id,
                backend=backend,
                db_path=db_path,
            )
        except (ValueError, services.TreeNotFound) as exc:
            errors[node_id] = str(exc)
            continue
        proposals.append(proposal)

    return CommitteeBatchResult(batch_id=resolved_batch_id, proposals=proposals, errors=errors)


__all__ = ["PRODUCER_ID", "CommitteeBatchResult", "run_committee_batch"]
