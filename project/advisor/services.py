"""Application services wiring ``lazyportfolio.advisor`` into Tree Studio.

docs/node-advisor-operational-plan.md §13 Fase 3. Every function takes the
caller's identity as an explicit parameter (``caller_id``/``approved_by``/
``user_id``), never read from an implicit request-scoped global --
docs/adr/0001-node-advisor-architecture.md Decision 3 point 4, so this same
layer stays callable from a future scheduled job (Fase 6's Investment
Committee) as well as an HTTP request.

MVP scope: ``create_fixture_proposal`` is the whole "proposal preparation"
pipeline with no LLM anywhere -- ``views`` are supplied directly by the
caller (a fixture, or eventually a human), validated and counterfactually
evaluated exactly as an LLM-produced candidate would be in Fase 4.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from advisor import jobs
from lazyportfolio.advisor import approval_service, node_universe
from lazyportfolio.advisor import conversation_repository as conversations
from lazyportfolio.advisor import counterfactual as counterfactual_service
from lazyportfolio.advisor import proposal_repository as proposals
from lazyportfolio.advisor import repository as tree_repository
from lazyportfolio.advisor import snapshot as snapshot_service
from lazyportfolio.advisor.canonical import content_hash
from lazyportfolio.advisor.contracts import (
    ChangeProposal,
    JsonPatchOperation,
    ModelProvenance,
    ProposedView,
)
from lazyportfolio.advisor.patch import views_patch_path
from lazyportfolio.v2.mode import mode_from_config

if TYPE_CHECKING:
    from advisor.jobs import JobRecord

    from lazyportfolio import OptimizationDataBackend

#: How long a fixture-created proposal stays approvable before it must be
#: re-derived. Arbitrary but generous for a local single-user tool -- not a
#: production SLA.
_DEFAULT_PROPOSAL_TTL = timedelta(hours=24)


class TreeNotFound(ValueError):
    """``tree_id`` has no revisions (see ``lazyportfolio.advisor.repository``)."""


class ProposalNotFound(ValueError):
    pass


# --------------------------------------------------------------------- #
# Node context
# --------------------------------------------------------------------- #
def get_node_context(
    tree_id: str, node_id: str, *, db_path: str | os.PathLike[str] | None = None
) -> Any:
    head = tree_repository.get_head(tree_id, db_path=db_path)
    if head is None:
        raise TreeNotFound(tree_id)
    mode = mode_from_config(head.config)
    return node_universe.resolve_node_context(
        head.config,
        node_id,
        mode=mode,
        tree_id=UUID(head.tree_id),
        revision_id=UUID(head.revision_id),
    )


# --------------------------------------------------------------------- #
# Conversations / messages
# --------------------------------------------------------------------- #
def create_conversation(
    tree_id: str, node_id: str, *, caller_id: str, db_path: str | os.PathLike[str] | None = None
) -> conversations.Conversation:
    return conversations.create_conversation(
        tree_id, node_id, user_id=caller_id, db_path=db_path
    )


def list_messages(
    conversation_id: str, *, db_path: str | os.PathLike[str] | None = None
) -> list[conversations.Message]:
    return conversations.list_messages(conversation_id, db_path=db_path)


def post_message_and_enqueue(
    conversation_id: str,
    content: dict[str, Any],
    *,
    caller_id: str,
    db_path: str | os.PathLike[str] | None = None,
) -> tuple[conversations.Message, str]:
    """Record a user message and enqueue the (MVP: fixture) job it triggers.

    Returns ``(message, job_id)``. The HTTP layer responds with the job id
    immediately -- the actual proposal preparation runs on the worker
    thread, never in the request thread (§11).
    """

    del caller_id  # recorded on the conversation itself, not per-message in the MVP
    message = conversations.add_message(conversation_id, "user", content, db_path=db_path)
    job_id = jobs.enqueue_job(
        conversation_id, message.message_id, jobs.FIXTURE_PROPOSAL, db_path=db_path
    )
    return message, job_id


# --------------------------------------------------------------------- #
# Proposal preparation (the fixture job handler)
# --------------------------------------------------------------------- #
def create_fixture_proposal(
    tree_id: str,
    node_id: str,
    views: list[dict[str, Any]],
    *,
    caller_id: str,
    rationale: str = "Fixture proposal (Fase 3: no LLM in this phase).",
    backend: OptimizationDataBackend | None = None,
    db_path: str | os.PathLike[str] | None = None,
) -> ChangeProposal:
    """Validate, counterfactually evaluate, and persist a
    ``pending_approval`` proposal for ``views`` on ``node_id`` -- the exact
    pipeline steps 5-9 of §8.2's Plan, minus the LLM-only steps 2-4
    (``retrieve_evidence``/``synthesize_candidate_views``) that Fase 4 adds.
    """

    head = tree_repository.get_head(tree_id, db_path=db_path)
    if head is None:
        raise TreeNotFound(tree_id)
    mode = mode_from_config(head.config)
    proposed_views = [ProposedView(**view) for view in views]

    validation = node_universe.validate_view_set(head.config, node_id, proposed_views, mode=mode)
    if not validation.valid:
        messages = "; ".join(f"{e.code}: {e.message}" for e in validation.errors)
        raise ValueError(f"proposed views failed validation: {messages}")

    _, dataset, snapshot = snapshot_service.load_snapshot(head.config, backend=backend)
    counterfactual = counterfactual_service.evaluate_view_counterfactual(
        head.config, node_id, proposed_views, dataset, mode=mode, periods_per_year=252.0
    )

    now = datetime.now(UTC)
    patch = [
        JsonPatchOperation(op="replace", path=views_patch_path(node_id), value=None),
    ]
    provenance = ModelProvenance(
        producer_kind="interactive_chat", producer_id="fixture", model="none (Fase 3, no LLM)"
    )
    draft = ChangeProposal(
        id=uuid4(),
        schema_version="1.0",
        kind="replace_node_views",
        tree_id=UUID(head.tree_id),
        base_revision_id=UUID(head.revision_id),
        node_id=node_id,
        snapshot=snapshot,
        information_cutoff=now,
        patch=patch,
        proposed_views=proposed_views,
        rationale=rationale,
        caveats=[],
        evidence=[],
        model_provenance=provenance,
        validation=validation,
        counterfactual=counterfactual,
        expires_at=now + _DEFAULT_PROPOSAL_TTL,
        content_hash="",
    )
    payload = draft.model_dump(mode="json", exclude={"content_hash"})
    proposal = draft.model_copy(update={"content_hash": content_hash(payload)})

    proposals.create(proposal, status="drafting", db_path=db_path)
    proposals.transition(proposal.id, "drafting", "pending_approval", db_path=db_path)
    return proposal


def handle_fixture_proposal_job(
    job: JobRecord,
    *,
    backend: OptimizationDataBackend | None = None,
    db_path: str | os.PathLike[str] | None = None,
) -> None:
    """The one MVP job handler: reads the triggering message's structured
    content (``{"node_id": ..., "views": [...]}, ...) and runs
    :func:`create_fixture_proposal`. Registered against
    ``advisor.jobs.FIXTURE_PROPOSAL`` by the worker's caller."""

    conversation = conversations.get_conversation(job.conversation_id, db_path=db_path)
    if conversation is None:
        raise ValueError(f"conversation {job.conversation_id!r} not found")
    message = next(
        (
            m
            for m in conversations.list_messages(job.conversation_id, db_path=db_path)
            if m.message_id == job.request_message_id
        ),
        None,
    )
    if message is None:
        raise ValueError(f"message {job.request_message_id!r} not found")
    node_id = str(message.content["node_id"])
    views = list(message.content["views"])

    proposal = create_fixture_proposal(
        conversation.tree_id,
        node_id,
        views,
        caller_id=conversation.user_id,
        backend=backend,
        db_path=db_path,
    )
    conversations.add_message(
        job.conversation_id,
        "assistant",
        {"proposal_id": str(proposal.id), "status": "pending_approval"},
        db_path=db_path,
    )


# --------------------------------------------------------------------- #
# Approval / rejection
# --------------------------------------------------------------------- #
def get_proposal(
    proposal_id: UUID, *, db_path: str | os.PathLike[str] | None = None
) -> proposals.ProposalRecord:
    record = proposals.get(proposal_id, db_path=db_path)
    if record is None:
        raise ProposalNotFound(str(proposal_id))
    return record


def approve_proposal(
    proposal_id: UUID,
    *,
    proposal_hash: str,
    approved_by: str,
    idempotency_key: str,
    db_path: str | os.PathLike[str] | None = None,
) -> approval_service.ApprovalResult:
    """``proposal_hash`` must come from the caller (the UI's rendered card),
    never re-derived here from the stored row -- comparing a value against
    itself would defeat §8.3 step 2's whole purpose: catching a proposal
    that changed between when the UI displayed it and when the user clicked
    approve."""

    return approval_service.apply_proposal(
        proposal_id,
        proposal_hash=proposal_hash,
        approved_by=approved_by,
        idempotency_key=idempotency_key,
        db_path=db_path,
    )


def reject_proposal(
    proposal_id: UUID,
    *,
    rejected_by: str,
    reason: str | None = None,
    db_path: str | os.PathLike[str] | None = None,
) -> None:
    del rejected_by, reason  # not yet persisted as separate columns in the MVP schema
    record = get_proposal(proposal_id, db_path=db_path)
    proposals.transition(proposal_id, record.status, "rejected", db_path=db_path)


__all__ = [
    "ProposalNotFound",
    "TreeNotFound",
    "approve_proposal",
    "create_conversation",
    "create_fixture_proposal",
    "get_node_context",
    "get_proposal",
    "handle_fixture_proposal_job",
    "list_messages",
    "post_message_and_enqueue",
    "reject_proposal",
]
