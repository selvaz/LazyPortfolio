"""Application services wiring ``lazyportfolio.advisor`` into Tree Studio.

docs/node-advisor-operational-plan.md §13 Fase 3. Every function takes the
caller's identity as an explicit parameter (``caller_id``/``approved_by``/
``user_id``), never read from an implicit request-scoped global --
docs/adr/0001-node-advisor-architecture.md Decision 3 point 4, so this same
layer stays callable from a future scheduled job (Fase 6's Investment
Batch producer) as well as an HTTP request.

``create_proposal`` is the whole "proposal preparation" pipeline, shared by
Fase 3's fixture job handler and Fase 4's LLM-driven
``advisor.agent.run_advisor_turn``: ``views`` are validated and
counterfactually evaluated identically regardless of whether they came from
a fixture or an LLM's structured output -- the pipeline itself never knows
or cares which.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

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
    ProducerKind,
    ProposedView,
)
from lazyportfolio.advisor.patch import views_patch_path
from lazyportfolio.v2.mode import mode_from_config
from lazyportfolio.v2.model import V2Model
from project.advisor import jobs

if TYPE_CHECKING:
    from lazyportfolio import OptimizationDataBackend
    from project.advisor.jobs import JobRecord

#: How long a fixture-created proposal stays approvable before it must be
#: re-derived. Arbitrary but generous for a local single-user tool -- not a
#: production SLA.
_DEFAULT_PROPOSAL_TTL = timedelta(hours=24)


#: ``node_id`` for a coordinator conversation, which is scoped to a tree
#: rather than a node. Safe as a sentinel because ``Conversation.node_id`` is
#: a plain string never validated against the tree, and the advisor job
#: handler reads the *message's* node_id, never the conversation's.
COORDINATOR_SCOPE = "__coordinator__"


class TreeNotFound(ValueError):
    """``tree_id`` has no revisions (see ``lazyportfolio.advisor.repository``)."""


class ProposalNotFound(ValueError):
    pass


class StaleBaseRevision(ValueError):
    """The tree moved away from the revision the caller pinned."""


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


def get_head_revision_id(
    tree_id: str, *, db_path: str | os.PathLike[str] | None = None
) -> str:
    """The tree's current head revision id.

    Exists so a long, multi-step caller (the tree coordinator) can pin the
    revision it reasoned against and re-check it right before persisting.
    ``get_node_context`` and ``create_proposal`` each read the head
    independently, so without this a tree that moved mid-turn would be
    reasoned about at R1 and written against R2 with nothing noticing.
    """

    head = tree_repository.get_head(tree_id, db_path=db_path)
    if head is None:
        raise TreeNotFound(tree_id)
    return str(head.revision_id)


def get_tree_summary(
    tree_id: str, *, db_path: str | os.PathLike[str] | None = None
) -> dict[str, Any]:
    """The whole tree's structure, from a single config parse.

    ``get_node_context`` rebuilds ``V2Model.from_config`` on every call, so
    "resolve every node" costs N parses of the same dict. A tree-wide caller
    wants the shape once; this gives it that for one parse, and deliberately
    returns far less per node than a :class:`NodeContext` (see
    ``node_universe.build_tree_summary``).
    """

    head = tree_repository.get_head(tree_id, db_path=db_path)
    if head is None:
        raise TreeNotFound(tree_id)
    model = V2Model.from_config(head.config)
    summary = node_universe.build_tree_summary(model)
    return {"tree_id": str(head.tree_id), "revision_id": str(head.revision_id), **summary}


# --------------------------------------------------------------------- #
# Conversations / messages
# --------------------------------------------------------------------- #
def create_conversation(
    tree_id: str, node_id: str, *, caller_id: str, db_path: str | os.PathLike[str] | None = None
) -> conversations.Conversation:
    return conversations.create_conversation(
        tree_id, node_id, user_id=caller_id, db_path=db_path
    )


def get_conversation(
    conversation_id: str, *, db_path: str | os.PathLike[str] | None = None
) -> conversations.Conversation | None:
    return conversations.get_conversation(conversation_id, db_path=db_path)


def list_messages(
    conversation_id: str, *, db_path: str | os.PathLike[str] | None = None
) -> list[conversations.Message]:
    return conversations.list_messages(conversation_id, db_path=db_path)


def post_message_and_enqueue(
    conversation_id: str,
    content: dict[str, Any],
    *,
    caller_id: str,
    job_kind: str = jobs.FIXTURE_PROPOSAL,
    db_path: str | os.PathLike[str] | None = None,
) -> tuple[conversations.Message, str]:
    """Record a user message and enqueue the job it triggers.

    ``job_kind`` selects which worker handler processes it --
    ``jobs.FIXTURE_PROPOSAL`` (deterministic, no LLM, ``content`` carries
    pre-supplied ``views``) or ``jobs.ADVISOR_TURN`` (real LLM call via
    :func:`advisor.agent.run_advisor_turn`, ``content`` carries free-text
    ``text``) -- this function itself is agnostic to the shape of
    ``content``, it only routes.

    Returns ``(message, job_id)``. The HTTP layer responds with the job id
    immediately -- the actual proposal preparation runs on the worker
    thread, never in the request thread (§11).
    """

    del caller_id  # recorded on the conversation itself, not per-message in the MVP
    message = conversations.add_message(conversation_id, "user", content, db_path=db_path)
    job_id = jobs.enqueue_job(conversation_id, message.message_id, job_kind, db_path=db_path)
    return message, job_id


# --------------------------------------------------------------------- #
# Proposal preparation (the fixture job handler)
# --------------------------------------------------------------------- #
def create_proposal(
    tree_id: str,
    node_id: str,
    views: list[dict[str, Any]],
    *,
    caller_id: str,
    rationale: str = "Fixture proposal (Fase 3: no LLM in this phase).",
    producer_kind: ProducerKind = "interactive_chat",
    producer_id: str = "fixture",
    model: str = "none (Fase 3, no LLM)",
    batch_id: UUID | None = None,
    expected_revision_id: str | None = None,
    backend: OptimizationDataBackend | None = None,
    db_path: str | os.PathLike[str] | None = None,
) -> ChangeProposal:
    """Validate, counterfactually evaluate, and persist a
    ``pending_approval`` proposal for ``views`` on ``node_id`` -- pipeline
    steps 5-9 of §8.2's Plan, shared by Fase 3's fixture job handler
    (default ``producer_id="fixture"``), Fase 4's LLM-driven
    :func:`advisor.agent.run_advisor_turn` (real model name,
    ``producer_id="node-advisor-agent"``), and Fase 6's
    :func:`advisor.batch_producer.run_proposal_batch` (``producer_kind=
    "scheduled_batch"``, a shared ``batch_id`` across one run's proposals)
    -- the exact producer-agnostic reuse docs/adr/0001-node-advisor-architecture.md
    Decision 3 was written to make possible: this function has no branch on
    which producer called it.

    ``batch_id`` is ``None`` for the Node Advisor's own conversational flow
    (the default) and set by a batch producer to group one run's proposals
    (§3.4 point 2).

    ``expected_revision_id`` pins the revision the caller reasoned against.
    A caller that spent minutes thinking (the tree coordinator walks a whole
    hierarchy of nested LLM calls) would otherwise validate and attach its
    proposal to whichever head exists *now*, not the one it saw. Checked both
    before the work and again after the slow snapshot/counterfactual step,
    which is itself long enough for the head to move. This narrows the window
    to the gap between the second check and the insert; closing it entirely
    needs the insert to be conditional on the head inside one transaction,
    which is a change to the write path rather than to this caller-supplied
    guard.
    """

    head = tree_repository.get_head(tree_id, db_path=db_path)
    if head is None:
        raise TreeNotFound(tree_id)
    if expected_revision_id is not None and str(head.revision_id) != expected_revision_id:
        raise StaleBaseRevision(
            f"tree {tree_id} moved from {expected_revision_id} to {head.revision_id}"
        )
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

    if expected_revision_id is not None:
        current = tree_repository.get_head(tree_id, db_path=db_path)
        if current is None or str(current.revision_id) != expected_revision_id:
            raise StaleBaseRevision(
                f"tree {tree_id} moved to "
                f"{current.revision_id if current else 'nothing'} while this proposal "
                "was being evaluated"
            )

    now = datetime.now(UTC)
    patch = [
        JsonPatchOperation(op="replace", path=views_patch_path(node_id), value=None),
    ]
    provenance = ModelProvenance(
        producer_kind=producer_kind, producer_id=producer_id, model=model
    )
    draft = ChangeProposal(
        id=uuid4(),
        schema_version="1.0",
        kind="replace_node_views",
        batch_id=batch_id,
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

    try:
        # Inserted at its final status in one commit, rather than written as
        # "drafting" and then transitioned. Two commits meant a window where
        # the proposal existed but the transition had failed -- the caller
        # sees an exception and reasonably assumes nothing was written, then
        # retries and files a second one. Nothing ever observed the
        # intermediate state.
        proposals.create(
            proposal,
            status="pending_approval",
            require_head_revision_id=expected_revision_id,
            db_path=db_path,
        )
    except proposals.StaleBaseRevisionOnCreate as exc:
        # The database refused the insert because the head moved between the
        # last check and this write. Re-raised as this module's own stale
        # error so every caller sees one exception type for "the tree moved",
        # whichever of the three checks caught it.
        raise StaleBaseRevision(str(exc)) from exc
    return proposal


def handle_fixture_proposal_job(
    job: JobRecord,
    *,
    backend: OptimizationDataBackend | None = None,
    db_path: str | os.PathLike[str] | None = None,
) -> None:
    """The one MVP job handler: reads the triggering message's structured
    content (``{"node_id": ..., "views": [...]}, ...) and runs
    :func:`create_proposal`. Registered against
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

    proposal = create_proposal(
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


def handle_advisor_turn_job(
    job: JobRecord,
    *,
    backend: OptimizationDataBackend | None = None,
    db_path: str | os.PathLike[str] | None = None,
) -> None:
    """The Fase 4/5 job handler: reads the triggering message's free-text
    content (``{"node_id": ..., "text": ...}``) and runs
    :func:`advisor.agent.run_advisor_turn`. Registered against
    ``advisor.jobs.ADVISOR_TURN`` by the worker's caller (Fase 5: wired
    into ``project/tree_studio.py``'s live worker, not just exercised by
    tests -- see docs/node-advisor-operational-plan.md §13 Fase 5).

    Imports ``project.advisor.agent`` lazily (module-level would be a
    circular import: the agent itself imports this module).
    """

    from project.advisor import agent as advisor_agent

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
    text = str(message.content["text"])

    result = advisor_agent.run_advisor_turn(
        conversation.tree_id,
        node_id,
        text,
        caller_id=conversation.user_id,
        backend=backend,
        db_path=db_path,
    )
    proposal = result["proposal"]
    conversations.add_message(
        job.conversation_id,
        "assistant",
        {
            "route": result["route"],
            "message": result["message"],
            "proposal_id": proposal["id"] if proposal else None,
            "status": "pending_approval" if proposal else None,
        },
        db_path=db_path,
    )


def handle_coordinator_turn_job(
    job: JobRecord,
    *,
    backend: OptimizationDataBackend | None = None,
    db_path: str | os.PathLike[str] | None = None,
) -> None:
    """Run one tree-coordinator turn on the worker thread.

    Mirrors :func:`handle_advisor_turn_job`, minus the ``node_id``: a
    coordinator conversation is scoped to the tree, and which nodes get
    consulted is the model's decision, not the request's.

    Imports :mod:`project.advisor.coordinator` lazily for the same reason
    the advisor handler does -- that module imports this one.
    """

    from project.advisor import coordinator as tree_coordinator

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

    result = tree_coordinator.run_coordinator_turn(
        conversation.tree_id,
        str(message.content["text"]),
        caller_id=conversation.user_id,
        conversation_id=job.conversation_id,
        current_message_id=job.request_message_id,
        backend=backend,
        db_path=db_path,
    )
    conversations.add_message(
        job.conversation_id,
        "assistant",
        {
            "kind": "coordinator_turn_result",
            "message": result["message"],
            "batch_id": result["batch_id"],
            "draft_config": result["draft_config"],
            "draft_error": result["draft_error"],
        },
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


def list_proposals(
    tree_id: str,
    *,
    batch_id: UUID | str | None = None,
    db_path: str | os.PathLike[str] | None = None,
) -> list[proposals.ProposalRecord]:
    """Proposals for one tree, newest first, optionally one batch of them.

    Always scoped by ``tree_id`` even when a ``batch_id`` is given: a batch
    id alone is an unowned identifier, and querying by it in isolation would
    let a caller who guessed one read across trees.
    """

    records = proposals.list_by_tree(UUID(tree_id), db_path=db_path)
    if batch_id is None:
        return records
    wanted = UUID(str(batch_id))
    return [r for r in records if r.proposal.batch_id == wanted]


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
    approve.

    Passes the real ``recompute_snapshot_fingerprint`` (Fase 2's
    ``SnapshotService``) so this, the actual HTTP-reachable approval path,
    rechecks live market data rather than trusting the stored snapshot's own
    fingerprint back at itself -- ``apply_proposal``'s bare default
    (``_trust_stored_fingerprint``) is for fixture-driven tests only."""

    return approval_service.apply_proposal(
        proposal_id,
        proposal_hash=proposal_hash,
        approved_by=approved_by,
        idempotency_key=idempotency_key,
        recompute_fingerprint=snapshot_service.recompute_snapshot_fingerprint,
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
    "COORDINATOR_SCOPE",
    "ProposalNotFound",
    "StaleBaseRevision",
    "TreeNotFound",
    "approve_proposal",
    "create_conversation",
    "create_proposal",
    "get_conversation",
    "get_head_revision_id",
    "get_node_context",
    "get_proposal",
    "get_tree_summary",
    "handle_advisor_turn_job",
    "handle_coordinator_turn_job",
    "handle_fixture_proposal_job",
    "list_messages",
    "list_proposals",
    "post_message_and_enqueue",
    "reject_proposal",
]
