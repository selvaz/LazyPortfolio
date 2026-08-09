"""Fase 5 exit criterion (docs/node-advisor-operational-plan.md §13):
"audit ricostruisce domanda -> evidenze -> proposal hash -> approval ->
revision -> confirmation run (provato end-to-end su un caso reale, non
solo per costruzione)".

"confirmation run" is not implemented anywhere in this codebase (a gap in
the original plan itself, not introduced by this phase -- see
docs/node-advisor-runbook.md §5): ``approval_service.apply_proposal``
stops at status ``"applied"``, no code ever transitions a proposal to
``confirmation_pending``/``confirmed``. This test reconstructs the chain
as far as it actually exists -- question -> proposal content_hash ->
approval record -> new tree revision -- entirely from the domain
repositories, proving the audit trail is real and queryable, not asserting
a confirmation step that was never built.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from lazyportfolio.advisor.approval_service import apply_proposal
from lazyportfolio.advisor.contracts import (
    ChangeProposal,
    CounterfactualResult,
    JsonPatchOperation,
    ModelProvenance,
    ProposedView,
    SnapshotDescriptor,
    ValidationResult,
)
from lazyportfolio.advisor.conversation_repository import (
    add_message,
    create_conversation,
    list_messages,
)
from lazyportfolio.advisor.proposal_repository import create as create_proposal_record
from lazyportfolio.advisor.proposal_repository import get as get_proposal_record
from lazyportfolio.advisor.repository import create_tree, get_head

_NOW = datetime(2026, 8, 9, tzinfo=UTC)


def _config() -> dict[str, object]:
    return {
        "root_id": "root",
        "currency": "USD",
        "nodes": [
            {
                "id": "root",
                "name": "Root",
                "children": ["equity"],
                "instruments": [],
                "goal": {"objective": "min_risk"},
                "constraints": {},
            },
            {
                "id": "equity",
                "name": "Equity",
                "children": [],
                "instruments": ["ticker:VTI", "ticker:VXUS"],
                "proxy": "ticker:VTI",
                "goal": {"objective": "min_risk"},
                "constraints": {},
            },
        ],
        "backtest": {
            "benchmark": {"name": "B0", "weights": {"ticker:VTI": 0.5, "ticker:VXUS": 0.5}}
        },
    }


def test_the_full_chain_from_question_to_new_revision_is_reconstructable(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "db.sqlite3"

    # 1. Question -- a conversation and its triggering user message.
    tree = create_tree(_config(), actor_type="human", actor_id="audit-user", db_path=db_path)
    conversation = create_conversation(
        tree.tree_id, "equity", user_id="audit-user", db_path=db_path
    )
    user_message = add_message(
        conversation.conversation_id,
        "user",
        {"node_id": "equity", "text": "propose a relative equity view"},
        db_path=db_path,
    )

    # 2. Proposal, with content_hash -- the immutable artifact a human
    # would have reviewed before approving.
    patch = [JsonPatchOperation(op="replace", path="/nodes/equity/constraints/views", value=[])]
    proposal = ChangeProposal(
        id=uuid4(),
        schema_version="1.0",
        kind="replace_node_views",
        tree_id=UUID(tree.tree_id),
        base_revision_id=UUID(tree.revision_id),
        node_id="equity",
        snapshot=SnapshotDescriptor(
            schema_version="1.0",
            source="market-data-hub",
            database_identity="audit-test",
            universe=["ticker:VTI", "ticker:VXUS"],
            field="close",
            currency="USD",
            frequency="D",
            fingerprint="sha256:audit-fixture",
        ),
        information_cutoff=_NOW,
        patch=patch,
        proposed_views=[
            ProposedView(
                instruments={"ticker:VTI": 1.0, "ticker:VXUS": -1.0},
                expected_return=0.02,
                confidence=0.6,
                rationale="audit reconstruction test",
            )
        ],
        rationale="audit reconstruction test",
        model_provenance=ModelProvenance(
            producer_kind="interactive_chat", producer_id="node-advisor-agent", model="test-model"
        ),
        validation=ValidationResult(valid=True),
        counterfactual=CounterfactualResult(),
        expires_at=datetime(2026, 12, 31, tzinfo=UTC),
        content_hash="sha256:audit-reconstruction-fixture",
    )
    create_proposal_record(proposal, status="pending_approval", db_path=db_path)
    assistant_message = add_message(
        conversation.conversation_id,
        "assistant",
        {"route": "propose", "message": proposal.rationale, "proposal_id": str(proposal.id)},
        db_path=db_path,
    )

    # 3. Approval.
    result = apply_proposal(
        proposal.id,
        proposal_hash=proposal.content_hash,
        approved_by="audit-user",
        idempotency_key="audit-reconstruction-key",
        db_path=db_path,
    )

    # --- Reconstruct the chain from scratch, using only proposal_id --- #
    record = get_proposal_record(proposal.id, db_path=db_path)
    assert record is not None
    assert record.status == "applied"

    # Question: the conversation this proposal came from, and the message
    # that triggered it, are both still reachable.
    messages = list_messages(conversation.conversation_id, db_path=db_path)
    assert any(m.message_id == user_message.message_id and m.role == "user" for m in messages)
    assert any(
        m.message_id == assistant_message.message_id
        and m.content.get("proposal_id") == str(proposal.id)
        for m in messages
    )

    # Proposal hash: the exact value a human would have compared before
    # clicking approve is still the one stored -- unchanged, unforgeable.
    assert record.proposal.content_hash == proposal.content_hash

    # Approval -> new revision: the applied proposal points at a real,
    # fetchable revision that is now the tree's head.
    assert result.new_revision_id
    head = get_head(tree.tree_id, db_path=db_path)
    assert head is not None
    assert head.revision_id == result.new_revision_id
    assert head.parent_revision_id == tree.revision_id
    equity = next(n for n in head.config["nodes"] if n["id"] == "equity")  # type: ignore[index]
    assert equity["constraints"]["views"][0]["instruments"] == {
        "ticker:VTI": 1.0,
        "ticker:VXUS": -1.0,
    }

    # No "confirmation run" state exists to reconstruct further -- see
    # docs/node-advisor-runbook.md §5 for why this is a disclosed gap, not
    # an oversight of this test.
    from lazyportfolio.advisor.state_machine import TERMINAL_STATUSES

    assert "applied" not in TERMINAL_STATUSES  # confirmed by design (§4.5) -- just never reached
