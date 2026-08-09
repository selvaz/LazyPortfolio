from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from lazyportfolio.copilot.contracts import (
    ChangeProposal,
    CounterfactualResult,
    JsonPatchOperation,
    ModelProvenance,
    SnapshotDescriptor,
    ValidationResult,
)
from lazyportfolio.copilot.proposal_repository import (
    ConcurrentProposalWrite,
    create,
    get,
    list_by_tree,
    transition,
)
from lazyportfolio.copilot.state_machine import IllegalProposalTransition

_NOW = datetime(2026, 8, 9, tzinfo=UTC)


def _snapshot() -> SnapshotDescriptor:
    return SnapshotDescriptor(
        schema_version="1.0",
        source="market-data-hub",
        database_identity="test",
        universe=["ticker:VTI"],
        field="close",
        currency="USD",
        frequency="D",
        fingerprint="sha256:abc",
    )


def _proposal(*, tree_id: UUID | None = None, node_id: str = "equity") -> ChangeProposal:
    return ChangeProposal(
        id=uuid4(),
        schema_version="1.0",
        kind="replace_node_views",
        tree_id=tree_id or uuid4(),
        base_revision_id=uuid4(),
        node_id=node_id,
        snapshot=_snapshot(),
        information_cutoff=_NOW,
        patch=[
            JsonPatchOperation(op="replace", path=f"/nodes/{node_id}/constraints/views", value=[])
        ],
        rationale="test",
        model_provenance=ModelProvenance(
            producer_kind="interactive_chat", producer_id="node-copilot", model="test-model"
        ),
        validation=ValidationResult(valid=True),
        counterfactual=CounterfactualResult(),
        expires_at=_NOW,
        content_hash="sha256:0000",
    )


def test_create_then_get_round_trips(tmp_path: Path) -> None:
    db_path = tmp_path / "db.sqlite3"
    proposal = _proposal()
    record = create(proposal, db_path=db_path)
    assert record.status == "drafting"

    fetched = get(proposal.id, db_path=db_path)
    assert fetched is not None
    assert fetched.status == "drafting"
    assert fetched.proposal.id == proposal.id
    assert fetched.proposal.content_hash == proposal.content_hash


def test_get_missing_proposal_returns_none(tmp_path: Path) -> None:
    assert get(uuid4(), db_path=tmp_path / "db.sqlite3") is None


def test_list_by_tree_returns_only_that_trees_proposals(tmp_path: Path) -> None:
    db_path = tmp_path / "db.sqlite3"
    tree_a = uuid4()
    tree_b = uuid4()
    proposal_a = _proposal(tree_id=tree_a)
    proposal_b = _proposal(tree_id=tree_b)
    create(proposal_a, db_path=db_path)
    create(proposal_b, db_path=db_path)

    results = list_by_tree(tree_a, db_path=db_path)
    assert [record.proposal.id for record in results] == [proposal_a.id]


def test_legal_transition_updates_status(tmp_path: Path) -> None:
    db_path = tmp_path / "db.sqlite3"
    proposal = _proposal()
    create(proposal, db_path=db_path)
    transition(proposal.id, "drafting", "pending_approval", db_path=db_path)
    fetched = get(proposal.id, db_path=db_path)
    assert fetched is not None
    assert fetched.status == "pending_approval"


def test_illegal_transition_raises_before_touching_the_database(tmp_path: Path) -> None:
    db_path = tmp_path / "db.sqlite3"
    proposal = _proposal()
    create(proposal, db_path=db_path)
    with pytest.raises(IllegalProposalTransition):
        transition(proposal.id, "drafting", "applied", db_path=db_path)
    fetched = get(proposal.id, db_path=db_path)
    assert fetched is not None
    assert fetched.status == "drafting"


def test_transition_from_the_wrong_current_status_raises_concurrent_write(
    tmp_path: Path,
) -> None:
    """The DB's actual status is 'pending_approval', but the caller believes
    it is still 'drafting' -- e.g. it raced with another transition."""

    db_path = tmp_path / "db.sqlite3"
    proposal = _proposal()
    create(proposal, db_path=db_path)
    transition(proposal.id, "drafting", "pending_approval", db_path=db_path)

    with pytest.raises(ConcurrentProposalWrite):
        transition(proposal.id, "drafting", "failed", db_path=db_path)
