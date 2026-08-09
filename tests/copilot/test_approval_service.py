"""docs/node-copilot-operational-plan.md §13 Fase 1 exit criteria for the
approval transaction: happy path applies exactly one new revision; a stale
base revision or a stale data snapshot both refuse (and expire) instead of
silently applying; a retried idempotency_key returns the same result rather
than creating a second revision."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from lazyportfolio.copilot.approval_service import (
    ApprovalHashMismatch,
    ProposalExpired,
    ProposalNotFound,
    ProposalNotPendingApproval,
    StaleDataError,
    StaleRevisionError,
    apply_proposal,
)
from lazyportfolio.copilot.contracts import (
    ChangeProposal,
    CounterfactualResult,
    JsonPatchOperation,
    ModelProvenance,
    ProposedView,
    SnapshotDescriptor,
    ValidationResult,
)
from lazyportfolio.copilot.proposal_repository import create as create_proposal
from lazyportfolio.copilot.proposal_repository import get as get_proposal
from lazyportfolio.copilot.repository import create_tree, get_head, save_revision

_NOW = datetime(2026, 8, 9, tzinfo=UTC)
_FUTURE = datetime(2026, 12, 31, tzinfo=UTC)


def _base_config() -> dict[str, object]:
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
            "benchmark": {
                "name": "B0",
                "weights": {"ticker:VTI": 0.5, "ticker:VXUS": 0.5},
            }
        },
    }


def _snapshot(fingerprint: str = "sha256:fixture") -> SnapshotDescriptor:
    return SnapshotDescriptor(
        schema_version="1.0",
        source="market-data-hub",
        database_identity="test",
        universe=["ticker:VTI", "ticker:VXUS"],
        field="close",
        currency="USD",
        frequency="D",
        fingerprint=fingerprint,
    )


def _pending_proposal(
    *, tree_id: UUID, base_revision_id: str, db_path: Path, expires_at: datetime = _FUTURE
) -> ChangeProposal:
    proposal = ChangeProposal(
        id=uuid4(),
        schema_version="1.0",
        kind="replace_node_views",
        tree_id=tree_id,
        base_revision_id=UUID(base_revision_id),
        node_id="equity",
        snapshot=_snapshot(),
        information_cutoff=_NOW,
        patch=[
            JsonPatchOperation(op="replace", path="/nodes/equity/constraints/views", value=[])
        ],
        proposed_views=[
            ProposedView(
                instruments={"ticker:VTI": 1.0, "ticker:VXUS": -1.0},
                expected_return=0.02,
                confidence=0.6,
                rationale="test view",
            )
        ],
        rationale="test",
        model_provenance=ModelProvenance(
            producer_kind="interactive_chat", producer_id="node-copilot", model="test-model"
        ),
        validation=ValidationResult(valid=True),
        counterfactual=CounterfactualResult(),
        expires_at=expires_at,
        content_hash="sha256:proposalhash",
    )
    create_proposal(proposal, status="pending_approval", db_path=db_path)
    return proposal


def test_happy_path_applies_the_views_and_advances_the_head(tmp_path: Path) -> None:
    db_path = tmp_path / "db.sqlite3"
    tree = create_tree(_base_config(), actor_type="human", actor_id="local-user", db_path=db_path)
    proposal = _pending_proposal(
        tree_id=UUID(tree.tree_id), base_revision_id=tree.revision_id, db_path=db_path
    )

    result = apply_proposal(
        proposal.id,
        proposal_hash=proposal.content_hash,
        approved_by="local-user",
        idempotency_key="key-1",
        db_path=db_path,
    )

    assert result.new_revision_id != tree.revision_id
    head = get_head(tree.tree_id, db_path=db_path)
    assert head is not None
    assert head.revision_id == result.new_revision_id
    equity = next(n for n in head.config["nodes"] if n["id"] == "equity")  # type: ignore[index]
    assert equity["constraints"]["views"][0]["instruments"] == {
        "ticker:VTI": 1.0,
        "ticker:VXUS": -1.0,
    }
    fetched = get_proposal(proposal.id, db_path=db_path)
    assert fetched is not None
    assert fetched.status == "applied"


def test_missing_proposal_raises(tmp_path: Path) -> None:
    with pytest.raises(ProposalNotFound):
        apply_proposal(
            uuid4(),
            proposal_hash="sha256:x",
            approved_by="local-user",
            idempotency_key="key-1",
            db_path=tmp_path / "db.sqlite3",
        )


def test_proposal_not_pending_approval_raises(tmp_path: Path) -> None:
    db_path = tmp_path / "db.sqlite3"
    tree = create_tree(_base_config(), actor_type="human", actor_id="local-user", db_path=db_path)
    proposal = ChangeProposal(
        id=uuid4(),
        schema_version="1.0",
        kind="replace_node_views",
        tree_id=UUID(tree.tree_id),
        base_revision_id=UUID(tree.revision_id),
        node_id="equity",
        snapshot=_snapshot(),
        information_cutoff=_NOW,
        patch=[],
        rationale="test",
        model_provenance=ModelProvenance(
            producer_kind="interactive_chat", producer_id="node-copilot", model="test-model"
        ),
        validation=ValidationResult(valid=True),
        counterfactual=CounterfactualResult(),
        expires_at=_FUTURE,
        content_hash="sha256:proposalhash",
    )
    create_proposal(proposal, status="drafting", db_path=db_path)  # never moved to pending_approval

    with pytest.raises(ProposalNotPendingApproval):
        apply_proposal(
            proposal.id,
            proposal_hash=proposal.content_hash,
            approved_by="local-user",
            idempotency_key="key-1",
            db_path=db_path,
        )


def test_hash_mismatch_raises_without_applying_anything(tmp_path: Path) -> None:
    db_path = tmp_path / "db.sqlite3"
    tree = create_tree(_base_config(), actor_type="human", actor_id="local-user", db_path=db_path)
    proposal = _pending_proposal(
        tree_id=UUID(tree.tree_id), base_revision_id=tree.revision_id, db_path=db_path
    )

    with pytest.raises(ApprovalHashMismatch):
        apply_proposal(
            proposal.id,
            proposal_hash="sha256:not-the-real-hash",
            approved_by="local-user",
            idempotency_key="key-1",
            db_path=db_path,
        )
    head = get_head(tree.tree_id, db_path=db_path)
    assert head is not None
    assert head.revision_id == tree.revision_id  # unchanged


def test_expired_proposal_raises_and_is_marked_expired(tmp_path: Path) -> None:
    db_path = tmp_path / "db.sqlite3"
    tree = create_tree(_base_config(), actor_type="human", actor_id="local-user", db_path=db_path)
    past = datetime(2020, 1, 1, tzinfo=UTC)
    proposal = _pending_proposal(
        tree_id=UUID(tree.tree_id),
        base_revision_id=tree.revision_id,
        db_path=db_path,
        expires_at=past,
    )

    with pytest.raises(ProposalExpired):
        apply_proposal(
            proposal.id,
            proposal_hash=proposal.content_hash,
            approved_by="local-user",
            idempotency_key="key-1",
            db_path=db_path,
        )
    fetched = get_proposal(proposal.id, db_path=db_path)
    assert fetched is not None
    assert fetched.status == "expired"


def test_stale_revision_raises_when_head_moved_since_the_proposal_was_drafted(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "db.sqlite3"
    tree = create_tree(_base_config(), actor_type="human", actor_id="local-user", db_path=db_path)
    proposal = _pending_proposal(
        tree_id=UUID(tree.tree_id), base_revision_id=tree.revision_id, db_path=db_path
    )
    # A human save moves the head to a revision the proposal was never based on.
    save_revision(
        tree.tree_id, _base_config(), actor_type="human", actor_id="someone-else", db_path=db_path
    )

    with pytest.raises(StaleRevisionError):
        apply_proposal(
            proposal.id,
            proposal_hash=proposal.content_hash,
            approved_by="local-user",
            idempotency_key="key-1",
            db_path=db_path,
        )
    fetched = get_proposal(proposal.id, db_path=db_path)
    assert fetched is not None
    assert fetched.status == "expired"


def test_stale_data_raises_when_the_recomputed_fingerprint_differs(tmp_path: Path) -> None:
    db_path = tmp_path / "db.sqlite3"
    tree = create_tree(_base_config(), actor_type="human", actor_id="local-user", db_path=db_path)
    proposal = _pending_proposal(
        tree_id=UUID(tree.tree_id), base_revision_id=tree.revision_id, db_path=db_path
    )

    with pytest.raises(StaleDataError):
        apply_proposal(
            proposal.id,
            proposal_hash=proposal.content_hash,
            approved_by="local-user",
            idempotency_key="key-1",
            recompute_fingerprint=lambda snapshot: "sha256:a-corrected-history",
            db_path=db_path,
        )
    fetched = get_proposal(proposal.id, db_path=db_path)
    assert fetched is not None
    assert fetched.status == "expired"


def test_retrying_the_same_idempotency_key_returns_the_same_result_not_a_new_revision(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "db.sqlite3"
    tree = create_tree(_base_config(), actor_type="human", actor_id="local-user", db_path=db_path)
    proposal = _pending_proposal(
        tree_id=UUID(tree.tree_id), base_revision_id=tree.revision_id, db_path=db_path
    )

    first = apply_proposal(
        proposal.id,
        proposal_hash=proposal.content_hash,
        approved_by="local-user",
        idempotency_key="same-key",
        db_path=db_path,
    )
    second = apply_proposal(
        proposal.id,
        proposal_hash=proposal.content_hash,
        approved_by="local-user",
        idempotency_key="same-key",
        db_path=db_path,
    )

    assert first == second
    head = get_head(tree.tree_id, db_path=db_path)
    assert head is not None
    assert head.revision_id == first.new_revision_id  # only ever advanced once
