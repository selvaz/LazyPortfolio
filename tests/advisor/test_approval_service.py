"""docs/node-advisor-operational-plan.md §13 Fase 1 exit criteria for the
approval transaction: happy path applies exactly one new revision; a stale
base revision or a stale data snapshot both refuse (and expire) instead of
silently applying; a retried idempotency_key returns the same result rather
than creating a second revision."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest

import sqlite3

from lazyportfolio.advisor import approval_service
from lazyportfolio.advisor.approval_service import (
    ApprovalHashMismatch,
    ProposalExpired,
    ProposalNotFound,
    ProposalNotPendingApproval,
    StaleDataError,
    StaleRevisionError,
    apply_proposal,
)
from lazyportfolio.advisor.contracts import (
    ChangeProposal,
    CounterfactualResult,
    JsonPatchOperation,
    ModelProvenance,
    ProposedView,
    SnapshotDescriptor,
    ValidationResult,
)
from lazyportfolio.advisor.proposal_repository import create as create_proposal
from lazyportfolio.advisor.proposal_repository import get as get_proposal
from lazyportfolio.advisor.repository import create_tree, get_head, save_revision

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
            producer_kind="interactive_chat", producer_id="node-advisor", model="test-model"
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
            producer_kind="interactive_chat", producer_id="node-advisor", model="test-model"
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


def test_a_concurrent_reject_between_the_status_read_and_the_write_wins(
    tmp_path: Path, monkeypatch
) -> None:
    """Regression for a real race: step 1 reads status='pending_approval',
    but sqlite only opens an implicit transaction at the first WRITE
    (step 8) -- so a reject_proposal() committing in that window must stop
    the apply, not lose to it. validate_patch() runs at step 5, strictly
    between the read and the first write, so patching it to also perform
    the concurrent reject (via a SEPARATE connection, exactly as a second
    process/request would) reproduces the real interleaving deterministically
    instead of relying on thread timing."""

    db_path = tmp_path / "db.sqlite3"
    tree = create_tree(_base_config(), actor_type="human", actor_id="local-user", db_path=db_path)
    proposal = _pending_proposal(
        tree_id=UUID(tree.tree_id), base_revision_id=tree.revision_id, db_path=db_path
    )

    real_validate_patch = approval_service.validate_patch

    def _validate_then_concurrent_reject(patch: object, node_id: str) -> None:
        real_validate_patch(patch, node_id)  # type: ignore[arg-type]
        # A second connection, exactly as reject_proposal()'s own separate
        # call would use -- and it commits before this function's caller
        # (apply_proposal) has made ANY write, so nothing blocks it.
        other_conn = sqlite3.connect(str(db_path))
        try:
            other_conn.execute(
                "UPDATE change_proposals SET status = 'rejected' WHERE proposal_id = ?",
                (str(proposal.id),),
            )
            other_conn.commit()
        finally:
            other_conn.close()

    monkeypatch.setattr(approval_service, "validate_patch", _validate_then_concurrent_reject)

    with pytest.raises(ProposalNotPendingApproval) as excinfo:
        apply_proposal(
            proposal.id,
            proposal_hash=proposal.content_hash,
            approved_by="local-user",
            idempotency_key="race-key",
            db_path=db_path,
        )
    assert excinfo.value.status == "rejected"  # the real current status, not prose

    # The whole transaction must have rolled back -- not just the final
    # status write. A partial apply (revision inserted, head moved, but
    # proposal left 'rejected') would be worse than a clean failure: check
    # every write step 8-10 staged, not just the head, so a bug that rolls
    # back the head but leaves an orphan revision/approval/outbox row can't
    # hide behind this test.
    head = get_head(tree.tree_id, db_path=db_path)
    assert head is not None
    assert head.revision_id == tree.revision_id  # unchanged
    fetched = get_proposal(proposal.id, db_path=db_path)
    assert fetched is not None
    assert fetched.status == "rejected"  # the concurrent reject, untouched

    check_conn = sqlite3.connect(str(db_path))
    try:
        revisions = check_conn.execute(
            "SELECT COUNT(*) FROM tree_revisions WHERE tree_id = ?", (str(tree.tree_id),)
        ).fetchone()[0]
        approvals = check_conn.execute(
            "SELECT COUNT(*) FROM proposal_approvals WHERE proposal_id = ?",
            (str(proposal.id),),
        ).fetchone()[0]
        outbox = check_conn.execute("SELECT COUNT(*) FROM outbox_events").fetchone()[0]
    finally:
        check_conn.close()
    assert revisions == 1, "no orphan tree_revisions row from the rolled-back insert"
    assert approvals == 0, "no orphan proposal_approvals row from the rolled-back insert"
    assert outbox == 0, "no orphan outbox_events row from the rolled-back insert"


def test_services_approve_proposal_wires_a_real_fingerprint_recheck(
    tmp_path: Path, monkeypatch
) -> None:
    """project/advisor/services.py:approve_proposal is the actual
    HTTP-reachable approval path (project/advisor/api.py's only caller).
    It must NOT fall back to apply_proposal's bare default
    (_trust_stored_fingerprint, which always matches itself and can never
    detect stale data) -- assert this by making the real recompute function
    return a different value and checking the mismatch is actually seen."""

    from lazyportfolio.advisor import snapshot as snapshot_service
    from project.advisor import services

    db_path = tmp_path / "db.sqlite3"
    tree = create_tree(_base_config(), actor_type="human", actor_id="local-user", db_path=db_path)
    proposal = _pending_proposal(
        tree_id=UUID(tree.tree_id), base_revision_id=tree.revision_id, db_path=db_path
    )

    monkeypatch.setattr(
        snapshot_service, "recompute_snapshot_fingerprint", lambda snapshot: "sha256:changed"
    )

    with pytest.raises(StaleDataError):
        services.approve_proposal(
            proposal.id,
            proposal_hash=proposal.content_hash,
            approved_by="local-user",
            idempotency_key="wiring-key",
            db_path=db_path,
        )
