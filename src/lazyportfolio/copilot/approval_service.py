"""Proposal approval: the single atomic transaction that turns an approved
``ChangeProposal`` into a new tree revision (docs/node-copilot-operational-plan.md §8.3).

All 11 steps run inside one sqlite transaction: read proposal+status,
constant-time hash check, expiry check, head-vs-base-revision check,
server-side patch reconstruction, data-fingerprint recheck, apply the patch
on a copy and validate the full ``V2Model``, insert the new revision, CAS
the head, insert the approval + outbox event, commit. A stale revision or
stale snapshot both expire the proposal rather than silently reusing it.
"""

from __future__ import annotations

import hmac
import json
import os
import sqlite3
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid4

from lazyportfolio.copilot.canonical import content_hash
from lazyportfolio.copilot.contracts import ChangeProposal, SnapshotDescriptor
from lazyportfolio.copilot.node_universe import apply_views_to_config
from lazyportfolio.copilot.patch import validate_patch
from lazyportfolio.v2 import db as _db
from lazyportfolio.v2.model import V2Model


class ApprovalError(RuntimeError):
    """Base class for every reason an approval can be refused."""


class ProposalNotFound(ApprovalError):
    pass


class ProposalNotPendingApproval(ApprovalError):
    def __init__(self, status: str) -> None:
        self.status = status
        super().__init__(f"proposal is {status!r}, not 'pending_approval'")


class ApprovalHashMismatch(ApprovalError):
    """The caller-supplied ``proposal_hash`` does not match the stored content hash."""


class ProposalExpired(ApprovalError):
    pass


class StaleRevisionError(ApprovalError):
    """The tree's head has moved since this proposal's ``base_revision_id``."""


class StaleDataError(ApprovalError):
    """The market data snapshot has changed since this proposal was drafted."""


@dataclass(frozen=True)
class ApprovalResult:
    proposal_id: UUID
    new_revision_id: str
    approval_id: str


def _trust_stored_fingerprint(snapshot: SnapshotDescriptor) -> str:
    """Fase 1 default for ``recompute_fingerprint``: no live re-check.

    Correct only for fixture-driven proposals with no real market data
    behind them. Fase 2's ``SnapshotService`` must supply a real
    implementation before this runs against live data (docs/adr/0001-node-copilot-architecture.md).
    """

    return snapshot.fingerprint


def apply_proposal(
    proposal_id: UUID,
    *,
    proposal_hash: str,
    approved_by: str,
    idempotency_key: str,
    recompute_fingerprint: Callable[[SnapshotDescriptor], str] = _trust_stored_fingerprint,
    db_path: str | os.PathLike[str] | None = None,
) -> ApprovalResult:
    """Approve and apply ``proposal_id`` in one atomic transaction (§8.3).

    A retry with the same ``idempotency_key`` after a prior success returns
    that same :class:`ApprovalResult` without creating a second revision.
    """

    with closing(_db.connect(db_path)) as conn:
        replay = _existing_result_for_idempotency_key(conn, idempotency_key)
        if replay is not None:
            return replay

        # Step 1 -- read proposal + status.
        row = conn.execute(
            "SELECT payload_json, status FROM change_proposals WHERE proposal_id = ?",
            (str(proposal_id),),
        ).fetchone()
        if row is None:
            raise ProposalNotFound(str(proposal_id))
        payload_json, status = row
        if status != "pending_approval":
            raise ProposalNotPendingApproval(status)
        proposal = ChangeProposal.model_validate_json(payload_json)

        # Step 2 -- constant-time hash comparison (never a plain `==` on a
        # value an attacker could use to time-probe the stored hash).
        if not hmac.compare_digest(proposal.content_hash, proposal_hash):
            raise ApprovalHashMismatch(proposal.content_hash)

        # Step 3 -- expiry.
        now_dt = datetime.now(UTC)
        if proposal.expires_at <= now_dt:
            _expire(conn, proposal_id)
            raise ProposalExpired(proposal.expires_at.isoformat())

        # Step 4 -- head vs base_revision_id.
        head_row = conn.execute(
            "SELECT head_revision_id FROM tree_heads WHERE tree_id = ?",
            (str(proposal.tree_id),),
        ).fetchone()
        current_head = head_row[0] if head_row else None
        if current_head != str(proposal.base_revision_id):
            _expire(conn, proposal_id)
            raise StaleRevisionError(
                f"head is {current_head!r}, proposal was based on "
                f"{str(proposal.base_revision_id)!r}"
            )

        # Step 5 -- rebuild/validate the patch server-side; a client-supplied
        # patch is never trusted, only node_id + proposed_views are read.
        validate_patch(proposal.patch, proposal.node_id)

        # Step 6 -- recompute the data fingerprint.
        current_fingerprint = recompute_fingerprint(proposal.snapshot)
        if current_fingerprint != proposal.snapshot.fingerprint:
            _expire(conn, proposal_id)
            raise StaleDataError(
                f"snapshot fingerprint is now {current_fingerprint!r}, proposal was "
                f"drafted against {proposal.snapshot.fingerprint!r}"
            )

        # Step 7 -- apply the patch on a copy, validate the full V2Model.
        head_config_row = conn.execute(
            "SELECT config_json FROM tree_revisions WHERE revision_id = ?",
            (current_head,),
        ).fetchone()
        assert head_config_row is not None  # tree_heads' FK guarantees this row exists
        base_config = json.loads(head_config_row[0])
        new_config = apply_views_to_config(base_config, proposal.node_id, proposal.proposed_views)
        V2Model.from_config(new_config)  # raises ValueError on an invalid resulting tree

        # Step 8 -- insert the new tree_revision.
        new_revision_id = str(uuid4())
        applied_at = now_dt.isoformat()
        new_config_json = json.dumps(new_config, sort_keys=True, default=str)
        conn.execute(
            "INSERT INTO tree_revisions (revision_id, tree_id, parent_revision_id, "
            "config_json, config_hash, created_at, actor_type, actor_id, reason) "
            "VALUES (?, ?, ?, ?, ?, ?, 'human', ?, ?)",
            (
                new_revision_id,
                str(proposal.tree_id),
                current_head,
                new_config_json,
                content_hash(new_config),
                applied_at,
                approved_by,
                f"approved proposal {proposal_id}",
            ),
        )

        # Step 9 -- CAS the head.
        cursor = conn.execute(
            "UPDATE tree_heads SET head_revision_id = ? WHERE tree_id = ? "
            "AND head_revision_id = ?",
            (new_revision_id, str(proposal.tree_id), current_head),
        )
        if cursor.rowcount != 1:
            conn.rollback()
            raise StaleRevisionError("head moved between step 4's read and this commit")

        # Step 10 -- approval + outbox event, same transaction.
        approval_id = str(uuid4())
        conn.execute(
            "INSERT INTO proposal_approvals (approval_id, proposal_id, approved_by, "
            "approved_at, approved_hash, idempotency_key, applied_revision_id, result_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                approval_id,
                str(proposal_id),
                approved_by,
                applied_at,
                proposal.content_hash,
                idempotency_key,
                new_revision_id,
                json.dumps({"new_revision_id": new_revision_id, "approval_id": approval_id}),
            ),
        )
        conn.execute(
            "UPDATE change_proposals SET status = 'applied' "
            "WHERE proposal_id = ? AND status = 'pending_approval'",
            (str(proposal_id),),
        )
        conn.execute(
            "INSERT INTO outbox_events (event_id, aggregate_type, aggregate_id, event_type, "
            "payload_json, created_at, delivered_at) "
            "VALUES (?, 'tree_revision', ?, 'proposal_applied', ?, ?, NULL)",
            (
                str(uuid4()),
                new_revision_id,
                json.dumps({"proposal_id": str(proposal_id), "revision_id": new_revision_id}),
                applied_at,
            ),
        )

        # Step 11 -- commit.
        conn.commit()

    return ApprovalResult(
        proposal_id=proposal_id, new_revision_id=new_revision_id, approval_id=approval_id
    )


def _existing_result_for_idempotency_key(
    conn: sqlite3.Connection, idempotency_key: str
) -> ApprovalResult | None:
    row = conn.execute(
        "SELECT proposal_id, applied_revision_id, approval_id FROM proposal_approvals "
        "WHERE idempotency_key = ?",
        (idempotency_key,),
    ).fetchone()
    if row is None:
        return None
    proposal_id, applied_revision_id, approval_id = row
    return ApprovalResult(
        proposal_id=UUID(proposal_id),
        new_revision_id=applied_revision_id,
        approval_id=approval_id,
    )


def _expire(conn: sqlite3.Connection, proposal_id: UUID) -> None:
    conn.execute(
        "UPDATE change_proposals SET status = 'expired' "
        "WHERE proposal_id = ? AND status = 'pending_approval'",
        (str(proposal_id),),
    )
    conn.commit()


__all__ = [
    "ApprovalError",
    "ApprovalHashMismatch",
    "ApprovalResult",
    "ProposalExpired",
    "ProposalNotFound",
    "ProposalNotPendingApproval",
    "StaleDataError",
    "StaleRevisionError",
    "apply_proposal",
]
