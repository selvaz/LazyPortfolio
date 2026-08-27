"""Proposal persistence and CAS-guarded state transitions (§5.1/§13 Fase 1).

``status`` is stored as its own column, never inside the immutable
``ChangeProposal`` payload (§4.3: "status non fa parte del payload
immutabile"). A transition is a single ``UPDATE ... WHERE status = ?`` --
the same compare-and-swap shape as
:func:`lazyportfolio.advisor.repository.save_revision`, and for the same
reason: two concurrent transitions racing on the same proposal must not
both succeed.
"""

from __future__ import annotations

import os
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from lazyportfolio.advisor.contracts import ChangeProposal, ProposalStatus
from lazyportfolio.advisor.state_machine import validate_transition
from lazyportfolio.v2 import db as _db


class ConcurrentProposalWrite(RuntimeError):
    """A transition's compare-and-swap lost a race: the status moved since it was read."""


@dataclass(frozen=True)
class ProposalRecord:
    proposal: ChangeProposal
    status: ProposalStatus
    created_at: str


class StaleBaseRevisionOnCreate(RuntimeError):
    """The tree head moved away from ``require_head_revision_id`` before the insert."""


def create(
    proposal: ChangeProposal,
    *,
    status: ProposalStatus = "drafting",
    require_head_revision_id: str | None = None,
    db_path: str | os.PathLike[str] | None = None,
) -> ProposalRecord:
    """Insert a new, immutable proposal at ``status`` (``"drafting"`` by default).

    ``require_head_revision_id`` makes the insert conditional on the tree's
    head still being that revision, evaluated by the database in the same
    statement. A caller that checks the head itself and then inserts leaves a
    window in which another editor advances the head -- the proposal is then
    persisted as ``pending_approval`` against a revision approval will
    immediately reject as stale, so the UI offers a card that can never be
    applied. Same conditional-write shape as
    :func:`lazyportfolio.advisor.repository.create_tree`'s head insert.
    """

    now = datetime.now(UTC).isoformat()
    payload_json = proposal.model_dump_json()
    columns = (
        "proposal_id, batch_id, supersedes_proposal_id, tree_id, base_revision_id, "
        "node_id, kind, producer_kind, producer_id, payload_json, content_hash, "
        "status, expires_at, created_at"
    )
    values = (
        str(proposal.id),
        str(proposal.batch_id) if proposal.batch_id is not None else None,
        str(proposal.supersedes_proposal_id)
        if proposal.supersedes_proposal_id is not None
        else None,
        str(proposal.tree_id),
        str(proposal.base_revision_id),
        proposal.node_id,
        proposal.kind,
        proposal.model_provenance.producer_kind,
        proposal.model_provenance.producer_id,
        payload_json,
        proposal.content_hash,
        status,
        proposal.expires_at.isoformat(),
        now,
    )
    with closing(_db.connect(db_path)) as conn:
        if require_head_revision_id is None:
            conn.execute(
                f"INSERT INTO change_proposals ({columns}) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                values,
            )
        else:
            cursor = conn.execute(
                f"INSERT INTO change_proposals ({columns}) "
                "SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ? "
                "WHERE EXISTS (SELECT 1 FROM tree_heads "
                "WHERE tree_id = ? AND head_revision_id = ?)",
                (*values, str(proposal.tree_id), require_head_revision_id),
            )
            if cursor.rowcount != 1:
                conn.rollback()
                raise StaleBaseRevisionOnCreate(
                    f"tree {proposal.tree_id} is no longer at revision "
                    f"{require_head_revision_id}; the proposal was not created"
                )
        conn.commit()
    return ProposalRecord(proposal=proposal, status=status, created_at=now)


def get(
    proposal_id: UUID, *, db_path: str | os.PathLike[str] | None = None
) -> ProposalRecord | None:
    with closing(_db.connect(db_path)) as conn:
        row = conn.execute(
            "SELECT payload_json, status, created_at FROM change_proposals WHERE proposal_id = ?",
            (str(proposal_id),),
        ).fetchone()
    if row is None:
        return None
    return _record_from_row(row)


def list_by_tree(
    tree_id: UUID, *, db_path: str | os.PathLike[str] | None = None
) -> list[ProposalRecord]:
    with closing(_db.connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT payload_json, status, created_at FROM change_proposals "
            "WHERE tree_id = ? ORDER BY created_at DESC",
            (str(tree_id),),
        ).fetchall()
    return [_record_from_row(row) for row in rows]


def transition(
    proposal_id: UUID,
    from_status: ProposalStatus,
    to_status: ProposalStatus,
    *,
    db_path: str | os.PathLike[str] | None = None,
) -> None:
    """Move a proposal from ``from_status`` to ``to_status``, or raise.

    Raises :class:`~lazyportfolio.advisor.state_machine.IllegalProposalTransition`
    if the transition is not in §4.5's diagram (checked before touching the
    database), or :class:`ConcurrentProposalWrite` if the proposal's status
    had already moved away from ``from_status`` by the time this call's
    ``UPDATE`` ran.
    """

    validate_transition(from_status, to_status)
    with closing(_db.connect(db_path)) as conn:
        cursor = conn.execute(
            "UPDATE change_proposals SET status = ? WHERE proposal_id = ? AND status = ?",
            (to_status, str(proposal_id), from_status),
        )
        if cursor.rowcount != 1:
            conn.rollback()
            raise ConcurrentProposalWrite(
                f"proposal {proposal_id!r} status moved before this transition committed "
                f"(expected {from_status!r})"
            )
        conn.commit()


def _record_from_row(row: tuple[str, str, str]) -> ProposalRecord:
    payload_json, status, created_at = row
    return ProposalRecord(
        proposal=ChangeProposal.model_validate_json(payload_json),
        status=status,  # type: ignore[arg-type]
        created_at=created_at,
    )


__all__ = [
    "ConcurrentProposalWrite",
    "ProposalRecord",
    "StaleBaseRevisionOnCreate",
    "create",
    "get",
    "list_by_tree",
    "transition",
]
