"""Append-only tree revisions (docs/node-copilot-operational-plan.md §5.1/§13 Fase 1).

Every save is a new ``tree_revisions`` row; the current head is a
compare-and-swap on ``tree_heads.head_revision_id``. The pattern mirrors
LazyFin's already-shipped ``StorePortfolioLedger.compare_and_swap``
(``LazyFin/src/lazyfin/kernel/persist.py``), reimplemented here rather than
imported since LazyPortfolio does not depend on LazyFin
(docs/adr/0001-node-copilot-architecture.md Decision 1).

Function-based, one connection per call, mirroring
:mod:`lazyportfolio.v2.store` and :mod:`lazyportfolio.v2.run_history`'s
existing convention rather than introducing a class-based repository --
this package has no precedent for the latter and one module more of it
would be inconsistent with its two siblings.
"""

from __future__ import annotations

import json
import os
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from lazyportfolio.copilot.canonical import content_hash
from lazyportfolio.v2 import db as _db


class ConcurrentTreeWrite(RuntimeError):
    """A save_revision's compare-and-swap lost a race: the head moved since it was read."""


@dataclass(frozen=True)
class TreeRevision:
    revision_id: str
    tree_id: str
    parent_revision_id: str | None
    config: dict[str, Any]
    config_hash: str
    created_at: str
    actor_type: str
    actor_id: str
    reason: str | None


def create_tree(
    config: dict[str, Any],
    *,
    actor_type: str,
    actor_id: str,
    reason: str | None = None,
    db_path: str | os.PathLike[str] | None = None,
) -> TreeRevision:
    """Create a brand-new tree with a single initial revision (no parent)."""

    tree_id = str(uuid4())
    revision_id = str(uuid4())
    now, config_hash, config_json = _prepare_revision(config)
    with closing(_db.connect(db_path)) as conn:
        conn.execute(
            "INSERT INTO tree_revisions (revision_id, tree_id, parent_revision_id, "
            "config_json, config_hash, created_at, actor_type, actor_id, reason) "
            "VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?)",
            (revision_id, tree_id, config_json, config_hash, now, actor_type, actor_id, reason),
        )
        cursor = conn.execute(
            "INSERT INTO tree_heads (tree_id, head_revision_id) "
            "SELECT ?, ? WHERE NOT EXISTS (SELECT 1 FROM tree_heads WHERE tree_id = ?)",
            (tree_id, revision_id, tree_id),
        )
        if cursor.rowcount != 1:
            conn.rollback()
            raise ConcurrentTreeWrite(f"tree {tree_id!r} already has a head")
        conn.commit()
    return TreeRevision(
        revision_id=revision_id,
        tree_id=tree_id,
        parent_revision_id=None,
        config=config,
        config_hash=config_hash,
        created_at=now,
        actor_type=actor_type,
        actor_id=actor_id,
        reason=reason,
    )


def save_revision(
    tree_id: str,
    config: dict[str, Any],
    *,
    actor_type: str,
    actor_id: str,
    reason: str | None = None,
    expected_head: str | None = None,
    db_path: str | os.PathLike[str] | None = None,
) -> TreeRevision:
    """Append a new revision onto an existing tree's history.

    The write is always compare-and-swapped against the head this call
    observed (``expected_head`` lets a caller pin an *earlier* head it
    already knew about, e.g. when re-validating a client-supplied base
    revision before writing).
    """

    current = get_head(tree_id, db_path=db_path)
    if current is None:
        raise ConcurrentTreeWrite(f"tree {tree_id!r} has no existing head to save onto")
    guard_head = expected_head if expected_head is not None else current.revision_id

    revision_id = str(uuid4())
    now, config_hash, config_json = _prepare_revision(config)
    with closing(_db.connect(db_path)) as conn:
        conn.execute(
            "INSERT INTO tree_revisions (revision_id, tree_id, parent_revision_id, "
            "config_json, config_hash, created_at, actor_type, actor_id, reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                revision_id,
                tree_id,
                current.revision_id,
                config_json,
                config_hash,
                now,
                actor_type,
                actor_id,
                reason,
            ),
        )
        cursor = conn.execute(
            "UPDATE tree_heads SET head_revision_id = ? WHERE tree_id = ? AND head_revision_id = ?",
            (revision_id, tree_id, guard_head),
        )
        if cursor.rowcount != 1:
            conn.rollback()
            raise ConcurrentTreeWrite(
                f"tree {tree_id!r}'s head moved before this save committed "
                f"(expected {guard_head!r})"
            )
        conn.commit()
    return TreeRevision(
        revision_id=revision_id,
        tree_id=tree_id,
        parent_revision_id=current.revision_id,
        config=config,
        config_hash=config_hash,
        created_at=now,
        actor_type=actor_type,
        actor_id=actor_id,
        reason=reason,
    )


def _prepare_revision(config: dict[str, Any]) -> tuple[str, str, str]:
    """``(created_at, config_hash, config_json)`` shared by both insert paths."""

    now = datetime.now(UTC).isoformat()
    config_hash = content_hash(config)
    config_json = json.dumps(config, sort_keys=True, default=str)
    return now, config_hash, config_json


def get_head(
    tree_id: str, *, db_path: str | os.PathLike[str] | None = None
) -> TreeRevision | None:
    """The current head revision of ``tree_id``, or ``None`` if it has none."""

    with closing(_db.connect(db_path)) as conn:
        row = conn.execute(
            "SELECT r.revision_id, r.parent_revision_id, r.config_json, r.config_hash, "
            "r.created_at, r.actor_type, r.actor_id, r.reason "
            "FROM tree_heads h JOIN tree_revisions r ON r.revision_id = h.head_revision_id "
            "WHERE h.tree_id = ?",
            (tree_id,),
        ).fetchone()
    if row is None:
        return None
    return TreeRevision(
        revision_id=row[0],
        tree_id=tree_id,
        parent_revision_id=row[1],
        config=json.loads(row[2]),
        config_hash=row[3],
        created_at=row[4],
        actor_type=row[5],
        actor_id=row[6],
        reason=row[7],
    )


__all__ = ["ConcurrentTreeWrite", "TreeRevision", "create_tree", "get_head", "save_revision"]
