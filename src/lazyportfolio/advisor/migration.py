"""One-time, idempotent migration of legacy ``trees`` rows into revisions.

docs/node-advisor-schema-migration-draft.md step 2: every legacy row (keyed
by name, mutable in place) becomes a ``tree_id`` + one initial
``tree_revisions`` row + a ``legacy_tree_names`` mapping row. The legacy
``trees`` table itself is never modified or deleted from -- it remains the
name index :mod:`lazyportfolio.v2.store`'s existing callers read.
"""

from __future__ import annotations

import json
import os
from contextlib import closing
from datetime import UTC, datetime
from uuid import uuid4

from lazyportfolio.advisor.canonical import content_hash
from lazyportfolio.v2 import db as _db


def migrate_legacy_trees(
    *,
    actor_id: str = "migration",
    db_path: str | os.PathLike[str] | None = None,
) -> list[str]:
    """Migrate every not-yet-migrated row of the legacy ``trees`` table.

    Safe to call more than once: a ``name`` already present in
    ``legacy_tree_names`` is skipped, not re-migrated (mirrors
    :func:`lazyportfolio.v2.store.migrate_legacy_json_models`'s own
    idempotency contract). Returns the names migrated by *this* call --
    an empty list on a second call means everything was already migrated,
    not that nothing exists.
    """

    migrated: list[str] = []
    with closing(_db.connect(db_path)) as conn:
        already_migrated = {
            row[0] for row in conn.execute("SELECT name FROM legacy_tree_names").fetchall()
        }
        legacy_rows = conn.execute(
            "SELECT name, config, created_at FROM trees ORDER BY name"
        ).fetchall()
        now = datetime.now(UTC).isoformat()
        for name, config_json, legacy_created_at in legacy_rows:
            if name in already_migrated:
                continue
            config = json.loads(config_json)
            tree_id = str(uuid4())
            revision_id = str(uuid4())
            conn.execute(
                "INSERT INTO tree_revisions (revision_id, tree_id, parent_revision_id, "
                "config_json, config_hash, created_at, actor_type, actor_id, reason) "
                "VALUES (?, ?, NULL, ?, ?, ?, 'system', ?, ?)",
                (
                    revision_id,
                    tree_id,
                    config_json,
                    content_hash(config),
                    legacy_created_at or now,
                    actor_id,
                    "migrated from legacy trees table",
                ),
            )
            conn.execute(
                "INSERT INTO tree_heads (tree_id, head_revision_id) VALUES (?, ?)",
                (tree_id, revision_id),
            )
            conn.execute(
                "INSERT INTO legacy_tree_names (tree_id, name) VALUES (?, ?)",
                (tree_id, name),
            )
            migrated.append(name)
        conn.commit()
    return migrated


def tree_id_for_name(
    name: str, *, db_path: str | os.PathLike[str] | None = None
) -> str | None:
    """The migrated ``tree_id`` for a legacy-named tree, or ``None`` if not yet migrated."""

    with closing(_db.connect(db_path)) as conn:
        row = conn.execute(
            "SELECT tree_id FROM legacy_tree_names WHERE name = ?", (name,)
        ).fetchone()
    return None if row is None else str(row[0])


__all__ = ["migrate_legacy_trees", "tree_id_for_name"]
