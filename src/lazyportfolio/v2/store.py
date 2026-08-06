"""Shared, SQLite-backed persistence for named V2 tree configurations.

Both Tree Studio (the local visual editor, ``project/tree_studio.py``) and any
external caller (LazyTools' MCP ``portfolio_tree_*`` tools) read and write
through this module, never through their own copy of the logic -- so a tree
saved by one is immediately visible to the other: same database (shared with
:mod:`lazyportfolio.v2.run_history` via :mod:`lazyportfolio.v2.db`), same name
sanitization, same validate-before-write gate. Stdlib-only.

Trees used to be one JSON file per name under ``reports/tree_studio/models/``.
That directory of loose files is gone -- a tree is now one row in the shared
``trees`` table, keyed by its sanitized name -- but ``sanitize_model_name``'s
character policy is unchanged, so an existing name still normalizes exactly
as it always has.
"""

from __future__ import annotations

import json
import os
import re
from contextlib import closing
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from lazyportfolio.v2 import db as _db
from lazyportfolio.v2.model import V2Model

#: Same character policy Tree Studio has always used for a tree's stored
#: name: collapse anything else to a hyphen, then trim stray separators.
_MODEL_NAME = re.compile(r"[^A-Za-z0-9._ -]+")


class ModelStoreError(ValueError):
    """A model name or configuration cannot be persisted or found."""


def _as_json(value: Any) -> Any:
    """Best-effort JSON coercion, same fallback Tree Studio's own responses use."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        return _as_json(asdict(value))
    if hasattr(value, "to_dict"):
        return {str(k): _as_json(v) for k, v in value.to_dict().items()}
    if isinstance(value, dict):
        return {str(k): _as_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_json(v) for v in value]
    return value


def resolve_store_path(store_path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the shared database file saved tree configurations live in.

    Precedence: an explicit ``store_path`` argument, then the
    ``LAZYPORTFOLIO_TREE_DB`` env var, then the default -- a sibling of the
    run-history database, ``<repo>/reports/tree_studio/tree_studio.sqlite3``.
    """
    return _db.resolve_db_path(store_path)


def sanitize_model_name(name: Any) -> str:
    """Reduce a model name to a safe, stable stored key."""
    cleaned = _MODEL_NAME.sub("-", str(name).strip()).strip(" .-")
    if not cleaned:
        raise ModelStoreError("model name cannot be blank")
    return cleaned[:120]


def list_saved_models(*, store_path: str | os.PathLike[str] | None = None) -> list[dict[str, str]]:
    """List saved models as ``{"name", "updated_at"}`` pairs, newest first."""
    with closing(_db.connect(store_path)) as conn:
        rows = conn.execute("SELECT name, updated_at FROM trees ORDER BY updated_at DESC").fetchall()
    return [{"name": name, "updated_at": updated_at} for name, updated_at in rows]


def read_model(name: Any, *, store_path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Read a saved model's raw configuration by name (no re-validation)."""
    key = sanitize_model_name(name)
    with closing(_db.connect(store_path)) as conn:
        row = conn.execute("SELECT config FROM trees WHERE name = ?", (key,)).fetchone()
    if row is None:
        raise FileNotFoundError(f"no saved model named {key!r}")
    loaded: dict[str, Any] = json.loads(row[0])
    return loaded


def write_model(
    name: Any,
    config: dict[str, Any],
    *,
    store_path: str | os.PathLike[str] | None = None,
) -> str:
    """Validate ``config`` and persist it; never writes on a validation failure.

    Validation is the same gate Tree Studio's own save endpoint has always
    used: constructing ``V2Model.from_config(config)`` and discarding the
    result (this call is for the side-effecting validation, not the model).

    Returns the sanitized name the tree was stored under.
    """
    if not isinstance(config, dict):
        raise ModelStoreError("model config must be an object")
    V2Model.from_config(config)
    key = sanitize_model_name(name)
    now = datetime.now(timezone.utc).isoformat()
    payload = json.dumps(config, default=_as_json)
    with closing(_db.connect(store_path)) as conn:
        conn.execute(
            "INSERT INTO trees (name, config, created_at, updated_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET config = excluded.config, updated_at = excluded.updated_at",
            (key, payload, now, now),
        )
        conn.commit()
    return key


def delete_model(name: Any, *, store_path: str | os.PathLike[str] | None = None) -> str:
    """Delete a saved model by name, returning the sanitized name that was removed."""
    key = sanitize_model_name(name)
    with closing(_db.connect(store_path)) as conn:
        cursor = conn.execute("DELETE FROM trees WHERE name = ?", (key,))
        conn.commit()
        deleted = cursor.rowcount
    if not deleted:
        raise FileNotFoundError(f"no saved model named {key!r}")
    return key


def migrate_legacy_json_models(
    models_dir: str | os.PathLike[str],
    *,
    store_path: str | os.PathLike[str] | None = None,
) -> list[str]:
    """One-time import of the pre-SQLite ``*.json`` tree files into the shared store.

    Safe to call more than once: an already-migrated name is simply
    overwritten with the same content. Source files are left untouched --
    this only ever adds rows, never deletes the originals. Returns the list
    of names imported.
    """
    directory = Path(models_dir)
    if not directory.is_dir():
        return []
    imported: list[str] = []
    for path in sorted(directory.glob("*.json")):
        config = json.loads(path.read_text(encoding="utf-8"))
        name = write_model(path.stem, config, store_path=store_path)
        imported.append(name)
    return imported


__all__ = [
    "ModelStoreError",
    "delete_model",
    "list_saved_models",
    "migrate_legacy_json_models",
    "read_model",
    "resolve_store_path",
    "sanitize_model_name",
    "write_model",
]
