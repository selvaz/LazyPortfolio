"""Shared SQLite persistence for Tree Studio: one database, three tables.

Saved tree configurations (:mod:`lazyportfolio.v2.store`) and run history/
artifacts (:mod:`lazyportfolio.v2.run_history`) used to live in different
places -- one directory of loose JSON files per tree, one opaque cache blob
table keyed by an unstructured hash. Both are single-user, personal data that
needs to survive a process restart, so they share one on-disk database and
one connection helper instead of three copies of the "resolve path / create
parent dir / open sqlite3 connection" boilerplate. Stdlib-only.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

#: Environment variable overriding where the shared database file lives.
_ENV_VAR = "LAZYPORTFOLIO_TREE_DB"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trees (
    name TEXT PRIMARY KEY,
    config TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cache_key TEXT NOT NULL UNIQUE,
    path TEXT NOT NULL,
    kind TEXT NOT NULL,
    tree_id TEXT,
    config_hash TEXT NOT NULL,
    data_as_of TEXT,
    data_fingerprint TEXT NOT NULL,
    created_at TEXT NOT NULL,
    weights TEXT,
    metrics TEXT,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_tree_id ON runs(tree_id);
CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at);

CREATE TABLE IF NOT EXISTS run_artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    kind TEXT NOT NULL,
    content_type TEXT NOT NULL,
    filename TEXT NOT NULL,
    blob BLOB,
    external_artifact_id TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_run_artifacts_run_id ON run_artifacts(run_id);
"""


def resolve_db_path(db_path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the one shared sqlite file Tree Studio's persistence lives in.

    Precedence: an explicit ``db_path`` argument, then the
    ``LAZYPORTFOLIO_TREE_DB`` env var, then the default --
    ``<repo>/reports/tree_studio/tree_studio.sqlite3``.
    """
    if db_path:
        return Path(db_path).resolve()
    env = os.environ.get(_ENV_VAR)
    if env:
        return Path(env).resolve()
    # .../src/lazyportfolio/v2/db.py -> v2 -> lazyportfolio -> src -> repo root
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "reports" / "tree_studio" / "tree_studio.sqlite3"


def connect(db_path: str | os.PathLike[str] | None = None) -> sqlite3.Connection:
    """Open a connection to the shared database, creating the schema if needed."""
    path = resolve_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.executescript(_SCHEMA)
    return conn


__all__ = ["connect", "resolve_db_path"]
