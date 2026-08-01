"""Persistent, file-based cache for V2 run results and the client report.

Sibling to :mod:`lazyportfolio.v2.store` (saved tree configs): same directory
convention, same stdlib-only policy (``sqlite3``). Tree Studio's in-memory
``_run_cache``/``_artifact_cache`` are wiped on every process restart -- this
module lets the two cheap-to-store kinds survive one: numeric run results
(estimate/backtest/scientific-study JSON) and the client HTML report. The
audit ZIP bundle is deliberately NOT persisted here (multi-MB per entry,
often tens of MB across a session) -- a repeat request for it re-runs the
full computation, which is fine since the underlying estimate/backtest is
the expensive part regardless of which artifact is requested first.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

#: Environment variable overriding where the cache file lives, same pattern
#: as store.py's LAZYPORTFOLIO_TREE_MODELS_DIR.
_ENV_VAR = "LAZYPORTFOLIO_TREE_CACHE_DB"


def resolve_cache_path(cache_path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the SQLite file this cache reads/writes.

    Precedence: an explicit ``cache_path`` argument, then the
    ``LAZYPORTFOLIO_TREE_CACHE_DB`` env var, then the default -- a sibling of
    the saved-tree-models directory, ``<repo>/reports/tree_studio/run_cache.sqlite3``.
    """
    if cache_path:
        return Path(cache_path).resolve()
    env = os.environ.get(_ENV_VAR)
    if env:
        return Path(env).resolve()
    # .../src/lazyportfolio/v2/run_cache.py -> v2 -> lazyportfolio -> src -> repo root
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "reports" / "tree_studio" / "run_cache.sqlite3"


def _connect(cache_path: str | os.PathLike[str] | None = None) -> sqlite3.Connection:
    path = resolve_cache_path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS cache ("
        "key TEXT PRIMARY KEY, kind TEXT NOT NULL, payload TEXT, "
        "blob BLOB, content_type TEXT, filename TEXT, created_at TEXT NOT NULL)"
    )
    return conn


def get_run_result(key: str, *, cache_path: str | os.PathLike[str] | None = None) -> dict[str, Any] | None:
    """Look up a cached JSON run result (estimate/backtest/scientific-study) by key."""
    with closing(_connect(cache_path)) as conn:
        row = conn.execute("SELECT payload FROM cache WHERE key = ? AND kind = 'run'", (key,)).fetchone()
    return json.loads(row[0]) if row else None


def put_run_result(
    key: str, payload: dict[str, Any], *, cache_path: str | os.PathLike[str] | None = None
) -> None:
    """Persist a JSON run result under ``key``, replacing any prior entry."""
    with closing(_connect(cache_path)) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO cache (key, kind, payload, created_at) "
            "VALUES (?, 'run', ?, datetime('now'))",
            (key, json.dumps(payload, default=str)),
        )
        conn.commit()


def get_report(
    key: str, *, cache_path: str | os.PathLike[str] | None = None
) -> tuple[bytes, str, str] | None:
    """Look up a cached client HTML report by key -> ``(body, content_type, filename)``."""
    with closing(_connect(cache_path)) as conn:
        row = conn.execute(
            "SELECT blob, content_type, filename FROM cache WHERE key = ? AND kind = 'report'", (key,)
        ).fetchone()
    return (row[0], row[1], row[2]) if row else None


def put_report(
    key: str,
    body: bytes,
    content_type: str,
    filename: str,
    *,
    cache_path: str | os.PathLike[str] | None = None,
) -> None:
    """Persist a client HTML report under ``key``, replacing any prior entry."""
    with closing(_connect(cache_path)) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO cache (key, kind, blob, content_type, filename, created_at) "
            "VALUES (?, 'report', ?, ?, ?, datetime('now'))",
            (key, body, content_type, filename),
        )
        conn.commit()


__all__ = [
    "get_report",
    "get_run_result",
    "put_report",
    "put_run_result",
    "resolve_cache_path",
]
