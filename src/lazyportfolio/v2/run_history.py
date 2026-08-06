"""Structured, queryable history of Tree Studio runs (SQLite-backed).

Sibling to :mod:`lazyportfolio.v2.store` (saved tree configs): same shared
database (:mod:`lazyportfolio.v2.db`), same stdlib-only policy. This replaces
the old ``run_cache`` module's opaque ``key -> JSON blob`` table with one row
per run carrying real columns -- ``tree_id``, ``created_at``, ``data_as_of``,
``config_hash``, ``weights``, ``metrics`` -- so a run can be audited or
listed later, not just replayed back to the same request that produced it.

A row is looked up by ``cache_key`` for the fast-path "have we already
computed this" check (same role the old ``get_run_result``/``put_run_result``
played), and separately by ``tree_id`` for a saved tree's execution history.
``cache_key`` MUST already fold in a data-freshness signal (see
``project/tree_studio.py``'s ``_data_fingerprint``) -- this module stores
whatever key it's given, it doesn't compute or validate one, so a stale key
here is a caller bug, not a storage bug.

The audit ZIP bundle is deliberately never persisted as a ``run_artifacts``
blob (multi-MB per entry, often tens of MB across a session) -- a repeat
request for it re-runs the full computation, which is fine since the
underlying estimate/backtest is the expensive part regardless of which
artifact is requested first. Only the (small) HTML report is persisted.
"""

from __future__ import annotations

import json
import os
from contextlib import closing
from datetime import UTC, datetime
from typing import Any

from lazyportfolio.v2 import db as _db


def _dumps(value: Any) -> str | None:
    return None if value is None else json.dumps(value, default=str)


def _loads(value: str | None) -> Any:
    return None if value is None else json.loads(value)


def record_run(
    *,
    cache_key: str,
    path: str,
    kind: str,
    tree_id: str | None,
    config_hash: str,
    data_as_of: str | None,
    data_fingerprint: str,
    weights: dict[str, Any] | list[Any] | None,
    metrics: dict[str, Any] | None,
    payload: dict[str, Any],
    db_path: str | os.PathLike[str] | None = None,
) -> int:
    """Persist one run, replacing any prior row under the same ``cache_key``.

    Returns the run's row id, needed by :func:`attach_artifact` to link a
    report/audit blob back to the run that produced it.
    """
    now = datetime.now(UTC).isoformat()
    with closing(_db.connect(db_path)) as conn:
        conn.execute(
            "INSERT INTO runs (cache_key, path, kind, tree_id, config_hash, data_as_of, "
            "data_fingerprint, created_at, weights, metrics, payload) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(cache_key) DO UPDATE SET "
            "path = excluded.path, kind = excluded.kind, tree_id = excluded.tree_id, "
            "data_as_of = excluded.data_as_of, created_at = excluded.created_at, "
            "weights = excluded.weights, metrics = excluded.metrics, payload = excluded.payload",
            (
                cache_key,
                path,
                kind,
                tree_id,
                config_hash,
                data_as_of,
                data_fingerprint,
                now,
                _dumps(weights),
                _dumps(metrics),
                json.dumps(payload, default=str),
            ),
        )
        conn.commit()
        row = conn.execute("SELECT id FROM runs WHERE cache_key = ?", (cache_key,)).fetchone()
    return int(row[0])


def get_by_cache_key(
    cache_key: str, *, db_path: str | os.PathLike[str] | None = None
) -> dict[str, Any] | None:
    """Look up a run's full JSON payload by cache key -- the cache-hit fast path."""
    with closing(_db.connect(db_path)) as conn:
        row = conn.execute(
            "SELECT id, payload FROM runs WHERE cache_key = ?", (cache_key,)
        ).fetchone()
    return None if row is None else {"id": row[0], "payload": json.loads(row[1])}


def list_runs(
    *,
    tree_id: str | None = None,
    limit: int = 50,
    db_path: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    """List runs newest-first, without their (potentially large) payload."""
    columns = [
        "id", "cache_key", "path", "kind", "tree_id", "config_hash",
        "data_as_of", "created_at", "weights", "metrics",
    ]
    query = f"SELECT {', '.join(columns)} FROM runs"
    params: list[Any] = []
    if tree_id is not None:
        query += " WHERE tree_id = ?"
        params.append(tree_id)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    with closing(_db.connect(db_path)) as conn:
        rows = conn.execute(query, params).fetchall()
    results = []
    for row in rows:
        item = dict(zip(columns, row, strict=True))
        item["weights"] = _loads(item["weights"])
        item["metrics"] = _loads(item["metrics"])
        results.append(item)
    return results


def get_run(run_id: int, *, db_path: str | os.PathLike[str] | None = None) -> dict[str, Any] | None:
    """One run's full record (including payload) plus its attached artifacts."""
    columns = [
        "id", "cache_key", "path", "kind", "tree_id", "config_hash", "data_as_of",
        "data_fingerprint", "created_at", "weights", "metrics", "payload",
    ]
    with closing(_db.connect(db_path)) as conn:
        row = conn.execute(
            f"SELECT {', '.join(columns)} FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        artifact_rows = conn.execute(
            "SELECT id, kind, content_type, filename, external_artifact_id, created_at "
            "FROM run_artifacts WHERE run_id = ? ORDER BY id",
            (run_id,),
        ).fetchall()
    item = dict(zip(columns, row, strict=True))
    item["weights"] = _loads(item["weights"])
    item["metrics"] = _loads(item["metrics"])
    item["payload"] = json.loads(item["payload"])
    item["artifacts"] = [
        {
            "id": artifact_id,
            "kind": kind,
            "content_type": content_type,
            "filename": filename,
            "external_artifact_id": external_artifact_id,
            "created_at": created_at,
        }
        for artifact_id, kind, content_type, filename, external_artifact_id, created_at
        in artifact_rows
    ]
    return item


def attach_artifact(
    run_id: int,
    *,
    kind: str,
    content_type: str,
    filename: str,
    blob: bytes,
    external_artifact_id: str | None = None,
    db_path: str | os.PathLike[str] | None = None,
) -> int:
    """Attach a binary/text artifact (e.g. the HTML report) to an existing run."""
    now = datetime.now(UTC).isoformat()
    with closing(_db.connect(db_path)) as conn:
        cursor = conn.execute(
            "INSERT INTO run_artifacts (run_id, kind, content_type, filename, blob, "
            "external_artifact_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, kind, content_type, filename, blob, external_artifact_id, now),
        )
        conn.commit()
        assert cursor.lastrowid is not None
        return cursor.lastrowid


def get_report_artifact(
    cache_key: str, *, db_path: str | os.PathLike[str] | None = None
) -> tuple[bytes, str, str] | None:
    """The most recent 'report' artifact for a run, by that run's cache key."""
    with closing(_db.connect(db_path)) as conn:
        row = conn.execute(
            "SELECT ra.blob, ra.content_type, ra.filename FROM run_artifacts ra "
            "JOIN runs r ON r.id = ra.run_id "
            "WHERE r.cache_key = ? AND ra.kind = 'report' "
            "ORDER BY ra.id DESC LIMIT 1",
            (cache_key,),
        ).fetchone()
    return None if row is None else (row[0], row[1], row[2])


__all__ = [
    "attach_artifact",
    "get_by_cache_key",
    "get_report_artifact",
    "get_run",
    "list_runs",
    "record_run",
]
