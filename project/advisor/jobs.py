"""Local job queue: claim, heartbeat, complete, reap.

docs/node-advisor-operational-plan.md §9.2/§13 Fase 3.

``agent_jobs`` (Fase 1's schema) is the durable queue -- a row survives a
server restart. Claiming a job is a compare-and-swap
(``queued -> running``), the same pattern as
:mod:`lazyportfolio.advisor.repository`'s revision CAS, so two worker
threads racing on the same job can never both claim it.

MVP scope (§13 Fase 3): one job kind, ``fixture_proposal`` -- no LLM call
anywhere in this module or its handler. A message's structured content
(node_id + views, not free text) is the "fixture" a real LLM step would
someday replace (Fase 4).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Event
from typing import Any
from uuid import uuid4

from lazyportfolio.v2 import db as _db

FIXTURE_PROPOSAL = "fixture_proposal"


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    conversation_id: str
    request_message_id: str
    kind: str
    status: str
    checkpoint_key: str
    budget: dict[str, Any]
    started_at: str | None
    heartbeat_at: str | None
    finished_at: str | None
    error: str | None


def _row_to_job(row: tuple[Any, ...]) -> JobRecord:
    (
        job_id,
        conversation_id,
        request_message_id,
        kind,
        status,
        checkpoint_key,
        budget_json,
        started_at,
        heartbeat_at,
        finished_at,
        error_json,
    ) = row
    return JobRecord(
        job_id=job_id,
        conversation_id=conversation_id,
        request_message_id=request_message_id,
        kind=kind,
        status=status,
        checkpoint_key=checkpoint_key,
        budget=json.loads(budget_json),
        started_at=started_at,
        heartbeat_at=heartbeat_at,
        finished_at=finished_at,
        error=None if error_json is None else json.loads(error_json).get("message"),
    )


_JOB_COLUMNS = (
    "job_id, conversation_id, request_message_id, kind, status, checkpoint_key, "
    "budget_json, started_at, heartbeat_at, finished_at, error_json"
)


def enqueue_job(
    conversation_id: str,
    request_message_id: str,
    kind: str,
    *,
    budget: dict[str, Any] | None = None,
    db_path: str | os.PathLike[str] | None = None,
) -> str:
    job_id = str(uuid4())
    with closing(_db.connect(db_path)) as conn:
        conn.execute(
            "INSERT INTO agent_jobs (job_id, conversation_id, request_message_id, kind, "
            "status, checkpoint_key, budget_json) VALUES (?, ?, ?, ?, 'queued', ?, ?)",
            (
                job_id,
                conversation_id,
                request_message_id,
                kind,
                f"advisor/{job_id}",
                json.dumps(budget or {}),
            ),
        )
        conn.commit()
    return job_id


def get_job(job_id: str, *, db_path: str | os.PathLike[str] | None = None) -> JobRecord | None:
    with closing(_db.connect(db_path)) as conn:
        row = conn.execute(
            f"SELECT {_JOB_COLUMNS} FROM agent_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
    return None if row is None else _row_to_job(row)


def claim_next_job(*, db_path: str | os.PathLike[str] | None = None) -> JobRecord | None:
    """Atomically claim the oldest ``queued`` job, or ``None`` if there is none.

    ``UPDATE ... WHERE status = 'queued'`` on the row selected by the
    read is the CAS: if another worker claimed it first, the read-then-write
    is repeated against the next candidate, never silently double-claiming.
    """

    now = datetime.now(UTC).isoformat()
    with closing(_db.connect(db_path)) as conn:
        candidates = conn.execute(
            "SELECT job_id FROM agent_jobs WHERE status = 'queued' ORDER BY rowid ASC"
        ).fetchall()
        for (job_id,) in candidates:
            cursor = conn.execute(
                "UPDATE agent_jobs SET status = 'running', started_at = ?, heartbeat_at = ? "
                "WHERE job_id = ? AND status = 'queued'",
                (now, now, job_id),
            )
            if cursor.rowcount == 1:
                conn.commit()
                row = conn.execute(
                    f"SELECT {_JOB_COLUMNS} FROM agent_jobs WHERE job_id = ?", (job_id,)
                ).fetchone()
                return _row_to_job(row)
            conn.rollback()
    return None


def heartbeat(job_id: str, *, db_path: str | os.PathLike[str] | None = None) -> None:
    now = datetime.now(UTC).isoformat()
    with closing(_db.connect(db_path)) as conn:
        conn.execute(
            "UPDATE agent_jobs SET heartbeat_at = ? WHERE job_id = ? AND status = 'running'",
            (now, job_id),
        )
        conn.commit()


def complete_job(
    job_id: str,
    *,
    status: str,
    error: str | None = None,
    db_path: str | os.PathLike[str] | None = None,
) -> None:
    if status not in ("succeeded", "failed"):
        raise ValueError(f"complete_job status must be 'succeeded' or 'failed', got {status!r}")
    now = datetime.now(UTC).isoformat()
    error_json = json.dumps({"message": error}) if error is not None else None
    with closing(_db.connect(db_path)) as conn:
        conn.execute(
            "UPDATE agent_jobs SET status = ?, finished_at = ?, error_json = ? "
            "WHERE job_id = ? AND status = 'running'",
            (status, now, error_json, job_id),
        )
        conn.commit()


def reap_orphaned_jobs(
    *,
    heartbeat_timeout_seconds: float,
    db_path: str | os.PathLike[str] | None = None,
) -> list[str]:
    """Reset ``running`` jobs whose heartbeat is older than the timeout back
    to ``queued``, so a crashed worker's job is picked up again instead of
    stuck forever (§13 Fase 3: "restart durante ogni stato critico
    recuperabile"). Returns the reaped job ids."""

    cutoff = datetime.now(UTC).timestamp() - heartbeat_timeout_seconds
    reaped: list[str] = []
    with closing(_db.connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT job_id, heartbeat_at FROM agent_jobs WHERE status = 'running'"
        ).fetchall()
        for job_id, heartbeat_at in rows:
            if heartbeat_at is None:
                continue
            if datetime.fromisoformat(heartbeat_at).timestamp() >= cutoff:
                continue
            cursor = conn.execute(
                "UPDATE agent_jobs SET status = 'queued', started_at = NULL, "
                "heartbeat_at = NULL WHERE job_id = ? AND status = 'running'",
                (job_id,),
            )
            if cursor.rowcount == 1:
                reaped.append(job_id)
        conn.commit()
    return reaped


JobHandler = Callable[[JobRecord], None]


def run_worker_once(
    *,
    handlers: dict[str, JobHandler],
    db_path: str | os.PathLike[str] | None = None,
) -> bool:
    """Claim and run exactly one queued job, if any. Returns whether a job ran."""

    job = claim_next_job(db_path=db_path)
    if job is None:
        return False
    heartbeat(job.job_id, db_path=db_path)
    handler = handlers.get(job.kind)
    if handler is None:
        complete_job(
            job.job_id,
            status="failed",
            error=f"no handler for kind {job.kind!r}",
            db_path=db_path,
        )
        return True
    try:
        handler(job)
    except Exception as exc:  # the job failed; never crash the worker loop over it
        complete_job(job.job_id, status="failed", error=str(exc), db_path=db_path)
        return True
    complete_job(job.job_id, status="succeeded", db_path=db_path)
    return True


def run_worker_loop(
    *,
    handlers: dict[str, JobHandler],
    stop_event: Event,
    db_path: str | os.PathLike[str] | None = None,
    poll_interval_seconds: float = 0.2,
) -> None:
    """Run ``run_worker_once`` until ``stop_event`` is set -- the whole
    point of no work being done in the HTTP request thread (§11)."""

    while not stop_event.is_set():
        ran = run_worker_once(handlers=handlers, db_path=db_path)
        if not ran:
            stop_event.wait(poll_interval_seconds)


__all__ = [
    "FIXTURE_PROPOSAL",
    "JobHandler",
    "JobRecord",
    "claim_next_job",
    "complete_job",
    "enqueue_job",
    "get_job",
    "heartbeat",
    "reap_orphaned_jobs",
    "run_worker_loop",
    "run_worker_once",
]
