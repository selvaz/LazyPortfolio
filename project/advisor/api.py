"""HTTP-agnostic dispatch for the Node Advisor's REST surface.

docs/node-advisor-operational-plan.md §9.1/§13 Fase 3. Pure functions:
parsed path/query/body in, ``(status_code, payload)`` out -- no dependency
on ``http.server``, so this is unit-testable without a real server, and
``project/tree_studio.py``'s ``StudioHandler`` only has to translate an
HTTP request into these plain arguments and write the JSON response.

The SSE endpoint (``GET .../jobs/{id}/events``) is not here: streaming a
response is an HTTP-layer concern this module deliberately has no opinion
on -- ``StudioHandler`` implements it directly against ``jobs.get_job``.
"""

from __future__ import annotations

import re
from typing import Any
from uuid import UUID

from advisor import jobs, services
from lazyportfolio.advisor.node_universe import NodeNotFoundError

_NODE_CONTEXT = re.compile(
    r"^/api/trees/(?P<tree_id>[^/]+)/nodes/(?P<node_id>[^/]+)/advisor/context$"
)
_CONVERSATIONS = re.compile(r"^/api/advisor/conversations$")
_CONVERSATION_MESSAGES = re.compile(
    r"^/api/advisor/conversations/(?P<conversation_id>[^/]+)/messages$"
)
_JOB = re.compile(r"^/api/advisor/jobs/(?P<job_id>[^/]+)$")
_PROPOSAL = re.compile(r"^/api/advisor/proposals/(?P<proposal_id>[^/]+)$")
_PROPOSAL_APPROVE = re.compile(r"^/api/advisor/proposals/(?P<proposal_id>[^/]+)/approve$")
_PROPOSAL_REJECT = re.compile(r"^/api/advisor/proposals/(?P<proposal_id>[^/]+)/reject$")


class ApiError(Exception):
    """Carries the HTTP status the caller should respond with."""

    def __init__(self, status: int, message: str) -> None:
        self.status = status
        self.message = message
        super().__init__(message)


def handle_get(
    path: str, *, db_path: str | None = None
) -> tuple[int, dict[str, Any]]:
    match = _NODE_CONTEXT.match(path)
    if match:
        try:
            context = services.get_node_context(
                match["tree_id"], match["node_id"], db_path=db_path
            )
        except services.TreeNotFound as exc:
            raise ApiError(404, f"tree not found: {exc}") from exc
        except NodeNotFoundError as exc:
            raise ApiError(404, f"node not found: {exc}") from exc
        return 200, {"ok": True, "context": context.model_dump(mode="json")}

    match = _CONVERSATION_MESSAGES.match(path)
    if match:
        messages = services.list_messages(match["conversation_id"], db_path=db_path)
        return 200, {
            "ok": True,
            "messages": [
                {
                    "message_id": m.message_id,
                    "role": m.role,
                    "content": m.content,
                    "created_at": m.created_at,
                }
                for m in messages
            ],
        }

    match = _JOB.match(path)
    if match:
        job = jobs.get_job(match["job_id"], db_path=db_path)
        if job is None:
            raise ApiError(404, "job not found")
        return 200, {"ok": True, "job": _job_payload(job)}

    match = _PROPOSAL.match(path)
    if match:
        try:
            proposal_id = UUID(match["proposal_id"])
        except ValueError as exc:
            raise ApiError(400, "proposal_id must be a UUID") from exc
        try:
            record = services.get_proposal(proposal_id, db_path=db_path)
        except services.ProposalNotFound as exc:
            raise ApiError(404, f"proposal not found: {exc}") from exc
        return 200, {
            "ok": True,
            "status": record.status,
            "proposal": record.proposal.model_dump(mode="json"),
        }

    raise ApiError(404, "not found")


def handle_post(
    path: str, body: dict[str, Any], *, db_path: str | None = None
) -> tuple[int, dict[str, Any]]:
    if _CONVERSATIONS.match(path):
        tree_id = _require_str(body, "tree_id")
        node_id = _require_str(body, "node_id")
        caller_id = str(body.get("caller_id") or "local-user")
        conversation = services.create_conversation(
            tree_id, node_id, caller_id=caller_id, db_path=db_path
        )
        return 201, {
            "ok": True,
            "conversation_id": conversation.conversation_id,
            "tree_id": conversation.tree_id,
            "node_id": conversation.node_id,
        }

    match = _CONVERSATION_MESSAGES.match(path)
    if match:
        node_id = _require_str(body, "node_id")
        views = body.get("views")
        if not isinstance(views, list):
            raise ApiError(400, "'views' must be a list")
        caller_id = str(body.get("caller_id") or "local-user")
        message, job_id = services.post_message_and_enqueue(
            match["conversation_id"],
            {"node_id": node_id, "views": views},
            caller_id=caller_id,
            db_path=db_path,
        )
        return 202, {"ok": True, "message_id": message.message_id, "job_id": job_id}

    match = _PROPOSAL_APPROVE.match(path)
    if match:
        proposal_id = _require_uuid(match["proposal_id"])
        proposal_hash = _require_str(body, "proposal_hash")
        idempotency_key = _require_str(body, "idempotency_key")
        approved_by = str(body.get("approved_by") or "local-user")
        try:
            result = services.approve_proposal(
                proposal_id,
                proposal_hash=proposal_hash,
                approved_by=approved_by,
                idempotency_key=idempotency_key,
                db_path=db_path,
            )
        except Exception as exc:  # translated to a stable 409 contract by _approval_error_response
            return _approval_error_response(exc)
        return 200, {
            "ok": True,
            "new_revision_id": result.new_revision_id,
            "approval_id": result.approval_id,
        }

    match = _PROPOSAL_REJECT.match(path)
    if match:
        proposal_id = _require_uuid(match["proposal_id"])
        rejected_by = str(body.get("rejected_by") or "local-user")
        reason = body.get("reason")
        try:
            services.reject_proposal(
                proposal_id, rejected_by=rejected_by, reason=reason, db_path=db_path
            )
        except services.ProposalNotFound as exc:
            raise ApiError(404, f"proposal not found: {exc}") from exc
        return 200, {"ok": True}

    raise ApiError(404, "not found")


def _approval_error_response(exc: Exception) -> tuple[int, dict[str, Any]]:
    """§8.3: a stale revision or stale data both come back as 409 with a
    stable machine-readable code, never a generic 500."""

    from lazyportfolio.advisor import approval_service

    if isinstance(exc, approval_service.StaleRevisionError):
        return 409, {"ok": False, "code": "stale_revision", "error": str(exc)}
    if isinstance(exc, approval_service.StaleDataError):
        return 409, {"ok": False, "code": "stale_data", "error": str(exc)}
    if isinstance(exc, approval_service.ProposalExpired):
        return 409, {"ok": False, "code": "expired", "error": str(exc)}
    if isinstance(exc, approval_service.ProposalNotPendingApproval):
        return 409, {"ok": False, "code": "not_pending_approval", "error": str(exc)}
    if isinstance(exc, approval_service.ApprovalHashMismatch):
        return 409, {"ok": False, "code": "hash_mismatch", "error": str(exc)}
    if isinstance(exc, approval_service.ProposalNotFound):
        raise ApiError(404, f"proposal not found: {exc}") from exc
    raise exc


def _job_payload(job: jobs.JobRecord) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "kind": job.kind,
        "status": job.status,
        "started_at": job.started_at,
        "heartbeat_at": job.heartbeat_at,
        "finished_at": job.finished_at,
        "error": job.error,
    }


def _require_str(body: dict[str, Any], key: str) -> str:
    value = body.get(key)
    if not isinstance(value, str) or not value:
        raise ApiError(400, f"'{key}' is required")
    return value


def _require_uuid(raw: str) -> UUID:
    try:
        return UUID(raw)
    except ValueError as exc:
        raise ApiError(400, "id must be a UUID") from exc


__all__ = ["ApiError", "handle_get", "handle_post"]
