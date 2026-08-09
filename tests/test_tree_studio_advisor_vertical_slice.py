"""End-to-end vertical slice of the Node Advisor's Fase 3 REST surface,
with NO LLM anywhere (docs/node-advisor-operational-plan.md §13 Fase 3 exit
criteria): fixture views -> conversation -> job -> worker -> proposal card
-> approve -> new revision -> confirm. Also proves a crashed worker's job is
recoverable (heartbeat-based reap), not stuck forever.

``project/tree_studio.py`` is a script, not an installed package -- same
sys.path pattern as ``tests/test_tree_studio_cache_freshness.py``.
"""

from __future__ import annotations

import functools
import http.client
import importlib
import json
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from lazyportfolio.advisor.conversation_repository import create_conversation
from lazyportfolio.advisor.repository import create_tree, get_head
from lazyportfolio.backend import OptimizationDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_DIR = REPO_ROOT / "project"


def _config() -> dict[str, Any]:
    return {
        "root_id": "root",
        "currency": "USD",
        "nodes": [
            {
                "id": "root",
                "name": "Root",
                "children": ["equity"],
                "instruments": [],
                "goal": {"objective": "min_risk"},
                "constraints": {},
            },
            {
                "id": "equity",
                "name": "Equity",
                "children": [],
                "instruments": ["ticker:VTI", "ticker:VXUS"],
                "proxy": "ticker:VTI",
                # max_ratio: consumes expected returns, so the injected view
                # actually moves the solved weights (see Fase 2's
                # test_counterfactual_evaluator.py for why min_risk/hrp would not).
                "goal": {"objective": "max_ratio"},
                "constraints": {},
            },
        ],
        "backtest": {
            "benchmark": {"name": "B0", "weights": {"ticker:VTI": 0.5, "ticker:VXUS": 0.5}}
        },
    }


class _FakeBackend:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def load_returns(self, instruments, *, start="", end="", frequency="D", currency=None):
        return OptimizationDataset(
            returns=self.frame.loc[:, instruments],
            metadata={"source": "fake-hub", "database_identity": "fake-hub"},
        )


@pytest.fixture()
def frame() -> pd.DataFrame:
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(20260809)
    index = pd.bdate_range("2020-01-01", periods=300)
    return pd.DataFrame(
        {
            "ticker:VTI": rng.normal(0.0005, 0.01, len(index)),
            "ticker:VXUS": rng.normal(0.0003, 0.008, len(index)),
        },
        index=index,
    )


@pytest.fixture()
def studio(monkeypatch, tmp_path):
    store_path = tmp_path / "store.sqlite3"
    monkeypatch.setenv("LAZYPORTFOLIO_TREE_DB", str(store_path))
    sys.path.insert(0, str(PROJECT_DIR))
    try:
        module = importlib.import_module("tree_studio")
        module = importlib.reload(module)
        server = ThreadingHTTPServer(("127.0.0.1", 0), module.StudioHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield module, server.server_address[1], store_path
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
    finally:
        sys.path.remove(str(PROJECT_DIR))


def _post(port: int, path: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        body = json.dumps(payload).encode("utf-8")
        conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read())
    finally:
        conn.close()


def _get(port: int, path: str) -> tuple[int, dict[str, Any]]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read())
    finally:
        conn.close()


def _run_fixture_worker_once(module: Any, *, db_path: Path, backend: _FakeBackend) -> bool:
    """Fase 3 has no persistent worker in the test process (main() isn't
    called) -- run exactly one job synchronously, with the fake backend
    threaded through, matching how the real background thread would call
    the same handler."""

    handlers = {
        module._advisor_jobs.FIXTURE_PROPOSAL: functools.partial(
            module._advisor_services.handle_fixture_proposal_job,
            backend=backend,
            db_path=str(db_path),
        )
    }
    return module._advisor_jobs.run_worker_once(handlers=handlers, db_path=str(db_path))


def test_fixture_to_approval_vertical_slice(studio, frame) -> None:
    module, port, store_path = studio
    backend = _FakeBackend(frame)

    tree = create_tree(_config(), actor_type="human", actor_id="test", db_path=str(store_path))

    status, conv = _post(
        port, "/api/advisor/conversations", {"tree_id": tree.tree_id, "node_id": "equity"}
    )
    assert status == 201
    conversation_id = conv["conversation_id"]

    status, message_response = _post(
        port,
        f"/api/advisor/conversations/{conversation_id}/messages",
        {
            "node_id": "equity",
            "views": [
                {
                    "instruments": {"ticker:VTI": 1.0, "ticker:VXUS": -1.0},
                    "expected_return": 0.03,
                    "confidence": 0.6,
                    "rationale": "vertical slice test",
                }
            ],
        },
    )
    assert status == 202
    job_id = message_response["job_id"]

    status, job_before = _get(port, f"/api/advisor/jobs/{job_id}")
    assert status == 200
    assert job_before["job"]["status"] == "queued"

    ran = _run_fixture_worker_once(module, db_path=store_path, backend=backend)
    assert ran is True

    status, job_after = _get(port, f"/api/advisor/jobs/{job_id}")
    assert status == 200
    assert job_after["job"]["status"] == "succeeded", job_after["job"]["error"]

    status, messages = _get(port, f"/api/advisor/conversations/{conversation_id}/messages")
    assert status == 200
    assistant_messages = [m for m in messages["messages"] if m["role"] == "assistant"]
    assert len(assistant_messages) == 1
    proposal_id = assistant_messages[0]["content"]["proposal_id"]

    status, proposal_response = _get(port, f"/api/advisor/proposals/{proposal_id}")
    assert status == 200
    assert proposal_response["status"] == "pending_approval"
    proposal = proposal_response["proposal"]
    assert proposal["node_id"] == "equity"

    status, approve_response = _post(
        port,
        f"/api/advisor/proposals/{proposal_id}/approve",
        {
            "proposal_hash": proposal["content_hash"],
            "idempotency_key": "vertical-slice-test",
            "approved_by": "test",
        },
    )
    assert status == 200, approve_response
    new_revision_id = approve_response["new_revision_id"]
    assert new_revision_id != tree.revision_id

    head = get_head(tree.tree_id, db_path=str(store_path))
    assert head is not None
    assert head.revision_id == new_revision_id
    equity = next(n for n in head.config["nodes"] if n["id"] == "equity")
    assert equity["constraints"]["views"][0]["instruments"] == {
        "ticker:VTI": 1.0,
        "ticker:VXUS": -1.0,
    }

    status, proposal_after = _get(port, f"/api/advisor/proposals/{proposal_id}")
    assert proposal_after["status"] == "applied"


def test_approving_with_the_wrong_hash_is_rejected_with_409(studio, frame) -> None:
    module, port, store_path = studio
    backend = _FakeBackend(frame)
    tree = create_tree(_config(), actor_type="human", actor_id="test", db_path=str(store_path))
    _, conv = _post(
        port, "/api/advisor/conversations", {"tree_id": tree.tree_id, "node_id": "equity"}
    )
    _, message_response = _post(
        port,
        f"/api/advisor/conversations/{conv['conversation_id']}/messages",
        {
            "node_id": "equity",
            "views": [
                {
                    "instruments": {"ticker:VTI": 1.0, "ticker:VXUS": -1.0},
                    "expected_return": 0.03,
                    "confidence": 0.6,
                    "rationale": "test",
                }
            ],
        },
    )
    _run_fixture_worker_once(module, db_path=store_path, backend=backend)
    status, messages = _get(port, f"/api/advisor/conversations/{conv['conversation_id']}/messages")
    proposal_id = next(m for m in messages["messages"] if m["role"] == "assistant")["content"][
        "proposal_id"
    ]

    status, response = _post(
        port,
        f"/api/advisor/proposals/{proposal_id}/approve",
        {"proposal_hash": "sha256:not-the-real-hash", "idempotency_key": "k"},
    )
    assert status == 409
    assert response["code"] == "hash_mismatch"


def test_rejecting_a_proposal_marks_it_rejected_and_leaves_the_head_untouched(
    studio, frame
) -> None:
    module, port, store_path = studio
    backend = _FakeBackend(frame)
    tree = create_tree(_config(), actor_type="human", actor_id="test", db_path=str(store_path))
    _, conv = _post(
        port, "/api/advisor/conversations", {"tree_id": tree.tree_id, "node_id": "equity"}
    )
    _post(
        port,
        f"/api/advisor/conversations/{conv['conversation_id']}/messages",
        {
            "node_id": "equity",
            "views": [
                {
                    "instruments": {"ticker:VTI": 1.0, "ticker:VXUS": -1.0},
                    "expected_return": 0.03,
                    "confidence": 0.6,
                    "rationale": "test",
                }
            ],
        },
    )
    _run_fixture_worker_once(module, db_path=store_path, backend=backend)
    _, messages = _get(port, f"/api/advisor/conversations/{conv['conversation_id']}/messages")
    proposal_id = next(m for m in messages["messages"] if m["role"] == "assistant")["content"][
        "proposal_id"
    ]

    status, response = _post(
        port, f"/api/advisor/proposals/{proposal_id}/reject", {"rejected_by": "test"}
    )
    assert status == 200 and response["ok"] is True

    status, proposal_after = _get(port, f"/api/advisor/proposals/{proposal_id}")
    assert proposal_after["status"] == "rejected"
    head = get_head(tree.tree_id, db_path=str(store_path))
    assert head is not None
    assert head.revision_id == tree.revision_id  # unchanged


def test_node_context_endpoint(studio) -> None:
    module, port, store_path = studio
    del module
    tree = create_tree(_config(), actor_type="human", actor_id="test", db_path=str(store_path))

    status, response = _get(port, f"/api/trees/{tree.tree_id}/nodes/equity/advisor/context")

    assert status == 200
    assert response["context"]["node_id"] == "equity"
    assert set(response["context"]["allowed_view_instruments"]) == {
        "ticker:VTI",
        "ticker:VXUS",
    }


def test_worker_crash_is_recovered_by_the_heartbeat_reaper(studio) -> None:
    """§13 Fase 3 exit criterion: restart during any critical state is
    recoverable. Simulates a worker that claims a job and dies before
    finishing -- the job must not be stuck in 'running' forever."""

    module, port, store_path = studio
    del port
    jobs = module._advisor_jobs
    conversation = create_conversation(
        "fake-tree", "fake-node", user_id="test", db_path=str(store_path)
    )

    job_id = jobs.enqueue_job(
        conversation.conversation_id,
        "fake-message",
        jobs.FIXTURE_PROPOSAL,
        db_path=str(store_path),
    )
    claimed = jobs.claim_next_job(db_path=str(store_path))
    assert claimed is not None
    assert claimed.job_id == job_id
    assert claimed.status == "running"
    # The "worker" now crashes -- nothing marks the job succeeded/failed.

    # heartbeat_timeout_seconds=-1 makes the cutoff strictly in the future,
    # so the already-set heartbeat (however recent) is always reaped --
    # deterministic, no timing-dependent sleep needed.
    reaped = jobs.reap_orphaned_jobs(heartbeat_timeout_seconds=-1, db_path=str(store_path))
    assert reaped == [job_id]

    job = jobs.get_job(job_id, db_path=str(store_path))
    assert job is not None
    assert job.status == "queued"
    assert job.started_at is None
    assert job.heartbeat_at is None

    # And it can be claimed again -- recovery actually works end to end.
    reclaimed = jobs.claim_next_job(db_path=str(store_path))
    assert reclaimed is not None
    assert reclaimed.job_id == job_id


def test_fresh_heartbeat_is_not_reaped(studio) -> None:
    module, port, store_path = studio
    del port
    jobs = module._advisor_jobs
    conversation = create_conversation(
        "fake-tree", "fake-node", user_id="test", db_path=str(store_path)
    )
    job_id = jobs.enqueue_job(
        conversation.conversation_id,
        "fake-message",
        jobs.FIXTURE_PROPOSAL,
        db_path=str(store_path),
    )
    jobs.claim_next_job(db_path=str(store_path))

    reaped = jobs.reap_orphaned_jobs(heartbeat_timeout_seconds=3600, db_path=str(store_path))

    assert reaped == []
    job = jobs.get_job(job_id, db_path=str(store_path))
    assert job is not None
    assert job.status == "running"


# --------------------------------------------------------------------- #
# Fase 5: the real LLM path (advisor_turn), wired end to end into the same
# worker/HTTP surface the fixture path uses -- docs/node-advisor-operational-plan.md
# §13 Fase 5. The LLM call itself is mocked (same pattern as
# tests/test_advisor_agent.py); this file's job is proving the *wiring*
# (API routing by body shape, job kind, worker handler, assistant message
# shape), not re-proving run_advisor_turn's own behavior.
# --------------------------------------------------------------------- #
def _run_advisor_turn_worker_once(module: Any, *, db_path: Path, backend: _FakeBackend) -> bool:
    handlers = {
        module._advisor_jobs.ADVISOR_TURN: functools.partial(
            module._advisor_services.handle_advisor_turn_job,
            backend=backend,
            db_path=str(db_path),
        )
    }
    return module._advisor_jobs.run_worker_once(handlers=handlers, db_path=str(db_path))


def _stub_llm(monkeypatch, payload: Any) -> None:
    from types import SimpleNamespace

    class _FakeAgent:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __call__(self, prompt: str) -> Any:
            return SimpleNamespace(payload=payload, error=None)

    class _FakeLLMEngine:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    lazybridge = pytest.importorskip("lazybridge")
    monkeypatch.setattr(lazybridge, "Agent", _FakeAgent)
    monkeypatch.setattr(lazybridge, "LLMEngine", _FakeLLMEngine)


def test_advisor_turn_propose_route_via_http_creates_a_pending_proposal(
    studio, frame, monkeypatch
) -> None:
    pytest.importorskip("lazybridge")
    pytest.importorskip("lazytools")
    from advisor.agent import AdvisorTurnResult, CandidateView

    payload = AdvisorTurnResult(
        route="propose",
        message="SPY should outperform TLT.",
        proposed_views=[
            CandidateView(
                instruments={"ticker:VTI": 1.0, "ticker:VXUS": -1.0},
                expected_return=0.03,
                confidence=0.6,
                rationale="test",
            )
        ],
    )
    _stub_llm(monkeypatch, payload)

    module, port, store_path = studio
    backend = _FakeBackend(frame)
    tree = create_tree(_config(), actor_type="human", actor_id="test", db_path=str(store_path))
    _, conv = _post(
        port, "/api/advisor/conversations", {"tree_id": tree.tree_id, "node_id": "equity"}
    )

    status, message_response = _post(
        port,
        f"/api/advisor/conversations/{conv['conversation_id']}/messages",
        {"node_id": "equity", "text": "propose a relative view"},
    )
    assert status == 202

    ran = _run_advisor_turn_worker_once(module, db_path=store_path, backend=backend)
    assert ran is True

    status, job_after = _get(port, f"/api/advisor/jobs/{message_response['job_id']}")
    assert job_after["job"]["status"] == "succeeded", job_after["job"]["error"]

    status, messages = _get(port, f"/api/advisor/conversations/{conv['conversation_id']}/messages")
    assistant_message = next(m for m in messages["messages"] if m["role"] == "assistant")
    assert assistant_message["content"]["route"] == "propose"
    proposal_id = assistant_message["content"]["proposal_id"]
    assert proposal_id

    status, proposal_response = _get(port, f"/api/advisor/proposals/{proposal_id}")
    assert proposal_response["status"] == "pending_approval"
    assert proposal_response["proposal"]["model_provenance"]["producer_kind"] == (
        "interactive_chat"
    )


def test_advisor_turn_explain_route_via_http_creates_no_proposal(studio, monkeypatch) -> None:
    pytest.importorskip("lazybridge")
    pytest.importorskip("lazytools")
    from advisor.agent import AdvisorTurnResult

    payload = AdvisorTurnResult(
        route="explain", message="This node is currently min_risk with no views.", proposed_views=[]
    )
    _stub_llm(monkeypatch, payload)

    module, port, store_path = studio
    tree = create_tree(_config(), actor_type="human", actor_id="test", db_path=str(store_path))
    _, conv = _post(
        port, "/api/advisor/conversations", {"tree_id": tree.tree_id, "node_id": "equity"}
    )
    _post(
        port,
        f"/api/advisor/conversations/{conv['conversation_id']}/messages",
        {"node_id": "equity", "text": "why is this node weighted this way?"},
    )

    ran = _run_advisor_turn_worker_once(module, db_path=store_path, backend=None)
    assert ran is True

    status, messages = _get(port, f"/api/advisor/conversations/{conv['conversation_id']}/messages")
    assistant_message = next(m for m in messages["messages"] if m["role"] == "assistant")
    assert assistant_message["content"]["route"] == "explain"
    assert assistant_message["content"]["proposal_id"] is None
    head = get_head(tree.tree_id, db_path=str(store_path))
    assert head is not None
    assert head.revision_id == tree.revision_id  # unchanged -- no proposal, no revision


def test_posting_both_views_and_text_is_rejected_with_400(studio) -> None:
    module, port, store_path = studio
    del module
    tree = create_tree(_config(), actor_type="human", actor_id="test", db_path=str(store_path))
    _, conv = _post(
        port, "/api/advisor/conversations", {"tree_id": tree.tree_id, "node_id": "equity"}
    )

    status, response = _post(
        port,
        f"/api/advisor/conversations/{conv['conversation_id']}/messages",
        {"node_id": "equity", "views": [], "text": "hello"},
    )
    assert status == 400


def test_posting_neither_views_nor_text_is_rejected_with_400(studio) -> None:
    module, port, store_path = studio
    del module
    tree = create_tree(_config(), actor_type="human", actor_id="test", db_path=str(store_path))
    _, conv = _post(
        port, "/api/advisor/conversations", {"tree_id": tree.tree_id, "node_id": "equity"}
    )

    status, response = _post(
        port,
        f"/api/advisor/conversations/{conv['conversation_id']}/messages",
        {"node_id": "equity"},
    )
    assert status == 400
