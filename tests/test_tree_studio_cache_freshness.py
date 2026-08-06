"""A cached /api/v2/estimate|backtest|scientific-study result must not be
served across a Market Data Hub refresh: the cache key has to fold in a
data-freshness fingerprint, not just the tree config's own hash, or a
persistent cache can hand back a report/backtest computed against
pre-refresh data forever.

Drives the real HTTP handler (mirrors test_tree_studio_artifact_registry.py's
pattern) with ``tree_studio._data_fingerprint`` monkeypatched to a
controllable stub -- this isolates the cache-key behavior under test from
needing a real Market Data Hub database, while still exercising the actual
request path (in-memory cache, disk-backed run_history fallback, producer
invocation) rather than just the key-formatting helper in isolation.

``project/tree_studio.py`` is a script, not an installed package, so it is
imported the same way ``tests/test_tree_studio_artifact_registry.py`` does:
put ``project/`` on ``sys.path`` first.
"""

from __future__ import annotations

import http.client
import importlib
import json
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_DIR = REPO_ROOT / "project"


def _config() -> dict[str, Any]:
    return {
        "root_id": "root",
        "nodes": [
            {
                "id": "root",
                "name": "Freshness Tree",
                "children": [],
                "instruments": ["AAA"],
                "proxy": "",
                "goal": {"objective": "min_risk"},
                "constraints": {},
            }
        ],
        "data": {"start": "", "end": ""},
        "backtest": {"benchmark": {"name": "B0", "weights": {"AAA": 1.0}}},
    }


def _post(port: int, path: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        body = json.dumps(payload).encode("utf-8")
        conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read())
    finally:
        conn.close()


@pytest.fixture()
def studio(monkeypatch, tmp_path):
    monkeypatch.setenv("LAZYPORTFOLIO_TREE_DB", str(tmp_path / "store.sqlite3"))
    sys.path.insert(0, str(PROJECT_DIR))
    try:
        module = importlib.import_module("tree_studio")
        module = importlib.reload(module)  # fresh in-memory caches, re-read env vars

        calls: list[int] = []

        def _fake_estimate_payload(config: dict[str, Any]) -> dict[str, Any]:
            calls.append(1)
            return {
                "ok": True,
                "engine": "fake",
                "terminal_weights": {"AAA": 1.0},
                "call_count": len(calls),
            }

        monkeypatch.setattr(module, "_v2_estimate_payload", _fake_estimate_payload)

        server = ThreadingHTTPServer(("127.0.0.1", 0), module.StudioHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield module, server.server_address[1], calls
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
    finally:
        sys.path.remove(str(PROJECT_DIR))


def test_same_config_same_fingerprint_is_a_cache_hit(studio, monkeypatch):
    module, port, calls = studio
    monkeypatch.setattr(module, "_data_fingerprint", lambda config: ("2026-08-01", "fp-A"))

    status1, body1 = _post(port, "/api/v2/estimate", _config())
    status2, body2 = _post(port, "/api/v2/estimate", _config())

    assert status1 == status2 == 200
    assert body1["cached"] is False
    assert body2["cached"] is True
    assert body2["call_count"] == body1["call_count"]
    assert len(calls) == 1


def test_same_config_different_fingerprint_recomputes(studio, monkeypatch):
    """The exact bug being fixed: a MarketDataHub refresh (modeled here as the
    coverage fingerprint changing) must invalidate the cached result for an
    otherwise-unchanged tree config."""
    module, port, calls = studio
    monkeypatch.setattr(module, "_data_fingerprint", lambda config: ("2026-08-01", "fp-A"))
    status1, body1 = _post(port, "/api/v2/estimate", _config())
    assert status1 == 200
    assert body1["cached"] is False

    monkeypatch.setattr(module, "_data_fingerprint", lambda config: ("2026-08-02", "fp-B"))
    status2, body2 = _post(port, "/api/v2/estimate", _config())

    assert status2 == 200
    assert body2["cached"] is False
    assert len(calls) == 2


def test_stale_fingerprint_result_remains_retrievable_from_disk_history(studio, monkeypatch):
    """Old fingerprint's result isn't lost -- it's a distinct, still-queryable
    run_history row (not overwritten by the fresh one), served from the
    disk-backed history even after the in-memory cache is evicted/restarted."""
    module, port, calls = studio
    monkeypatch.setattr(module, "_data_fingerprint", lambda config: ("2026-08-01", "fp-A"))
    _post(port, "/api/v2/estimate", _config())

    monkeypatch.setattr(module, "_data_fingerprint", lambda config: ("2026-08-02", "fp-B"))
    _post(port, "/api/v2/estimate", _config())

    module.StudioHandler._run_cache.clear()  # simulate a process restart

    monkeypatch.setattr(module, "_data_fingerprint", lambda config: ("2026-08-01", "fp-A"))
    status3, body3 = _post(port, "/api/v2/estimate", _config())

    assert status3 == 200
    assert body3["cached"] is True
    assert len(calls) == 2  # third request reused fp-A's already-computed (disk) result
