"""Tree Studio's /api/v2/client-report and /api/v2/audit-bundle handlers must
catalog a genuinely (re)generated HTML report into the shared LazyTools
artifact registry (``lazytools.registry``), exactly once per config, and
never for the audit ZIP.

Uses a REAL temporary sqlite artifact DB (no mocks for the registry itself)
plus a real running ``ThreadingHTTPServer`` driving the actual
``StudioHandler.do_POST`` -- the same request path a browser hits. The one
thing monkeypatched is ``tree_studio._v2_export_artifacts`` itself, so these
tests exercise the artifact-registration wiring without needing a live
Market Data Hub database to produce a real backtest/report.

``project/tree_studio.py`` is a script, not an installed package, so it is
imported the same way ``tests/test_studio_compat.py`` already does: put
``project/`` on ``sys.path`` first.
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

pytest.importorskip(
    "lazytools", reason="these tests exercise the real lazytools.registry artifact DB"
)

from lazytools.registry import search_artifacts  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_DIR = REPO_ROOT / "project"


def _config(name: str) -> dict[str, Any]:
    return {
        "root_id": "root",
        "nodes": [
            {
                "id": "root",
                "name": name,
                "children": ["equity"],
                "instruments": ["AGG"],
                "proxy": "",
                "goal": {"objective": "min_risk"},
                "constraints": {},
            },
            {
                "id": "equity",
                "name": "Equity",
                "children": [],
                "instruments": ["SPY", "VGK"],
                "proxy": "ACWI",
                "goal": {"objective": "min_risk"},
                "constraints": {},
            },
        ],
        "data": {"start": "2018-01-01", "end": ""},
        "backtest": {
            "id": "tree-studio",
            "benchmark": {"name": "B0", "weights": {"ACWI": 0.7, "AGG": 0.3}},
        },
    }


def _fake_export_artifacts(config: dict[str, Any]) -> dict[str, tuple[bytes, str, str]]:
    """Stand-in for the real (expensive, Market-Data-Hub-backed) exporter."""
    root_name = config["nodes"][0]["name"]
    html = f"<html><body>Report for {root_name}</body></html>".encode()
    zip_bytes = b"PK\x03\x04-fake-audit-zip-bytes"
    return {
        "audit": (zip_bytes, "application/zip", "fake-audit.zip"),
        "report": (html, "text/html; charset=utf-8", "fake-report.html"),
    }


def _post(port: int, path: str, payload: dict[str, Any]) -> tuple[int, bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        body = json.dumps(payload).encode("utf-8")
        conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


@pytest.fixture()
def studio(monkeypatch, tmp_path):
    """Import (or reload) tree_studio with isolated model/cache directories,
    then serve it on a real ephemeral localhost port for the duration of the
    test."""
    monkeypatch.setenv("LAZYPORTFOLIO_TREE_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("LAZYPORTFOLIO_TREE_CACHE_DB", str(tmp_path / "run_cache.sqlite3"))
    sys.path.insert(0, str(PROJECT_DIR))
    try:
        module = importlib.import_module("tree_studio")
        module = importlib.reload(module)  # fresh in-memory caches, re-read env vars
        monkeypatch.setattr(module, "_v2_export_artifacts", _fake_export_artifacts)

        server = ThreadingHTTPServer(("127.0.0.1", 0), module.StudioHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield module, server.server_address[1]
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
    finally:
        sys.path.remove(str(PROJECT_DIR))


def test_first_report_request_registers_exactly_one_artifact(studio, monkeypatch, tmp_path):
    module, port = studio
    db_path = tmp_path / "artifacts.sqlite"
    monkeypatch.setenv("LAZYPORTFOLIO_ARTIFACTS_DB", str(db_path))

    status, _ = _post(port, "/api/v2/client-report", _config("Alpha Tree"))
    assert status == 200

    results = search_artifacts(str(db_path), kind="report")
    assert len(results) == 1
    row = results[0]
    assert row["repo"] == "lazyportfolio"
    assert row["kind"] == "report"
    assert "tree-studio" in row["tags"]
    assert "Alpha Tree" in row["title"]
    assert "Alpha Tree" in row["summary"]


def test_repeat_request_same_config_in_memory_cache_hit_registers_once(
    studio, monkeypatch, tmp_path
):
    module, port = studio
    db_path = tmp_path / "artifacts.sqlite"
    monkeypatch.setenv("LAZYPORTFOLIO_ARTIFACTS_DB", str(db_path))
    config = _config("Beta Tree")

    status1, _ = _post(port, "/api/v2/client-report", config)
    status2, _ = _post(port, "/api/v2/client-report", config)
    assert status1 == 200
    assert status2 == 200

    results = search_artifacts(str(db_path), kind="report")
    assert len(results) == 1


def test_repeat_request_after_memory_eviction_disk_cache_hit_registers_once(
    studio, monkeypatch, tmp_path
):
    module, port = studio
    db_path = tmp_path / "artifacts.sqlite"
    monkeypatch.setenv("LAZYPORTFOLIO_ARTIFACTS_DB", str(db_path))
    config = _config("Gamma Tree")

    status1, _ = _post(port, "/api/v2/client-report", config)
    assert status1 == 200

    # Simulate the in-memory artifact cache having been evicted/restarted --
    # the disk-backed run_cache (lazyportfolio.v2.run_cache) should still
    # serve the report without re-registering an artifact.
    module.StudioHandler._artifact_cache.clear()

    status2, _ = _post(port, "/api/v2/client-report", config)
    assert status2 == 200

    results = search_artifacts(str(db_path), kind="report")
    assert len(results) == 1


def test_env_var_unset_skips_silently_without_breaking_the_response(studio, monkeypatch, tmp_path):
    module, port = studio
    monkeypatch.delenv("LAZYPORTFOLIO_ARTIFACTS_DB", raising=False)

    status, body = _post(port, "/api/v2/client-report", _config("Delta Tree"))

    assert status == 200
    assert body  # the HTML report body still comes through
    assert list(tmp_path.glob("*.sqlite")) == []


def test_registration_failure_does_not_break_the_response(studio, monkeypatch, tmp_path):
    module, port = studio
    db_path = tmp_path / "artifacts.sqlite"
    monkeypatch.setenv("LAZYPORTFOLIO_ARTIFACTS_DB", str(db_path))

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated lazytools.registry failure")

    monkeypatch.setattr("lazyportfolio.artifact_registry.register_artifact", _boom)

    status, body = _post(port, "/api/v2/client-report", _config("Epsilon Tree"))

    assert status == 200
    assert body


def test_audit_bundle_never_registers_an_artifact(studio, monkeypatch, tmp_path):
    module, port = studio
    db_path = tmp_path / "artifacts.sqlite"
    monkeypatch.setenv("LAZYPORTFOLIO_ARTIFACTS_DB", str(db_path))

    status, body = _post(port, "/api/v2/audit-bundle", _config("Zeta Tree"))

    assert status == 200
    assert body
    assert search_artifacts(str(db_path), kind="report") == []
