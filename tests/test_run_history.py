"""Structured run history: record/lookup/list, and attached report artifacts.

Sibling to test_v2_store.py -- same shared-database module family
(lazyportfolio.v2.db), same "pass an explicit db_path per test" isolation.
"""

from __future__ import annotations

from lazyportfolio.v2 import run_history


def _record(db_path, **overrides):
    fields = {
        "cache_key": "key-1",
        "path": "/api/v2/estimate",
        "kind": "estimate",
        "tree_id": "My-Tree",
        "config_hash": "hash-1",
        "data_as_of": "2026-08-01",
        "data_fingerprint": "fp-1",
        "weights": {"AAA": 0.6, "BBB": 0.4},
        "metrics": None,
        "payload": {"ok": True, "terminal_weights": {"AAA": 0.6, "BBB": 0.4}},
    }
    fields.update(overrides)
    return run_history.record_run(db_path=db_path, **fields)


def test_record_then_get_by_cache_key_round_trips(tmp_path) -> None:
    db_path = tmp_path / "store.sqlite3"
    _record(db_path)
    found = run_history.get_by_cache_key("key-1", db_path=db_path)
    assert found is not None
    assert found["payload"] == {"ok": True, "terminal_weights": {"AAA": 0.6, "BBB": 0.4}}


def test_get_by_cache_key_miss_returns_none(tmp_path) -> None:
    db_path = tmp_path / "store.sqlite3"
    assert run_history.get_by_cache_key("nope", db_path=db_path) is None


def test_record_run_with_same_cache_key_replaces_the_row(tmp_path) -> None:
    db_path = tmp_path / "store.sqlite3"
    first_id = _record(db_path, payload={"ok": True, "terminal_weights": {"AAA": 1.0}})
    second_id = _record(db_path, payload={"ok": True, "terminal_weights": {"AAA": 0.5, "BBB": 0.5}})
    assert first_id == second_id
    found = run_history.get_by_cache_key("key-1", db_path=db_path)
    assert found["payload"]["terminal_weights"] == {"AAA": 0.5, "BBB": 0.5}
    assert len(run_history.list_runs(db_path=db_path)) == 1


def test_list_runs_orders_newest_first_and_omits_payload(tmp_path) -> None:
    db_path = tmp_path / "store.sqlite3"
    _record(db_path, cache_key="key-a", tree_id="Tree-A")
    _record(db_path, cache_key="key-b", tree_id="Tree-B")
    rows = run_history.list_runs(db_path=db_path)
    assert [row["cache_key"] for row in rows] == ["key-b", "key-a"]
    assert "payload" not in rows[0]


def test_list_runs_filters_by_tree_id(tmp_path) -> None:
    db_path = tmp_path / "store.sqlite3"
    _record(db_path, cache_key="key-a", tree_id="Tree-A")
    _record(db_path, cache_key="key-b", tree_id="Tree-B")
    rows = run_history.list_runs(tree_id="Tree-A", db_path=db_path)
    assert [row["cache_key"] for row in rows] == ["key-a"]


def test_list_runs_respects_limit(tmp_path) -> None:
    db_path = tmp_path / "store.sqlite3"
    for index in range(5):
        _record(db_path, cache_key=f"key-{index}")
    assert len(run_history.list_runs(limit=2, db_path=db_path)) == 2


def test_get_run_includes_payload_and_attached_artifacts(tmp_path) -> None:
    db_path = tmp_path / "store.sqlite3"
    run_id = _record(db_path)
    run_history.attach_artifact(
        run_id,
        kind="report",
        content_type="text/html",
        filename="r.html",
        blob=b"<html></html>",
        external_artifact_id="ext-1",
        db_path=db_path,
    )
    row = run_history.get_run(run_id, db_path=db_path)
    assert row is not None
    assert row["payload"]["ok"] is True
    assert len(row["artifacts"]) == 1
    assert row["artifacts"][0]["external_artifact_id"] == "ext-1"


def test_get_run_missing_id_returns_none(tmp_path) -> None:
    db_path = tmp_path / "store.sqlite3"
    assert run_history.get_run(999, db_path=db_path) is None


def test_get_report_artifact_returns_the_most_recent_report(tmp_path) -> None:
    db_path = tmp_path / "store.sqlite3"
    run_id = _record(db_path)
    run_history.attach_artifact(
        run_id, kind="report", content_type="text/html", filename="old.html", blob=b"old", db_path=db_path
    )
    run_history.attach_artifact(
        run_id, kind="report", content_type="text/html", filename="new.html", blob=b"new", db_path=db_path
    )
    body, content_type, filename = run_history.get_report_artifact("key-1", db_path=db_path)
    assert (body, content_type, filename) == (b"new", "text/html", "new.html")


def test_get_report_artifact_miss_returns_none(tmp_path) -> None:
    db_path = tmp_path / "store.sqlite3"
    assert run_history.get_report_artifact("nope", db_path=db_path) is None
