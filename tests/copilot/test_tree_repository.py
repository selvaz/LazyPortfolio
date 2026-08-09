"""docs/node-copilot-operational-plan.md §13 Fase 1 exit criteria: every save
creates a revision, no overwrite loses history, two concurrent applies
cannot both succeed."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from lazyportfolio.copilot.repository import (
    ConcurrentTreeWrite,
    create_tree,
    get_head,
    save_revision,
)


def _config(value: str) -> dict[str, object]:
    return {
        "root_id": "root",
        "currency": "USD",
        "nodes": [
            {
                "id": "root",
                "name": "Root",
                "children": [],
                "instruments": [value],
                "goal": {"objective": "min_risk"},
                "constraints": {},
            }
        ],
        "backtest": {"benchmark": {"name": "B0", "weights": {value: 1.0}}},
    }


def test_create_tree_makes_one_initial_revision_with_no_parent(tmp_path: Path) -> None:
    db_path = tmp_path / "db.sqlite3"
    revision = create_tree(
        _config("AAA"), actor_type="human", actor_id="local-user", db_path=db_path
    )
    assert revision.parent_revision_id is None
    head = get_head(revision.tree_id, db_path=db_path)
    assert head is not None
    assert head.revision_id == revision.revision_id
    assert head.config == _config("AAA")


def test_save_revision_appends_and_never_overwrites_history(tmp_path: Path) -> None:
    db_path = tmp_path / "db.sqlite3"
    first = create_tree(_config("AAA"), actor_type="human", actor_id="local-user", db_path=db_path)
    second = save_revision(
        first.tree_id, _config("BBB"), actor_type="human", actor_id="local-user", db_path=db_path
    )
    assert second.parent_revision_id == first.revision_id
    assert second.revision_id != first.revision_id
    head = get_head(first.tree_id, db_path=db_path)
    assert head is not None
    assert head.revision_id == second.revision_id
    assert head.config == _config("BBB")


def test_save_revision_with_stale_expected_head_raises(tmp_path: Path) -> None:
    db_path = tmp_path / "db.sqlite3"
    first = create_tree(_config("AAA"), actor_type="human", actor_id="local-user", db_path=db_path)
    save_revision(
        first.tree_id, _config("BBB"), actor_type="human", actor_id="local-user", db_path=db_path
    )
    # first.revision_id is no longer the head -- a save pinned to it must fail, not silently rebase.
    with pytest.raises(ConcurrentTreeWrite):
        save_revision(
            first.tree_id,
            _config("CCC"),
            actor_type="human",
            actor_id="local-user",
            expected_head=first.revision_id,
            db_path=db_path,
        )
    # The rejected write left no trace: head is still BBB's revision.
    head = get_head(first.tree_id, db_path=db_path)
    assert head is not None
    assert head.config == _config("BBB")


def test_save_revision_on_a_tree_with_no_head_raises(tmp_path: Path) -> None:
    db_path = tmp_path / "db.sqlite3"
    with pytest.raises(ConcurrentTreeWrite):
        save_revision(
            "no-such-tree",
            _config("AAA"),
            actor_type="human",
            actor_id="local-user",
            db_path=db_path,
        )


def test_creating_the_same_tree_id_twice_is_impossible_through_the_public_api(
    tmp_path: Path,
) -> None:
    """create_tree always mints its own tree_id, so this exercises the CAS
    guard indirectly: two saves racing on the same tree_id can't both
    "create" it."""

    db_path = tmp_path / "db.sqlite3"
    first = create_tree(_config("AAA"), actor_type="human", actor_id="local-user", db_path=db_path)
    second = create_tree(_config("AAA"), actor_type="human", actor_id="local-user", db_path=db_path)
    assert first.tree_id != second.tree_id


def test_two_concurrent_saves_racing_on_the_same_head_only_one_succeeds(tmp_path: Path) -> None:
    """Real concurrency, not just sequential simulation: two threads each
    open their own connection (WAL + busy_timeout, see db.connect) and race
    a save pinned to the same expected_head. Exactly one must win."""

    db_path = tmp_path / "db.sqlite3"
    first = create_tree(_config("AAA"), actor_type="human", actor_id="local-user", db_path=db_path)

    results: list[str] = []
    errors: list[Exception] = []
    barrier = threading.Barrier(2)

    def _attempt(value: str) -> None:
        barrier.wait()
        try:
            save_revision(
                first.tree_id,
                _config(value),
                actor_type="human",
                actor_id="local-user",
                expected_head=first.revision_id,
                db_path=db_path,
            )
            results.append(value)
        except ConcurrentTreeWrite as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_attempt, args=(v,)) for v in ("WINNER", "LOSER")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 1
    assert len(errors) == 1
    head = get_head(first.tree_id, db_path=db_path)
    assert head is not None
    assert head.config == _config(results[0])
