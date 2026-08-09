"""docs/node-advisor-operational-plan.md §13 Fase 1 exit criterion: the
legacy trees table migrates into revisions without ever being modified
itself, and a migrated tree's head config round-trips byte-equivalent to
what lazyportfolio.v2.store.read_model already returned."""

from __future__ import annotations

from pathlib import Path

from lazyportfolio.advisor.migration import migrate_legacy_trees, tree_id_for_name
from lazyportfolio.advisor.repository import get_head
from lazyportfolio.v2.store import list_saved_models, read_model, write_model


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


def test_migrate_creates_a_revision_matching_the_legacy_config(tmp_path: Path) -> None:
    db_path = tmp_path / "db.sqlite3"
    write_model("Legacy Tree", _config("AAA"), store_path=db_path)

    migrated = migrate_legacy_trees(db_path=db_path)
    assert migrated == ["Legacy Tree"]

    tree_id = tree_id_for_name("Legacy Tree", db_path=db_path)
    assert tree_id is not None
    head = get_head(tree_id, db_path=db_path)
    assert head is not None
    assert head.config == read_model("Legacy Tree", store_path=db_path)
    assert head.parent_revision_id is None


def test_migration_never_touches_the_legacy_trees_table(tmp_path: Path) -> None:
    db_path = tmp_path / "db.sqlite3"
    write_model("Legacy Tree", _config("AAA"), store_path=db_path)
    before = list_saved_models(store_path=db_path)

    migrate_legacy_trees(db_path=db_path)

    after = list_saved_models(store_path=db_path)
    assert before == after
    # The legacy read path is completely unaffected by the migration.
    assert read_model("Legacy Tree", store_path=db_path) == _config("AAA")


def test_migration_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "db.sqlite3"
    write_model("Legacy Tree", _config("AAA"), store_path=db_path)

    first_pass = migrate_legacy_trees(db_path=db_path)
    second_pass = migrate_legacy_trees(db_path=db_path)

    assert first_pass == ["Legacy Tree"]
    assert second_pass == []  # already migrated, not re-migrated
    # Still exactly one tree_id for that name, not a second one.
    tree_id_first = tree_id_for_name("Legacy Tree", db_path=db_path)
    assert tree_id_first is not None


def test_migration_handles_multiple_trees_independently(tmp_path: Path) -> None:
    db_path = tmp_path / "db.sqlite3"
    write_model("Tree A", _config("AAA"), store_path=db_path)
    write_model("Tree B", _config("BBB"), store_path=db_path)

    migrated = migrate_legacy_trees(db_path=db_path)
    assert set(migrated) == {"Tree A", "Tree B"}

    tree_a_id = tree_id_for_name("Tree A", db_path=db_path)
    tree_b_id = tree_id_for_name("Tree B", db_path=db_path)
    assert tree_a_id != tree_b_id

    head_a = get_head(tree_a_id, db_path=db_path)  # type: ignore[arg-type]
    head_b = get_head(tree_b_id, db_path=db_path)  # type: ignore[arg-type]
    assert head_a is not None and head_a.config == _config("AAA")
    assert head_b is not None and head_b.config == _config("BBB")


def test_tree_id_for_unmigrated_name_is_none(tmp_path: Path) -> None:
    db_path = tmp_path / "db.sqlite3"
    write_model("Never Migrated", _config("AAA"), store_path=db_path)
    assert tree_id_for_name("Never Migrated", db_path=db_path) is None
