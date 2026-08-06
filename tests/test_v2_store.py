"""Shared model store: DB resolution, sanitization, validate-before-write.

Both Tree Studio and LazyTools' MCP `portfolio_tree_*` tools go through
`lazyportfolio.v2.store` so a tree saved by one is immediately visible to the
other -- these tests pin the contract that makes that true.
"""

from __future__ import annotations

import json

import pytest

from lazyportfolio.v2.store import (
    ModelStoreError,
    delete_model,
    list_saved_models,
    migrate_legacy_json_models,
    read_model,
    resolve_store_path,
    sanitize_model_name,
    write_model,
)


def _minimal_config() -> dict[str, object]:
    return {
        "root_id": "root",
        "nodes": [
            {
                "id": "root",
                "name": "Root",
                "children": [],
                "instruments": ["AAA", "BBB"],
                "proxy": "",
                "goal": {"objective": "min_risk"},
                "constraints": {},
            }
        ],
        "data": {"start": "", "end": ""},
        "backtest": {"benchmark": {"name": "B0", "weights": {"AAA": 0.5, "BBB": 0.5}}},
    }


# --------------------------------------------------------------------------- #
# Database path resolution precedence
# --------------------------------------------------------------------------- #


def test_explicit_store_path_wins_over_everything(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LAZYPORTFOLIO_TREE_DB", str(tmp_path / "from-env.sqlite3"))
    explicit = tmp_path / "explicit.sqlite3"
    assert resolve_store_path(explicit) == explicit.resolve()


def test_env_var_wins_over_computed_default(tmp_path, monkeypatch) -> None:
    db = tmp_path / "from-env.sqlite3"
    monkeypatch.setenv("LAZYPORTFOLIO_TREE_DB", str(db))
    assert resolve_store_path() == db.resolve()


def test_computed_default_is_repo_reports_tree_studio(monkeypatch) -> None:
    monkeypatch.delenv("LAZYPORTFOLIO_TREE_DB", raising=False)
    default = resolve_store_path()
    assert default.parts[-3:] == ("reports", "tree_studio", "tree_studio.sqlite3")


# --------------------------------------------------------------------------- #
# Sanitization
# --------------------------------------------------------------------------- #


def test_sanitize_collapses_unsafe_characters() -> None:
    assert sanitize_model_name("My Tree!!") == "My Tree"
    assert sanitize_model_name("a/b\\c") == "a-b-c"


def test_sanitize_rejects_blank_name() -> None:
    with pytest.raises(ModelStoreError, match="cannot be blank"):
        sanitize_model_name("   ")


def test_sanitize_truncates_to_120_chars() -> None:
    assert len(sanitize_model_name("x" * 500)) == 120


# --------------------------------------------------------------------------- #
# Save / list / read / delete round-trip
# --------------------------------------------------------------------------- #


def test_write_then_list_then_read_round_trips(tmp_path) -> None:
    store_path = tmp_path / "store.sqlite3"
    config = _minimal_config()
    name = write_model("First Tree", config, store_path=store_path)
    assert name == "First Tree"
    expected_names = ["First Tree"]
    assert [item["name"] for item in list_saved_models(store_path=store_path)] == expected_names
    assert read_model("First Tree", store_path=store_path) == config


def test_list_is_empty_when_nothing_saved_yet(tmp_path) -> None:
    assert list_saved_models(store_path=tmp_path / "never-written.sqlite3") == []


def test_list_is_sorted_newest_first(tmp_path) -> None:
    import time

    store_path = tmp_path / "store.sqlite3"
    write_model("older", _minimal_config(), store_path=store_path)
    time.sleep(0.02)
    write_model("newer", _minimal_config(), store_path=store_path)
    names = [item["name"] for item in list_saved_models(store_path=store_path)]
    assert names == ["newer", "older"]


def test_read_missing_model_raises_file_not_found(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        read_model("does-not-exist", store_path=tmp_path / "store.sqlite3")


def test_write_overwrites_an_existing_name(tmp_path) -> None:
    store_path = tmp_path / "store.sqlite3"
    write_model("dup", _minimal_config(), store_path=store_path)
    updated = _minimal_config()
    updated["nodes"][0]["instruments"] = ["CCC"]
    write_model("dup", updated, store_path=store_path)
    assert read_model("dup", store_path=store_path) == updated
    assert len(list_saved_models(store_path=store_path)) == 1


def test_delete_removes_the_row_and_returns_its_name(tmp_path) -> None:
    store_path = tmp_path / "store.sqlite3"
    write_model("to-delete", _minimal_config(), store_path=store_path)
    deleted = delete_model("to-delete", store_path=store_path)
    assert deleted == "to-delete"
    assert list_saved_models(store_path=store_path) == []


def test_delete_missing_model_raises_file_not_found(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        delete_model("never-saved", store_path=tmp_path / "store.sqlite3")


# --------------------------------------------------------------------------- #
# Validate-before-write
# --------------------------------------------------------------------------- #


def test_write_rejects_a_non_dict_config(tmp_path) -> None:
    store_path = tmp_path / "store.sqlite3"
    with pytest.raises(ModelStoreError, match="must be an object"):
        write_model("bad", "not a dict", store_path=store_path)  # type: ignore[arg-type]
    assert list_saved_models(store_path=store_path) == []


def test_write_rejects_an_invalid_tree_and_writes_nothing(tmp_path) -> None:
    store_path = tmp_path / "store.sqlite3"
    config = _minimal_config()
    config["nodes"][0]["children"] = ["missing-child"]  # type: ignore[index]
    with pytest.raises(ValueError, match="unknown child id"):
        write_model("invalid", config, store_path=store_path)
    assert list_saved_models(store_path=store_path) == []


def test_write_creates_parent_directories(tmp_path) -> None:
    nested = tmp_path / "a" / "b" / "c" / "store.sqlite3"
    write_model("nested", _minimal_config(), store_path=nested)
    assert nested.is_file()


# --------------------------------------------------------------------------- #
# Legacy *.json migration
# --------------------------------------------------------------------------- #


def test_migrate_legacy_json_models_imports_every_file(tmp_path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    config_a = _minimal_config()
    config_b = _minimal_config()
    config_b["nodes"][0]["instruments"] = ["CCC", "DDD"]
    (models_dir / "Alpha.json").write_text(json.dumps(config_a), encoding="utf-8")
    (models_dir / "Beta.json").write_text(json.dumps(config_b), encoding="utf-8")

    store_path = tmp_path / "store.sqlite3"
    imported = migrate_legacy_json_models(models_dir, store_path=store_path)

    assert sorted(imported) == ["Alpha", "Beta"]
    assert read_model("Alpha", store_path=store_path) == config_a
    assert read_model("Beta", store_path=store_path) == config_b
    # The source files are left untouched.
    assert (models_dir / "Alpha.json").is_file()


def test_migrate_legacy_json_models_on_missing_directory_returns_empty(tmp_path) -> None:
    assert migrate_legacy_json_models(tmp_path / "never-existed", store_path=tmp_path / "store.sqlite3") == []
