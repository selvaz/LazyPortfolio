"""Shared model store: directory resolution, sanitization, validate-before-write.

Both Tree Studio and LazyTools' MCP `portfolio_tree_*` tools go through
`lazyportfolio.v2.store` so a tree saved by one is immediately visible to the
other -- these tests pin the contract that makes that true.
"""

from __future__ import annotations

import pytest

from lazyportfolio.v2.store import (
    ModelStoreError,
    delete_model,
    list_saved_models,
    model_path,
    read_model,
    resolve_models_dir,
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
# Directory resolution precedence
# --------------------------------------------------------------------------- #


def test_explicit_store_dir_wins_over_everything(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LAZYPORTFOLIO_TREE_MODELS_DIR", str(tmp_path / "from-env"))
    explicit = tmp_path / "explicit"
    assert resolve_models_dir(explicit) == explicit.resolve()


def test_env_var_wins_over_computed_default(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LAZYPORTFOLIO_TREE_MODELS_DIR", str(tmp_path))
    assert resolve_models_dir() == tmp_path.resolve()


def test_computed_default_is_repo_reports_tree_studio_models(monkeypatch) -> None:
    monkeypatch.delenv("LAZYPORTFOLIO_TREE_MODELS_DIR", raising=False)
    default = resolve_models_dir()
    assert default.parts[-3:] == ("reports", "tree_studio", "models")


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


def test_model_path_is_traversal_safe(tmp_path) -> None:
    path = model_path("../../etc/passwd", store_dir=tmp_path)
    assert path.parent == tmp_path.resolve()
    assert ".." not in path.parts


# --------------------------------------------------------------------------- #
# Save / list / read / delete round-trip
# --------------------------------------------------------------------------- #


def test_write_then_list_then_read_round_trips(tmp_path) -> None:
    config = _minimal_config()
    path = write_model("First Tree", config, store_dir=tmp_path)
    assert path == tmp_path / "First Tree.json"
    expected = [{"name": "First Tree", "file": "First Tree.json"}]
    assert list_saved_models(store_dir=tmp_path) == expected
    assert read_model("First Tree", store_dir=tmp_path) == config


def test_list_is_empty_when_directory_does_not_exist_yet(tmp_path) -> None:
    assert list_saved_models(store_dir=tmp_path / "never-created") == []


def test_list_is_sorted_newest_first(tmp_path) -> None:
    import time

    write_model("older", _minimal_config(), store_dir=tmp_path)
    time.sleep(0.02)
    write_model("newer", _minimal_config(), store_dir=tmp_path)
    names = [item["name"] for item in list_saved_models(store_dir=tmp_path)]
    assert names == ["newer", "older"]


def test_read_missing_model_raises_file_not_found(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        read_model("does-not-exist", store_dir=tmp_path)


def test_delete_removes_the_file_and_returns_its_path(tmp_path) -> None:
    write_model("to-delete", _minimal_config(), store_dir=tmp_path)
    deleted = delete_model("to-delete", store_dir=tmp_path)
    assert deleted == tmp_path / "to-delete.json"
    assert list_saved_models(store_dir=tmp_path) == []


def test_delete_missing_model_raises_file_not_found(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        delete_model("never-saved", store_dir=tmp_path)


# --------------------------------------------------------------------------- #
# Validate-before-write
# --------------------------------------------------------------------------- #


def test_write_rejects_a_non_dict_config(tmp_path) -> None:
    with pytest.raises(ModelStoreError, match="must be an object"):
        write_model("bad", "not a dict", store_dir=tmp_path)  # type: ignore[arg-type]
    assert list_saved_models(store_dir=tmp_path) == []


def test_write_rejects_an_invalid_tree_and_writes_nothing(tmp_path) -> None:
    config = _minimal_config()
    config["nodes"][0]["children"] = ["missing-child"]  # type: ignore[index]
    with pytest.raises(ValueError, match="unknown child id"):
        write_model("invalid", config, store_dir=tmp_path)
    assert list_saved_models(store_dir=tmp_path) == []
    assert not (tmp_path / "invalid.json").exists()


def test_write_creates_parent_directories(tmp_path) -> None:
    nested = tmp_path / "a" / "b" / "c"
    write_model("nested", _minimal_config(), store_dir=nested)
    assert (nested / "nested.json").is_file()
