"""Tree Studio's own model-store/mode wrappers must delegate to the shared
``lazyportfolio.v2.store``/``v2.mode`` modules, not reimplement the logic --
that delegation is what guarantees the GUI and LazyTools' MCP
``portfolio_tree_*`` tools read/write byte-identical rows for the same name.

"""

from __future__ import annotations

import importlib

import pytest
from project import tree_studio as studio_module


@pytest.fixture()
def tree_studio(monkeypatch, tmp_path):
    monkeypatch.setenv("LAZYPORTFOLIO_TREE_DB", str(tmp_path / "store.sqlite3"))
    # Reloaded so the env var is re-read even if another test imported it first.
    return importlib.reload(studio_module)


def _config() -> dict[str, object]:
    return {
        "root_id": "root",
        "currency": "USD",
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


def test_saved_models_reflects_a_tree_written_through_the_store(tree_studio, tmp_path) -> None:
    from lazyportfolio.v2.store import write_model

    write_model("from-the-store", _config(), store_path=tmp_path / "store.sqlite3")
    assert [item["name"] for item in tree_studio._saved_models()] == ["from-the-store"]


def test_saved_models_isolated_per_env_database(tree_studio) -> None:
    # The fixture points LAZYPORTFOLIO_TREE_DB at a fresh, empty tmp_path
    # database -- nothing written by another test (or a prior real Tree
    # Studio session) should leak in here.
    assert tree_studio._saved_models() == []


def test_v2_mode_delegates_to_mode_from_config(tree_studio) -> None:
    assert tree_studio._v2_mode({"backtest": {"forward_enabled": False}}) == "flat"
    forward_config = {"backtest": {"forward_enabled": True, "hierarchy_mode": "proxy"}}
    assert tree_studio._v2_mode(forward_config) == "forward"


def test_v2_mode_reraises_as_studio_config_error(tree_studio) -> None:
    reason = "iterative mode is intentionally disabled"
    with pytest.raises(tree_studio.StudioConfigError, match=reason):
        tree_studio._v2_mode({"backtest": {"hierarchy_mode": "current_root_synthetic"}})
