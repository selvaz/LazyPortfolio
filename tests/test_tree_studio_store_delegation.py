"""Tree Studio's own model-store/mode wrappers must delegate to the shared
``lazyportfolio.v2.store``/``v2.mode`` modules, not reimplement the logic --
that delegation is what guarantees the GUI and LazyTools' MCP
``portfolio_tree_*`` tools read/write byte-identical rows for the same name.

``project/tree_studio.py`` is a script, not an installed package, so it is
imported the same way ``project/tree_studio_v2/validate_exports.py`` already
does elsewhere in this repo: put ``project/`` on ``sys.path`` first.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_DIR = REPO_ROOT / "project"


@pytest.fixture()
def tree_studio(monkeypatch, tmp_path):
    monkeypatch.setenv("LAZYPORTFOLIO_TREE_DB", str(tmp_path / "store.sqlite3"))
    sys.path.insert(0, str(PROJECT_DIR))
    try:
        module = importlib.import_module("tree_studio")
        yield importlib.reload(module)  # re-read the env var if already imported by another test
    finally:
        sys.path.remove(str(PROJECT_DIR))


def _config() -> dict[str, object]:
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
