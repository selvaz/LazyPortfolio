"""docs/node-copilot-operational-plan.md §13 Fase 2 exit criterion: Tree
Studio and LazyTools must emit the identical fingerprint for the same
config. ``project/tree_studio.py``'s ``_config_hash``/``_data_fingerprint``/
``_config_instruments``/``_load_instruments`` were moved (not reimplemented)
into ``lazyportfolio.copilot.snapshot`` -- these tests pin that the module
script's wrappers still delegate, byte-for-byte, rather than having drifted
back into a second copy of the logic.

``project/tree_studio.py`` is a script, not an installed package -- same
sys.path pattern as ``tests/test_tree_studio_cache_freshness.py``.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

from lazyportfolio.copilot import snapshot

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_DIR = REPO_ROOT / "project"


def _config() -> dict[str, Any]:
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


@pytest.fixture()
def tree_studio_module():
    sys.path.insert(0, str(PROJECT_DIR))
    try:
        module = importlib.import_module("tree_studio")
        yield importlib.reload(module)
    finally:
        sys.path.remove(str(PROJECT_DIR))


def test_config_hash_matches_a_hand_computed_golden_vector() -> None:
    """Pins the exact scheme (sorted keys, compact separators, sha256) so a
    change to lazyportfolio.v2.store._as_json's float/Decimal handling
    doesn't silently change every existing cache key/run_history row."""

    config = {"b": 2, "a": [1, 2, 3]}
    # sha256('{"a":[1,2,3],"b":2}') -- computed once, not re-derived from
    # the function under test.
    assert snapshot.config_hash(config) == (
        "17df395fb77661fb2f96417b64819b03367b9a00303e18b0445ac09534f134e1"
    )


def test_config_hash_is_stable_regardless_of_key_insertion_order() -> None:
    config = {"b": 2, "a": [1, 2, 3]}
    reordered = {"a": [1, 2, 3], "b": 2}
    assert snapshot.config_hash(config) == snapshot.config_hash(reordered)


def test_tree_studio_config_hash_delegates_to_snapshot_service(tree_studio_module) -> None:
    config = _config()
    assert tree_studio_module._config_hash(config) == snapshot.config_hash(config)


def test_tree_studio_data_fingerprint_delegates_to_snapshot_service(tree_studio_module) -> None:
    config = _config()
    assert tree_studio_module._data_fingerprint(config) == snapshot.data_fingerprint(config)


def test_tree_studio_config_instruments_delegates_to_snapshot_service(
    tree_studio_module,
) -> None:
    model = tree_studio_module.V2Model.from_config(_config())
    assert tree_studio_module._config_instruments(model) == snapshot.config_instruments(model)


def test_tree_studio_load_instruments_translates_snapshot_load_error(
    tree_studio_module, monkeypatch
) -> None:
    def _raise(*args: Any, **kwargs: Any) -> Any:
        raise snapshot.SnapshotLoadError("no data for this universe")

    monkeypatch.setattr(snapshot, "load_dataset", _raise)
    with pytest.raises(tree_studio_module.StudioConfigError, match="no data for this universe"):
        tree_studio_module._load_instruments(["ticker:AAA"], {}, "USD")


def test_invalid_config_returns_the_same_sentinel_both_ways(tree_studio_module) -> None:
    broken = {"not": "a valid tree config"}
    assert tree_studio_module._data_fingerprint(broken) == (None, "invalid-config")
    assert snapshot.data_fingerprint(broken) == (None, "invalid-config")
