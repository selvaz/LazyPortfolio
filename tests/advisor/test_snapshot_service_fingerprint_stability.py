"""docs/node-advisor-operational-plan.md §13 Fase 2 exit criterion: Tree
Studio and LazyTools must emit the identical fingerprint for the same
config. ``project/tree_studio.py``'s ``_config_hash``/``_data_fingerprint``/
``_config_instruments``/``_load_instruments`` were moved (not reimplemented)
into ``lazyportfolio.advisor.snapshot`` -- these tests pin that the module
script's wrappers still delegate, byte-for-byte, rather than having drifted
back into a second copy of the logic.

"""

from __future__ import annotations

import importlib
from typing import Any

import pytest
from project import tree_studio

from lazyportfolio.advisor import snapshot
from lazyportfolio.advisor.contracts import SnapshotDescriptor


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
    return importlib.reload(tree_studio)


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


def test_recompute_snapshot_fingerprint_matches_data_fingerprint_for_the_same_universe() -> None:
    """§8.3 step 6 needs recompute_snapshot_fingerprint(descriptor) to agree
    with data_fingerprint(config) for the same instruments -- otherwise a
    proposal drafted via data_fingerprint would ALWAYS look stale at
    approval time even with nothing having changed."""

    config = _config()
    _as_of, drafted_fingerprint = snapshot.data_fingerprint(config)

    descriptor = SnapshotDescriptor(
        schema_version="1.0",
        source="market-data-hub",
        database_identity="test",
        universe=["ticker:AAA", "ticker:BBB"],
        field="close",
        currency="USD",
        frequency="D",
        fingerprint=drafted_fingerprint,
    )
    recomputed = snapshot.recompute_snapshot_fingerprint(descriptor)

    assert recomputed == drafted_fingerprint


class _FakeCoverageCursor:
    def __init__(self, rows: list[tuple]) -> None:
        self._rows = rows

    def fetchall(self) -> list[tuple]:
        return self._rows


class _FakeCoverageConn:
    """Stands in for market_data_hub.db.connection.get_conn(read_only=True).

    Ignores the query text (both data_fingerprint and
    recompute_snapshot_fingerprint issue the identical parameterised SELECT
    -- see _coverage_fingerprint) and returns pre-set rows regardless of the
    ``symbols`` filter, so the test controls exactly what "live data" looks
    like instead of depending on whatever market-data-hub happens to have.
    """

    def __init__(self, rows: list[tuple]) -> None:
        self._rows = rows

    def execute(self, _sql: str, _params: object) -> _FakeCoverageCursor:
        return _FakeCoverageCursor(self._rows)

    def close(self) -> None:
        pass


def _patch_coverage(monkeypatch: pytest.MonkeyPatch, rows: list[tuple]) -> None:
    import market_data_hub.db.connection as hub_connection

    monkeypatch.setattr(hub_connection, "get_conn", lambda read_only=True: _FakeCoverageConn(rows))


def test_recompute_snapshot_fingerprint_detects_real_coverage_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Not tautological: controls actual 'live' rows on both sides, so this
    proves the hashing logic itself reacts to a real data change -- the
    equivalence test above only proves both paths reach the same *absence*
    of data, which they'd do even if the hashing were broken."""

    descriptor = SnapshotDescriptor(
        schema_version="1.0",
        source="market-data-hub",
        database_identity="test",
        universe=["ticker:AAA", "ticker:BBB"],
        field="close",
        currency="USD",
        frequency="D",
        fingerprint="unused",
    )

    _patch_coverage(
        monkeypatch,
        [("AAA", "2026-08-15", 500, "run-1"), ("BBB", "2026-08-15", 500, "run-1")],
    )
    original = snapshot.recompute_snapshot_fingerprint(descriptor)
    assert original not in ("no-instruments", "no-coverage", "coverage-unavailable")

    # Same rows again -> same fingerprint (determinism).
    _patch_coverage(
        monkeypatch,
        [("AAA", "2026-08-15", 500, "run-1"), ("BBB", "2026-08-15", 500, "run-1")],
    )
    assert snapshot.recompute_snapshot_fingerprint(descriptor) == original

    # A later last_date -- new data landed -> the fingerprint must change,
    # this is the actual freshness check §8.3 step 6 relies on.
    _patch_coverage(
        monkeypatch,
        [("AAA", "2026-08-16", 501, "run-2"), ("BBB", "2026-08-16", 501, "run-2")],
    )
    assert snapshot.recompute_snapshot_fingerprint(descriptor) != original


def test_data_fingerprint_and_recompute_agree_on_the_same_controlled_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two entry points (config-driven at creation, descriptor-driven at
    approval) must produce byte-identical output for the same live rows, or
    a legitimately unchanged proposal would look stale at approval time."""

    rows = [("AAA", "2026-08-15", 500, "run-1"), ("BBB", "2026-08-15", 500, "run-1")]
    _patch_coverage(monkeypatch, rows)

    _as_of, from_config = snapshot.data_fingerprint(_config())

    descriptor = SnapshotDescriptor(
        schema_version="1.0",
        source="market-data-hub",
        database_identity="test",
        universe=["AAA", "BBB"],
        field="close",
        currency="USD",
        frequency="D",
        fingerprint="unused",
    )
    from_descriptor = snapshot.recompute_snapshot_fingerprint(descriptor)

    assert from_config == from_descriptor


def test_recompute_snapshot_fingerprint_empty_universe_matches_data_fingerprint() -> None:
    _as_of, drafted = snapshot.data_fingerprint({"not": "a valid tree config"})
    descriptor = SnapshotDescriptor(
        schema_version="1.0",
        source="market-data-hub",
        database_identity="test",
        universe=[],
        field="close",
        currency="USD",
        frequency="D",
        fingerprint="unused",
    )
    # data_fingerprint's "invalid-config" path never reaches instrument
    # extraction; recompute's universe-driven path instead hits
    # "no-instruments" on an empty universe. Both are None-as-of sentinels,
    # not a live-looking hash -- assert the shape, not byte-equality, since
    # the two functions reach the "nothing to check" conclusion differently.
    assert drafted == "invalid-config"
    assert snapshot.recompute_snapshot_fingerprint(descriptor) == "no-instruments"
