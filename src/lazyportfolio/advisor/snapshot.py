"""Public snapshot/fingerprint service (docs/node-advisor-operational-plan.md §6.2).

Moved -- not reimplemented -- from ``project/tree_studio.py``'s private
``_config_hash``/``_data_fingerprint``/``_config_instruments``: those were
each Tree Studio's own copy of logic LazyTools would otherwise have had to
duplicate to agree on the same fingerprint for the same config. This module
is now the one place either caller imports from; ``project/tree_studio.py``
becomes a thin wrapper (see the regression test pinning byte-identical
output before/after the move).

``load_snapshot`` is the "one load" half of the counterfactual evaluator's
one-load/two-solve invariant (§11): baseline and variant must never load
the dataset independently, or a mid-comparison data refresh could make them
silently disagree on what they are comparing.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Any
from uuid import uuid4

from lazyportfolio.advisor.contracts import SnapshotDescriptor
from lazyportfolio.backend import (
    MarketDataHubOptimizationBackend,
    OptimizationDataBackend,
    OptimizationDataset,
)
from lazyportfolio.v2.model import V2Model
from lazyportfolio.v2.store import _as_json


class SnapshotLoadError(ValueError):
    """The data for a config's instrument universe could not be loaded."""


def config_hash(config: dict[str, Any]) -> str:
    """Byte-for-byte identical to the former ``project/tree_studio.py:_config_hash``."""

    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=_as_json)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def config_instruments(model: V2Model) -> list[str]:
    """The de-duplicated instrument set a built V2Model actually references.

    Byte-for-byte identical to the former
    ``project/tree_studio.py:_config_instruments``.
    """

    return list(
        dict.fromkeys(
            [
                *model.root.terminal_instruments(),
                *(node.proxy for node in model.root.walk() if node.proxy),
                *model.benchmark.weights,
            ]
        )
    )


def _coverage_fingerprint(symbols: list[str]) -> tuple[str | None, str]:
    """Query live ``coverage_report`` for ``symbols`` and hash the rows.

    Shared by :func:`data_fingerprint` (which derives ``symbols`` from a
    tree config) and :func:`recompute_snapshot_fingerprint` (which derives
    them from an already-built :class:`~lazyportfolio.advisor.contracts.SnapshotDescriptor`
    -- approval time has no tree config to rebuild a ``V2Model`` from, only
    the descriptor's own ``universe``). Fails soft with a deterministic
    sentinel string, never raises: a proposal drafted before market-data-hub
    was configured (fixture-driven dev/test use) must compare equal to
    itself at approval time, not spuriously block on unrelated
    infrastructure absence.
    """

    if not symbols:
        return None, "no-instruments"
    try:
        from market_data_hub.db.connection import get_conn

        con = get_conn(read_only=True)
        try:
            placeholders = ", ".join("?" for _ in symbols)
            rows = con.execute(
                "SELECT symbol, last_date, obs_count, last_run_id FROM coverage_report "
                f"WHERE upper(symbol) IN ({placeholders}) ORDER BY symbol",
                symbols,
            ).fetchall()
        finally:
            con.close()
    except Exception:
        return None, "coverage-unavailable"
    if not rows:
        return None, "no-coverage"
    as_of = max((str(row[1]) for row in rows if row[1] is not None), default=None)
    canonical = json.dumps([[str(value) for value in row] for row in rows], separators=(",", ":"))
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return as_of, fingerprint


def data_fingerprint(config: dict[str, Any]) -> tuple[str | None, str]:
    """Cheap freshness signal for the instruments a tree config references.

    Byte-for-byte identical to the former
    ``project/tree_studio.py:_data_fingerprint`` -- see that function's
    original docstring for the coverage_report/degradation rationale, which
    still applies unchanged here.
    """

    try:
        model = V2Model.from_config(config)
    except (KeyError, TypeError, ValueError):
        return None, "invalid-config"
    symbols = sorted(
        {
            instrument.split(":", 1)[-1].strip().upper()
            for instrument in config_instruments(model)
            if instrument
        }
    )
    if not symbols:
        return None, "no-instruments"
    return _coverage_fingerprint(symbols)


def recompute_snapshot_fingerprint(snapshot: SnapshotDescriptor) -> str:
    """The real ``recompute_fingerprint`` implementation for approval time
    (docs/adr/0001-node-advisor-architecture.md Fase 2 ``SnapshotService``).

    Re-derives the same coverage-based fingerprint :func:`data_fingerprint`
    computed when the proposal was drafted, from ``snapshot.universe``
    instead of a tree config (approval time has no config to rebuild a
    ``V2Model`` from). Wired at the ``project/advisor/services.py`` layer,
    not as :func:`~lazyportfolio.advisor.approval_service.apply_proposal`'s
    default -- fixture-driven tests that call ``apply_proposal`` directly
    with no market-data-hub configured keep using the Fase 1
    ``_trust_stored_fingerprint`` default deliberately.
    """

    symbols = sorted(
        {
            instrument.split(":", 1)[-1].strip().upper()
            for instrument in snapshot.universe
            if instrument
        }
    )
    _as_of, fingerprint = _coverage_fingerprint(symbols)
    return fingerprint


def load_dataset(
    instruments: list[str],
    data: dict[str, Any],
    currency: str,
    *,
    backend: OptimizationDataBackend | None = None,
) -> OptimizationDataset:
    """Load a complete daily return matrix, converted to ``currency``.

    Same logic as the former ``project/tree_studio.py:_load_instruments``,
    raising :class:`SnapshotLoadError` instead of Tree Studio's own
    ``StudioConfigError`` -- this module has no dependency on ``project/``
    (docs/adr/0001-node-advisor-architecture.md Decision 1), so Tree
    Studio's wrapper catches this and re-raises its own exception type.

    ``backend`` defaults to the real Market Data Hub, but is injectable --
    LazyTools' ``NodeAdvisorReadTools`` (and any test) needs a fake one, the
    same reason ``PortfolioTreeTools``/``PortfolioOptimizationTools`` already
    accept a ``backend`` constructor argument instead of hardcoding one.
    """

    resolved_backend = backend or MarketDataHubOptimizationBackend()
    dataset = resolved_backend.load_returns(
        instruments,
        start=str(data.get("start") or ""),
        end=str(data.get("end") or ""),
        currency=currency,
    )
    missing = [i for i in instruments if i not in dataset.returns.columns]
    if missing:
        display = [instrument.removeprefix("ticker:") for instrument in missing]
        raise SnapshotLoadError("Market Data Hub has no return series for: " + ", ".join(display))
    clean = dataset.returns.dropna(how="any")
    if len(clean) < 3:
        raise SnapshotLoadError("Market Data Hub returned fewer than three complete observations")
    return OptimizationDataset(
        returns=clean, metadata={**dataset.metadata, "complete_rows": len(clean)}
    )


def load_snapshot(
    config: dict[str, Any],
    *,
    field: str = "close",
    frequency: str = "D",
    backend: OptimizationDataBackend | None = None,
) -> tuple[V2Model, OptimizationDataset, SnapshotDescriptor]:
    """Load a config's model, dataset and :class:`SnapshotDescriptor` once.

    The single entry point ``CounterfactualEvaluator`` (§6.3) and, later,
    ``NodeUniverseResolver``'s ``NodeContext.snapshot`` must both go
    through -- baseline and variant solves share the exact same in-memory
    ``OptimizationDataset`` object returned here, never independently
    reloaded ones. ``backend`` is forwarded to :func:`load_dataset`.
    """

    model = V2Model.from_config(config)
    instruments = config_instruments(model)
    raw_data = config.get("data")
    data: dict[str, Any] = raw_data if isinstance(raw_data, dict) else {}
    dataset = load_dataset(instruments, data, model.reference_currency, backend=backend)
    as_of_str, fingerprint = data_fingerprint(config)
    as_of: date | None = None
    if as_of_str is not None:
        try:
            as_of = date.fromisoformat(as_of_str[:10])
        except ValueError:
            as_of = None
    index = dataset.returns.index
    descriptor = SnapshotDescriptor(
        schema_version="1.0",
        source="market-data-hub",
        database_identity=str(dataset.metadata.get("database_identity", uuid4())),
        universe=instruments,
        start=index.min().date() if len(index) else None,
        end=index.max().date() if len(index) else None,
        data_as_of=as_of,
        field=field,
        currency=model.reference_currency,
        frequency=frequency,
        coverage=[],
        source_run_ids=[],
        fingerprint=fingerprint,
    )
    return model, dataset, descriptor


__all__ = [
    "SnapshotLoadError",
    "config_hash",
    "config_instruments",
    "data_fingerprint",
    "load_dataset",
    "load_snapshot",
    "recompute_snapshot_fingerprint",
]
