"""Local visual editor and runner for LazyPortfolio V2 hierarchical allocation trees.

Run with ``python project/tree_studio.py`` and open the URL printed by the
server.  The application deliberately binds to localhost: it is a workstation
tool and it can access the user's local Market Data Hub database.

Tree Studio is V2-only.  There is one hierarchical engine
(``lazyportfolio.hierarchical_v2``); it does not call any legacy
allocation tree, backend, or backtester.

A daily production allocation is a call to ``/api/v2/estimate`` (a single
point-in-time solve) -- it never touches ``_run_full_backtest`` or
``_v2_export_artifacts``, so it never builds the walk-forward fold ledger,
the audit ZIP, or the client HTML report. There is no separate
daily-allocation endpoint or job abstraction; reuse this one.
"""

# ruff: noqa: E501

from __future__ import annotations

import gzip
import hashlib
import json
import mimetypes
import re
import sys
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any, Literal
from urllib.parse import parse_qs, unquote, urlparse

from tree_studio_v2.exports import build_audit_bundle, build_client_report

from lazyportfolio.artifact_registry import register_report_artifact
from lazyportfolio.backend import MarketDataHubOptimizationBackend, OptimizationDataset
from lazyportfolio.calendar import _annualization_factor, _resample_simple_returns
from lazyportfolio.hierarchical_v2 import (
    HierarchicalV2Backtester,
    HierarchicalV2Estimator,
    V2Model,
    V2OptimizationError,
)
from lazyportfolio.scientific_study import (
    ScientificStudyProtocol,
    ScientificStudyResult,
    run_scientific_study,
)
from lazyportfolio.v2 import run_history as _run_history
from lazyportfolio.v2 import store as _store
from lazyportfolio.v2.mode import mode_from_config as _mode_from_config
from lazyportfolio.v2.store import _as_json

APP_DIR = Path(__file__).resolve().parent
INDEX_FILE = APP_DIR / "tree_studio.html"
_TICKER = re.compile(r"^[A-Za-z0-9.\-]+$")
#: Used only for export filename stems (audit ZIP / client report) below --
#: model persistence itself goes through lazyportfolio.v2.store, which owns
#: the equivalent pattern for saved-model names.
_MODEL_NAME = re.compile(r"[^A-Za-z0-9._ -]+")


class StudioConfigError(ValueError):
    """A configuration cannot be represented by the V2 contract."""


def _saved_models() -> list[dict[str, str]]:
    return _store.list_saved_models()


def _config_instruments(model: V2Model) -> list[str]:
    """The de-duplicated instrument set a built V2Model actually references."""
    return list(
        dict.fromkeys(
            [
                *model.root.terminal_instruments(),
                *(node.proxy for node in model.root.walk() if node.proxy),
                *model.benchmark.weights,
            ]
        )
    )


def _v2_inputs(config: dict[str, Any]) -> tuple[V2Model, OptimizationDataset]:
    model = V2Model.from_config(config)
    data = config.get("data") if isinstance(config.get("data"), dict) else {}
    return model, _load_instruments(_config_instruments(model), data, model.reference_currency)


def _config_hash(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=_as_json)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _data_fingerprint(config: dict[str, Any]) -> tuple[str | None, str]:
    """Cheap freshness signal for the instruments a tree config references.

    Reads Market Data Hub's ``coverage_report`` (symbol, last_date,
    obs_count, last_run_id) -- no price history is loaded -- so this stays
    cheap enough to run on every cache lookup, before deciding whether to
    reuse a cached result. A MarketDataHub refresh that touches any
    referenced instrument changes the fingerprint (and therefore the cache
    key derived from it), so a result computed before the refresh is never
    served as if it were still current.

    Degrades to a constant fallback (never raises) when market-data-hub
    isn't installed or its DB isn't reachable, matching this module's
    existing best-effort MDH-metadata pattern (see ``instrument_labels``).
    """
    try:
        model = V2Model.from_config(config)
    except (KeyError, TypeError, ValueError):
        return None, "invalid-config"
    symbols = sorted(
        {
            instrument.split(":", 1)[-1].strip().upper()
            for instrument in _config_instruments(model)
            if instrument
        }
    )
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


def _cache_key(path: str, config_hash: str, data_fingerprint: str) -> str:
    return f"{path}:{config_hash}:{data_fingerprint}"


def _tree_id_for_config(config: dict[str, Any]) -> str | None:
    """Best-effort link back to a saved tree.

    Only set when this exact config (canonicalized) byte-for-byte matches a
    currently-saved tree under the name its root node carries -- an ad-hoc,
    unsaved, or since-edited config stays unlinked (``None``) rather than
    guessing which saved tree it might correspond to.
    """
    nodes = config.get("nodes") if isinstance(config.get("nodes"), list) else []
    root_id = str(config.get("root_id") or "")
    root = next((node for node in nodes if str(node.get("id")) == root_id), None)
    root_name = str((root or {}).get("name") or "").strip()
    if not root_name:
        return None
    try:
        saved = _store.read_model(root_name)
    except FileNotFoundError:
        return None

    def canonical(value: dict[str, Any]) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_as_json)

    if canonical(config) != canonical(saved):
        return None
    return _store.sanitize_model_name(root_name)


def _run_summary_fields(path: str, payload: dict[str, Any]) -> tuple[Any, Any]:
    """Extract (weights, metrics) from a producer's response for the run_history row."""
    if path == "/api/v2/estimate":
        return payload.get("terminal_weights"), None
    if path == "/api/v2/backtest":
        return None, (payload.get("report") or {}).get("metrics")
    if path == "/api/v2/scientific-study":
        return None, payload.get("metrics")
    return None, None


def _load_instruments(
    instruments: list[str], data: dict[str, Any], currency: str
) -> OptimizationDataset:
    """Load a complete daily return matrix, converted to ``currency``, and
    identify missing series clearly."""
    dataset = MarketDataHubOptimizationBackend().load_returns(
        instruments,
        start=str(data.get("start") or ""),
        end=str(data.get("end") or ""),
        currency=currency,
    )
    missing = [instrument for instrument in instruments if instrument not in dataset.returns.columns]
    if missing:
        display = [instrument.removeprefix("ticker:") for instrument in missing]
        raise StudioConfigError(
            "Market Data Hub has no return series for: " + ", ".join(display)
        )
    clean = dataset.returns.dropna(how="any")
    if len(clean) < 3:
        raise StudioConfigError("Market Data Hub returned fewer than three complete observations")
    return OptimizationDataset(returns=clean, metadata={**dataset.metadata, "complete_rows": len(clean)})


def _scientific_study_settings(config: dict[str, Any]) -> dict[str, Any] | None:
    """Return the study's settings dict, or None when the flag is off/absent.

    This is an app-level flag (``backtest.scientific_study``), not part of the
    canonical V2 contract validated by ``V2Model.from_config`` -- it only
    decides whether Tree Studio ALSO runs the offline bootstrap-significance
    harness (``lazyportfolio.scientific_study``) alongside the normal
    backtest, so nothing in ``lazyportfolio.v2`` needs to know about it.
    """

    backtest = config.get("backtest") if isinstance(config.get("backtest"), dict) else {}
    settings = backtest.get("scientific_study")
    if not isinstance(settings, dict) or not settings.get("enabled"):
        return None
    return settings


def _scientific_study_result(
    config: dict[str, Any],
    model: V2Model,
    dataset: OptimizationDataset,
    mode: str,
) -> ScientificStudyResult | None:
    settings = _scientific_study_settings(config)
    if settings is None:
        return None
    backtest = config["backtest"]
    protocol = ScientificStudyProtocol(
        train_size=int(backtest.get("train_size") or 104),
        estimation_frequency=str(backtest.get("estimation_frequency") or "W"),
        rebalance_frequency=str(backtest.get("rebalance_frequency") or "M"),
        transaction_cost_bps=float(backtest.get("transaction_cost_bps") or 0),
        bootstrap_samples=int(settings.get("bootstrap_samples") or 2_000),
        bootstrap_block_size=int(settings.get("bootstrap_block_size") or 20),
        random_seed=int(settings.get("random_seed") or 7),
    )
    return run_scientific_study(model, dataset.returns, mode=mode, protocol=protocol)


def _v2_scientific_study_payload(config: dict[str, Any]) -> dict[str, Any]:
    model, dataset = _v2_inputs(config)
    mode = _v2_mode(config)
    result = _scientific_study_result(config, model, dataset, mode)
    if result is None:
        raise StudioConfigError(
            "scientific study is not enabled: set backtest.scientific_study.enabled = true"
        )
    return {
        "ok": True,
        "engine": "scientific-study",
        "mode": mode,
        "fold_count": result.fold_count,
        "common_oos_start": result.common_oos_start,
        "common_oos_end": result.common_oos_end,
        "metrics": result.metrics,
        "comparisons": [asdict(item) for item in result.comparisons],
        "dropped_observations": result.dropped_observations,
    }


def _v2_mode(config: dict[str, Any]) -> str:
    """Thin wrapper: derivation itself lives in ``lazyportfolio.v2.mode`` so
    Tree Studio and any other caller (LazyTools' MCP ``portfolio_tree_*``
    tools) can never compute a different mode for the same config."""
    try:
        return _mode_from_config(config)
    except ValueError as exc:
        raise StudioConfigError(str(exc)) from exc


def _v2_node_payload(results: dict[str, Any]) -> dict[str, Any]:
    return {
        name: {
            "local_weights": result.local_weights,
            "terminal_weights": result.terminal_weights,
            "audit": asdict(result.audit),
        }
        for name, result in results.items()
    }


def _v2_estimate_payload(config: dict[str, Any]) -> dict[str, Any]:
    model, dataset = _v2_inputs(config)
    backtest = config["backtest"]
    estimation_frequency = str(backtest.get("estimation_frequency") or "W")
    estimation = _resample_simple_returns(dataset.returns, estimation_frequency)
    train_size = int(backtest.get("train_size") or 104)
    train = estimation.tail(train_size)
    if len(train) < train_size:
        raise StudioConfigError("not enough observations for the V2 estimation window")
    mode = _v2_mode(config)
    estimate = HierarchicalV2Estimator().estimate(
        model,
        train,
        mode=mode,
        periods_per_year=_annualization_factor(estimation_frequency),
    )
    return {
        "ok": True,
        "engine": "hierarchical-v2",
        "mode": mode,
        "reference_policy": "immutable_raw",
        "metadata": {
            **dataset.metadata,
            "estimation_start": train.index.min(),
            "estimation_end": train.index.max(),
            "estimation_observations": len(train),
        },
        "terminal_weights": estimate.terminal_weights,
        "synthetic_benchmark_weights": estimate.synthetic_benchmark_weights,
        "nodes": _v2_node_payload(estimate.node_results),
        "forward_nodes": _v2_node_payload(estimate.forward_node_results),
    }


#: Raw (non-JSON) cache for the one canonical walk-forward backtest per
#: config -- shared by /api/v2/backtest AND the report/audit export path, so
#: clicking "Report" after "Backtest" reuses the same already-computed
#: V2BacktestReport instead of re-running the whole walk-forward loop a
#: second time. In-memory only (pandas Series/dataclasses inside don't round
#: -trip through JSON), separate from the JSON _run_cache on StudioHandler.
_raw_backtest_cache: dict[str, tuple[V2Model, OptimizationDataset, Any]] = {}
_raw_backtest_cache_lock = Lock()
_RAW_BACKTEST_CACHE_LIMIT = 8


def _raw_backtest_key(config: dict[str, Any], *, capture_audit_series: bool) -> str:
    _, data_fingerprint = _data_fingerprint(config)
    path = "/api/v2/raw-backtest" + ("/with-series" if capture_audit_series else "")
    return _cache_key(path, _config_hash(config), data_fingerprint)


def _run_full_backtest(
    config: dict[str, Any], *, capture_audit_series: bool
) -> tuple[V2Model, OptimizationDataset, Any]:
    """Run (or reuse) THE walk-forward backtest for this config.

    ``capture_audit_series`` is part of the cache key: a plain backtest view
    never reads per-fold estimation series (only the audit ZIP export does,
    see ``tree_studio_v2/exports.py``'s ``build_audit_bundle``), so it's
    requested with ``capture_audit_series=False`` and gets a cheaper,
    separately-cached run -- a later audit-bundle request recomputes once
    with series captured rather than paying that extra bookkeeping on every
    backtest view. Keyed on the exact config AND a Market Data Hub freshness
    fingerprint (same policy as ``_cache_key``), so any change to the tree,
    or a data refresh, both invalidate the cache naturally instead of
    silently serving a pre-refresh result for the rest of the process's
    lifetime.
    """
    key = _raw_backtest_key(config, capture_audit_series=capture_audit_series)
    with _raw_backtest_cache_lock:
        cached = _raw_backtest_cache.get(key)
    if cached is not None:
        return cached
    model, dataset = _v2_inputs(config)
    backtest = config["backtest"]
    mode = _v2_mode(config)
    report = HierarchicalV2Backtester().run(
        model,
        dataset.returns,
        mode=mode,
        train_size=int(backtest.get("train_size") or 104),
        estimation_frequency=str(backtest.get("estimation_frequency") or "W"),
        rebalance_frequency=str(backtest.get("rebalance_frequency") or "M"),
        transaction_cost_bps=float(backtest.get("transaction_cost_bps") or 0),
        include_partial_last_period=bool(backtest.get("include_partial_last_period", False)),
        capture_audit_series=capture_audit_series,
    )
    result = (model, dataset, report)
    with _raw_backtest_cache_lock:
        if len(_raw_backtest_cache) >= _RAW_BACKTEST_CACHE_LIMIT and key not in _raw_backtest_cache:
            _raw_backtest_cache.pop(next(iter(_raw_backtest_cache)))
        _raw_backtest_cache[key] = result
    return result


def _v2_backtest_payload(config: dict[str, Any]) -> dict[str, Any]:
    model, dataset, report = _run_full_backtest(config, capture_audit_series=False)
    mode = _v2_mode(config)
    curves = {name: _chart_curve(series) for name, series in report.curves.items()}
    return {
        "ok": True,
        "engine": "hierarchical-v2",
        "report": {
            "mode": mode,
            "reference_policy": "immutable_raw",
            "n_folds": len(report.folds),
            "oos_start": report.curves["FINAL"].index.min(),
            "oos_end": report.curves["FINAL"].index.max(),
            "metrics": report.metrics,
            "curves": curves,
            "folds": report.folds,
            "transaction_cost_paid": report.transaction_cost_paid,
            "node_names": [node.name for node in model.root.walk()],
            "father_arms": {
                node.name: f"FATHER:{node.name}"
                for node in model.root.walk()
                if node.proxy is not None
            },
        },
    }


def _v2_export_artifacts(
    config: dict[str, Any], *, kind: Literal["audit", "report"]
) -> dict[str, tuple[bytes, str, str]]:
    """Build only the requested artifact -- a client-report request must
    never pay for (or even construct) the audit ZIP, and vice versa. Only
    ``kind == "audit"`` needs per-fold estimation series, so that's the only
    case that asks ``_run_full_backtest`` for the more expensive capture.
    """
    model, dataset, report = _run_full_backtest(config, capture_audit_series=(kind == "audit"))
    backtest = config["backtest"]
    mode = _v2_mode(config)
    estimation_frequency = str(backtest.get("estimation_frequency") or "W")
    train_size = int(backtest.get("train_size") or 104)
    estimation = _resample_simple_returns(dataset.returns, estimation_frequency)
    train = estimation.tail(train_size)
    if len(train) < train_size:
        raise StudioConfigError("not enough observations for the V2 export estimation window")
    estimate = HierarchicalV2Estimator().estimate(
        model,
        train,
        mode=mode,
        periods_per_year=_annualization_factor(estimation_frequency),
    )
    root = next(node for node in config["nodes"] if str(node["id"]) == str(config["root_id"]))
    stem = _MODEL_NAME.sub("-", str(root.get("name") or "hierarchical-model")).strip(" .-")
    stem = stem or "hierarchical-model"
    scientific_study = _scientific_study_result(config, model, dataset, mode)
    if kind == "audit":
        audit = build_audit_bundle(
            config=config,
            data_metadata=dataset.metadata,
            daily_returns=dataset.returns,
            estimate=estimate,
            report=report,
            scientific_study=scientific_study,
        )
        return {"audit": (audit, "application/zip", f"{stem}-v2-audit.zip")}
    client = build_client_report(
        config=config,
        data_metadata=dataset.metadata,
        estimate=estimate,
        report=report,
        scientific_study=scientific_study,
    )
    return {"report": (client, "text/html; charset=utf-8", f"{stem}-v2-report.html")}


def _report_artifact_fields(config: dict[str, Any]) -> tuple[str, str]:
    """Derive a human-recognizable title/summary for a generated Tree Studio
    report directly from the tree config.

    There is no session/date concept here -- report generation is on-demand
    and human-triggered, not a scheduled/dated batch job -- so this uses the
    root node's name instead (the same lookup ``_v2_export_artifacts`` above
    already does to derive the export filename stem), plus the tree's node
    names and instruments for a keyword-dense summary (``search_artifacts``
    only ever matches title/summary/tags, never the stored HTML content).
    """
    nodes = config.get("nodes") if isinstance(config.get("nodes"), list) else []
    root_id = str(config.get("root_id") or "")
    root = next((node for node in nodes if str(node.get("id")) == root_id), None)
    root_name = str((root or {}).get("name") or "hierarchical-model")
    title = f"Tree Studio report: {root_name}"

    node_names = [str(node.get("name") or node.get("id") or "") for node in nodes if isinstance(node, dict)]
    instruments: list[str] = []
    for node in nodes:
        if isinstance(node, dict):
            instruments.extend(str(item) for item in (node.get("instruments") or []) if item)
    backtest = config.get("backtest") if isinstance(config.get("backtest"), dict) else {}
    benchmark = backtest.get("benchmark") if isinstance(backtest.get("benchmark"), dict) else {}
    weights = benchmark.get("weights") if isinstance(benchmark.get("weights"), dict) else {}
    instruments.extend(str(symbol) for symbol in weights)
    unique_instruments = list(dict.fromkeys(instruments))
    summary = (
        f"Tree Studio HTML report for '{root_name}' "
        f"({len(nodes)} nodes: {', '.join(node_names) or 'n/a'}); "
        f"instruments: {', '.join(unique_instruments) or 'n/a'}"
    )
    return title, summary


def _chart_curve(series: Any, *, max_points: int = 600) -> list[dict[str, float | str]]:
    """Compress daily simple returns without changing the compounded curve."""
    if series.empty:
        return []
    step = max(1, -(-len(series) // max_points))
    points: list[dict[str, float | str]] = []
    for start in range(0, len(series), step):
        chunk = series.iloc[start : start + step]
        points.append(
            {
                "date": str(chunk.index[-1].date()),
                "return": float((1.0 + chunk).prod() - 1.0),
            }
        )
    return points


def _v2_validation_payload(config: dict[str, Any]) -> dict[str, Any]:
    """Build the V2 model to report required instruments and a forward-mode check."""
    try:
        model = V2Model.from_config(config)
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "ok": True,
            "warnings": [],
            "instruments": [],
            "nested_v0": {
                "eligible": False,
                "reason": str(exc),
                "requirements": [
                    "a root node and at least one child",
                    "every child node needs a proxy ticker",
                    "the backtest benchmark weights must be declared",
                ],
            },
        }
    instruments = list(
        dict.fromkeys(
            [
                *model.root.terminal_instruments(),
                *(node.proxy for node in model.root.walk() if node.proxy),
                *model.benchmark.weights,
            ]
        )
    )
    children = model.root.children
    return {
        "ok": True,
        "warnings": [],
        "instruments": instruments,
        "nested_v0": {
            "eligible": bool(children),
            "reason": (
                "Configurazione valida per Forward."
                if children
                else "Il nodo root non ha nodi figli da espandere in Forward."
            ),
            "requirements": [],
            "level_0_components": model.root.instruments,
            "sleeves": [
                {
                    "node": child.name,
                    "proxy": child.proxy,
                    "instruments": child.instruments,
                    "terminal_instruments": child.terminal_instruments(),
                }
                for child in children
            ],
        },
    }


def sample_config() -> dict[str, Any]:
    return {
        "root_id": "root",
        "currency": "USD",
        "nodes": [
            {
                "id": "root",
                "name": "Global allocation",
                "children": ["equity", "bonds"],
                "instruments": [],
                "proxy": "",
                "goal": {"objective": "min_risk"},
                "constraints": {},
            },
            {
                "id": "equity",
                "name": "Equity",
                "children": [],
                "instruments": ["SPY", "VGK", "EWJ"],
                "proxy": "ACWI",
                "goal": {"objective": "min_risk"},
                "constraints": {},
            },
            {
                "id": "bonds",
                "name": "Bonds",
                "children": [],
                "instruments": ["SHY", "IEF", "TLT"],
                "proxy": "AGG",
                "goal": {"objective": "min_risk"},
                "constraints": {},
            },
        ],
        "data": {"start": "2018-01-01", "end": ""},
        "backtest": {
            "id": "tree-studio",
            "train_size": 104,
            "rebalance_frequency": "M",
            "estimation_frequency": "W",
            "transaction_cost_bps": "5",
            "forward_enabled": True,
            "hierarchy_mode": "proxy",
            "benchmark": {"id": "B0", "name": "Global 70/30", "weights": {"ACWI": "0.7", "AGG": "0.3"}},
        },
    }


def minimal_sample_config() -> dict[str, Any]:
    """A smaller, two-node configuration for a quick first run."""
    return {
        "root_id": "root",
        "currency": "USD",
        "nodes": [
            {
                "id": "root",
                "name": "70/30 global",
                "children": ["equity"],
                "instruments": ["AGG"],
                "proxy": "",
                "goal": {"objective": "min_risk"},
                "constraints": {},
            },
            {
                "id": "equity",
                "name": "Equity sleeve",
                "children": [],
                "instruments": ["SPY", "VGK", "VWO"],
                "proxy": "ACWI",
                "goal": {"objective": "min_risk"},
                "constraints": {},
            },
        ],
        "data": {"start": "2015-01-01", "end": ""},
        "backtest": {
            "id": "tree-studio-minimal",
            "train_size": 104,
            "rebalance_frequency": "M",
            "estimation_frequency": "W",
            "transaction_cost_bps": "5",
            "forward_enabled": True,
            "hierarchy_mode": "proxy",
            "benchmark": {"id": "B0", "name": "ACWI / AGG 70/30", "weights": {"ACWI": "0.7", "AGG": "0.3"}},
        },
    }


def instrument_catalog(
    query: str,
    *,
    limit: int = 30,
    min_observations: int = 0,
    start_before: str = "",
) -> list[dict[str, Any]]:
    """Read active listings from Market Data Hub's identity tables.

    This is intentionally a read-only lookup. The editor receives only the
    metadata needed to choose an instrument, never a copied static universe.
    """
    text = query.strip()
    minimum = max(0, int(min_observations))
    cutoff = start_before.strip()
    if len(text) < 2 and not minimum and not cutoff:
        return []
    bounded_limit = max(1, min(int(limit), 100))
    pattern = f"%{text.upper()}%"
    try:
        from market_data_hub.db.connection import get_conn

        con = get_conn(read_only=True)
        try:
            rows = con.execute(
                """
                SELECT l.symbol, coalesce(i.name, ''), coalesce(i.kind, ''),
                       coalesce(l.exchange, ''), coalesce(l.currency, ''),
                       coalesce(c.asset_class, ''), coalesce(c.area, ''),
                       coalesce(c.category, ''), coalesce(c.sub_group, ''),
                       coalesce(c.sector, ''), coalesce(c.theme, ''),
                       coalesce(c.benchmark_proxy, ''), c.priority,
                       coalesce(p.observations, 0), p.start_date, p.end_date
                FROM listings l
                JOIN instruments i USING (instrument_id)
                LEFT JOIN etf_classification c ON upper(c.symbol) = upper(l.symbol)
                LEFT JOIN (
                    SELECT listing_id, count(*) AS observations,
                           min(date) AS start_date, max(date) AS end_date
                    FROM prices_daily
                    GROUP BY listing_id
                ) p USING (listing_id)
                WHERE l.active_to IS NULL
                  AND coalesce(p.observations, 0) >= ?
                  AND (? = '' OR p.start_date <= CAST(? AS DATE))
                  AND (
                      upper(l.symbol) LIKE ? OR upper(coalesce(i.name, '')) LIKE ?
                      OR upper(coalesce(c.asset_class, '')) LIKE ?
                      OR upper(coalesce(c.area, '')) LIKE ?
                      OR upper(coalesce(c.category, '')) LIKE ?
                      OR upper(coalesce(c.sub_group, '')) LIKE ?
                      OR upper(coalesce(c.sector, '')) LIKE ?
                      OR upper(coalesce(c.theme, '')) LIKE ?
                  )
                ORDER BY CASE WHEN upper(l.symbol) = ? THEN 0 ELSE 1 END,
                         l.symbol, l.exchange
                LIMIT ?
                """,
                [minimum, cutoff, cutoff,
                 pattern, pattern, pattern, pattern, pattern, pattern, pattern, pattern,
                 text.upper(), bounded_limit],
            ).fetchall()
        finally:
            con.close()
    except Exception as exc:
        raise StudioConfigError(f"Market Data Hub catalog is unavailable: {type(exc).__name__}: {exc}") from exc
    return [
        {
            "symbol": str(symbol), "name": str(name), "kind": str(kind),
            "exchange": str(exchange), "currency": str(currency),
            "asset_class": str(asset_class), "area": str(area),
            "category": str(category), "sub_group": str(sub_group),
            "sector": str(sector), "theme": str(theme),
            "benchmark_proxy": str(benchmark_proxy), "priority": priority,
            "observations": int(observations or 0),
            "start_date": str(start_date or ""), "end_date": str(end_date or ""),
        }
        for (symbol, name, kind, exchange, currency, asset_class, area, category,
             sub_group, sector, theme, benchmark_proxy, priority, observations,
             start_date, end_date) in rows
    ]


def instrument_labels(symbols: list[str]) -> list[dict[str, str]]:
    """Return display names for a bounded set of visible tree tickers."""
    normalized = list(dict.fromkeys(
        symbol.split(":", 1)[-1].strip().upper()
        for symbol in symbols
        if symbol and _TICKER.fullmatch(symbol.split(":", 1)[-1].strip())
    ))[:200]
    if not normalized:
        return []
    placeholders = ", ".join("?" for _ in normalized)
    try:
        from market_data_hub.db.connection import get_conn

        con = get_conn(read_only=True)
        try:
            rows = con.execute(
                f"""
                SELECT l.symbol, coalesce(i.name, '')
                FROM listings l JOIN instruments i USING (instrument_id)
                WHERE l.active_to IS NULL AND upper(l.symbol) IN ({placeholders})
                ORDER BY l.symbol
                """,
                normalized,
            ).fetchall()
        finally:
            con.close()
    except Exception as exc:
        raise StudioConfigError(f"Market Data Hub labels are unavailable: {type(exc).__name__}: {exc}") from exc
    return [{"symbol": str(symbol), "name": str(name)} for symbol, name in rows]


class StudioHandler(BaseHTTPRequestHandler):
    server_version = "LazyPortfolioTreeStudio/0.1"
    _run_cache: dict[str, dict[str, Any]] = {}
    _cache_lock = Lock()
    _cache_limit = 32
    _artifact_cache: dict[str, dict[str, tuple[bytes, str, str]]] = {}
    _artifact_cache_limit = 4

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/sample":
            self._json(HTTPStatus.OK, sample_config())
            return
        if path == "/api/sample/nested-v0":
            self._json(HTTPStatus.OK, minimal_sample_config())
            return
        if path == "/api/models":
            self._json(
                HTTPStatus.OK,
                {"ok": True, "database": str(_store.resolve_store_path()), "items": _saved_models()},
            )
            return
        if path.startswith("/api/models/"):
            name = unquote(path.removeprefix("/api/models/"))
            try:
                config = _store.read_model(name)
            except FileNotFoundError:
                self._json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Model not found"})
                return
            self._json(HTTPStatus.OK, config)
            return
        if path == "/api/instrument-catalog":
            query = parse_qs(parsed.query).get("q", [""])[0]
            min_observations = parse_qs(parsed.query).get("min_observations", ["0"])[0]
            start_before = parse_qs(parsed.query).get("start_before", [""])[0]
            self._json(HTTPStatus.OK, {
                "ok": True,
                "items": instrument_catalog(
                    query,
                    min_observations=int(min_observations or 0),
                    start_before=start_before,
                ),
            })
            return
        if path == "/api/instrument-labels":
            symbols = parse_qs(parsed.query).get("symbols", [""])[0].split(",")
            self._json(HTTPStatus.OK, {"ok": True, "items": instrument_labels(symbols)})
            return
        if path in {"/", "/index.html"}:
            self._file(INDEX_FILE)
            return
        self._json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 2_000_000:
                raise StudioConfigError("request is too large")
            path = urlparse(self.path).path
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise StudioConfigError("configuration must be an object")
            if path == "/api/models":
                name = payload.get("name")
                config = payload.get("config")
                if not isinstance(config, dict):
                    raise StudioConfigError("model config must be an object")
                try:
                    stored_name = _store.write_model(name, config)
                except _store.ModelStoreError as exc:
                    raise StudioConfigError(str(exc)) from exc
                self._json(HTTPStatus.OK, {"ok": True, "name": stored_name})
                return
            if path == "/api/cache/clear":
                with self._cache_lock:
                    cleared = len(self._run_cache) + len(self._artifact_cache)
                    self._run_cache.clear()
                    self._artifact_cache.clear()
                self._json(HTTPStatus.OK, {"ok": True, "cleared": cleared})
                return
            config = payload
            if path == "/api/validate":
                self._json(HTTPStatus.OK, _v2_validation_payload(config))
            elif path in {"/api/v2/estimate", "/api/v2/backtest", "/api/v2/scientific-study"}:
                config_hash = _config_hash(config)
                data_as_of, data_fingerprint = _data_fingerprint(config)
                key = _cache_key(path, config_hash, data_fingerprint)
                with self._cache_lock:
                    cached = self._run_cache.get(key)
                if cached is not None:
                    self._json(HTTPStatus.OK, {**cached, "cached": True})
                    return
                disk_cached = _run_history.get_by_cache_key(key)
                if disk_cached is not None:
                    disk_payload = disk_cached["payload"]
                    with self._cache_lock:
                        if len(self._run_cache) >= self._cache_limit and key not in self._run_cache:
                            self._run_cache.pop(next(iter(self._run_cache)))
                        self._run_cache[key] = disk_payload
                    self._json(HTTPStatus.OK, {**disk_payload, "cached": True})
                    return
                producer = {
                    "/api/v2/estimate": _v2_estimate_payload,
                    "/api/v2/backtest": _v2_backtest_payload,
                    "/api/v2/scientific-study": _v2_scientific_study_payload,
                }[path]
                payload = _as_json(producer(config))
                with self._cache_lock:
                    if len(self._run_cache) >= self._cache_limit and key not in self._run_cache:
                        self._run_cache.pop(next(iter(self._run_cache)))
                    self._run_cache[key] = payload
                weights, metrics = _run_summary_fields(path, payload)
                _run_history.record_run(
                    cache_key=key,
                    path=path,
                    kind=path.rsplit("/", 1)[-1],
                    tree_id=_tree_id_for_config(config),
                    config_hash=config_hash,
                    data_as_of=data_as_of,
                    data_fingerprint=data_fingerprint,
                    weights=weights,
                    metrics=metrics,
                    payload=payload,
                )
                self._json(HTTPStatus.OK, {**payload, "cached": False})
            elif path in {"/api/v2/audit-bundle", "/api/v2/client-report"}:
                kind: Literal["audit", "report"] = (
                    "audit" if path.endswith("audit-bundle") else "report"
                )
                config_hash = _config_hash(config)
                data_as_of, data_fingerprint = _data_fingerprint(config)
                # kind is part of the key: audit and report are cached (and,
                # for report, persisted) independently since _v2_export_artifacts
                # now builds only the one that was actually requested -- see
                # that function's docstring.
                key = _cache_key(f"/api/v2/artifacts/{kind}", config_hash, data_fingerprint)
                with self._cache_lock:
                    artifacts = self._artifact_cache.get(key)
                if artifacts is None and kind == "report":
                    disk_report = _run_history.get_report_artifact(key)
                    if disk_report is not None:
                        body, content_type, filename = disk_report
                        self._binary(HTTPStatus.OK, body, content_type, filename)
                        return
                if artifacts is None:
                    artifacts = _v2_export_artifacts(config, kind=kind)
                    with self._cache_lock:
                        if len(self._artifact_cache) >= self._artifact_cache_limit:
                            self._artifact_cache.pop(next(iter(self._artifact_cache)))
                        self._artifact_cache[key] = artifacts
                    # Persist only the (small) HTML report -- the audit ZIP is
                    # multi-MB per entry and deliberately stays in-memory-only,
                    # see lazyportfolio.v2.run_history's module docstring. An
                    # audit-bundle request no longer builds a report at all
                    # (see _v2_export_artifacts), so there is nothing to
                    # persist/register for that kind.
                    if kind == "report":
                        report_body, report_ct, report_fn = artifacts["report"]
                        title, summary = _report_artifact_fields(config)
                        # Best-effort catalog entry into LazyTools' shared
                        # artifact registry, only right here where the report
                        # is genuinely (re)generated, never on a cache hit --
                        # a re-view of an already-cached report must not
                        # insert a fresh artifact row every time.
                        external_id = register_report_artifact(
                            title=title,
                            summary=summary,
                            tags=["tree-studio"],
                            content=report_body.decode("utf-8"),
                        )
                        run_id = _run_history.record_run(
                            cache_key=key,
                            path="/api/v2/artifacts",
                            kind="artifacts",
                            tree_id=_tree_id_for_config(config),
                            config_hash=config_hash,
                            data_as_of=data_as_of,
                            data_fingerprint=data_fingerprint,
                            weights=None,
                            metrics=None,
                            payload={"title": title, "summary": summary},
                        )
                        _run_history.attach_artifact(
                            run_id,
                            kind="report",
                            content_type=report_ct,
                            filename=report_fn,
                            blob=report_body,
                            external_artifact_id=external_id,
                        )
                body, content_type, filename = artifacts[kind]
                self._binary(HTTPStatus.OK, body, content_type, filename)
            else:
                self._json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})
        except (StudioConfigError, V2OptimizationError, ValueError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
        except Exception as exc:  # Keep tracebacks in the console, not in the browser.
            print(f"[tree-studio] {type(exc).__name__}: {exc}", file=sys.stderr)
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    def do_DELETE(self) -> None:  # noqa: N802
        try:
            path = urlparse(self.path).path
            if not path.startswith("/api/models/"):
                self._json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})
                return
            name = unquote(path.removeprefix("/api/models/"))
            try:
                deleted = _store.delete_model(name)
            except FileNotFoundError as exc:
                self._json(HTTPStatus.NOT_FOUND, {"ok": False, "error": str(exc)})
                return
            self._json(HTTPStatus.OK, {"ok": True, "name": deleted})
        except (StudioConfigError, _store.ModelStoreError, ValueError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
        except Exception as exc:  # Keep tracebacks in the console, not in the browser.
            print(f"[tree-studio] {type(exc).__name__}: {exc}", file=sys.stderr)
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    def _file(self, path: Path) -> None:
        body = path.read_bytes()
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{mime}; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, default=_as_json).encode("utf-8")
        compressed = len(body) >= 64_000 and "gzip" in self.headers.get("Accept-Encoding", "").lower()
        if compressed:
            body = gzip.compress(body)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        if compressed:
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Vary", "Accept-Encoding")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _binary(
        self,
        status: HTTPStatus,
        body: bytes,
        content_type: str,
        filename: str,
    ) -> None:
        safe_filename = re.sub(r"[^A-Za-z0-9._-]+", "-", filename)
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Disposition", f'attachment; filename="{safe_filename}"')
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        view = memoryview(body)
        for offset in range(0, len(view), 64 * 1024):
            self.wfile.write(view[offset : offset + 64 * 1024])
        self.wfile.flush()

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[tree-studio] {format % args}")


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    httpd = ThreadingHTTPServer(("127.0.0.1", port), StudioHandler)
    print(f"LazyPortfolio Tree Studio running at http://127.0.0.1:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nTree Studio stopped.")


if __name__ == "__main__":
    main()
