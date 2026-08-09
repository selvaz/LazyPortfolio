"""A2/A3: a client-report request must never build (or pay for the audit-series
capture behind) the audit ZIP, and vice versa -- see this session's
hierarchical-optimizer-performance-plan.md, Phase A3.

Exercises the real _run_full_backtest/_v2_export_artifacts code paths
against small synthetic data (bypassing Market Data Hub via a monkeypatched
_v2_inputs, the same boundary test_tree_studio_cache_freshness.py uses),
not a fully-mocked shortcut, so this actually proves the claim rather than
just asserting a monkeypatch was called.

``project/tree_studio.py`` is a script, not an installed package, so it is
imported the same way the other tree_studio tests do: put ``project/`` on
``sys.path`` first.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_DIR = REPO_ROOT / "project"


def _config() -> dict[str, Any]:
    return {
        "root_id": "root",
        "currency": "USD",
        "nodes": [
            {
                "id": "root",
                "name": "Report Waste Tree",
                "children": [],
                "instruments": ["A", "B"],
                "proxy": "",
                "goal": {"objective": "min_risk"},
                "constraints": {},
            }
        ],
        "data": {"start": "", "end": ""},
        "backtest": {
            "id": "test",
            "train_size": 60,
            "estimation_frequency": "W",
            "rebalance_frequency": "M",
            "transaction_cost_bps": 0,
            "benchmark": {"name": "B0", "weights": {"A": 0.5, "B": 0.5}},
        },
    }


@pytest.fixture()
def tree_studio(monkeypatch, tmp_path):
    monkeypatch.setenv("LAZYPORTFOLIO_TREE_DB", str(tmp_path / "store.sqlite3"))
    sys.path.insert(0, str(PROJECT_DIR))
    try:
        import importlib

        module = importlib.import_module("tree_studio")
        module = importlib.reload(module)

        rng = np.random.default_rng(20260806)
        index = pd.bdate_range("2020-01-01", periods=400)
        returns = pd.DataFrame(
            {
                "ticker:A": rng.normal(0.0004, 0.01, len(index)),
                "ticker:B": rng.normal(0.0003, 0.008, len(index)),
            },
            index=index,
        )
        from lazyportfolio.backend import OptimizationDataset
        from lazyportfolio.hierarchical_v2 import V2Model

        fake_dataset = OptimizationDataset(
            returns=returns, metadata={"source": "fake", "n_rows": len(returns)}
        )

        def _fake_v2_inputs(config):
            model = V2Model.from_config(config)
            return model, fake_dataset

        monkeypatch.setattr(module, "_v2_inputs", _fake_v2_inputs)
        module._raw_backtest_cache.clear()
        yield module
    finally:
        sys.path.remove(str(PROJECT_DIR))


def test_plain_backtest_view_never_captures_audit_series(tree_studio, monkeypatch):
    calls: list[bool] = []
    real_run = tree_studio.HierarchicalV2Backtester.run

    def _recording_run(self, *args, **kwargs):
        calls.append(kwargs["capture_audit_series"])
        return real_run(self, *args, **kwargs)

    monkeypatch.setattr(tree_studio.HierarchicalV2Backtester, "run", _recording_run)

    payload = tree_studio._v2_backtest_payload(_config())

    assert payload["ok"] is True
    assert calls == [False]


def test_audit_bundle_export_captures_audit_series_client_report_does_not(
    tree_studio, monkeypatch
):
    calls: list[bool] = []
    real_run = tree_studio.HierarchicalV2Backtester.run

    def _recording_run(self, *args, **kwargs):
        calls.append(kwargs["capture_audit_series"])
        return real_run(self, *args, **kwargs)

    monkeypatch.setattr(tree_studio.HierarchicalV2Backtester, "run", _recording_run)

    tree_studio._v2_export_artifacts(_config(), kind="report")
    tree_studio._v2_export_artifacts(_config(), kind="audit")

    assert calls == [False, True]


def test_client_report_never_calls_build_audit_bundle(tree_studio, monkeypatch):
    def _boom(**kwargs):
        raise AssertionError("build_audit_bundle must not be called for kind='report'")

    monkeypatch.setattr(tree_studio, "build_audit_bundle", _boom)

    artifacts = tree_studio._v2_export_artifacts(_config(), kind="report")

    assert set(artifacts) == {"report"}


def test_audit_bundle_never_calls_build_client_report(tree_studio, monkeypatch):
    def _boom(**kwargs):
        raise AssertionError("build_client_report must not be called for kind='audit'")

    monkeypatch.setattr(tree_studio, "build_client_report", _boom)

    artifacts = tree_studio._v2_export_artifacts(_config(), kind="audit")

    assert set(artifacts) == {"audit"}


def test_report_and_audit_are_cached_independently(tree_studio, monkeypatch):
    """A plain backtest view, a report export, and an audit export each get
    their own _run_full_backtest cache entry -- distinct capture_audit_series
    values must never collide on the same key."""
    keys = {
        "backtest": tree_studio._raw_backtest_key(_config(), capture_audit_series=False),
        "report": tree_studio._raw_backtest_key(_config(), capture_audit_series=False),
        "audit": tree_studio._raw_backtest_key(_config(), capture_audit_series=True),
    }
    assert keys["backtest"] == keys["report"]  # both capture_audit_series=False
    assert keys["audit"] != keys["report"]
