from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import lazyportfolio
import lazyportfolio.v2.adaptive_pruning as adaptive_pruning_module
from lazyportfolio.backend import OptimizationDataset
from lazyportfolio.v2.adaptive_pruning import (
    AdaptivePruningPolicy,
    accumulated_node_metrics,
    run_adaptive_pruning,
    summarize_pruning_decisions,
)
from lazyportfolio.v2.backtest import HierarchicalV2Backtester
from lazyportfolio.v2.model import V2Model

ROOT = Path(__file__).resolve().parents[1]


def test_tests_import_lazyportfolio_from_the_worktree_under_review() -> None:
    assert Path(lazyportfolio.__file__).resolve().is_relative_to(ROOT)
    assert Path(adaptive_pruning_module.__file__).resolve().is_relative_to(ROOT)


def test_policy_is_closed_and_round_trips_backend_parameters() -> None:
    policy = AdaptivePruningPolicy.from_mapping(
        {
            "enabled": True,
            "burn_in_years": 1.5,
            "evidence_window_years": 3.0,
            "min_sharpe_improvement": 0.05,
            "max_drawdown_per_vol_ratio": 1.2,
            "workers": 4,
            "max_folds": 24,
            "expanding": True,
        }
    )
    assert policy.payload() == {
        "burn_in_years": 1.5,
        "evidence_window_years": 3.0,
        "min_sharpe_improvement": 0.05,
        "max_drawdown_per_vol_ratio": 1.2,
        "workers": 4,
        "max_folds": 24,
        "expanding": True,
    }
    assert policy.pruning_rule().required_protocols == ("accumulated",)
    with pytest.raises(ValueError, match="unknown adaptive pruning settings"):
        AdaptivePruningPolicy.from_mapping({"enabled": True, "private_tree": "IC"})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("burn_in_years", 0, "burn_in_years"),
        ("evidence_window_years", -1, "evidence_window_years"),
        ("max_drawdown_per_vol_ratio", 0, "max_drawdown_per_vol_ratio"),
        ("workers", 0, "workers"),
        ("workers", True, "workers"),
        ("max_folds", 0, "max_folds"),
        ("expanding", 1, "expanding"),
    ],
)
def test_policy_refuses_invalid_parameters(field: str, value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        AdaptivePruningPolicy(**{field: value})  # type: ignore[arg-type]


def test_accumulated_metrics_never_use_the_signal_date_or_future() -> None:
    dates = pd.date_range("2025-01-01", periods=4, freq="D")
    node = pd.Series([0.01, -0.005, 0.80, -0.90], index=dates)
    father = pd.Series([0.005, -0.002, -0.40, 0.70], index=dates)
    metrics = accumulated_node_metrics(
        {"NODE:Child": node, "FATHER:Child": father},
        ["Child"],
        as_of=dates[2],
    )
    assert metrics["NODE:Child"] == HierarchicalV2Backtester._metrics(node.iloc[:2])
    assert metrics["FATHER:Child"] == HierarchicalV2Backtester._metrics(father.iloc[:2])


def test_decision_summary_is_computed_by_backend() -> None:
    summary = summarize_pruning_decisions(
        [
            {
                "signal": "2025-01-31",
                "burn_in": False,
                "candidate_nodes": 3,
                "target_l1_distance": 0.25,
                "nodes": [
                    {"decision": "prune"},
                    {"decision": "retain"},
                    {"decision": "retain"},
                ],
            }
        ]
    )
    assert summary == [
        {
            "signal": "2025-01-31",
            "burn_in": False,
            "candidate_nodes": 3,
            "pruned_nodes": 1,
            "retained_nodes": 2,
            "target_l1_distance": 0.25,
        }
    ]


def test_backend_runs_adaptive_pruning_without_a_script_dependency() -> None:
    config = {
        "root_id": "root",
        "currency": "USD",
        "nodes": [
            {
                "id": "root",
                "name": "Root",
                "instruments": ["ticker:BOND"],
                "children": ["equity"],
                "goal": {"objective": "min_risk"},
                "constraints": {},
            },
            {
                "id": "equity",
                "name": "Equity",
                "instruments": ["ticker:A", "ticker:B"],
                "children": [],
                "proxy": "ticker:EQUITY",
                "goal": {"objective": "min_risk"},
                "constraints": {},
            },
        ],
        "backtest": {
            "train_size": 8,
            "estimation_frequency": "W",
            "rebalance_frequency": "M",
            "include_partial_last_period": True,
            "transaction_cost_bps": 0,
            "benchmark": {
                "name": "B0",
                "weights": {"ticker:EQUITY": 0.7, "ticker:BOND": 0.3},
            },
        },
    }
    rng = np.random.default_rng(42)
    index = pd.bdate_range("2023-01-02", periods=360)
    returns = pd.DataFrame(
        {
            "ticker:A": rng.normal(0.0004, 0.010, len(index)),
            "ticker:B": rng.normal(0.0003, 0.012, len(index)),
            "ticker:EQUITY": rng.normal(0.00035, 0.009, len(index)),
            "ticker:BOND": rng.normal(0.0001, 0.004, len(index)),
        },
        index=index,
    )
    model = V2Model.from_config(config)
    reference = HierarchicalV2Backtester().run(
        model,
        returns,
        mode="forward",
        train_size=8,
        estimation_frequency="W",
        rebalance_frequency="M",
        include_partial_last_period=True,
    )
    result = run_adaptive_pruning(
        config,
        model=model,
        dataset=OptimizationDataset(returns=returns, metadata={}),
        reference_report=reference,
        mode="forward",
        policy=AdaptivePruningPolicy(burn_in_years=0.25, max_folds=3),
    )
    assert result.report.folds
    assert result.decisions[-1]["burn_in"] is False
    assert result.last_candidate["root_id"] == "root"
    assert set(result.report.metrics) == {"B0", "STATIC_FINAL", "FINAL", "FORWARD_FINAL"}

    last_signal = pd.Timestamp(result.decisions[-1]["signal"])
    expected = accumulated_node_metrics(
        reference.curves,
        ["Equity"],
        as_of=last_signal,
    )
    observation = result.decisions[-1]["nodes"][0]["observations"][0]
    assert observation["node_sharpe"] == pytest.approx(
        expected["NODE:Equity"]["annualized_sharpe"]
    )
    assert observation["father_sharpe"] == pytest.approx(
        expected["FATHER:Equity"]["annualized_sharpe"]
    )
    assert observation["node_max_drawdown"] == pytest.approx(
        expected["NODE:Equity"]["max_drawdown"]
    )
    assert observation["father_max_drawdown"] == pytest.approx(
        expected["FATHER:Equity"]["max_drawdown"]
    )


def test_daily_job_propagates_a_configured_pruning_failure() -> None:
    source = (ROOT / "scripts" / "rolling_vs_expanding_backtest.py").read_text(
        encoding="utf-8"
    )
    module = ast.parse(source)
    evaluator = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_evaluate_and_send_adaptive_pruning"
    )
    main = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )

    def calls_pruning(node: ast.AST) -> bool:
        return any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "_evaluate_and_send_adaptive_pruning"
            for child in ast.walk(node)
        )

    assert calls_pruning(main)
    assert not any(isinstance(node, ast.Try) for node in ast.walk(evaluator))
    assert not any(
        isinstance(node, ast.Try) and calls_pruning(node) for node in ast.walk(main)
    )


def test_tree_studio_frontend_calls_the_backend_endpoint() -> None:
    html = (ROOT / "project" / "tree_studio.html").read_text(encoding="utf-8")
    server = (ROOT / "project" / "tree_studio.py").read_text(encoding="utf-8")
    assert "'/api/v2/adaptive-pruning'" in html
    assert 'id="prune-enabled"' in html
    assert 'id="prune-burn-in"' in html
    assert 'id="prune-window"' in html
    assert 'id="prune-sharpe"' in html
    assert 'id="prune-drawdown"' in html
    assert '"/api/v2/adaptive-pruning": _v2_adaptive_pruning_payload' in server
    assert "run_adaptive_pruning(" in server
