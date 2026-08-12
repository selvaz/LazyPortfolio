"""Regression guard for scripts/benchmark_v2.py's local-solve counting.

Deterministic, synthetic, no Market Data Hub -- unlike the benchmark script
itself (a live measurement tool, not a test), this only pins down that
local_solves() counts what it should on a small known tree: a wiring bug in
a later v3 phase (an accidental double-solve, a skipped node, a route that
silently drops a node) should move this count, even though the local-solve
count itself is expected to stay identical across every future solver route
-- routing changes *which* solver runs, never how many (node, pass)
combinations need solving.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scripts.benchmark_v2 import local_solves

from lazyportfolio.hierarchical_v2 import (  # noqa: E402
    HierarchicalV2Estimator,
    V2Benchmark,
    V2Constraints,
    V2Model,
    V2Node,
)


@pytest.fixture()
def two_node_model_and_returns() -> tuple[V2Model, pd.DataFrame]:
    rng = np.random.default_rng(20260806)
    index = pd.bdate_range("2020-01-01", periods=300)
    returns = pd.DataFrame(
        {
            "ticker:CHILD_A": rng.normal(0.0005, 0.01, len(index)),
            "ticker:CHILD_B": rng.normal(0.0003, 0.008, len(index)),
            "ticker:BOND": rng.normal(0.0002, 0.004, len(index)),
        },
        index=index,
    )
    returns["ticker:EQUITY"] = 0.5 * returns["ticker:CHILD_A"] + 0.5 * returns["ticker:CHILD_B"]
    child = V2Node(
        id="child", name="Equity", instruments=["ticker:CHILD_A", "ticker:CHILD_B"],
        children=[], proxy="ticker:EQUITY", objective="min_risk", constraints=V2Constraints(),
    )
    root = V2Node(
        id="root", name="Root", instruments=["ticker:BOND"], children=[child],
        proxy=None, objective="min_risk", constraints=V2Constraints(),
    )
    model = V2Model(
        root=root,
        benchmark=V2Benchmark(name="B0", weights={"ticker:EQUITY": 0.6, "ticker:BOND": 0.4}),
        reference_currency="USD",
    )
    return model, returns


def test_flat_mode_solve_count(two_node_model_and_returns) -> None:
    """flat also runs the full Forward pass best-effort for diagnostics
    (see hierarchy.py's _estimate_flat docstring), so a 2-node tree is
    2 (forward: root + child) + 1 (the flat pool itself) = 3, not 1."""
    model, returns = two_node_model_and_returns
    estimate = HierarchicalV2Estimator().estimate(
        model, returns, mode="flat", periods_per_year=252.0
    )
    solves, _, _, _ = local_solves(estimate)
    assert solves == 3


def test_forward_mode_solve_count_is_one_per_node(two_node_model_and_returns) -> None:
    model, returns = two_node_model_and_returns
    estimate = HierarchicalV2Estimator().estimate(
        model, returns, mode="forward", periods_per_year=252.0
    )
    solves, _, _, _ = local_solves(estimate)
    assert solves == 2  # root + child, once each


def test_forward_backward_solve_count_excludes_reused_leaf(two_node_model_and_returns) -> None:
    """The leaf reuses its Forward result in Backward (never re-solved) --
    local_solves() must not double-count it via forward_node_results just
    because it also appears there under a different pass_kind label."""
    model, returns = two_node_model_and_returns
    estimate = HierarchicalV2Estimator().estimate(
        model, returns, mode="forward_backward", periods_per_year=252.0
    )
    solves, _, _, _ = local_solves(estimate)
    assert solves == 3  # forward: root + child; backward: root only (child reused)


def _audit(component_id, pass_kind, solve_seconds):
    return SimpleNamespace(
        component_id=component_id,
        pass_kind=pass_kind,
        solve_seconds=solve_seconds,
        restart_candidate_count=1,
        problem_class="synthetic",
    )


def _result(primary, forward):
    return SimpleNamespace(
        node_results={name: SimpleNamespace(audit=audit) for name, audit in primary.items()},
        forward_node_results={
            name: SimpleNamespace(audit=audit) for name, audit in forward.items()
        },
    )


def test_reused_audit_is_counted_once_even_if_its_recorded_time_differs() -> None:
    result = _result(
        {"leaf": _audit("node:leaf", "forward", 0.2)},
        {"leaf": _audit("node:leaf", "forward", 9.9)},
    )
    solves, seconds, _, _ = local_solves(result)
    assert solves == 1
    assert seconds == pytest.approx(0.2)


def test_distinct_passes_are_counted_even_if_their_times_are_identical() -> None:
    result = _result(
        {"root": _audit("node:root", "backward", 0.3)},
        {"root": _audit("node:root", "forward", 0.3)},
    )
    solves, seconds, _, _ = local_solves(result)
    assert solves == 2
    assert seconds == pytest.approx(0.6)


def test_forward_backward_counts_root_twice_and_reused_leaf_once() -> None:
    result = _result(
        {
            "root": _audit("node:root", "backward", 0.3),
            "leaf": _audit("node:leaf", "forward", 0.1),
        },
        {
            "root": _audit("node:root", "forward", 0.2),
            "leaf": _audit("node:leaf", "forward", 8.0),
        },
    )
    solves, seconds, slsqp_calls, problem_classes = local_solves(result)
    assert solves == 3
    assert seconds == pytest.approx(0.6)
    assert slsqp_calls == 3
    assert problem_classes == ["synthetic"]


@pytest.mark.parametrize(
    ("component_id", "pass_kind", "message"),
    [("", "forward", "component_id"), ("node:root", "", "pass_kind")],
)
def test_missing_audit_identity_is_rejected(component_id, pass_kind, message) -> None:
    result = _result(
        {"root": _audit(component_id, pass_kind, 0.1)},
        {},
    )
    with pytest.raises(ValueError, match=message):
        local_solves(result)
