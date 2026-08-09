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

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_v2 import local_solves  # noqa: E402

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


def test_forward_backward_does_not_double_count_solve_seconds(two_node_model_and_returns) -> None:
    model, returns = two_node_model_and_returns
    estimate_fb = HierarchicalV2Estimator().estimate(
        model, returns, mode="forward_backward", periods_per_year=252.0
    )
    estimate_fwd = HierarchicalV2Estimator().estimate(
        model, returns, mode="forward", periods_per_year=252.0
    )
    _, seconds_fb, _, _ = local_solves(estimate_fb)
    _, seconds_fwd, _, _ = local_solves(estimate_fwd)
    # forward_backward does exactly one more real solve than forward (the
    # root's backward re-solve) -- its total solve time must be at least
    # forward's, not roughly double it (which double-counting would produce).
    assert seconds_fb >= seconds_fwd
    assert seconds_fb < seconds_fwd * 1.9
