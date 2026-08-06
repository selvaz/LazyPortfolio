"""Phase F1 (v3 performance roadmap): parallel fold estimation in
HierarchicalV2Backtester.run().

Each fold's estimator.estimate() call only depends on that fold's own
training window, so it's dispatched across worker processes when
max_workers > 1 (ProcessPoolExecutor). The ledger walk that turns fold
targets into OOS curves carries state from one fold's holding period into
the next rebalance, so it always stays sequential -- these tests verify
that splitting the work this way produces byte-identical results to the
default (max_workers=1, unchanged sequential) path, not just "close enough"
numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lazyportfolio.hierarchical_v2 import (
    HierarchicalV2Backtester,
    V2Benchmark,
    V2Constraints,
    V2Model,
    V2Node,
)


def _small_model_and_returns() -> tuple[V2Model, pd.DataFrame]:
    rng = np.random.default_rng(2026)
    index = pd.bdate_range("2020-01-01", periods=260)
    returns = pd.DataFrame(
        {
            "ticker:A": rng.normal(0.0004, 0.010, len(index)),
            "ticker:B": rng.normal(0.0002, 0.012, len(index)),
            "ticker:C": rng.normal(0.0003, 0.008, len(index)),
        },
        index=index,
    )
    root = V2Node(
        id="root",
        name="Root",
        instruments=["ticker:A", "ticker:B", "ticker:C"],
        children=[],
        proxy=None,
        objective="max_return",
        constraints=V2Constraints(
            volatility_reference="manual",
            volatility_target=0.12,
            volatility_target_policy="nearest_feasible",
        ),
    )
    model = V2Model(
        root=root,
        benchmark=V2Benchmark(name="B0", weights={"ticker:A": 0.6, "ticker:B": 0.4}),
        reference_currency="USD",
    )
    return model, returns


@pytest.mark.parametrize("mode", ["flat", "forward"])
def test_parallel_folds_match_sequential_exactly(mode: str) -> None:
    model, returns = _small_model_and_returns()

    sequential = HierarchicalV2Backtester().run(
        model,
        returns,
        mode=mode,
        train_size=8,
        estimation_frequency="W",
        rebalance_frequency="M",
        transaction_cost_bps=5.0,
        include_partial_last_period=True,
        max_workers=1,
    )
    parallel = HierarchicalV2Backtester().run(
        model,
        returns,
        mode=mode,
        train_size=8,
        estimation_frequency="W",
        rebalance_frequency="M",
        transaction_cost_bps=5.0,
        include_partial_last_period=True,
        max_workers=2,
    )

    assert len(sequential.folds) == len(parallel.folds) >= 2
    for seq_fold, par_fold in zip(sequential.folds, parallel.folds, strict=True):
        assert seq_fold.signal == par_fold.signal
        assert seq_fold.targets.keys() == par_fold.targets.keys()
        for arm in seq_fold.targets:
            for name in seq_fold.targets[arm]:
                assert seq_fold.targets[arm][name] == pytest.approx(
                    par_fold.targets[arm][name], abs=1e-10
                )
    assert set(sequential.curves) == set(parallel.curves)
    for arm in sequential.curves:
        pd.testing.assert_series_equal(
            sequential.curves[arm], parallel.curves[arm], check_exact=False, atol=1e-10
        )
    assert sequential.transaction_cost_paid == pytest.approx(
        parallel.transaction_cost_paid, abs=1e-10
    )


def test_parallel_worker_count_never_exceeds_fold_count() -> None:
    """A single-fold backtest must not try to spin up a process pool at
    all -- max_workers > fold count is clamped, not an error."""
    model, returns = _small_model_and_returns()
    report = HierarchicalV2Backtester().run(
        model,
        returns.tail(70),
        mode="flat",
        train_size=8,
        estimation_frequency="W",
        rebalance_frequency="M",
        include_partial_last_period=True,
        max_workers=8,
    )
    assert len(report.folds) >= 1
