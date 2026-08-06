"""Expanding-window walk-forward mode for HierarchicalV2Backtester.run().

`expanding=False` (default) keeps today's behavior: each fold's training
window is exactly `train_size` observations, sliding forward
(`.tail(train_size)`). `expanding=True` uses every observation up to the
signal date instead -- `train_size` becomes only the minimum size before
the first fold is emitted, not a cap, so the window grows fold over fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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


def test_expanding_window_grows_while_rolling_stays_fixed() -> None:
    model, returns = _small_model_and_returns()
    kwargs = dict(
        mode="flat",
        train_size=8,
        estimation_frequency="W",
        rebalance_frequency="M",
        include_partial_last_period=True,
    )

    rolling = HierarchicalV2Backtester().run(model, returns, **kwargs, expanding=False)
    expanding = HierarchicalV2Backtester().run(model, returns, **kwargs, expanding=True)

    assert len(rolling.folds) == len(expanding.folds) >= 3

    rolling_training_lengths = {
        (fold.training_end - fold.training_start) for fold in rolling.folds
    }
    assert len(rolling_training_lengths) == 1, "rolling window must stay a fixed span"

    expanding_spans = [
        (fold.training_end - fold.training_start) for fold in expanding.folds
    ]
    assert expanding_spans == sorted(expanding_spans), "expanding window must only grow"
    assert expanding_spans[-1] > expanding_spans[0], (
        "expanding window must actually grow across folds, not stay flat"
    )
    assert expanding.folds[0].training_start <= rolling.folds[0].training_start, (
        "expanding must never discard early data, even on the first fold "
        "that already clears the train_size floor -- if the first "
        "qualifying signal has more than train_size observations "
        "available, expanding keeps all of them while rolling still "
        "trims to exactly train_size"
    )
    assert expanding.folds[-1].training_start < rolling.folds[-1].training_start, (
        "the expanding window's last fold must reach further back than the "
        "rolling window's same-position fold, since it never drops early data"
    )


def test_expanding_window_still_requires_the_minimum_train_size() -> None:
    """train_size is a floor, not a target, under expanding -- a window
    that starts below it must still be skipped exactly like rolling does."""
    model, returns = _small_model_and_returns()
    rolling = HierarchicalV2Backtester().run(
        model,
        returns,
        mode="flat",
        train_size=8,
        estimation_frequency="W",
        rebalance_frequency="M",
        include_partial_last_period=True,
        expanding=False,
    )
    expanding = HierarchicalV2Backtester().run(
        model,
        returns,
        mode="flat",
        train_size=8,
        estimation_frequency="W",
        rebalance_frequency="M",
        include_partial_last_period=True,
        expanding=True,
    )
    assert rolling.folds[0].signal == expanding.folds[0].signal, (
        "both modes must skip the same early folds that don't yet have "
        "train_size observations"
    )
