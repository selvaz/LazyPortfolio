from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lazyportfolio import V2Model
from lazyportfolio.scientific_study import (
    ScientificStudyProtocol,
    _holm_adjust,
    baseline_allocations,
    paired_block_bootstrap,
    run_scientific_study,
)


def _model() -> V2Model:
    return V2Model.from_config(
        {
            "root_id": "root",
            "nodes": [
                {
                    "id": "root",
                    "name": "Root",
                    "children": [],
                    "instruments": ["A", "B", "C", "D"],
                    "goal": {"objective": "min_risk"},
                    "constraints": {"risk_free_rate": 0.02},
                }
            ],
            "backtest": {
                "benchmark": {
                    "name": "B0",
                    "weights": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
                }
            },
        }
    )


def _training_returns() -> pd.DataFrame:
    rng = np.random.default_rng(31)
    common = rng.normal(0.0005, 0.006, size=(180, 1))
    residual = rng.normal(0.0003, 0.012, size=(180, 4))
    return pd.DataFrame(
        common + residual,
        columns=["ticker:A", "ticker:B", "ticker:C", "ticker:D"],
    )


def _daily_returns() -> pd.DataFrame:
    frame = _training_returns().iloc[:120].copy()
    frame.index = pd.bdate_range("2024-01-02", periods=len(frame))
    return frame


def test_required_baselines_are_fitted_and_audited() -> None:
    allocations = baseline_allocations(
        _model(),
        _training_returns(),
        risk_aversion=1.5,
        risk_free_rate=0.02,
    )
    assert set(allocations) == {
        "EQUAL_WEIGHT",
        "DECLARED_BENCHMARK",
        "SAMPLE_MIN_VARIANCE",
        "SHRUNK_FIXED_MIN_VARIANCE",
        "LEDOIT_WOLF_MIN_VARIANCE",
        "HRP_WARD_PEARSON",
    }
    for weights in allocations.values():
        vector = np.array(list(weights.values()), dtype=float)
        assert np.all(np.isfinite(vector))
        assert vector.sum() == pytest.approx(1.0)
        assert np.all(vector >= -1e-8)


def test_scientific_study_uses_one_common_oos_grid() -> None:
    result = run_scientific_study(
        _model(),
        _daily_returns(),
        mode="flat",
        protocol=ScientificStudyProtocol(
            train_size=30,
            estimation_frequency="D",
            rebalance_frequency="M",
            transaction_cost_bps=5.0,
            bootstrap_samples=100,
            bootstrap_block_size=5,
            random_seed=19,
        ),
    )
    assert result.fold_count >= 2
    assert set(result.curves) == {
        "V2_FINAL",
        "EQUAL_WEIGHT",
        "DECLARED_BENCHMARK",
        "SAMPLE_MIN_VARIANCE",
        "SHRUNK_FIXED_MIN_VARIANCE",
        "LEDOIT_WOLF_MIN_VARIANCE",
        "HRP_WARD_PEARSON",
        # Proxy-vs-synthetic representation ablation (mode="flat" requested
        # V2_FINAL here, so both representation modes are additional runs)
        # and the direct-bottom-up arm (Phase 8, clean-engine follow-up).
        "V2_FORWARD",
        "V2_FORWARD_BACKWARD",
        "V2_DIRECT_BOTTOM_UP",
    }
    indexes = [curve.index for curve in result.curves.values()]
    assert all(index.equals(indexes[0]) for index in indexes[1:])
    assert result.common_oos_start == indexes[0].min()
    assert result.common_oos_end == indexes[0].max()
    assert all(metric["risk_free_rate"] == pytest.approx(0.02) for metric in result.metrics.values())
    # Deliberate, documented pin update (not silent): inference now covers
    # all six declared baselines, not just EQUAL_WEIGHT/DECLARED_BENCHMARK -
    # user-confirmed decision to extend rather than merely document the
    # narrower scope (see docs/optimizer-v2-remediation-status.md).
    assert {comparison.baseline for comparison in result.comparisons} == {
        "EQUAL_WEIGHT",
        "DECLARED_BENCHMARK",
        "SAMPLE_MIN_VARIANCE",
        "SHRUNK_FIXED_MIN_VARIANCE",
        "LEDOIT_WOLF_MIN_VARIANCE",
        "HRP_WARD_PEARSON",
    }
    assert all(
        comparison.holm_adjusted_p_value >= comparison.p_value
        for comparison in result.comparisons
    )


def test_block_bootstrap_is_reproducible() -> None:
    differences = np.linspace(-0.001, 0.003, 300)
    first = paired_block_bootstrap(
        differences,
        samples=500,
        block_size=15,
        random_seed=11,
    )
    second = paired_block_bootstrap(
        differences,
        samples=500,
        block_size=15,
        random_seed=11,
    )
    assert first == second
    low, high, p_value = first
    assert low < high
    assert 0.0 <= p_value <= 1.0


def test_block_bootstrap_rejects_invalid_protocol() -> None:
    with pytest.raises(ValueError, match="at least 100"):
        paired_block_bootstrap(
            np.array([0.1, 0.2]),
            samples=10,
            block_size=1,
            random_seed=1,
        )
    with pytest.raises(ValueError, match="positive"):
        paired_block_bootstrap(
            np.array([0.1, 0.2]),
            samples=100,
            block_size=0,
            random_seed=1,
        )


def test_holm_adjustment_is_order_invariant_and_not_smaller_than_raw() -> None:
    raw = [0.04, 0.01, 0.20]
    adjusted = _holm_adjust(raw)
    assert all(value >= original for value, original in zip(adjusted, raw, strict=True))
    permuted = [raw[2], raw[0], raw[1]]
    permuted_adjusted = _holm_adjust(permuted)
    restored = [permuted_adjusted[1], permuted_adjusted[2], permuted_adjusted[0]]
    assert restored == pytest.approx(adjusted)


def test_protocol_defaults_are_explicit() -> None:
    protocol = ScientificStudyProtocol(train_size=52)
    assert protocol.bootstrap_samples == 2_000
    assert protocol.bootstrap_block_size == 20
    assert protocol.random_seed == 7


def test_ablation_arms_are_separate_from_strategy_comparisons() -> None:
    """The proxy-vs-synthetic representation ablation (V2_FORWARD/
    V2_FORWARD_BACKWARD) and the direct-bottom-up arm are fitted, audited and
    reported like any other curve, but must never be folded into the
    baseline strategy-comparison bootstrap: they are an estimator/
    representation ablation, not a claim about financial superiority over a
    baseline.
    """

    result = run_scientific_study(
        _model(),
        _daily_returns(),
        mode="forward_backward",
        protocol=ScientificStudyProtocol(
            train_size=30,
            estimation_frequency="D",
            rebalance_frequency="M",
            transaction_cost_bps=5.0,
            bootstrap_samples=100,
            bootstrap_block_size=5,
            random_seed=19,
        ),
    )
    for arm in ("V2_FORWARD", "V2_FORWARD_BACKWARD", "V2_DIRECT_BOTTOM_UP"):
        assert arm in result.curves
        assert arm in result.metrics
        assert np.isfinite(result.metrics[arm]["cagr"])
    comparison_baselines = {comparison.baseline for comparison in result.comparisons}
    assert not comparison_baselines & {
        "V2_FORWARD", "V2_FORWARD_BACKWARD", "V2_DIRECT_BOTTOM_UP",
    }
    # mode="forward_backward" was requested as the primary V2_FINAL curve, so
    # its own ablation-labelled twin must be an alias of the same curve
    # rather than a redundant second backtest run.
    pd.testing.assert_series_equal(
        result.curves["V2_FINAL"], result.curves["V2_FORWARD_BACKWARD"]
    )
