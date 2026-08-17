"""``significance_report``/``sharpe_significance_report`` below the block bootstrap floor.

D23 in ecosystem-cleanup/docs/deferred-fixes.md: with ``n_obs <= block_size``
there are no whole blocks left to resample, so every draw reproduces the
original series verbatim -- the CI collapses onto the point estimate and the
p-value bottoms out at its bootstrap floor regardless of whether the
difference is real. Reproduced there with real numbers (4/6/10 observations,
block_size=4): p-value pinned at ~0.0005 every time. Fixed by reporting the
point difference with no CI/p-value below that threshold, rather than a
number that looks like a real significance test and is not one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pruning_runner import sharpe_significance_report, significance_report  # noqa: E402


def _curves(n_obs: int, gap: float = 0.01) -> dict[str, pd.Series]:
    """Two daily-return curves, CANDIDATE consistently ahead of BASELINE by
    ``gap`` per period plus small noise -- a real, nonzero difference, so a
    valid statistical test on enough data should find it significant."""
    idx = pd.bdate_range("2026-01-01", periods=n_obs)
    rng = np.random.default_rng(3)
    baseline = rng.normal(0.0003, 0.0005, n_obs)
    candidate = baseline + gap + rng.normal(0.0, 0.0001, n_obs)
    return {
        "CANDIDATE": pd.Series(candidate, index=idx),
        "BASELINE": pd.Series(baseline, index=idx),
    }


@pytest.mark.parametrize("n_obs", [4, 6, 10, 20])
def test_significance_report_omits_ci_and_p_value_at_or_below_block_size(n_obs):
    report = significance_report(_curves(n_obs), [("CANDIDATE", "BASELINE")], block_size=4)[0]
    if n_obs <= 4:
        assert report["ci_low"] is None
        assert report["ci_high"] is None
        assert report["p_value"] is None
        assert report["holm_adjusted_p_value"] is None
        assert "note" in report and "too small" in report["note"]
    else:
        assert report["ci_low"] is not None
        assert report["ci_high"] is not None
        assert report["p_value"] is not None
        assert "note" not in report


def test_significance_report_point_difference_matches_the_raw_mean_gap():
    """The point estimate shown when insufficient must be the real mean gap,
    not a leftover/zeroed field from skipping the bootstrap call."""
    curves = _curves(4, gap=0.02)
    report = significance_report(curves, [("CANDIDATE", "BASELINE")], block_size=4)[0]
    expected = float((curves["CANDIDATE"] - curves["BASELINE"]).mean() * 252.0)
    assert report["annualized_mean_difference"] == pytest.approx(expected)


def test_significance_report_holm_adjustment_ignores_untestable_comparisons():
    """Two comparisons, both insufficient: neither gets a p-value, so Holm
    has nothing to adjust -- it must not crash or fabricate a value."""
    curves = _curves(4)
    curves["OTHER"] = curves["CANDIDATE"] * 1.01
    reports = significance_report(
        curves, [("CANDIDATE", "BASELINE"), ("OTHER", "BASELINE")], block_size=4,
    )
    assert all(r["p_value"] is None and r["holm_adjusted_p_value"] is None for r in reports)


@pytest.mark.parametrize("n_obs", [4, 10, 20])
def test_sharpe_significance_report_omits_ci_and_p_value_at_or_below_block_size(n_obs):
    reports = sharpe_significance_report(
        _curves(n_obs), [("CANDIDATE", "BASELINE")], block_size=4,
    )
    report = reports[0]
    if n_obs <= 4:
        assert report["ci_low"] is None
        assert report["ci_high"] is None
        assert report["p_value"] is None
        assert report["holm_adjusted_p_value"] is None
        assert "note" in report
        # sharpe_difference is a plain arithmetic comparison, not a resample
        # product, so it must still be a real, finite number
        assert np.isfinite(report["sharpe_difference"])
    else:
        assert report["ci_low"] is not None
        assert "note" not in report


def test_sufficient_data_still_finds_the_planted_gap_significant():
    """Not just "doesn't crash below the floor" -- above it, a real,
    generously-sized planted difference should still come out significant,
    so the guard isn't quietly disabling the test for everyone."""
    report = significance_report(_curves(120, gap=0.02), [("CANDIDATE", "BASELINE")])[0]
    assert report["p_value"] < 0.05
    assert report["ci_low"] > 0
