"""Phase B.5 (v3 performance roadmap): target/cap/TEV volatility must be
measured with the same risk measure the target/cap itself is computed
from (the reference series' own historical std), never the
shrunk-covariance-implied figure that drives weight estimation.

Real bug this covers: when a node's father/benchmark proxy is also one of
its own candidate instruments (e.g. a Precious Metals node holding
GLD/SLV/PPLT/PALL with father proxy GLD), covariance shrinkage makes a
single-asset "portfolio's" implied volatility diverge from that asset's own
raw historical volatility -- so a 100%-proxy allocation, provably feasible
and provably matching its own reference by construction, could still fail
a hard_fail equality check under the old covariance-implied measure. None
of this behavior had a dedicated test before, even though it landed
alongside Phase B2's QP route.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lazyportfolio.v2.contracts import V2Constraints
from lazyportfolio.v2.moments import estimate_moments
from lazyportfolio.v2.solver import V2LocalOptimizer


def _precious_metals_returns() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    index = pd.bdate_range("2020-01-01", periods=500)
    return pd.DataFrame(
        {
            "GLD": rng.normal(0.0003, 0.010, len(index)),
            "SLV": rng.normal(0.0002, 0.018, len(index)),
            "PPLT": rng.normal(0.0001, 0.014, len(index)),
            "PALL": rng.normal(0.0004, 0.020, len(index)),
        },
        index=index,
    )


def test_covariance_implied_and_raw_std_diverge_for_a_single_asset() -> None:
    """Sanity check the premise: shrinkage really does move a single-asset
    "portfolio's" implied volatility away from that asset's own raw
    historical std -- if this ever converges to ~0 the rest of this file's
    regression coverage becomes meaningless."""
    returns = _precious_metals_returns()
    names = list(returns.columns)
    constraints = V2Constraints()
    covariance, _, _ = estimate_moments(
        returns, names, None, constraints.mean_estimator, 1.0, 0.0,
        constraints.covariance_estimator,
    )
    weights = np.array([1.0, 0.0, 0.0, 0.0])
    covariance_implied = float(np.sqrt(weights @ covariance @ weights)) * (252**0.5)
    raw_std = float(returns["GLD"].std(ddof=1)) * (252**0.5)
    assert abs(covariance_implied - raw_std) / raw_std > 0.01


def test_hard_fail_does_not_raise_when_proxy_is_also_a_direct_instrument() -> None:
    returns = _precious_metals_returns()
    father = returns["GLD"]
    constraints = V2Constraints(
        volatility_reference="father_proxy",
        volatility_target_policy="hard_fail",
    )
    weights, audit = V2LocalOptimizer().solve(
        returns,
        objective="max_return",
        constraints=constraints,
        periods_per_year=252.0,
        target_reference_series=father,
        cap_reference_series=None,
        tracking_reference_series=None,
        reference_weights=None,
    )
    assert audit.target_status == "matched"
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)


def test_actual_volatility_matches_target_to_the_realized_measure() -> None:
    """`_resolve_volatility` computes the target from the reference series'
    own std (`series.std(ddof=1) * sqrt(periods_per_year)`); the solved
    portfolio's reported actual_volatility must be measured the exact same
    way, not via the covariance -- otherwise "matched" would be a lie."""
    returns = _precious_metals_returns()
    father = returns["GLD"]
    target_annual = float(father.std(ddof=1)) * (252**0.5)
    constraints = V2Constraints(
        volatility_reference="father_proxy",
        volatility_target_policy="hard_fail",
    )
    _, audit = V2LocalOptimizer().solve(
        returns,
        objective="max_return",
        constraints=constraints,
        periods_per_year=252.0,
        target_reference_series=father,
        cap_reference_series=None,
        tracking_reference_series=None,
        reference_weights=None,
    )
    assert audit.target_volatility == pytest.approx(target_annual, abs=1e-9)
    assert audit.actual_volatility == pytest.approx(target_annual, abs=1e-4)


def test_proxy_seed_is_offered_when_other_instruments_are_much_more_volatile() -> None:
    """When blending in any other instrument moves realized volatility
    measurably away from the proxy's own std, the deterministic 100%-proxy
    seed should let the solver converge to a proxy-heavy allocation instead
    of needing multi-start luck or falling back to nearest_feasible."""
    rng = np.random.default_rng(7)
    index = pd.bdate_range("2020-01-01", periods=500)
    returns = pd.DataFrame(
        {
            "GLD": rng.normal(0.0003, 0.010, len(index)),
            "SLV": rng.normal(0.0002, 0.05, len(index)),
            "PPLT": rng.normal(0.0001, 0.06, len(index)),
            "PALL": rng.normal(0.0004, 0.07, len(index)),
        },
        index=index,
    )
    father = returns["GLD"]
    constraints = V2Constraints(
        volatility_reference="father_proxy",
        volatility_target_policy="hard_fail",
    )
    weights, audit = V2LocalOptimizer().solve(
        returns,
        objective="max_return",
        constraints=constraints,
        periods_per_year=252.0,
        target_reference_series=father,
        cap_reference_series=None,
        tracking_reference_series=None,
        reference_weights=None,
    )
    assert audit.target_status == "matched"
    assert weights["GLD"] > 0.5


def test_max_volatility_cap_is_enforced_with_the_realized_measure() -> None:
    """The cap constraint (`max_volatility_reference`) must use the same
    realized measure as the exact target, not the covariance-implied one --
    the solved portfolio's realized volatility must sit at or under the
    cap, and the cap value itself must equal the reference's own raw std."""
    returns = _precious_metals_returns()
    father = returns["GLD"]
    cap_annual = float(father.std(ddof=1)) * (252**0.5)
    constraints = V2Constraints(max_volatility_reference="father_proxy")
    weights, audit = V2LocalOptimizer().solve(
        returns,
        objective="max_return",
        constraints=constraints,
        periods_per_year=252.0,
        target_reference_series=None,
        cap_reference_series=father,
        tracking_reference_series=None,
        reference_weights=None,
    )
    assert audit.volatility_cap == pytest.approx(cap_annual, abs=1e-9)
    realized = float(np.std(returns.to_numpy() @ np.array(
        [weights[name] for name in returns.columns]
    ), ddof=1)) * (252**0.5)
    assert realized <= cap_annual + 1e-6
