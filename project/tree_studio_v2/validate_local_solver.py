"""External gate 1 for the independent hierarchical V2 local optimiser.

Run from the repository root.  The script exits non-zero on the first violated
invariant and does not import Tree Studio or any legacy optimisation engine.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lazyportfolio.hierarchical_v2 import (  # noqa: E402
    V2Constraints,
    V2LocalOptimizer,
    V2OptimizationError,
)


def synthetic_returns() -> pd.DataFrame:
    rng = np.random.default_rng(20260720)
    index = pd.bdate_range("2020-01-01", periods=520)
    defensive = rng.normal(0.00010, 0.0030, len(index))
    growth = rng.normal(0.00065, 0.0120, len(index))
    diversifier = rng.normal(0.00030, 0.0070, len(index))
    father = 0.55 * growth + 0.35 * defensive + 0.10 * diversifier
    return pd.DataFrame(
        {
            "ticker:GROWTH": growth,
            "ticker:DEFENSIVE": defensive,
            "ticker:DIVERSIFIER": diversifier,
            "ticker:FATHER": father,
        },
        index=index,
    )


def assert_close(left: float, right: float, tolerance: float, label: str) -> None:
    if abs(left - right) > tolerance:
        raise AssertionError(f"{label}: {left:.10f} != {right:.10f}")


def main() -> None:
    returns = synthetic_returns()
    optimiser = V2LocalOptimizer()
    father = returns["ticker:FATHER"]

    # Target and TEV are both relative to the father. Bounds are local to the
    # series in this solve and must survive the optimisation unchanged.
    weights, audit = optimiser.solve(
        returns,
        objective="max_return",
        constraints=V2Constraints(
            min_weights={"ticker:DEFENSIVE": 0.10},
            max_weights={"ticker:GROWTH": 0.45},
            volatility_reference="father_proxy",
            max_tracking_error=0.03,
            tracking_error_reference="father_proxy",
        ),
        periods_per_year=252.0,
        target_reference_series=father,
        cap_reference_series=None,
        tracking_reference_series=father,
        reference_weights={"ticker:FATHER": 1.0},
    )
    assert_close(sum(weights.values()), 1.0, 1e-8, "fully invested")
    assert_close(audit.actual_volatility, audit.target_volatility or 0.0, 5e-5, "target vol")
    if (audit.actual_tracking_error or 0.0) > 0.03001:
        raise AssertionError("TEV limit violated")
    if weights["ticker:GROWTH"] > 0.450001:
        raise AssertionError("maximum series allocation violated")
    if weights["ticker:DEFENSIVE"] < 0.099999:
        raise AssertionError("minimum series allocation violated")
    print("PASS target father + TEV + local min/max")

    # Root-relative cap is an inequality and must not be converted into a target.
    _, cap_audit = optimiser.solve(
        returns.drop(columns="ticker:FATHER"),
        objective="min_risk",
        constraints=V2Constraints(max_volatility_reference="forward_root_reference"),
        periods_per_year=252.0,
        target_reference_series=None,
        cap_reference_series=father,
        tracking_reference_series=None,
        reference_weights=None,
    )
    if cap_audit.actual_volatility > (cap_audit.volatility_cap or 0.0) + 5e-5:
        raise AssertionError("root-relative volatility cap violated")
    if cap_audit.target_volatility is not None:
        raise AssertionError("volatility cap was incorrectly reported as a target")
    print("PASS root-relative volatility cap")

    # Bounds that cannot produce a fully-invested portfolio are rejected before solve.
    try:
        optimiser.solve(
            returns[["ticker:GROWTH", "ticker:DEFENSIVE"]],
            objective="max_return",
            constraints=V2Constraints(
                max_weights={"ticker:GROWTH": 0.40, "ticker:DEFENSIVE": 0.40}
            ),
            periods_per_year=252.0,
            target_reference_series=None,
            cap_reference_series=None,
            tracking_reference_series=None,
            reference_weights=None,
        )
    except V2OptimizationError:
        print("PASS infeasible local bounds rejected")
    else:
        raise AssertionError("infeasible local bounds were accepted")

    # A father is a reference, not an implicit candidate. If its volatility or
    # TEV cannot be reached, V2 projects to the nearest result in the declared
    # candidate universe and exposes that relaxation in the audit.
    candidates = returns.drop(columns="ticker:FATHER")
    projected_weights, projected_vol = optimiser.solve(
        candidates,
        objective="max_return",
        constraints=V2Constraints(
            volatility_reference="father_proxy",
            volatility_target_policy="nearest_feasible",
        ),
        periods_per_year=252.0,
        target_reference_series=father * 3.0,
        cap_reference_series=None,
        tracking_reference_series=None,
        reference_weights={"ticker:FATHER": 1.0},
    )
    if "ticker:FATHER" in projected_weights:
        raise AssertionError("father was inserted into the candidate universe")
    if projected_vol.target_status != "nearest_feasible":
        raise AssertionError("unreachable volatility target was not projected")
    print("PASS absent father + nearest feasible volatility")

    distant_father = father + np.linspace(-0.02, 0.02, len(father))
    tev_weights, projected_tev = optimiser.solve(
        candidates,
        objective="max_return",
        constraints=V2Constraints(
            max_tracking_error=0.001,
            tracking_error_reference="father_proxy",
            tracking_error_policy="nearest_feasible",
        ),
        periods_per_year=252.0,
        target_reference_series=None,
        cap_reference_series=None,
        tracking_reference_series=distant_father,
        reference_weights={"ticker:FATHER": 1.0},
    )
    if "ticker:FATHER" in tev_weights:
        raise AssertionError("father was inserted into the candidate universe")
    if projected_tev.tracking_error_status != "nearest_feasible":
        raise AssertionError("unreachable TEV limit was not projected")
    print("PASS absent father + nearest feasible TEV")


if __name__ == "__main__":
    main()
