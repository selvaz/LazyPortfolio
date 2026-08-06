"""Characterization tests for the optimizer V2 clean-engine follow-up remediation.

These tests pin *current* behavior of `lazyportfolio.v2` before the
component-identity / reference-separation / lexicographic-fallback refactor
described in `docs/optimizer-v2-clean-engine-follow-up.md`. They are the
regression baseline: later phases either keep a test green unchanged
(behavior that was already correct) or deliberately replace/invert a test
when a documented defect is fixed (never silently update an assertion to
match new output without noting why in the commit message).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from lazyportfolio.hierarchical_v2 import (
    HierarchicalV2Estimator,
    V2Benchmark,
    V2Constraints,
    V2LocalOptimizer,
    V2Model,
    V2Node,
    V2OptimizationError,
)


def _returns() -> pd.DataFrame:
    rng = np.random.default_rng(20260722)
    index = pd.bdate_range("2020-01-01", periods=520)
    leaf_a = rng.normal(0.00070, 0.0130, len(index))
    leaf_b = rng.normal(0.00015, 0.0035, len(index))
    leaf_proxy = rng.normal(0.00040, 0.0080, len(index))
    middle_proxy = rng.normal(0.00035, 0.0065, len(index))
    bond = rng.normal(0.00012, 0.0028, len(index))
    return pd.DataFrame(
        {
            "ticker:LEAF_A": leaf_a,
            "ticker:LEAF_B": leaf_b,
            "ticker:LEAF_PROXY": leaf_proxy,
            "ticker:MIDDLE_PROXY": middle_proxy,
            "ticker:BOND": bond,
        },
        index=index,
    )


def _three_level_model() -> tuple[V2Model, pd.DataFrame]:
    """Root -> Middle sleeve -> Leaf sleeve, each with its own father_proxy TEV."""

    returns = _returns()
    leaf = V2Node(
        id="leaf",
        name="Leaf sleeve",
        instruments=["ticker:LEAF_A", "ticker:LEAF_B", "ticker:LEAF_PROXY"],
        children=[],
        proxy="ticker:LEAF_PROXY",
        objective="max_return",
        constraints=V2Constraints(
            volatility_reference="father_proxy",
            max_tracking_error=0.15,
            tracking_error_reference="father_proxy",
        ),
    )
    middle = V2Node(
        id="middle",
        name="Middle sleeve",
        instruments=["ticker:MIDDLE_PROXY"],
        children=[leaf],
        proxy="ticker:MIDDLE_PROXY",
        objective="max_return",
        constraints=V2Constraints(
            volatility_reference="father_proxy",
            max_tracking_error=0.15,
            tracking_error_reference="father_proxy",
        ),
    )
    root = V2Node(
        id="root",
        name="Root",
        instruments=["ticker:BOND"],
        children=[middle],
        proxy=None,
        objective="max_return",
        constraints=V2Constraints(
            volatility_reference="benchmark",
            max_tracking_error=0.15,
            tracking_error_reference="benchmark",
        ),
    )
    model = V2Model(
        root=root,
        benchmark=V2Benchmark(
            name="B0",
            weights={"ticker:MIDDLE_PROXY": 0.7, "ticker:BOND": 0.3},
        ),
        reference_currency="USD",
    )
    return model, returns


def test_current_backward_child_is_synthetic_father_is_raw_baseline() -> None:
    """Required test #1 baseline, at a non-root nesting depth (3 levels).

    Middle's own local solve during the backward pass must use Leaf's
    *synthetic* series as the candidate column, while Middle's own
    father_proxy TEV/volatility reference stays the *raw* MIDDLE_PROXY series
    regardless of that substitution.
    """

    model, returns = _three_level_model()
    estimate = HierarchicalV2Estimator().estimate(
        model, returns, mode="forward_backward", periods_per_year=252.0
    )
    middle_result = estimate.node_results["Middle sleeve"]
    leaf_result = estimate.node_results["Leaf sleeve"]

    raw_middle_proxy_vol = returns["ticker:MIDDLE_PROXY"].std(ddof=1) * np.sqrt(252.0)
    assert middle_result.audit.target_volatility == pytest.approx(
        raw_middle_proxy_vol, abs=1e-10
    )
    raw_middle_tev = (
        (middle_result.synthetic_returns - returns["ticker:MIDDLE_PROXY"])
        .std(ddof=1)
        * np.sqrt(252.0)
    )
    assert middle_result.audit.actual_tracking_error == pytest.approx(
        raw_middle_tev, abs=1e-10
    )
    # Leaf's synthetic diverges materially from its own raw proxy (otherwise this
    # test would not be exercising the substitution at all).
    leaf_divergence = (
        (leaf_result.synthetic_returns - returns["ticker:LEAF_PROXY"]).std(ddof=1)
        * np.sqrt(252.0)
    )
    assert leaf_divergence > 1e-4


def test_current_father_reference_ignores_proxy_vs_synthetic_divergence() -> None:
    """Required test #2 baseline: Middle's TEV against its own father_proxy is
    computed against the raw proxy series even though Middle's synthetic
    return (built from Leaf's synthetic sleeve) is numerically very different
    from that raw proxy.
    """

    model, returns = _three_level_model()
    estimate = HierarchicalV2Estimator().estimate(
        model, returns, mode="forward_backward", periods_per_year=252.0
    )
    middle_result = estimate.node_results["Middle sleeve"]
    divergence = (
        (middle_result.synthetic_returns - returns["ticker:MIDDLE_PROXY"])
        .std(ddof=1)
        * np.sqrt(252.0)
    )
    # The synthetic sleeve and its raw father proxy are not the same series...
    assert divergence > 1e-4
    # ...yet the audited TEV is exactly that divergence against the RAW proxy,
    # never against some other (e.g. root benchmark, or a smoothed) reference.
    assert middle_result.audit.actual_tracking_error == pytest.approx(
        divergence, abs=1e-10
    )


def test_current_direct_bottom_up_vs_forward_backward_relationship() -> None:
    """Baseline for required test #12 (direct-bottom-up == final backward).

    Today this is only an *expectation*, not an enforced invariant (per
    `docs/optimizer-remediation-plan.md`: "A regression establishing this
    equivalence is required before it is used as a formal methodological
    claim"). This test records today's actual relationship without asserting
    strict equality; Phase 4/5 should tighten this into a real equality
    assertion once flat-mode independence (current bug, see
    `test_current_flat_mode_fails_if_any_forward_node_fails` below) is fixed.
    """

    model, returns = _three_level_model()
    estimator = HierarchicalV2Estimator()
    forward_backward = estimator.estimate(
        model, returns, mode="forward_backward", periods_per_year=252.0
    )
    flat = estimator.estimate(model, returns, mode="flat", periods_per_year=252.0)

    # Both succeed and are fully invested; they are not required to (and today
    # generally will not) agree exactly, since flat solves a single global
    # frame while forward_backward composes nested local solves.
    assert sum(forward_backward.terminal_weights.values()) == pytest.approx(1.0)
    assert sum(flat.terminal_weights.values()) == pytest.approx(1.0)


def test_current_same_estimator_resolution_forward_vs_backward() -> None:
    """Required test #6, now a strict equality assertion (Phase 4).

    Before Phase 4, `reference_weights` fed to the mean estimator was derived
    from whichever risk reference resolved truthy first (`hierarchy.py`'s old
    `target_weights or cap_weights or tracking_weights`), which happened to
    still agree between passes on this fixture but via the coupled
    mechanism this remediation removes. Now that hierarchy.py passes only a
    genuine, independent mean reference (`none` here, since this fixture
    declares no `mean_reference_kind`), `mean_estimator="auto"` resolves to
    `bayes_stein` identically in both passes for the right reason: no
    complete mean reference exists in either.
    """

    model, returns = _three_level_model()
    estimator = HierarchicalV2Estimator()
    forward_only = estimator.estimate(
        model, returns, mode="forward", periods_per_year=252.0
    )
    forward_backward = estimator.estimate(
        model, returns, mode="forward_backward", periods_per_year=252.0
    )
    root_forward_audit = forward_only.node_results["Root"].audit
    root_backward_audit = forward_backward.node_results["Root"].audit
    assert root_forward_audit.resolved_mean_estimator == "bayes_stein"
    assert root_backward_audit.resolved_mean_estimator == "bayes_stein"
    assert root_forward_audit.mean_reference_source == "none"
    assert root_backward_audit.mean_reference_source == "none"


def test_current_auto_mean_estimator_special_cases_are_pinned() -> None:
    """Locks in the two existing `mean_estimator="auto"` special cases so
    Phase 4 cannot silently remove them (user has confirmed: these must keep
    winning over the new general mean_reference-based auto rule).
    """

    returns = _returns()
    frame = returns[["ticker:LEAF_A", "ticker:LEAF_B", "ticker:LEAF_PROXY"]]

    _, cash_audit = V2LocalOptimizer().solve(
        frame,
        objective="min_risk",
        constraints=V2Constraints(mean_estimator="auto", cash_enabled=True),
        periods_per_year=252.0,
        target_reference_series=None,
        cap_reference_series=None,
        tracking_reference_series=None,
        reference_weights=None,
    )
    assert cash_audit.mean_resolution_reason == "auto_for_cash_financing_uses_bayes_stein"
    assert cash_audit.resolved_mean_estimator == "bayes_stein"

    _, utility_audit = V2LocalOptimizer().solve(
        frame,
        objective="max_utility",
        constraints=V2Constraints(mean_estimator="auto"),
        periods_per_year=252.0,
        target_reference_series=None,
        cap_reference_series=None,
        tracking_reference_series=None,
        reference_weights=None,
    )
    assert (
        utility_audit.mean_resolution_reason
        == "auto_for_max_utility_avoids_reference_reconstruction"
    )
    assert utility_audit.resolved_mean_estimator == "bayes_stein"


def _unreachable_target_and_tev() -> tuple[Any, Any, Any]:
    returns = _returns()
    father = returns["ticker:LEAF_PROXY"] * 4.0  # deliberately unreachable vol target
    tev_reference = returns["ticker:LEAF_PROXY"] + np.linspace(
        -0.05, 0.05, len(returns)
    )  # deliberately unreachable TEV
    candidates = returns[["ticker:LEAF_A", "ticker:LEAF_B"]]
    return father, tev_reference, candidates


def test_hard_fail_is_the_default_and_raises_without_projection() -> None:
    """Required test groundwork: with the default `hard_fail` policy on both
    axes, an unreachable TEV limit raises directly — no nearest-feasible
    projection is attempted at all (Phase 4 replaces the old automatic
    combined-violation fallback; hard_fail is the new default per the
    confirmed methodology, `docs/optimizer-v2-clean-engine-follow-up.md` §3).
    """

    father, tev_reference, candidates = _unreachable_target_and_tev()
    with pytest.raises(V2OptimizationError, match="tracking_error_policy='hard_fail'"):
        V2LocalOptimizer().solve(
            candidates,
            objective="max_return",
            constraints=V2Constraints(
                max_tracking_error=0.001,
                tracking_error_reference="father_proxy",
            ),
            periods_per_year=252.0,
            target_reference_series=None,
            cap_reference_series=None,
            tracking_reference_series=tev_reference,
            reference_weights={"ticker:LEAF_PROXY": 1.0},
        )


def test_tev_resolved_before_volatility_in_lexicographic_fallback() -> None:
    """Required tests #10/#11: with both TEV and volatility target unreachable
    and both set to `nearest_feasible`, TEV's minimal excess is fixed first
    (stage A); the volatility stage (B) then operates only within that fixed
    TEV band; the economic objective (stage C) is optimized only once both
    minima are fixed. `constraint_stage_results` records all three stages, in
    order.
    """

    father, tev_reference, candidates = _unreachable_target_and_tev()
    _, audit = V2LocalOptimizer().solve(
        candidates,
        objective="max_return",
        constraints=V2Constraints(
            volatility_reference="father_proxy",
            volatility_target_policy="nearest_feasible",
            max_tracking_error=0.001,
            tracking_error_reference="father_proxy",
            tracking_error_policy="nearest_feasible",
        ),
        periods_per_year=252.0,
        target_reference_series=father,
        cap_reference_series=None,
        tracking_reference_series=tev_reference,
        reference_weights={"ticker:LEAF_PROXY": 1.0},
    )
    assert audit.solver_message.startswith("nearest feasible projection")
    assert audit.target_status == "nearest_feasible"
    assert audit.tracking_error_status == "nearest_feasible"
    stages = [stage["stage"] for stage in audit.constraint_stage_results]
    assert stages == ["tracking_error", "volatility", "objective"]
    tev_stage, vol_stage, objective_stage = audit.constraint_stage_results
    assert tev_stage["policy"] == "nearest_feasible"
    assert vol_stage["policy"] == "nearest_feasible"
    assert objective_stage["status"] == "optimized"
    # Stage B's achieved volatility must respect stage A's fixed TEV bound:
    # the final tracking error must not exceed what stage A already fixed.
    assert (audit.actual_tracking_error or 0.0) <= tev_stage["achieved"] * (
        252.0**0.5
    ) + 1e-6


def test_tev_hard_and_volatility_cap_hard_simultaneously() -> None:
    """Required test #9: a hard volatility cap and a hard (default) TEV
    policy are enforced simultaneously — the cap is never relaxed to
    accommodate TEV, and vice versa, when both are jointly feasible.
    """

    returns = _returns()
    father = returns["ticker:LEAF_PROXY"]
    candidates = returns[["ticker:LEAF_A", "ticker:LEAF_B", "ticker:LEAF_PROXY"]]

    weights, audit = V2LocalOptimizer().solve(
        candidates,
        objective="max_return",
        constraints=V2Constraints(
            max_volatility_reference="father_proxy",
            max_volatility=float(father.std(ddof=1) * (252.0**0.5) * 1.5),
            max_tracking_error=0.05,
            tracking_error_reference="father_proxy",
        ),
        periods_per_year=252.0,
        target_reference_series=None,
        cap_reference_series=father,
        tracking_reference_series=father,
        reference_weights={"ticker:LEAF_PROXY": 1.0},
    )
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-8)
    assert audit.actual_volatility <= (audit.volatility_cap or 0.0) + 5e-5
    assert (audit.actual_tracking_error or 0.0) <= 0.05 + 5e-5


def test_flat_mode_succeeds_when_forward_pass_fails_elsewhere_in_tree() -> None:
    """Required fix #8 (Phase 5), now asserting the fixed behavior.

    This test previously (Phase 1 characterization,
    `test_current_flat_mode_fails_if_any_forward_node_fails`) pinned the bug:
    `flat` mode was blocked by an unrelated Forward-pass failure anywhere in
    the tree, even though flat conceptually only needs one single global
    solve over terminals. Phase 5 decoupled flat from the recursive Forward
    pass (best-effort diagnostics only); this test now asserts the opposite
    of what it asserted before — deliberately inverted, not silently edited,
    per the Phase 1 test's own note.
    """

    model, returns = _three_level_model()
    # Break the leaf sleeve's forward solve: bounds that cannot sum to one.
    broken_leaf = V2Node(
        id="leaf",
        name="Leaf sleeve",
        instruments=["ticker:LEAF_A", "ticker:LEAF_B", "ticker:LEAF_PROXY"],
        children=[],
        proxy="ticker:LEAF_PROXY",
        objective="max_return",
        constraints=V2Constraints(
            max_weights={
                "ticker:LEAF_A": 0.1,
                "ticker:LEAF_B": 0.1,
                "ticker:LEAF_PROXY": 0.1,
            }
        ),
    )
    broken_middle = V2Node(
        id="middle",
        name="Middle sleeve",
        instruments=["ticker:MIDDLE_PROXY"],
        children=[broken_leaf],
        proxy="ticker:MIDDLE_PROXY",
        objective="max_return",
        constraints=V2Constraints(),
    )
    broken_root = V2Node(
        id="root",
        name="Root",
        instruments=["ticker:BOND"],
        children=[broken_middle],
        proxy=None,
        objective="max_return",
        constraints=V2Constraints(),
    )
    broken_model = V2Model(
        root=broken_root,
        benchmark=V2Benchmark(
            name="B0", weights={"ticker:MIDDLE_PROXY": 0.7, "ticker:BOND": 0.3}
        ),
        reference_currency="USD",
    )

    # The broken bounds only apply to the leaf's own forward solve; flat's
    # own solve uses the root's (unconstrained) constraints over every
    # terminal instrument, so it must succeed even though the tree's forward
    # pass cannot.
    estimate = HierarchicalV2Estimator().estimate(
        broken_model, returns, mode="flat", periods_per_year=252.0
    )
    assert sum(estimate.terminal_weights.values()) == pytest.approx(1.0, abs=1e-6)
    # No forward diagnostics are present since the forward pass failed.
    assert "Leaf sleeve" not in estimate.node_results
    assert "Global flat terminal allocation" in estimate.node_results


def test_current_reference_kind_strings_are_informal() -> None:
    """Inventory of today's valid `reference_kind` strings, the migration
    source-of-truth for Phase 2's typed `ReferencePolicy`. Also pins that
    unknown strings raise, and that the two future-iterative-mode names are
    now explicitly rejected with a dedicated, actionable message (Phase 2
    added this: previously they fell through to the generic "unsupported
    reference" error like any other typo).

    Phase 2 also renamed the ambiguous generic ``"root"`` label to the typed
    ``"forward_root_reference"`` (see `docs/optimizer-v2-clean-engine-follow-up.md`
    §4) — the plain string ``"root"`` is therefore no longer recognized here;
    this test was updated accordingly as part of that deliberate, documented
    rename (not silently, to hide a regression).
    """

    from lazyportfolio.v2.hierarchy import HierarchicalV2Estimator as _Impl

    model, returns = _three_level_model()
    frame = returns[["ticker:MIDDLE_PROXY", "ticker:BOND"]]

    for benign in ("none", "manual", "declared"):
        assert _Impl._risk_reference(model.root, model, returns, frame, benign, None) == (
            None,
            None,
        )

    with pytest.raises(V2OptimizationError, match="invalid on the root node"):
        _Impl._risk_reference(model.root, model, returns, frame, "forward_root_reference", None)

    with pytest.raises(V2OptimizationError, match="unsupported reference 'root'"):
        _Impl._risk_reference(model.root, model, returns, frame, "root", None)

    series, weights = _Impl._risk_reference(model.root, model, returns, frame, "benchmark", None)
    assert series is not None
    assert weights == dict(model.benchmark.weights)

    for reserved in ("current_parent_synthetic", "current_root_synthetic"):
        with pytest.raises(V2OptimizationError, match="reserved for a future iterative"):
            _Impl._risk_reference(model.root, model, returns, frame, reserved, None)


def _model_with_root_mean_reference(
    **mean_reference_kwargs: object,
) -> tuple[V2Model, pd.DataFrame]:
    from dataclasses import replace

    model, returns = _three_level_model()
    root = replace(
        model.root,
        constraints=replace(model.root.constraints, **mean_reference_kwargs),
    )
    return replace(model, root=root), returns


def test_complete_local_mean_reference_resolves_in_forward_and_backward() -> None:
    """Required test #4/#5: an explicit, complete `local_weights` mean
    reference resolves correctly to whatever the node's current candidate
    columns are (raw proxy in Forward, synthetic column in Backward) — Phase
    3's `_mean_reference` resolver, exercised end-to-end through a real
    solve, not just called directly.
    """

    model, returns = _model_with_root_mean_reference(
        mean_estimator="equilibrium",
        mean_reference_kind="local_weights",
        mean_reference_weights={"ticker:MIDDLE_PROXY": 0.7, "ticker:BOND": 0.3},
    )
    estimator = HierarchicalV2Estimator()
    for mode in ("forward", "forward_backward"):
        estimate = estimator.estimate(model, returns, mode=mode, periods_per_year=252.0)
        root_audit = estimate.node_results["Root"].audit
        assert root_audit.resolved_mean_estimator == "equilibrium"
        assert root_audit.mean_reference_source == "local_weights"


def test_no_invented_residual_weight_for_incomplete_local_mean_reference() -> None:
    """Required test #8: an incomplete `local_weights` mean reference (missing
    a component of the node's actual solved universe) must fail loudly, never
    fall back to inventing a zero weight for the missing sleeve.
    """

    model, returns = _model_with_root_mean_reference(
        mean_estimator="equilibrium",
        mean_reference_kind="local_weights",
        mean_reference_weights={"ticker:MIDDLE_PROXY": 1.0},
    )
    with pytest.raises(V2OptimizationError, match="missing an entry for"):
        HierarchicalV2Estimator().estimate(
            model, returns, mode="forward", periods_per_year=252.0
        )


def test_mean_reference_independent_of_configured_risk_reference() -> None:
    """Required test #7 groundwork: configuring a risk reference
    (volatility_reference='benchmark' at the root, already present in the
    fixture) does not implicitly populate a mean reference — mean_reference_
    source stays 'none' unless mean_reference_kind is explicitly set.
    """

    model, returns = _three_level_model()
    estimate = HierarchicalV2Estimator().estimate(
        model, returns, mode="forward", periods_per_year=252.0
    )
    root_audit = estimate.node_results["Root"].audit
    assert root_audit.mean_reference_source == "none"
    assert root_audit.risk_reference_source == "volatility_reference"


def test_auto_mean_estimator_resolves_equilibrium_only_with_complete_mean_reference() -> None:
    """Phase 4: `mean_estimator="auto"` (the default) resolves to `equilibrium`
    if and only if a complete mean reference is configured; otherwise
    `bayes_stein`. Neither resolution is influenced by the root's own risk
    reference (`volatility_reference="benchmark"` in this fixture).
    """

    model, returns = _three_level_model()
    estimate = HierarchicalV2Estimator().estimate(
        model, returns, mode="forward", periods_per_year=252.0
    )
    assert (
        estimate.node_results["Root"].audit.resolved_mean_estimator == "bayes_stein"
    )

    with_reference, _ = _model_with_root_mean_reference(
        mean_reference_kind="local_weights",
        mean_reference_weights={"ticker:MIDDLE_PROXY": 0.7, "ticker:BOND": 0.3},
    )
    estimate_with_reference = HierarchicalV2Estimator().estimate(
        with_reference, returns, mode="forward", periods_per_year=252.0
    )
    root_audit = estimate_with_reference.node_results["Root"].audit
    assert root_audit.resolved_mean_estimator == "equilibrium"
    assert root_audit.mean_resolution_reason == "auto_resolved_from_reference_availability"


def test_auto_special_cases_still_win_over_a_complete_mean_reference() -> None:
    """User-confirmed decision: the two existing `auto` special cases (cash/
    leverage financing active, `objective="max_utility"`) keep forcing
    `bayes_stein` even when a complete `mean_reference` is configured — they
    are checked before, and take priority over, the new general auto rule.
    """

    from dataclasses import replace

    model, returns = _model_with_root_mean_reference(
        mean_reference_kind="local_weights",
        mean_reference_weights={"ticker:MIDDLE_PROXY": 0.7, "ticker:BOND": 0.3},
    )

    financed_root = replace(
        model.root, constraints=replace(model.root.constraints, cash_enabled=True)
    )
    financed_model = replace(model, root=financed_root)
    financed_estimate = HierarchicalV2Estimator().estimate(
        financed_model, returns, mode="forward", periods_per_year=252.0
    )
    financed_audit = financed_estimate.node_results["Root"].audit
    assert financed_audit.resolved_mean_estimator == "bayes_stein"
    assert (
        financed_audit.mean_resolution_reason
        == "auto_for_cash_financing_uses_bayes_stein"
    )

    utility_root = replace(
        model.root, objective="max_utility", constraints=model.root.constraints
    )
    utility_model = replace(model, root=utility_root)
    utility_estimate = HierarchicalV2Estimator().estimate(
        utility_model, returns, mode="forward", periods_per_year=252.0
    )
    utility_audit = utility_estimate.node_results["Root"].audit
    assert utility_audit.resolved_mean_estimator == "bayes_stein"
    assert (
        utility_audit.mean_resolution_reason
        == "auto_for_max_utility_avoids_reference_reconstruction"
    )


def test_direct_bottom_up_equals_final_backward_within_tolerance() -> None:
    """Required test #12: `estimate_direct_bottom_up` (solves every leaf
    directly, bypassing the recursive Forward pass entirely - no
    `_solve_forward_root_first` call at all) produces the same final result
    as `forward_backward`'s own backward composition, within numerical
    tolerance. This is the "direct bottom-up" resolver/harness the
    methodology docs require: proof that Forward is a diagnostic, never a
    hidden dependency of the final backward result (for a tree where no node
    requests `forward_root_reference`, which none in this fixture do).
    """

    model, returns = _three_level_model()
    estimator = HierarchicalV2Estimator()

    direct = estimator.estimate_direct_bottom_up(model, returns, 252.0)
    full_estimate = estimator.estimate(
        model, returns, mode="forward_backward", periods_per_year=252.0
    )

    assert direct.terminal_weights == pytest.approx(
        full_estimate.terminal_weights, abs=1e-9
    )
    for name in ("Leaf sleeve", "Middle sleeve", "Root"):
        assert direct.node_results[name].local_weights == pytest.approx(
            full_estimate.node_results[name].local_weights, abs=1e-9
        )


def test_point_estimate_and_backtest_resolve_mean_reference_identically() -> None:
    """Required test #16: `HierarchicalV2Backtester.run` calls the exact same
    `HierarchicalV2Estimator.estimate` per fold — no parallel/duplicate
    reference-resolution path exists in `backtest.py`. Confirm a fold's
    resolved mean/risk reference provenance matches a direct point estimate
    on that fold's own training window.
    """

    from dataclasses import replace

    from lazyportfolio.hierarchical_v2 import HierarchicalV2Backtester

    model, _ = _model_with_root_mean_reference(
        mean_estimator="equilibrium",
        mean_reference_kind="local_weights",
        mean_reference_weights={"ticker:MIDDLE_PROXY": 0.7, "ticker:BOND": 0.3},
        volatility_target_policy="nearest_feasible",
        tracking_error_policy="nearest_feasible",
    )
    # Rolling folds can occasionally make the middle/leaf sleeves' own
    # father-matching target/TEV marginally infeasible; tolerate that with
    # nearest-feasible projection, as a real backtest config would.
    middle = model.root.children[0]
    leaf = middle.children[0]
    lenient_leaf = replace(
        leaf,
        constraints=replace(
            leaf.constraints,
            volatility_target_policy="nearest_feasible",
            tracking_error_policy="nearest_feasible",
        ),
    )
    lenient_middle = replace(
        middle,
        children=[lenient_leaf],
        constraints=replace(
            middle.constraints,
            volatility_target_policy="nearest_feasible",
            tracking_error_policy="nearest_feasible",
        ),
    )
    model = replace(model, root=replace(model.root, children=[lenient_middle]))

    rng = np.random.default_rng(20260723)
    index = pd.bdate_range("2020-01-01", periods=400)
    daily = pd.DataFrame(
        {
            "ticker:LEAF_A": rng.normal(0.0006, 0.012, len(index)),
            "ticker:LEAF_B": rng.normal(0.0002, 0.004, len(index)),
            "ticker:LEAF_PROXY": rng.normal(0.0004, 0.008, len(index)),
            "ticker:MIDDLE_PROXY": rng.normal(0.00035, 0.0065, len(index)),
            "ticker:BOND": rng.normal(0.00012, 0.0028, len(index)),
        },
        index=index,
    )

    report = HierarchicalV2Backtester().run(
        model,
        daily,
        mode="forward",
        train_size=60,
        estimation_frequency="W",
        rebalance_frequency="M",
    )
    fold = report.folds[0]
    fold_root_audit = fold.audits["Root"]

    train = daily.loc[daily.index <= fold.training_end].tail(60).resample(
        "W-FRI"
    ).apply(lambda values: (1.0 + values).prod() - 1.0)
    point_estimate = HierarchicalV2Estimator().estimate(
        model, train, mode="forward", periods_per_year=52.0
    )
    point_root_audit = point_estimate.node_results["Root"].audit

    assert fold_root_audit.mean_reference_source == point_root_audit.mean_reference_source
    assert (
        fold_root_audit.resolved_mean_estimator
        == point_root_audit.resolved_mean_estimator
    )
    assert fold_root_audit.risk_reference_source == point_root_audit.risk_reference_source


def test_mean_reference_kind_benchmark_resolves_in_forward_and_backward() -> None:
    """Regression for a review finding: mean_reference_kind="benchmark" used
    to require the raw benchmark tickers to literally be columns in the
    node's local frame, which only holds in Forward - in Backward the
    columns are the child's *_SYNTH names, so it always raised. It must now
    resolve via component identity in both passes identically.
    """

    model, returns = _model_with_root_mean_reference(
        mean_estimator="equilibrium", mean_reference_kind="benchmark",
    )
    estimator = HierarchicalV2Estimator()
    for mode in ("forward", "forward_backward"):
        estimate = estimator.estimate(model, returns, mode=mode, periods_per_year=252.0)
        root_audit = estimate.node_results["Root"].audit
        assert root_audit.resolved_mean_estimator == "equilibrium"
        assert root_audit.mean_reference_source == "benchmark"


def test_mean_reference_kind_rejects_father_proxy_at_solve_time() -> None:
    """father_proxy has no coherent meaning as a mean reference (a node's own
    proxy is never one of its own candidate columns); confirm this is
    rejected even if a caller bypasses config validation and constructs
    V2Constraints directly.
    """

    from dataclasses import replace

    model, returns = _model_with_root_mean_reference(
        mean_estimator="equilibrium", mean_reference_kind="benchmark",
    )
    broken_root = replace(
        model.root,
        constraints=replace(model.root.constraints, mean_reference_kind="father_proxy"),
    )
    broken_model = replace(model, root=broken_root)
    with pytest.raises(V2OptimizationError, match="unsupported mean_reference_kind"):
        HierarchicalV2Estimator().estimate(
            broken_model, returns, mode="forward", periods_per_year=252.0
        )


def test_sleeve_absent_from_benchmark_uses_complete_local_weights() -> None:
    """Required scenario from the methodology doc: root candidates are
    Equity/Bonds/Gold but B0 only declares ACWI/AGG (Gold is absent). A
    complete local_weights mean reference covering all three (including the
    absent-from-B0 Gold sleeve) must resolve equilibrium correctly in both
    passes, without inventing or dropping any weight, and without changing
    the raw B0 TEV/volatility reference.
    """

    returns = _returns().rename(columns={"ticker:MIDDLE_PROXY": "ticker:GOLD_PROXY"})
    gold = V2Node(
        id="gold",
        name="Gold sleeve",
        instruments=["ticker:GOLD_PROXY"],
        children=[],
        proxy="ticker:GOLD_PROXY",
        objective="max_return",
        constraints=V2Constraints(),
    )
    equity = V2Node(
        id="leaf",
        name="Leaf sleeve",
        instruments=["ticker:LEAF_A", "ticker:LEAF_B", "ticker:LEAF_PROXY"],
        children=[],
        proxy="ticker:LEAF_PROXY",
        objective="max_return",
        constraints=V2Constraints(),
    )
    root = V2Node(
        id="root",
        name="Root",
        instruments=["ticker:BOND"],
        children=[equity, gold],
        proxy=None,
        objective="min_risk",
        constraints=V2Constraints(
            volatility_reference="benchmark",
            mean_estimator="equilibrium",
            mean_reference_kind="local_weights",
            mean_reference_weights={
                "ticker:LEAF_PROXY": 0.6,
                "ticker:GOLD_PROXY": 0.1,
                "ticker:BOND": 0.3,
            },
        ),
    )
    model = V2Model(
        root=root,
        benchmark=V2Benchmark(
            name="B0",
            # Gold is absent from B0 - only Equity (via its proxy) and Bond.
            weights={"ticker:LEAF_PROXY": 0.7, "ticker:BOND": 0.3},
        ),
        reference_currency="USD",
    )
    estimator = HierarchicalV2Estimator()
    raw_b0 = 0.7 * returns["ticker:LEAF_PROXY"] + 0.3 * returns["ticker:BOND"]
    for mode in ("forward", "forward_backward"):
        estimate = estimator.estimate(model, returns, mode=mode, periods_per_year=252.0)
        root_audit = estimate.node_results["Root"].audit
        assert root_audit.resolved_mean_estimator == "equilibrium"
        assert root_audit.mean_reference_source == "local_weights"
        # The risk reference (B0) stays raw and un-rescaled by the mean
        # reference, in *both* passes (Backward substitutes child candidates,
        # never father/B0).
        assert root_audit.target_volatility == pytest.approx(
            raw_b0.std(ddof=1) * np.sqrt(252.0), abs=1e-10
        )


def test_explicit_zero_risk_free_rate_end_to_end() -> None:
    """Comprehensive end-to-end chain for an explicit child RF=0.0 override
    while the root declares 3%: point estimate, backtest fold audits, and
    financing rates must all agree it is a *node*-sourced explicit zero, not
    an inherited or default value - and a positive cash position must accrue
    exactly zero return while borrowing costs only the spread.
    """

    from lazyportfolio.hierarchical_v2 import HierarchicalV2Backtester

    rng = np.random.default_rng(20260724)
    index = pd.bdate_range("2020-01-01", periods=400)
    daily = pd.DataFrame(
        {
            "ticker:A": rng.normal(0.0006, 0.012, len(index)),
            "ticker:B": rng.normal(0.0003, 0.006, len(index)),
            "ticker:EQUITY": rng.normal(0.0004, 0.009, len(index)),
        },
        index=index,
    )
    child = V2Node(
        id="child",
        name="Zero RF sleeve",
        instruments=["ticker:A", "ticker:B"],
        children=[],
        proxy="ticker:EQUITY",
        objective="max_return",
        constraints=V2Constraints(
            cash_enabled=True,
            # max_leverage stays at 1.0 (no borrowing regime attempted at
            # all) and risky exposure is capped well below full investment,
            # so a positive cash (lending) position is *forced*, not merely
            # possible - the regime is deterministic, not solver-dependent.
            max_weights={"ticker:A": 0.3, "ticker:B": 0.3},
            borrow_spread_bps=50.0,
            risk_free_rate=0.0,  # explicit override, root declares 3%
        ),
    )
    root = V2Node(
        id="root",
        name="Root",
        instruments=[],
        children=[child],
        proxy=None,
        objective="max_return",
        constraints=V2Constraints(risk_free_rate=0.03),
    )
    model = V2Model(
        root=root,
        benchmark=V2Benchmark(name="B0", weights={"ticker:EQUITY": 1.0}),
        reference_currency="USD",
    )

    estimate = HierarchicalV2Estimator().estimate(
        model, daily, mode="forward", periods_per_year=252.0
    )
    child_audit = estimate.node_results["Zero RF sleeve"].audit
    root_audit = estimate.node_results["Root"].audit

    assert child_audit.risk_free_rate == 0.0
    assert child_audit.risk_free_rate_source == "node"
    assert child_audit.cash_lending_rate == 0.0
    # Borrowing costs only the declared spread when the local RF is exactly 0.
    assert child_audit.cash_borrowing_rate == pytest.approx(0.005)
    assert root_audit.risk_free_rate == pytest.approx(0.03)
    assert root_audit.risk_free_rate_source == "node"

    # A positive local cash position is forced (risky exposure capped at
    # 0.6, no leverage available) and accrues exactly zero return, never the
    # root's 3% - this is deterministic, not conditional on what the solver
    # happens to choose.
    assert child_audit.financing_regime == "cash_lending"
    assert child_audit.cash_weight == pytest.approx(0.4, abs=1e-6)
    assert child_audit.cash_lending_rate == 0.0
    child_result = estimate.node_results["Zero RF sleeve"]
    cash_return_contribution = child_result.local_weights.get("cash:RF", 0.0) * (
        child_audit.cash_lending_rate / 252.0
    )
    assert cash_return_contribution == 0.0

    # Same chain through the walk-forward backtester: fold audits must agree.
    report = HierarchicalV2Backtester().run(
        model, daily, mode="forward", train_size=60,
        estimation_frequency="W", rebalance_frequency="M",
    )
    fold_child_audit = report.folds[0].audits["Zero RF sleeve"]
    assert fold_child_audit.risk_free_rate == 0.0
    assert fold_child_audit.risk_free_rate_source == "node"
    assert fold_child_audit.cash_lending_rate == 0.0
    assert fold_child_audit.cash_borrowing_rate == pytest.approx(0.005)

    # Point estimate and backtest resolve the same provenance (required test #16
    # territory) for this RF-specific chain too.
    assert (
        fold_child_audit.risk_free_rate_source == child_audit.risk_free_rate_source
    )
