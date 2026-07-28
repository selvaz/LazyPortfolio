from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lazyportfolio.hierarchical_v2 import (
    HierarchicalV2Backtester,
    HierarchicalV2Estimator,
    V2Benchmark,
    V2Constraints,
    V2LocalOptimizer,
    V2Model,
    V2Node,
    V2OptimizationError,
    V2View,
)


def _returns() -> pd.DataFrame:
    rng = np.random.default_rng(20260720)
    index = pd.bdate_range("2020-01-01", periods=520)
    low = rng.normal(0.00010, 0.0030, len(index))
    high = rng.normal(0.00065, 0.0120, len(index))
    middle = rng.normal(0.00030, 0.0070, len(index))
    father = 0.55 * high + 0.35 * low + 0.10 * middle
    return pd.DataFrame(
        {"ticker:HIGH": high, "ticker:LOW": low, "ticker:MIDDLE": middle,
         "ticker:FATHER": father},
        index=index,
    )


def test_local_solver_audits_father_target_tev_and_series_bounds() -> None:
    returns = _returns()
    father = returns["ticker:FATHER"]
    weights, audit = V2LocalOptimizer().solve(
        returns,
        objective="max_return",
        constraints=V2Constraints(
            min_weights={"ticker:LOW": 0.10},
            max_weights={"ticker:HIGH": 0.45},
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

    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-8)
    assert audit.actual_volatility == pytest.approx(audit.target_volatility, abs=5e-5)
    assert audit.actual_tracking_error is not None
    assert audit.actual_tracking_error <= 0.03001
    assert weights["ticker:LOW"] >= 0.099999
    assert weights["ticker:HIGH"] <= 0.450001


def test_local_solver_audits_root_cap_and_rejects_impossible_bounds() -> None:
    returns = _returns()
    father = returns["ticker:FATHER"]
    _, audit = V2LocalOptimizer().solve(
        returns.drop(columns="ticker:FATHER"),
        objective="min_risk",
        constraints=V2Constraints(max_volatility_reference="forward_root_reference"),
        periods_per_year=252.0,
        target_reference_series=None,
        cap_reference_series=father,
        tracking_reference_series=None,
        reference_weights=None,
    )
    assert audit.actual_volatility <= (audit.volatility_cap or 0.0) + 5e-5
    assert audit.target_volatility is None

    with pytest.raises(V2OptimizationError, match="bounds cannot sum to one"):
        V2LocalOptimizer().solve(
            returns[["ticker:HIGH", "ticker:LOW"]],
            objective="max_return",
            constraints=V2Constraints(
                max_weights={"ticker:HIGH": 0.40, "ticker:LOW": 0.40}
            ),
            periods_per_year=252.0,
            target_reference_series=None,
            cap_reference_series=None,
            tracking_reference_series=None,
            reference_weights=None,
        )


def test_reference_is_not_added_and_unreachable_volatility_uses_nearest_point() -> None:
    returns = _returns()
    father = returns["ticker:FATHER"] * 3.0
    candidates = returns[["ticker:HIGH", "ticker:LOW", "ticker:MIDDLE"]]

    weights, audit = V2LocalOptimizer().solve(
        candidates,
        objective="max_return",
        constraints=V2Constraints(
            volatility_reference="father_proxy",
            volatility_target_policy="nearest_feasible",
        ),
        periods_per_year=252.0,
        target_reference_series=father,
        cap_reference_series=None,
        tracking_reference_series=None,
        reference_weights={"ticker:FATHER": 1.0},
    )

    assert "ticker:FATHER" not in weights
    assert audit.target_status == "nearest_feasible"
    assert audit.actual_volatility < (audit.target_volatility or 0.0)
    assert audit.solver_message.startswith("nearest feasible projection")


def test_unreachable_tev_limit_uses_minimum_excess_without_inserting_father() -> None:
    returns = _returns()
    father = returns["ticker:FATHER"] + np.linspace(-0.02, 0.02, len(returns))
    candidates = returns[["ticker:HIGH", "ticker:LOW", "ticker:MIDDLE"]]

    weights, audit = V2LocalOptimizer().solve(
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
        tracking_reference_series=father,
        reference_weights={"ticker:FATHER": 1.0},
    )

    assert "ticker:FATHER" not in weights
    assert audit.tracking_error_status == "nearest_feasible"
    assert (audit.actual_tracking_error or 0.0) > 0.001
    assert audit.solver_message.startswith("nearest feasible projection")


def test_backward_root_uses_raw_benchmark_and_exposes_synthetic_diagnostic() -> None:
    returns = _returns().rename(
        columns={
            "ticker:HIGH": "ticker:CHILD_A",
            "ticker:LOW": "ticker:CHILD_B",
            "ticker:MIDDLE": "ticker:BOND",
            "ticker:FATHER": "ticker:EQUITY",
        }
    )
    child = V2Node(
        id="child",
        name="Equity sleeve",
        instruments=["ticker:CHILD_A", "ticker:CHILD_B", "ticker:EQUITY"],
        children=[],
        proxy="ticker:EQUITY",
        objective="max_return",
        constraints=V2Constraints(volatility_reference="father_proxy"),
    )
    root = V2Node(
        id="root",
        name="Root",
        instruments=["ticker:BOND"],
        children=[child],
        proxy=None,
        objective="max_return",
        constraints=V2Constraints(
            volatility_reference="benchmark",
            max_tracking_error=0.03,
            tracking_error_reference="benchmark",
        ),
    )
    model = V2Model(
        root=root,
        benchmark=V2Benchmark(
            name="B0",
            weights={"ticker:EQUITY": 0.7, "ticker:BOND": 0.3},
        ),
    )

    estimate = HierarchicalV2Estimator().estimate(
        model, returns, mode="forward_backward", periods_per_year=252.0
    )
    b0_raw = 0.7 * returns["ticker:EQUITY"] + 0.3 * returns["ticker:BOND"]
    root_result = estimate.node_results["Root"]

    assert root_result.audit.target_volatility == pytest.approx(
        b0_raw.std(ddof=1) * np.sqrt(252.0), abs=1e-10
    )
    assert root_result.audit.actual_tracking_error == pytest.approx(
        (root_result.synthetic_returns - b0_raw).std(ddof=1) * np.sqrt(252.0),
        abs=1e-10,
    )
    assert sum(estimate.synthetic_benchmark_weights.values()) == pytest.approx(1.0)
    expected_synthetic = {"ticker:BOND": 0.3}
    for instrument, weight in estimate.node_results["Equity sleeve"].terminal_weights.items():
        expected_synthetic[instrument] = expected_synthetic.get(instrument, 0.0) + 0.7 * weight
    assert estimate.synthetic_benchmark_weights == pytest.approx(expected_synthetic)


def _hierarchical_fixture() -> tuple[V2Model, pd.DataFrame]:
    returns = _returns().rename(
        columns={
            "ticker:HIGH": "ticker:CHILD_A",
            "ticker:LOW": "ticker:CHILD_B",
            "ticker:MIDDLE": "ticker:BOND",
            "ticker:FATHER": "ticker:EQUITY",
        }
    )
    child = V2Node(
        id="child",
        name="Equity sleeve",
        instruments=["ticker:CHILD_A", "ticker:CHILD_B", "ticker:EQUITY"],
        children=[],
        proxy="ticker:EQUITY",
        objective="max_ratio",
        constraints=V2Constraints(
            per_asset_cap=0.8,
            volatility_reference="father_proxy",
            max_tracking_error=0.08,
            tracking_error_reference="father_proxy",
            # A rolling walk-forward backtest can hit a fold where matching
            # father's exact realized volatility, or an 8% TEV, is only
            # marginally infeasible on a short, noisy window; tolerate that
            # with a nearest-feasible projection rather than aborting the
            # whole backtest (both policies default to hard_fail since the
            # clean-engine follow-up remediation).
            volatility_target_policy="nearest_feasible",
            tracking_error_policy="nearest_feasible",
        ),
    )
    root = V2Node(
        id="root",
        name="Root",
        instruments=["ticker:BOND"],
        children=[child],
        proxy=None,
        objective="max_return",
        constraints=V2Constraints(
            volatility_reference="benchmark",
            max_tracking_error=0.08,
            tracking_error_reference="benchmark",
            volatility_target_policy="nearest_feasible",
            tracking_error_policy="nearest_feasible",
        ),
    )
    model = V2Model(
        root=root,
        benchmark=V2Benchmark(
            name="B0",
            weights={"ticker:EQUITY": 0.7, "ticker:BOND": 0.3},
        ),
    )
    return model, returns


def test_model_from_config_normalizes_full_contract() -> None:
    config = {
        "root_id": "root",
        "nodes": [
            {
                "id": "root",
                "name": "Configured root",
                "instruments": ["agg"],
                "children": ["equity"],
                "goal": {"objective": "max_return"},
                "constraints": {
                    "min_weights": {"agg": "0.1", "ignored": ""},
                    "max_weights": {"agg": "0.9"},
                    "per_asset_cap": "0.95",
                    "vol_target": "0.10",
                    "max_volatility": "0.12",
                    "max_tracking_error": "0.03",
                    "tracking_error_reference": "benchmark",
                },
            },
            {
                "id": "equity",
                "instruments": ["spy", "vgk"],
                "children": [],
                "proxy": "acwi",
                "constraints": {},
            },
        ],
        "backtest": {
            "benchmark": {
                "name": "70/30",
                "weights": {"acwi": "0.7", "agg": "0.3"},
            }
        },
    }

    model = V2Model.from_config(config)

    assert [node.id for node in model.root.walk()] == ["root", "equity"]
    assert model.root.terminal_instruments() == ["ticker:AGG", "ticker:SPY", "ticker:VGK"]
    assert model.root.constraints.volatility_reference == "manual"
    assert model.root.constraints.max_volatility_reference == "manual"
    assert model.root.constraints.min_weights == {"ticker:AGG": 0.1}
    assert model.root.constraints.max_weights == {"ticker:AGG": 0.9}
    assert model.root.constraints.per_asset_cap == 0.95
    assert model.root.children[0].name == "equity"
    assert model.root.children[0].proxy == "ticker:ACWI"
    assert model.benchmark.weights == {"ticker:ACWI": 0.7, "ticker:AGG": 0.3}


@pytest.mark.parametrize("mode", ["flat", "forward", "forward_backward"])
def test_walk_forward_backtester_reconciles_all_public_modes(mode: str) -> None:
    model, daily = _hierarchical_fixture()

    report = HierarchicalV2Backtester().run(
        model,
        daily,
        mode=mode,  # type: ignore[arg-type]
        train_size=8,
        estimation_frequency="W",
        rebalance_frequency="M",
        transaction_cost_bps=7.0,
        include_partial_last_period=True,
        capture_audit_series=True,
    )

    assert report.folds
    assert set(report.metrics) == set(report.curves)
    assert {len(curve) for curve in report.curves.values()} == {
        report.metrics["FINAL"]["n_obs"]
    }
    assert report.transaction_cost_paid["FINAL"] > 0
    first = report.folds[0]
    assert "B0" in first.targets
    assert "FINAL" in first.targets
    assert "LOCAL:Root" in first.targets
    assert "FATHER:Equity sleeve" in first.targets
    assert "REFERENCE:B0_RAW" in first.estimation_series
    assert "REFERENCE:FATHER:Equity sleeve" in first.estimation_series
    assert "RESULT_OUTPUT:Root" in first.estimation_series
    assert any(name.startswith("RAW:") for name in first.estimation_series)
    assert report.metrics["FINAL"]["max_drawdown"] <= 0

    if mode == "forward_backward":
        assert "B0_SYNTH" in report.curves
        assert "FORWARD_FINAL" in report.curves
        assert "FORWARD_LOCAL:Root" in first.targets
        assert "BACKWARD_INPUT:ticker:EQUITY_SYNTH" in first.estimation_series
        assert "DIAGNOSTIC:B0_SYNTH" in first.estimation_series
        assert "FORWARD_OUTPUT:Root" in first.estimation_series
    else:
        assert "B0_SYNTH" not in report.curves


def test_backtester_rejects_a_window_without_complete_folds() -> None:
    model, daily = _hierarchical_fixture()

    with pytest.raises(V2OptimizationError, match="produced no complete folds"):
        HierarchicalV2Backtester().run(
            model,
            daily.head(20),
            mode="forward",
            train_size=104,
        )


def test_local_solver_validates_reference_and_input_data() -> None:
    returns = _returns()
    candidates = returns[["ticker:HIGH", "ticker:LOW", "ticker:MIDDLE"]]

    with pytest.raises(V2OptimizationError, match="at least three"):
        V2LocalOptimizer().solve(
            candidates.head(2),
            objective="max_ratio",
            constraints=V2Constraints(),
            periods_per_year=252.0,
            target_reference_series=None,
            cap_reference_series=None,
            tracking_reference_series=None,
            reference_weights=None,
        )
    with pytest.raises(V2OptimizationError, match="requires a reference series"):
        V2LocalOptimizer().solve(
            candidates,
            objective="max_ratio",
            constraints=V2Constraints(volatility_reference="forward_root_reference"),
            periods_per_year=252.0,
            target_reference_series=None,
            cap_reference_series=None,
            tracking_reference_series=None,
            reference_weights=None,
        )
    with pytest.raises(V2OptimizationError, match="TEV limit requires"):
        V2LocalOptimizer().solve(
            candidates,
            objective="max_ratio",
            constraints=V2Constraints(max_tracking_error=0.03),
            periods_per_year=252.0,
            target_reference_series=None,
            cap_reference_series=None,
            tracking_reference_series=None,
            reference_weights=None,
        )
    with pytest.raises(V2OptimizationError, match="invalid local min/max"):
        V2LocalOptimizer().solve(
            candidates,
            objective="max_ratio",
            constraints=V2Constraints(min_weights={"ticker:HIGH": -0.1}),
            periods_per_year=252.0,
            target_reference_series=None,
            cap_reference_series=None,
            tracking_reference_series=None,
            reference_weights=None,
        )
    incomplete_reference = returns["ticker:FATHER"].copy()
    incomplete_reference.iloc[0] = np.nan
    with pytest.raises(V2OptimizationError, match="reference series is incomplete"):
        V2LocalOptimizer().solve(
            candidates,
            objective="max_ratio",
            constraints=V2Constraints(max_tracking_error=0.03),
            periods_per_year=252.0,
            target_reference_series=None,
            cap_reference_series=None,
            tracking_reference_series=incomplete_reference,
            reference_weights=None,
        )
    _, ratio_audit = V2LocalOptimizer().solve(
        candidates,
        objective="max_ratio",
        constraints=V2Constraints(),
        periods_per_year=252.0,
        target_reference_series=None,
        cap_reference_series=None,
        tracking_reference_series=None,
        reference_weights=None,
    )
    assert ratio_audit.effective_objective == "max_ratio"
    assert np.isfinite(ratio_audit.objective_value)


def test_estimator_reference_resolution_and_local_synthetic_series() -> None:
    model, returns = _hierarchical_fixture()
    estimator = HierarchicalV2Estimator()
    child = model.root.children[0]
    frame = returns[["ticker:EQUITY", "ticker:BOND"]]

    none_series, none_weights = estimator._reference(
        child, model, returns, frame, {}, False, "manual", None
    )
    assert none_series is None and none_weights is None

    benchmark, weights = estimator._reference(
        model.root, model, returns, frame, {}, False, "benchmark", None
    )
    assert benchmark.equals(0.7 * frame["ticker:EQUITY"] + 0.3 * frame["ticker:BOND"])
    assert weights == model.benchmark.weights

    father, father_weights = estimator._reference(
        child, model, returns, frame, {}, False, "father_proxy", None
    )
    assert father.equals(returns["ticker:EQUITY"])
    assert father_weights == {"ticker:EQUITY": 1.0}

    forward_root_reference = returns["ticker:EQUITY"] * 0.5
    resolved_root, root_weights = estimator._reference(
        child, model, returns, frame, {}, False, "forward_root_reference", forward_root_reference
    )
    assert resolved_root.equals(forward_root_reference)
    assert root_weights is None

    with pytest.raises(V2OptimizationError, match="invalid on the root"):
        estimator._reference(
            model.root, model, returns, frame, {}, False,
            "forward_root_reference", forward_root_reference,
        )
    with pytest.raises(V2OptimizationError, match="not available"):
        estimator._reference(
            child, model, returns, frame, {}, False, "forward_root_reference", None
        )
    with pytest.raises(V2OptimizationError, match="requires a proxy"):
        estimator._reference(
            model.root, model, returns, frame, {}, False, "father_proxy", None
        )
    with pytest.raises(V2OptimizationError, match="unsupported reference"):
        estimator._reference(child, model, returns, frame, {}, False, "mystery", None)
    with pytest.raises(V2OptimizationError, match="benchmark reference series missing"):
        estimator._reference(
            model.root,
            model,
            returns.drop(columns="ticker:EQUITY"),
            frame[["ticker:BOND"]],
            {},
            False,
            "benchmark",
            None,
        )

    child_result = HierarchicalV2Estimator().estimate(
        model, returns, mode="forward", periods_per_year=252.0
    ).node_results[child.name]
    synthetic = estimator._local_series(
        returns,
        {"ticker:BOND": 0.4, "ticker:EQUITY_SYNTH": 0.6},
        {child.name: child_result},
        {child.name: "ticker:EQUITY_SYNTH"},
    )
    expected = 0.4 * returns["ticker:BOND"] + 0.6 * child_result.synthetic_returns
    assert synthetic.equals(expected)
    with pytest.raises(V2OptimizationError, match="cannot resolve local series"):
        estimator._local_series(returns, {"ticker:MISSING": 1.0}, {}, {})


def test_estimator_rejects_missing_child_proxy() -> None:
    model, returns = _hierarchical_fixture()
    model.root.children[0].proxy = None

    with pytest.raises(V2OptimizationError, match="child proxy is required"):
        HierarchicalV2Estimator().estimate(
            model, returns, mode="forward", periods_per_year=252.0
        )


def test_estimate_moments_shrinks_covariance_away_from_the_raw_sample() -> None:
    frame = _returns().drop(columns="ticker:FATHER")
    names = list(frame.columns)
    raw_covariance = np.cov(frame.to_numpy(dtype=float), rowvar=False, ddof=1)

    covariance, _, resolved = V2LocalOptimizer._estimate_moments(frame, names, None, "auto")

    assert covariance.shape == raw_covariance.shape
    assert not np.allclose(covariance, raw_covariance)
    assert resolved == "bayes_stein"


def test_estimate_moments_bayes_stein_shrinks_the_mean_toward_the_grand_mean() -> None:
    frame = _returns().drop(columns="ticker:FATHER")
    names = list(frame.columns)
    raw_mean = frame.to_numpy(dtype=float).mean(axis=0)

    # No reference, and a reference that does not sum to a fully-invested
    # portfolio, both fall back to Bayes-Stein shrinkage toward the grand mean.
    for reference_weights in (None, {"ticker:HIGH": 0.4}):
        _, means, resolved = V2LocalOptimizer._estimate_moments(
            frame, names, reference_weights, "auto"
        )
        assert resolved == "bayes_stein"
        assert not np.allclose(means, raw_mean)
        # Shrinkage narrows cross-sectional dispersion around the grand mean;
        # it must never widen it.
        assert np.std(means) < np.std(raw_mean)


def test_estimate_moments_uses_equilibrium_prior_from_a_full_reference() -> None:
    frame = _returns().drop(columns="ticker:FATHER")
    names = list(frame.columns)
    reference_weights = {"ticker:HIGH": 0.5, "ticker:LOW": 0.3, "ticker:MIDDLE": 0.2}

    covariance, means, resolved = V2LocalOptimizer._estimate_moments(
        frame, names, reference_weights, "auto"
    )

    assert resolved == "equilibrium"
    weights_vector = np.array([reference_weights[name] for name in names])
    expected = covariance @ weights_vector  # risk_aversion=1.0, skfolio's EquilibriumMu default
    assert means == pytest.approx(expected, rel=1e-9)


def test_estimate_moments_explicit_method_overrides_auto_selection() -> None:
    frame = _returns().drop(columns="ticker:FATHER")
    names = list(frame.columns)
    raw_mean = frame.to_numpy(dtype=float).mean(axis=0)
    reference_weights = {"ticker:HIGH": 0.5, "ticker:LOW": 0.3, "ticker:MIDDLE": 0.2}

    # Explicit "empirical" is the raw sample mean, unlike the "auto" default.
    _, empirical_means, resolved = V2LocalOptimizer._estimate_moments(
        frame, names, reference_weights, "empirical"
    )
    assert resolved == "empirical"
    assert empirical_means == pytest.approx(raw_mean)

    # Explicit "james_stein"/"bodnar_okhrin" are honoured even with a full
    # reference available (which "auto" would have resolved to "equilibrium").
    for method in ("james_stein", "bodnar_okhrin"):
        _, means, resolved = V2LocalOptimizer._estimate_moments(
            frame, names, reference_weights, method
        )
        assert resolved == method
        assert not np.allclose(means, raw_mean)

    # "equilibrium" without a usable reference is a configuration error, not
    # a silent fallback to Bayes-Stein.
    with pytest.raises(V2OptimizationError, match="requires a full"):
        V2LocalOptimizer._estimate_moments(frame, names, None, "equilibrium")

    with pytest.raises(V2OptimizationError, match="unsupported mean_estimator"):
        V2LocalOptimizer._estimate_moments(frame, names, None, "not_a_method")


def test_model_from_config_parses_mean_estimator_per_node() -> None:
    config = {
        "root_id": "root",
        "nodes": [
            {
                "id": "root",
                "proxy": "",
                "instruments": ["ACWI", "AGG"],
                "constraints": {"mean_estimator": "james_stein"},
            },
        ],
        "backtest": {"benchmark": {"weights": {"ACWI": 0.6, "AGG": 0.4}}},
    }
    model = V2Model.from_config(config)
    assert model.root.constraints.mean_estimator == "james_stein"

    default_config = {
        "root_id": "root",
        "nodes": [{"id": "root", "proxy": "", "instruments": ["ACWI"], "constraints": {}}],
        "backtest": {"benchmark": {"weights": {"ACWI": 1.0}}},
    }
    assert V2Model.from_config(default_config).root.constraints.mean_estimator == "auto"


def test_apply_views_is_a_no_op_without_declared_views() -> None:
    frame = _returns().drop(columns="ticker:FATHER")
    names = list(frame.columns)
    covariance, means, _ = V2LocalOptimizer._estimate_moments(frame, names, None, "auto")

    posterior_covariance, posterior_means, details = V2LocalOptimizer._apply_views(
        covariance, means, names, (), 0.05, 252.0
    )

    assert posterior_covariance is covariance
    assert posterior_means is means
    assert details == ()


def test_apply_views_matches_the_closed_form_single_view_blend() -> None:
    """For one view, the posterior view-return is exactly a convex blend of the
    prior and the declared view, weighted by confidence — independent of tau.
    Derivation: with K=1, the BL system collapses to
    ``posterior = (1 - confidence) * prior + confidence * Q``.
    """
    frame = _returns().drop(columns="ticker:FATHER")
    names = list(frame.columns)
    periods_per_year = 252.0
    covariance, means, _ = V2LocalOptimizer._estimate_moments(frame, names, None, "auto")
    prior_view_return = float(means[names.index("ticker:HIGH")]) * periods_per_year

    for confidence in (0.9999, 0.5, 0.001):
        view = V2View(
            instruments={"ticker:HIGH": 1.0},
            expected_return=0.25,
            confidence=confidence,
            source="test-agent",
        )
        for tau in (0.02, 0.05, 0.2):  # the closed form is tau-independent for K=1
            _, posterior_means, details = V2LocalOptimizer._apply_views(
                covariance, means, names, (view,), tau, periods_per_year
            )
            expected = (1.0 - confidence) * prior_view_return + confidence * 0.25
            posterior_view_return = (
                float(posterior_means[names.index("ticker:HIGH")]) * periods_per_year
            )
            assert posterior_view_return == pytest.approx(expected, rel=1e-6)
            assert len(details) == 1
            assert details[0]["confidence"] == confidence
            assert details[0]["source"] == "test-agent"
            assert details[0]["posterior_view_return_annualized"] == pytest.approx(
                expected, rel=1e-6
            )

    # High confidence pulls the posterior close to the view; low confidence barely moves it.
    _, high_conf_means, _ = V2LocalOptimizer._apply_views(
        covariance, means, names, (V2View({"ticker:HIGH": 1.0}, 0.25, 0.999),), 0.05,
        periods_per_year,
    )
    _, low_conf_means, _ = V2LocalOptimizer._apply_views(
        covariance, means, names, (V2View({"ticker:HIGH": 1.0}, 0.25, 0.001),), 0.05,
        periods_per_year,
    )
    idx = names.index("ticker:HIGH")
    assert abs(float(high_conf_means[idx]) * periods_per_year - 0.25) < abs(
        float(low_conf_means[idx]) * periods_per_year - 0.25
    )


def test_apply_views_rejects_invalid_input() -> None:
    frame = _returns().drop(columns="ticker:FATHER")
    names = list(frame.columns)
    covariance, means, _ = V2LocalOptimizer._estimate_moments(frame, names, None, "auto")

    for bad_confidence in (0.0, -0.1, 1.5):
        with pytest.raises(V2OptimizationError, match="confidence must be in"):
            V2LocalOptimizer._apply_views(
                covariance, means, names,
                (V2View({"ticker:HIGH": 1.0}, 0.1, bad_confidence),), 0.05, 252.0,
            )

    with pytest.raises(V2OptimizationError, match="not part of this node's solved universe"):
        V2LocalOptimizer._apply_views(
            covariance, means, names,
            (V2View({"ticker:UNKNOWN": 1.0}, 0.1, 0.5),), 0.05, 252.0,
        )

    with pytest.raises(V2OptimizationError, match="no instrument coefficients"):
        V2LocalOptimizer._apply_views(
            covariance, means, names,
            (V2View({}, 0.1, 0.5),), 0.05, 252.0,
        )

    with pytest.raises(V2OptimizationError, match="view_tau must be positive"):
        V2LocalOptimizer._apply_views(
            covariance, means, names,
            (V2View({"ticker:HIGH": 1.0}, 0.1, 0.5),), 0.0, 252.0,
        )


def test_model_from_config_parses_views_per_node() -> None:
    config = {
        "root_id": "root",
        "nodes": [
            {
                "id": "root",
                "proxy": "",
                "instruments": ["ACWI", "AGG"],
                "constraints": {
                    "views": [
                        {
                            "instruments": {"ACWI": 1.0},
                            "expected_return": 0.08,
                            "confidence": 0.6,
                            "source": "agent:macro-2026-07-21",
                        },
                        {
                            "instruments": {"ACWI": 1.0, "AGG": -1.0},
                            "expected_return": 0.02,
                            "confidence": 0.3,
                        },
                    ],
                    "view_tau": 0.1,
                },
            },
        ],
        "backtest": {"benchmark": {"weights": {"ACWI": 0.6, "AGG": 0.4}}},
    }
    model = V2Model.from_config(config)
    views = model.root.constraints.views
    assert len(views) == 2
    assert views[0].instruments == {"ticker:ACWI": 1.0}
    assert views[0].expected_return == 0.08
    assert views[0].confidence == 0.6
    assert views[0].source == "agent:macro-2026-07-21"
    assert views[1].instruments == {"ticker:ACWI": 1.0, "ticker:AGG": -1.0}
    assert views[1].source == "manual"
    assert model.root.constraints.view_tau == 0.1

    default_config = {
        "root_id": "root",
        "nodes": [{"id": "root", "proxy": "", "instruments": ["ACWI"], "constraints": {}}],
        "backtest": {"benchmark": {"weights": {"ACWI": 1.0}}},
    }
    default_constraints = V2Model.from_config(default_config).root.constraints
    assert default_constraints.views == ()
    assert default_constraints.view_tau == 0.05


def test_model_from_config_treats_empty_string_view_tau_as_default() -> None:
    """A GUI payload sends '' (not a missing key) for an unset optional field."""
    config = {
        "root_id": "root",
        "nodes": [
            {
                "id": "root",
                "proxy": "",
                "instruments": ["ACWI"],
                "constraints": {"view_tau": ""},
            },
        ],
        "backtest": {"benchmark": {"weights": {"ACWI": 1.0}}},
    }
    assert V2Model.from_config(config).root.constraints.view_tau == 0.05


def test_local_solver_end_to_end_with_a_strong_view_shifts_the_allocation() -> None:
    """A confident, bullish view on the lowest-mean asset should raise its weight
    relative to the same solve with no views declared."""
    returns = _returns().drop(columns="ticker:FATHER")

    def solve(views: tuple[V2View, ...]) -> dict[str, float]:
        weights, _ = V2LocalOptimizer().solve(
            returns,
            objective="max_return",
            constraints=V2Constraints(views=views),
            periods_per_year=252.0,
            target_reference_series=None,
            cap_reference_series=None,
            tracking_reference_series=None,
            reference_weights=None,
        )
        return weights

    baseline = solve(())
    with_view = solve(
        (V2View(instruments={"ticker:LOW": 1.0}, expected_return=0.5, confidence=0.95),)
    )
    assert with_view["ticker:LOW"] > baseline["ticker:LOW"]


def test_effective_setting_resolves_node_then_root_then_default() -> None:
    from lazyportfolio.hierarchical_v2 import _effective_setting

    assert _effective_setting(2.5, None, 1.0) == 2.5
    assert _effective_setting(2.5, 4.0, 1.0) == 2.5  # node wins over root
    assert _effective_setting(None, 4.0, 1.0) == 4.0  # root is the tree-wide default
    assert _effective_setting(None, None, 1.0) == 1.0  # hard default, never estimated


def test_solve_rejects_an_unrecognized_objective() -> None:
    returns = _returns().drop(columns="ticker:FATHER")
    with pytest.raises(V2OptimizationError, match="unsupported objective"):
        V2LocalOptimizer().solve(
            returns,
            objective="risk_budget",  # a phantom objective the old UI used to offer
            constraints=V2Constraints(),
            periods_per_year=252.0,
            target_reference_series=None,
            cap_reference_series=None,
            tracking_reference_series=None,
            reference_weights=None,
        )


def test_max_ratio_is_a_true_sharpe_ratio_using_risk_free_rate() -> None:
    returns = _returns().drop(columns="ticker:FATHER")

    def solve(risk_free_rate: float) -> tuple[dict[str, float], object]:
        return V2LocalOptimizer().solve(
            returns, objective="max_ratio", constraints=V2Constraints(), periods_per_year=252.0,
            target_reference_series=None, cap_reference_series=None,
            tracking_reference_series=None, reference_weights=None, risk_free_rate=risk_free_rate,
        )

    zero_rf_weights, _ = solve(0.0)
    weights, audit = solve(0.05)

    assert audit.risk_free_rate == 0.05
    assert audit.objective_value == pytest.approx(
        (audit.expected_return_annualized - 0.05) / audit.actual_volatility, rel=1e-6
    )
    # A positive risk-free rate is not just a constant added to the audit number:
    # dividing by volatility means it changes which portfolio maximizes the ratio.
    assert any(
        abs(weights[name] - zero_rf_weights[name]) > 1e-6 for name in weights
    )


def test_equilibrium_mean_uses_risk_aversion_and_adds_back_the_risk_free_rate() -> None:
    frame = _returns().drop(columns="ticker:FATHER")
    names = list(frame.columns)
    reference = {"ticker:HIGH": 0.5, "ticker:LOW": 0.3, "ticker:MIDDLE": 0.2}
    weights_vector = np.array([reference[name] for name in names])

    covariance, means_default, _ = V2LocalOptimizer._estimate_moments(
        frame, names, reference, "auto", risk_aversion=1.0, risk_free_periodic=0.0
    )
    assert means_default == pytest.approx(covariance @ weights_vector, rel=1e-9)

    rf_periodic = 0.05 / 252.0
    covariance2, means_with_rf, _ = V2LocalOptimizer._estimate_moments(
        frame, names, reference, "auto", risk_aversion=1.0, risk_free_periodic=rf_periodic
    )
    assert means_with_rf == pytest.approx(means_default + rf_periodic, rel=1e-9)

    _, means_higher_aversion, _ = V2LocalOptimizer._estimate_moments(
        frame, names, reference, "auto", risk_aversion=3.0, risk_free_periodic=0.0
    )
    assert means_higher_aversion == pytest.approx(3.0 * (covariance @ weights_vector), rel=1e-9)


def test_max_utility_matches_the_two_asset_closed_form() -> None:
    """Unconstrained fully-invested quadratic utility over 2 assets has an exact
    closed form; derived by hand and verified against the solver here."""
    rng = np.random.default_rng(7)
    index = pd.bdate_range("2020-01-01", periods=400)
    frame = pd.DataFrame(
        {
            "ticker:A": rng.normal(0.0006, 0.010, len(index)),
            "ticker:B": rng.normal(0.0003, 0.007, len(index)),
        },
        index=index,
    )
    names = list(frame.columns)
    periods_per_year = 252.0
    risk_free_rate = 0.02
    risk_aversion = 3.0

    covariance, means, _ = V2LocalOptimizer._estimate_moments(frame, names, None, "auto")
    excess = means - risk_free_rate / periods_per_year
    mu1, mu2 = excess[0], excess[1]
    s1, s2, s12 = covariance[0, 0], covariance[1, 1], covariance[0, 1]
    spread_variance = s1 + s2 - 2 * s12
    expected_w1 = ((mu1 - mu2) / risk_aversion + (s2 - s12)) / spread_variance

    weights, audit = V2LocalOptimizer().solve(
        frame,
        objective="max_utility",
        constraints=V2Constraints(),
        periods_per_year=periods_per_year,
        target_reference_series=None,
        cap_reference_series=None,
        tracking_reference_series=None,
        reference_weights=None,
        risk_aversion=risk_aversion,
        risk_free_rate=risk_free_rate,
    )
    assert weights["ticker:A"] == pytest.approx(expected_w1, rel=1e-4)
    assert audit.configured_objective == "max_utility"
    assert audit.risk_aversion == risk_aversion
    assert audit.risk_free_rate == risk_free_rate


def test_hrp_produces_a_valid_portfolio_respecting_bounds() -> None:
    returns = _returns().drop(columns="ticker:FATHER")
    weights, audit = V2LocalOptimizer().solve(
        returns,
        objective="hrp",
        constraints=V2Constraints(
            min_weights={"ticker:LOW": 0.15}, max_weights={"ticker:HIGH": 0.4}
        ),
        periods_per_year=252.0,
        target_reference_series=None,
        cap_reference_series=None,
        tracking_reference_series=None,
        reference_weights=None,
    )
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-8)
    assert weights["ticker:LOW"] >= 0.15 - 1e-6
    assert weights["ticker:HIGH"] <= 0.4 + 1e-6
    assert audit.configured_objective == "hrp"
    assert audit.effective_objective == "hrp"
    assert audit.target_status == "not_requested"
    assert audit.tracking_error_status == "not_requested"


def test_hrp_rejects_constraints_it_would_otherwise_silently_ignore() -> None:
    returns = _returns().drop(columns="ticker:FATHER")

    def solve(constraints: V2Constraints) -> None:
        V2LocalOptimizer().solve(
            returns, objective="hrp", constraints=constraints, periods_per_year=252.0,
            target_reference_series=None,
            cap_reference_series=None, tracking_reference_series=None, reference_weights=None,
        )

    with pytest.raises(V2OptimizationError, match="does not support a volatility target or cap"):
        solve(V2Constraints(volatility_reference="manual", volatility_target=0.1))
    with pytest.raises(V2OptimizationError, match="does not support a volatility target or cap"):
        solve(V2Constraints(max_volatility_reference="manual", max_volatility=0.1))
    with pytest.raises(V2OptimizationError, match="does not support a tracking-error limit"):
        solve(V2Constraints(max_tracking_error=0.03))
    with pytest.raises(V2OptimizationError, match="does not use an expected-return estimator"):
        solve(V2Constraints(mean_estimator="bayes_stein"))
    with pytest.raises(V2OptimizationError, match="does not use views"):
        one_view = V2View(
            instruments={"ticker:HIGH": 1.0}, expected_return=0.1, confidence=0.5
        )
        solve(V2Constraints(views=(one_view,)))


def test_model_from_config_parses_risk_aversion_and_risk_free_rate() -> None:
    config = {
        "root_id": "root",
        "nodes": [
            {
                "id": "root",
                "proxy": "",
                "instruments": ["ACWI", "AGG"],
                "constraints": {"risk_aversion": 2.5, "risk_free_rate": 0.03},
            },
        ],
        "backtest": {"benchmark": {"weights": {"ACWI": 0.6, "AGG": 0.4}}},
    }
    constraints = V2Model.from_config(config).root.constraints
    assert constraints.risk_aversion == 2.5
    assert constraints.risk_free_rate == 0.03

    default_config = {
        "root_id": "root",
        "nodes": [{"id": "root", "proxy": "", "instruments": ["ACWI"], "constraints": {}}],
        "backtest": {"benchmark": {"weights": {"ACWI": 1.0}}},
    }
    default_constraints = V2Model.from_config(default_config).root.constraints
    assert default_constraints.risk_aversion is None
    assert default_constraints.risk_free_rate is None


def test_estimator_inherits_risk_aversion_from_the_root_as_a_global_default() -> None:
    model, returns = _hierarchical_fixture()
    model.root.constraints = V2Constraints(
        volatility_reference=model.root.constraints.volatility_reference,
        max_tracking_error=model.root.constraints.max_tracking_error,
        tracking_error_reference=model.root.constraints.tracking_error_reference,
        risk_aversion=2.5,
        risk_free_rate=0.04,
    )
    estimate = HierarchicalV2Estimator().estimate(
        model, returns, mode="forward", periods_per_year=252.0
    )
    # The child does not declare its own risk_aversion/risk_free_rate: it inherits
    # the root's as the tree-wide global default, not the hard-coded 1.0 / 0.0.
    child_audit = estimate.node_results["Equity sleeve"].audit
    assert child_audit.risk_aversion == 2.5
    assert child_audit.risk_free_rate == 0.04
    root_audit = estimate.node_results["Root"].audit
    assert root_audit.risk_aversion == 2.5
    assert root_audit.risk_free_rate == 0.04


@pytest.mark.parametrize("mode", ["flat", "forward", "forward_backward"])
def test_hrp_and_max_utility_survive_a_full_tree_and_walk_forward_backtest(mode: str) -> None:
    """HRP bypasses the whole SLSQP path and max_utility is a new objective;
    neither had been exercised through the estimator/backtester, only through
    a single V2LocalOptimizer.solve() call. Confirm both compose correctly
    inside a real tree, across every mode, and through a walk-forward run."""
    model, daily = _hierarchical_fixture()
    model.root.objective = "max_utility"
    model.root.constraints = V2Constraints(
        volatility_reference=model.root.constraints.volatility_reference,
        volatility_target_policy=model.root.constraints.volatility_target_policy,
        max_tracking_error=model.root.constraints.max_tracking_error,
        tracking_error_reference=model.root.constraints.tracking_error_reference,
        tracking_error_policy=model.root.constraints.tracking_error_policy,
        risk_aversion=2.0,
        risk_free_rate=0.02,
    )
    model.root.children[0].objective = "hrp"
    model.root.children[0].constraints = V2Constraints()

    estimate = HierarchicalV2Estimator().estimate(
        model, daily.tail(60), mode=mode, periods_per_year=252.0  # type: ignore[arg-type]
    )
    assert sum(estimate.terminal_weights.values()) == pytest.approx(1.0, abs=1e-6)
    root_audit = estimate.node_results["Root"].audit
    assert root_audit.configured_objective == "max_utility"
    child_audit = estimate.node_results["Equity sleeve"].audit
    assert child_audit.configured_objective == "hrp"
    assert child_audit.target_status == "not_requested"

    report = HierarchicalV2Backtester().run(
        model, daily, mode=mode, train_size=8,  # type: ignore[arg-type]
        estimation_frequency="W", rebalance_frequency="M",
        include_partial_last_period=True,
    )
    assert report.folds
    assert sum(report.folds[0].targets["FINAL"].values()) == pytest.approx(1.0, abs=1e-6)
