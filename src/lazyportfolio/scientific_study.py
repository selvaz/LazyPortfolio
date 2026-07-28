"""Reproducible scientific comparison harness for the V2 optimizer.

The harness separates engineering validation from financial claims. It evaluates
V2 and simple baselines on the same walk-forward folds, OOS grid, cost convention
and configured risk-free rate.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lazyportfolio.calendar import (
    _annualization_factor,
    _resample_simple_returns,
)
from lazyportfolio.hierarchical_v2 import (
    HierarchicalV2Backtester,
    HierarchicalV2Estimator,
    Mode,
    V2Constraints,
    V2LocalOptimizer,
    V2Model,
    V2OptimizationError,
    _effective_setting,
)
from lazyportfolio.v2.moments import is_financing_instrument

#: All six required baseline arms (declared in `baseline_allocations`) now
#: receive full block-bootstrap + Holm-adjusted inference against V2_FINAL -
#: not just EQUAL_WEIGHT/DECLARED_BENCHMARK. Keep this in one place so the
#: inference loop and any future "what does this study actually claim"
#: assertion stay in sync by construction.
BOOTSTRAPPED_BASELINES = (
    "EQUAL_WEIGHT",
    "DECLARED_BENCHMARK",
    "SAMPLE_MIN_VARIANCE",
    "SHRUNK_FIXED_MIN_VARIANCE",
    "LEDOIT_WOLF_MIN_VARIANCE",
    "HRP_WARD_PEARSON",
)


@dataclass(frozen=True)
class ScientificStudyProtocol:
    train_size: int
    estimation_frequency: str = "W"
    rebalance_frequency: str = "M"
    transaction_cost_bps: float = 0.0
    include_partial_last_period: bool = False
    bootstrap_samples: int = 2_000
    bootstrap_block_size: int = 20
    random_seed: int = 7
    #: By default every arm's OOS index must be *exactly* identical - a
    #: silent intersection can hide that one arm lost observations (e.g. a
    #: baseline that failed to fit on some fold) without ever surfacing it.
    #: Set False to opt into the old intersect-and-continue behavior; when
    #: dropped, the count of dropped observations per arm is still recorded
    #: in ``ScientificStudyResult.dropped_observations`` for audit.
    require_identical_oos_index: bool = True


@dataclass(frozen=True)
class PairedComparison:
    candidate: str
    baseline: str
    annualized_mean_difference: float
    confidence_interval_low: float
    confidence_interval_high: float
    p_value: float
    holm_adjusted_p_value: float


@dataclass
class ScientificStudyResult:
    curves: dict[str, Any]
    metrics: dict[str, dict[str, float | int]]
    comparisons: list[PairedComparison]
    fold_count: int
    common_oos_start: Any
    common_oos_end: Any
    protocol: ScientificStudyProtocol
    #: Observations each arm lost to the common-grid intersection, keyed by
    #: arm name. Empty when every arm's index was already identical (the
    #: default, ``require_identical_oos_index=True``, never populates this -
    #: it raises instead). Only non-empty when
    #: ``protocol.require_identical_oos_index=False`` was explicitly set.
    dropped_observations: dict[str, int] = field(default_factory=dict)


class _StudyLedger:
    def __init__(self, transaction_cost_bps: float) -> None:
        self.cost_rate = transaction_cost_bps / 10_000.0
        self.weights: dict[str, float] = {}

    def rebalance(self, target: dict[str, float]) -> float:
        names = set(self.weights) | set(target)
        turnover = sum(
            abs(target.get(name, 0.0) - self.weights.get(name, 0.0))
            for name in names
        )
        self.weights = dict(target)
        return turnover * self.cost_rate

    def step(self, row: Any) -> float:
        # A financing leg (cash:RF / cash:BORROW@<node>) is a ledger-only
        # position, not a priced instrument -- it has no column in the raw
        # daily return frame. Treat it as a zero-return position: its dollar
        # amount is flat, so its *weight* still drifts with the rest of the
        # book via the shared denominator below, exactly like a real
        # un-remunerated cash balance would.
        value = sum(
            0.0 if is_financing_instrument(name) else weight * float(row[name])
            for name, weight in self.weights.items()
        )
        denominator = 1.0 + value
        if denominator != 0.0:
            self.weights = {
                name: weight * (
                    1.0 if is_financing_instrument(name) else (1.0 + float(row[name]))
                ) / denominator
                for name, weight in self.weights.items()
            }
        return value


def _bounded_minimum_variance(frame: Any) -> dict[str, float]:
    from scipy.optimize import minimize

    clean = frame.dropna(how="any")
    names = list(clean.columns)
    covariance = np.atleast_2d(
        np.cov(clean.to_numpy(dtype=float), rowvar=False, ddof=1)
    )
    start = np.full(len(names), 1.0 / len(names), dtype=float)
    result = minimize(
        lambda weights: float(weights @ covariance @ weights),
        start,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(names),
        constraints=[
            {
                "type": "eq",
                "fun": lambda weights: float(weights.sum() - 1.0),
            }
        ],
        options={"ftol": 1e-12, "maxiter": 2_000, "disp": False},
    )
    if not result.success:
        raise V2OptimizationError(
            f"sample minimum-variance baseline failed: {result.message}"
        )
    weights = np.asarray(result.x, dtype=float)
    if (
        not np.all(np.isfinite(weights))
        or abs(float(weights.sum()) - 1.0) > 2e-6
    ):
        raise V2OptimizationError("sample minimum-variance baseline failed audit")
    return dict(zip(names, weights, strict=True))


def _optimizer_weights(
    frame: Any,
    *,
    covariance_estimator: str,
    periods_per_year: float,
    objective: str = "min_risk",
    risk_aversion: float = 1.0,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    constraints_type: Any = V2Constraints
    weights, _ = V2LocalOptimizer().solve(
        frame,
        objective=objective,
        constraints=constraints_type(
            covariance_estimator=covariance_estimator,
            mean_estimator="auto" if objective == "hrp" else "empirical",
        ),
        periods_per_year=periods_per_year,
        target_reference_series=None,
        cap_reference_series=None,
        tracking_reference_series=None,
        reference_weights=None,
        risk_aversion=risk_aversion,
        risk_free_rate=risk_free_rate,
    )
    return weights


def baseline_allocations(
    model: V2Model,
    training_returns: Any,
    *,
    risk_aversion: float,
    risk_free_rate: float,
    periods_per_year: float = 52.0,
) -> dict[str, dict[str, float]]:
    """Fit the required baseline matrix on one training fold."""
    terminals = model.root.terminal_instruments()
    frame = training_returns.loc[:, terminals].dropna(how="any")
    if len(frame) < 3:
        raise V2OptimizationError(
            "scientific baselines require at least three observations"
        )
    equal = 1.0 / len(terminals)
    return {
        "EQUAL_WEIGHT": {name: equal for name in terminals},
        "DECLARED_BENCHMARK": dict(model.benchmark.weights),
        "SAMPLE_MIN_VARIANCE": _bounded_minimum_variance(frame),
        "SHRUNK_FIXED_MIN_VARIANCE": _optimizer_weights(
            frame,
            covariance_estimator="shrunk_fixed",
            periods_per_year=periods_per_year,
            risk_aversion=risk_aversion,
            risk_free_rate=risk_free_rate,
        ),
        "LEDOIT_WOLF_MIN_VARIANCE": _optimizer_weights(
            frame,
            covariance_estimator="ledoit_wolf",
            periods_per_year=periods_per_year,
            risk_aversion=risk_aversion,
            risk_free_rate=risk_free_rate,
        ),
        "HRP_WARD_PEARSON": _optimizer_weights(
            frame,
            covariance_estimator="shrunk_fixed",
            periods_per_year=periods_per_year,
            objective="hrp",
            risk_aversion=risk_aversion,
            risk_free_rate=risk_free_rate,
        ),
    }


def paired_block_bootstrap(
    differences: np.ndarray,
    *,
    samples: int,
    block_size: int,
    random_seed: int,
) -> tuple[float, float, float]:
    """Return a percentile CI and null-centred two-sided p-value for a paired mean."""
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("paired bootstrap requires at least two observations")
    if not np.all(np.isfinite(values)):
        raise ValueError("paired bootstrap differences must be finite")
    if samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    if block_size < 1:
        raise ValueError("bootstrap_block_size must be positive")

    rng = np.random.default_rng(random_seed)
    n_obs = len(values)
    blocks_needed = int(np.ceil(n_obs / block_size))
    bootstrap_means = np.empty(samples, dtype=float)
    null_means = np.empty(samples, dtype=float)
    offsets = np.arange(block_size)
    centered = values - float(np.mean(values))
    for sample in range(samples):
        starts = rng.integers(0, n_obs, size=blocks_needed)
        indexes = (
            ((starts[:, None] + offsets[None, :]) % n_obs)
            .reshape(-1)[:n_obs]
        )
        bootstrap_means[sample] = float(np.mean(values[indexes]))
        null_means[sample] = float(np.mean(centered[indexes]))
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    observed = abs(float(np.mean(values)))
    p_value = (1.0 + float(np.sum(np.abs(null_means) >= observed))) / (
        samples + 1.0
    )
    return float(low), float(high), float(p_value)


def _holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _fold_curve(
    folds: list[Any],
    estimation: Any,
    daily_returns: Any,
    allocation_factory: Callable[[Any], dict[str, dict[str, float]]],
    transaction_cost_bps: float,
) -> dict[str, Any]:
    import pandas as pd

    ledgers: dict[str, _StudyLedger] = {}
    points: dict[str, list[tuple[Any, float]]] = {}
    for fold in folds:
        train = estimation.loc[
            (estimation.index >= fold.training_start)
            & (estimation.index <= fold.training_end)
        ]
        holding = daily_returns.loc[
            (daily_returns.index >= fold.holding_start)
            & (daily_returns.index <= fold.holding_end)
        ]
        allocations = allocation_factory(train)
        for arm, target in allocations.items():
            ledger = ledgers.setdefault(arm, _StudyLedger(transaction_cost_bps))
            cost = ledger.rebalance(target)
            arm_points = points.setdefault(arm, [])
            first = True
            for day, row in holding.iterrows():
                value = ledger.step(row)
                if first:
                    value -= cost
                    first = False
                arm_points.append((day, value))
    return {
        arm: pd.Series(
            [value for _, value in values],
            index=pd.DatetimeIndex([day for day, _ in values]),
            dtype=float,
        )
        for arm, values in points.items()
    }


def run_scientific_study(
    model: V2Model,
    daily_returns: Any,
    *,
    mode: Mode,
    protocol: ScientificStudyProtocol,
) -> ScientificStudyResult:
    """Run V2 and required baselines on the same causal walk-forward folds.

    Beyond the naive/covariance-ablation baselines, this also runs two
    dedicated ablations that isolate effects the naive baselines cannot:
    a proxy-vs-synthetic representation ablation (the *same* model/estimator
    solved once ``forward`` and once ``forward_backward`` - everything except
    candidate representation held fixed) and a direct-bottom-up arm (the
    same final backward composition, computed with no Forward pass at all -
    the empirical companion to the unit-tested
    ``estimate_direct_bottom_up`` invariant). These are reported as their own
    curves/metrics, not folded into the baseline bootstrap comparisons: they
    are an estimator/representation ablation, not a strategy comparison.
    """
    v2_report = HierarchicalV2Backtester().run(
        model,
        daily_returns,
        mode=mode,
        train_size=protocol.train_size,
        estimation_frequency=protocol.estimation_frequency,
        rebalance_frequency=protocol.rebalance_frequency,
        transaction_cost_bps=protocol.transaction_cost_bps,
        include_partial_last_period=protocol.include_partial_last_period,
    )
    estimation = _resample_simple_returns(
        daily_returns, protocol.estimation_frequency
    )
    periods_per_year = _annualization_factor(protocol.estimation_frequency)
    root_risk_aversion = _effective_setting(
        model.root.constraints.risk_aversion,
        model.root.constraints.risk_aversion,
        1.0,
    )
    root_risk_free = _effective_setting(
        model.root.constraints.risk_free_rate,
        model.root.constraints.risk_free_rate,
        0.0,
    )

    baseline_curves = _fold_curve(
        v2_report.folds,
        estimation,
        daily_returns,
        lambda train: baseline_allocations(
            model,
            train,
            risk_aversion=root_risk_aversion,
            risk_free_rate=root_risk_free,
            periods_per_year=periods_per_year,
        ),
        protocol.transaction_cost_bps,
    )

    representation_label: dict[Mode, str] = {
        "forward": "V2_FORWARD",
        "forward_backward": "V2_FORWARD_BACKWARD",
    }
    ablation_curves: dict[str, Any] = {}
    if mode in representation_label:
        ablation_curves[representation_label[mode]] = v2_report.curves["FINAL"]
    for ablation_mode in ("forward", "forward_backward"):
        if ablation_mode == mode:
            continue
        ablation_report = HierarchicalV2Backtester().run(
            model,
            daily_returns,
            mode=ablation_mode,
            train_size=protocol.train_size,
            estimation_frequency=protocol.estimation_frequency,
            rebalance_frequency=protocol.rebalance_frequency,
            transaction_cost_bps=protocol.transaction_cost_bps,
            include_partial_last_period=protocol.include_partial_last_period,
        )
        ablation_curves[representation_label[ablation_mode]] = (
            ablation_report.curves["FINAL"]
        )

    direct_bottom_up_estimator = HierarchicalV2Estimator()
    direct_bottom_up_curves = _fold_curve(
        v2_report.folds,
        estimation,
        daily_returns,
        lambda train: {
            "V2_DIRECT_BOTTOM_UP": direct_bottom_up_estimator.estimate_direct_bottom_up(
                model, train, periods_per_year,
            ).terminal_weights
        },
        protocol.transaction_cost_bps,
    )

    curves = {
        "V2_FINAL": v2_report.curves["FINAL"],
        **baseline_curves,
        **ablation_curves,
        **direct_bottom_up_curves,
    }
    common_index = next(iter(curves.values())).index
    for curve in curves.values():
        common_index = common_index.intersection(curve.index)
    if common_index.empty:
        raise V2OptimizationError(
            "scientific study has no common OOS observations"
        )
    dropped_observations = {
        name: int(len(curve.index) - len(common_index))
        for name, curve in curves.items()
    }
    if any(dropped_observations.values()):
        if protocol.require_identical_oos_index:
            offenders = {
                name: count for name, count in dropped_observations.items() if count
            }
            raise V2OptimizationError(
                "scientific study arms do not share one identical OOS index "
                f"(observations that would be silently dropped: {offenders}); "
                "set protocol.require_identical_oos_index=False to opt into "
                "the previous intersect-and-continue behavior"
            )
        dropped_observations = {
            name: count for name, count in dropped_observations.items() if count
        }
    else:
        dropped_observations = {}
    curves = {name: curve.reindex(common_index) for name, curve in curves.items()}
    if any(curve.isna().any() for curve in curves.values()):
        raise V2OptimizationError(
            "scientific study curves are incomplete on the common OOS grid"
        )

    metrics_function: Any = HierarchicalV2Backtester._metrics
    metrics = {
        name: metrics_function(curve, root_risk_free)
        for name, curve in curves.items()
    }
    raw_comparisons: list[
        tuple[str, str, float, float, float, float]
    ] = []
    for baseline in BOOTSTRAPPED_BASELINES:
        differences = (
            curves["V2_FINAL"].to_numpy(dtype=float)
            - curves[baseline].to_numpy(dtype=float)
        )
        low, high, p_value = paired_block_bootstrap(
            differences,
            samples=protocol.bootstrap_samples,
            block_size=protocol.bootstrap_block_size,
            random_seed=protocol.random_seed,
        )
        raw_comparisons.append(
            (
                "V2_FINAL",
                baseline,
                float(np.mean(differences) * 252.0),
                low * 252.0,
                high * 252.0,
                p_value,
            )
        )
    adjusted = _holm_adjust([item[-1] for item in raw_comparisons])
    comparisons = [
        PairedComparison(
            candidate=item[0],
            baseline=item[1],
            annualized_mean_difference=item[2],
            confidence_interval_low=item[3],
            confidence_interval_high=item[4],
            p_value=item[5],
            holm_adjusted_p_value=adjusted[index],
        )
        for index, item in enumerate(raw_comparisons)
    ]
    return ScientificStudyResult(
        curves=curves,
        metrics=metrics,
        comparisons=comparisons,
        fold_count=len(v2_report.folds),
        common_oos_start=common_index.min(),
        common_oos_end=common_index.max(),
        protocol=protocol,
        dropped_observations=dropped_observations,
    )
