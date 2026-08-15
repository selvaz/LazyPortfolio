"""Causal adaptive pruning for hierarchical V2 portfolios.

This module owns the reusable computation.  CLIs, schedulers and Tree Studio
may choose a tree and a policy, but they must all call this implementation.
"""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any

import pandas as pd

from lazyportfolio.backend import OptimizationDataset
from lazyportfolio.calendar import _annualization_factor
from lazyportfolio.models import BacktestSpec
from lazyportfolio.v2.backtest import HierarchicalV2Backtester, _V2Ledger
from lazyportfolio.v2.contracts import Mode, V2BacktestReport, V2Estimate, V2Fold
from lazyportfolio.v2.hierarchy import HierarchicalV2Estimator
from lazyportfolio.v2.model import V2Model
from lazyportfolio.v2.tree_pruning import PruningRule, prune_config
from lazyportfolio.walk_forward import prepare_walk_forward_inputs

DEFAULT_BURN_IN_YEARS = 2.0
_POLICY_KEYS = {
    "enabled",
    "burn_in_years",
    "evidence_window_years",
    "min_sharpe_improvement",
    "max_drawdown_per_vol_ratio",
    "workers",
    "max_folds",
    "expanding",
}


@dataclass(frozen=True)
class AdaptivePruningPolicy:
    """Validated policy shared by the backend API and scheduled jobs."""

    burn_in_years: float = DEFAULT_BURN_IN_YEARS
    evidence_window_years: float | None = None
    min_sharpe_improvement: float = 0.03
    max_drawdown_per_vol_ratio: float = 1.10
    workers: int = 1
    max_folds: int | None = None
    expanding: bool = False

    def __post_init__(self) -> None:
        if not isfinite(self.burn_in_years) or self.burn_in_years <= 0:
            raise ValueError("burn_in_years must be a positive finite number")
        if self.evidence_window_years is not None and (
            not isfinite(self.evidence_window_years) or self.evidence_window_years <= 0
        ):
            raise ValueError("evidence_window_years must be null or a positive finite number")
        if not isfinite(self.min_sharpe_improvement):
            raise ValueError("min_sharpe_improvement must be finite")
        if (
            not isfinite(self.max_drawdown_per_vol_ratio)
            or self.max_drawdown_per_vol_ratio <= 0
        ):
            raise ValueError("max_drawdown_per_vol_ratio must be a positive finite number")
        if isinstance(self.workers, bool) or not isinstance(self.workers, int):
            raise ValueError("workers must be an integer")
        if not 1 <= self.workers <= 32:
            raise ValueError("workers must be between 1 and 32")
        if self.max_folds is not None and (
            isinstance(self.max_folds, bool)
            or not isinstance(self.max_folds, int)
            or self.max_folds < 1
        ):
            raise ValueError("max_folds must be null or a positive integer")
        if not isinstance(self.expanding, bool):
            raise ValueError("expanding must be boolean")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> AdaptivePruningPolicy:
        """Build a closed-schema policy from an API/config mapping."""

        values = dict(raw or {})
        unknown = sorted(set(values) - _POLICY_KEYS)
        if unknown:
            raise ValueError(f"unknown adaptive pruning settings: {', '.join(unknown)}")
        values.pop("enabled", None)
        return cls(**values)

    def pruning_rule(self) -> PruningRule:
        return PruningRule(
            min_sharpe_improvement=self.min_sharpe_improvement,
            max_drawdown_per_vol_ratio=self.max_drawdown_per_vol_ratio,
            required_protocols=("accumulated",),
        )

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AdaptivePruningResult:
    """Backend result before transport/report formatting."""

    report: V2BacktestReport
    estimate: V2Estimate
    decisions: list[dict[str, Any]]
    policy: AdaptivePruningPolicy
    rule: PruningRule
    burn_in_cutoff: pd.Timestamp
    last_candidate: dict[str, Any]


def _history_before(curve: pd.Series, as_of: Any, window_start: Any | None) -> pd.Series:
    """Return strictly prior observations; the signal date itself is future to the decision."""

    selected = curve.loc[curve.index < pd.Timestamp(as_of)]
    if window_start is not None:
        selected = selected.loc[selected.index >= pd.Timestamp(window_start)]
    return selected


def _years_offset(years: float) -> pd.DateOffset | pd.Timedelta:
    """Keep whole years calendar-aware and make fractional years unambiguous."""

    return (
        pd.DateOffset(years=int(years))
        if float(years).is_integer()
        else pd.Timedelta(days=round(years * 365.2425))
    )


def accumulated_node_metrics(
    curves: Mapping[str, pd.Series],
    node_names: list[str],
    as_of: Any,
    window_start: Any | None = None,
) -> dict[str, dict[str, Any]]:
    """Build NODE/FATHER evidence using observations strictly before ``as_of``."""

    metrics: dict[str, dict[str, Any]] = {}
    for name in node_names:
        node_key, father_key = f"NODE:{name}", f"FATHER:{name}"
        if node_key not in curves or father_key not in curves:
            continue
        node_curve = _history_before(curves[node_key], as_of, window_start)
        father_curve = _history_before(curves[father_key], as_of, window_start)
        common = node_curve.index.intersection(father_curve.index)
        if common.empty:
            continue
        metrics[node_key] = HierarchicalV2Backtester._metrics(node_curve.loc[common])
        metrics[father_key] = HierarchicalV2Backtester._metrics(father_curve.loc[common])
    return metrics


def summarize_pruning_decisions(
    decisions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return transport-ready fold summaries without moving policy into the UI."""

    return [
        {
            "signal": fold["signal"],
            "burn_in": bool(fold["burn_in"]),
            "candidate_nodes": int(fold["candidate_nodes"]),
            "pruned_nodes": sum(
                1 for node in fold.get("nodes", []) if node.get("decision") == "prune"
            ),
            "retained_nodes": sum(
                1 for node in fold.get("nodes", []) if node.get("decision") == "retain"
            ),
            "target_l1_distance": float(fold["target_l1_distance"]),
        }
        for fold in decisions
    ]


def _prepare_fold(
    config: dict[str, Any],
    train: pd.DataFrame,
    mode: Mode,
    periods_per_year: float,
    metrics: dict[str, dict[str, Any]],
    rule: PruningRule,
) -> tuple[dict[str, Any], list[dict[str, Any]], V2Estimate, dict[str, float]]:
    candidate, decisions = prune_config(config, {"accumulated": metrics}, rule)
    candidate_model = V2Model.from_config(candidate)
    estimate = HierarchicalV2Estimator().estimate(
        candidate_model,
        train,
        mode=mode,
        periods_per_year=periods_per_year,
    )
    forward_target = (
        dict(estimate.forward_node_results[candidate_model.root.name].terminal_weights)
        if estimate.forward_node_results
        else dict(estimate.terminal_weights)
    )
    return candidate, decisions, estimate, forward_target


def run_adaptive_pruning(
    config: dict[str, Any],
    *,
    model: V2Model,
    dataset: OptimizationDataset,
    reference_report: V2BacktestReport,
    mode: Mode,
    policy: AdaptivePruningPolicy,
) -> AdaptivePruningResult:
    """Run the universal causal adaptive-pruning backtest.

    ``reference_report`` must be the ordinary unpruned walk-forward report for
    the same config.  Reusing it makes the decision both auditable and fast:
    only each pruned candidate needs a new solve.
    """

    reference_folds = reference_report.folds
    if not reference_folds:
        raise ValueError("adaptive pruning requires at least one reference fold")

    backtest = config["backtest"]
    train_size = int(backtest.get("train_size") or 104)
    estimation_frequency = str(backtest.get("estimation_frequency") or "W")
    rebalance_frequency = str(backtest.get("rebalance_frequency") or "M")
    instruments = list(
        dict.fromkeys(
            [
                *model.root.terminal_instruments(),
                *(node.proxy for node in model.root.walk() if node.proxy),
                *model.benchmark.weights,
            ]
        )
    )
    node_names = [node.name for node in model.root.walk() if node.proxy is not None]
    periods_per_year = _annualization_factor(estimation_frequency)
    rule = policy.pruning_rule()
    burn_in_cutoff = reference_folds[0].holding_start + _years_offset(policy.burn_in_years)

    valuation, estimation, _ = prepare_walk_forward_inputs(
        dataset.returns,
        instruments,
        BacktestSpec(
            id="adaptive-pruning",
            train_size=train_size,
            rebalance_frequency=rebalance_frequency,
        ),
        estimation_frequency,
    )
    report_folds = (
        reference_folds[-(policy.max_folds + 1) :]
        if policy.max_folds is not None
        else reference_folds
    )
    specs = [
        (
            fold,
            estimation.loc[fold.training_start : fold.training_end],
            valuation.loc[fold.holding_start : fold.holding_end],
        )
        for fold in report_folds
    ]

    def prepare(spec: tuple[Any, pd.DataFrame, pd.DataFrame]) -> tuple[
        dict[str, Any], list[dict[str, Any]], V2Estimate, dict[str, float]
    ]:
        fold, train, _holding = spec
        window_start = (
            fold.signal - _years_offset(policy.evidence_window_years)
            if policy.evidence_window_years is not None
            else None
        )
        metrics = accumulated_node_metrics(
            reference_report.curves,
            node_names,
            fold.signal,
            window_start,
        )
        return _prepare_fold(config, train, mode, periods_per_year, metrics, rule)

    post_burn_in = [
        (index, spec)
        for index, spec in enumerate(specs)
        if spec[0].signal >= burn_in_cutoff
    ]
    if not post_burn_in:
        raise ValueError("adaptive pruning has no folds after the configured burn-in")
    with ThreadPoolExecutor(max_workers=policy.workers) as pool:
        prepared = dict(
            zip(
                (index for index, _ in post_burn_in),
                pool.map(prepare, (spec for _, spec in post_burn_in)),
                strict=True,
            )
        )

    arms = ("FINAL", "FORWARD_FINAL")
    ledgers = {
        arm: _V2Ledger(float(backtest.get("transaction_cost_bps") or 0)) for arm in arms
    }
    points: dict[str, list[tuple[Any, float]]] = {arm: [] for arm in arms}
    folds: list[V2Fold] = []
    decisions: list[dict[str, Any]] = []
    last_estimate: V2Estimate | None = None
    last_candidate: dict[str, Any] | None = None

    for index, (reference_fold, train, holding) in enumerate(specs):
        if index in prepared:
            candidate, fold_decisions, estimate, forward_target = prepared[index]
            adaptive_target = dict(estimate.terminal_weights)
            last_estimate = estimate
            last_candidate = candidate
            audits = {name: result.audit for name, result in estimate.node_results.items()}
            candidate_nodes = len(candidate["nodes"])
        else:
            fold_decisions = []
            adaptive_target = dict(reference_fold.targets["FINAL"])
            forward_target = dict(
                reference_fold.targets.get("FORWARD_FINAL", adaptive_target)
            )
            audits = dict(reference_fold.audits)
            candidate_nodes = len(config["nodes"])

        targets = {"FINAL": adaptive_target, "FORWARD_FINAL": forward_target}
        for arm, target in targets.items():
            cost = ledgers[arm].rebalance(target)
            for first, (day, row) in enumerate(holding.iterrows()):
                points[arm].append((day, ledgers[arm].step(row) - (cost if first == 0 else 0)))
        folds.append(
            V2Fold(
                reference_fold.signal,
                train.index.min(),
                train.index.max(),
                holding.index.min(),
                holding.index.max(),
                {
                    "B0": dict(model.benchmark.weights),
                    "STATIC_FINAL": dict(reference_fold.targets["FINAL"]),
                    **targets,
                },
                audits,
            )
        )
        static = reference_fold.targets["FINAL"]
        names = set(static) | set(adaptive_target)
        decisions.append(
            {
                "signal": str(reference_fold.signal.date()),
                "burn_in": index not in prepared,
                "nodes": fold_decisions,
                "candidate_nodes": candidate_nodes,
                "static_weights": static,
                "adaptive_weights": adaptive_target,
                "target_l1_distance": sum(
                    abs(static.get(name, 0) - adaptive_target.get(name, 0)) for name in names
                ),
            }
        )

    if last_estimate is None or last_candidate is None:
        raise ValueError("adaptive pruning did not produce a post-burn-in estimate")
    report_start, report_end = specs[0][0].holding_start, specs[-1][0].holding_end
    curves = {
        "B0": reference_report.curves["B0"].loc[report_start:report_end],
        "STATIC_FINAL": reference_report.curves["FINAL"].loc[report_start:report_end],
        **{
            arm: pd.Series(
                [point[1] for point in points[arm]],
                index=pd.DatetimeIndex([point[0] for point in points[arm]]),
            )
            for arm in arms
        },
    }
    report = V2BacktestReport(
        mode=mode,
        folds=folds,
        curves=curves,
        metrics={arm: HierarchicalV2Backtester._metrics(curve) for arm, curve in curves.items()},
        transaction_cost_paid={
            "B0": 0.0,
            "STATIC_FINAL": 0.0,
            **{arm: ledgers[arm].total_cost for arm in arms},
        },
    )
    return AdaptivePruningResult(
        report=report,
        estimate=last_estimate,
        decisions=decisions,
        policy=policy,
        rule=rule,
        burn_in_cutoff=burn_in_cutoff,
        last_candidate=last_candidate,
    )


__all__ = [
    "AdaptivePruningPolicy",
    "AdaptivePruningResult",
    "DEFAULT_BURN_IN_YEARS",
    "accumulated_node_metrics",
    "run_adaptive_pruning",
    "summarize_pruning_decisions",
]
