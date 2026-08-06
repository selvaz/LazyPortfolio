"""Walk-forward ledger and reporting for hierarchical optimizer V2."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from math import sqrt
from typing import Any

import numpy as np

from lazyportfolio.v2.contracts import (
    Mode,
    V2BacktestReport,
    V2Estimate,
    V2Fold,
    V2OptimizationError,
    effective_setting,
)
from lazyportfolio.v2.hierarchy import HierarchicalV2Estimator
from lazyportfolio.v2.model import V2Model
from lazyportfolio.v2.moments import (
    CASH_BORROW,
    CASH_LEND,
    financing_instrument,
)


def _limit_worker_blas_threads() -> None:
    """``ProcessPoolExecutor`` initializer: cap each worker to a single BLAS
    thread.

    Without this, every worker process's numpy/scipy calls spin up their
    *own* OpenBLAS thread pool sized to the machine's full core count -- N
    worker processes x M BLAS threads each oversubscribes the machine by
    N*M threads competing for N*M cores' worth of work. Verified live: 5
    workers on a 6-core machine crashed with "OpenBLAS error: Memory
    allocation still failed after 10 retries, giving up" (each worker's
    BLAS pool alone tried to claim most of the machine). Setting the
    env vars here (before any worker imports numpy -- this initializer
    runs first) plus threadpool_limits at call time (belt-and-suspenders
    for whichever BLAS build honors runtime limits vs import-time-only env
    vars) keeps each worker single-threaded, so the process-level
    parallelism from Phase F1 is the only parallelism in play.
    """
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"


def _estimate_fold(
    model: V2Model, train: Any, mode: Mode, periods_per_year: float
) -> V2Estimate:
    """Module-level, picklable worker for parallel fold estimation (v3
    performance roadmap Phase F1). Must stay top-level and reconstruct
    ``HierarchicalV2Estimator()`` itself, not a bound method or closure --
    ``ProcessPoolExecutor`` needs an importable callable, and each worker
    process (Windows uses the "spawn" start method, no fork) starts fresh
    with no inherited parent state to reuse anyway.
    """
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        return HierarchicalV2Estimator().estimate(
            model, train, mode=mode, periods_per_year=periods_per_year
        )


class _V2Ledger:
    """Self-financing daily ledger; target weights change only at rebalance."""

    def __init__(self, transaction_cost_bps: float) -> None:
        self.cost_rate = transaction_cost_bps / 10_000.0
        self.weights: dict[str, float] = {}
        self.total_cost = 0.0

    def rebalance(self, target: dict[str, float]) -> float:
        names = set(self.weights) | set(target)
        turnover = sum(
            abs(target.get(name, 0.0) - self.weights.get(name, 0.0))
            for name in names
        )
        cost = turnover * self.cost_rate
        self.weights = dict(target)
        self.total_cost += cost
        return cost

    def step(self, day_returns: Any) -> float:
        portfolio_return = sum(
            weight * float(day_returns[instrument])
            for instrument, weight in self.weights.items()
        )
        denominator = 1.0 + portfolio_return
        if denominator != 0.0:
            self.weights = {
                instrument: (
                    weight * (1.0 + float(day_returns[instrument])) / denominator
                )
                for instrument, weight in self.weights.items()
            }
        return portfolio_return


class HierarchicalV2Backtester:
    """One causal walk-forward implementation for every V2 methodology."""

    def __init__(self, estimator: HierarchicalV2Estimator | None = None) -> None:
        self.estimator = estimator or HierarchicalV2Estimator()

    def run(
        self,
        model: V2Model,
        daily_returns: Any,
        *,
        mode: Mode,
        train_size: int,
        estimation_frequency: str = "W",
        rebalance_frequency: str = "M",
        transaction_cost_bps: float = 0.0,
        include_partial_last_period: bool = False,
        capture_audit_series: bool = False,
        max_workers: int = 1,
        expanding: bool = False,
    ) -> V2BacktestReport:
        import pandas as pd

        from lazyportfolio.calendar import _annualization_factor
        from lazyportfolio.models import BacktestSpec
        from lazyportfolio.walk_forward import prepare_walk_forward_inputs

        effective_returns = self._with_financing_returns(model, daily_returns, 252.0)
        financing_columns = [
            name for name in effective_returns.columns if name not in daily_returns.columns
        ]
        instruments = list(
            dict.fromkeys(
                [
                    *model.root.terminal_instruments(),
                    *(node.proxy for node in model.root.walk() if node.proxy),
                    *model.benchmark.weights,
                    *financing_columns,
                ]
            )
        )
        protocol = BacktestSpec(
            id=f"hierarchical-v2-{mode}",
            train_size=train_size,
            rebalance_frequency=rebalance_frequency,
            include_partial_last_period=include_partial_last_period,
        )
        valuation, estimation, schedule = prepare_walk_forward_inputs(
            effective_returns,
            instruments,
            protocol,
            estimation_frequency,
        )
        periods_per_year = _annualization_factor(estimation_frequency)

        # Phase 1 (sequential, cheap): resolve every valid fold's train/
        # holding slices up front -- no solving happens here. `expanding`
        # controls the training window's shape: rolling (default) keeps it
        # pinned to exactly train_size observations (`.tail(train_size)`),
        # sliding forward each fold; expanding uses every observation up to
        # the signal date, so the window only grows -- train_size is then
        # just the minimum size before the first fold is emitted, not a cap.
        fold_specs: list[tuple[Any, Any, Any]] = []
        for index, signal in enumerate(schedule):
            next_signal = schedule[index + 1] if index + 1 < len(schedule) else None
            if next_signal is None and not include_partial_last_period:
                continue
            available = estimation.loc[estimation.index <= signal]
            train = available if expanding else available.tail(train_size)
            if len(train) < train_size:
                continue
            holding_mask = valuation.index > signal
            if next_signal is not None:
                holding_mask &= valuation.index <= next_signal
            holding = valuation.loc[holding_mask]
            if holding.empty:
                continue
            fold_specs.append((signal, train, holding))

        # Phase 2 (v3 performance roadmap Phase F1; embarrassingly parallel
        # when max_workers > 1): each fold's estimate() call only depends on
        # that fold's own training window, not on any other fold's result --
        # the *ledger* is what carries state across folds (Phase 3), so it's
        # the only part that must stay sequential. max_workers=1 (default)
        # keeps today's exact sequential behavior and error semantics; opt
        # into >1 explicitly once this is validated on real backtests, since
        # ProcessPoolExecutor's per-worker interpreter/import cost (observed
        # in this environment to spike well past a second under some
        # conditions) can outweigh the savings for small fold counts.
        estimates: list[V2Estimate] = []
        if max_workers > 1 and len(fold_specs) > 1:
            worker_count = min(max_workers, len(fold_specs), os.cpu_count() or 1)
            with ProcessPoolExecutor(
                max_workers=worker_count, initializer=_limit_worker_blas_threads
            ) as pool:
                futures = [
                    pool.submit(_estimate_fold, model, train, mode, periods_per_year)
                    for _, train, _ in fold_specs
                ]
                for (signal, _, _), future in zip(fold_specs, futures, strict=True):
                    try:
                        estimates.append(future.result())
                    except Exception as exc:
                        raise V2OptimizationError(
                            f"fold {signal.date()}: {type(exc).__name__}: {exc}"
                        ) from exc
        else:
            # Sequential path keeps using self.estimator (not the
            # module-level _estimate_fold helper), preserving a custom
            # injected estimator/optimiser -- the parallel path above can't
            # honor that (an arbitrary custom estimator isn't guaranteed
            # picklable across the process boundary), so it always
            # constructs the default HierarchicalV2Estimator() itself.
            for signal, train, _ in fold_specs:
                try:
                    estimates.append(
                        self.estimator.estimate(
                            model,
                            train,
                            mode=mode,
                            periods_per_year=periods_per_year,
                        )
                    )
                except Exception as exc:
                    raise V2OptimizationError(
                        f"fold {signal.date()}: {type(exc).__name__}: {exc}"
                    ) from exc

        # Phase 3 (sequential, required): the ledger's weights carry
        # forward from one fold's holding period into the next rebalance,
        # so this walk cannot parallelize regardless of worker count.
        ledgers: dict[str, _V2Ledger] = {}
        points: dict[str, list[tuple[Any, float]]] = {}
        folds: list[V2Fold] = []
        for (signal, train, holding), estimate in zip(
            fold_specs, estimates, strict=True
        ):
            targets = self._targets(model, estimate)
            for arm, target in targets.items():
                ledger = ledgers.setdefault(arm, _V2Ledger(transaction_cost_bps))
                arm_points = points.setdefault(arm, [])
                rebalance_cost = ledger.rebalance(target)
                first = True
                for day, row in holding.iterrows():
                    value = ledger.step(row)
                    if first:
                        value -= rebalance_cost
                        first = False
                    arm_points.append((day, value))
            folds.append(
                V2Fold(
                    signal=signal,
                    training_start=train.index.min(),
                    training_end=train.index.max(),
                    holding_start=holding.index.min(),
                    holding_end=holding.index.max(),
                    targets={
                        **{arm: dict(weights) for arm, weights in targets.items()},
                        **{
                            f"LOCAL:{name}": dict(result.local_weights)
                            for name, result in estimate.node_results.items()
                        },
                        **{
                            f"FORWARD_LOCAL:{name}": dict(result.local_weights)
                            for name, result in estimate.forward_node_results.items()
                        },
                    },
                    audits={
                        name: result.audit
                        for name, result in estimate.node_results.items()
                    },
                    forward_audits={
                        name: result.audit
                        for name, result in estimate.forward_node_results.items()
                    },
                    candidate_series={
                        **{
                            f"RESULT:{name}": list(result.local_weights)
                            for name, result in estimate.node_results.items()
                        },
                        **{
                            f"FORWARD:{name}": list(result.local_weights)
                            for name, result in estimate.forward_node_results.items()
                        },
                    },
                    estimation_series=(
                        self._audit_series(model, train, estimate)
                        if capture_audit_series
                        else {}
                    ),
                )
            )

        if not folds:
            raise V2OptimizationError("V2 backtest produced no complete folds")
        curves = {
            arm: pd.Series(
                [value for _, value in values],
                index=pd.DatetimeIndex([day for day, _ in values]),
                dtype=float,
            )
            for arm, values in points.items()
        }
        lengths = {len(curve) for curve in curves.values()}
        indexes = {tuple(curve.index) for curve in curves.values()}
        if len(lengths) != 1 or len(indexes) != 1:
            raise V2OptimizationError("V2 arms do not share one common OOS grid")

        root_rf = effective_setting(
            model.root.constraints.risk_free_rate,
            model.root.constraints.risk_free_rate,
            0.0,
        )
        nodes_by_name = {node.name: node for node in model.root.walk()}

        def rate_for_arm(arm: str) -> float:
            for prefix in ("NODE:", "FORWARD_NODE:", "FATHER:"):
                if arm.startswith(prefix):
                    node = nodes_by_name.get(arm.removeprefix(prefix))
                    if node is not None:
                        return effective_setting(
                            node.constraints.risk_free_rate,
                            model.root.constraints.risk_free_rate,
                            0.0,
                        )
            return root_rf

        return V2BacktestReport(
            mode=mode,
            folds=folds,
            curves=curves,
            metrics={
                arm: self._metrics(curve, rate_for_arm(arm))
                for arm, curve in curves.items()
            },
            transaction_cost_paid={
                arm: ledger.total_cost for arm, ledger in ledgers.items()
            },
        )

    @staticmethod
    def _with_financing_returns(
        model: V2Model,
        returns: Any,
        periods_per_year: float,
    ) -> Any:
        effective = returns.copy()
        root_rate = model.root.constraints.risk_free_rate
        for node in model.root.walk():
            constraints = node.constraints
            if not constraints.cash_enabled and constraints.max_leverage <= 1.0:
                continue
            risk_free = effective_setting(
                constraints.risk_free_rate,
                root_rate,
                0.0,
            )
            is_root = node is model.root
            effective[
                financing_instrument(CASH_LEND, node.id, is_root=is_root)
            ] = risk_free / periods_per_year
            effective[
                financing_instrument(CASH_BORROW, node.id, is_root=is_root)
            ] = (
                risk_free + constraints.borrow_spread_bps / 10_000.0
            ) / periods_per_year
        return effective

    @staticmethod
    def _targets(
        model: V2Model,
        estimate: V2Estimate,
    ) -> dict[str, dict[str, float]]:
        targets = {
            "B0": dict(model.benchmark.weights),
            "FINAL": dict(estimate.terminal_weights),
        }
        if estimate.synthetic_benchmark_weights:
            targets["B0_SYNTH"] = dict(estimate.synthetic_benchmark_weights)
        for name, result in estimate.node_results.items():
            targets[f"NODE:{name}"] = dict(result.terminal_weights)
        if estimate.forward_node_results:
            targets["FORWARD_FINAL"] = dict(
                estimate.forward_node_results[model.root.name].terminal_weights
            )
            for name, result in estimate.forward_node_results.items():
                targets[f"FORWARD_NODE:{name}"] = dict(result.terminal_weights)
        for node in model.root.walk():
            if node.proxy is not None:
                targets[f"FATHER:{node.name}"] = {node.proxy: 1.0}
        return targets

    @staticmethod
    def _audit_series(
        model: V2Model,
        train: Any,
        estimate: V2Estimate,
    ) -> dict[str, Any]:
        series = {f"RAW:{name}": train[name] for name in train.columns}
        b0_raw = (
            train.loc[:, list(model.benchmark.weights)]
            .mul(model.benchmark.weights, axis="columns")
            .sum(axis="columns")
        )
        series["REFERENCE:B0_RAW"] = b0_raw
        for node in model.root.walk():
            if node.proxy is not None:
                series[f"REFERENCE:FATHER:{node.name}"] = train[node.proxy]
        for name, result in estimate.forward_node_results.items():
            series[f"FORWARD_OUTPUT:{name}"] = result.synthetic_returns
        for name, result in estimate.node_results.items():
            series[f"RESULT_OUTPUT:{name}"] = result.synthetic_returns
        for node in model.root.walk():
            for child in node.children:
                series[f"BACKWARD_INPUT:{child.proxy}_SYNTH"] = (
                    estimate.node_results[child.name].synthetic_returns
                )
        if estimate.synthetic_benchmark_weights:
            series["DIAGNOSTIC:B0_SYNTH"] = (
                train.loc[:, list(estimate.synthetic_benchmark_weights)]
                .mul(estimate.synthetic_benchmark_weights, axis="columns")
                .sum(axis="columns")
            )
        return series

    @staticmethod
    def _metrics(
        returns: Any,
        risk_free_rate: float = 0.0,
    ) -> dict[str, float | int]:
        values = returns.to_numpy(dtype=float)
        observations = len(values)
        wealth = np.cumprod(1.0 + values)
        years = observations / 252.0
        cagr = float(wealth[-1] ** (1.0 / years) - 1.0) if years > 0 else 0.0
        volatility = (
            float(np.std(values, ddof=1) * sqrt(252.0))
            if observations > 1
            else 0.0
        )
        periodic_rf = risk_free_rate / 252.0
        excess = values - periodic_rf
        excess_std = float(np.std(excess, ddof=1)) if observations > 1 else 0.0
        sharpe = (
            float(np.mean(excess) / excess_std * sqrt(252.0))
            if excess_std > 0.0
            else 0.0
        )
        downside = np.minimum(excess, 0.0)
        downside_deviation = float(
            np.sqrt(np.mean(downside**2)) * sqrt(252.0)
        )
        annualized_excess = (
            float(np.mean(excess) * 252.0) if observations else 0.0
        )
        sortino = (
            annualized_excess / downside_deviation
            if downside_deviation > 0.0
            else 0.0
        )
        peaks = np.maximum.accumulate(wealth)
        max_drawdown = float(np.min(wealth / peaks - 1.0))
        return {
            "cagr": cagr,
            "annualized_excess_return": annualized_excess,
            "annualized_volatility": volatility,
            "annualized_sharpe": sharpe,
            "annualized_sortino": sortino,
            "risk_free_rate": float(risk_free_rate),
            "max_drawdown": max_drawdown,
            "n_obs": observations,
        }


__all__ = ["HierarchicalV2Backtester", "_V2Ledger"]
