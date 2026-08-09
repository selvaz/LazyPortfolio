"""Phase A1 benchmark harness for the v3 optimizer performance roadmap.

Loads real saved trees and runs each through every applicable
(mode, workload) combination, recording solve counts and wall-clock time
into the shared run_history DB (kind="benchmark") -- a durable, queryable
baseline later phases (B onward) diff against, not a throwaway report.

Needs a real, populated Market Data Hub database (MARKET_DATA_DB) -- this
is a live measurement tool, not a unit test; see
tests/test_benchmark_v2.py for the deterministic, MDH-free regression
guard on solve *counting* itself.

Run: python scripts/benchmark_v2.py [tree name ...]
(default: the 3 real benchmark trees + the Black-Litterman views fixture)
"""

from __future__ import annotations

import sys
import time
import tracemalloc
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "project"))

import tree_studio  # noqa: E402  (reuses _v2_inputs/_config_hash/_data_fingerprint/_config_instruments)

from lazyportfolio.calendar import _annualization_factor, _resample_simple_returns  # noqa: E402
from lazyportfolio.hierarchical_v2 import (  # noqa: E402
    HierarchicalV2Backtester,
    HierarchicalV2Estimator,
    V2OptimizationError,
)
from lazyportfolio.v2 import run_history, store  # noqa: E402

#: Small / medium / large real trees + the Black-Litterman views fixture.
#: "Global Multi-Asset" (13 nodes) is a real saved tree, not synthetic.
DEFAULT_TREES = [
    "MS_7030_base_2_level",
    "ACWI_AGG_70_30",
    "Global Multi-Asset",
    "V3 Benchmark Views Fixture -posterior_all",
]
MODES = ["flat", "forward", "forward_backward"]
#: Walk-forward multiplies the point-estimate cost by the fold count (a
#: single point-estimate solve already took ~90s/local-solve against real
#: data in early testing) -- default to the cheap workload only; pass
#: --walk-forward for the expensive one.
DEFAULT_WORKLOADS = ("point_estimate",)


def _merge_pass_audits(primary: dict[str, Any], forward: dict[str, Any]) -> list[Any]:
    """Combine a pass's authoritative audits (backward, or a fold's own) with
    the Forward pass's, without double-counting a component neither pass
    actually re-solved.

    forward_backward's ``node_results`` (backward) and
    ``forward_node_results`` overlap: a leaf reuses its Forward audit rather
    than re-solving, so the identical audit (same ``component_id`` *and*
    ``solve_seconds`` -- an unchanged carried-forward value, not a
    coincidence) shows up in both dicts. An internal node that genuinely
    re-solves in Backward has a *different* ``solve_seconds`` in each dict --
    two distinct real solves, both must count. Comparing IDs alone (as an
    earlier version of this function did) can't tell these apart and
    undercounts genuine re-solves; comparing ``solve_seconds`` can.
    """
    by_id = {a.component_id: a for a in primary.values()}
    audits = list(by_id.values())
    for component_id, forward_audit in {a.component_id: a for a in forward.values()}.items():
        primary_audit = by_id.get(component_id)
        if primary_audit is None or primary_audit.solve_seconds != forward_audit.solve_seconds:
            audits.append(forward_audit)
    return audits


def local_solves(result: Any) -> tuple[int, float, int, list[str]]:
    """(solve_count, total_solve_seconds, total_slsqp_calls, problem_classes)
    across every audit a V2Estimate or V2BacktestReport carries -- see
    ``_merge_pass_audits`` for how forward_backward's overlapping Forward/
    Backward audit sets are combined without over- or under-counting.
    """
    if hasattr(result, "node_results"):
        audits = _merge_pass_audits(
            {name: nr.audit for name, nr in result.node_results.items()},
            {name: nr.audit for name, nr in result.forward_node_results.items()},
        )
    else:
        audits = []
        for fold in result.folds:
            audits.extend(_merge_pass_audits(fold.audits, fold.forward_audits))
    solve_seconds = sum(a.solve_seconds for a in audits)
    slsqp_calls = sum(a.restart_candidate_count for a in audits)
    problem_classes = sorted({a.problem_class for a in audits})
    return len(audits), solve_seconds, slsqp_calls, problem_classes


def _benchmark_point_estimate(
    model: Any, dataset: Any, mode: str, backtest: dict[str, Any]
) -> dict[str, Any]:
    estimation_frequency = str(backtest.get("estimation_frequency") or "W")
    train_size = int(backtest.get("train_size") or 104)
    estimation = _resample_simple_returns(dataset.returns, estimation_frequency)
    train = estimation.tail(train_size)
    if len(train) < train_size:
        raise V2OptimizationError("not enough observations for point-estimate benchmark")
    tracemalloc.start()
    started = time.perf_counter()
    estimate = HierarchicalV2Estimator().estimate(
        model, train, mode=mode, periods_per_year=_annualization_factor(estimation_frequency)
    )
    wall_clock = time.perf_counter() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    solves, solver_seconds, slsqp_calls, problem_classes = local_solves(estimate)
    return {
        "workload": "point_estimate",
        "wall_clock_seconds": wall_clock,
        "local_solve_count": solves,
        "solver_seconds": solver_seconds,
        "slsqp_call_count": slsqp_calls,
        "peak_memory_mb": peak_bytes / (1024 * 1024),
        "problem_classes": problem_classes,
        "terminal_weights": estimate.terminal_weights,
    }


def _benchmark_walk_forward(
    model: Any, dataset: Any, mode: str, backtest: dict[str, Any]
) -> dict[str, Any]:
    tracemalloc.start()
    started = time.perf_counter()
    report = HierarchicalV2Backtester().run(
        model,
        dataset.returns,
        mode=mode,
        train_size=int(backtest.get("train_size") or 104),
        estimation_frequency=str(backtest.get("estimation_frequency") or "W"),
        rebalance_frequency=str(backtest.get("rebalance_frequency") or "M"),
        transaction_cost_bps=float(backtest.get("transaction_cost_bps") or 0),
        capture_audit_series=False,
    )
    wall_clock = time.perf_counter() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    solves, solver_seconds, slsqp_calls, problem_classes = local_solves(report)
    return {
        "workload": "walk_forward_monthly",
        "wall_clock_seconds": wall_clock,
        "local_solve_count": solves,
        "solver_seconds": solver_seconds,
        "slsqp_call_count": slsqp_calls,
        "peak_memory_mb": peak_bytes / (1024 * 1024),
        "problem_classes": problem_classes,
        "fold_count": len(report.folds),
        "metrics": report.metrics,
    }


def benchmark_tree(
    name: str, *, workloads: tuple[str, ...] = DEFAULT_WORKLOADS
) -> list[dict[str, Any]]:
    config = store.read_model(name)
    model, dataset = tree_studio._v2_inputs(config)
    config_hash = tree_studio._config_hash(config)
    data_as_of, data_fingerprint = tree_studio._data_fingerprint(config)
    node_count = len(model.root.walk())
    instrument_count = len(tree_studio._config_instruments(model))
    backtest = config.get("backtest") or {}

    workload_fns = {
        "point_estimate": (_benchmark_point_estimate, "/api/v2/estimate"),
        "walk_forward": (_benchmark_walk_forward, "/api/v2/backtest"),
    }
    results = []
    for mode in MODES:
        for workload_name in workloads:
            run_workload, path = workload_fns[workload_name]
            try:
                measured = run_workload(model, dataset, mode, backtest)
            except V2OptimizationError as exc:
                print(f"  [skip] {name} mode={mode} {run_workload.__name__}: {exc}")
                continue
            measured["tree"] = name
            measured["mode"] = mode
            measured["node_count"] = node_count
            measured["instrument_count"] = instrument_count
            results.append(measured)

            cache_key = (
                f"benchmark:{name}:{mode}:{measured['workload']}:"
                f"{datetime.now(UTC).isoformat()}"
            )
            run_history.record_run(
                cache_key=cache_key,
                path=path,
                kind="benchmark",
                tree_id=store.sanitize_model_name(name),
                config_hash=config_hash,
                data_as_of=data_as_of,
                data_fingerprint=data_fingerprint,
                weights=measured.get("terminal_weights"),
                metrics={k: v for k, v in measured.items() if k != "terminal_weights"},
                payload=measured,
            )
            print(
                f"  {name} mode={mode:16s} {measured['workload']:20s} "
                f"solves={measured['local_solve_count']:4d} "
                f"solver_s={measured['solver_seconds']:.3f} "
                f"wall_s={measured['wall_clock_seconds']:.3f} "
                f"peak_mb={measured['peak_memory_mb']:.1f}"
            )
    return results


def main(argv: list[str] | None = None) -> int:
    args = argv or []
    walk_forward = "--walk-forward" in args
    names = [a for a in args if a != "--walk-forward"] or DEFAULT_TREES
    workloads = ("point_estimate", "walk_forward") if walk_forward else DEFAULT_WORKLOADS
    all_results: list[dict[str, Any]] = []
    for name in names:
        print(f"Benchmarking {name!r} (workloads={workloads})...")
        all_results.extend(benchmark_tree(name, workloads=workloads))
    print(f"\n{len(all_results)} benchmark runs recorded to run_history.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or None))
