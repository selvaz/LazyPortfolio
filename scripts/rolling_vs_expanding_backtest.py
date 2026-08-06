"""Overnight comparison: rolling vs expanding training-window backtests on
a real tree, full history, results + an HTML report saved into the
registered LazyPortfolio store (lazyportfolio.v2.run_history -- the same
DB as scripts/benchmark_v2.py, registered in LazyTools as
lazyportfolio_store / LAZYPORTFOLIO_TREE_DB).

Runs both methodologies with weekly estimation / monthly rebalance / zero
transaction cost / zero risk-free rate (this tree's own existing config
and defaults), max_workers=4 (Phase F1 parallelization -- kept at 4, not
higher, on this 6-core machine: 5 workers reproduced the OpenBLAS
oversubscription crash documented in docs/optimizer-v3-rollout.md even
with each worker pinned to 1 BLAS thread, most likely from per-process
memory/handle pressure beyond just BLAS threading) on the tree's complete
available history -- meant to run unattended (e.g. via Windows Task
Scheduler overnight) since a full multi-year run can take 15-40+ minutes
per methodology.

Run: python scripts/rolling_vs_expanding_backtest.py [tree name]
(default: Global Multi-Asset)
"""

from __future__ import annotations

import html
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "project"))

import tree_studio  # noqa: E402  (reuses _v2_inputs/_config_hash/_data_fingerprint)

from lazyportfolio.hierarchical_v2 import HierarchicalV2Backtester  # noqa: E402
from lazyportfolio.v2 import run_history, store  # noqa: E402

DEFAULT_TREE = "Global Multi-Asset"
MAX_WORKERS = 4


def _log(message: str) -> None:
    print(f"[{datetime.now(UTC).isoformat()}] {message}", flush=True)


def _metrics_table_html(metrics: dict[str, dict[str, Any]]) -> str:
    columns = [
        "cagr", "annualized_volatility", "annualized_sharpe", "annualized_sortino",
        "max_drawdown", "n_obs",
    ]
    rows = []
    for arm in sorted(metrics):
        cells = "".join(
            f"<td>{metrics[arm].get(col, ''):.4f}</td>"
            if isinstance(metrics[arm].get(col), float)
            else f"<td>{metrics[arm].get(col, '')}</td>"
            for col in columns
        )
        rows.append(f"<tr><td>{html.escape(arm)}</td>{cells}</tr>")
    header = "".join(f"<th>{col}</th>" for col in columns)
    return (
        f"<table><tr><th>arm</th>{header}</tr>{''.join(rows)}</table>"
    )


def _build_report_html(
    *,
    tree_name: str,
    window_mode: str,
    wall_seconds: float,
    fold_count: int,
    solve_count: int,
    train_size: int,
    date_range: tuple[str, str],
    metrics: dict[str, dict[str, Any]],
    terminal_weights: dict[str, float],
) -> str:
    weights_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{weight:.4f}</td></tr>"
        for name, weight in sorted(terminal_weights.items(), key=lambda kv: -kv[1])
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{html.escape(tree_name)} -- {html.escape(window_mode)} backtest</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 800px;
          margin: 2rem auto; padding: 0 1rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  th, td {{ border: 1px solid #ccc; padding: .4rem .6rem; text-align: left; }}
  th {{ background: #f2f2f2; }}
  .meta {{ color: #555; font-size: .92rem; }}
</style></head><body>
<h1>{html.escape(tree_name)} &mdash; {html.escape(window_mode)} window</h1>
<p class="meta">
  Date range: {date_range[0]} to {date_range[1]}<br>
  train_size: {train_size} &middot; folds: {fold_count} &middot; local solves: {solve_count}<br>
  wall-clock: {wall_seconds:.1f}s ({wall_seconds / 60:.1f} min)
  &middot; max_workers={MAX_WORKERS}<br>
  weekly estimation, monthly rebalance, 0 transaction cost, 0 risk-free rate
</p>
<h2>Metrics by arm</h2>
{_metrics_table_html(metrics)}
<h2>Terminal weights (FINAL)</h2>
<table><tr><th>instrument</th><th>weight</th></tr>{weights_rows}</table>
</body></html>
"""


def run_variant(name: str, *, expanding: bool) -> dict[str, Any]:
    label = "expanding" if expanding else "rolling"
    config = store.read_model(name)
    model, dataset = tree_studio._v2_inputs(config)
    config_hash = tree_studio._config_hash(config)
    data_as_of, data_fingerprint = tree_studio._data_fingerprint(config)
    backtest = config.get("backtest", {}) or {}
    train_size = int(backtest.get("train_size") or 104)
    estimation_frequency = str(backtest.get("estimation_frequency") or "W")
    mode = "forward" if backtest.get("forward_enabled", True) else "flat"

    _log(
        f"{name!r} [{label}]: starting -- range {dataset.returns.index.min().date()} "
        f"to {dataset.returns.index.max().date()}, train_size={train_size}, mode={mode}"
    )
    started = time.perf_counter()
    report = HierarchicalV2Backtester().run(
        model,
        dataset.returns,
        mode=mode,
        train_size=train_size,
        estimation_frequency=estimation_frequency,
        rebalance_frequency="M",
        transaction_cost_bps=0.0,
        capture_audit_series=False,
        max_workers=MAX_WORKERS,
        expanding=expanding,
    )
    wall = time.perf_counter() - started
    solve_count = sum(
        len(fold.audits) + len(fold.forward_audits) for fold in report.folds
    )
    _log(
        f"{name!r} [{label}]: done -- wall={wall:.1f}s folds={len(report.folds)} "
        f"solves~={solve_count}"
    )

    terminal_weights = report.folds[-1].targets.get("FINAL", {}) if report.folds else {}
    payload = {
        "tree": name,
        "window_mode": label,
        "wall_clock_seconds": wall,
        "fold_count": len(report.folds),
        "solve_count_approx": solve_count,
        "train_size": train_size,
        "metrics": report.metrics,
        "transaction_cost_paid": report.transaction_cost_paid,
        "terminal_weights": terminal_weights,
    }
    cache_key = f"rolling_vs_expanding:{name}:{label}:{datetime.now(UTC).isoformat()}"
    run_id = run_history.record_run(
        cache_key=cache_key,
        path="/scripts/rolling_vs_expanding_backtest",
        kind="rolling_vs_expanding",
        tree_id=store.sanitize_model_name(name),
        config_hash=config_hash,
        data_as_of=data_as_of,
        data_fingerprint=data_fingerprint,
        weights=terminal_weights,
        metrics={k: v for k, v in payload.items() if k not in ("metrics", "terminal_weights")},
        payload=payload,
    )
    report_html = _build_report_html(
        tree_name=name,
        window_mode=label,
        wall_seconds=wall,
        fold_count=len(report.folds),
        solve_count=solve_count,
        train_size=train_size,
        date_range=(
            str(dataset.returns.index.min().date()),
            str(dataset.returns.index.max().date()),
        ),
        metrics=report.metrics,
        terminal_weights=terminal_weights,
    )
    run_history.attach_artifact(
        run_id,
        kind="report",
        content_type="text/html",
        filename=f"{store.sanitize_model_name(name)}_{label}_report.html",
        blob=report_html.encode("utf-8"),
    )
    _log(f"{name!r} [{label}]: recorded run_id={run_id}, HTML report attached")
    return payload


def main(argv: list[str] | None = None) -> int:
    args = argv or []
    name = args[0] if args else DEFAULT_TREE
    _log(f"=== rolling vs expanding backtest job starting for {name!r} ===")
    rolling = run_variant(name, expanding=False)
    expanding = run_variant(name, expanding=True)
    _log(
        "=== job complete === "
        f"rolling: wall={rolling['wall_clock_seconds']:.1f}s folds={rolling['fold_count']} | "
        f"expanding: wall={expanding['wall_clock_seconds']:.1f}s folds={expanding['fold_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or None))
