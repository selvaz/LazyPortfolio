"""Overnight comparison: rolling vs expanding training-window backtests on
a real tree, full history, results + an HTML report saved into the
registered LazyPortfolio store (lazyportfolio.v2.run_history -- the same
DB as scripts/benchmark_v2.py, registered in LazyTools as
lazyportfolio_store / LAZYPORTFOLIO_TREE_DB).

Runs both methodologies with weekly estimation / monthly rebalance / zero
transaction cost / zero risk-free rate (this tree's own existing config
and defaults) with a stated worker count (Phase F1 parallelization -- on a
6-core machine 5 workers reproduced the OpenBLAS
oversubscription crash documented in docs/optimizer-v3-rollout.md even
with each worker pinned to 1 BLAS thread, most likely from per-process
memory/handle pressure beyond just BLAS threading) on the tree's complete
available history -- meant to run unattended (e.g. via Windows Task
Scheduler overnight) since a full multi-year run can take 15-40+ minutes
per methodology.

Each variant's real Tree Studio client report is also sent as a Telegram
document (best-effort -- a delivery failure is logged, never fatal, since
the results are already durably saved to run_history by that point) via
LazyTools' TelegramClient, using TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID from
the environment (User-level env vars on this machine).

Nothing about which portfolio is watched lives here. The trees, which of
them also get the pruning evaluation, how many workers to use and whether
anything is sent are all stated on the command line -- there is no default
list, because that is a judgement about a desk's coverage rather than a
property of the method.

Run:

    python scripts/rolling_vs_expanding_backtest.py
        --tree "<name>" [--tree "<name>" ...]
        [--pruning-tree "<name>" ...]
        --max-workers N (--telegram | --no-telegram) [--skip-pruning]

After both windows are done for the trees named with --pruning-tree (a
TEV variant, not the leverage variants), the dynamic adaptive pruning
backtest (scripts/adaptive_pruning_backtest.evaluate_adaptive_pruning) is
also run, with cumulative evidence (evidence_window_years=None) -- validated
on 2026-08-09 to clearly beat both a 3-year rolling evidence window and the
old static one-shot gate this replaced. Its reference walk-forward
(expanding=False) hits this job's own rolling run's in-memory cache, so the
only added cost is the per-fold candidate re-estimation. Never writes a new
tree: this is evidence, not promotion. Use --skip-pruning to leave a run's
wall-clock unchanged.
"""

from __future__ import annotations

import argparse
import html
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

# NOTE (ecosystem cleanup, 2026-08-11): these two inserts are still here on
# purpose. Removing them means turning `project/` into a package and
# rewriting the flat sibling imports inside it — `tree_studio` imports
# `tree_studio_v2`, which imports `advisor`, and so on. That is a change to
# public engine files with a real blast radius, and it belongs in its own
# scoped step rather than being done in passing while extracting a preset.
# What this commit does remove is the committee's configuration: which trees,
# which of them are pruned, how many workers and whether anything is sent are
# now all stated on the command line.
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "project"))

import tree_studio  # noqa: E402  (reuses _run_full_backtest/_v2_export_artifacts)
from adaptive_pruning_backtest import evaluate_adaptive_pruning  # noqa: E402

from lazyportfolio.v2 import run_history, store  # noqa: E402


def _log(message: str) -> None:
    print(f"[{datetime.now(UTC).isoformat()}] {message}", flush=True)


def _send_telegram_document(*, content: bytes, filename: str, caption: str) -> None:
    """Best-effort delivery -- an unattended overnight run must never lose
    the backtest results (already durably saved to run_history at this
    point) just because Telegram is unreachable or credentials are
    missing. Logs and returns instead of raising.
    """
    import os

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        _log("Telegram: TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID not set, skipping send")
        return
    try:
        from lazytools.connectors.telegram.client import TelegramClient

        client = TelegramClient.from_token(token)
        try:
            client.send_document(
                chat_id=chat_id, document=content, filename=filename, caption=caption
            )
        finally:
            client.close()
        _log(f"Telegram: sent {filename!r} ({len(content)} bytes)")
    except Exception as exc:  # noqa: BLE001 -- best-effort notification, never fatal
        _log(f"Telegram: send failed ({type(exc).__name__}: {exc}), continuing anyway")


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
    max_workers: int,
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
  &middot; max_workers={max_workers}<br>
  weekly estimation, monthly rebalance, 0 transaction cost, 0 risk-free rate
</p>
<h2>Metrics by arm</h2>
{_metrics_table_html(metrics)}
<h2>Terminal weights (FINAL)</h2>
<table><tr><th>instrument</th><th>weight</th></tr>{weights_rows}</table>
</body></html>
"""


def run_variant(name: str, *, expanding: bool, max_workers: int,
                send_telegram: bool = True) -> dict[str, Any]:
    """Run one methodology and attach TWO artifacts to its run_history row:
    the real Tree Studio client report (``kind="report"``, built by
    ``tree_studio._v2_export_artifacts`` -- the exact same
    ``build_client_report`` output the app itself would produce for this
    config, not a reimplementation) and a lightweight ``kind="summary"``
    comparison table for a fast side-by-side glance without opening both
    full reports.

    Calls ``tree_studio._run_full_backtest`` directly (not
    ``HierarchicalV2Backtester().run()``) so the walk-forward computation
    happens exactly once per config/expanding combination -- the later
    ``_v2_export_artifacts`` call hits ``_run_full_backtest``'s own
    in-memory cache (same config_hash/data_fingerprint/expanding key)
    instead of re-running the backtest a second time.
    """
    label = "expanding" if expanding else "rolling"
    config = store.read_model(name)
    config_hash = tree_studio._config_hash(config)
    data_as_of, data_fingerprint = tree_studio._data_fingerprint(config)
    backtest = config.get("backtest", {}) or {}
    train_size = int(backtest.get("train_size") or 104)

    _log(
        f"{name!r} [{label}]: starting -- train_size={train_size}, "
        f"max_workers={max_workers}"
    )
    started = time.perf_counter()
    model, dataset, report = tree_studio._run_full_backtest(
        config, capture_audit_series=False, max_workers=max_workers, expanding=expanding
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

    export_started = time.perf_counter()
    artifacts = tree_studio._v2_export_artifacts(
        config, kind="report", max_workers=max_workers, expanding=expanding
    )
    client_blob, client_content_type, _client_filename = artifacts["report"]
    export_wall = time.perf_counter() - export_started
    run_history.attach_artifact(
        run_id,
        kind="report",
        content_type=client_content_type,
        filename=f"{store.sanitize_model_name(name)}_{label}_client_report.html",
        blob=client_blob,
    )
    _log(
        f"{name!r} [{label}]: real Tree Studio client report attached "
        f"({len(client_blob)} bytes, +{export_wall:.1f}s -- cheap, hit the "
        f"_run_full_backtest cache from above)"
    )
    final_metrics = report.metrics.get("FINAL", {})
    caption = (
        f"{name} -- {label} window\n"
        f"{len(report.folds)} folds, {wall:.0f}s wall-clock, {max_workers} workers\n"
        f"CAGR {final_metrics.get('cagr', 0):.2%} | "
        f"vol {final_metrics.get('annualized_volatility', 0):.2%} | "
        f"Sharpe {final_metrics.get('annualized_sharpe', 0):.2f} | "
        f"max DD {final_metrics.get('max_drawdown', 0):.2%}"
    )
    if send_telegram:
        _send_telegram_document(
            content=client_blob,
            filename=f"{store.sanitize_model_name(name)}_{label}_client_report.html",
            caption=caption,
        )

    summary_html = _build_report_html(
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
        max_workers=max_workers,
    )
    run_history.attach_artifact(
        run_id,
        kind="summary",
        content_type="text/html",
        filename=f"{store.sanitize_model_name(name)}_{label}_summary.html",
        blob=summary_html.encode("utf-8"),
    )
    _log(f"{name!r} [{label}]: recorded run_id={run_id}, both artifacts attached")
    return payload


def _evaluate_and_send_adaptive_pruning(name: str, *, max_workers: int,
                                        send_telegram: bool = True) -> None:
    """Dynamic, per-fold, causal pruning evidence -- reused evidence window
    is cumulative (``evidence_window_years=None``): each rebalance date
    prunes from the node-vs-father OOS track record accumulated since
    burn-in, not a fixed lookback. Validated on the two base variants
    (2026-08-09) to clearly beat both a 3-year rolling evidence window and
    the previous static one-shot pruning gate this replaces here.

    ``expanding=False`` (the ``run()`` default) so the reference walk-forward
    inside ``evaluate_adaptive_pruning`` hits ``_run_full_backtest``'s
    in-memory cache from this job's own ``run_variant(name, expanding=False)``
    call above instead of recomputing it. A failure here must never take
    down the job that already durably saved the rolling/expanding results.
    """
    try:
        payload = evaluate_adaptive_pruning(name, workers=max_workers, evidence_window_years=None)
    except Exception as exc:  # noqa: BLE001 -- pruning evidence is a bonus, never fatal to the job
        _log(f"{name!r}: adaptive pruning failed ({type(exc).__name__}: {exc}), continuing")
        return
    final = payload["metrics"].get("FINAL", {})
    static = payload["metrics"].get("STATIC_FINAL", {})
    _log(
        f"{name!r}: adaptive pruning evaluated -- Sharpe {final.get('annualized_sharpe', 0):.3f} "
        f"vs static {static.get('annualized_sharpe', 0):.3f} (run_id={payload['run_id']})"
    )
    _send_telegram_document(
        content=payload["report_html"],
        filename=f"{store.sanitize_model_name(name)}_adaptive_pruning_report.html",
        caption=(
            f"{name} -- adaptive pruning (dynamic, cumulative evidence)\n"
            f"Sharpe {final.get('annualized_sharpe', 0):.2f} vs static "
            f"{static.get('annualized_sharpe', 0):.2f}\n"
            f"CAGR {final.get('cagr', 0):.2%} | MaxDD {final.get('max_drawdown', 0):.2%}"
        ),
    )


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(
        description="Rolling versus expanding walk-forward back-test, with an "
                    "optional adaptive-pruning evaluation per tree.")
    parser.add_argument(
        "--tree", action="append", dest="trees", required=True, metavar="NAME",
        help="A tree to back-test. Required and repeatable: there is no default "
             "list, because which trees a desk watches is its judgement, not a "
             "property of the method.",
    )
    parser.add_argument(
        "--pruning-tree", action="append", dest="pruning_trees", default=[],
        metavar="NAME",
        help="A tree that also gets the adaptive-pruning evaluation, which "
             "costs considerably more. Repeatable; must name a tree given with "
             "--tree.",
    )
    parser.add_argument(
        "--max-workers", type=int, required=True,
        help="Parallel workers for the backtest. Stated rather than defaulted: "
             "it is a property of the machine the job runs on.",
    )
    parser.add_argument(
        "--telegram", dest="telegram", action="store_true", default=None,
        help="Send each variant's report. Best-effort: an unattended run must "
             "not lose results already saved because a token is missing.",
    )
    parser.add_argument(
        "--no-telegram", dest="telegram", action="store_false",
        help="Do not send anything.",
    )
    parser.add_argument(
        "--skip-pruning", action="store_true",
        help="skip the per-tree pruning evaluation (report-only; leaves wall-clock unchanged)",
    )
    args = parser.parse_args(argv)
    if args.telegram is None:
        parser.error("choose --telegram or --no-telegram; sending is not a default")
    if args.max_workers <= 0:
        parser.error("--max-workers must be positive")
    names = args.trees
    unknown = [t for t in args.pruning_trees if t not in names]
    if unknown:
        parser.error(
            f"--pruning-tree names {unknown}, which are not among --tree; a tree "
            f"cannot be pruned in a run that does not back-test it")
    _log(f"=== rolling vs expanding backtest job starting for {names!r} ===")
    for name in names:
        rolling = run_variant(name, expanding=False, max_workers=args.max_workers,
                              send_telegram=args.telegram)
        expanding = run_variant(name, expanding=True, max_workers=args.max_workers,
                                send_telegram=args.telegram)
        _log(
            f"=== {name!r} done === "
            f"rolling: wall={rolling['wall_clock_seconds']:.1f}s "
            f"folds={rolling['fold_count']} | "
            f"expanding: wall={expanding['wall_clock_seconds']:.1f}s "
            f"folds={expanding['fold_count']}"
        )
        if not args.skip_pruning and name in args.pruning_trees:
            _evaluate_and_send_adaptive_pruning(name, max_workers=args.max_workers,
                                                send_telegram=args.telegram)
    _log("=== job complete ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
