"""Causal walk-forward backtest of deterministic adaptive tree pruning.

One reference walk-forward of the *unpruned* tree is run over the whole
available history (the same computation every plain backtest already does).
Its per-node OOS curves (``report.curves["NODE:<name>"]`` /
``"FATHER:<name>"``) already contain, for every node, exactly the realized
out-of-sample track record a live analyst would have observed at any past
date -- nothing needs to be invented or re-solved to get it.

After an initial burn-in window (no pruning yet -- there isn't enough OOS
history to judge anything), every subsequent rebalance slices those curves
from the very start up to (not including) the current date, scores node vs
father on that accumulated slice, and prunes accordingly -- a fresh,
reversible decision every time, using only information available as of that
date. The candidate tree is then re-estimated for that one fold to get the
period's actual holding weights. ``B0`` and ``STATIC_FINAL`` need no extra
work at all: they are just the reference run's own ``B0``/``FINAL`` curves.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import UTC, datetime
from html import escape

from project import tree_studio
from project.tree_studio_v2.exports import _tree_html, build_client_report

from lazyportfolio.v2 import run_history, store
from lazyportfolio.v2.adaptive_pruning import (
    DEFAULT_BURN_IN_YEARS,
    AdaptivePruningPolicy,
    run_adaptive_pruning,
)
from lazyportfolio.v2.tree_pruning import rule_payload
from scripts.pruning_runner import significance_report


def _node_prune_stats(decisions: list[dict]) -> list[dict]:
    """Per-node prune/retain tally across every post-burn-in fold -- how
    often a branch's own accumulated OOS record actually justified cutting
    it, versus a single point-in-time verdict that says nothing about
    stability over the whole backtest.
    """
    counts: dict[str, dict[str, int]] = {}
    for fold in decisions:
        for node in fold.get("nodes", []):
            name = str(node["node_name"])
            bucket = counts.setdefault(name, {"prune": 0, "retain": 0})
            bucket[node["decision"]] = bucket.get(node["decision"], 0) + 1
    stats = []
    for name, bucket in sorted(counts.items()):
        total = bucket.get("prune", 0) + bucket.get("retain", 0)
        stats.append(
            {
                "node_name": name,
                "pruned": bucket.get("prune", 0),
                "retained": bucket.get("retain", 0),
                "total": total,
                "prune_rate": bucket.get("prune", 0) / total if total else 0.0,
            }
        )
    return stats


def _pruning_evidence_html(decisions: list[dict], last_candidate: dict | None) -> str:
    """Extra report sections a plain client report doesn't have: which
    branches were actually cut (not just aggregate performance numbers) and
    how stable that verdict was over time. Spliced into ``build_client_
    report``'s own HTML rather than duplicating its whole layout.
    """
    stats = _node_prune_stats(decisions)
    if not stats:
        return ""
    stats_rows = "".join(
        "<tr><td>" + escape(s["node_name"]) + f"</td><td>{s['pruned']}</td><td>{s['retained']}</td>"
        f"<td>{s['total']}</td><td>{s['prune_rate']:.0%}</td></tr>"
        for s in stats
    )
    stats_html = (
        "<h2>Statistiche di pruning per nodo (tutti i fold post burn-in)</h2>"
        '<p class="note">Quante volte ogni ramo e stato tagliato vs mantenuto, '
        "rivalutando la decisione ad ogni ribilanciamento sull'evidenza OOS "
        "accumulata fino a quel momento. "
        "Un nodo tagliato quasi sempre e un candidato stabile alla potatura permanente; "
        "uno con un mix vicino al 50/50 e un ramo instabile, la cui classificazione dipende "
        "da quanta storia si e accumulata, non da un giudizio solido.</p>"
        "<table><thead><tr><th>Nodo</th><th>Volte prunato</th><th>Volte mantenuto</th>"
        f"<th>Fold valutati</th><th>% prunato</th></tr></thead><tbody>{stats_rows}</tbody></table>"
    )
    tree_html = ""
    if last_candidate is not None:
        tree_html = (
            "<h2>Ultimo albero pruned (fold piu recente)</h2>"
            '<p class="note">Struttura del candidato realmente usato per i pesi '
            "correnti dell'ultimo fold -- confronta con l'albero completo sopra "
            '("Albero di allocazione") per vedere esattamente '
            "quali rami sono stati tagliati o promossi a quella data.</p>"
            + _tree_html(last_candidate)
        )
    return stats_html + tree_html


def evaluate_adaptive_pruning(
    tree_name: str,
    *,
    workers: int = 4,
    expanding: bool = False,
    max_folds: int | None = None,
    burn_in_years: float = DEFAULT_BURN_IN_YEARS,
    evidence_window_years: float | None = None,
    min_sharpe_improvement: float = 0.03,
    max_drawdown_per_vol_ratio: float = 1.10,
    rebalance_frequency: str | None = None,
) -> dict:
    """Run the dynamic (per-fold, causal) adaptive pruning backtest for
    ``tree_name``, record it to ``run_history`` and attach the client
    report, and return a payload including the rendered HTML.

    The callable core behind both the CLI (``main`` below) and the nightly
    job (``rolling_vs_expanding_backtest.py``), which wires this in with
    ``evidence_window_years=None`` (cumulative evidence -- the variant
    validated on Global Multi-Asset to clearly beat both a 3-year rolling
    evidence window and the old static one-shot pruning gate). ``expanding``
    defaults to False (rolling training window) so that, when called right
    after the nightly job's own ``run_variant(name, expanding=False)``, this
    hits ``_run_full_backtest``'s in-memory cache for the reference run
    instead of recomputing it.
    """
    config = store.read_model(tree_name)
    if rebalance_frequency:
        config = deepcopy(config)
        config["backtest"]["rebalance_frequency"] = rebalance_frequency
    model, dataset, reference_report = tree_studio._run_full_backtest(
        config,
        capture_audit_series=False,
        max_workers=workers,
        expanding=expanding,
    )
    policy = AdaptivePruningPolicy(
        burn_in_years=burn_in_years,
        evidence_window_years=evidence_window_years,
        min_sharpe_improvement=min_sharpe_improvement,
        max_drawdown_per_vol_ratio=max_drawdown_per_vol_ratio,
        workers=workers,
        max_folds=max_folds,
        expanding=expanding,
    )
    result = run_adaptive_pruning(
        config,
        model=model,
        dataset=dataset,
        reference_report=reference_report,
        mode=tree_studio._v2_mode(config),
        policy=policy,
    )
    report = result.report
    html = build_client_report(
        config=config,
        data_metadata=dataset.metadata,
        estimate=result.estimate,
        report=report,
    )
    extra_html = _pruning_evidence_html(result.decisions, result.last_candidate)
    if extra_html:
        marker = b"</main></body></html>"
        html = html.replace(marker, extra_html.encode("utf-8") + marker)
    as_of, fingerprint = tree_studio._data_fingerprint(config)
    significance = significance_report(
        report.curves,
        [("FINAL", "STATIC_FINAL")],
        resample_frequency=config["backtest"].get("rebalance_frequency"),
    )
    payload = {
        "tree": tree_name,
        "policy": policy.payload(),
        "rule": rule_payload(result.rule),
        "burn_in_years": burn_in_years,
        "evidence_window_years": evidence_window_years,
        "rebalance_frequency": rebalance_frequency or config["backtest"].get("rebalance_frequency"),
        "burn_in_cutoff": str(result.burn_in_cutoff.date()),
        "decisions": result.decisions,
        "metrics": report.metrics,
        "significance": significance,
        "fold_count": len(report.folds),
    }
    run_id = run_history.record_run(
        cache_key=f"adaptive-pruning:{tree_name}:{datetime.now(UTC).isoformat()}",
        path="/scripts/adaptive_pruning_backtest",
        kind="adaptive_pruning",
        tree_id=store.sanitize_model_name(tree_name),
        config_hash=tree_studio._config_hash(config),
        data_as_of=as_of,
        data_fingerprint=fingerprint,
        weights=result.estimate.terminal_weights,
        metrics=report.metrics,
        payload=payload,
    )
    run_history.attach_artifact(
        run_id,
        kind="report",
        content_type="text/html; charset=utf-8",
        filename="adaptive_pruning_report.html",
        blob=html,
    )
    payload["run_id"] = run_id
    payload["report_html"] = html
    return payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tree")
    p.add_argument("--expanding", action="store_true")
    p.add_argument("--max-folds", type=int)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--burn-in-years", type=float, default=DEFAULT_BURN_IN_YEARS)
    p.add_argument("--min-sharpe-improvement", type=float, default=0.03)
    p.add_argument("--max-drawdown-per-vol-ratio", type=float, default=1.10)
    p.add_argument(
        "--evidence-window-years",
        type=float,
        default=None,
        help=(
            "rolling lookback (years) for accumulated NODE/FATHER evidence; "
            "default: everything since burn-in (pure expanding)"
        ),
    )
    p.add_argument(
        "--rebalance-frequency",
        default=None,
        help=(
            "override backtest.rebalance_frequency for this run only (e.g. 'W'); "
            "the stored tree is never modified"
        ),
    )
    args = p.parse_args()
    payload = evaluate_adaptive_pruning(
        args.tree,
        workers=args.workers,
        expanding=args.expanding,
        max_folds=args.max_folds,
        burn_in_years=args.burn_in_years,
        evidence_window_years=args.evidence_window_years,
        min_sharpe_improvement=args.min_sharpe_improvement,
        max_drawdown_per_vol_ratio=args.max_drawdown_per_vol_ratio,
        rebalance_frequency=args.rebalance_frequency,
    )
    print(
        json.dumps(
            {
                "run_id": payload["run_id"],
                "metrics": payload["metrics"],
                "significance": payload["significance"],
                "folds": payload["fold_count"],
            },
            default=str,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
