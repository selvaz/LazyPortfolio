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
import argparse, json, sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "project")]
import tree_studio  # noqa: E402
from lazyportfolio.calendar import _annualization_factor  # noqa: E402
from lazyportfolio.models import BacktestSpec  # noqa: E402
from lazyportfolio.walk_forward import prepare_walk_forward_inputs  # noqa: E402
from lazyportfolio.v2.backtest import HierarchicalV2Backtester, _V2Ledger  # noqa: E402
from lazyportfolio.v2.contracts import V2BacktestReport, V2Fold  # noqa: E402
from lazyportfolio.v2.hierarchy import HierarchicalV2Estimator  # noqa: E402
from lazyportfolio.v2.model import V2Model  # noqa: E402
from lazyportfolio.v2 import run_history, store  # noqa: E402
from lazyportfolio.v2.tree_pruning import PruningRule, prune_config, rule_payload  # noqa: E402
from pruning_runner import significance_report  # noqa: E402
from tree_studio_v2.exports import build_client_report, _tree_html  # noqa: E402
from html import escape  # noqa: E402

DEFAULT_BURN_IN_YEARS = 2.0


def _accumulated_metrics(curves: dict, node_names: list[str], as_of, window_start=None) -> dict[str, dict]:
    """NODE:/FATHER: metrics from the reference run's own curves, sliced to
    ``[window_start, as_of]`` -- no future information, nothing re-solved.

    ``window_start=None`` means "from the very start of the backtest" (pure
    expanding evidence, growing every fold). A concrete date instead makes
    the evidence a rolling lookback of fixed length, so a profile/branch
    that won early cannot stay locked in purely from accumulated inertia.
    """
    metrics = {}
    for name in node_names:
        node_curve = curves[f"NODE:{name}"].loc[window_start:as_of]
        father_curve = curves[f"FATHER:{name}"].loc[window_start:as_of]
        if node_curve.empty or father_curve.empty:
            continue
        metrics[f"NODE:{name}"] = HierarchicalV2Backtester._metrics(node_curve)
        metrics[f"FATHER:{name}"] = HierarchicalV2Backtester._metrics(father_curve)
    return metrics


def _prepare_adaptive_fold(config, train, holding, mode, ppy, accumulated_metrics, rule):
    """One post-burn-in fold: prune from accumulated evidence, re-estimate
    only the resulting candidate for this fold's own training window.

    ``mode="forward_backward"`` already solves both passes in one call; the
    candidate must be re-estimated the same way as the reference tree, so
    both FINAL (backward) and FORWARD_FINAL stay available for the pruned
    structure too, not just the unpruned baseline.
    """
    candidate, decisions = prune_config(config, {"accumulated": accumulated_metrics}, rule)
    candidate_model = V2Model.from_config(candidate)
    estimator = HierarchicalV2Estimator()
    estimate = estimator.estimate(candidate_model, train, mode=mode, periods_per_year=ppy)
    forward_target = (
        dict(estimate.forward_node_results[candidate_model.root.name].terminal_weights)
        if estimate.forward_node_results else dict(estimate.terminal_weights)
    )
    return candidate, decisions, estimate, forward_target


def run(
    config: dict, *, expanding: bool = False, max_folds: int | None = None,
    workers: int = 1, burn_in_years: float = DEFAULT_BURN_IN_YEARS,
    evidence_window_years: float | None = None,
):
    model, dataset = tree_studio._v2_inputs(config)
    bt, mode = config["backtest"], tree_studio._v2_mode(config)
    train_size, freq = int(bt.get("train_size") or 104), str(bt.get("estimation_frequency") or "W")
    instruments = list(dict.fromkeys(
        [*model.root.terminal_instruments(), *(n.proxy for n in model.root.walk() if n.proxy), *model.benchmark.weights]
    ))
    node_names = [node.name for node in model.root.walk() if node.proxy is not None]
    rule = PruningRule(required_protocols=("accumulated",))
    ppy = _annualization_factor(freq)

    # One reference walk-forward of the unpruned tree over the WHOLE
    # available history: gives us B0/STATIC_FINAL for free (its own B0/FINAL
    # curves) and every node's accumulated-OOS evidence for the decisions
    # below.  Not truncated by max_folds -- burn-in and the accumulated
    # slices both need the true start of history regardless of how many
    # folds get reported.
    _, _, reference_report = tree_studio._run_full_backtest(
        config, capture_audit_series=False, max_workers=workers, expanding=expanding
    )
    reference_folds = reference_report.folds
    burn_in_cutoff = reference_folds[0].holding_start + pd.DateOffset(years=burn_in_years)

    valuation, estimation, _schedule = prepare_walk_forward_inputs(
        dataset.returns, instruments,
        BacktestSpec(id="adaptive-pruning", train_size=train_size, rebalance_frequency=str(bt.get("rebalance_frequency") or "M")),
        freq,
    )
    report_folds = reference_folds[-(max_folds + 1):] if max_folds is not None else reference_folds

    specs = []
    for ref_fold in report_folds:
        train = estimation.loc[ref_fold.training_start:ref_fold.training_end]
        holding = valuation.loc[ref_fold.holding_start:ref_fold.holding_end]
        specs.append((ref_fold, train, holding))

    def _window_start(signal):
        return signal - pd.DateOffset(years=evidence_window_years) if evidence_window_years else None

    post_burn_in = [(i, spec) for i, spec in enumerate(specs) if spec[0].signal >= burn_in_cutoff]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        prepared_by_index = dict(zip(
            (i for i, _ in post_burn_in),
            pool.map(
                lambda spec: _prepare_adaptive_fold(
                    config, spec[1], spec[2], mode, ppy,
                    _accumulated_metrics(
                        reference_report.curves, node_names, spec[0].signal, _window_start(spec[0].signal),
                    ),
                    rule,
                ),
                (spec for _, spec in post_burn_in),
            ),
        ))

    adaptive_arms = ("FINAL", "FORWARD_FINAL")
    ledgers = {name: _V2Ledger(float(bt.get("transaction_cost_bps") or 0)) for name in adaptive_arms}
    points = {name: [] for name in adaptive_arms}
    folds: list[V2Fold] = []
    decisions = []
    last_estimate = None
    last_candidate = None
    for i, (ref_fold, train, holding) in enumerate(specs):
        if i in prepared_by_index:
            candidate, fold_decisions, estimate, forward_target = prepared_by_index[i]
            adaptive_target = dict(estimate.terminal_weights)
            last_estimate = estimate
            last_candidate = candidate
            audits = {name: r.audit for name, r in estimate.node_results.items()}
            candidate_nodes = len(candidate["nodes"])
        else:
            # Burn-in: no evidence yet, hold the unpruned tree's own decision for this fold.
            fold_decisions = []
            adaptive_target = dict(ref_fold.targets["FINAL"])
            forward_target = dict(ref_fold.targets.get("FORWARD_FINAL", adaptive_target))
            audits = dict(ref_fold.audits)
            candidate_nodes = len(config["nodes"])
        targets_this_fold = {"FINAL": adaptive_target, "FORWARD_FINAL": forward_target}
        for arm, target in targets_this_fold.items():
            cost = ledgers[arm].rebalance(target)
            for first, (day, row) in enumerate(holding.iterrows()):
                points[arm].append((day, ledgers[arm].step(row) - (cost if first == 0 else 0)))
        folds.append(V2Fold(
            ref_fold.signal, train.index.min(), train.index.max(),
            holding.index.min(), holding.index.max(),
            {
                "B0": dict(model.benchmark.weights), "STATIC_FINAL": dict(ref_fold.targets["FINAL"]),
                **targets_this_fold,
            },
            audits,
        ))
        static_weights = ref_fold.targets["FINAL"]
        names = set(static_weights) | set(adaptive_target)
        decisions.append({
            "signal": str(ref_fold.signal.date()),
            "burn_in": i not in prepared_by_index,
            "nodes": fold_decisions,
            "candidate_nodes": candidate_nodes,
            "static_weights": static_weights,
            "adaptive_weights": adaptive_target,
            "target_l1_distance": sum(abs(static_weights.get(x, 0) - adaptive_target.get(x, 0)) for x in names),
        })

    report_start, report_end = specs[0][0].holding_start, specs[-1][0].holding_end
    curves = {
        "B0": reference_report.curves["B0"].loc[report_start:report_end],
        "STATIC_FINAL": reference_report.curves["FINAL"].loc[report_start:report_end],
        **{
            arm: pd.Series(
                [x[1] for x in points[arm]], index=pd.DatetimeIndex([x[0] for x in points[arm]]),
            )
            for arm in adaptive_arms
        },
    }
    report = V2BacktestReport(
        mode=mode, folds=folds, curves=curves,
        metrics={a: HierarchicalV2Backtester._metrics(c) for a, c in curves.items()},
        transaction_cost_paid={
            "B0": 0.0, "STATIC_FINAL": 0.0,
            **{arm: ledgers[arm].total_cost for arm in adaptive_arms},
        },
    )
    return dataset, report, last_estimate, decisions, rule, burn_in_cutoff, last_candidate


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
        stats.append({
            "node_name": name, "pruned": bucket.get("prune", 0), "retained": bucket.get("retain", 0),
            "total": total, "prune_rate": bucket.get("prune", 0) / total if total else 0.0,
        })
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
        "rivalutando la decisione ad ogni ribilanciamento sull'evidenza OOS accumulata fino a quel momento. "
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
            '<p class="note">Struttura del candidato realmente usato per i pesi correnti dell\'ultimo fold -- '
            "confronta con l'albero completo sopra (\"Albero di allocazione\") per vedere esattamente "
            "quali rami sono stati tagliati o promossi a quella data.</p>" + _tree_html(last_candidate)
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
    dataset, report, estimate, decisions, rule, burn_in_cutoff, last_candidate = run(
        config, expanding=expanding, max_folds=max_folds,
        workers=workers, burn_in_years=burn_in_years,
        evidence_window_years=evidence_window_years,
    )
    html = build_client_report(config=config, data_metadata=dataset.metadata, estimate=estimate, report=report)
    extra_html = _pruning_evidence_html(decisions, last_candidate)
    if extra_html:
        marker = b"</main></body></html>"
        html = html.replace(marker, extra_html.encode("utf-8") + marker)
    as_of, fingerprint = tree_studio._data_fingerprint(config)
    significance = significance_report(
        report.curves, [("FINAL", "STATIC_FINAL")],
        resample_frequency=config["backtest"].get("rebalance_frequency"),
    )
    payload = {
        "tree": tree_name, "rule": rule_payload(rule), "burn_in_years": burn_in_years,
        "evidence_window_years": evidence_window_years,
        "rebalance_frequency": rebalance_frequency or config["backtest"].get("rebalance_frequency"),
        "burn_in_cutoff": str(burn_in_cutoff.date()), "decisions": decisions, "metrics": report.metrics,
        "significance": significance, "fold_count": len(report.folds),
    }
    run_id = run_history.record_run(
        cache_key=f"adaptive-pruning:{tree_name}:{datetime.now(UTC).isoformat()}", path="/scripts/adaptive_pruning_backtest",
        kind="adaptive_pruning", tree_id=store.sanitize_model_name(tree_name), config_hash=tree_studio._config_hash(config),
        data_as_of=as_of, data_fingerprint=fingerprint, weights=estimate.terminal_weights, metrics=report.metrics, payload=payload,
    )
    run_history.attach_artifact(run_id, kind="report", content_type="text/html; charset=utf-8", filename="adaptive_pruning_report.html", blob=html)
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
    p.add_argument(
        "--evidence-window-years", type=float, default=None,
        help="rolling lookback (years) for the accumulated NODE:/FATHER: evidence; default: everything since burn-in (pure expanding)",
    )
    p.add_argument(
        "--rebalance-frequency", default=None,
        help="override backtest.rebalance_frequency for this run only (e.g. 'W'); the stored tree is never modified",
    )
    args = p.parse_args()
    payload = evaluate_adaptive_pruning(
        args.tree, workers=args.workers, expanding=args.expanding, max_folds=args.max_folds,
        burn_in_years=args.burn_in_years, evidence_window_years=args.evidence_window_years,
        rebalance_frequency=args.rebalance_frequency,
    )
    print(json.dumps({
        "run_id": payload["run_id"], "metrics": payload["metrics"],
        "significance": payload["significance"], "folds": payload["fold_count"],
    }, default=str, indent=2))


if __name__ == "__main__":
    main()
