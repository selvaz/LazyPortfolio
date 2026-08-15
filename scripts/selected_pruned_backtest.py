"""Causal walk-forward backtest of node-wise profile selection, then pruning.

Same idea as ``adaptive_pruning_backtest.py``, extended to a multi-way
choice: given N profile trees that share one economic topology (same
children/instruments/proxy per node) but differ in per-node goal/
constraints, one reference walk-forward is run once per profile over the
whole history. After burn-in, at every rebalance and for every node, the
profiles' accumulated OOS ``NODE:<name>`` curves so far are compared and the
best one is picked -- nothing new to solve for that step, it is a straight
Sharpe comparison on data every profile's reference run already produced.
The winning goal/constraints are assembled into a SELECTED config (topology
never changes), which is re-estimated for this one fold. Pruning is then
applied on top of SELECTED using the same accumulated evidence, borrowed
from whichever profile is currently winning that node -- producing FINAL.

Reports B0, <base>'s own STATIC_FINAL, SELECTED (chosen profiles, not
pruned), FINAL (chosen profiles, pruned) and FORWARD_FINAL for FINAL.
"""
from __future__ import annotations
import argparse, json, sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
from tree_studio_v2.exports import build_client_report  # noqa: E402

DEFAULT_BURN_IN_YEARS = 2.0
_BACKTEST_METHOD_KEYS = (
    "train_size", "estimation_frequency", "rebalance_frequency",
    "include_partial_last_period", "hierarchy_mode", "transaction_cost_bps",
)


def _validate_topology(base_config: dict, profile_configs: dict[str, dict]) -> None:
    ref_nodes = {str(n["id"]): n for n in base_config["nodes"]}
    ref_method = {k: (base_config.get("backtest", {}) or {}).get(k) for k in _BACKTEST_METHOD_KEYS}
    for name, config in profile_configs.items():
        method = {k: (config.get("backtest", {}) or {}).get(k) for k in _BACKTEST_METHOD_KEYS}
        if method != ref_method:
            raise ValueError(f"profile {name!r} has a different backtest method than the base tree")
        nodes = {str(n["id"]): n for n in config["nodes"]}
        if set(nodes) != set(ref_nodes):
            raise ValueError(f"profile {name!r} has different node ids than the base tree")
        for node_id, ref in ref_nodes.items():
            node = nodes[node_id]
            if (
                node.get("children", []) != ref.get("children", [])
                or node.get("instruments", []) != ref.get("instruments", [])
                or node.get("proxy", "") != ref.get("proxy", "")
            ):
                raise ValueError(f"profile {name!r} changes economic topology at node {node_id!r}")


def _accumulated_sharpe(curve, as_of, window_start=None) -> float:
    sliced = curve.loc[window_start:as_of]
    if sliced.empty:
        return float("-inf")
    return float(HierarchicalV2Backtester._metrics(sliced).get("annualized_sharpe", float("-inf")))


def _select_profiles(reference_reports: dict[str, Any], node_names: list[str], as_of, window_start=None) -> dict[str, str]:
    """Winning profile name per node, purely from accumulated OOS Sharpe.

    ``window_start=None`` compares each profile's whole track record since
    the start (pure expanding); a concrete date makes it a rolling lookback,
    so an early winner cannot stay locked in by inertia alone.
    """
    chosen = {}
    for name in node_names:
        chosen[name] = max(
            reference_reports,
            key=lambda profile: _accumulated_sharpe(reference_reports[profile].curves[f"NODE:{name}"], as_of, window_start),
        )
    return chosen


def _apply_selection(base_config: dict, profile_configs: dict[str, dict], chosen: dict[str, str]) -> dict:
    selected = deepcopy(base_config)
    profile_nodes = {name: {str(n["id"]): n for n in cfg["nodes"]} for name, cfg in profile_configs.items()}
    for node in selected["nodes"]:
        node_id = str(node["id"])
        profile = chosen.get(node.get("name"))
        if profile is None:
            continue
        source = profile_nodes[profile][node_id]
        node["goal"] = deepcopy(source.get("goal", {}))
        node["constraints"] = deepcopy(source.get("constraints", {}))
    return selected


def _prepare_selected_pruned_fold(
    base_config, profile_configs, train, mode, ppy, chosen, accumulated_metrics, rule,
):
    selected_config = _apply_selection(base_config, profile_configs, chosen)
    selected_model = V2Model.from_config(selected_config)
    estimator = HierarchicalV2Estimator()
    selected_estimate = estimator.estimate(selected_model, train, mode=mode, periods_per_year=ppy)

    candidate_config, decisions = prune_config(selected_config, {"accumulated": accumulated_metrics}, rule)
    candidate_model = V2Model.from_config(candidate_config)
    final_estimate = estimator.estimate(candidate_model, train, mode=mode, periods_per_year=ppy)
    forward_target = (
        dict(final_estimate.forward_node_results[candidate_model.root.name].terminal_weights)
        if final_estimate.forward_node_results else dict(final_estimate.terminal_weights)
    )
    return selected_config, dict(selected_estimate.terminal_weights), candidate_config, decisions, final_estimate, forward_target


def run(
    base: str, profiles: list[str], *, expanding: bool = False, max_folds: int | None = None,
    workers: int = 1, burn_in_years: float = DEFAULT_BURN_IN_YEARS,
    evidence_window_years: float | None = None, rebalance_frequency: str | None = None,
):
    base_config = store.read_model(base)
    profile_configs = {name: store.read_model(name) for name in profiles}
    if rebalance_frequency:
        # Override for this run only -- the stored trees are never
        # rewritten. Applied identically to base and every profile so
        # ``_validate_topology``'s backtest-method check still agrees.
        base_config = deepcopy(base_config)
        base_config["backtest"]["rebalance_frequency"] = rebalance_frequency
        profile_configs = {name: deepcopy(cfg) for name, cfg in profile_configs.items()}
        for cfg in profile_configs.values():
            cfg["backtest"]["rebalance_frequency"] = rebalance_frequency
    _validate_topology(base_config, profile_configs)

    model, dataset = tree_studio._v2_inputs(base_config)
    bt, mode = base_config["backtest"], tree_studio._v2_mode(base_config)
    train_size, freq = int(bt.get("train_size") or 104), str(bt.get("estimation_frequency") or "W")
    instruments = list(dict.fromkeys(
        [*model.root.terminal_instruments(), *(n.proxy for n in model.root.walk() if n.proxy), *model.benchmark.weights]
    ))
    node_names = [node.name for node in model.root.walk() if node.proxy is not None]
    # Selection covers every node including the root -- a profile's own
    # root-level goal/constraints (e.g. a TEV target) are exactly the kind
    # of thing that should be selectable, not just its children. Pruning
    # stays restricted to non-root nodes (``node_names`` above): the root
    # can never be contracted into a parent it doesn't have.
    selection_node_names = [node.name for node in model.root.walk()]
    rule = PruningRule(required_protocols=("accumulated",))
    ppy = _annualization_factor(freq)

    # One reference walk-forward PER profile, each over the whole available
    # history -- this alone gives every node's accumulated-OOS evidence for
    # both the profile-selection and the pruning decision below; nothing
    # else needs solving until a fold's SELECTED/FINAL structure is fixed.
    reference_reports = {
        name: tree_studio._run_full_backtest(cfg, capture_audit_series=False, max_workers=workers, expanding=expanding)[2]
        for name, cfg in profile_configs.items()
    }
    base_folds = reference_reports[base].folds if base in reference_reports else (
        tree_studio._run_full_backtest(base_config, capture_audit_series=False, max_workers=workers, expanding=expanding)[2].folds
    )
    burn_in_cutoff = base_folds[0].holding_start + pd.DateOffset(years=burn_in_years)

    valuation, estimation, _schedule = prepare_walk_forward_inputs(
        dataset.returns, instruments,
        BacktestSpec(id="selected-pruned", train_size=train_size, rebalance_frequency=str(bt.get("rebalance_frequency") or "M")),
        freq,
    )
    report_folds = base_folds[-(max_folds + 1):] if max_folds is not None else base_folds

    specs = []
    for ref_fold in report_folds:
        train = estimation.loc[ref_fold.training_start:ref_fold.training_end]
        holding = valuation.loc[ref_fold.holding_start:ref_fold.holding_end]
        specs.append((ref_fold, train, holding))

    def _window_start(signal):
        return signal - pd.DateOffset(years=evidence_window_years) if evidence_window_years else None

    post_burn_in = [(i, spec) for i, spec in enumerate(specs) if spec[0].signal >= burn_in_cutoff]

    def _prepare(spec):
        ref_fold, train, _holding = spec
        window_start = _window_start(ref_fold.signal)
        chosen = _select_profiles(reference_reports, selection_node_names, ref_fold.signal, window_start)
        metrics = {}
        for name in node_names:
            winner_curves = reference_reports[chosen[name]].curves
            node_curve = winner_curves[f"NODE:{name}"].loc[window_start:ref_fold.signal]
            father_curve = winner_curves[f"FATHER:{name}"].loc[window_start:ref_fold.signal]
            if node_curve.empty or father_curve.empty:
                continue
            metrics[f"NODE:{name}"] = HierarchicalV2Backtester._metrics(node_curve)
            metrics[f"FATHER:{name}"] = HierarchicalV2Backtester._metrics(father_curve)
        result = _prepare_selected_pruned_fold(base_config, profile_configs, train, mode, ppy, chosen, metrics, rule)
        return chosen, *result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        prepared_by_index = dict(zip((i for i, _ in post_burn_in), pool.map(_prepare, (spec for _, spec in post_burn_in))))

    arms = ("SELECTED", "FINAL", "FORWARD_FINAL")
    ledgers = {name: _V2Ledger(float(bt.get("transaction_cost_bps") or 0)) for name in arms}
    points = {name: [] for name in arms}
    folds: list[V2Fold] = []
    decisions_log = []
    last_estimate = None
    base_reference = reference_reports.get(base)
    for i, (ref_fold, train, holding) in enumerate(specs):
        if i in prepared_by_index:
            chosen, selected_config, selected_target, candidate_config, fold_decisions, final_estimate, forward_target = prepared_by_index[i]
            final_target = dict(final_estimate.terminal_weights)
            last_estimate = final_estimate
            audits = {name: r.audit for name, r in final_estimate.node_results.items()}
            candidate_nodes = len(candidate_config["nodes"])
        else:
            chosen, fold_decisions = {}, []
            base_target = dict((base_reference.folds[i] if base_reference else ref_fold).targets["FINAL"])
            selected_target = base_target
            final_target = base_target
            forward_target = dict((base_reference.folds[i] if base_reference else ref_fold).targets.get("FORWARD_FINAL", base_target))
            audits = dict(ref_fold.audits)
            candidate_nodes = len(base_config["nodes"])
        targets_this_fold = {"SELECTED": selected_target, "FINAL": final_target, "FORWARD_FINAL": forward_target}
        for arm, target in targets_this_fold.items():
            cost = ledgers[arm].rebalance(target)
            for first, (day, row) in enumerate(holding.iterrows()):
                points[arm].append((day, ledgers[arm].step(row) - (cost if first == 0 else 0)))
        folds.append(V2Fold(
            ref_fold.signal, train.index.min(), train.index.max(),
            holding.index.min(), holding.index.max(),
            {"B0": dict(model.benchmark.weights), "STATIC_FINAL": dict(ref_fold.targets["FINAL"]), **targets_this_fold},
            audits,
        ))
        decisions_log.append({
            "signal": str(ref_fold.signal.date()),
            "burn_in": i not in prepared_by_index,
            "profiles": chosen,
            "nodes": fold_decisions,
            "candidate_nodes": candidate_nodes,
        })

    report_start, report_end = specs[0][0].holding_start, specs[-1][0].holding_end
    curves = {
        "B0": (base_reference or reference_reports[next(iter(reference_reports))]).curves["B0"].loc[report_start:report_end],
        "STATIC_FINAL": (base_reference or reference_reports[next(iter(reference_reports))]).curves["FINAL"].loc[report_start:report_end],
        **{arm: pd.Series([x[1] for x in points[arm]], index=pd.DatetimeIndex([x[0] for x in points[arm]])) for arm in arms},
    }
    report = V2BacktestReport(
        mode=mode, folds=folds, curves=curves,
        metrics={a: HierarchicalV2Backtester._metrics(c) for a, c in curves.items()},
        transaction_cost_paid={"B0": 0.0, "STATIC_FINAL": 0.0, **{arm: ledgers[arm].total_cost for arm in arms}},
    )
    return dataset, report, last_estimate, decisions_log, rule, burn_in_cutoff


def main():
    p = argparse.ArgumentParser()
    p.add_argument("base")
    p.add_argument("profiles", nargs="+")
    p.add_argument("--expanding", action="store_true")
    p.add_argument("--max-folds", type=int)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--burn-in-years", type=float, default=DEFAULT_BURN_IN_YEARS)
    p.add_argument(
        "--evidence-window-years", type=float, default=None,
        help="rolling lookback (years) for profile-selection and pruning evidence; default: everything since burn-in (pure expanding)",
    )
    p.add_argument(
        "--rebalance-frequency", default=None,
        help="override backtest.rebalance_frequency for this run only (e.g. 'W'); the stored trees are never modified",
    )
    args = p.parse_args()
    dataset, report, estimate, decisions, rule, burn_in_cutoff = run(
        args.base, args.profiles, expanding=args.expanding, max_folds=args.max_folds,
        workers=args.workers, burn_in_years=args.burn_in_years,
        evidence_window_years=args.evidence_window_years, rebalance_frequency=args.rebalance_frequency,
    )
    base_config = store.read_model(args.base)
    if args.rebalance_frequency:
        base_config = deepcopy(base_config)
        base_config["backtest"]["rebalance_frequency"] = args.rebalance_frequency
    html = build_client_report(config=base_config, data_metadata=dataset.metadata, estimate=estimate, report=report)
    as_of, fingerprint = tree_studio._data_fingerprint(base_config)
    significance = significance_report(
        report.curves,
        [("SELECTED", "STATIC_FINAL"), ("FINAL", "STATIC_FINAL"), ("FINAL", "SELECTED")],
        resample_frequency=base_config["backtest"].get("rebalance_frequency"),
    )
    payload = {
        "base": args.base, "profiles": args.profiles, "rule": rule_payload(rule),
        "burn_in_years": args.burn_in_years, "evidence_window_years": args.evidence_window_years,
        "rebalance_frequency": base_config["backtest"].get("rebalance_frequency"),
        "burn_in_cutoff": str(burn_in_cutoff.date()),
        "decisions": decisions, "metrics": report.metrics, "significance": significance,
    }
    run_id = run_history.record_run(
        cache_key=f"selected-pruned:{args.base}:{datetime.now(UTC).isoformat()}", path="/scripts/selected_pruned_backtest",
        kind="selected_pruned", tree_id=store.sanitize_model_name(args.base), config_hash=tree_studio._config_hash(base_config),
        data_as_of=as_of, data_fingerprint=fingerprint, weights=estimate.terminal_weights, metrics=report.metrics, payload=payload,
    )
    run_history.attach_artifact(run_id, kind="report", content_type="text/html; charset=utf-8", filename="selected_pruned_report.html", blob=html)
    print(json.dumps({"run_id": run_id, "metrics": report.metrics, "significance": significance, "folds": len(report.folds)}, default=str, indent=2))


if __name__ == "__main__":
    main()
