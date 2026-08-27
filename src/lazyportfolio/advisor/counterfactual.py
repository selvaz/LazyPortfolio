"""One-load/two-solve counterfactual evaluator (docs/node-advisor-operational-plan.md §6.3).

Security/correctness invariant (§11): baseline and variant must share the
exact same in-memory dataset -- never independently reloaded, or a
mid-comparison data refresh could make them silently disagree on what they
are comparing. ``dataset`` is therefore always a parameter here, never
loaded inside this module (:func:`lazyportfolio.advisor.snapshot.load_snapshot`
is the one place that loads it, once, for both solves).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from lazyportfolio.advisor.contracts import CounterfactualResult, ProposedView
from lazyportfolio.advisor.node_universe import apply_node_views_to_config, find_node
from lazyportfolio.backend import OptimizationDataset
from lazyportfolio.v2.contracts import Mode
from lazyportfolio.v2.hierarchy import HierarchicalV2Estimator
from lazyportfolio.v2.model import V2Model


def evaluate_view_counterfactual(
    base_config: dict[str, Any],
    node_id: str,
    proposed_views: list[ProposedView],
    dataset: OptimizationDataset,
    *,
    mode: Mode,
    periods_per_year: float,
    seed: int | None = None,
) -> CounterfactualResult:
    """Solve ``base_config`` and its ``proposed_views`` variant on the exact
    same ``dataset``, and return the diff (§6.3's 8-step sequence).

    Never mutates or persists ``base_config``: the variant is a deep copy
    that swaps ``node_id``'s ``constraints.views`` for ``proposed_views``
    and nothing else -- the same single-node, single-field scope §11's
    patch allowlist enforces at apply time, exercised here before any
    proposal exists.

    A thin wrapper over :func:`evaluate_node_views_counterfactual` for the
    single-node case, which is what every proposal was before compound ones.
    """

    return evaluate_node_views_counterfactual(
        base_config,
        {node_id: proposed_views},
        dataset,
        mode=mode,
        periods_per_year=periods_per_year,
        seed=seed,
    )


def evaluate_node_views_counterfactual(
    base_config: dict[str, Any],
    node_views: dict[str, list[ProposedView]],
    dataset: OptimizationDataset,
    *,
    mode: Mode,
    periods_per_year: float,
    seed: int | None = None,
) -> CounterfactualResult:
    """Baseline versus *all* the proposed views applied together, in one solve.

    Not the sum of per-node previews: views do not compose. Two nodes each
    previewed alone against the same baseline produce two allocations neither
    of which is what you get when both are applied, because the hierarchy
    propagates each change upward through shared parents. A compound proposal
    must therefore show the human one combined solve -- which is also what
    the confirmation run will later compare the applied tree against.

    Local deltas and audits are keyed by node id, and cover every node whose
    local weights actually moved, not only the declared ones: a change at one
    node can shift a sibling's solve through their common parent, and hiding
    that would understate what approving the proposal does.
    """

    variant_config = apply_node_views_to_config(base_config, node_views)

    baseline_model = V2Model.from_config(base_config)
    variant_model = V2Model.from_config(variant_config)
    # node_results is keyed by node *name*, not id (see _validate_tree_structure).
    name_by_id = {
        node_id: find_node(variant_model, node_id).name for node_id in node_views
    }
    id_by_name = {name: node_id for node_id, name in name_by_id.items()}

    estimator = HierarchicalV2Estimator()
    baseline = estimator.estimate(
        baseline_model, dataset.returns, mode=mode, periods_per_year=periods_per_year
    )
    variant = estimator.estimate(
        variant_model, dataset.returns, mode=mode, periods_per_year=periods_per_year
    )

    delta_terminal = _delta(baseline.terminal_weights, variant.terminal_weights)
    turnover_one_way = 0.5 * sum(abs(value) for value in delta_terminal.values())

    local_weights: dict[str, dict[str, float]] = {}
    baseline_audits: dict[str, Any] = {}
    variant_audits: dict[str, Any] = {}
    solver_versions: dict[str, str] = {}
    for node_name, variant_node in variant.node_results.items():
        baseline_node = baseline.node_results.get(node_name)
        if baseline_node is None:
            continue
        delta_local = _delta(baseline_node.local_weights, variant_node.local_weights)
        declared = id_by_name.get(node_name)
        if declared is None and not any(delta_local.values()):
            continue
        key = declared or node_name
        local_weights[key] = delta_local
        baseline_audits[key] = asdict(baseline_node.audit)
        variant_audits[key] = asdict(variant_node.audit)
        if declared is not None:
            solver_versions[f"{declared}.solver_strategy"] = variant_node.audit.solver_strategy
            solver_versions[f"{declared}.problem_class"] = variant_node.audit.problem_class

    return CounterfactualResult(
        baseline={
            "terminal_weights": baseline.terminal_weights,
            "node_audits": baseline_audits,
        },
        variant={
            "terminal_weights": variant.terminal_weights,
            "node_audits": variant_audits,
        },
        delta={"terminal_weights": delta_terminal, "local_weights": local_weights},
        turnover_one_way=turnover_one_way,
        solver_versions=solver_versions,
        seed=seed,
    )


def _delta(baseline: dict[str, float], variant: dict[str, float]) -> dict[str, float]:
    """``{key: variant[key] - baseline[key]}`` over the union of both key sets."""

    return {
        instrument: variant.get(instrument, 0.0) - baseline.get(instrument, 0.0)
        for instrument in {*baseline, *variant}
    }


__all__ = ["evaluate_view_counterfactual"]
