"""One-load/two-solve counterfactual evaluator (docs/node-copilot-operational-plan.md §6.3).

Security/correctness invariant (§11): baseline and variant must share the
exact same in-memory dataset -- never independently reloaded, or a
mid-comparison data refresh could make them silently disagree on what they
are comparing. ``dataset`` is therefore always a parameter here, never
loaded inside this module (:func:`lazyportfolio.copilot.snapshot.load_snapshot`
is the one place that loads it, once, for both solves).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from lazyportfolio.backend import OptimizationDataset
from lazyportfolio.copilot.contracts import CounterfactualResult, ProposedView
from lazyportfolio.copilot.node_universe import apply_views_to_config, find_node
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
    (:func:`~lazyportfolio.copilot.node_universe.apply_views_to_config`)
    that swaps ``node_id``'s ``constraints.views`` for ``proposed_views``
    and nothing else -- the same single-node, single-field scope §11's
    patch allowlist enforces at apply time, exercised here before any
    proposal exists.
    """

    variant_config = apply_views_to_config(base_config, node_id, proposed_views)

    baseline_model = V2Model.from_config(base_config)
    variant_model = V2Model.from_config(variant_config)
    node_name = find_node(variant_model, node_id).name

    estimator = HierarchicalV2Estimator()
    baseline = estimator.estimate(
        baseline_model, dataset.returns, mode=mode, periods_per_year=periods_per_year
    )
    variant = estimator.estimate(
        variant_model, dataset.returns, mode=mode, periods_per_year=periods_per_year
    )

    delta_terminal = _delta(baseline.terminal_weights, variant.terminal_weights)
    turnover_one_way = 0.5 * sum(abs(value) for value in delta_terminal.values())

    baseline_node = baseline.node_results.get(node_name)
    variant_node = variant.node_results.get(node_name)
    delta_local = (
        _delta(baseline_node.local_weights, variant_node.local_weights)
        if baseline_node is not None and variant_node is not None
        else {}
    )

    solver_versions: dict[str, str] = {}
    if variant_node is not None:
        solver_versions = {
            "solver_strategy": variant_node.audit.solver_strategy,
            "problem_class": variant_node.audit.problem_class,
        }

    return CounterfactualResult(
        baseline={
            "terminal_weights": baseline.terminal_weights,
            "node_audit": asdict(baseline_node.audit) if baseline_node is not None else None,
        },
        variant={
            "terminal_weights": variant.terminal_weights,
            "node_audit": asdict(variant_node.audit) if variant_node is not None else None,
        },
        delta={"terminal_weights": delta_terminal, "local_weights": delta_local},
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
