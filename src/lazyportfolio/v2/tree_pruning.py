"""Deterministic, conservative pruning for hierarchical V2 trees.

The decision is deliberately made from out-of-sample node-versus-father
metrics.  A branch that does not clear the rule is contracted: its proxy is
made a direct candidate of its parent.  This keeps the economic exposure but
removes the local optimisation layer.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PruningRule:
    """Explicit rule used to classify a node in every required OOS protocol."""

    min_sharpe_improvement: float = 0.03
    max_drawdown_per_vol_ratio: float = 1.10
    required_protocols: tuple[str, ...] = ("rolling", "expanding")


def _arm(metrics: dict[str, dict[str, Any]], prefix: str, name: str) -> dict[str, Any] | None:
    return metrics.get(f"{prefix}:{name}")


def classify_node(
    node: dict[str, Any],
    protocol_metrics: dict[str, dict[str, dict[str, Any]]],
    rule: PruningRule = PruningRule(),
) -> dict[str, Any]:
    """Return a serialisable retain/prune decision for one non-root node.

    Missing metrics are a deterministic prune.  This is intentional: a new
    layer must prove that it adds value before it earns production complexity.
    """
    name = str(node["name"])
    observations: list[dict[str, Any]] = []
    passed = True
    reasons: list[str] = []
    for protocol in rule.required_protocols:
        metrics = protocol_metrics.get(protocol, {})
        child, father = _arm(metrics, "NODE", name), _arm(metrics, "FATHER", name)
        if child is None or father is None:
            passed = False
            reasons.append(f"{protocol}: missing NODE/FATHER out-of-sample metrics")
            continue
        sharpe_delta = float(child.get("annualized_sharpe", 0.0)) - float(
            father.get("annualized_sharpe", 0.0)
        )
        drawdown_delta = float(child.get("max_drawdown", 0.0)) - float(father.get("max_drawdown", 0.0))
        child_vol = float(child.get("annualized_volatility", 0.0))
        father_vol = float(father.get("annualized_volatility", 0.0))
        child_drawdown_per_vol = abs(float(child.get("max_drawdown", 0.0))) / child_vol if child_vol > 0 else float("inf")
        father_drawdown_per_vol = abs(float(father.get("max_drawdown", 0.0))) / father_vol if father_vol > 0 else float("inf")
        protocol_pass = (
            sharpe_delta >= rule.min_sharpe_improvement
            and child_drawdown_per_vol <= father_drawdown_per_vol * rule.max_drawdown_per_vol_ratio
        )
        observations.append(
            {
                "protocol": protocol,
                "node_sharpe": child.get("annualized_sharpe"),
                "father_sharpe": father.get("annualized_sharpe"),
                "sharpe_delta": sharpe_delta,
                "node_max_drawdown": child.get("max_drawdown"),
                "father_max_drawdown": father.get("max_drawdown"),
                "max_drawdown_delta": drawdown_delta,
                "node_drawdown_per_vol": child_drawdown_per_vol,
                "father_drawdown_per_vol": father_drawdown_per_vol,
                "passed": protocol_pass,
            }
        )
        if not protocol_pass:
            passed = False
            reasons.append(
                f"{protocol}: ΔSharpe {sharpe_delta:.3f}, "
                f"MDD/vol {child_drawdown_per_vol:.3f} vs {father_drawdown_per_vol:.3f}"
            )
    return {
        "node_id": node["id"],
        "node_name": name,
        "proxy": node.get("proxy", ""),
        "decision": "retain" if passed else "prune",
        "reason": "passes every required out-of-sample protocol" if passed else "; ".join(reasons),
        "observations": observations,
    }


def _depths(config: dict[str, Any]) -> dict[str, int]:
    by_id = {node["id"]: node for node in config["nodes"]}
    root = config["root_id"]
    result = {root: 0}
    frontier = [root]
    while frontier:
        parent_id = frontier.pop()
        for child_id in by_id[parent_id].get("children", []):
            result[child_id] = result[parent_id] + 1
            frontier.append(child_id)
    return result


def prune_config(
    config: dict[str, Any],
    protocol_metrics: dict[str, dict[str, dict[str, Any]]],
    rule: PruningRule = PruningRule(),
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Classify and contract all non-root branches, shallow to deep.

    The source config is never mutated.  A single pruned node contracts into
    its (surviving) parent, promoting its proxy once.  A retained node whose
    immediate parent is pruned is lifted to the grandparent -- but only
    across exactly one pruned hop.  If the grandparent is *also* pruned, the
    whole branch below it is cut outright: no multi-hop rescue is attempted,
    and every descendant is dropped regardless of its own classification.
    This keeps every reachability guarantee local and non-recursive.
    """
    result = deepcopy(config)
    by_id = {node["id"]: node for node in result["nodes"]}
    root_id = result["root_id"]
    parent_of = {
        child_id: node["id"]
        for node in result["nodes"]
        for child_id in node.get("children", [])
    }
    decisions = [
        classify_node(node, protocol_metrics, rule)
        for node in result["nodes"]
        if node["id"] != root_id
    ]
    decision_by_id = {item["node_id"]: item for item in decisions}
    depths = _depths(result)

    current_parent = dict(parent_of)
    removed_ids: set[str] = set()
    cut_ids: set[str] = set()

    for node_id in sorted(decision_by_id, key=lambda item: (depths[item], item)):
        decision = decision_by_id[node_id]
        parent_id = current_parent[node_id]

        if parent_id in cut_ids:
            # Inside an already-cut branch: no lifting, unconditional drop.
            cut_ids.add(node_id)
            removed_ids.add(node_id)
            decision["action"] = {"removed_with_ancestor": parent_id}
            continue

        parent_decision = decision_by_id.get(parent_id, {}).get("decision")
        parent_pruned = parent_id != root_id and parent_decision == "prune"
        if not parent_pruned:
            if decision["decision"] == "prune":
                node = by_id[node_id]
                parent = by_id[parent_id]
                parent_children = parent.get("children", [])
                parent["children"] = [child for child in parent_children if child != node_id]
                proxy = str(node.get("proxy") or "").strip()
                if proxy and proxy not in parent.setdefault("instruments", []):
                    parent["instruments"].append(proxy)
                decision["action"] = {
                    "contracted_into_parent": parent_id,
                    "promoted_proxy": proxy or None,
                }
                removed_ids.add(node_id)
            # else: retained and the parent is fine -- stays exactly as is.
            continue

        grandparent_id = current_parent.get(parent_id)
        grandparent_pruned = (
            grandparent_id is not None
            and grandparent_id != root_id
            and decision_by_id.get(grandparent_id, {}).get("decision") == "prune"
        )
        if grandparent_pruned:
            # Two consecutive pruned levels: cut the whole branch here down.
            cut_ids.add(node_id)
            removed_ids.add(node_id)
            decision["action"] = {"removed_with_ancestor": grandparent_id}
        elif decision["decision"] == "retain":
            old_parent = by_id[parent_id]
            old_children = old_parent.get("children", [])
            old_parent["children"] = [child for child in old_children if child != node_id]
            new_parent_id = grandparent_id if grandparent_id is not None else root_id
            new_parent = by_id[new_parent_id]
            if node_id not in new_parent.setdefault("children", []):
                new_parent["children"].append(node_id)
            current_parent[node_id] = new_parent_id
            decision["action"] = {"lifted_from_pruned_ancestor": parent_id}
        else:
            # A single pruned hop above, and this node fails its own father
            # too -- it is swept away together with that already-contracted
            # parent rather than lifted.
            removed_ids.add(node_id)
            decision["action"] = {"removed_with_ancestor": parent_id}

    result["nodes"] = [node for node in result["nodes"] if node["id"] not in removed_ids]
    return result, decisions


def rule_payload(rule: PruningRule) -> dict[str, Any]:
    """Stable JSON-friendly representation used in reports and history."""
    return asdict(rule)


__all__ = ["PruningRule", "classify_node", "prune_config", "rule_payload"]
