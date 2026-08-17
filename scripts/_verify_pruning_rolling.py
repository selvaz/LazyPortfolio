"""Ad-hoc verification run: rolling-only, source vs pruned-candidate, no
Telegram noise, no tree persisted.  Not part of the permanent script set.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from lazyportfolio.v2.tree_pruning import PruningRule  # noqa: E402
from pruning_runner import evaluate_pruning  # noqa: E402
from rolling_vs_expanding_backtest import run_variant  # noqa: E402

ROLLING_ONLY = PruningRule(required_protocols=("rolling",))
TREES = ["Global Multi-Asset", "Global Multi-Asset - TEV 7-10-5"]


def fmt(metrics: dict) -> str:
    return (
        f"CAGR {metrics.get('cagr', 0):.2%}  Vol {metrics.get('annualized_volatility', 0):.2%}  "
        f"Sharpe {metrics.get('annualized_sharpe', 0):.3f}  MaxDD {metrics.get('max_drawdown', 0):.2%}"
    )


def main() -> None:
    for tree in TREES:
        print(f"\n=== {tree} (rolling only) ===", flush=True)
        payload = evaluate_pruning(tree, run_variant, rule=ROLLING_ONLY, write_tree=False)
        src = payload["baseline_final"]["rolling"]
        cand = payload["candidate_final"]["rolling"]
        pruned_count = sum(1 for d in payload["decisions"] if d["decision"] == "prune")
        print(f"source (no pruning):    {fmt(src)}")
        print(f"pruned candidate:       {fmt(cand)}")
        print(f"{pruned_count}/{len(payload['decisions'])} branches would be cut, "
              f"whole-tree guard: {'PASSED' if payload['global_guard_passed'] else 'failed'}")
        print("decisions:")
        for d in payload["decisions"]:
            print(f"  {d['node_name']:30s} proxy={d.get('proxy',''):8s} -> {d['decision']:6s} {d.get('action')}")


if __name__ == "__main__":
    main()
