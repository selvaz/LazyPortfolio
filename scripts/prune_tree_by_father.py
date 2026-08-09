"""Build a deterministic, OOS-validated pruned V2 tree.

The script first measures the source tree with rolling and expanding windows,
then contracts every branch that does not improve on its father under both
protocols.  It measures the candidate again and only persists a new tree when
the candidate clears the whole-tree guard in both protocols.

Usage: python scripts/prune_tree_by_father.py "Global Multi-Asset" --write-tree
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "project"))

from pruning_runner import evaluate_pruning  # noqa: E402
from rolling_vs_expanding_backtest import run_variant  # noqa: E402

from lazyportfolio.v2.tree_pruning import PruningRule  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tree")
    parser.add_argument("--write-tree", action="store_true", help="persist the guarded candidate under a new name")
    parser.add_argument("--output-name")
    parser.add_argument("--min-sharpe-improvement", type=float, default=0.03)
    parser.add_argument("--max-drawdown-per-vol-ratio", type=float, default=1.10)
    args = parser.parse_args(argv)
    rule = PruningRule(args.min_sharpe_improvement, args.max_drawdown_per_vol_ratio)
    payload = evaluate_pruning(
        args.tree, run_variant, rule=rule, write_tree=args.write_tree, output_name=args.output_name,
    )
    print(json.dumps({
        "run_id": payload["run_id"],
        "global_guard_passed": payload["global_guard_passed"],
        "persisted_tree": payload["persisted_tree"],
        "decisions": payload["decisions"],
    }, indent=2))
    return 0 if payload["global_guard_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
