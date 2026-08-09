# Deterministic tree pruning

`scripts/prune_tree_by_father.py` turns the node-versus-father analysis into a
repeatable production gate.  It never overwrites the source tree.

For each non-root node it runs the same tree through two out-of-sample
protocols: rolling and expanding estimation windows.  A node is retained only
when, in **both** protocols:

- its annualised Sharpe exceeds its father by at least `0.03`; and
- its drawdown per unit of annualised volatility (`|max drawdown| / volatility`)
  does not exceed 110% of its father's ratio. A proportional rise in
  volatility and drawdown is therefore acceptable, with a 10% buffer.

Otherwise the branch is contracted: the node is removed and its proxy
becomes a direct instrument of its parent.  Thus a failed optimisation
layer falls back to the identical father exposure rather than disappearing
from the allocation universe.

A retained node whose immediate parent is pruned is lifted to the
grandparent instead of disappearing with it -- but only across exactly one
pruned hop.  If the grandparent is **also** pruned (two consecutive pruned
levels), no rescue is attempted: the whole branch below is cut outright,
including any retained descendants further down.  This keeps every
reachability guarantee local to at most two hops, with no risk of a node
ending up declared but unreachable from root.

The proposed whole tree is then run again under both protocols.  It is saved
only when its final portfolio Sharpe does not fall and its drawdown-per-volatility
ratio does not worsen in either protocol. Every decision,
the underlying OOS observations, and the final gate are saved in
`lazyportfolio.v2.run_history` with an attached HTML report.

Example:

```powershell
python scripts/prune_tree_by_father.py "Global Multi-Asset - Selective Depth" --write-tree --output-name "Global Multi-Asset - Production Pruned"
```

Use `--min-sharpe-improvement` and `--max-drawdown-per-vol-ratio` only to change
the explicitly recorded policy.  Without `--write-tree` the process runs and
records its evidence but does not create a final saved tree.
