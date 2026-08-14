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
python -m scripts.prune_tree_by_father "Global Multi-Asset - Selective Depth" --write-tree --output-name "Global Multi-Asset - Production Pruned"
```

Use `--min-sharpe-improvement` and `--max-drawdown-per-vol-ratio` only to change
the explicitly recorded policy.  Without `--write-tree` the process runs and
records its evidence but does not create a final saved tree.

## Adaptive pruning

Adaptive pruning is a separate, causal backtest of the same universal method.
At each rebalance it compares every node with its father using only
out-of-sample observations strictly earlier than the signal date. The evidence
can be expanding or limited to a rolling window, and a burn-in period prevents
decisions before enough evidence exists.

The reusable implementation lives in
`lazyportfolio.v2.adaptive_pruning`. It owns policy validation, evidence
selection, pruning decisions, candidate re-estimation, weights and summary
metrics. Both the command-line adapter and Tree Studio call this backend; they
do not reimplement the method.

Tree Studio exposes the backend at `POST /api/v2/adaptive-pruning`. The browser
may enable the feature and choose burn-in, evidence window, Sharpe threshold,
drawdown/volatility threshold, worker count, fold limit and expanding mode. It
only sends these parameters and renders the returned decisions, metrics and
weights. No pruning or portfolio calculation runs in JavaScript.

The generic rolling-versus-expanding job accepts repeatable `--pruning-tree`
arguments. When pruning is requested for a tree, a pruning failure is fatal to
the job so the scheduler cannot report a complete daily run with missing
evidence. Telegram delivery remains best-effort because results are persisted
before notification.
