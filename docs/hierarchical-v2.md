# Hierarchical optimizer V2

V2 is the only hierarchical allocation engine in LazyFin. Python and Tree Studio
use the same validated model, local solver, hierarchy traversal and walk-forward
ledger. Invalid or unsupported contracts fail loudly; the engine never substitutes a
different objective, reference, estimator or financing regime silently.

For the ticker-by-ticker canonical example, see
[`hierarchical-v2-step-by-step.md`](hierarchical-v2-step-by-step.md). For the
post-PR-36 remediation record, see
[`optimizer-remediation-plan.md`](optimizer-remediation-plan.md).

## Engine architecture

`lazyfin.optimization.hierarchical_v2` is a compatibility facade. Numerical code
lives only under `lazyfin.optimization.v2`:

| Module | Responsibility |
|---|---|
| `contracts.py` | Typed constraints, hierarchy nodes, audits and reports |
| `validation.py` | Fail-loud validation and legacy JSON migration |
| `model.py` | Validated hierarchy and benchmark construction |
| `moments.py` | Covariance, expected-return and Black-Litterman calculations |
| `solver.py` | Local SLSQP/HRP optimization and financing regimes |
| `hierarchy.py` | Flat, forward and forward-backward traversal/composition |
| `backtest.py` | Causal ledger, costs, financing accrual and metrics |
| `api.py` | Stable public V2 assembly |

There are no remediation finalizers, import-time mutations, numerical shims or
runtime class replacements.

## Shared local contract

Every node solves one self-financing local portfolio:

```text
sum(local risky weights) + local cash weight = 1
```

Risky instruments remain long-only unless a separate public feature explicitly
changes that contract. Without financing controls, the local solve is fully
invested in risky instruments. `min_weights`, `max_weights` and `per_asset_cap`
apply to risky series in that local solve; the per-asset cap never applies to a
financing instrument.

Reference series are external economic anchors. Father, `forward_root_reference`
and benchmark references are not inserted into the candidate universe and are not
fallback allocations. Turnover-aware optimization remains unsupported; transaction
costs and drift belong to the walk-forward ledger.

## Optional financing on every node

Cash and leverage may be declared on the root, a leaf or an intermediate node:

```json
{
  "id": "equity_sleeve",
  "constraints": {
    "cash_enabled": true,
    "max_leverage": 1.40,
    "risk_free_rate": 0.03,
    "borrow_spread_bps": 75
  }
}
```

`cash_enabled=true` enlarges the feasible set by permitting positive local cash.
It does not require the solver to hold cash. `max_leverage` must be finite and at
least `1.0`; a value greater than one enables the borrowing regime and may produce
negative local cash. It does not require leverage to be used.

The local cash bound is:

```text
1 - max_leverage <= local cash weight <= 1
```

A node with `max_leverage=1.40` can therefore reach 140% local risky gross
exposure with -40% local cash. Positive cash earns the node's effective risk-free
rate. Negative cash pays:

```text
node effective risk-free rate + borrow_spread_bps / 10000
```

The risk-free rate resolves from node, then root, then the hard default `0.0`.
Borrow spread resolves from an explicit node value, then the root/global value,
then `0.0`. A positive spread with `max_leverage == 1` and cash disabled is a
contradictory direct financing declaration and fails loudly; spread alone never
enables borrowing.

The local optimizer evaluates lending and borrowing as separate regimes. Failure
of one regime does not discard a feasible candidate from the other. The selected
candidate must be feasible and economically preferable under the configured
objective. A fully invested point remains available inside either enabled regime,
so allowed cash or leverage can remain unused.

HRP continues to reject cash and leverage. The implemented HRP contract is a
long-only clustering allocation and has no scientifically validated financing
decision. `mean_estimator=equilibrium` is also rejected while financing is active;
`auto` resolves to Bayes-Stein for financing solves.

Black-Litterman views cannot target `cash:RF`, `cash:BORROW` or their node-qualified
ledger forms.

## Hierarchical composition

A parent sees each child as one synthetic risky asset. The child's synthetic
return already includes its local lending income or borrowing cost. The parent
never adds the child's internal cash directly to its own local cash decision.

Terminal ledger names are collision-free:

```text
cash:RF                  root lending
cash:BORROW              root borrowing
cash:RF@<node-id>        non-root lending
cash:BORROW@<node-id>    non-root borrowing
```

Root names remain unchanged for public API compatibility. Qualified non-root names
exist only at composition and ledger boundaries; the local solver continues to use
the canonical `cash:RF` or `cash:BORROW` instrument for one node at a time.

All child terminal exposures are multiplied by the weight assigned by the parent.
A sleeve held at 40% with 1.5x local risky exposure contributes 60% global risky
exposure and -20% global borrowing. The child's internal cash is not also counted
as root cash.

This rule applies recursively to leaves and intermediate nodes in flat, forward
and forward-backward modes.

## Modes

### Flat

Hierarchy nodes are solved for diagnostics, then the final portfolio is optimized
once over unique terminal risky instruments. Root financing controls apply to the
flat final solve. Non-root financing does not leak into that independent terminal
solve.

### Forward: proxy baseline and attribution

The root is solved over direct instruments and child proxies. Each child is then
solved locally. Composition replaces each proxy exposure with the child's scaled
terminal risky and financing exposures while preserving the child's synthetic
return as the sleeve return.

The forward result is a proxy-based counterfactual. It records what the parent
would allocate before the optimized child sleeves are represented by their actual
synthetic return series. It is retained for attribution and audit; it is not
required to determine the final backward solution when child solves do not depend
on the parent's forward weights.

### Forward plus backward: implemented two-pass decomposition

V2 freezes the forward pass, reconstructs child synthetic series including local
financing, and resolves ancestors bottom-up with those synthetic series. Forward
and backward audits are stored independently. Financing remains local to the node
being solved in each pass.

The backward result is the final hierarchical allocation. The difference

```text
backward parent weights - forward parent weights
```

measures the effect of replacing raw child proxies with the realized optimized
sleeves. That difference can be attributed to local asset selection, local
constraints, cash or leverage decisions, and propagation through intermediate
nodes.

Under the current dependency contract, child optimization does not consume the
forward parent weights. If every ancestor is fully re-solved with the same data,
estimators, constraints, deterministic initialization and tie-breaking, the final
forward-backward result should equal a direct bottom-up solve within numerical
tolerance. This is a tested invariant, not merely an intended one:
`HierarchicalV2Estimator.estimate_direct_bottom_up` solves every leaf directly —
with no Forward pass at all, no `forward_root_reference` — then reuses the same
recursive backward composition, and both a unit test and the scientific study's
`V2_DIRECT_BOTTOM_UP` arm confirm it agrees with `forward_backward` within
tolerance. It is not a claim that the forward pass improves the optimum.

### Planned step 3: iterative hierarchical equilibrium — not implemented

A future mode may repeatedly re-optimize nodes using synthetic series produced by
the preceding iteration. This is intentionally outside the current public V2
contract and must not be inferred from `forward_backward`.

Additional iterations can change results only when the hierarchy contains a real
feedback dependency from an ancestor to a descendant. Examples include a child
constraint relative to the current optimized father or root, an ancestor-assigned
risk budget, a conditional leverage limit, or moments that depend on the current
aggregate hierarchy. If dependencies remain only child-to-parent, one bottom-up
pass is sufficient and repeated passes are idempotent apart from solver noise.

The candidate iterative contract is:

```text
W^(k+1) = T(W^k)
residual_k = max_node ||w_node^(k+1) - w_node^k||
stop when residual_k <= tolerance
```

where `W` contains all node allocations and `T` is one complete, explicitly
ordered hierarchy update. A production implementation would require:

- a separate mode and schema rather than silently changing `forward_backward`;
- declared ancestor-to-descendant dependencies and cycle validation;
- deterministic update ordering, initialization and tie-breaking;
- configurable tolerance, maximum iterations and optional damping;
- detection of oscillations, longer cycles and financing-regime switches;
- an audit of every iteration, residual and local/global constraint state;
- fail-loud non-convergence rather than returning an unlabelled partial iterate.

The scientific study must establish existence, uniqueness and convergence only
under stated assumptions. It must also compare direct bottom-up, the implemented
two-pass attribution mode, the iterative candidate and the independent flat solve.

### Planned step 4: auditable LLM-supplied views — final research objective

The final intended layer is to integrate views produced with an LLM into the
optimized portfolios without allowing the model to emit weights, bypass constraints
or replace the numerical optimizer. The LLM acts only as a structured view generator:
a timestamped information set is transformed into typed, node-scoped Black-Litterman
views that must pass the same deterministic validation as manually supplied views.

Each accepted view must record at least its target instruments and coefficients,
direction and expected-return value, horizon, confidence, node scope, source
references, information cutoff, model and prompt version, generation parameters and
a stable identifier for the raw and parsed payloads. Unsupported instruments,
financing instruments, non-finite values, invalid confidence or data unavailable at
the decision date must fail loudly. Rejected views remain in the audit trail and do
not silently disappear.

The implementation must preserve an explicit causal chain:

```text
source snapshot
  -> LLM raw response
  -> validated typed views
  -> prior-to-posterior moment change
  -> local optimized weight change
  -> hierarchy composition change
  -> financing and final terminal exposures
```

Counterfactual runs must make each contribution observable. At minimum, reports must
compare the same configuration with no views, with the validated LLM views, and with
every later hierarchy stage held or enabled consistently. For each node and the final
portfolio, the audit should expose posterior-minus-prior moments, weights with-views
minus weights-without-views, local-to-global propagation, financing changes and the
resulting risk and performance deltas. If the future iterative step is enabled, view
effects must also be separated by iteration rather than reported only at convergence.

Walk-forward evaluation must enforce that retrieval, source documents and prompts use
only information available by the signal date. Reproduction requires immutable source
snapshots or hashes, prompt templates, model identifiers, parsed views, validation
decisions and optimizer configuration. The scientific claim is not that LLM views are
superior, but that their incremental contribution can be replayed, ablated and measured
independently from estimation, optimization, hierarchy and financing effects.

## Volatility and tracking references

Exact targets and hard caps are different contracts.

- `volatility_target_mode: "exact"` requests an equality against `constraints.
  volatility_target_policy` (`hard_fail` by default: the solve raises rather than
  project when the target cannot be matched; `nearest_feasible` opts into the
  lexicographic projection below).
- `volatility_target_mode: "cap"`, or `max_volatility`, requests an at-most hard
  inequality that is **never** relaxed — `constraints.volatility_cap_policy` may
  only ever be `hard_fail`.
- `constraints.max_tracking_error` follows `constraints.tracking_error_policy`
  (same two values, same `hard_fail` default).

Manual, father-proxy, benchmark and `forward_root_reference` references are
supported where valid (the legacy label `father` normalizes to `father_proxy`).
`forward_root_reference` names the frozen root synthetic series computed once
during the Forward diagnostic pass — the ambiguous generic `"root"` label from
earlier revisions has been renamed and is no longer accepted. Root-relative
(`forward_root_reference`) constraints are invalid on the root itself.
`current_parent_synthetic`/`current_root_synthetic` are reserved for a future,
not-yet-implemented iterative mode and are rejected explicitly in `flat`,
`forward` and `forward_backward`. Cash or borrowing may be selected when it
improves the objective or makes a target, cap, TEV limit or local weight
contract feasible; no target implicitly enables cash.

### Constraint fallback is lexicographic, never a weighted sum

When a relaxable constraint (`tracking_error_policy`/`volatility_target_policy`
set to `nearest_feasible`) cannot be matched exactly, V2 resolves it in three
explicit stages, never by combining violations into one weighted score:

1. Minimize the TEV excess in isolation (subject only to the always-hard budget/
   bounds/leverage/volatility-cap constraints).
2. Given that minimal TEV excess held fixed, minimize the volatility-target
   deviation.
3. Given both minima held fixed, optimize the actual economic objective.

`V2Audit.constraint_stage_results` records every stage that ran, in order, with
its policy, requested value, achieved value and status. A volatility cap is
never relaxed by this fallback at any stage.

## Moment estimation and views

`constraints.covariance_estimator` supports `shrunk_fixed` and `ledoit_wolf`.
`constraints.mean_estimator` supports `auto`, `equilibrium`, `bayes_stein`,
`james_stein`, `bodnar_okhrin` and `empirical`. Every audit records configured and
resolved methods and covariance roles.

### Mean reference is independent of the risk references

`constraints.mean_reference_kind` (`none`/`father_proxy`/`benchmark`/
`local_weights`) is a separate axis from `volatility_reference`/
`max_volatility_reference`/`tracking_error_reference`: configuring a TEV,
volatility target or volatility cap never implicitly selects the equilibrium
mean-estimation prior, and vice versa. `mean_estimator="auto"` resolves to
`equilibrium` if and only if a complete mean reference is configured this way,
otherwise `bayes_stein` — except two pre-existing special cases that still win
over this general rule: active cash/leverage financing, and
`objective="max_utility"`, both of which always force `bayes_stein` regardless of
the mean reference. `mean_reference_kind="local_weights"` requires
`mean_reference_weights` to cover every component actually solved by that node
(raw proxies in Forward, synthetic child series in Backward) — no weight is ever
invented for a component absent from the declared mapping, and an incomplete
mapping fails loudly rather than falling back silently. Every audit records
`mean_reference_source` and `risk_reference_source` so the two are always
distinguishable in the trace.

### Component identity

Every node's local solve resolves a `V2SolveContext`: a stable `component_id` per
economic component (a direct instrument, or a child sleeve), distinct from its raw
proxy series, its synthetic series (when one exists) and the solver column
currently used. Forward uses each child's raw proxy as its candidate; Backward
substitutes the child's own synthetic series — but this is a property of
`V2Component.kind` and the pass, not of a `_SYNTH` column-name suffix, which is
purely an internal naming detail.

Black-Litterman views are node-scoped and typed. Confidence must be in `(0, 1]`,
`view_tau` must be positive and finite, and all coefficients and expected returns
must be finite. Financing instruments are excluded from views and from per-asset
caps.

## Audit contract

Every node audit records:

- whether cash was enabled and which financing instrument was used;
- local cash weight, local risky gross exposure and local leverage limit;
- lending rate, borrowing rate and spread;
- selected financing regime;
- parent weight and cumulative node weight;
- global effective risky exposure and signed cash/borrowing exposure;
- portfolio aggregate risky, cash and net exposure;
- node/root/default provenance for risk-free rate, risk aversion, cash, leverage
  and spread;
- objective, risk constraints, bounds, estimator provenance and solver strategy.

A zero financing position is normalized to the `fully_invested` regime and is not
emitted as a terminal ledger instrument.

## Walk-forward ledger

All arms share one causal schedule and common out-of-sample grid. Training ends no
later than the signal date and holding begins after the signal. The ledger applies
normal drift, turnover and transaction costs to flattened terminal positions.

Each node-qualified lending or borrowing instrument has its own daily return
column at the correct effective local rate. This prevents collisions and ensures
that root and child financing can coexist without double counting. Terminal
weights remain economically equivalent to the composed synthetic sleeves.

Sharpe, Sortino and annualized excess return use the effective risk-free rate of
the arm being reported: node arms use the node/root/default resolution, while
portfolio arms use the root rate.

## Compatibility and validation

Existing root-only configurations retain `cash:RF` and `cash:BORROW` names and the
same public imports. `max_leverage > 1` continues to enable financing implicitly.
Non-finite controls, leverage below one, negative spread, contradictory legacy and
canonical cash flags, unsupported HRP financing, financing views and invalid
references fail before a result is accepted.

Run the engineering gates:

```text
pytest -q --cov=lazyfin --cov-fail-under=95
ruff check src tests project/tree_studio.py
mypy src/lazyfin
python project/tree_studio_v2/validate_local_solver.py
python project/tree_studio_v2/validate_example_estimates.py
python project/tree_studio_v2/validate_backtests.py
python project/tree_studio_v2/validate_exports.py
```
