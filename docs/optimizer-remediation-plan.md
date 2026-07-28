# Optimizer V2 remediation plan

Status: canonical engine rationalization, per-node financing and the matching Tree
Studio contract are complete on the implementation branch. Live environment
validation and human numerical review remain open. Scientific-study adaptation, the
iterative hierarchy and the auditable LLM-view layer described below remain deferred
until the implemented numerical contract is frozen and reviewed.

**2026-07-22 update:** the clean-engine follow-up findings this plan's own
"Hierarchy-pass semantics and research roadmap" section flagged as open — the
bottom-up/forward-backward equivalence regression, and the broader component-
identity/mean-reference-separation/lexicographic-fallback work — are now implemented
and tested; see `docs/optimizer-v2-clean-engine-follow-up.md` for the detailed spec
that was implemented against and `docs/optimizer-v2-remediation-status.md` for the
phase-by-phase record. This document's content below is retained as the historical
post-PR-#36 record and is superseded where it conflicts with those two files (notably:
`auto` resolution is no longer scoped to `max_utility` only, and the generic `root`
reference label has been renamed `forward_root_reference`).

Branch: `agent/optimizer-methodology-remediation`

## Scope

This branch addresses the methodological and implementation findings identified
after PR #36. The governing rules remain:

- one canonical V2 engine;
- no import-time mutation, monkey patching, numeric shims or patch finalizers;
- no silent reinterpretation of economic controls;
- no loss of flat, forward, forward-backward, cash, leverage, views, HRP or
  walk-forward functionality;
- engineering validation and financial-performance claims remain separate.

## Canonical architecture

`src/lazyfin/optimization/hierarchical_v2.py` is a compatibility facade only.
Numerical behavior lives under `lazyfin.optimization.v2`:

| Module | Responsibility |
|---|---|
| `contracts.py` | Typed public constraints, hierarchy, audits and reports |
| `validation.py` | Fail-loud validation and legacy JSON migration |
| `model.py` | Validated hierarchy and benchmark construction |
| `moments.py` | Covariance, expected-return and Black-Litterman calculations |
| `solver.py` | Local SLSQP/HRP solve and independent financing regimes |
| `hierarchy.py` | Flat, forward and forward-backward traversal/composition |
| `backtest.py` | Causal ledger, financing accrual, costs and metrics |
| `api.py` | Stable public V2 assembly |

The former transitional modules `remediation`, `financing`,
`financing_finalize`, `audit_finalize` and `studio_compat` remain absent. Strict
Mypy applies to the canonical engine without engine-specific exclusions.

## Completed methodological corrections

### Covariance and expected-return provenance

- `shrunk_fixed` resolves to Skfolio `ShrunkCovariance`.
- `ledoit_wolf` resolves to Skfolio `LedoitWolf`.
- `mean_estimator=auto` resolves to Bayes-Stein for `max_utility`.
- Explicit equilibrium means remain available when their fully invested risky
  reference contract is valid.
- Every audit records configured and resolved estimators and their provenance.

### Black-Litterman covariance roles

- `prior_risk` remains the default: views update expected returns while prior
  covariance continues to enforce risk controls.
- `posterior_all` remains an explicit opt-in.
- Views cannot target root or qualified per-node financing instruments.

### Risk-free and borrowing-rate semantics

- The annual risk-free rate resolves per node through node, root, then `0.0`.
- The borrowing spread resolves through node, root/global, then `0.0` for a node
  whose financing contract is active.
- Optimization, synthetic sleeve returns, ledger accrual, Sharpe, Sortino and
  annualized excess return use the same effective rate for the arm analyzed.
- A positive spread with `max_leverage == 1` and `cash_enabled == false` is
  rejected because no borrowing or lending contract is active. With explicit
  cash permission it is accepted, but borrowing remains unavailable until
  `max_leverage > 1`.

### Volatility, references and HRP

- Exact target and at-most cap remain distinct contracts.
- Father, father-proxy, root and benchmark series remain external economic
  references rather than automatically inserted candidates.
- Exact equality uses audited multi-start SLSQP and may report a nearest feasible
  projection; no global efficient-frontier claim is emitted.
- HRP remains the Skfolio Pearson/Ward/variance variant and rejects cash,
  leverage, volatility targets/caps, TEV limits, views and unsupported moments.

## Per-node cash and leverage

Cash and leverage are optional controls on every node.

For node `n`, risky weights `w_n`, local cash `c_n` and leverage limit `L_n` obey:

```text
1' w_n + c_n = 1
w_n >= 0
1 - L_n <= c_n <= 1
```

- `cash_enabled=true` admits positive local cash.
- `max_leverage > 1` admits negative local cash and continues to enable the
  borrowing contract implicitly for backward compatibility.
- Lending and borrowing are solved as separate audited regimes.
- An infeasible regime cannot suppress another feasible regime.
- Financing is selected only when it improves the configured objective or is
  needed to satisfy the declared feasible constraints.
- Per-asset caps and risky min/max bounds do not apply to financing instruments.

The local synthetic sleeve return already includes its own lending or borrowing
cash flow. A parent consumes that return as one synthetic asset and never adds the
child cash directly to parent cash. During terminal composition, every child
position, including its qualified financing position, is multiplied by the
parent allocation.

Root financing retains `cash:RF` and `cash:BORROW` for API compatibility.
Non-root financing uses collision-free ledger labels such as `cash:RF@equity` and
`cash:BORROW@duration`. This permits simultaneous root and child financing without
double counting.

Audit output now records local financing permission, chosen instrument, local cash,
local risky gross exposure, leverage limit, lending/borrowing rates, spread,
regime, parent and cumulative node weights, globally scaled risky/cash exposure,
portfolio aggregates and parameter provenance.

## Hierarchy-pass semantics and research roadmap

The current engine implements a two-pass decomposition, not an iterative fixed-point
algorithm.

### Step 1 — forward proxy baseline

The parent is solved using raw child proxies; children are then solved locally and
composed into terminal exposures. The forward allocation is retained as a
counterfactual baseline. It measures what the parent selected before the proxy was
replaced by the optimized child's realized synthetic series.

This pass is primarily diagnostic. Under the current contract, child solves do not
depend on the parent's forward weights, so the forward result need not enter the
final backward solution.

### Step 2 — realized-sleeve backward solve, implemented

Child synthetic series, including local lending or borrowing cash flows, are
reconstructed and ancestors are re-solved bottom-up. The backward allocation is
the final hierarchical result. Separate forward and backward audits permit a
proxy-to-realized-sleeve attribution of:

- local asset selection;
- local constraints and estimator choices;
- cash, leverage and borrowing spread decisions;
- propagation through intermediate nodes.

When dependencies are only child-to-parent, all ancestors are fully re-solved, and
data, estimators, initialization and tie-breaking are held constant, the
forward-backward final result should equal a direct bottom-up-only solve within
numerical tolerance. A regression establishing this equivalence is required before
it is used as a formal methodological claim. Repeating the same bottom-up pass with
unchanged inputs should otherwise be idempotent apart from solver noise.

### Step 3 — iterative hierarchical equilibrium, not implemented

A future mode may repeatedly optimize every affected node using the synthetic
series constructed at the preceding iteration:

```text
W^(k+1) = T(W^k)
residual_k = max_node ||w_node^(k+1) - w_node^k||
```

This step changes the result only if a genuine ancestor-to-descendant feedback
exists in addition to the existing child-to-parent composition. Candidate feedback
channels include:

- child constraints measured against the current optimized father or root;
- risk, capital or leverage budgets assigned by an ancestor;
- conditional financing terms determined by aggregate exposure;
- moment estimates or views conditional on the current hierarchy.

If no such feedback exists, further iterations add no economic information and
must not be presented as a new optimizer.

The iterative mode is outside the current public V2 contract and is not part of
this remediation merge gate. Before implementation it requires a separate schema
and a methodological specification covering:

- the dependency graph and admissible cycles;
- synchronous versus ordered node updates;
- initialization, deterministic tie-breaking and reproducibility;
- convergence tolerance and maximum iterations;
- optional damping or relaxation;
- oscillation, cycle and financing-regime-switch detection;
- fail-loud non-convergence;
- complete iteration-level audits and residual histories.

The research phase must not assume that a fixed point exists, is unique, or is
independent of initialization and update order. Those properties must be proved
under explicit assumptions or measured empirically.

### Step 4 — auditable LLM-supplied views, final research objective

The intended end state is to feed portfolio optimizers with node-scoped views derived
by an LLM while keeping the numerical path deterministic, typed and fully auditable.
The LLM must never produce final weights, alter constraints or directly select a
financing regime. It converts a causally available source bundle into proposed views;
the canonical parser validates them and Black-Litterman or another explicitly declared
view model translates accepted views into moments before the existing optimizer runs.

The minimum view artifact includes:

- source snapshot identifiers, timestamps and decision-date cutoff;
- retrieval query or source-selection rule;
- prompt template, model identifier, model version and generation parameters;
- raw response identifier or immutable hash;
- parsed node, instruments, coefficients, horizon, expected return and confidence;
- validation result, rejection reason and all normalization applied;
- prior and posterior moments and the configured covariance-role policy.

The audit and study must decompose the complete chain rather than attribute the final
result generically to the LLM:

- data and estimator contribution before any view;
- LLM-to-typed-view transformation;
- prior-to-posterior moment contribution;
- posterior-moment-to-local-weight contribution;
- local-to-global hierarchy propagation;
- cash, leverage and borrowing-spread contribution;
- per-iteration contribution if step 3 is active.

Required counterfactuals use identical data, constraints and solver settings with views
disabled and enabled. Additional ablations should vary source sets, prompt templates,
confidence calibration, node placement and deterministic/manual view baselines. Every
rejected or missing view, parser failure and model failure must be counted. Walk-forward
runs must prevent future information from entering retrieval, prompts or source bundles.

This LLM layer is outside the current public V2 contract and remediation merge gate.
Its objective is not to assume predictive superiority, but to make the incremental
effect of LLM-generated information exactly replayable and separable from every later
optimization, hierarchy, financing and iterative step.

## Validation contract

The canonical parser and direct solver fail loudly for:

- non-finite or non-positive risk aversion;
- non-finite risk-free rate, leverage, spread, view payloads or `view_tau`;
- leverage below one or negative spread;
- contradictory cash aliases or a spread without an active financing contract;
- inconsistent risky bounds and risk limits;
- unsupported objectives, estimators, covariance policies or turnover controls;
- Black-Litterman views on financing instruments;
- HRP combined with financing.

The parser no longer rejects financing on non-root nodes. Existing root-only JSON
configurations preserve their public financing names and economics.

## Backtest and ledger invariants

- All arms share one causal OOS schedule.
- Sleeve targets use synthetic returns already inclusive of local financing.
- Qualified financing labels prevent collisions across nodes.
- Lending and borrowing accrue at the effective local rates exactly once.
- Drift, turnover and transaction costs operate on the flattened terminal target.
- Terminal targets satisfy the same net-investment identity as the hierarchy.
- Arm metrics use the risk-free rate effective for that node or portfolio.

## Verification gate

The current head is complete only when it passes:

- Python 3.11, 3.12 and 3.13 test matrix;
- existing 95% coverage floor without exclusions or a reduced threshold;
- Ruff;
- strict Mypy without V2 escape hatches;
- dependency-boundary validation;
- all-extras integration tests;
- architecture tests proving transitional patch modules remain absent;
- financing regressions for leaf, intermediate and root nodes, all hierarchy
  modes, reference constraints, independent regimes, audits and ledger accounting;
- Tree Studio financing controls, serialization and HRP compatibility tests;
- complete embedded JavaScript syntax validation through `node --check`.

## Engine-adjacent work before scientific study

Tree Studio now exposes the same local cash, leverage, risk-free and spread
contract without inventing a second numerical path. Private Market Data Hub model,
backtest and export validators remain environment-dependent checks. The PR stays
draft until final-head CI, live validators and human numerical review are recorded.

## Scientific study phase

No financial-superiority claim follows from the engineering suite. After engine
sign-off, the study must use common causal OOS dates, explicit baselines and
ablations, cost and universe sensitivity, paired uncertainty estimates,
multiple-testing controls, immutable configurations and complete failure counts.

The hierarchy study should distinguish at least:

- independent flat optimization;
- direct bottom-up hierarchical optimization;
- the implemented forward plus backward attribution mode;
- the future iterative mode, only after its contract is implemented;
- ablations without local financing, with root-only financing and with financing
  enabled per node;
- no-view, manual-view and validated LLM-view counterfactuals using identical numerical
  configurations;
- attribution of source, parsing, posterior moments, local weights, hierarchy,
  financing and any iterative updates.

The primary questions for step 3 are convergence, stability and hierarchical
constraint consistency, not an assumed performance improvement. Report iteration
counts, residuals, non-convergence, initialization sensitivity, update-order
sensitivity and all financing-regime transitions.

The primary questions for step 4 are causal validity, confidence calibration,
reproducibility and incremental attribution. Report accepted and rejected views,
source and prompt versions, prior/posterior moment changes, local and final weight
deltas, risk and performance deltas, and sensitivity to source, prompt, confidence and
node placement. LLM output must never be conflated with optimizer output.

## Merge gate

The PR can leave draft only when every final-head CI job passes, documentation
matches runtime behavior, financing and estimator provenance are exported, no
control is silently ignored, live validators and human review are recorded, and
scientific claims remain separately supported.
