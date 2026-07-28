# Optimizer V2 clean-engine follow-up

## Status and purpose

**2026-07-22: implemented, on branch `agent/optimizer-methodology-remediation` (PR
#37).** This document was the spec implemented against; it is retained as the
governing methodology record. See `docs/optimizer-v2-remediation-status.md` for the
phase-by-phase implementation log (contracts → hierarchy resolver → moments/solver
→ flat independence → backtest/audit/export → Tree Studio → scientific harness),
including two residual limitations not yet closed: a pre-existing (not introduced by
this work) invariant gap in `project/tree_studio_v2/validate_example_estimates.py`'s
synthetic-series reconstruction check for a forward-mode node with children, and a
still-missing fixed-mean-estimator scientific-study ablation arm (only a covariance-
estimator ablation exists today, via the three `*_MIN_VARIANCE` baselines).

This document defines the next implementation cycle for the canonical V2 hierarchical optimizer after PR #37.

The primary objective is not to patch the current implementation. The work must preserve one clean numerical engine under `lazyfin.optimization.v2`, update its public contracts explicitly, and only afterwards adapt Tree Studio, exports, documentation and scientific tooling to the same contract.

The target behavior is:

1. **Forward** is a proxy-based diagnostic counterfactual.
2. **Backward** is the final non-iterative bottom-up allocation.
3. In standard backward mode, child candidates are replaced by their optimized synthetic series, while father and benchmark risk references remain raw external series.
4. A future iterative mode may allow current synthetic ancestors as references, but that is a separate contract and must never be inferred from `forward_backward`.

This plan also resolves the remaining audit findings concerning expected-return references, constraint fallback priority, root-relative references, GUI round-trip fidelity and scientific claims.

---

## 1. Canonical methodology

### 1.1 Forward is diagnostic, backward is final

For a hierarchy such as:

```text
B0: 70% ACWI / 30% AGG

Root
├── Equity sleeve, proxy ACWI
│   ├── MSCI World
│   └── Emerging sleeve, proxy MSCI Emerging
│       ├── China
│       └── Emerging ex-China
└── Fixed-income sleeve, proxy AGG
    ├── IG Developed
    └── HY Developed
```

Forward solves every parent using raw child proxies:

```text
Root:            ACWI + AGG
Equity:          MSCI World + MSCI Emerging, versus raw ACWI when requested
Emerging:        China + Emerging ex-China, versus raw MSCI Emerging when requested
Fixed income:    IG Developed + HY Developed, versus raw AGG when requested
```

The forward result is retained for diagnostics and attribution only.

Backward starts at the lowest sleeves, builds their synthetic returns and resolves ancestors bottom-up:

```text
Emerging* = w_China R_China + w_ExChina R_ExChina
AGG*      = w_IG R_IG + w_HY R_HY

Equity solve candidates:
    MSCI World + Emerging*
Risk father reference, if requested:
    raw ACWI

ACWI* = w_World R_World + w_Emerging R_Emerging*

Root solve candidates:
    ACWI* + AGG*
Risk benchmark reference, if requested:
    raw B0 = 70% raw ACWI + 30% raw AGG
```

Final terminal weights are recursively composed from the backward local weights.

When descendants do not consume optimized ancestor state, the final `forward_backward` result must equal a direct bottom-up-only solve within numerical tolerance. The forward pass is therefore not required to improve the final optimum.

### 1.2 Identity is not representation

Each economic component needs a stable logical identity, but raw and synthetic series must remain distinct.

Example:

```text
component_id: child:emerging
raw proxy:   ticker:MSCI_EMERGING
synthetic:   internal:child:emerging:SYNTH
```

The active candidate representation depends on the pass:

```text
Forward candidate representation:   raw proxy
Backward candidate representation:  child synthetic series
Iterative candidate representation: current synthetic series
```

A common `component_id` exists only to keep constraints, strategic weights, views and audit records stable across representations. It must never imply that raw proxy and synthetic series are interchangeable for every role.

### 1.3 Candidate series and reference series are separate roles

The engine must resolve series by role, not by a generic alias lookup.

| Role | Forward | Backward | Future iterative mode |
|---|---|---|---|
| Direct instrument candidate | Raw instrument | Raw instrument | Raw instrument |
| Child candidate | Raw child proxy | Child synthetic | Current child synthetic |
| Father TEV/volatility reference | Raw father proxy | Raw father proxy | Explicit policy only |
| Global benchmark reference | Raw benchmark | Raw benchmark | Normally raw benchmark |
| Mean-reference weights | Applied to active candidates | Applied to active candidates | Applied to current candidates |

The non-iterative backward mode must therefore satisfy this invariant:

```text
child candidate = synthetic
father risk reference = raw proxy
benchmark risk reference = raw B0
```

Using the synthetic father as a backward risk reference would introduce an ancestor/descendant feedback dependency and belongs only to an explicitly iterative mode.

### 1.4 Raw references are external anchors

Father and benchmark risk references are external economic anchors. They are not candidate assets, fallback allocations or synthetic replacements.

For example, in the backward Equity solve:

```text
candidate return = w_World R_World + w_EM R_Emerging*
father TEV = std(candidate return - raw R_ACWI)
```

The raw ACWI series is not inserted into the candidate universe unless it is also explicitly declared as a direct instrument.

---

## 2. Separate benchmark, risk references and mean references

### 2.1 Current problem

The current engine derives `reference_weights` for the expected-return estimator from the first available risk reference:

```text
volatility target weights
or volatility cap weights
or tracking-error weights
```

This couples expected-return estimation to risk-control configuration. It can also make `mean_estimator="auto"` resolve differently between forward and backward when raw proxy names are replaced by synthetic candidate columns.

At the root, for example:

```text
Forward candidates:  ACWI, AGG
Backward candidates: ACWI_SYNTH, AGG_SYNTH
Raw B0 weights:       ACWI 70%, AGG 30%
```

The forward solve can see a complete equilibrium reference while the backward solve may not, even though the intended strategic allocation is unchanged.

### 2.2 Required separation

The contracts must distinguish:

```text
benchmark_reference
    External reporting and comparison benchmark.

risk_reference
    Series used by TEV, volatility targets and volatility caps.

mean_reference
    Strategic weights used to construct equilibrium expected returns.
```

Risk-reference configuration must not implicitly select the mean-reference portfolio.

### 2.3 Mean reference must support sleeves absent from the benchmark

A node may optimize components that do not appear in B0.

Example:

```text
Root candidates:
    Equity sleeve
    Bond sleeve
    Gold sleeve

B0:
    70% ACWI
    30% AGG
```

The raw B0 series remains valid for TEV:

```text
TEV(R_root, 0.70 R_ACWI + 0.30 R_AGG)
```

No instrument-by-instrument overlap is required to compute TEV. Only complete aligned return series are required.

Equilibrium expected returns, however, require a complete strategic weight vector over the actual solved components. Therefore a separate local mean reference may be declared:

```json
{
  "mean_estimator": "equilibrium",
  "mean_reference": {
    "kind": "local_weights",
    "weights": {
      "child:equity": 0.60,
      "child:bonds": 0.30,
      "child:gold": 0.10
    }
  }
}
```

These weights are applied to the active candidate representation:

```text
Forward:
    child:equity -> raw ACWI proxy
    child:bonds  -> raw AGG proxy
    child:gold   -> raw Gold proxy

Backward:
    child:equity -> Equity synthetic
    child:bonds  -> Bonds synthetic
    child:gold   -> Gold synthetic
```

The father and benchmark risk references remain raw.

### 2.4 No invented weights

The engine must never silently infer strategic weights for components absent from the benchmark.

Disallowed implicit behavior:

```text
component absent from benchmark -> zero strategic weight
benchmark-covered components -> silently renormalized
first available risk reference -> mean reference
```

Optional benchmark projection may be supported only through an explicit complete mapping and explicit residual allocation policy.

Example:

```json
{
  "mean_reference": {
    "kind": "benchmark_projection",
    "mapping": {
      "ticker:ACWI": "child:equity",
      "ticker:AGG": "child:bonds"
    },
    "additional_weights": {
      "child:gold": 0.05
    },
    "rescale_mapped_benchmark_to": 0.95
  }
}
```

If the resulting weights do not cover every solved component and sum to one, validation must fail.

### 2.5 `auto` resolution

The recommended rule is:

```text
if mean_estimator == auto and a complete mean_reference exists:
    resolve to equilibrium
else if mean_estimator == auto:
    resolve to bayes_stein
```

`auto` must not inspect TEV, volatility-target or volatility-cap references.

For `mean_estimator="equilibrium"`, an incomplete mean reference must fail loudly.

---

## 3. Constraint semantics and priority

### 3.1 Hard constraints are simultaneous

Hard constraints are not sequentially optimized. They must all hold at the accepted solution.

Recommended defaults:

```text
minimum/maximum weights: hard
full-investment or financing budget: hard
maximum leverage: hard
volatility cap: hard
TEV maximum: explicit policy, preferably hard by default
exact volatility target: exact attempt with explicit nearest-feasible policy
```

If TEV and a hard volatility cap are both configured, the accepted solution must satisfy both or fail.

### 3.2 Priority applies only to fallback

When one or more constraints are configured as `nearest_feasible`, fallback must be lexicographic rather than a weighted sum of normalized violations.

Required priority:

```text
1. Minimize TEV-limit excess, if TEV is present and relaxable.
2. Subject to the minimum TEV excess, minimize volatility violation or exact-target deviation.
3. Subject to both minimum violations, optimize the economic objective.
```

This prevents a smaller volatility deviation from compensating for a larger TEV breach.

Suggested schema:

```json
{
  "constraint_priority": ["tracking_error", "volatility"],
  "tracking_error_policy": "hard_fail",
  "volatility_target_policy": "nearest_feasible",
  "volatility_cap_policy": "hard_fail"
}
```

Supported policies should be explicit and narrowly defined:

```text
hard_fail
nearest_feasible
```

A hard cap must never be relaxed merely because an exact target or TEV is infeasible.

### 3.3 Auditing fallback

Every local audit must record:

```text
configured policy for each constraint
whether fallback was entered
minimum TEV excess
minimum volatility target deviation or cap excess
final economic objective after violation levels were fixed
solver stages and messages
```

A nearest-feasible result must never be reported as satisfying the original limit.

---

## 4. Root-relative references and iterative mode boundary

### 4.1 Standard non-iterative contract

The standard `forward_backward` mode supports child-to-parent information flow only:

```text
child optimized series -> parent candidate
```

It must not contain a dependency such as:

```text
current optimized parent/root -> child constraint
```

A reference to the forward root may be retained only if it is explicitly named and documented as a frozen forward counterfactual, for example:

```text
forward_root_reference
```

A generic `root` label is ambiguous and should be deprecated.

### 4.2 Future iterative contract

References such as these belong to a separate future mode:

```text
current_parent_synthetic
current_root_synthetic
```

That mode must define initialization, update order, damping, tolerance, maximum iterations, cycle detection, financing-regime switches and full iteration auditability.

The standard backward resolver must reject iterative-only reference policies.

---

## 5. Clean engine design

### 5.1 No patch layer

Implementation must modify the canonical modules directly:

```text
src/lazyfin/optimization/v2/contracts.py
src/lazyfin/optimization/v2/validation.py
src/lazyfin/optimization/v2/model.py
src/lazyfin/optimization/v2/moments.py
src/lazyfin/optimization/v2/solver.py
src/lazyfin/optimization/v2/hierarchy.py
src/lazyfin/optimization/v2/backtest.py
src/lazyfin/optimization/v2/api.py
```

Do not introduce:

```text
monkey patches
import-time mutation
wrapper finalizers
parallel solver paths
legacy compatibility engines with different numerical behavior
post-solve weight substitution
```

Compatibility must be implemented through explicit parsing and migration into the canonical typed contract.

### 5.2 Proposed typed objects

The exact naming may change, but the engine should have equivalent concepts.

```python
@dataclass(frozen=True)
class V2Component:
    id: str
    kind: Literal["direct", "child"]
    raw_series: str
    child_id: str | None = None


@dataclass(frozen=True)
class V2MeanReference:
    kind: Literal["none", "local_weights", "benchmark_projection"]
    weights: dict[str, float]
    mapping: dict[str, str]
    additional_weights: dict[str, float]
    rescale_mapped_benchmark_to: float | None


ReferencePolicy = Literal[
    "none",
    "manual",
    "father_raw_proxy",
    "benchmark_raw",
    "forward_root_reference",
    "current_parent_synthetic",
    "current_root_synthetic",
]


ConstraintPolicy = Literal["hard_fail", "nearest_feasible"]
```

Iterative-only values may exist in the schema before the iterative engine exists, but validation must reject them in flat, forward and forward-backward modes.

### 5.3 Pass-specific resolution context

The hierarchy should build an explicit context rather than relying on column-name suffixes.

```python
@dataclass
class V2SolveContext:
    pass_kind: Literal["forward", "backward", "iterative"]
    candidate_series_by_component: dict[str, Series]
    raw_proxy_series_by_component: dict[str, Series]
    synthetic_series_by_component: dict[str, Series]
    component_to_solver_column: dict[str, str]
    solver_column_to_component: dict[str, str]
```

Dedicated resolvers should exist:

```python
resolve_candidate_frame(node, context)
resolve_risk_reference(node, reference_policy, context)
resolve_mean_reference_weights(node, mean_reference, context)
```

`resolve_risk_reference` and `resolve_mean_reference_weights` must not call each other or share an implicit fallback chain.

---

## 6. Step-by-step engine implementation

### Step 0 — Freeze existing intended behavior

Before refactoring, add characterization tests for:

1. standard forward proxy solves;
2. backward synthetic-child substitution;
3. raw father reference in backward;
4. raw B0 reference in backward;
5. recursive final-weight composition;
6. direct bottom-up equivalence when no ancestor-to-descendant dependency exists;
7. local cash and leverage propagation.

These tests define the accepted behavior, not every incidental implementation detail.

### Step 1 — Add stable component identities

Modify model construction so that every direct instrument and child edge receives a stable component ID.

Recommended conventions:

```text
direct instrument: ticker:ACWI
child edge:        child:<node-id>
```

Validate uniqueness within each node. Preserve the raw proxy as separate metadata on child components.

Do not change numerical behavior in this step.

### Step 2 — Introduce pass-specific candidate resolution

Refactor hierarchy frame construction to use component IDs and a pass context.

Forward:

```text
direct component -> raw instrument
child component  -> raw child proxy
```

Backward:

```text
direct component -> raw instrument
child component  -> resolved child synthetic series
```

Continue passing solver-column aliases for per-component bounds, but do not use those aliases as a generic reference-resolution mechanism.

### Step 3 — Make risk references role-specific

Replace ambiguous reference values with explicit raw policies:

```text
father -> father_raw_proxy
benchmark -> benchmark_raw
root -> forward_root_reference, only where intentionally supported
```

Backward `father_raw_proxy` must always resolve from the immutable raw returns matrix using `node.proxy`.

Add tests that deliberately create a synthetic series very different from the proxy and prove that TEV/volatility is still calculated against the raw father.

### Step 4 — Add independent mean references

Add `mean_reference` to the canonical constraint or node contract.

Implement complete validation:

```text
all referenced component IDs exist
all weights are finite
weights are compatible with long-only policy
all solved components are explicitly covered
sum equals one within tolerance
projection mappings are one-to-one unless explicitly supported otherwise
no invented residual weights
```

Remove the current `target_weights or cap_weights or tracking_weights` coupling.

The moments layer must receive only the resolved, solver-column-aligned mean-reference weights.

### Step 5 — Resolve mean weights across representations

Map strategic component weights to the active solver columns:

```text
child:equity 0.60 -> ACWI in Forward
child:equity 0.60 -> internal:child:equity:SYNTH in Backward
```

Add regression tests for:

1. sleeve absent from B0;
2. complete local mean reference including the extra sleeve;
3. identical resolved estimator in Forward and Backward;
4. explicit equilibrium succeeding in Backward;
5. no change to raw father or raw B0 risk references.

### Step 6 — Change `auto` resolution

Implement:

```text
auto + complete mean reference -> equilibrium
auto + no complete mean reference -> bayes_stein
```

Retain explicit special handling for financing and any objective-specific policy only if it is methodologically intended and fully audited.

Audit the reason and reference source.

### Step 7 — Implement lexicographic fallback

Refactor the local solver into explicit stages.

Suggested approach:

```text
Stage A: solve all hard constraints and exact requests.

If accepted:
    optimize the economic objective and return.

If fallback is allowed:
    Stage B1: minimize TEV excess.
    Stage B2: constrain TEV excess to its minimum plus tolerance;
              minimize volatility deviation/excess.
    Stage B3: constrain both violations to their minima plus tolerance;
              optimize the economic objective.
```

Do not encode priority through arbitrary penalty coefficients.

Hard volatility caps and hard TEV limits remain in every fallback stage.

### Step 8 — Clarify root-reference behavior

Rename or migrate the current root-relative policy so the frozen dependency is explicit.

Recommended immediate behavior:

```text
forward_root_reference:
    valid only on descendants;
    resolves to the frozen forward-root synthetic return;
    clearly audited as a forward reference.

current_root_synthetic:
    rejected outside future iterative mode.
```

Do not silently reinterpret old `root` configurations. Migrate with a warning and documented compatibility rule, or fail if the intended meaning cannot be determined.

### Step 9 — Extend audits and exports

Add fields equivalent to:

```text
component IDs and active solver columns
pass kind
candidate representation per component
risk-reference policy and resolved raw series
mean-reference kind, source and component weights
resolved mean-reference solver weights
mean-reference coverage
configured and resolved mean estimator
constraint policies and priority
lexicographic fallback stage results
forward-root versus final-root reference provenance
```

The Audit ZIP must contain enough information to replay each local solve exactly.

### Step 10 — Backtest integration

Ensure every fold builds all references only from the training window and uses the same pass-specific resolver as point estimation.

The fold audit must preserve both component identity and actual series names so that forward and backward can be compared without confusing representation changes with identity changes.

---

## 7. Tree Studio adaptation after engine completion

Tree Studio must be adapted only after the canonical engine contracts and tests are complete.

### 7.1 Lossless configuration round-trip

The current form must not reconstruct a partial `constraints` object and discard fields it does not expose.

Required behavior:

```text
import JSON
edit one visible field
export JSON
all unrelated supported fields remain unchanged
```

Use schema-backed state or merge edited values into the existing canonical object.

Explicit zero values must remain zero and must not be converted into inheritance through JavaScript truthiness.

### 7.2 Correct terminology

UI labels and help text must state:

```text
Backward candidate child: synthetic sleeve
Backward father risk reference: raw father proxy
Backward benchmark risk reference: raw B0
Synthetic father/root reference: iterative-only future option
```

Remove any wording that says Father automatically becomes synthetic in standard backward mode.

### 7.3 Expose independent mean-reference controls

At node level, expose:

```text
mean estimator
mean-reference kind
local strategic weights by component
optional explicit benchmark projection
coverage and sum validation
```

The UI should display stable component names, not internal `_SYNTH` column names. A technical preview may show how those components resolve in Forward and Backward.

### 7.4 Expose constraint policies

Expose separately:

```text
TEV limit and hard/nearest-feasible policy
volatility target and hard/nearest-feasible policy
volatility cap and hard/nearest-feasible policy
priority preview: TEV, then volatility, then objective
```

Disable contradictory combinations and show the actual serialized policy.

### 7.5 GUI tests

Add browser-independent state/serialization tests for:

1. full supported contract round-trip;
2. preservation of explicit zero values;
3. preservation of views and covariance fields not currently edited;
4. local mean-reference editing;
5. sleeve absent from benchmark;
6. correct raw/synthetic labels;
7. no iterative reference in standard mode;
8. imported legacy reference migration.

String-presence tests are not sufficient.

---

## 8. Scientific harness follow-up

The scientific harness is useful as an engineering comparison, but claims must match the tested contrasts.

Required changes:

1. state clearly whether baselines use their native constraints or a harmonized mandate;
2. distinguish strategy comparison from estimator/solver ablation;
3. run inference for every baseline claimed in the documentation, or narrow the claims;
4. report proxy-to-synthetic attribution separately from any expected-return-estimator change;
5. add an arm with an explicitly fixed statistical mean estimator when studying hierarchy effects;
6. add a direct bottom-up arm and verify numerical equivalence with final backward when assumptions hold;
7. preserve raw reference, mean-reference and representation provenance for every fold.

No engineering gate should be presented as evidence of financial superiority.

---

## 9. Required regression matrix

### Hierarchy and references

- Forward uses raw child proxies.
- Backward uses child synthetic candidates.
- Backward father TEV uses raw father proxy.
- Backward father volatility target/cap uses raw father proxy.
- Backward root benchmark uses raw B0.
- Standard backward rejects current synthetic father/root policies.
- Direct bottom-up equals final backward within tolerance when dependencies are child-to-parent only.

### Mean estimation

- `auto` does not inspect risk references.
- `auto` plus complete local mean reference resolves to equilibrium.
- `auto` without mean reference resolves to Bayes-Stein.
- Explicit equilibrium works with synthetic child candidates.
- Sleeve absent from B0 works with a complete local mean reference.
- Incomplete local mean reference fails loudly.
- No benchmark residual weight is invented.
- Forward and backward use the same configured/resolved mean estimator when intended.

### Constraints

- Hard TEV and hard volatility cap are simultaneous.
- TEV fallback has priority over volatility fallback.
- Economic objective is optimized only after minimum violations are fixed.
- A nearest-feasible result is explicitly marked.
- Hard caps are never relaxed by a soft target fallback.

### GUI

- Full lossless JSON round-trip.
- Explicit zeros preserved.
- Raw/synthetic roles shown correctly.
- Mean-reference weights use component IDs.
- Unsupported iterative policies cannot be selected in standard mode.

### Audit and backtest

- Every resolved series is recorded.
- Every component-to-column mapping is recorded.
- Point estimate and walk-forward use the same resolver.
- No look-ahead in reference or synthetic-series construction.

---

## 10. Delivery sequence

Use small, reviewable commits in this order:

```text
1. Characterization tests and accepted invariants.
2. Component identity contract and model validation.
3. Pass-specific candidate and risk-reference resolver.
4. Independent mean-reference contract and moment integration.
5. Forward/backward estimator consistency tests.
6. Lexicographic constraint fallback.
7. Root-reference naming and iterative-mode validation boundary.
8. Audit, backtest and export provenance.
9. Tree Studio lossless state and serialization.
10. Tree Studio reference, mean and policy controls.
11. Scientific harness and documentation claim alignment.
12. Final architecture and dependency-boundary review.
```

Do not start GUI changes before steps 1–8 are green.

---

## 11. Merge acceptance criteria

The follow-up implementation is acceptable only when all of the following hold:

1. There is one canonical numerical engine and no patch/finalizer layer.
2. Standard backward replaces child candidates with synthetic series but keeps father and B0 risk references raw.
3. Component identity does not erase representation or reference-role distinctions.
4. Sleeves absent from B0 can be optimized with TEV versus raw B0 and a separate complete local mean reference.
5. Expected-return estimation is independent of TEV and volatility-reference selection.
6. Explicit equilibrium works in both Forward and Backward when a complete mean reference is declared.
7. Constraint fallback is lexicographic with TEV before volatility.
8. Hard constraints never become soft without an explicit policy.
9. Iterative synthetic-ancestor references are rejected outside a separately named iterative mode.
10. Tree Studio round-trips the complete supported contract without losing fields or explicit zeros.
11. GUI terminology matches the engine's raw/synthetic semantics.
12. Audit artifacts can replay every local solve.
13. Scientific claims match the comparisons and inference actually performed.
14. CI passes on all supported Python versions, strict typing, lint, architecture boundaries and GUI serialization tests.
