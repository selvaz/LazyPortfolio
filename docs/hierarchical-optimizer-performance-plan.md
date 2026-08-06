# Hierarchical Optimizer Performance Plan

## Purpose

Make multi-level V2 trees fast enough for daily production while preserving
their economic and audit contracts.

The primary production case is a multi-level `max_return` tree in which a
node can have:

- long-only local weights, budget and per-asset bounds;
- an exact volatility target derived from its immutable father proxy;
- an optional tracking-error (TEV) upper bound relative to that same father;
- Forward and Forward-plus-Backward hierarchical composition.

The plan must not silently turn an exact volatility target into a cap, replace
the immutable father reference, weaken TEV, or return an unaudited approximate
allocation.

## Current bottleneck

For each local solve, the current V2 SLSQP path can evaluate:

- one structured initial point;
- two boundary initial points;
- eight deterministic random restarts;
- up to 2,000 iterations for each restart.

`forward_backward` solves every node once in Forward and once in Backward.
Consequently, a three-node tree can invoke up to 66 local SLSQP runs per fold.
A monthly ten-year out-of-sample backtest, after its training period, can
therefore create several thousand local solves.

The report pipeline adds avoidable cost because a normal client report builds
the full audit ZIP and captures every fold's estimation series.

## Target architecture

Use a solver router, not a single replacement solver.

```text
validated node problem
        |
        +-- linear max-return ----------------> analytical / LP route
        |
        +-- convex QP ------------------------> OSQP route
        |
        +-- max-return + vol-cap + TEV ------- > Clarabel SOCP route
        |
        +-- exact target-volatility ----------> SOCP candidate, then audited SLSQP only if needed
        |
        +-- max-ratio / non-convex fallback --> audited multi-start SLSQP
        |
        +-- HRP ------------------------------> existing Skfolio HRP route
```

Every route must emit the same canonical weights and a `V2Audit` record. The
audit must identify the route, solver status, tolerance, feasibility checks,
solve duration, and any SLSQP fallback.

## Step 1 — Establish a reproducible performance baseline

1. Select three representative saved trees:
   - a small two-level 70/30 tree;
   - a medium multi-asset tree with father target-volatility and TEV;
   - a larger research tree with multiple sleeves.
2. Run each in `forward`, `forward_backward`, point-estimate, and monthly
   walk-forward modes.
3. Record wall-clock time, number of folds, local solves, SLSQP calls,
   iterations, node count, instruments per node, data fingerprint, and peak
   memory.
4. Store benchmark snapshots in a versioned test fixture or benchmark artifact.
5. Add a non-flaky regression threshold for the number of local solves; keep
   wall-clock results informational because they depend on hardware.

**Exit condition:** a benchmark report identifies time spent in data loading,
moment estimation, solver calls, ledger replay, CSV generation, and ZIP
compression.

## Step 2 — Add solver classification and audit fields

1. Introduce an internal `V2ProblemClass` derived from the already validated
   node contract:
   - objective;
   - exact target versus volatility cap;
   - TEV presence;
   - financing regime;
   - views;
   - bounds and asset count.
2. Add a solver-router entry point inside `V2LocalOptimizer`; keep the public
   V2 configuration schema unchanged initially.
3. Extend `V2Audit` with:
   - `solver_route`;
   - `solver_status`;
   - `solve_seconds`;
   - `warm_started`;
   - `fallback_reason`.
4. Preserve `solver_strategy` for compatibility, but make it describe the
   selected route precisely.
5. Add unit tests proving that unsupported/problematic combinations still fail
   loudly rather than being routed to a weaker formulation.

**Exit condition:** every local solve is classified deterministically and its
route can be reconstructed from the audit export.

## Step 3 — Implement exact fast paths with no methodology change

### 3.1 Linear max-return

Use an analytical bounded-allocation algorithm, or SciPy HiGHS linear
programming, only when the node has:

- `objective=max_return`;
- no volatility target or cap;
- no TEV constraint;
- no financing, or a separately proven linear financing formulation;
- only budget and box constraints.

Allocate remaining capital in descending expected-return order after applying
minimum weights. Validate the budget and bounds independently.

### 3.2 Convex QP routes

Add an optional OSQP-backed route for:

- minimum variance with linear constraints;
- mean-variance utility with linear constraints.

Use the same covariance and expected-return estimators as the current engine.
Compare weights, objective values, and constraint slacks against the current
SLSQP implementation on all deterministic test fixtures.

**Exit condition:** the fast-path outputs are equivalent to SLSQP within
declared numerical tolerances and do not alter any existing test result.

## Step 4 — Add Clarabel for father-relative max-return constraints

Formulate the common convex relaxation as a second-order cone problem (SOCP):

```text
maximize        expected_return' * w
subject to      sum(w) = 1
                lower <= w <= upper
                volatility(w) <= target_volatility_from_father
                TEV(w, father_returns) <= maximum_tracking_error
```

Implementation requirements:

1. Build volatility and TEV from the same sample data and annualization as the
   current V2 solver.
2. Treat the father proxy as an immutable reference series, never as a
   candidate allocation unless it was already a declared candidate.
3. Use an optional CVXPY + Clarabel dependency behind a lazy import so the
   package remains testable without the production solver extra.
4. Record primal feasibility and post-solve independent checks in `V2Audit`.
5. Use the SOCP result only if it satisfies all canonical tolerance checks.

**Important:** this route solves `volatility <= target`, not exact equality.
It is globally optimal for the cap formulation and is a high-quality candidate
for the exact-target route in Step 5.

**Exit condition:** SOCP matches or improves feasibility against SLSQP for all
cap-volatility + TEV fixtures, with an audited global-optimum claim limited to
the convex formulation.

## Step 5 — Preserve exact father target-volatility

Exact volatility equality is non-convex. Do not silently replace it with a
volatility cap.

For `volatility_target_mode=exact`:

1. Solve the Step 4 SOCP cap problem first.
2. If the SOCP optimum reaches the volatility target within the existing
   equality tolerance, accept it. It is also a valid solution to the exact
   target problem.
3. If it remains below target, use that feasible SOCP allocation as the first
   SLSQP initial point for the exact equality problem.
4. Run a small, explicit, deterministic fallback restart set only when needed.
5. Retain the current lexicographic infeasibility policy for TEV and target
   volatility; every fallback stage remains independently recorded.
6. Compare the hybrid result to the current full multi-start SLSQP result over
   historical folds before enabling the route by default.

The initial production policy should prefer correctness over speed: fall back
to the existing SLSQP policy whenever the hybrid candidate fails a strict
equivalence or feasibility test.

**Exit condition:** the hybrid route either matches the established solution
within tolerances or transparently uses the existing audited fallback.

## Step 6 — Add analytic derivatives to the retained SLSQP path

For paths that remain on SLSQP, provide analytical gradients for:

- expected-return objective;
- variance and volatility;
- mean-variance utility;
- TEV;
- volatility-cap and target-equality constraints.

Use finite-difference derivatives only as a controlled fallback in tests.
Benchmark iteration counts and function evaluations before/after. This step is
independent of the solver-router work and benefits max-ratio and exact-target
fallbacks.

**Exit condition:** retained SLSQP routes have materially fewer function
evaluations without a feasibility or reproducibility regression.

## Step 7 — Parallelize the independent portion of backtests

Separate each fold into two phases:

```text
parallel phase:    training window -> hierarchical estimate -> fold targets/audits
sequential phase:  ordered fold targets -> daily ledger drift, turnover, costs
```

1. Compute fold estimates in a bounded `ProcessPoolExecutor`.
2. Keep process count configurable and default it conservatively to prevent
   BLAS/OpenMP oversubscription.
3. Sort completed estimates by signal date before the ledger replay.
4. Use immutable inputs and deterministic seeds so parallel and sequential
   executions produce the same fold targets and final curves.
5. Do not parallelize dependent Forward/Backward nodes within a fold until a
   dependency graph proves that a given subtree is independent.
6. Add a test comparing one-worker and multi-worker runs exactly within the
   existing numerical tolerance.

**Exit condition:** fold estimates scale across available CPU cores while
ledger outcomes remain identical and deterministic.

## Step 8 — Add a versioned fold and production-run store

Create a dedicated SQLite/DuckDB run store, separate from MarketDataHub's raw
market-data database, with at least:

- `tree_id`, `config_hash`, optimizer code version and solver profile;
- data fingerprint/as-of timestamp;
- signal date and training-window hash;
- weights, audit summary, metrics, duration and artifact references;
- run status and error details.

Cache reuse rules:

1. A fold can be reused only when its model hash and complete input-window hash
   match exactly.
2. A new market-data refresh invalidates only folds whose input windows changed.
3. Corporate-action or historical data revisions must invalidate every affected
   fold, not only the last one.
4. A cached report must be keyed by both configuration and data fingerprint.

**Exit condition:** repeated research runs reuse unchanged folds safely, and
daily production never presents a stale allocation as current.

## Step 9 — Split daily production from research artifacts

Create separate commands/jobs:

| Job | Frequency | Output | Must not do |
|---|---|---|---|
| Daily allocation | daily after data refresh | current weights, changes, node audit summary, compact HTML/JSON | full ten-year backtest, scientific bootstrap, audit ZIP |
| Research backtest | manual or weekly | fold ledger, all OOS metrics, solver comparison | daily notification path |
| Scientific study | manual or monthly | baselines, ablations, bootstrap inference | routine production execution |
| Audit export | on demand | full reconstruction ZIP | client-report-only request |

Change the Tree Studio endpoint flow so a client HTML report never constructs
an audit ZIP. Capture full fold estimation series only for the explicit audit
export path.

**Exit condition:** a daily report performs one current estimate per tree and
an HTML report request never triggers unnecessary ZIP creation.

## Step 10 — Validation and rollout

1. Run SLSQP and hybrid routes side by side on representative trees and ten
   years of historical folds.
2. Compare per-node weights, target/cap volatility, TEV, feasibility, terminal
   weights, turnover, OOS curves, and runtime.
3. Define tolerances before examining results.
4. Enable each new route behind an explicit solver profile:
   - `research_strict`: current multi-start SLSQP behavior;
   - `hybrid_validated`: router with strict fallback;
   - `production_fast`: only after equivalence thresholds are met.
5. Keep the old route available until all selected trees pass the agreed
   historical comparison.
6. Publish a per-tree migration report showing solver-route usage and realized
   speed improvement.

## Order of implementation

1. Steps 1, 2 and 9: measure correctly and remove report-only waste.
2. Step 3: exact linear fast path.
3. Step 4: Clarabel SOCP cap-volatility + TEV path.
4. Step 5: hybrid exact-target strategy.
5. Step 6: analytic SLSQP gradients.
6. Step 7: parallel fold estimation.
7. Step 8: durable run/fold store.
8. Step 10: controlled production rollout.

This order delivers early speed gains without making the daily tree runner
dependent on an unvalidated optimizer migration.
