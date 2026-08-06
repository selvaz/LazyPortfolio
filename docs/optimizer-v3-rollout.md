# V2 optimizer v3 performance roadmap: what changed and what's proven

Status: **Phases 0, A, B, B.5, C landed** on `agent/optimizer-v3-performance`. This is a
living document — it gets updated as later phases (E, F, G's remaining validation) land,
not rewritten from scratch. See `docs/hierarchical-optimizer-performance-plan.md` for the
original diagnosis and the plan (kept in this session's working plan file) for phase-by-phase
detail.

**Read this section first if you only read one section.** The honest summary:

- **Correctness is proven.** Every new solver route (LP, QP, SOCP) has been checked against
  SLSQP on the exact same problem — not just in isolated unit tests, but on a real saved
  tree (`MS_7030_base_2_level`), where forcing every fast path off and running pure SLSQP
  with the Phase B.5 risk-measure fix reproduces the fast-path-enabled result to 4+ decimal
  places. Nothing about routing changes *what* gets solved for, only *how fast*.
- **Fast paths B1/B2 (LP/QP) have zero measured benefit on your real trees today.** Every
  node in your 3 saved real trees that has a volatility target uses the default
  `volatility_target_mode="exact"`, and B1/B2 only engage when *no* volatility constraint is
  present at all. This was known and expected going in — it's why Phase C exists.
- **C2 (the exact-target hybrid route) does engage on real trees** — verified live,
  node-by-node, on `MS_7030_base_2_level`: 2 of its 4 node solves used the SOCP warm start.
  This is the route that matters for your portfolios.
- **Wall-clock speedup is NOT reliably measured yet.** This environment showed a real,
  reproducible ~6.5-second `sklearn` import-time spike after installing `cvxpy`/`clarabel`/
  `osqp` (almost certainly antivirus re-scanning the newly-touched site-packages tree),
  which contaminated several timing comparisons during this work by 10-30x before it was
  caught and root-caused. Wall-clock numbers below are reported for transparency but flagged
  wherever this noise could be a factor — don't treat them as a clean before/after speedup
  claim yet.

---

## 1. What each phase actually does

| Phase | What it adds | Real-tree engagement (verified) |
|---|---|---|
| B1 (LP, HiGHS) | Exact route for `max_return`, no vol constraint, no financing | **0 of 30 nodes** across the 3 real trees |
| B2 (QP, OSQP) | Exact route for `min_risk`/`max_utility`, no vol constraint | **0 of 30 nodes** |
| C1 (SOCP, Clarabel) | Exact route for `max_return` + a volatility **cap** (not exact target) | **0 of 30 nodes** — no real node uses cap mode |
| C2 (SOCP warm start) | Solves the cap-relaxed problem first; if it lands exactly on the target, accepts it directly instead of running the expensive multi-SLSQP `cap_results` search | **Confirmed engaging**: 2 of 4 nodes on `MS_7030_base_2_level`'s flat-mode solve |

B1/B2/C1 are real, tested, and will engage the moment a node's shape matches their gate —
but as of this writing, none of your saved trees have that shape. C2 is the one that
matters today because your trees' father/benchmark-relative volatility targets all use
`volatility_target_mode="exact"` (the default), which C2 specifically targets.

## 2. Why B1/B2/C1 show 0% real-tree engagement

Checked live against the 3 real saved trees (`MS_7030_base_2_level`, `ACWI_AGG_70_30`,
`Global Multi-Asset` — 30 nodes total):

| Tree | Nodes | Nodes with an exact volatility target | Nodes with a volatility **cap** (not exact) |
|---|---|---|---|
| MS_7030_base_2_level | 3 | 3 | 0 |
| ACWI_AGG_70_30 | 14 | 13 | 0 |
| Global Multi-Asset | 13 | 13 | 0 |

Every constrained node uses the exact-equality target (the default), never the cap. This
is expected given how you build trees (target vol relative to a father/benchmark), not a
data quality problem. It's the reason B1/B2/C1 were built as the *foundation* for C2 rather
than as standalone deliverables — see the "Phase C merged" note in the plan for the full
reasoning.

## 3. Correctness verification (the part that matters most)

**Unit-level:** every route (`tests/test_v2_solver_lp_route.py`, `test_v2_solver_qp_route.py`,
`test_v2_solver_socp_route.py`, `test_v2_solver_risk_measure_consistency.py` — 42 tests
total) compares real solver output against SLSQP on the same problem, including under
Black-Litterman views with `view_covariance_policy="posterior_all"`. All pass.

**Real-tree level:** `MS_7030_base_2_level`, flat mode, checked two ways in the same run:

1. Full pipeline (all fast paths live): terminal weights
   `{VEA: 0.4692, EEM: 0.1097, CEMB: 0.4212}` (AGG drops to ~0).
2. Every fast-path entry point (`_solve_lp_max_return`, `_solve_qp_convex`,
   `_solve_socp_max_return_cap`, `_solve_socp_cap_weights`) monkeypatched off, forcing pure
   SLSQP with the Phase B.5 risk-measure fix still active: terminal weights
   `{VEA: 0.4692, EEM: 0.1097, CEMB: 0.4212}`.

Match to 4 decimal places. The routing layer changes nothing about the answer.

**Node-level engagement, same run** (`estimate.node_results[...].audit`):

| Node | `solver_strategy` | `warm_started` | `target_status` |
|---|---|---|---|
| Equity sleeve | slsqp_multistart_audited | False | nearest_feasible |
| AGG sleeve | slsqp_multistart_audited | False | matched |
| 70/30 global | slsqp_multistart_audited | **True** | matched |
| Global flat terminal allocation | slsqp_multistart_audited | **True** | matched |

Two of four nodes used the C2 SOCP warm start and landed exactly on target. `solver_strategy`
stays `slsqp_multistart_audited` even when warm-started, by design — C2 hands SLSQP a
pre-solved starting point and lets the same audited final SLSQP pass confirm it, rather than
returning a SOCP-only result unverified by the existing feasibility machinery.

## 4. Why this real-tree test also proves the Phase B.5 fix works

The weights above (`VEA: 0.4692, EEM: 0.1097, CEMB: 0.4212`, **no AGG**) differ from an older
benchmark snapshot recorded before Phase B.5 landed (`VEA: 0.4350, EEM: 0.1630, AGG: 0.0392,
CEMB: 0.3629`). That's not a regression — it's Phase B.5's risk-measure-consistency fix
changing *which* portfolio satisfies the (now correctly measured) volatility target,
exactly as intended. Confirmed directly above: SLSQP-with-B.5-fix and
fast-paths-with-B.5-fix agree with each other, not with the pre-fix number.

## 5. Solve-count / SLSQP-call-count, all 4 benchmark trees

Recorded via `scripts/benchmark_v2.py`, `run_history` (`kind="benchmark"`):

| Tree | Mode | Local solves | SLSQP calls (old, pre-B/C) | SLSQP calls (new) |
|---|---|---|---|---|
| MS_7030_base_2_level | flat | 4 | 36 | 36 |
| MS_7030_base_2_level | forward | 3 | 27 | 27 |
| MS_7030_base_2_level | forward_backward | 4 | 36 | 36 |
| ACWI_AGG_70_30 | flat | 15 | *(not recorded pre-B/C)* | 137 |
| ACWI_AGG_70_30 | forward | 14 | — | 126 |
| ACWI_AGG_70_30 | forward_backward | 19 | — | 173 |
| Global Multi-Asset | flat | 14 | — | 126 |
| Global Multi-Asset | forward | 13 | — | 117 |
| Global Multi-Asset | forward_backward | 19 | — | 171 |
| Views fixture (`posterior_all`) | all 3 modes | 2-3 | — | 2-3 (1 SLSQP call each — B1/B2 fully replace SLSQP here) |

**Important limitation, found while preparing this report:** `slsqp_call_count` is
unchanged between the old and new `MS_7030_base_2_level` runs (36/27/36 both times) *despite*
C2 demonstrably engaging on 2 of 4 nodes (section 3). This is because the benchmark
harness's `local_solves()` only counts the final audited SLSQP loop's
`restart_candidate_count`, which stays at 9 candidates whether or not C2 supplied one of
them as a pre-solved warm start — it does **not** count the separate, more expensive
`cap_results` search loop (2-4 more SLSQP calls) that C2 skips entirely when it lands
binding. The real savings are happening but this specific metric doesn't see them. Fixing
`local_solves()` to count `cap_results`-loop calls too is a good next step for anyone who
wants a trustworthy solve-count delta instead of the wall-clock numbers below.

The views fixture (2 unconstrained nodes, `min_risk` + `max_utility`) is the one case where
the improvement is unambiguous: B1/B2 fully replace the 8-9-candidate SLSQP search with a
single exact convex solve, dropping SLSQP call count to 1 per node.

## 6. Wall-clock: reported, but not trustworthy yet

| Tree | Mode | Old wall-clock | New wall-clock | Apparent ratio |
|---|---|---|---|---|
| MS_7030_base_2_level | flat | 377.1s | 233.0s | 1.6x |
| MS_7030_base_2_level | forward | 304.9s | 200.9s | 1.5x |
| MS_7030_base_2_level | forward_backward | 319.4s | 210.9s | 1.5x |

These look like a real win, and might be — but a follow-up isolation test on the *same*
tree/mode, with the current code and B.5's fix active but every fast path forced off, ran
in **29.9 seconds** (not ~300s). That's faster than both the "old" and "new" numbers above,
which is not physically consistent with fast paths helping — it's consistent with this
machine's timing being dominated by something external (background CPU contention, the
antivirus-scanning artifact already found once, OS-level noise) rather than by the code
being timed. Until wall-clock measurements are repeatable within a reasonable band on this
machine, treat every wall-clock number in this document as a data point, not a proof.

## 7. What's still open

- **Phase B.5**, item 2 (deterministic 100%-proxy seed) and item 1 (realized-volatility
  measure) are both landed and tested (`tests/test_v2_solver_risk_measure_consistency.py`).
- **Phase E** (analytic SLSQP gradients) and **Phase F** (parallel fold estimation,
  fold-level run-history reuse) haven't been started.
- **Phase G's formal gating** (`research_strict`/`hybrid_validated`/`production_fast`
  solver profile on `V2Constraints`) hasn't been built — today, every fast path is live
  unconditionally the moment its gate conditions match, for every solve. Given the
  correctness evidence in section 3, this seems reasonable, but it's a deliberate
  simplification from the original plan's staged-rollout design, worth flagging explicitly:
  there is currently no way to force "old" SLSQP-only behavior in production without
  monkeypatching, the way this report's isolation tests did.
- **A trustworthy wall-clock benchmark** needs either a quieter environment, more repeated
  runs to average out noise, or `time.process_time()`/CPU-time based measurement instead of
  wall-clock — plus the `local_solves()` fix noted in section 5 to make solve-count itself
  reflect C2's savings.
- Only `MS_7030_base_2_level` was checked at the node/`warm_started` level. The same check
  wasn't extended to `ACWI_AGG_70_30`/`Global Multi-Asset` in this pass for time — worth
  doing before treating C2's real-tree benefit as fully characterized rather than
  "confirmed to engage on at least one real tree."
