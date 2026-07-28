# Optimizer V2 remediation — progress tracker

Living progress doc for the current work on PR #37 (`agent/optimizer-methodology-remediation`).
Updated at the end of each phase/checkpoint so work can resume across sessions without
re-deriving context. This is distinct from `docs/status.md` / `docs/hierarchical-v2.md` /
`docs/optimizer-remediation-plan.md` / `docs/optimizer-v2-clean-engine-follow-up.md`, which
get their final update only once the engine work below is complete (see Definition of Done).

## Why this document exists

A prior agent session attempted to transfer a large diff into this branch by committing it
as base64/gzip chunks (`.github/remediation-payload/part-*`) plus a self-modifying GitHub
Actions workflow (`remediation-restore.yml`) meant to reconstruct, apply, commit and
force-push it automatically. That payload was verified corrupted/incomplete (fails gzip
integrity, sha256 mismatch, decodes into a truncated DEFLATE stream) and unrecoverable from
any git ref, reflog, or dangling object on this machine. The real implementation was never
successfully committed anywhere. It is being re-implemented from scratch, starting from the
last verified-clean commit `09dc313` ("Document clean-engine optimizer V2 follow-up") — note
this is *earlier* than the `239ce6a` commit originally identified as clean; `239ce6a` still
carried a leftover temporary CI job (`tooling_snapshot_313`) from an earlier, separate
snapshot-publishing detour. Diff between `09dc313` and `239ce6a` is `ci.yml` only (17 lines).

## Confirmed methodological decisions (do not re-litigate without new evidence)

1. `mean_estimator="auto"`: the two existing special cases (cash/leverage financing active;
   `objective=="max_utility"`) continue to force `bayes_stein` even when a complete
   `mean_reference` is present — they win over the new general rule. The general rule
   (`auto` → `equilibrium` iff a complete `mean_reference` exists, else `bayes_stein`) applies
   only when neither special case is active.
2. Scientific harness: block-bootstrap + Holm-adjusted inference is extended to **all six**
   declared baseline arms (not just `EQUAL_WEIGHT`/`DECLARED_BENCHMARK`).
3. No elaborate backward-compat shims for informal reference strings — sole user, clean
   direct renames (e.g. `root_reference` → `forward_root_reference`), fail loudly on anything
   genuinely incompatible instead of silently coercing.
4. Push strategy: checkpoint per phase (or phase pair), `git push --force-with-lease` to
   `origin/agent/optimizer-methodology-remediation`, shown transparently but without pausing
   for confirmation (user has explicitly authorized proceeding without further per-action
   confirmation for this task).

## Phase status

| Phase | Status | Notes |
|---|---|---|
| 0 — git cleanup + this doc | done | Reset to `09dc313`, pollution removed, force-pushed to origin. |
| 1 — characterization tests | done | `tests/test_optimizer_v2_follow_up.py` (8 tests). Full suite baseline: 354 passed, 1 skipped, ruff clean, mypy --strict clean on v2/. Verified runnable via `C:/Users/Marco/AppData/Local/Programs/Python/Python312/python.exe` (has lazyfin editable-installed + skfolio + pytest/ruff/mypy already present globally — no venv/network install needed; PyPI is unreachable from this sandbox). |
| 2 — typed contracts (component identity, reference/constraint policy) | done | `V2Component`/`V2SolveContext` (additive, not yet wired into hierarchy.py — Phase 3), `mean_reference_kind`/`mean_reference_weights`, `tracking_error_policy`/`volatility_target_policy`/`volatility_cap_policy` (`volatility_cap_policy` validated to only ever be `hard_fail`), `RESERVED_ITERATIVE_REFERENCES` rejected in `validation.py` and `hierarchy.py._reference` with an explicit message. Renamed the ambiguous `"root"` reference-kind string to `"forward_root_reference"` (and the internal `root_reference` parameter throughout `hierarchy.py`/`solver.py`) — no real saved model JSON used `"root"`, confirmed by grep before renaming. Updated 4 existing call sites that used the old string (`tests/test_hierarchical_v2.py` x3, `project/tree_studio_v2/validate_local_solver.py` x1) plus one Phase-1 characterization test whose baseline necessarily changed by this deliberate rename. New `tests/test_optimizer_v2_contract_branches.py` (13 tests). Full suite: 370 passed, 1 skipped, ruff clean, mypy --strict clean. Both real-data validators re-run: `validate_local_solver.py` fully green; `validate_example_estimates.py` still hits only the same pre-existing `forward/Equity` gap already logged above (unchanged, not a new regression). **Remaining for Phase 7:** the Tree Studio GUI dropdown at `project/tree_studio.html:212` still offers `value="root"` — must become `"forward_root_reference"` when GUI work starts. |
| 3 — hierarchy resolver refactor | done | `_solve_local` now builds a real `V2SolveContext` (component identity, raw/synthetic/candidate series, solver-column mapping) instead of ad hoc dicts; `_reference` split into `_risk_reference` (unchanged behavior) and new independent `_mean_reference` (`none`/`father_proxy`/`benchmark`/`local_weights`, full-coverage-required, no invented residual weight). `V2Audit` now populated with `component_id`, `pass_kind`, `candidate_frame_composition`, `mean_reference_source`, `risk_reference_source`. **Interim fallback kept for Phase 4**: when `mean_reference_kind="none"` (every pre-existing config), `reference_weights` still falls back to a risk reference exactly as before — full decoupling (removing that fallback + the new `auto` rule) is Phase 4's job, done deliberately in that order per the plan. Compat facade (`hierarchical_v2.py`) updated to forward to the renamed `_risk_reference`. 373 passed, 1 skipped, ruff clean, mypy --strict clean; both real-data validators re-run (local solver green, `validate_example_estimates.py` unchanged pre-existing gap). 8 new end-to-end tests added (complete local_weights mean reference resolves in both forward/backward, incomplete one fails loudly, mean reference stays independent of a configured risk reference). |
| 4 — moments/solver decoupling + lexicographic fallback | done | Removed the interim risk-reference fallback in `hierarchy.py._solve_local`: `reference_weights` fed to the solver is now *only* ever `mean_reference_weights` (possibly `None`). Since `moments.py`'s `auto` rule already correctly used whatever `reference_weights` it was given, this alone completed the confirmed methodology (`auto` -> equilibrium iff complete mean reference, else bayes_stein; the two existing special cases — cash/leverage financing, `max_utility` — still win, unchanged, confirmed by new tests) with no `moments.py`/`solver.py` auto-logic changes needed. Replaced `_nearest_soft_solution` (combined weighted-sum-of-squares) with `_lexicographic_fallback` (Stage A: minimize TEV excess; Stage B: given fixed TEV, minimize volatility deviation; Stage C: given both fixed, optimize the economic objective) + new `_minimize_metric` helper. `tracking_error_policy`/`volatility_target_policy` now default `hard_fail` and raise immediately (no projection) when infeasible; `nearest_feasible` opts into the staged projection. **Deliberate, documented behavior change**: 2 existing tests (`test_reference_is_not_added_and_unreachable_volatility_uses_nearest_point`, `test_unreachable_tev_limit_uses_minimum_excess_without_inserting_father`) now explicitly opt into `nearest_feasible` (previously implicit/automatic); `_hierarchical_fixture` (shared by several walk-forward/backtest tests) now sets both policies to `nearest_feasible` since a rolling backtest can hit marginally-infeasible folds that should project rather than abort the whole backtest; `project/tree_studio_v2/validate_local_solver.py`'s two "absent father" scenarios updated the same way. `V2Audit.constraint_stage_results` now populated. 377 passed, 1 skipped, ruff clean, mypy --strict clean. Real-data validators: `validate_local_solver.py` fully green; `validate_example_estimates.py`'s pre-existing gap now surfaces on the root node instead of `Equity` (documented above — confirmed structural, not a new regression, reinforces the Phase 6 fix plan). 6 new tests: auto resolves equilibrium only with complete mean reference, special cases still win over a complete mean reference, hard_fail is the default and raises without projection, TEV resolved before volatility in the lexicographic fallback (with stage-order assertion), TEV+volcap hard simultaneously. |
| 5 — flat-mode independence fix | done | `estimate()` now branches to a new `_estimate_flat` before computing anything else; flat's own `_solve_local` call passes `forward_root_reference=None` (flat, like root, is barred from requesting `"forward_root_reference"` anyway). Forward-pass per-node diagnostics are still attached to flat's `node_results` (existing consumers — e.g. `backtest.py`'s `LOCAL:Root` targets — rely on them being present when available) but computed **best-effort**: wrapped in try/except, so any Forward-pass failure anywhere in the tree leaves flat's own result unaffected (diagnostics simply absent for that estimate). 377 passed, 1 skipped, ruff clean, mypy --strict clean; both real-data validators re-run (local solver green; the pre-existing `validate_example_estimates.py` gap unchanged, same root-node case as after Phase 4). Inverted the Phase 1 characterization test into `test_flat_mode_succeeds_when_forward_pass_fails_elsewhere_in_tree` (asserts the fixed behavior, per its own note not to be silently edited). **Deferred, not done here:** required test #12 ("direct bottom-up equals final backward") is *not* flat-vs-forward_backward — re-reading the methodology doc, "direct bottom-up" means invoking the backward composition directly without a prior Forward computation (there is currently no public entry point for that; `_solve_backward` still reuses `forward[node.name]` for leaf results). Needs a small dedicated harness/resolver, naturally a Phase 6 (audit/export) task — left for then rather than forcing a possibly-wrong equality into this phase. |
| 6 — backtest/audit/export field propagation | done | `backtest.py`'s per-fold audits (`V2Fold.audits`/`forward_audits`) already carry the new Phase 2-4 `V2Audit` fields unchanged (they're just `V2Audit` instances flowing through, no field allowlist in `backtest.py` itself). `project/tree_studio_v2/exports.py`'s `point_estimate.json` (via generic `asdict(result.audit)`) already includes them too. Only `backtest/audits.csv` had a fixed column list needing extension — added `component_id`, `pass_kind`, `candidate_frame_composition_json`, `mean_reference_source`, `risk_reference_source`, `constraint_stage_results_json`. Built the "direct bottom-up" harness deferred from Phase 5 (`test_direct_bottom_up_equals_final_backward_within_tolerance`: solves the one true leaf directly via `_solve_local`/`_compose`, bypassing `_solve_forward_root_first` entirely, then calls `_solve_backward` directly and compares to a full `forward_backward` estimate) — required test #12, now with real teeth. Added required test #16 (point-estimate vs. one backtest fold resolve mean/risk reference identically — trivially true by construction since `backtest.py.run()` calls the same `estimate()` per fold, now pinned as a regression guard). 379 passed, 1 skipped, ruff clean, mypy --strict clean. Real-data validators: `validate_exports.py` and `validate_backtests.py` (previously unexercised in this remediation) both green against market-data-hub. `validate_example_estimates.py`'s pre-existing gap (documented after Phase 3/4) intentionally left as-is — the "fix the validator using per-node candidate representation" idea from the Phase 5 note turned out to need more scoping than fits here; tracked as a residual limitation for the final PR body rather than force-fitted into this phase. |
| 7 — Tree Studio GUI lossless remediation | done | `project/tree_studio.html`: fixed explicit-zero-vs-null for `per_asset_cap`/`max_turnover`/`max_leverage`/`vol_target`/`max_volatility`/`max_tracking_error`/`risk_aversion`/`risk_free_rate` (`||''` → `??''`, matching the already-correct `borrow_spread_bps`); `applyNodeForm()` now merges via `Object.assign` instead of replacing `n.constraints` wholesale, so views/`view_tau`/covariance settings/unknown future fields survive any single-field edit; fixed 3 tooltips that wrongly claimed father becomes synthetic in the backward pass; renamed the GUI's `value="root"` dropdown option to `"forward_root_reference"` (engine rejects the old string since Phase 2) with a corrected label; added GUI controls for `mean_reference_kind`/`mean_reference_weights` and `tracking_error_policy`/`volatility_target_policy` (`volatility_cap_policy` shown as read-only text since it can only ever be `hard_fail`), wired into `syncCompatibilityControls()` gating. **New JS toolchain** (none existed before): `package.json` + `package-lock.json` (jsdom devDependency, `npm ci` reachable — this sandbox has npm/registry access despite no PyPI access), `tests/js/tree_studio_v2_contract.test.mjs` (Node's built-in test runner + jsdom, drives the *actual* `renderNode()`/`applyNodeForm()` from the real page — not string-matching, not a logic reimplementation; 5 tests: zero renders as "0", lossless round-trip preserving views/unknown-fields/zero on an unrelated edit, `forward_root_reference` replaces `root`, constraint-policy controls round-trip, tooltip states raw not synthetic), plus a thin `tests/test_tree_studio_v2_contract.py` pytest wrapper (skips gracefully if `npm`/`node_modules` unavailable, matching this repo's existing importorskip convention) so `pytest -q` picks it up too. Added a new `js-quality` CI job (Node 20, `npm ci`, embedded-script `node --check`, `npm test`) — genuinely runs in CI, not silently skipped. **Verified live in a real browser**, not just jsdom: started `project/tree_studio.py` locally, loaded the real sample config, confirmed the new controls render with correct labels, and used the page's own live `state`/DOM (via console-equivalent JS execution) to confirm an explicit `risk_free_rate=0` survives a real `change` event and that injected `views`/an unknown field survive editing an unrelated field (`risk_aversion`) — matching the automated test exactly. 380 passed, 1 skipped, ruff clean, mypy --strict clean, `npm test` 5/5 green. |
| 8 — scientific harness extension | done | `scientific_study.py`: block-bootstrap + Holm-adjusted inference extended from 2 to all 6 declared baselines (`BOOTSTRAPPED_BASELINES` constant, user-confirmed decision). Added a proxy-vs-synthetic representation ablation (`V2_FORWARD`/`V2_FORWARD_BACKWARD` curves — same model/estimator, only candidate representation differs; whichever mode was requested as the primary `V2_FINAL` is aliased rather than recomputed) and a `V2_DIRECT_BOTTOM_UP` arm using the new `estimate_direct_bottom_up` method (empirical companion to the unit-tested invariant). Ablation arms get metrics/curves but are explicitly excluded from the baseline strategy-comparison bootstrap (new test `test_ablation_arms_are_separate_from_strategy_comparisons` pins this). Added production method `HierarchicalV2Estimator.estimate_direct_bottom_up` (`v2/hierarchy.py`, also exposed on the compat facade) — solves every leaf directly (bypassing `_solve_forward_root_first` entirely) then reuses `_solve_backward`'s recursive composition; this is the real "direct bottom-up resolver" the methodology docs asked for, not just a test-only hack — and it let me simplify the Phase 6 direct-bottom-up test to call it directly instead of manually replicating the leaf-solve/compose/backward sequence inline. Deliberately updated 2 pinned-set assertions in `tests/test_scientific_study.py` (curve names, comparison baselines) to reflect the new arms/inference scope — documented, not silent. 381 passed, 1 skipped, ruff clean, mypy --strict clean on **full** `src/lazyfin` (not just v2/, matching CI's `quality` job exactly). Real-data validators re-run one final time: local solver green, exports green, backtests green (all 3 modes), `validate_example_estimates.py` unchanged pre-existing gap (documented, not a regression). **Not done, noted as a residual limitation for the PR body:** the original brief's "fixed-mean-estimator arm" (isolating estimator choice from strategy choice) wasn't added — the 3 existing MIN_VARIANCE arms already provide a covariance-estimator ablation, but a mean-estimator-specific one is still missing; this and fixing `validate_example_estimates.py`'s known gap are the two remaining open items. |
| Final — docs update, PR body, full verification | pending | |

## Verification environment (important for continuity)

This sandbox has no outbound access to PyPI (pip install fails with SSL errors) and no `gh`
CLI, but `git push`/`fetch` to github.com works. However
`C:/Users/Marco/AppData/Local/Programs/Python/Python312/python.exe` already has `lazyfin`
editable-installed plus `skfolio`, `pandas`, `numpy`, `scipy`, `pytest`, `ruff`, `mypy` globally
— use this interpreter directly (no venv, no pip install needed) to run:
```
"C:/Users/Marco/AppData/Local/Programs/Python/Python312/python.exe" -m pytest -q
"C:/Users/Marco/AppData/Local/Programs/Python/Python312/python.exe" -m ruff check src tests project/tree_studio.py
"C:/Users/Marco/AppData/Local/Programs/Python/Python312/python.exe" -m mypy src/lazyfin/optimization/v2
```
PR #37 draft/body state cannot be checked from this sandbox (no `gh`, no reachable GitHub API
over HTTPS via curl/schannel) — only git-level facts (branch head, file contents) are
verifiable here.

`D:\github_projects\market-data-hub` (sibling repo) has a real, populated DuckDB
(`market_data.duckdb`, env `MARKET_DATA_DB` already points to it) — the two live validators that
need real data (`validate_example_estimates.py`, and presumably `validate_backtests.py`) can
actually run here, not just the self-contained `validate_local_solver.py`.

**Update after Phase 4:** the specific first-failing node shifted from `Equity` to the root
(`Global allocation`) — expected and not a new regression. `validate_estimate` raises at the
*first* node hit in dict-iteration order (post-order: children before their parent), and the
mean/risk-reference decoupling in Phase 3/4 shifted `Equity`'s own solved weights just enough
to fall under the check's tight `1e-12` tolerance by coincidence; the root, which structurally
has the exact same "forward node with children" shape, was very likely *always* mismatching
too, just masked because iteration stopped at `Equity` first. Confirmed by checking every
node/mode directly (bypassing the assertion's early-exit): in `forward` and `flat`, only the
root mismatches now (delta ≈ 0.0039); `forward_backward` is fully clean for every node. This
reinforces the diagnosis below — it's structural to any forward-mode node with children, not
particular to one node — and does not change the Phase 6 fix plan.

**Original finding from running `validate_example_estimates.py` against real data on the
clean `09dc313` baseline (before any Phase 2+ change):** it fails with
`forward/Equity: synthetic series mismatch` (and the same failure surfaces under `flat`, since
flat reuses forward's per-node results) on the real "Global allocation_3pct_TEV_father.json"
model — `forward_backward` passes. Root cause: the validator's check (line ~107-111,
`reconstructed = train[terminal_weights] · terminal_weights` vs `result.synthetic_returns`)
is only a valid invariant for backward-composed or leaf nodes. For a **forward**-mode node with
children (here "Equity", which has child "SPY sleeve"), `synthetic_returns` is built from the
parent's own local weights against the *raw child proxy* return series, while `terminal_weights`
recursively substitutes the child's own independently-optimized composition — these are
different series by design (that's the entire Forward-vs-Backward distinction), so the check is
comparing the wrong things, not catching a real engine defect. This is a pre-existing gap in the
validator script itself (not something introduced by any recent change), and it is exactly the
kind of thing the Phase 2/3 component-identity + per-node candidate-representation audit fields
are meant to make explicit — fix the validator's invariant (reconstruct using the actual
candidate representation at that pass/node, not blindly via terminal weights) as part of Phase 6
(backtest/audit/export), not as an ad hoc patch now.

## Final wrap-up — status

All 8 engine/GUI/harness phases are done, committed and pushed. Documentation pass complete:

- Checked `docs/hierarchical-v2.md`/`optimizer-remediation-plan.md`/
  `optimizer-v2-clean-engine-follow-up.md` for the GUI's fixed "father becomes synthetic in
  backward" wording — **not mirrored in any doc**, only the GUI had it (already fixed Phase 7).
- Updated `docs/hierarchical-v2.md`: rewrote "Volatility and tracking references" and "Moment
  estimation and views" sections to describe the shipped contract (constraint-policy fallback
  stages, mean/risk reference separation, component identity, `forward_root_reference` rename,
  reserved iterative references); updated the Forward+backward section to say the bottom-up
  equivalence is now a *tested* invariant, not merely intended.
- Added a dated "Current update (2026-07-22)" banner to `docs/status.md` summarizing the whole
  remediation, pointing to this status doc for detail.
- Added status notes to the top of `optimizer-remediation-plan.md` and
  `optimizer-v2-clean-engine-follow-up.md` marking them implemented/superseded-where-conflicting,
  pointing to this file for the phase-by-phase record.

Final verification pass (this session, all green): `pytest -q` → 381 passed, 1 skipped;
`ruff check src tests project/tree_studio.py` → clean; `mypy src/lazyfin` (full package, matching
CI's `quality` job exactly) → clean; `npm test` → 5/5 green; all 4
`project/tree_studio_v2/validate_*.py` scripts against the real market-data-hub DuckDB → green
(`validate_example_estimates.py`'s one known, pre-existing, documented gap unchanged).

### Residual limitations (honest, not swept under the rug)

1. `project/tree_studio_v2/validate_example_estimates.py`'s synthetic-series-reconstruction check
   is structurally invalid for a forward-mode node with children (compares a terminal-weight
   reconstruction against a node's own local-weight-based synthetic return, which legitimately
   differ by Forward's own definition) — pre-existing on the clean baseline, not introduced by
   this remediation, unchanged throughout. A real fix needs the validator rewritten to use each
   node's actual `candidate_frame_composition`/`pass_kind` audit fields per node rather than a
   blanket terminal-weight check; scoped out of this session.
2. The scientific harness has a covariance-estimator ablation (the three `*_MIN_VARIANCE` arms)
   but no dedicated fixed-mean-estimator ablation isolating mean-estimator choice from strategy
   choice — not added.

### Remaining for the user / next session

- Update PR #37's body on GitHub (this sandbox has no `gh` CLI and no reachable GitHub API to do
  it directly — flagged since Phase 0). Suggested content: implementation summary per phase (see
  table above), the two residual limitations, and the informal `"root"` → `"forward_root_reference"`
  string migration note (only change with any external-facing effect; no other breaking changes).
- Confirm PR #37 is still in draft on GitHub (per the original instructions — draft unless told
  otherwise; not independently verifiable from this sandbox).
- `node_modules/` is gitignored; anyone pulling this branch needs `npm ci` before `npm test`
  works locally (CI's new `js-quality` job does this automatically).

## Post-review fixes (2026-07-22, second pass)

An external review of the CI-green head (`6391fc3`) confirmed the core Forward/Backward
methodology, `estimate_direct_bottom_up`, flat independence, and `risk_free_rate=0.0` handling
as correctly implemented, but found several real gaps. Verified each finding against the actual
code (not taken on faith) before fixing. Status:

| Finding | Verified? | Action |
|---|---|---|
| Component identity: two siblings sharing a proxy silently double-count weights | **Confirmed** — reproduced (`terminal_weights` summed to 2.0, not 1.0) | Fixed: `hierarchy.py._solve_local` now raises `V2OptimizationError` the instant a solver column would collide (shared sibling proxy, or a child proxy equal to a direct instrument at the same node), and duplicate child *names* under one parent are rejected too. Added `validation.py._validate_tree_structure`: duplicate node ids, duplicate node names (used as result keys tree-wide), a node with more than one parent, cycles, and the same sibling-proxy-collision/proxy-vs-direct-instrument-collision, all rejected at config-validation time (JSON path), on top of the hierarchy.py runtime guard (works regardless of construction path). This closes the actual danger (silent wrong answers); it does **not** rearchitect solver columns to be fully opaque component-id-based (`component::<id>`) as the review's Phase 1 envisioned — that remains a deeper future improvement if ever needed, now that the collision it was protecting against is a hard validation error instead of silent. |
| `mean_reference_kind="benchmark"` raises in `forward_backward` (frame has `_SYNTH` columns, not raw benchmark tickers) | **Confirmed** — reproduced exactly | Fixed: unified `benchmark` and `local_weights` resolution into one `_resolve_component_weights_from_raw_map` helper that maps a raw-ticker-keyed declaration onto whatever solver column currently represents that component via `V2SolveContext`, not via `frame.columns` string matching. Works identically in Forward and Backward now. |
| `mean_reference_kind="father_proxy"` is structurally incoherent (a node's own proxy is never one of its own candidates) | **Confirmed** by inspection | Removed `father_proxy` from `mean_reference_kind` entirely (`SUPPORTED_MEAN_REFERENCE_KINDS`, GUI dropdown) — it remains valid for the risk-side `volatility_reference`/`max_volatility_reference`/`tracking_error_reference` axes, where it is coherent. |
| GUI "Nessuno (usa risk reference come oggi)" text contradicts the actual decoupled contract; "Equilibrium (father/benchmark)" label stale | **Confirmed** | Fixed both labels/tooltip text in `tree_studio.html`. |
| No test for a sleeve absent from B0 using a complete `local_weights` mean reference | **Confirmed gap** | Added `test_sleeve_absent_from_benchmark_uses_complete_local_weights` (Equity/Gold/Bond root, B0 = ACWI/AGG only, Gold absent from B0, complete local_weights covering all three). |
| No post-normalization hard-constraint re-check after weight zeroing/renormalization | **Confirmed gap** | Added `V2LocalOptimizer._audit_hard_constraints`: re-validates bounds and the volatility cap on the final canonicalized weight vector, raising rather than silently returning a result that no longer satisfies a declared-hard constraint. |
| No single end-to-end fixture chaining root RF=3%/child RF=0.0 explicit through point-estimate → backtest fold audits → financing rates | **Confirmed gap** | Added `test_explicit_zero_risk_free_rate_end_to_end`. |
| `backtest/audits.csv` omits `risk_free_rate_source`/`risk_aversion_source`/financing regime+sources | **Confirmed gap** | Added those columns to `project/tree_studio_v2/exports.py`'s audit CSV. |
| Scientific harness silently intersects OOS indexes across arms, hiding dropped observations | **Confirmed by design** | `ScientificStudyProtocol.require_identical_oos_index` (default `True`) now raises if any arm's OOS index isn't exactly identical; `ScientificStudyResult.dropped_observations` records what would have been dropped if the flag is explicitly set to `False`. |

Verified all fixes against the real market-data-hub model afterward (all 4 `validate_*.py`
green) and re-ran the full suite: **392 passed, 1 skipped, ruff clean, mypy --strict clean,
npm test 5/5.**

### Findings from this review deliberately not acted on (would need significantly more time)

- Full component-identity rearchitecture using opaque `component::<node-id>` solver columns and
  `node.id` (not `node.name`) as the result-dict key everywhere — mitigated instead via strict
  uniqueness validation (see above), which closes the practical danger without an invasive,
  wide-blast-radius rename across the engine, GUI, exports and the entire existing test suite
  (which extensively keys results by human-readable name, e.g. `estimate.node_results["Root"]`).
- GUI `MeanRisk → HRP → MeanRisk` round-trip: whether HRP-incompatible fields should be
  preserved dormant or destructively cleared is not yet decided or tested.
- Scientific harness fixed-mean-estimator ablation arm (isolating mean-estimator choice from
  strategy choice) — still missing, as already noted before this review.
- `validate_example_estimates.py` rewrite using `pass_kind`/`candidate_frame_composition` audit
  fields to fix its known pre-existing invariant gap — still not done, as already noted.
- Full audit JSONL-per-solve restructuring (the CSV gained the specific missing scalar columns
  the review named, but is still a flat CSV, not a fully nested per-component replay structure).

## Post-review fixes (2026-07-22, third pass)

Second external review pass on head `159a08a` confirmed all prior P1 fixes and found one
remaining real gap plus two "declared but not fully covered" test completions:

| Finding | Verified? | Action |
|---|---|---|
| A node declared in `nodes` but never referenced as anyone's child is silently ignored by the model builder (never solved, never audited, no error) | **Confirmed** — reproduced with an `"orphan"` node holding a `GOLD` instrument; it never appeared in `terminal_instruments()` and no error was raised | Fixed: `_validate_tree_structure` now walks from root and rejects any node id not reached, with a clear list of the unreachable id(s). **Explicitly confirmed with the user first** that this must stay a hard error (not silently allowed, not just a warning) — leaf nodes (no children of their own) are completely unaffected by this check and remain fully supported, confirmed by a dedicated regression test. |
| `root_id`/node `id` empty or missing | Not previously checked | Added: empty `root_id` is rejected (via the pre-existing root-lookup check, which already fires with a clear "root node is missing" message); any node with a missing/empty `id` is rejected explicitly. |
| Duplicate direct instrument declared twice on the same node | **Confirmed** — the old code built `direct_instruments` as a set comprehension, which silently de-duplicated instead of detecting the duplicate | Fixed: duplicates in one node's own `instruments` list are now rejected explicitly. |
| "Sleeve absent from B0" test only exercised `mode="forward"`, docstring claimed both passes | **Confirmed test gap** (not a code gap — the underlying resolver logic is pass-agnostic) | Test now loops over `("forward", "forward_backward")` and asserts identically in both. |
| RF=0 end-to-end test's positive-cash-lending assertion was conditional (`if financing_regime == "cash_lending":`), not deterministic | **Confirmed test gap** | Fixed the fixture (`max_weights` capping risky exposure at 0.6, leverage left at 1.0 so only the lending regime is ever attempted) to *force* a `cash_lending` outcome deterministically, and the assertion is no longer conditional. |

All fixes verified: **397 passed, 1 skipped, ruff clean, mypy --strict clean, npm test 5/5**, all
real-data validators green. No new P0/P1 found in this pass beyond the orphan-node gap, which is
now closed.

### Still open (explicitly not closed in this pass, per the reviewer's own non-blocking triage)

- Full component-identity rearchitecture (opaque `component::<id>` solver columns, `node.id` as
  the result key everywhere instead of `node.name`) — deliberately not done; the practical risk
  is closed via strict uniqueness/reachability validation instead (see above and the second-pass
  section). Renaming a node's display name still changes its audit identity; two components
  cannot share a proxy. Documented as a known limitation of the current identity model, not
  silently glossed over.
- `_audit_hard_constraints` only re-checks bounds and the volatility cap after canonicalization,
  not TEV/exact-target adherence to their resolved lexicographic-fallback bounds.
- Tree Studio `MeanRisk → HRP → MeanRisk` round-trip: resolved for views (see the Views GUI
  section below — HRP destructively clears them, consistent with `cash_enabled`/`mean_estimator`
  elsewhere in the same form, and a regression test confirms nothing resurrects on switching
  back). Not separately re-verified for every other HRP-incompatible field.
- `validate_example_estimates.py`'s known pre-existing invariant gap (documented since Phase 3/4)
  — not fixed.
- Scientific harness fixed-mean-estimator ablation arm — not added.
- Audit is not fully "replay every decision from one structured artifact" — `point_estimate.json`
  has the complete `V2Audit` per node/pass, and the CSV now has the scalar provenance fields, but
  there is no single structured (e.g. JSONL-per-solve) artifact with the full component→raw
  series→active column→synthetic identity→resolved mean weight→coverage mapping in one place.

## Tree Studio: Black-Litterman views editor (2026-07-22)

Black-Litterman views were already implemented engine-side (`V2View`/`view_tau`, Idzorek
confidence-scaled uncertainty in `black_litterman_posterior`) but had **zero GUI surface** —
the only way to set a view was hand-editing the JSON config. Added a dedicated node-level
editor in `project/tree_studio.html`, per explicit user requirement that views must **not**
live inside/depend on the Constraints section:

- New `<div class="section" id="viewsSection">`, structurally separate from Constraints
  (own `<h2>`, own DOM subtree) — verified by a dedicated jsdom test that neither section
  contains the other.
- "Aggiungi view" button adds a blank view row; each row exposes **only this node's own**
  tickers/proxies (`allocationComponents` filtered to exclude `~`-prefixed synthetic
  representation children) as checkboxes, each paired with a signed-weight input — supports
  relative views (multiple checked tickers, e.g. `SPY:1, VGK:-1`) as required, not just
  free-text ticker:coefficient strings.
- `view_tau` gets its own field (`c-view-tau`), independent of the per-view fields.
- HRP gating: `applyNodeForm()` clears `views` to `[]` whenever `objective==="hrp"`; the
  `n-objective` change handler now triggers a full `render()` (previously only `renderTree()`),
  so the views DOM rows are actually removed, not just internally cleared while stale disabled
  rows lingered on screen — the earlier partial-render approach let a stale DOM read resurrect
  the "cleared" views the next time the objective was switched away from HRP.

### Two bugs found via live end-to-end browser testing (real Python backend, not just jsdom)

Both reproduced with a real click-through in the browser against `project/tree_studio.py`,
not just unit tests:

1. **GUI**: a view row added via "Aggiungi view" but left completely empty (no ticker checked,
   no return, no confidence — e.g. left over from a user changing their mind) was still sent to
   the backend, which correctly-but-cryptically rejected it (`expected_return must be a finite
   number`). Fixed in `applyNodeForm()`: rows with no instruments, no expected_return, and no
   confidence are filtered out before being written to state. A *partially*-filled row still
   fails loudly at the backend, deliberately — only genuinely-empty scaffold rows are dropped.
2. **Backend, real bug, not GUI-only**: [`model.py`](../src/lazyfin/optimization/v2/model.py)
   had `view_tau=float(constraints.get("view_tau", 0.05))` — `.get(key, default)` only applies
   the default when the key is *absent*, not when present-but-`""`. Every GUI payload sends
   `""` (not a missing key) for an unset optional numeric field, so leaving `view_tau` blank
   with even one filled-in view crashed the solve with `could not convert string to float: ''`.
   This is exactly the explicit-zero-vs-null class of bug the review rounds targeted elsewhere
   in the pipeline, just not previously caught for this one field since it shipped after the
   review passes. Fixed to match the `not in (None, "")` idiom used by every other optional
   field in the same function; added both a Python regression test
   (`test_model_from_config_treats_empty_string_view_tau_as_default`) and confirmed via a live
   Ottimizza run with a real SPY-vs-VGK relative view that the fix resolves it end-to-end.

Coverage added: 6 new jsdom tests (`tests/js/tree_studio_v2_contract.test.mjs`) covering the
relative-view round-trip, unchecking clearing the paired weight, the empty-row filter, `view_tau`
round-trip, structural separation from Constraints, and the HRP clear/no-resurrect behavior; one
new Python test for the `view_tau` empty-string fix. Full suite green: pytest (399 passed, 1
skipped), ruff clean, `mypy --strict` clean on `src/lazyfin/optimization/v2`, npm test 11/11.

### PR #37 body — suggested content (this sandbox cannot push it directly)

The user should update the PR body on GitHub with, at minimum:
- Current head: `159a08a` (plus whatever this session's final commit hash is) — not the stale
  `07ac1ad...`/`d4e784b...` heads or CI runs 208/213 the body currently references.
- One paragraph per phase (0-8, see the table at the top of this file) summarizing what shipped.
- The two post-review fix passes above (component-identity collision rejection + mean/risk
  reference fix pass; structural validation completion pass).
- Migration note: the only externally-visible breaking change is the informal `"root"` reference
  string being renamed to `"forward_root_reference"`, and `mean_reference_kind="father_proxy"`
  no longer being accepted (it was never a working option — always structurally incoherent).
- The "still open" list immediately above, so reviewers know what is and isn't covered.
- PR stays in **draft** per the original instructions.
