# Optimizer run persistence — decision record (v2)

*2026-08-18. Produced by a Wizengaimot deliberation (orchestrator: Claude Code;
participants: Codex gpt-5.6-sol medium, Claude sonnet — both reading this repo
and LazyTools). Status: **decided**, not yet implemented.*

## Reframing

Run persistence for the tree engine **already exists**: `lazyportfolio.v2.db`
declares `runs` and `run_artifacts` alongside `trees`;
`lazyportfolio.v2.run_history` implements `record_run`, `get_by_cache_key`,
`list_runs`, `get_run`, `attach_artifact`; Tree Studio calls `record_run`
unconditionally on every cache-miss estimate/backtest. Only the **MCP
surface** (LazyTools `portfolio_tree_estimate/_backtest`,
`portfolio_optimizer_run`) is ephemeral — those tools never touch
`run_history`.

Both reviewers discovered this independently. The task is therefore: **wire
the MCP tools into the existing mechanism and close its gaps** — not design a
new persistence subsystem.

## Decisions

1. **Reuse, don't fork.** Extend `lazyportfolio.v2.run_history` (not
   `store.py`, which is the validated tree-config repository). No new tables
   beyond what's listed below, no new `KNOWN_DBS` entry — update the existing
   `LAZYPORTFOLIO_TREE_DB` description to name LazyTools as a second
   producer. Real schema migrations via `PRAGMA user_version` (the current
   `CREATE TABLE IF NOT EXISTS` cannot evolve a deployed schema).

2. **History ≠ cache.** Today `runs.cache_key UNIQUE` +
   `ON CONFLICT DO UPDATE` collapses repeated executions into one row,
   erasing chronology. Split the concerns:
   - `runs` rows become **immutable, one per execution** (`run_id`).
   - A separate **`run_cache(cache_key → run_id)`** mapping preserves Tree
     Studio's cache-hit fast path (`get_by_cache_key`).
   - Optional **`idempotency_key`** (unique) guards against MCP retries
     creating duplicate rows.
   - Known cost: `tree_studio.py`'s existing `record_run`/`get_by_cache_key`
     call sites must be migrated — this is a behavior change to shipped code,
     not an addition.

3. **Persist the effective config, not just its hash.** Trees are mutable and
   deletable, call-time overrides exist, and objectives/estimators are
   per-node — a single `objective` column is wrong for trees. Store
   `effective_config_json` (the exact config executed), compute
   `config_hash` from that JSON, and denormalize only stable query
   dimensions: `kind`, mode, train size, tree id, config hash, data-as-of.
   Exception: the **flat engine** (`portfolio_optimizer_run`) has a genuinely
   scalar `objective` — give it a denormalized column there.

4. **Explicit, honest tree linkage.** Record whether the config was loaded by
   `name` or supplied inline — `tree_tools.py:_resolve_config` already knows —
   instead of promoting Tree Studio's `_tree_id_for_config` root-display-name
   inference (brittle; inline configs must stay unlinked even when their
   display name matches a saved tree).

5. **Namespaced cache keys.** Studio and MCP return different payload shapes.
   Share the key *construction* (promote the private helpers out of
   `tree_studio.py` into the `lazyportfolio.advisor.snapshot` module, the
   same place `config_hash`/`data_fingerprint` already live) but include a
   producer + result-contract version in the key, so one surface can never
   read the other's incompatible cached payload. Unified history does not
   require interchangeable response caches.

6. **Persistence only from compute calls.** No `portfolio_run_save(result_json)`
   tool — caller-supplied results are forgeable. `persist=` on the compute
   tools saves the internally produced result and returns `run_id`.
   **Kind-differentiated defaults**: `persist=True` on
   `portfolio_tree_backtest` (expensive, deliberate, one call per decision);
   `persist=False` on `portfolio_tree_estimate` / `portfolio_optimizer_run`
   (cheap, plausibly called in agent exploration loops). Policy divergence
   from Studio's unconditional recording is intentional and documented.
   Note: this refinement emerged in the last reply of a single-round
   deliberation and was adopted by the orchestrator without Codex's
   counter-reply.

7. **Privilege gating.** `PortfolioOptimizationTools` currently has neither
   `store_path` nor persistence privileges — add both. A compute tool rejects
   `persist=True` without `allow_persist`; deletion requires `allow_delete`.
   Otherwise read-only profiles gain an indirect write path.

8. **Partial failure is explicit.** Compute may succeed while SQLite
   persistence fails (locking, disk). Return the computed result with
   `persisted: false` plus the error — never silently claim success. Persist
   only completed successful runs.

9. **Soft deletion.** Runs may become evidence cited by change proposals:
   `deleted_at` instead of physical deletion. (The FK from `run_artifacts`
   has no cascade, so a naive hard `DELETE` raises anyway.) MCP surface:
   `portfolio_runs_list`, `portfolio_run_get`, `portfolio_run_delete`.
   Listing: `ORDER BY created_at DESC, run_id DESC`, capped `limit`, filters
   by tree/kind/date.

10. **Evidence convention, decided now.**
    `proposal_evidence.evidence_id = "optimizer_run:<run_id>"` with
    `metadata_json.kind = "optimizer_run"`, produced by one repository
    helper — so the advisor pipeline and this feature don't invent two
    incompatible conventions later.

11. **Flat engine refactor.** `PortfolioOptimizationTools._build_model` →
    `_build_config` + `V2Model.from_config`, so flat runs have an exact
    synthesized config to hash and persist, auditable like tree runs.

12. **Provenance and honesty about reproducibility.** Record *actual* data
    provenance (effective index start/end, estimation window, observation
    count, source, currency, frequencies, `data_fingerprint` — plus OOS dates
    and fold count for backtests), not just the requested `start/end`. The
    fingerprint is computed before data load in Studio today (race): mark it
    best-effort unless/until a backend-returned snapshot identity exists.
    The feature promises **config reproducibility and auditability**, not
    exact numerical reproduction — that would need the market-data store to
    retain immutable snapshots.

13. **Execution identity.** Store `result_schema_version`, Python + package
    versions, LazyPortfolio/LazyTools commit or build id when available,
    chosen solver/route, resolved estimator, and a versioned full
    `result_json` per run kind (`weights_json` nullable: estimates return
    weights, backtests return metrics without terminal weights).

## Deliberation notes

Full consensus on items 1, 3–5, 7–13. Two genuine disagreements, both
resolved on the merits:

- *Dedup-as-history vs immutable rows*: Codex's split (item 2) won; Claude
  conceded that the upsert erases the chronology its own "compare runs over
  time" motivation requires, and contributed the migration-cost caveat.
- *Automatic vs opt-in persistence*: Claude withdrew "automatic everywhere"
  against Codex's volume argument (agents sweep parameter variants; humans
  click once), then proposed the kind-differentiated defaults the
  orchestrator adopted (item 6, with the transparency note above).

Discarded along the way: a new `runs` table (exists), `save_run` in
`store.py` (wrong module), `portfolio_run_save(result_json)` (forgeable),
promoting `_tree_id_for_config` unchanged (brittle inference).
