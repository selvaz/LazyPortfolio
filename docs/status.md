# Project status & history

This engine and Tree Studio were extracted from [LazyFin](https://github.com/selvaz/LazyFin)
on 2026-07-28 (all of `src/lazyfin/optimization/`, `project/tree_studio*`, their tests and
docs). The history below predates the split and is carried over from LazyFin's own
`docs/status.md` for continuity.

> **2026-07-20, V2-only:** the legacy proxy-tree, nested-experiment and Skfolio-direct
> optimizer engines were removed. `hierarchical_v2.py` became the only hierarchical engine
> for Flat, Forward, and Forward plus backward estimates and walk-forward tests, and its
> local solver shrinks both the covariance (Ledoit-Wolf, unconditional) and the
> expected-return vector (equilibrium prior or Bayes-Stein by default, both a per-node
> `constraints.mean_estimator` choice — also editable in the Tree Studio node form) instead
> of using raw sample statistics. A node can also declare `constraints.views`
> (Black-Litterman, confidence-scoped, node-local) — the intended entry point for a future
> LLM agent's macro view, fully audited per view (`V2Audit.views_applied`/`view_details`),
> never a raw weight override. See `hierarchical-v2.md` for the binding contract; iterative
> feedback remains a separately gated next phase.

> **2026-07-22, clean-engine follow-up, PR #37:** component identity
> (`V2Component`/`V2SolveContext`) now backs candidate resolution in `hierarchy.py` instead
> of a `_SYNTH` column-suffix convention; `mean_reference_kind`
> (`none`/`father_proxy`/`benchmark`/`local_weights`) is a fully independent axis from the
> risk-side `volatility_reference`/`max_volatility_reference`/`tracking_error_reference` —
> `mean_estimator="auto"` now resolves off mean-reference completeness alone (the two
> pre-existing cash/leverage and `max_utility` special cases still win). Constraint fallback
> is lexicographic (TEV excess, then volatility deviation, then the objective —
> `V2Audit.constraint_stage_results`), and `tracking_error_policy`/`volatility_target_policy`
> default to `hard_fail` (a deliberate behavior change from the previous automatic
> soft-projection default; opt into `nearest_feasible` where wanted). `mode="flat"` no
> longer depends on the recursive Forward pass succeeding anywhere else in the tree. The
> ambiguous `"root"` reference string is renamed `forward_root_reference`;
> `current_parent_synthetic`/`current_root_synthetic` are reserved and explicitly rejected
> outside a future iterative mode. A new
> `HierarchicalV2Estimator.estimate_direct_bottom_up` proves the Forward pass is never a
> hidden dependency of the final Backward result. Tree Studio's node form is now a lossless
> merge (previously replaced `constraints` wholesale on every save) with explicit-zero
> preserved and corrected raw/synthetic terminology; a `npm test` (jsdom + Node's test
> runner) suite covers this behaviorally. See `optimizer-v2-remediation-status.md` for the
> full phase-by-phase implementation record, and the "Volatility and tracking
> references"/"Moment estimation and views" sections of `hierarchical-v2.md` for the
> updated binding contract.

> **2026-07-28:** extracted into this standalone repo, `lazyportfolio`. No functional
> change — import paths moved from `lazyfin.optimization`/`lazyfin.optimization.v2` to
> `lazyportfolio`/`lazyportfolio.v2`, and `BacktestSpec`'s pydantic base class was replaced
> with an in-repo equivalent (`_PortfolioModel`) so this package no longer depends on the
> LazyFin package at all.

## Hierarchical Tree Studio (V2-only)

The legacy proxy-tree engine, the two-level nested experiment (`nested_experiment.py`), the
Skfolio-backed direct optimizer (`engine.py`'s `SkfolioOptimizer`) and their Studio endpoints
(`/api/allocate`, `/api/backtest`, `/api/nested-experiment`, `/api/legacy-replication`) have
been removed. `hierarchical_v2.py` is the only hierarchical optimization engine and Tree
Studio (`project/tree_studio.py`) only exposes the V2 endpoints (`/api/v2/estimate`,
`/api/v2/backtest`, `/api/v2/audit-bundle`, `/api/v2/client-report`) plus the shared model
catalog, Market Data Hub loader, and instrument search. `iterative_feedback` mode remains
intentionally rejected until a separate coordinator and convergence-history gate are
implemented — see `hierarchical-v2.md`.

V2's local solver estimates the covariance with Skfolio's `ShrunkCovariance` (Ledoit-Wolf)
instead of the raw sample covariance — unconditionally, not configurable. The
expected-return vector is declared per node via `constraints.mean_estimator` (default
`"auto"`): `EquilibriumMu` (reverse-optimized from the node's own benchmark/father reference
weights) when a full reference is available, `ShrunkMu` toward the grand mean otherwise
(`bayes_stein` by default, or explicitly `james_stein` / `bodnar_okhrin`), or `"empirical"`
as an explicit, non-default opt-out to the raw sample mean. The Tree Studio node editor
exposes this as the "Mean estimator" control next to the volatility/TEV reference fields.
This addresses the best-known weakness of plug-in mean-variance: a two-year raw sample mean
is a poor forecast, and an un-regularized sample covariance is unstable at typical node
sizes. See `hierarchical-v2.md#moment-estimation`.

A node can additionally declare `constraints.views`: Black-Litterman views fused onto
whichever (covariance, mean) `mean_estimator` resolved, scoped to that node's own universe,
with confidence in `(0, 1]` converted to Black-Litterman uncertainty via Idzorek's method
(`view_tau` default `0.05`). This is the designed entry point for a future LLM agent's
macro/qualitative view: a bounded, typed payload (instruments, expected return, confidence,
source), never a direct weight override, and never free text merged into the numbers. Every
view is audited per fold — declared payload, prior view return, and posterior view return —
in `V2Audit.views_applied`/`view_details`. Not yet exposed in the Tree Studio node editor
(backend/model only so far). See `hierarchical-v2.md#views-black-litterman`.

`objective` is validated against a fixed set (`min_risk`, `max_return`, `max_ratio`,
`max_utility`, `hrp`) — an unrecognised value raises instead of silently behaving like
`min_risk`, which is what the old UI's `risk_budget`/`hierarchical` labels did (they have
been removed from the editor). `max_utility` is a real Markowitz quadratic-utility
objective; `hrp` is real Hierarchical Risk Parity via Skfolio, bypassing the mean-variance
solve and rejecting (not silently ignoring) a volatility target/cap, a TEV limit, views, or
a non-`"auto"` `mean_estimator` declared alongside it. `risk_aversion` and `risk_free_rate`
are per-node settings with an explicit node → root (tree-wide default) → fixed-constant
fallback — never estimated from data — editable per node in the Studio next to the mean
estimator control, and reused consistently: `risk_aversion` in both `EquilibriumMu` and
`max_utility`; `risk_free_rate` in both `max_ratio`'s Sharpe ratio and every excess-return
calculation. See `hierarchical-v2.md#objectives`.

The canonical external gates live in `project/tree_studio_v2/`; all modes are checked
against Market Data Hub data and the walk-forward FINAL stream is reconstructed by a second
ledger. Tree Studio can export the same cached run as a complete technical Audit ZIP or a
self-contained client HTML report. The audit bundle includes raw and synthetic series by
fold, every solve input, local/composed weights, constraint audits, OOS curves and
cryptographic hashes; sensitive configuration values are redacted.
`project/tree_studio_v2/validate_exports.py` is the independent gate — note three of the
four `validate_*.py` gates depend on a private, git-ignored model file and a live Market
Data Hub connection, so they are not runnable from a fresh clone or in CI; only
`validate_local_solver.py` is self-contained.

The binding reference policy is immutable-anchor: raw sleeve proxies are used at
intermediate nodes and raw `B0` at the root for target-vol, cap and TEV. Synthetic series
replace candidates only. Backward reports `B0_SYNTH` as a diagnostic
fixed-strategic-weight implementation, never as the solver benchmark.

To run the local Studio:

```powershell
scripts\install_tree_studio.ps1   # first run: creates .venv-tree-studio, installs deps
scripts\start_tree_studio.ps1     # subsequent runs
```

or directly:

```bash
python project/tree_studio.py 8766
```

Open `http://127.0.0.1:8766/`. The Studio requires a Market Data Hub database configured
for the active environment (`MARKET_DATA_DB` env var).

## Working agreement for future sessions

- **Determinism:** the optimizer is a deterministic tool, never LLM-driven.
- **Dependency invariant:** `LazyPortfolio → market-data-hub` (optional, `[datacore]`
  extra), never the reverse. No dependency on LazyFin.
- **Dependencies come from GitHub, not PyPI** for git-sourced extras
  (`market-data-hub` via `git+https://github.com/selvaz/...`);
  `tool.hatch.metadata.allow-direct-references` is set.

## Local setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
ruff check src tests project/tree_studio.py
mypy src/lazyportfolio
pytest -q --cov=lazyportfolio --cov-report=term-missing
```
