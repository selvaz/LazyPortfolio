# LazyPortfolio

**Hierarchical (V2) portfolio optimization engine and Tree Studio — extracted from LazyFin.**

LazyPortfolio owns the node-tree, goal-first hierarchical allocation engine (Flat, Forward,
and one-pass Forward-plus-backward optimization), its per-node rf/funding financing regimes,
the causal walk-forward drift backtester, the scientific-study significance harness, and
Tree Studio (the local visual editor/runner for allocation trees). It was split out of
[LazyFin](https://github.com/selvaz/LazyFin) so the domain layer (canonical data model, PM
agents, workflows) and the optimization engine can evolve independently.

## Setup

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_first_run.ps1
```

This installs LazyPortfolio, always installs and wires up **market-data-hub** (the one
supported way every function here reads price/return data — see "Data source" below), asks
only about the genuinely optional pieces (dev/test tooling, Tree Studio's JS test harness),
and locates or asks for the `.duckdb` database this environment should use, persisting it as
`MARKET_DATA_DB`. Idempotent — safe to re-run after pulling an update or on a new machine. Pass
`-MarketDataHubPath <dir>` to point at a specific checkout, or `-DbPath <file.duckdb>` to skip
the interactive database prompt.

On a brand-new machine with no market-data-hub checkout yet, clone
[market-data-hub](https://github.com/selvaz/market-data-hub) next to this repo and run **its**
`setup_first_run.ps1` first (it builds the database and its ingestion pipeline); then run this
repo's script, which will detect and reuse it.

## Data source

**market-data-hub is the preferred and only supported way to feed real data into every
function in this repo** — `MarketDataHubOptimizationBackend`, Tree Studio, the scientific-study
harness. It is packaged as an installable "extra" (`[datacore]`) purely because
`backend.py` imports it lazily so the package stays testable without it (synthetic returns,
unit tests), never because it's meant to be skipped in practice. `setup_first_run.ps1` always
installs it — you are not asked. Supplying your own `OptimizationDataBackend` is only meant
for testing or a from-scratch integration, not as an alternative production data source.

> **Hierarchical Tree Studio:** the V2 engine supports Flat, Forward, and one-pass Forward
> plus backward optimization, local per-series bounds, annualized volatility/TEV constraints,
> saved JSON configurations and a shared walk-forward ledger. Raw father proxies and raw B0
> remain immutable constraint references; synthetic series replace candidates only. Covariance
> estimation is explicit and audited: fixed-shrinkage `ShrunkCovariance` is the default, while
> data-adaptive `LedoitWolf` is a separate opt-in. Expected-return estimation supports
> equilibrium and shrinkage methods, but `max_utility` with `mean_estimator=auto` resolves to
> Bayes–Stein to avoid mechanically recreating a feasible reference portfolio. Black-Litterman
> views are node-scoped and the default `prior_risk` policy changes preferences without
> silently redefining volatility targets or caps. Every objective is validated against a fixed
> set (`min_risk`, `max_return`, `max_ratio`, `max_utility`, `hrp`); unrecognised or unsupported
> settings fail loudly. Risk aversion and the risk-free rate follow the same node → root →
> hard-default chain in optimization, OOS metrics and exports. HRP is identified as the Skfolio
> Pearson/Ward/variance variant and is independently audited after fit. Results can be
> downloaded as a reconstruction-ready Audit ZIP or self-contained report. Engineering gates do
> not imply financial superiority; the scientific harness compares V2 with equal weight, the
> declared benchmark, sample/shrunk minimum variance and HRP on the same OOS folds with
> block-bootstrap inference. See [`docs/hierarchical-v2.md`](docs/hierarchical-v2.md) and
> [`docs/optimizer-remediation-plan.md`](docs/optimizer-remediation-plan.md).

## Why a separate repo

This engine has no dependency on LazyFin's domain model — it consumes a plain
`instruments: list[str]` universe and a return matrix, and produces target weights plus a typed
audit trail. Keeping it standalone lets it be reused (or benchmarked) outside the PM-agent
context, and lets LazyFin's domain layer stay a thin, fast-installing library.

**Dependency invariant:** `LazyPortfolio → market-data-hub` (the `[datacore]` extra — see "Data
source" below; `setup_first_run.ps1` always installs it). Never the reverse — LazyPortfolio
does not depend on LazyFin. Anything that wants both (e.g. LazyTools'
`PortfolioOptimizationTools`) depends on both packages directly.

## Layout

```
setup_first_run.ps1     guided install (package + market-data-hub + db config)
src/lazyportfolio/
  backend.py          OptimizationDataBackend protocol + MarketDataHubOptimizationBackend
  calendar.py          rebalance-date / resampling helpers for the walk-forward backtester
  models.py            BacktestSpec (walk-forward protocol contract)
  hierarchical_v2.py    compatibility facade re-exporting v2/api.py
  scientific_study.py   paired block-bootstrap significance harness vs. baselines
  walk_forward.py       glue between BacktestSpec/calendar and the backtester
  v2/
    contracts.py        V2Node / V2Constraints / V2Audit / V2BacktestReport / ...
    validation.py        fail-loud config validation + legacy JSON migration
    model.py             config dict -> validated V2Model (node tree + benchmark)
    moments.py           covariance/mean estimation + Black-Litterman (skfolio-backed)
    solver.py            per-node local optimizer (SLSQP/HRP), rf/funding regimes
    hierarchy.py         Flat/Forward/Forward+Backward traversal across the node tree
    backtest.py          causal walk-forward ledger, costs, financing accrual, metrics
    api.py               stable public assembly point
project/
  tree_studio.py         local HTTP app (visual editor/runner) over the V2 engine
  tree_studio.html        single-page UI
  tree_studio_v2/         export/audit-bundle builder + standalone validation scripts
scripts/                 Tree Studio install/launch scripts (PowerShell)
docs/                    methodology, per-node financing, remediation history
tests/
```

## Manual install (advanced / non-Windows)

`setup_first_run.ps1` is the recommended path. To install by hand instead:

```bash
pip install -e ".[datacore]"   # core + market-data-hub (see "Data source" above)
pip install -e ".[dev]"        # + test/lint/type-check tooling
```

Then set `MARKET_DATA_DB` to your `.duckdb` file's path yourself.

## Development

```bash
pip install -e ".[dev]"
ruff check src tests project/tree_studio.py   # lint + import order
mypy src/lazyportfolio                         # strict type checking
pytest -q                                      # tests
```

Tree Studio's HTML/JS contract has its own Node-based test harness:

```bash
npm install
npm test    # node --test tests/js/tree_studio_v2_contract.test.mjs
```

## License

Apache-2.0 (to align with the rest of the LazyBridge ecosystem).
