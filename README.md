# LazyPortfolio

**Hierarchical (V2) portfolio optimization engine and Tree Studio — extracted from LazyFin.**

LazyPortfolio owns the node-tree, goal-first hierarchical allocation engine (Flat, Forward,
and one-pass Forward-plus-backward optimization), its per-node rf/funding financing regimes,
the causal walk-forward drift backtester, the scientific-study significance harness, and
Tree Studio (the local visual editor/runner for allocation trees). It was split out of
[LazyFin](https://github.com/selvaz/LazyFin) so the domain layer (canonical data model, PM
agents, workflows) and the optimization engine can evolve independently.

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

**Dependency invariant:** `LazyPortfolio → market-data-hub` (optional, `[datacore]` extra, for
`MarketDataHubOptimizationBackend`). Never the reverse — LazyPortfolio does not depend on
LazyFin. Anything that wants both (e.g. LazyTools' `PortfolioOptimizationTools`) depends on
both packages directly.

## Layout

```
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

## Optional integrations (extras)

```bash
pip install -e ".[datacore]"   # market-data-hub: MarketDataHubOptimizationBackend
pip install -e ".[dev]"        # test + lint + type-check tooling
```

Without `[datacore]`, supply your own `OptimizationDataBackend` implementation — `backend.py`
imports `market_data_hub` lazily, only inside `load_returns()`.

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
