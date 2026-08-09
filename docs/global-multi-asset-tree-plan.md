# Global Multi-Asset Optimization Tree Plan

## Purpose

Build an auditable multi-level strategic tree for LazyPortfolio V2 that can be
backtested over ten years and operated daily after a separate research approval.
The initial tree starts from a transparent global equity/bond benchmark and
adds commodities and diversifying alternatives as active root sleeves.

## Standard node policy

| Setting | Policy |
|---|---|
| Objective | `max_return` |
| Estimation window | 102 weekly observations |
| Estimation frequency | Weekly |
| Rebalance frequency | Monthly |
| Risk-free rate | `0` annualized |
| Borrowing spread | `0` bps |
| Transaction costs | `0` bps |
| Root volatility reference | B0 benchmark |
| Non-root volatility reference | Immutable raw father proxy |
| Volatility target | Volatility of the declared reference |
| Target mode | `exact` |
| Target policy | `nearest_feasible` |
| TEV reference | Immutable raw father proxy, where enabled |
| TEV policy | Hard fail unless explicitly changed |
| Final hierarchy mode | `forward_backward` |
| Covariance estimator | `shrunk_fixed` unless a separate study says otherwise |

The father and B0 remain reference series. They must never become undeclared
fallback allocations. Forward uses raw child proxies as candidates; Backward
uses the synthetic return series produced by the child sleeves.

## Proxy contract: reference and parent representation

Every non-root sleeve proxy has two simultaneous jobs:

1. **Node father reference.** It is the immutable raw series against which the
   node resolves its target volatility and, when configured, TEV.
2. **Parent candidate representation.** In the Forward pass, the parent solves
   over this raw proxy. In the Backward pass, that parent candidate is replaced
   by the optimized synthetic return series of the sleeve.

Therefore a proxy must be a real, broad, liquid, long-history economic anchor,
not merely a convenient ticker label. A new ETF may enter the tree only if it
can serve this contract at its proposed depth.

Configuration rules:

- a node proxy must be unique among its siblings;
- a node must not add its own father proxy as a direct local candidate;
- **sole approved exception:** `GLD` is both the precious-metals node proxy
  and an investible local candidate, so the optimizer may retain broad gold
  exposure rather than being forced to express the sleeve solely through the
  satellite metals;
- a sleeve that is decomposed must contain at least two meaningful non-proxy
  candidate instruments or child sleeves;
- a sleeve with no genuine decomposition stays a direct instrument in its
  parent rather than becoming a one-candidate child node;
- a deeper node is introduced only when a suitable father proxy and at least
  two distinct candidates exist.

## Benchmark B0

Initial globally diversified 60/40 benchmark:

```text
60% ACWI  Global all-country equity
20% AGG   USD aggregate bonds
20% BNDX  Global aggregate bonds ex-USD
```

Commodities and alternatives are intentionally outside B0. The root optimizer
may select them, subject to the root target-volatility and other constraints.

Before B0 is frozen, verify common adjusted-close history for all three ETFs on
the exact backtest start date.

## Validated tree topology

The following is the **only topology to configure as the initial production
candidate**. It contains no empty node, no placeholder ticker, and no node
whose proxy also appears in its own candidate set. A labelled sleeve has its
proxy in square brackets; unlabelled tickers are direct candidate instruments
of the preceding sleeve. It is divided into a core which runs with the current
database and clearly marked extensions that require their data backfill first.

```text
Global Multi-Asset [B0]
|- Global Equity [ACWI]
|  |- SPY United States
|  |- Developed ex-US [VEA]
|  |  |- Europe [VGK]: EWG, EWQ, EWI, EWU
|  |  |- EWJ Japan
|  |  |- EWC Canada
|  |  `- EWA Australia
|  `- Emerging Markets [VWO]
|     |- MCHI China
|     |- INDA India
|     |- EWT Taiwan
|     |- EWY South Korea
|     `- Optional approved EM country satellites
|
|- USD Fixed Income [AGG]
|  |- Government duration [IEF]
|  |  |- SHY short Treasury
|  |  |- IEI intermediate Treasury
|  |  `- TLT long Treasury
|  |- Investment-grade credit: LQD direct candidate
|  |- High yield: HYG direct candidate
|  |- Inflation-linked: TIP direct candidate
|  `- Municipal: MUB direct candidate
|
|- International Fixed Income [BNDX]
|  |- BWX international government direct candidate
|  |- EUHY euro high-yield credit direct candidate
|  |- IHY global ex-US high-yield credit direct candidate
|  `- Emerging debt [EMB]
|     |- CEMB hard-currency corporate
|     |- HYEM hard-currency high yield
|     `- EMLC local-currency government
|
|- Broad Commodities [DBC]
|  |- USO energy direct candidate
|  |- Industrial metals: DBB and CPER direct candidates
|  |- Precious metals [GLD]: GLD, SLV, PPLT, PALL
|  `- Agriculture [DBA]: CORN, SOYB, WEAT
|
`- Extension after data backfill: Diversifying Alternatives [QAI]
   |- WTMF managed futures / trend
   |- MNA merger arbitrage
   `- BIL liquidity stabilizer
```

Square brackets identify each sleeve's raw father proxy. They do not mean that
the proxy is automatically a child candidate.

The immediately runnable core is fully present in MarketDataHub and has a
common adjusted-price history from 2016-08-06 onwards. The alternatives node
is deliberately not configured until `QAI`, `WTMF`, and `MNA` pass the
insertion gate. The energy child node is likewise delayed until `BNO` and
`USL` are backfilled; meanwhile `USO` remains a direct DBC candidate.

## Equity design rules

### Global decomposition

Use `SPY`, `VEA`, and `VWO` under `ACWI`. `VEA` means developed markets
excluding the United States, so USA is a direct sibling and not a child node in
the baseline. `VGK` is a valid Europe proxy because its node has the four
separate country candidates `EWG`, `EWQ`, `EWI`, and `EWU`.

### Country, sector, and style axes

Countries, sectors, and styles overlap materially. Do not place all three as
peer candidates in one decomposition. Build sibling research variants instead:

```text
US sector variant: XLB, XLE, XLF, XLI, XLP, XLU, XLV, XLY, VGT
US style variant:  VUG, VTV, IWM
Europe countries:  EWG, EWQ, EWI, EWU, ...
Europe sectors:    EXH1.DE, EXH4.DE, EXV1.DE, EXV3.DE, EXV4.DE, EXH9.DE
Dev ex-US styles:  EFV, EFG, SCZ
```

Start production research with countries/regions. Sector and style variants
are controlled comparisons, not layers to be added simultaneously. In a US
sector variant, `SPY` becomes the proxy of a new US child node and is replaced
by the sector ETFs; in a US style variant it is replaced by the style ETFs.
The raw `SPY` proxy is never also a candidate in that child.

### Emerging markets and China

No active, clean, liquid ETF proxy for EM ex-China currently offers a usable
ten-year history in the target universe. `EMXC` and `XCNY` are too recent.

Initial policy:

1. Use `VWO` as EM father proxy.
2. Keep `MCHI` as an independent direct China candidate in the VWO node.
3. Permit the optimizer to underweight or set China to zero.
4. Never label the result “EM ex-China”, because its father still includes
   China.

A true EM ex-China tree requires an approved synthetic proxy or a shorter
historical window.

## Fixed-income design rules

The USD AGG branch separates duration, IG credit, HY credit, inflation-linked
and municipal risk. Treasury maturity ETFs are alternatives inside the
government sleeve, not strategic additive buckets.

The BNDX core is already valid: `BWX`, `EUHY`, and `IHY` are direct
instruments, while EM debt is a true `EMB` child sleeve. `EUHY` is the
database-native iShares euro high-yield corporate ETF, with history since
2012, and is preferred to introducing the separate UCITS listing `IHYG`.
`IHY` replaces the retired `HYXU`: it preserves global ex-US high-yield
exposure without inheriting HYXU's November 2025 mandate and ticker change.
`IGOV`, `ISHG`, and `PICB` are optional later additions; they remain direct
instruments inside the `BNDX` node, not child nodes. `EMB` has the distinct
candidates `CEMB`, `HYEM`, and `EMLC`, organized by the economically meaningful
hard/local-currency and sovereign/corporate risks.

The first AGG tree has one genuine duration child: `IEF` is its proxy and
`SHY`, `IEI`, and `TLT` are its non-proxy candidates. `LQD`, `HYG`, `TIP`,
and `MUB` remain direct AGG-node candidates until a later approved expansion
gives each a meaningful candidate universe. A future IG-credit sleeve can use
`LQD` as proxy with `VCSH`, `VCIT`, and `VCLT` as candidates.

## Commodity design rules

Keep `DBC` as broad father proxy. Put each specific futures ETF under exactly
one economic family: energy, industrial metals, precious metals, or agriculture.
Document futures roll behavior as ETF methodology, not as a data-quality error.
Do not mix commodity-producer equities with futures-based commodity ETFs in the
base commodity sleeve.

`USO` is initially a direct DBC candidate. Once `BNO` and `USL` have been
backfilled, it becomes the energy sleeve proxy and is removed from its own
candidate list; the energy node then uses `BNO`, `USL`, `UGA`, and `UNG`.
`GLD` is the one approved proxy/candidate exception: it remains investible in
the precious-metals node alongside `SLV`, `PPLT`, and `PALL`. `DBA` remains
only the agriculture proxy, with its local candidates represented by the other
specific agricultural ETFs. Industrial metals stays at the direct-DBC level:
`DBB` and `CPER` are both direct DBC
candidates, because no industrial-metals child has two credible non-proxy
candidates yet.

## Alternatives design rules

The initial alternatives branch is:

```text
QAI, WTMF, MNA, BIL
```

`QAI` is the sleeve proxy and father reference; it is not also a direct local
candidate of the QAI sleeve. `WTMF`, `MNA`, and `BIL` are the initial local
candidate instruments that reconstruct the sleeve for the parent. `BIL` is
intentional: it provides a low-volatility component that makes a father target
based on the lower-volatility QAI proxy more attainable without inventing cash
or enabling leverage.

Exclude VIX, VIX futures, VIX ETPs, and crypto from the base strategic sleeve.
Treat them as signal/tactical research instruments. Keep infrastructure and
global REITs in a later “listed real assets” variant because their equity beta
is material.

`WTMF` must receive a `strategy_break` metadata flag because the manager
materially changed its strategy in 2021. Evaluate both full-history and
post-change robustness results.

## ETF universe expansion

### Batch 1: required before the first full tree

| Ticker | Classification | Intended use |
|---|---|---|
| QAI | ALTERNATIVES / Global / multi-strategy | alternatives father proxy only |
| WTMF | ALTERNATIVES / Global / managed futures | trend sleeve |
| MNA | ALTERNATIVES / Global / merger arbitrage | relative-value sleeve |
| IGOV | FIXED_INCOME / Developed ex-US / government | international government |
| ISHG | FIXED_INCOME / Developed ex-US / short government | BNDX direct candidate |
| PICB | FIXED_INCOME / Developed ex-US / IG corporate | international IG |
| EFV | EQUITY / Developed ex-US / value | style research variant |
| EFG | EQUITY / Developed ex-US / growth | style research variant |
| SCZ | EQUITY / Developed ex-US / small cap | style research variant |
| BNO | COMMODITIES / Global / energy | energy sleeve candidate |
| USL | COMMODITIES / Global / energy | energy sleeve candidate |
| VCIT | FIXED_INCOME / USA / intermediate IG | future LQD child |
| VCLT | FIXED_INCOME / USA / long IG | future LQD child |

### Batch 2: EM country satellites

Research and validate before inclusion:

```text
ECH Chile       EPHE Philippines     THD Thailand
EIDO Indonesia  EPU Peru             TUR Türkiye
ARGT Argentina
```

They are satellites, not mandatory constituents. Start the EM tree with the
existing large sleeves and add countries only after common-history validation.

### Mandatory insertion gate

For every ETF candidate:

1. Confirm exact Yahoo ticker and exchange.
2. Backfill adjusted-close history from the configured historical start.
3. Verify latest-date freshness, gaps, zero/negative prices, and continuity.
4. Test common history with B0, father proxy, siblings, and terminal assets.
5. Review liquidity, delisting risk, and known methodology changes.
6. Add asset class, area, category, sector/theme, and notes to the taxonomy.
7. Run listing identity and ETF-classification backfills.
8. Rebuild coverage and reject instruments that shorten the agreed ten-year
   common sample.

## Construction sequence

### Phase A: data foundation

1. Add and backfill Batch 1.
2. Create and save a reproducible “10Y eligible universe” coverage report.
3. Correct taxonomy and quality issues before building V2 configurations.
4. Re-run the topology gate: every configured non-root node must have at
   least two non-proxy candidates; every child proxy must be distinct from all
   candidates in that child; every configured ticker must pass the common
   ten-year history check.

### Phase B: minimal strategic tree

Build only root-level sleeves:

```text
ACWI equity, AGG USD bonds, BNDX international bonds, DBC commodities, QAI alternatives
```

Validate point estimates, all V2 modes, and causal monthly walk-forward.

### Phase C: first depth expansion

Add global equity geography, USD duration/credit branches, international
government/IG/HY/EM debt, commodity families, and the alternatives children.

### Phase D: geographic depth

Add developed countries and approved EM satellites. Review target and TEV
feasibility at each new depth before adding another branch.

### Phase E: research variants

Create controlled sibling configurations:

```text
Global-Multi-Asset-Country
Global-Multi-Asset-US-Sectors
Global-Multi-Asset-US-Styles
Global-Multi-Asset-Europe-Sectors
Global-Multi-Asset-EM-Country-Satellites
```

Keep benchmark, calendar, zero transaction costs, zero funding cost, zero
risk-free rate, training size, and universe-quality policy identical across
comparisons.

## Validation and production

Each tree must pass schema validation, data/common-history validation, a
Forward-plus-Backward point estimate, local audit checks, and a causal monthly
walk-forward before it becomes a production tree.

Production uses only a current point estimate after the daily data refresh. It
stores configuration hash, data fingerprint, weights, audit summary, and report
reference. A ten-year backtest, scientific study, and audit ZIP remain separate
research or on-demand jobs.

## Open decisions

1. Confirm B0 as `ACWI 60 / AGG 20 / BNDX 20` or select another global 60/40.
2. Confirm country/region as the first equity decomposition.
3. Decide whether TEV applies to every node or only to strategic sleeves.
4. Decide whether `IGF` and `VNQI` become a later sixth root sleeve named
   “Listed Real Assets”.
