# Per-node financing invariants

This note is the implementation checklist for cash and leverage on hierarchical
nodes. The normative user-facing contract is in
[`hierarchical-v2.md`](hierarchical-v2.md).

## Mathematical model

For node `n`, risky weights `w_n` and local cash `c_n` satisfy:

```text
1' w_n + c_n = 1
w_n >= 0
1 - L_n <= c_n <= 1
```

`L_n >= 1` is the node leverage limit. Lending uses `c_n >= 0`; borrowing uses
`c_n <= 0`. Both regimes include `c_n = 0`, so financing is optional. The local
synthetic return is:

```text
r_n = w_n' r_risky + c_n * r_lend             when c_n >= 0
r_n = w_n' r_risky + c_n * r_borrow           when c_n <= 0
r_borrow = r_lend + spread_bps / 10000
```

The regimes are solved and audited independently. The feasible candidate with the
best configured economic objective is selected.

## Composition invariant

For parent weight `a_n`, every child terminal position is scaled by `a_n`:

```text
global risky exposure of n = a_n * sum(w_n)
global cash/borrowing of n = a_n * c_n
```

The parent consumes `r_n` as one synthetic asset. It does not add `c_n` to its own
cash variable. Flattening the tree therefore preserves:

```text
sum(global risky terminal weights) + sum(global financing weights) = 1
```

## Ledger identity

Root financing retains the compatibility names `cash:RF` and `cash:BORROW`.
Non-root instruments are qualified with the node id. A ledger contains at most one
lending or borrowing position for a given node and can contain financing positions
from multiple nodes without collisions.

## Provenance

Risk-free rate: node -> root -> `0.0`.

Borrow spread: node -> root/global -> `0.0` for a financing-enabled node. A direct
positive spread with no cash and no borrowing capability is rejected because the
parameter has no active economic regime.

Cash permission and leverage are local declarations; they are not inherited.
`max_leverage > 1` continues to enable financing implicitly for compatibility.

## Tree Studio controls

The node editor exposes `cash_enabled`, local `max_leverage`, local
`borrow_spread_bps`, and the node risk-free override. Cash and leverage remain
optional: enabling a regime only enlarges the feasible set. The editor disables
financing for HRP, distinguishes node overrides from root/global defaults, and
explains that a positive spread is unused while `max_leverage == 1`.

Estimate results display both local financing and parent-scaled global exposure,
including parameter provenance and aggregate risky/cash/net exposure. Root and
non-root financing instruments remain distinct in the displayed terminal weights.

Static UI tests verify the financing controls and serialization contract, while a
Node.js syntax check validates the complete embedded JavaScript program.
