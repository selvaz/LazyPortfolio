"""Golden fixtures for the Node Advisor's Fase 0 contracts.

Two V2 tree configurations, both in the plain-dict shape
``lazyportfolio.v2.model.V2Model.from_config`` already accepts (built from
``normalize_config``/``ticker`` conventions, not a new format):

* :data:`MULTI_LEVEL_CONFIG` -- root -> two pillars -> leaves, with a
  Black-Litterman view on the "equity" pillar (a Node Advisor conversation
  on a nested node).
* :data:`PILLAR_LEVEL_CONFIG` -- root -> four direct pillars
  (equity/bond/commodity/cash-equivalent), no views, matching the top-down
  ETF batch producer's granularity (``private workflow specification``).

For each, a :func:`NodeContext` is built *manually* here (not through an
automated resolver -- that is ``NodeUniverseResolver``, Fase 2) so Fase 0's
tests can prove the contract's fields make sense against both tree shapes,
in particular that a first-level pillar node has no different semantics
than a deeper interior node (docs/adr/0001-node-advisor-architecture.md,
Decision 3 / plan §3.4 point 5).
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from uuid import UUID

from lazyportfolio.advisor.contracts import NodeComponent, NodeContext, SnapshotDescriptor

#: Fixed, arbitrary UUIDs -- reproducible across test runs, not meaningful
#: beyond "this is tree/revision X in this fixture module".
MULTI_LEVEL_TREE_ID = UUID("11111111-1111-4111-8111-111111111111")
MULTI_LEVEL_REVISION_ID = UUID("22222222-2222-4222-8222-222222222222")
PILLAR_LEVEL_TREE_ID = UUID("33333333-3333-4333-8333-333333333333")
PILLAR_LEVEL_REVISION_ID = UUID("44444444-4444-4444-8444-444444444444")

MULTI_LEVEL_CONFIG: dict[str, object] = {
    "root_id": "root",
    "currency": "USD",
    "nodes": [
        {
            "id": "root",
            "name": "Root",
            "instruments": [],
            "children": ["equity", "bond"],
            "goal": {"objective": "min_risk"},
            "constraints": {},
        },
        {
            "id": "equity",
            "name": "Equity",
            "instruments": [],
            "children": ["equity_us", "equity_intl"],
            "proxy": "ticker:ACWI",
            "goal": {"objective": "min_risk"},
            "constraints": {
                "views": [
                    {
                        "instruments": {"ticker:VTI": 1.0, "ticker:VXUS": -1.0},
                        "expected_return": 0.02,
                        "confidence": 0.6,
                        "source": "node-advisor",
                    }
                ]
            },
        },
        {
            "id": "equity_us",
            "name": "Equity US",
            "instruments": ["ticker:VTI"],
            "children": [],
            "proxy": "ticker:VTI",
            "goal": {"objective": "min_risk"},
            "constraints": {},
        },
        {
            "id": "equity_intl",
            "name": "Equity Intl",
            "instruments": ["ticker:VXUS"],
            "children": [],
            "proxy": "ticker:VXUS",
            "goal": {"objective": "min_risk"},
            "constraints": {},
        },
        {
            "id": "bond",
            "name": "Bond",
            "instruments": ["ticker:AGG"],
            "children": [],
            "proxy": "ticker:AGG",
            "goal": {"objective": "min_risk"},
            "constraints": {},
        },
    ],
    "backtest": {
        "benchmark": {
            "name": "B0",
            "weights": {"ticker:VTI": 0.4, "ticker:VXUS": 0.2, "ticker:AGG": 0.4},
        }
    },
}

#: Root -> four direct pillars, no grandchildren -- coherent with the
#: top-down ETF batch producer's asset-class granularity. "cash" is represented
#: as a short-duration T-Bill ETF, not the special ``cash:rf``/``cash:borrow``
#: financing labels (those are a different, node-local financing axis --
#: see lazyportfolio.v2.validation._is_financing_view_label).
PILLAR_LEVEL_CONFIG: dict[str, object] = {
    "root_id": "root",
    "currency": "USD",
    "nodes": [
        {
            "id": "root",
            "name": "Root",
            "instruments": [],
            "children": ["equity", "bond", "commodity", "cash_equiv"],
            "goal": {"objective": "min_risk"},
            "constraints": {},
        },
        {
            "id": "equity",
            "name": "Equity",
            "instruments": ["ticker:VTI"],
            "children": [],
            "proxy": "ticker:VTI",
            "goal": {"objective": "min_risk"},
            "constraints": {},
        },
        {
            "id": "bond",
            "name": "Bond",
            "instruments": ["ticker:AGG"],
            "children": [],
            "proxy": "ticker:AGG",
            "goal": {"objective": "min_risk"},
            "constraints": {},
        },
        {
            "id": "commodity",
            "name": "Commodity",
            "instruments": ["ticker:GLD"],
            "children": [],
            "proxy": "ticker:GLD",
            "goal": {"objective": "min_risk"},
            "constraints": {},
        },
        {
            "id": "cash_equiv",
            "name": "Cash Equivalent",
            "instruments": ["ticker:SHV"],
            "children": [],
            "proxy": "ticker:SHV",
            "goal": {"objective": "min_risk"},
            "constraints": {},
        },
    ],
    "backtest": {
        "benchmark": {
            "name": "B0",
            "weights": {
                "ticker:VTI": 0.4,
                "ticker:AGG": 0.3,
                "ticker:GLD": 0.1,
                "ticker:SHV": 0.2,
            },
        }
    },
}

_FIXED_AS_OF = date(2026, 8, 9)
_FIXED_RETRIEVED_AT = datetime(2026, 8, 9, tzinfo=UTC)


def _snapshot(universe: list[str]) -> SnapshotDescriptor:
    return SnapshotDescriptor(
        schema_version="1.0",
        source="market-data-hub",
        database_identity="fixture",
        universe=universe,
        start=date(2016, 1, 1),
        end=_FIXED_AS_OF,
        data_as_of=_FIXED_AS_OF,
        field="close",
        currency="USD",
        frequency="D",
        coverage=[],
        source_run_ids=[],
        fingerprint="sha256:fixture",
    )


def multi_level_equity_node_context() -> NodeContext:
    """The "equity" pillar in :data:`MULTI_LEVEL_CONFIG` -- an interior node
    (has children, has a view, is itself a child of root)."""

    return NodeContext(
        schema_version="1.0",
        tree_id=MULTI_LEVEL_TREE_ID,
        revision_id=MULTI_LEVEL_REVISION_ID,
        node_id="equity",
        node_name="Equity",
        objective="min_risk",
        mode="forward_backward",
        solved_components=[
            NodeComponent(
                component_id="equity_us",
                kind="child",
                label="Equity US",
                candidate_instrument="ticker:VTI",
                child_node_id="equity_us",
            ),
            NodeComponent(
                component_id="equity_intl",
                kind="child",
                label="Equity Intl",
                candidate_instrument="ticker:VXUS",
                child_node_id="equity_intl",
            ),
        ],
        allowed_view_instruments=["ticker:VTI", "ticker:VXUS"],
        direct_instruments=[],
        child_node_ids=["equity_us", "equity_intl"],
        parent_node_id="root",
        parent_candidate_instrument="ticker:ACWI",
        constraints={},
        current_views=[
            {
                "instruments": {"ticker:VTI": 1.0, "ticker:VXUS": -1.0},
                "expected_return": 0.02,
                "confidence": 0.6,
                "source": "node-advisor",
            }
        ],
        snapshot=_snapshot(["ticker:VTI", "ticker:VXUS"]),
        recent_run=None,
    )


def pillar_level_equity_node_context() -> NodeContext:
    """The "equity" pillar in :data:`PILLAR_LEVEL_CONFIG` -- a first-level
    node directly under root, no children, no view. Structurally the same
    contract shape as :func:`multi_level_equity_node_context`, just with
    ``direct_instruments`` populated instead of ``solved_components``/
    ``child_node_ids`` -- proving the contract does not special-case depth.
    """

    return NodeContext(
        schema_version="1.0",
        tree_id=PILLAR_LEVEL_TREE_ID,
        revision_id=PILLAR_LEVEL_REVISION_ID,
        node_id="equity",
        node_name="Equity",
        objective="min_risk",
        mode="forward_backward",
        solved_components=[
            NodeComponent(
                component_id="equity",
                kind="direct",
                label="Equity",
                candidate_instrument="ticker:VTI",
                child_node_id=None,
            ),
        ],
        allowed_view_instruments=["ticker:VTI"],
        direct_instruments=["ticker:VTI"],
        child_node_ids=[],
        parent_node_id="root",
        parent_candidate_instrument="ticker:VTI",
        constraints={},
        current_views=[],
        snapshot=_snapshot(["ticker:VTI"]),
        recent_run=None,
    )


__all__ = [
    "MULTI_LEVEL_CONFIG",
    "MULTI_LEVEL_REVISION_ID",
    "MULTI_LEVEL_TREE_ID",
    "PILLAR_LEVEL_CONFIG",
    "PILLAR_LEVEL_REVISION_ID",
    "PILLAR_LEVEL_TREE_ID",
    "multi_level_equity_node_context",
    "pillar_level_equity_node_context",
]
