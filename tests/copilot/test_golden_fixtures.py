"""Fase 0 exit criterion (docs/adr/0001-node-copilot-architecture.md
Decision 3 / plan §3.4 point 5): a NodeContext built for a first-level
pillar node must validate against the same contract, with the same field
semantics, as one built for a deeper interior node. The automated resolver
that will build these from a live V2Model is Fase 2 (NodeUniverseResolver)
-- this only proves the contract and the fixtures are coherent with each
other and with V2Model.from_config.
"""

from __future__ import annotations

from fixtures import (
    MULTI_LEVEL_CONFIG,
    PILLAR_LEVEL_CONFIG,
    multi_level_equity_node_context,
    pillar_level_equity_node_context,
)

from lazyportfolio.v2.model import V2Model


def test_multi_level_config_parses_through_v2model() -> None:
    model = V2Model.from_config(MULTI_LEVEL_CONFIG)
    assert model.root.id == "root"
    equity = next(node for node in model.root.children if node.id == "equity")
    assert {child.id for child in equity.children} == {"equity_us", "equity_intl"}
    assert len(equity.constraints.views) == 1
    assert equity.constraints.views[0].instruments == {"ticker:VTI": 1.0, "ticker:VXUS": -1.0}


def test_pillar_level_config_parses_through_v2model() -> None:
    model = V2Model.from_config(PILLAR_LEVEL_CONFIG)
    assert model.root.id == "root"
    assert {child.id for child in model.root.children} == {
        "equity",
        "bond",
        "commodity",
        "cash_equiv",
    }
    equity = next(node for node in model.root.children if node.id == "equity")
    assert equity.children == []
    assert equity.instruments == ["ticker:VTI"]


def test_interior_node_context_validates_against_the_contract() -> None:
    context = multi_level_equity_node_context()
    assert context.node_id == "equity"
    assert context.child_node_ids == ["equity_us", "equity_intl"]
    assert context.allowed_view_instruments == ["ticker:VTI", "ticker:VXUS"]
    assert context.direct_instruments == []
    assert len(context.current_views) == 1


def test_pillar_node_context_validates_against_the_same_contract() -> None:
    context = pillar_level_equity_node_context()
    assert context.node_id == "equity"
    assert context.child_node_ids == []
    assert context.allowed_view_instruments == ["ticker:VTI"]
    assert context.direct_instruments == ["ticker:VTI"]
    assert context.current_views == []


def test_pillar_and_interior_node_contexts_share_the_same_field_shape() -> None:
    """The exit criterion itself: no field is populated for one and left
    structurally meaningless for the other -- both are the same NodeContext
    shape, just with different depths in the tree."""

    interior = multi_level_equity_node_context()
    pillar = pillar_level_equity_node_context()
    assert interior.model_fields_set == pillar.model_fields_set
    assert interior.schema_version == pillar.schema_version == "1.0"
    assert interior.parent_node_id == pillar.parent_node_id == "root"
    # An interior node's own candidate columns come from its children's
    # proxies; a pillar's come from its own direct instrument. Different
    # *source*, identical *contract field* (allowed_view_instruments) --
    # this is the point being proven, not a coincidence.
    assert len(interior.allowed_view_instruments) >= 1
    assert len(pillar.allowed_view_instruments) >= 1
