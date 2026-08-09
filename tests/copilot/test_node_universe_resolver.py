"""docs/node-copilot-operational-plan.md §13 Fase 2 exit criterion: a view
cannot be validated at the wrong economic level, and NodeContext resolves
correctly on the pillar-level fixture, not just the multi-level one -- this
is where Fase 0's manually-built fixtures.py contexts get *proven* against
the real resolver, not just asserted by hand."""

from __future__ import annotations

from uuid import uuid4

import pytest
from fixtures import (
    MULTI_LEVEL_CONFIG,
    PILLAR_LEVEL_CONFIG,
    multi_level_equity_node_context,
    pillar_level_equity_node_context,
)

from lazyportfolio.copilot.contracts import ProposedView
from lazyportfolio.copilot.node_universe import (
    NodeNotFoundError,
    apply_views_to_config,
    resolve_node_context,
    validate_view_set,
)


def test_resolve_node_context_matches_the_manually_built_multi_level_fixture() -> None:
    tree_id = uuid4()
    revision_id = uuid4()
    resolved = resolve_node_context(
        MULTI_LEVEL_CONFIG,  # type: ignore[arg-type]
        "equity",
        mode="forward_backward",
        tree_id=tree_id,
        revision_id=revision_id,
    )
    expected = multi_level_equity_node_context()
    assert resolved.node_id == expected.node_id
    assert resolved.allowed_view_instruments == expected.allowed_view_instruments
    assert resolved.direct_instruments == expected.direct_instruments
    assert resolved.child_node_ids == expected.child_node_ids
    assert resolved.parent_node_id == expected.parent_node_id
    assert resolved.parent_candidate_instrument == expected.parent_candidate_instrument
    assert len(resolved.current_views) == 1
    assert resolved.current_views[0]["instruments"] == {"ticker:VTI": 1.0, "ticker:VXUS": -1.0}


def test_resolve_node_context_matches_the_manually_built_pillar_level_fixture() -> None:
    """The exit criterion itself: a first-level pillar node resolves with
    the identical field semantics as a deeper interior node -- no special
    casing by depth."""

    resolved = resolve_node_context(
        PILLAR_LEVEL_CONFIG,  # type: ignore[arg-type]
        "equity",
        mode="forward_backward",
        tree_id=uuid4(),
        revision_id=uuid4(),
    )
    expected = pillar_level_equity_node_context()
    assert resolved.allowed_view_instruments == expected.allowed_view_instruments
    assert resolved.direct_instruments == expected.direct_instruments
    assert resolved.child_node_ids == expected.child_node_ids
    assert resolved.parent_node_id == expected.parent_node_id
    assert resolved.current_views == []


def test_resolve_node_context_raises_for_unknown_node_id() -> None:
    with pytest.raises(NodeNotFoundError):
        resolve_node_context(
            MULTI_LEVEL_CONFIG,  # type: ignore[arg-type]
            "does-not-exist",
            mode="forward_backward",
            tree_id=uuid4(),
            revision_id=uuid4(),
        )


def _view(**overrides: object) -> ProposedView:
    defaults: dict[str, object] = {
        "instruments": {"ticker:VTI": 1.0, "ticker:VXUS": -1.0},
        "expected_return": 0.02,
        "confidence": 0.6,
        "rationale": "test",
    }
    defaults.update(overrides)
    return ProposedView(**defaults)  # type: ignore[arg-type]


def test_validate_view_set_accepts_a_view_inside_the_universe() -> None:
    result = validate_view_set(
        MULTI_LEVEL_CONFIG,  # type: ignore[arg-type]
        "equity",
        [_view()],
        mode="forward_backward",
    )
    assert result.valid
    assert result.errors == []


def test_validate_view_set_empty_list_is_trivially_valid() -> None:
    result = validate_view_set(
        MULTI_LEVEL_CONFIG,  # type: ignore[arg-type]
        "equity",
        [],
        mode="forward_backward",
    )
    assert result.valid


def test_validate_view_set_rejects_an_instrument_outside_the_universe() -> None:
    result = validate_view_set(
        MULTI_LEVEL_CONFIG,  # type: ignore[arg-type]
        "equity",
        [_view(instruments={"ticker:AGG": 1.0})],  # AGG is the bond pillar's proxy, not equity's
        mode="forward_backward",
    )
    assert not result.valid
    assert any(e.code == "instrument_outside_universe" for e in result.errors)


def test_validate_view_set_rejects_a_financing_instrument() -> None:
    result = validate_view_set(
        PILLAR_LEVEL_CONFIG,  # type: ignore[arg-type]
        "equity",
        [_view(instruments={"cash:rf": 1.0})],
        mode="forward_backward",
    )
    assert not result.valid
    assert any(e.code == "financing_instrument_forbidden" for e in result.errors)


def test_validate_view_set_rejects_duplicate_views() -> None:
    view = _view()
    result = validate_view_set(
        MULTI_LEVEL_CONFIG,  # type: ignore[arg-type]
        "equity",
        [view, view],
        mode="forward_backward",
    )
    assert not result.valid
    assert any(e.code == "duplicate_view" for e in result.errors)


def test_validate_view_set_rejects_opposite_picks_on_the_same_instruments() -> None:
    result = validate_view_set(
        MULTI_LEVEL_CONFIG,  # type: ignore[arg-type]
        "equity",
        [
            _view(instruments={"ticker:VTI": 1.0, "ticker:VXUS": -1.0}),
            _view(instruments={"ticker:VTI": -1.0, "ticker:VXUS": 1.0}),
        ],
        mode="forward_backward",
    )
    assert not result.valid
    assert any(e.code == "opposite_pick_same_horizon" for e in result.errors)


def test_validate_view_set_warns_on_extreme_expected_return_without_invalidating() -> None:
    result = validate_view_set(
        MULTI_LEVEL_CONFIG,  # type: ignore[arg-type]
        "equity",
        [_view(expected_return=1.5)],
        mode="forward_backward",
    )
    assert result.valid  # a warning, not an error
    assert any(w.code == "extreme_expected_return" for w in result.warnings)


def test_validate_view_set_flags_no_effect_on_weights_for_min_risk_prior_risk() -> None:
    """A node with objective=min_risk and view_covariance_policy=prior_risk
    (the config default) never lets a view move the solved weights -- must
    warn, per plan §6.1, not silently accept a proposal that does nothing."""

    result = validate_view_set(
        PILLAR_LEVEL_CONFIG,  # type: ignore[arg-type]
        "equity",
        [_view(instruments={"ticker:VTI": 1.0})],
        mode="forward_backward",
    )
    assert any(w.code == "no_effect_on_weights" for w in result.warnings)


def test_apply_views_to_config_only_touches_the_target_nodes_views() -> None:
    patched = apply_views_to_config(MULTI_LEVEL_CONFIG, "equity", [_view(expected_return=0.05)])  # type: ignore[arg-type]
    equity = next(n for n in patched["nodes"] if n["id"] == "equity")
    assert equity["constraints"]["views"][0]["expected_return"] == 0.05
    # Every other node is untouched.
    for node in patched["nodes"]:
        if node["id"] != "equity":
            original = next(n for n in MULTI_LEVEL_CONFIG["nodes"] if n["id"] == node["id"])  # type: ignore[index]
            assert node == original


def test_apply_views_to_config_raises_for_unknown_node() -> None:
    with pytest.raises(NodeNotFoundError):
        apply_views_to_config(MULTI_LEVEL_CONFIG, "does-not-exist", [])  # type: ignore[arg-type]
