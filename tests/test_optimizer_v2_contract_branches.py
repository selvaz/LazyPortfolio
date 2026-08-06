"""Phase 2 contract tests: component identity, reference/constraint policy.

Covers the typed vocabulary introduced in
`docs/optimizer-v2-clean-engine-follow-up.md`: rejection of reserved
iterative-only reference names, independence of `mean_reference_kind` from
the risk-side reference axes, and the new constraint policy fields.
"""

from __future__ import annotations

import pytest

from lazyportfolio.v2.contracts import (
    RESERVED_ITERATIVE_REFERENCES,
    V2Component,
    V2SolveContext,
)
from lazyportfolio.v2.model import V2Model
from lazyportfolio.v2.validation import normalize_config


def _base_config(**node_constraints: object) -> dict[str, object]:
    return {
        "root_id": "root",
        "currency": "USD",
        "nodes": [
            {
                "id": "root",
                "instruments": ["acwi", "agg"],
                "children": [],
                "goal": {"objective": "min_risk"},
                "constraints": dict(node_constraints),
            }
        ],
        "backtest": {
            "benchmark": {"weights": {"acwi": "0.7", "agg": "0.3"}},
        },
    }


@pytest.mark.parametrize(
    "field_name",
    ["volatility_reference", "max_volatility_reference", "tracking_error_reference"],
)
@pytest.mark.parametrize("reserved", sorted(RESERVED_ITERATIVE_REFERENCES))
def test_iterative_references_rejected_on_risk_axes(field_name: str, reserved: str) -> None:
    config = _base_config(**{field_name: reserved})
    with pytest.raises(ValueError, match="reserved for a future iterative"):
        normalize_config(config)


@pytest.mark.parametrize("reserved", sorted(RESERVED_ITERATIVE_REFERENCES))
def test_iterative_references_rejected_on_mean_reference_kind(reserved: str) -> None:
    config = _base_config(mean_reference_kind=reserved)
    with pytest.raises(ValueError, match="reserved for a future iterative"):
        normalize_config(config)


def test_legacy_reference_strings_still_validate() -> None:
    for value in ("none", "manual", "declared", "benchmark", "father_proxy", "father"):
        config = _base_config(volatility_reference=value)
        normalized = normalize_config(config)
        resolved = normalized["nodes"][0]["constraints"]["volatility_reference"]
        assert resolved == ("father_proxy" if value == "father" else value)


def test_forward_root_reference_string_validates() -> None:
    config = _base_config(volatility_reference="forward_root_reference")
    normalized = normalize_config(config)
    assert (
        normalized["nodes"][0]["constraints"]["volatility_reference"]
        == "forward_root_reference"
    )


def test_v2component_and_solvecontext_round_trip() -> None:
    direct = V2Component(id="direct:acwi", kind="direct", raw_series_key="ticker:ACWI")
    child = V2Component(
        id="child:equity",
        kind="child",
        raw_series_key="ticker:EQUITY",
        child_id="equity",
    )
    context = V2SolveContext(
        pass_kind="backward",
        components={direct.id: direct, child.id: child},
        candidate_series_by_component={direct.id: "raw-acwi", child.id: "synthetic-equity"},
        raw_proxy_series_by_component={direct.id: "raw-acwi", child.id: "raw-equity"},
        synthetic_series_by_component={child.id: "synthetic-equity"},
        component_to_solver_column={direct.id: "ticker:ACWI", child.id: "ticker:EQUITY_SYNTH"},
        solver_column_to_component={
            "ticker:ACWI": direct.id,
            "ticker:EQUITY_SYNTH": child.id,
        },
    )
    for component_id, column in context.component_to_solver_column.items():
        assert context.solver_column_to_component[column] == component_id
    assert context.components[child.id].kind == "child"
    assert context.components[direct.id].kind == "direct"
    # Backward candidate for the child component is its synthetic series, not raw.
    assert (
        context.candidate_series_by_component[child.id]
        == context.synthetic_series_by_component[child.id]
    )


def test_constraint_policy_fields_default_and_validate() -> None:
    config = _base_config()
    normalized = normalize_config(config)
    constraints = normalized["nodes"][0]["constraints"]
    assert constraints["tracking_error_policy"] == "hard_fail"
    assert constraints["volatility_target_policy"] == "hard_fail"
    assert constraints["volatility_cap_policy"] == "hard_fail"

    for key in ("tracking_error_policy", "volatility_target_policy"):
        normalized = normalize_config(_base_config(**{key: "nearest_feasible"}))
        assert normalized["nodes"][0]["constraints"][key] == "nearest_feasible"

    with pytest.raises(ValueError, match="unsupported tracking_error_policy"):
        normalize_config(_base_config(tracking_error_policy="soft"))
    with pytest.raises(ValueError, match="volatility_cap_policy must be 'hard_fail'"):
        normalize_config(_base_config(volatility_cap_policy="nearest_feasible"))


def test_mean_reference_kind_is_independent_of_risk_axes() -> None:
    config = _base_config(volatility_reference="father_proxy")
    normalized = normalize_config(config)
    constraints = normalized["nodes"][0]["constraints"]
    assert constraints["volatility_reference"] == "father_proxy"
    assert constraints["mean_reference_kind"] == "none"
    assert constraints.get("mean_reference_weights") is None


def test_mean_reference_local_weights_requires_full_valid_mapping() -> None:
    with pytest.raises(ValueError, match="requires a non-empty mean_reference_weights"):
        normalize_config(_base_config(mean_reference_kind="local_weights"))

    with pytest.raises(ValueError, match="must sum to one"):
        normalize_config(
            _base_config(
                mean_reference_kind="local_weights",
                mean_reference_weights={"acwi": 0.5},
            )
        )

    normalized = normalize_config(
        _base_config(
            mean_reference_kind="local_weights",
            mean_reference_weights={"acwi": 0.7, "agg": 0.3},
        )
    )
    constraints = normalized["nodes"][0]["constraints"]
    assert constraints["mean_reference_weights"] == {"acwi": 0.7, "agg": 0.3}

    with pytest.raises(ValueError, match="requires mean_reference_kind='local_weights'"):
        normalize_config(_base_config(mean_reference_weights={"acwi": 1.0}))


def test_mean_reference_kind_rejects_unsupported_value() -> None:
    with pytest.raises(ValueError, match="unsupported mean_reference_kind"):
        normalize_config(_base_config(mean_reference_kind="mystery"))


def test_model_from_config_parses_mean_reference_and_constraint_policies() -> None:
    config = _base_config(
        mean_reference_kind="local_weights",
        mean_reference_weights={"acwi": 0.7, "agg": 0.3},
        tracking_error_policy="nearest_feasible",
    )
    model = V2Model.from_config(config)
    constraints = model.root.constraints
    assert constraints.mean_reference_kind == "local_weights"
    assert constraints.mean_reference_weights == {
        "ticker:ACWI": 0.7,
        "ticker:AGG": 0.3,
    }
    assert constraints.tracking_error_policy == "nearest_feasible"
    assert constraints.volatility_target_policy == "hard_fail"
    assert constraints.volatility_cap_policy == "hard_fail"


def test_mean_reference_kind_rejects_father_proxy() -> None:
    """father_proxy is deliberately not offered as a mean_reference_kind: a
    node's own proxy is what its *parent* sees it as, never one of the
    node's own candidate columns, so it has no coherent meaning here (it
    remains valid for the risk-side volatility/TEV references).
    """

    with pytest.raises(ValueError, match="unsupported mean_reference_kind"):
        normalize_config(_base_config(mean_reference_kind="father_proxy"))


def test_duplicate_node_id_rejected() -> None:
    config = {
        "root_id": "root",
        "currency": "USD",
        "nodes": [
            {"id": "root", "instruments": ["acwi"], "children": ["dup"], "constraints": {}},
            {"id": "dup", "instruments": ["agg"], "children": [], "constraints": {}},
            {"id": "dup", "instruments": ["dbc"], "children": [], "constraints": {}},
        ],
        "backtest": {"benchmark": {"weights": {"acwi": "1.0"}}},
    }
    with pytest.raises(ValueError, match="duplicate node id"):
        normalize_config(config)


def test_duplicate_node_name_rejected() -> None:
    config = {
        "root_id": "root",
        "currency": "USD",
        "nodes": [
            {
                "id": "root", "name": "Root", "instruments": [],
                "children": ["a", "b"], "constraints": {},
            },
            {
                "id": "a", "name": "Sleeve", "instruments": ["acwi"],
                "children": [], "constraints": {},
            },
            {
                "id": "b", "name": "Sleeve", "instruments": ["agg"],
                "children": [], "constraints": {},
            },
        ],
        "backtest": {"benchmark": {"weights": {"acwi": "0.7", "agg": "0.3"}}},
    }
    with pytest.raises(ValueError, match="duplicate node name"):
        normalize_config(config)


def test_sibling_proxy_collision_rejected() -> None:
    config = {
        "root_id": "root",
        "currency": "USD",
        "nodes": [
            {
                "id": "root", "name": "Root", "instruments": [],
                "children": ["a", "b"], "constraints": {},
            },
            {
                "id": "a", "name": "Equity US", "proxy": "acwi",
                "instruments": ["spy"], "children": [], "constraints": {},
            },
            {
                "id": "b", "name": "Equity Tactical", "proxy": "acwi",
                "instruments": ["vt"], "children": [], "constraints": {},
            },
        ],
        "backtest": {"benchmark": {"weights": {"acwi": "1.0"}}},
    }
    with pytest.raises(ValueError, match="share the same proxy"):
        normalize_config(config)


def test_child_proxy_colliding_with_direct_instrument_rejected() -> None:
    config = {
        "root_id": "root",
        "currency": "USD",
        "nodes": [
            {
                "id": "root", "name": "Root", "instruments": ["acwi"],
                "children": ["a"], "constraints": {},
            },
            {
                "id": "a", "name": "Equity", "proxy": "acwi",
                "instruments": ["spy"], "children": [], "constraints": {},
            },
        ],
        "backtest": {"benchmark": {"weights": {"acwi": "1.0"}}},
    }
    with pytest.raises(ValueError, match="collides with a direct instrument"):
        normalize_config(config)


def test_cycle_rejected() -> None:
    config = {
        "root_id": "root",
        "currency": "USD",
        "nodes": [
            {"id": "root", "instruments": [], "children": ["a"], "constraints": {}},
            {
                "id": "a", "proxy": "acwi", "instruments": [],
                "children": ["root"], "constraints": {},
            },
        ],
        "backtest": {"benchmark": {"weights": {"acwi": "1.0"}}},
    }
    with pytest.raises(ValueError):
        normalize_config(config)


def test_multiple_parents_rejected() -> None:
    config = {
        "root_id": "root",
        "currency": "USD",
        "nodes": [
            {"id": "root", "instruments": [], "children": ["a", "b"], "constraints": {}},
            {
                "id": "a", "proxy": "acwi", "instruments": [],
                "children": ["shared"], "constraints": {},
            },
            {
                "id": "b", "proxy": "agg", "instruments": [],
                "children": ["shared"], "constraints": {},
            },
            {"id": "shared", "proxy": "dbc", "instruments": [], "children": [], "constraints": {}},
        ],
        "backtest": {"benchmark": {"weights": {"acwi": "0.5", "agg": "0.5"}}},
    }
    with pytest.raises(ValueError, match="exactly one parent"):
        normalize_config(config)


def test_empty_root_id_rejected() -> None:
    config = _base_config()
    config["root_id"] = ""
    # normalize_config's pre-existing root-lookup check fires first (no node
    # has id ""); _validate_tree_structure's own root_id-empty check is a
    # second, later gate for the case where it would otherwise match.
    with pytest.raises(ValueError, match="root node is missing"):
        normalize_config(config)


def test_empty_node_id_rejected() -> None:
    config = {
        "root_id": "root",
        "currency": "USD",
        "nodes": [
            {"id": "root", "instruments": ["acwi"], "children": ["a"], "constraints": {}},
            {"id": "", "proxy": "acwi", "instruments": [], "children": [], "constraints": {}},
        ],
        "backtest": {"benchmark": {"weights": {"acwi": "1.0"}}},
    }
    with pytest.raises(ValueError, match="missing a non-empty id"):
        normalize_config(config)


def test_duplicate_direct_instrument_rejected() -> None:
    config = {
        "root_id": "root",
        "currency": "USD",
        "nodes": [
            {
                "id": "root", "instruments": ["acwi", "acwi"],
                "children": [], "constraints": {},
            },
        ],
        "backtest": {"benchmark": {"weights": {"acwi": "1.0"}}},
    }
    with pytest.raises(ValueError, match="duplicate direct instrument"):
        normalize_config(config)


def test_node_unreachable_from_root_rejected() -> None:
    """A node declared in the tree but never referenced as anyone's child
    used to be silently ignored by the model builder (it took no part in
    any solve or audit, with no error at all) - the exact same class of
    silent-configuration-loss the other structural checks close.
    """

    config = {
        "root_id": "root",
        "currency": "USD",
        "nodes": [
            {"id": "root", "instruments": ["acwi"], "children": [], "constraints": {}},
            {
                "id": "orphan", "proxy": "gold", "instruments": ["gold"],
                "children": [], "constraints": {},
            },
        ],
        "backtest": {"benchmark": {"weights": {"acwi": "1.0"}}},
    }
    with pytest.raises(ValueError, match="not reachable from root"):
        normalize_config(config)


def test_leaf_nodes_without_children_are_unaffected() -> None:
    """Sanity guard: a node without children of its own (a normal leaf
    sleeve) must never be rejected by the structural checks above - only
    nodes that are not reachable from root at all are.
    """

    config = {
        "root_id": "root",
        "currency": "USD",
        "nodes": [
            {
                "id": "root", "instruments": [], "children": ["leaf"],
                "constraints": {},
            },
            {
                "id": "leaf", "proxy": "acwi", "instruments": ["spy", "vt"],
                "children": [], "constraints": {},
            },
        ],
        "backtest": {"benchmark": {"weights": {"acwi": "1.0"}}},
    }
    normalized = normalize_config(config)
    assert [item["id"] for item in normalized["nodes"]] == ["root", "leaf"]
