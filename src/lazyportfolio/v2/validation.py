"""Validation and migration for the canonical V2 configuration contract."""

from __future__ import annotations

from copy import deepcopy
from math import isfinite
from typing import Any

from lazyportfolio.v2.contracts import (
    RESERVED_ITERATIVE_REFERENCES,
    V2OptimizationError,
)

SUPPORTED_COVARIANCE_ESTIMATORS = {"shrunk_fixed", "ledoit_wolf"}
SUPPORTED_VIEW_POLICIES = {"prior_risk", "posterior_all"}
SUPPORTED_TARGET_MODES = {"exact", "cap", "at_most"}
RECOGNIZED_OBJECTIVES = {"min_risk", "max_return", "max_ratio", "max_utility", "hrp"}
#: "father_proxy" is deliberately not offered as a mean_reference_kind: a
#: node's own proxy is what its *parent* sees it as, never one of the node's
#: own candidate columns, so "100% weight on my own father proxy" has no
#: coherent meaning for that node's own equilibrium mean estimation.
SUPPORTED_MEAN_REFERENCE_KINDS = {"none", "benchmark", "local_weights"}
SUPPORTED_CONSTRAINT_POLICIES = {"hard_fail", "nearest_feasible"}
#: The only reference currencies lazyportfolio.fx can convert between today
#: (EURUSD=X/GBPUSD=X/USDJPY=X give full USD-pivoted coverage of exactly
#: these four -- see lazyportfolio.fx's module docstring).
SUPPORTED_CURRENCIES = {"USD", "EUR", "GBP", "JPY"}


def finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def optional_number(mapping: dict[str, Any], key: str, label: str) -> float | None:
    value = mapping.get(key)
    if value in (None, ""):
        return None
    result = finite_float(value, label)
    mapping[key] = result
    return result


def boolean(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, "", 0, 0.0, "0", "false", "False", "no", "off"):
        return False
    if value in (1, 1.0, "1", "true", "True", "yes", "on"):
        return True
    raise ValueError(f"{label} must be boolean")


def setting_source(node_value: float | None, root_value: float | None) -> str:
    if node_value is not None:
        return "node"
    if root_value is not None:
        return "root"
    return "hard_default"


def _clean_weight_mapping(
    constraints: dict[str, Any],
    key: str,
    node_id: str,
) -> None:
    raw = constraints.get(key) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"node {node_id}: {key} must be an object")
    cleaned: dict[str, float] = {}
    for instrument, value in raw.items():
        if value in (None, ""):
            continue
        parsed = finite_float(value, f"node {node_id} {key}[{instrument!r}]")
        if not 0.0 <= parsed <= 1.0:
            raise ValueError(
                f"node {node_id}: {key}[{instrument!r}] must be in [0, 1]"
            )
        cleaned[str(instrument)] = parsed
    constraints[key] = cleaned


def _normalize_reference(value: Any) -> str:
    reference = str(value or "none")
    return "father_proxy" if reference == "father" else reference


def _reject_reserved_iterative_reference(value: str, node_id: str, field_name: str) -> None:
    """Fail loudly on a reference reserved for a future iterative mode.

    ``current_parent_synthetic``/``current_root_synthetic`` are schema values
    reserved for a not-yet-implemented iterative hierarchy mode. They must
    never be silently accepted or coerced in the standard ``flat``,
    ``forward`` or ``forward_backward`` modes this engine supports.
    """

    if value in RESERVED_ITERATIVE_REFERENCES:
        raise ValueError(
            f"node {node_id}: {field_name}={value!r} is reserved for a future "
            "iterative hierarchy mode and is not supported in flat, forward or "
            "forward_backward mode"
        )


def _is_financing_view_label(value: Any) -> bool:
    label = str(value).strip().lower()
    return label.startswith("cash:rf") or label.startswith("cash:borrow")


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a validated copy of a V2 config, including legacy Studio migration."""

    normalized = deepcopy(config)
    nodes = normalized.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("nodes must be a list")
    if any(not isinstance(item, dict) for item in nodes):
        raise ValueError("each node must be an object")

    root_id = str(normalized.get("root_id"))
    root = next((item for item in nodes if str(item.get("id")) == root_id), None)
    if root is None:
        raise ValueError("root node is missing")

    currency_raw = normalized.get("currency")
    if not isinstance(currency_raw, str) or not currency_raw.strip():
        raise ValueError(
            "currency is required: every tree must declare a portfolio "
            "reference currency (one of USD, EUR, GBP, JPY)"
        )
    currency = currency_raw.strip().upper()
    if currency not in SUPPORTED_CURRENCIES:
        raise ValueError(
            f"currency must be one of {sorted(SUPPORTED_CURRENCIES)}, got {currency_raw!r}"
        )
    normalized["currency"] = currency

    data = normalized.get("data")
    if data is None:
        data = {}
        normalized["data"] = data
    if not isinstance(data, dict):
        raise ValueError("data must be an object")

    root_constraints = root.get("constraints") or {}
    if not isinstance(root_constraints, dict):
        raise ValueError(f"node {root_id}: constraints must be an object")
    root["constraints"] = root_constraints

    legacy_rate_raw = data.get("risk_free_annual")
    if legacy_rate_raw not in (None, ""):
        legacy_rate = finite_float(legacy_rate_raw, "data.risk_free_annual")
        declared = root_constraints.get("risk_free_rate")
        if declared not in (None, ""):
            declared_rate = finite_float(declared, "root risk_free_rate")
            if abs(declared_rate - legacy_rate) > 1e-12:
                raise ValueError(
                    "conflicting risk-free rates: data.risk_free_annual and "
                    "root.constraints.risk_free_rate must agree"
                )
            root_constraints["risk_free_rate"] = declared_rate
        else:
            root_constraints["risk_free_rate"] = legacy_rate

    global_spread_raw = data.get("borrow_spread_bps")
    global_spread_declared = global_spread_raw not in (None, "")
    global_spread = (
        finite_float(global_spread_raw, "data.borrow_spread_bps")
        if global_spread_declared
        else 0.0
    )
    if global_spread < 0.0:
        raise ValueError("data.borrow_spread_bps cannot be negative")

    root_spread_declared = root_constraints.get("borrow_spread_bps") not in (None, "")
    if root_spread_declared:
        root_spread = finite_float(
            root_constraints["borrow_spread_bps"],
            f"node {root_id} borrow_spread_bps",
        )
        root_spread_source = "node"
    elif global_spread_declared:
        root_spread = global_spread
        root_spread_source = "root"
    else:
        root_spread = 0.0
        root_spread_source = "default"
    if root_spread < 0.0:
        raise ValueError(f"node {root_id}: borrow_spread_bps cannot be negative")

    financing_enabled_anywhere = False

    for raw_node in nodes:
        node_id = str(raw_node.get("id") or "<unknown>")
        objective = str((raw_node.get("goal") or {}).get("objective") or "min_risk")
        if objective not in RECOGNIZED_OBJECTIVES:
            raise ValueError(
                f"node {node_id}: unsupported objective {objective!r}; expected one of "
                f"{sorted(RECOGNIZED_OBJECTIVES)}"
            )

        constraints = raw_node.get("constraints") or {}
        if not isinstance(constraints, dict):
            raise ValueError(f"node {node_id}: constraints must be an object")
        raw_node["constraints"] = constraints

        _clean_weight_mapping(constraints, "min_weights", node_id)
        _clean_weight_mapping(constraints, "max_weights", node_id)
        minimums = constraints["min_weights"]
        maximums = constraints["max_weights"]
        for instrument in set(minimums) & set(maximums):
            if minimums[instrument] > maximums[instrument]:
                raise ValueError(
                    f"node {node_id}: min_weights[{instrument!r}] exceeds max_weights"
                )

        for key in ("maximum_turnover", "max_turnover"):
            turnover = constraints.pop(key, None)
            if turnover not in (None, ""):
                raise ValueError(
                    f"node {node_id}: {key} is unsupported until a turnover-aware solver exists"
                )

        covariance = str(constraints.get("covariance_estimator") or "shrunk_fixed")
        if covariance not in SUPPORTED_COVARIANCE_ESTIMATORS:
            raise ValueError(
                f"node {node_id}: unsupported covariance_estimator {covariance!r}; "
                f"expected one of {sorted(SUPPORTED_COVARIANCE_ESTIMATORS)}"
            )
        constraints["covariance_estimator"] = covariance

        view_policy = str(constraints.get("view_covariance_policy") or "prior_risk")
        if view_policy not in SUPPORTED_VIEW_POLICIES:
            raise ValueError(
                f"node {node_id}: unsupported view_covariance_policy {view_policy!r}"
            )
        constraints["view_covariance_policy"] = view_policy

        target_mode = str(constraints.get("volatility_target_mode") or "exact")
        if target_mode not in SUPPORTED_TARGET_MODES:
            raise ValueError(
                f"node {node_id}: unsupported volatility_target_mode {target_mode!r}"
            )
        if target_mode == "at_most":
            target_mode = "cap"
        constraints["volatility_target_mode"] = target_mode

        for key in (
            "volatility_reference",
            "max_volatility_reference",
            "tracking_error_reference",
        ):
            if key in constraints:
                constraints[key] = _normalize_reference(constraints[key])
                _reject_reserved_iterative_reference(constraints[key], node_id, key)

        mean_reference_kind = str(constraints.get("mean_reference_kind") or "none")
        _reject_reserved_iterative_reference(
            mean_reference_kind, node_id, "mean_reference_kind"
        )
        if mean_reference_kind not in SUPPORTED_MEAN_REFERENCE_KINDS:
            raise ValueError(
                f"node {node_id}: unsupported mean_reference_kind "
                f"{mean_reference_kind!r}; expected one of "
                f"{sorted(SUPPORTED_MEAN_REFERENCE_KINDS)}"
            )
        constraints["mean_reference_kind"] = mean_reference_kind
        mean_reference_weights = constraints.get("mean_reference_weights")
        if mean_reference_kind == "local_weights":
            if not isinstance(mean_reference_weights, dict) or not mean_reference_weights:
                raise ValueError(
                    f"node {node_id}: mean_reference_kind='local_weights' requires a "
                    "non-empty mean_reference_weights mapping"
                )
            cleaned_mean_weights = {
                str(instrument): finite_float(
                    value, f"node {node_id} mean_reference_weights[{instrument!r}]"
                )
                for instrument, value in mean_reference_weights.items()
            }
            if abs(sum(cleaned_mean_weights.values()) - 1.0) > 1e-6:
                raise ValueError(
                    f"node {node_id}: mean_reference_weights must sum to one"
                )
            constraints["mean_reference_weights"] = cleaned_mean_weights
        elif mean_reference_weights not in (None, {}):
            raise ValueError(
                f"node {node_id}: mean_reference_weights requires "
                "mean_reference_kind='local_weights'"
            )
        else:
            constraints["mean_reference_weights"] = None

        for key in (
            "tracking_error_policy",
            "volatility_target_policy",
            "volatility_cap_policy",
        ):
            policy = str(constraints.get(key) or "hard_fail")
            if policy not in SUPPORTED_CONSTRAINT_POLICIES:
                raise ValueError(
                    f"node {node_id}: unsupported {key} {policy!r}; expected one of "
                    f"{sorted(SUPPORTED_CONSTRAINT_POLICIES)}"
                )
            if key == "volatility_cap_policy" and policy != "hard_fail":
                raise ValueError(
                    f"node {node_id}: volatility_cap_policy must be 'hard_fail'; "
                    "a volatility cap is never relaxed for TEV/target infeasibility"
                )
            constraints[key] = policy

        risk_aversion = optional_number(
            constraints, "risk_aversion", f"node {node_id} risk_aversion"
        )
        if risk_aversion is not None and risk_aversion <= 0.0:
            raise ValueError(f"node {node_id}: risk_aversion must be positive")
        optional_number(constraints, "risk_free_rate", f"node {node_id} risk_free_rate")

        if constraints.get("view_tau") not in (None, ""):
            tau = finite_float(constraints["view_tau"], f"node {node_id} view_tau")
            if tau <= 0.0:
                raise ValueError(f"node {node_id}: view_tau must be positive")
            constraints["view_tau"] = tau

        for key in (
            "per_asset_cap",
            "vol_target",
            "max_volatility",
            "max_tracking_error",
        ):
            value = optional_number(constraints, key, f"node {node_id} {key}")
            if value is not None and value < 0.0:
                raise ValueError(f"node {node_id}: {key} cannot be negative")
            if key == "per_asset_cap" and value is not None and value > 1.0:
                raise ValueError(f"node {node_id}: per_asset_cap must be in [0, 1]")

        views = constraints.get("views") or []
        if not isinstance(views, list):
            raise ValueError(f"node {node_id}: views must be a list")
        for index, view in enumerate(views):
            if not isinstance(view, dict):
                raise ValueError(f"node {node_id}: view {index} must be an object")
            expected = finite_float(
                view.get("expected_return"),
                f"node {node_id} view {index} expected_return",
            )
            confidence = finite_float(
                view.get("confidence"), f"node {node_id} view {index} confidence"
            )
            if not 0.0 < confidence <= 1.0:
                raise ValueError(
                    f"node {node_id}: view {index} confidence must be in (0, 1]"
                )
            instruments = view.get("instruments") or {}
            if not isinstance(instruments, dict) or not instruments:
                raise ValueError(
                    f"node {node_id}: view {index} requires instrument coefficients"
                )
            for instrument, coefficient in instruments.items():
                if _is_financing_view_label(instrument):
                    raise ValueError(
                        f"node {node_id}: Black-Litterman views cannot target "
                        "financing instruments"
                    )
                finite_float(
                    coefficient,
                    f"node {node_id} view {index} coefficient {instrument!r}",
                )
            view["expected_return"] = expected
            view["confidence"] = confidence

        if target_mode == "cap" and constraints.get("vol_target") not in (None, ""):
            if constraints.get("max_volatility") not in (None, ""):
                raise ValueError(
                    f"node {node_id}: cap mode cannot declare both vol_target and max_volatility"
                )
            constraints["max_volatility"] = constraints["vol_target"]
            constraints["max_volatility_reference"] = _normalize_reference(
                constraints.get("volatility_reference") or "manual"
            )
            constraints["vol_target"] = None
            constraints["volatility_reference"] = "none"

        leverage_declared = constraints.get("max_leverage") not in (None, "")
        leverage_raw = constraints.get("max_leverage", 1.0)
        max_leverage = finite_float(leverage_raw, f"node {node_id} max_leverage")
        if max_leverage < 1.0:
            raise ValueError(f"node {node_id}: max_leverage must be at least 1.0")

        cash_declared = "cash_enabled" in constraints
        allow_cash_declared = "allow_cash" in constraints
        if cash_declared and allow_cash_declared:
            canonical = boolean(constraints["cash_enabled"], f"node {node_id} cash_enabled")
            legacy = boolean(constraints["allow_cash"], f"node {node_id} allow_cash")
            if canonical != legacy:
                raise ValueError(
                    f"node {node_id}: cash_enabled and allow_cash contradict each other"
                )
            cash_enabled = canonical
        else:
            cash_enabled = boolean(
                constraints.get("cash_enabled", constraints.get("allow_cash", False)),
                f"node {node_id} cash_enabled",
            )
        if max_leverage > 1.0:
            cash_enabled = True

        financing_enabled = cash_enabled or max_leverage > 1.0
        financing_enabled_anywhere = financing_enabled_anywhere or financing_enabled
        spread_declared = constraints.get("borrow_spread_bps") not in (None, "")
        if spread_declared:
            declared_spread = finite_float(
                constraints["borrow_spread_bps"],
                f"node {node_id} borrow_spread_bps",
            )
            if node_id == root_id and not financing_enabled:
                spread = 0.0
                spread_source = "default"
            else:
                spread = declared_spread
                spread_source = "node"
        elif financing_enabled:
            spread = root_spread
            spread_source = (
                root_spread_source
                if node_id == root_id
                else ("root" if root_spread_source != "default" else "default")
            )
        else:
            spread = 0.0
            spread_source = "default"
        if spread < 0.0:
            raise ValueError(f"node {node_id}: borrow_spread_bps cannot be negative")
        if spread > 0.0 and not financing_enabled:
            raise ValueError(
                f"node {node_id}: borrow_spread_bps requires cash_enabled or max_leverage > 1"
            )

        constraints["cash_enabled"] = cash_enabled
        constraints["max_leverage"] = max_leverage
        constraints["borrow_spread_bps"] = spread
        constraints["cash_enabled_source"] = (
            "node" if cash_declared or allow_cash_declared or max_leverage > 1.0 else "default"
        )
        constraints["max_leverage_source"] = "node" if leverage_declared else "default"
        constraints["borrow_spread_bps_source"] = spread_source
        constraints.pop("allow_cash", None)

    if root_spread > 0.0 and not financing_enabled_anywhere:
        raise ValueError(
            "borrow_spread_bps requires cash_enabled or max_leverage > 1"
        )

    _validate_tree_structure(nodes, root_id)

    return normalized


def _validate_tree_structure(nodes: list[dict[str, Any]], root_id: str) -> None:
    """Reject structural identity collisions before any solve is attempted.

    The hierarchy resolver keys results by node *name* (a human-readable
    label, not the node id) and builds each node's local candidate frame from
    its direct instruments plus one solver column per child, named after the
    child's *proxy* ticker. Two nodes sharing a name, or two children of the
    same parent sharing a proxy (or a child's proxy colliding with that same
    parent's own direct instrument), silently overwrite one another in those
    dicts today — this produces materially wrong (e.g. double-counted)
    terminal weights with no error at all. Reject all of these explicitly
    instead, along with basic tree-shape problems (duplicate ids, more than
    one parent, cycles) that the same silent-dict-overwrite pattern would
    otherwise mask.
    """

    if not root_id or root_id == "None":
        raise ValueError("root_id is required and cannot be empty")

    ids = [str(item.get("id")) for item in nodes]
    empty_ids = [index for index, node_id in enumerate(ids) if not node_id or node_id == "None"]
    if empty_ids:
        raise ValueError(f"node(s) at position {empty_ids} are missing a non-empty id")

    id_counts: dict[str, int] = {}
    for node_id in ids:
        id_counts[node_id] = id_counts.get(node_id, 0) + 1
    duplicate_ids = sorted(name for name, count in id_counts.items() if count > 1)
    if duplicate_ids:
        raise ValueError(f"duplicate node id(s): {duplicate_ids}")

    if root_id not in id_counts:
        raise ValueError(f"root_id {root_id!r} does not match any declared node id")

    names = [str(item.get("name") or item.get("id")) for item in nodes]
    name_counts: dict[str, int] = {}
    for name in names:
        name_counts[name] = name_counts.get(name, 0) + 1
    duplicate_names = sorted(name for name, count in name_counts.items() if count > 1)
    if duplicate_names:
        raise ValueError(
            f"duplicate node name(s): {duplicate_names}; node names must be "
            "unique across the whole tree, they are used as result keys"
        )

    by_id = {str(item.get("id")): item for item in nodes}
    parent_of: dict[str, str] = {}
    for item in nodes:
        node_id = str(item.get("id"))
        for child_id in item.get("children") or []:
            child_id = str(child_id)
            if child_id not in by_id:
                raise ValueError(f"node {node_id}: unknown child id {child_id!r}")
            if child_id in parent_of:
                raise ValueError(
                    f"node {child_id!r} is declared as a child of both "
                    f"{parent_of[child_id]!r} and {node_id!r}; every node must "
                    "have exactly one parent"
                )
            parent_of[child_id] = node_id

    if root_id in parent_of:
        raise ValueError(f"root node {root_id!r} cannot also be declared as a child")

    visiting: set[str] = set()
    visited: set[str] = set()

    def walk(node_id: str) -> None:
        if node_id in visited:
            return
        if node_id in visiting:
            raise ValueError(f"cycle detected in the hierarchy at node {node_id!r}")
        visiting.add(node_id)
        node = by_id[node_id]
        proxies_seen: dict[str, str] = {}
        instrument_list = [str(instrument) for instrument in node.get("instruments") or []]
        duplicate_instruments = sorted(
            {
                instrument
                for instrument in instrument_list
                if instrument_list.count(instrument) > 1
            }
        )
        if duplicate_instruments:
            raise ValueError(
                f"node {node_id}: duplicate direct instrument(s) declared: "
                f"{duplicate_instruments}"
            )
        direct_instruments = set(instrument_list)
        for child_id in node.get("children") or []:
            child_id = str(child_id)
            child_proxy = by_id[child_id].get("proxy")
            if not child_proxy:
                raise ValueError(f"node {child_id}: child proxy is required")
            child_proxy = str(child_proxy)
            if child_proxy in direct_instruments:
                raise ValueError(
                    f"node {node_id}: child {child_id!r}'s proxy {child_proxy!r} "
                    "collides with a direct instrument declared on this same "
                    "node; a child's proxy column and a direct-instrument "
                    "column would silently overwrite each other"
                )
            if child_proxy in proxies_seen:
                raise ValueError(
                    f"node {node_id}: children {proxies_seen[child_proxy]!r} and "
                    f"{child_id!r} share the same proxy {child_proxy!r}; sibling "
                    "children under the same parent must use distinct proxies, "
                    "or their solver columns silently collide"
                )
            proxies_seen[child_proxy] = child_id
            walk(child_id)
        visiting.discard(node_id)
        visited.add(node_id)

    walk(root_id)

    unreachable = sorted(set(by_id) - visited)
    if unreachable:
        raise ValueError(
            f"node(s) are declared but not reachable from root {root_id!r}: "
            f"{unreachable}; every declared node must be part of the tree "
            "rooted at root_id, or it silently takes no part in any solve "
            "or audit"
        )


def validate_economic_settings(risk_aversion: float, risk_free_rate: float) -> None:
    if not isfinite(risk_aversion) or risk_aversion <= 0.0:
        raise V2OptimizationError("risk_aversion must be positive and finite")
    if not isfinite(risk_free_rate):
        raise V2OptimizationError("risk_free_rate must be finite")
