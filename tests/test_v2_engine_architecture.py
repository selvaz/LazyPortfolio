from __future__ import annotations

import inspect
from importlib import import_module, util
from typing import Any, cast

from lazyportfolio import V2Constraints, V2LocalOptimizer, V2Model

hv2 = cast(Any, import_module("lazyportfolio.hierarchical_v2"))

_REMOVED_PATCH_MODULES = (
    "lazyportfolio.audit_finalize",
    "lazyportfolio.financing",
    "lazyportfolio.financing_finalize",
    "lazyportfolio.remediation",
    "lazyportfolio.studio_compat",
)


def _config() -> dict[str, object]:
    return {
        "root_id": "root",
        "currency": "USD",
        "nodes": [
            {
                "id": "root",
                "name": "Root",
                "children": [],
                "instruments": ["A", "B"],
                "goal": {"objective": "max_return"},
                "constraints": {
                    "cash_enabled": True,
                    "max_leverage": 1.25,
                    "borrow_spread_bps": 50,
                    "covariance_estimator": "ledoit_wolf",
                },
            }
        ],
        "backtest": {
            "benchmark": {
                "name": "B0",
                "weights": {"A": 0.5, "B": 0.5},
            }
        },
    }


def test_public_facade_exports_canonical_contracts_and_solver() -> None:
    assert hv2.V2Constraints is V2Constraints
    assert hv2.V2LocalOptimizer is V2LocalOptimizer
    assert V2LocalOptimizer.__module__ == "lazyportfolio.v2.solver"
    assert V2Constraints.__module__ == "lazyportfolio.v2.contracts"


def test_engine_has_no_patch_closure_or_import_time_markers() -> None:
    closure = inspect.getclosurevars(V2LocalOptimizer.solve)
    assert not closure.nonlocals
    source = inspect.getsource(hv2)
    assert "apply_optimizer_remediation" not in source
    assert "apply_financing_support" not in source
    assert "apply_financing_finalizers" not in source
    assert not hasattr(hv2, "_METHODOLOGY_REMEDIATION_APPLIED")
    assert not hasattr(hv2, "_FINANCING_SUPPORT_APPLIED")
    assert not hasattr(hv2, "_FINANCING_FINALIZERS_APPLIED")


def test_transitional_patch_modules_are_absent() -> None:
    for module_name in _REMOVED_PATCH_MODULES:
        assert util.find_spec(module_name) is None


def test_model_parser_builds_one_canonical_contract() -> None:
    model = V2Model.from_config(_config())
    constraints = model.root.constraints
    assert isinstance(constraints, V2Constraints)
    assert constraints.cash_enabled is True
    assert constraints.max_leverage == 1.25
    assert constraints.borrow_spread_bps == 50.0
    assert constraints.covariance_estimator == "ledoit_wolf"
