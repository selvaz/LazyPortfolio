from __future__ import annotations

from copy import deepcopy

import pytest

from lazyportfolio import V2Model


def _config() -> dict[str, object]:
    return {
        "root_id": "root",
        "currency": "USD",
        "nodes": [
            {
                "id": "root",
                "name": "Root",
                "children": [],
                "instruments": ["AAA", "BBB"],
                "goal": {"objective": "max_ratio"},
                "constraints": {},
            }
        ],
        "backtest": {
            "benchmark": {
                "name": "B0",
                "weights": {"AAA": 0.5, "BBB": 0.5},
            }
        },
        "data": {"risk_free_annual": "0.035"},
    }


def test_global_studio_risk_free_is_migrated_to_root() -> None:
    config = _config()
    original = deepcopy(config)
    model = V2Model.from_config(config)
    assert model.root.constraints.risk_free_rate == pytest.approx(0.035)
    assert config == original


def test_matching_root_and_global_rates_are_accepted() -> None:
    config = _config()
    nodes = config["nodes"]
    assert isinstance(nodes, list)
    root = nodes[0]
    assert isinstance(root, dict)
    root["constraints"] = {"risk_free_rate": 0.035}
    model = V2Model.from_config(config)
    assert model.root.constraints.risk_free_rate == pytest.approx(0.035)


def test_conflicting_root_and_global_rates_fail_loudly() -> None:
    config = _config()
    nodes = config["nodes"]
    assert isinstance(nodes, list)
    root = nodes[0]
    assert isinstance(root, dict)
    root["constraints"] = {"risk_free_rate": 0.01}
    with pytest.raises(ValueError, match="conflicting risk-free rates"):
        V2Model.from_config(config)


def test_non_finite_global_rate_is_rejected() -> None:
    config = _config()
    data = config["data"]
    assert isinstance(data, dict)
    data["risk_free_annual"] = "nan"
    with pytest.raises(ValueError, match="must be finite"):
        V2Model.from_config(config)


def test_financing_spread_requires_cash_or_leverage() -> None:
    config = _config()
    data = config["data"]
    assert isinstance(data, dict)
    data["borrow_spread_bps"] = "25"
    with pytest.raises(ValueError, match="requires cash_enabled"):
        V2Model.from_config(config)
