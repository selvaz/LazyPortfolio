from __future__ import annotations

import pandas as pd
import pytest

from lazyportfolio.v2.backtest import HierarchicalV2Backtester, _V2Ledger
from lazyportfolio.v2.model import V2Model
from lazyportfolio.v2.moments import (
    CASH_BORROW,
    CASH_LEND,
    financing_base,
    financing_instrument,
    is_financing_instrument,
)


def _config() -> dict[str, object]:
    return {
        "root_id": "root",
        "currency": "USD",
        "nodes": [
            {
                "id": "root",
                "name": "Root",
                "children": ["child"],
                "instruments": ["A"],
                "goal": {"objective": "max_return"},
                "constraints": {
                    "cash_enabled": True,
                    "risk_free_rate": 0.03,
                    "borrow_spread_bps": 50,
                },
            },
            {
                "id": "child",
                "name": "Child",
                "children": [],
                "instruments": ["B"],
                "proxy": "P",
                "goal": {"objective": "max_return"},
                "constraints": {
                    "max_leverage": 1.5,
                    "risk_free_rate": 0.04,
                    "borrow_spread_bps": 100,
                },
            },
        ],
        "backtest": {"benchmark": {"name": "B0", "weights": {"A": 0.5, "B": 0.5}}},
    }


def _returns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker:A": [0.01, 0.02],
            "ticker:B": [0.02, 0.01],
            "ticker:P": [0.015, 0.012],
        }
    )


def test_backtest_financing_columns_are_unique_and_rate_correct() -> None:
    model = V2Model.from_config(_config())
    effective = HierarchicalV2Backtester._with_financing_returns(
        model, _returns(), 12.0
    )
    child_lend = financing_instrument(CASH_LEND, "child", is_root=False)
    child_borrow = financing_instrument(CASH_BORROW, "child", is_root=False)
    assert effective[CASH_LEND].iloc[0] == pytest.approx(0.03 / 12.0)
    assert effective[CASH_BORROW].iloc[0] == pytest.approx(0.035 / 12.0)
    assert effective[child_lend].iloc[0] == pytest.approx(0.04 / 12.0)
    assert effective[child_borrow].iloc[0] == pytest.approx(0.05 / 12.0)
    assert len({CASH_LEND, CASH_BORROW, child_lend, child_borrow}) == 4


def test_root_only_configuration_keeps_public_cash_names() -> None:
    config = _config()
    config["nodes"] = [config["nodes"][0] | {"children": [], "instruments": ["A"]}]
    config["backtest"]["benchmark"]["weights"] = {"A": 1.0}
    model = V2Model.from_config(config)
    effective = HierarchicalV2Backtester._with_financing_returns(
        model, _returns(), 12.0
    )
    assert CASH_LEND in effective
    assert CASH_BORROW in effective
    assert all("@root" not in name for name in effective.columns)


def test_root_financing_does_not_activate_child_financing() -> None:
    config = _config()
    config["nodes"][1]["constraints"] = {}
    model = V2Model.from_config(config)
    child = model.root.children[0].constraints
    assert child.cash_enabled is False
    assert child.max_leverage == pytest.approx(1.0)
    assert child.borrow_spread_bps == pytest.approx(0.0)
    assert child.borrow_spread_bps_source == "default"


def test_ledger_accounts_for_root_and_child_financing_once() -> None:
    child_borrow = financing_instrument(CASH_BORROW, "child", is_root=False)
    target = {"ticker:A": 0.5, "ticker:B": 0.6, CASH_LEND: 0.1, child_borrow: -0.2}
    day = pd.Series(
        {
            "ticker:A": 0.01,
            "ticker:B": 0.02,
            CASH_LEND: 0.03 / 252.0,
            child_borrow: 0.05 / 252.0,
        }
    )
    ledger = _V2Ledger(transaction_cost_bps=10.0)
    assert ledger.rebalance(target) == pytest.approx(0.0014)
    assert ledger.step(day) == pytest.approx(
        sum(weight * day[name] for name, weight in target.items())
    )
    assert sum(ledger.weights.values()) == pytest.approx(1.0)


def test_financing_name_helpers_are_collision_safe() -> None:
    child_lend = financing_instrument(CASH_LEND, "child", is_root=False)
    assert child_lend == "cash:RF@child"
    assert is_financing_instrument(CASH_LEND) is True
    assert is_financing_instrument(child_lend) is True
    assert is_financing_instrument("ticker:A") is False
    assert financing_base(CASH_BORROW) == CASH_BORROW
    assert financing_base(child_lend) == CASH_LEND
    assert financing_base("ticker:A") is None
    with pytest.raises(ValueError, match="unsupported financing"):
        financing_instrument("cash:OTHER", "child", is_root=False)
