"""``mode_from_config``: the single source of truth Tree Studio and any other
caller (LazyTools' MCP `portfolio_tree_*` tools) both derive a run's mode
from, so the same saved config can never estimate differently in each place.
"""

from __future__ import annotations

import pytest

from lazyportfolio.v2.mode import mode_from_config


def _config(**backtest_overrides: object) -> dict[str, object]:
    return {"backtest": {"forward_enabled": True, "hierarchy_mode": "proxy", **backtest_overrides}}


def test_forward_disabled_is_flat_regardless_of_hierarchy_mode() -> None:
    assert mode_from_config(_config(forward_enabled=False)) == "flat"
    assert mode_from_config(_config(forward_enabled=False, hierarchy_mode="synthetic_reconstructed")) == "flat"


def test_proxy_hierarchy_mode_is_forward() -> None:
    assert mode_from_config(_config(hierarchy_mode="proxy")) == "forward"


def test_synthetic_reconstructed_hierarchy_mode_is_forward_backward() -> None:
    assert mode_from_config(_config(hierarchy_mode="synthetic_reconstructed")) == "forward_backward"


def test_missing_backtest_block_defaults_to_forward() -> None:
    assert mode_from_config({}) == "forward"


def test_missing_hierarchy_mode_defaults_to_proxy_forward() -> None:
    assert mode_from_config({"backtest": {"forward_enabled": True}}) == "forward"


def test_unrecognized_hierarchy_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="iterative mode is intentionally disabled"):
        mode_from_config(_config(hierarchy_mode="current_parent_synthetic"))
