"""Derive the estimator/backtester ``Mode`` from a tree configuration.

Promoted out of Tree Studio (``project/tree_studio.py``'s former ``_v2_mode``)
so any caller building a ``V2Model`` from a saved or inline config -- the
Studio UI, or LazyTools' MCP ``portfolio_tree_estimate``/``_backtest`` tools --
derives the same mode for the same config. Two processes independently
re-implementing this 12-line rule is exactly the kind of drift that would
make a tree behave differently in the GUI than over MCP for no visible reason.
"""

from __future__ import annotations

from typing import Any

from lazyportfolio.v2.contracts import Mode


def mode_from_config(config: dict[str, Any]) -> Mode:
    """Resolve ``flat``/``forward``/``forward_backward`` from ``config["backtest"]``.

    ``forward_enabled=False`` (default ``True``) forces ``flat``. Otherwise
    ``backtest.hierarchy_mode`` selects: ``"proxy"`` (default) -> ``forward``,
    ``"synthetic_reconstructed"`` -> ``forward_backward``. Any other value is a
    reserved, not-yet-implemented iterative mode and is rejected outright.
    """
    backtest_raw = config.get("backtest")
    backtest = backtest_raw if isinstance(backtest_raw, dict) else {}
    if not bool(backtest.get("forward_enabled", True)):
        return "flat"
    hierarchy_mode = str(backtest.get("hierarchy_mode") or "proxy")
    if hierarchy_mode == "proxy":
        return "forward"
    if hierarchy_mode == "synthetic_reconstructed":
        return "forward_backward"
    raise ValueError(
        "V2 iterative mode is intentionally disabled until the non-iterative engine is accepted"
    )


__all__ = ["mode_from_config"]
