"""Shared point-in-time data preparation for walk-forward engines."""

from __future__ import annotations

from typing import Any

from lazyportfolio.calendar import _rebalance_dates, _resample_simple_returns
from lazyportfolio.models import BacktestSpec


def prepare_walk_forward_inputs(
    daily_returns: Any,
    instruments: list[str],
    protocol: BacktestSpec,
    estimation_frequency: str,
) -> tuple[Any, Any, list[Any]]:
    """Return one complete-case valuation grid, estimation grid and schedule.

    Both generic and nested runs must derive their folds from this same grid;
    otherwise missing observations can change their signal dates independently.
    """
    valuation = daily_returns.loc[:, instruments].dropna(how="any")
    estimation = _resample_simple_returns(valuation, estimation_frequency)
    schedule = _rebalance_dates(
        valuation,
        protocol.rebalance_frequency,
        include_partial_last_period=protocol.include_partial_last_period,
    )
    return valuation, estimation, schedule
