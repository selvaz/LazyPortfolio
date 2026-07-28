"""Frequency, resampling and rebalance-calendar helpers shared by V2.

Kept separate from the optimization logic itself: these functions only
determine *when* a fold happens and how daily returns compound onto a
coarser grid, independent of which optimizer solves the fold.
"""

from __future__ import annotations

from typing import Any

_ANNUALIZATION_FACTORS = {"D": 252.0, "W": 52.0, "M": 12.0, "Q": 4.0}


def _annualization_factor(frequency: str) -> float:
    try:
        return _ANNUALIZATION_FACTORS[frequency]
    except KeyError as exc:  # pragma: no cover - model validation normally guards this
        raise ValueError(f"unsupported frequency {frequency!r}") from exc


def _resample_simple_returns(daily_returns: Any, frequency: str) -> Any:
    """Compound canonical daily simple returns onto an estimator's time grid."""
    if frequency == "D":
        return daily_returns
    return (
        daily_returns.add(1.0)
        .resample(_resample_rule(frequency))
        .prod(min_count=1)
        .sub(1.0)
        .dropna(how="any")
    )


def _resample_rule(frequency: str) -> Any:
    """Return pandas offsets supported by both older and current pandas releases."""
    if frequency == "D":
        return "D"
    from pandas.tseries.offsets import MonthEnd, QuarterEnd, Week

    rules = {
        "W": Week(weekday=4),
        "M": MonthEnd(),
        "Q": QuarterEnd(),
    }
    try:
        return rules[frequency]
    except KeyError as exc:  # pragma: no cover - model validation guards this
        raise ValueError(f"unsupported frequency {frequency!r}") from exc


def _final_period_is_complete(last_observation: Any, frequency: str) -> bool:
    """Whether the final daily observation reaches its business-period endpoint."""
    if frequency == "D":
        return True
    if frequency == "W":
        return bool(last_observation.weekday() == 4)

    from pandas.tseries.offsets import BMonthEnd, BQuarterEnd

    if frequency == "M":
        return bool(BMonthEnd().rollback(last_observation) == last_observation)
    if frequency == "Q":
        return bool(BQuarterEnd().rollback(last_observation) == last_observation)
    raise ValueError(f"unsupported frequency {frequency!r}")


def _rebalance_dates(
    daily_returns: Any,
    frequency: str,
    *,
    include_partial_last_period: bool = False,
) -> list[Any]:
    """Calendar endpoints of complete daily holding periods at ``frequency``."""
    if frequency == "D":
        return list(daily_returns.index)
    schedule = list(
        daily_returns.resample(_resample_rule(frequency)).last().dropna().index
    )
    if (
        schedule
        and not include_partial_last_period
        and not _final_period_is_complete(daily_returns.index.max(), frequency)
    ):
        schedule.pop()
    return schedule
