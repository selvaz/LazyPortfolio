"""Shared level-frame alignment: bounded forward-fill, never a fabricated live edge.

Used by both :mod:`lazyportfolio.backend` (ETF/instrument price levels) and
:mod:`lazyportfolio.fx` (FX rate levels) -- the same two market realities
apply to either: a holiday closes some but not all markets (an interior
gap, safe to forward-fill up to a bounded allowance), and a live quote
sometimes simply hasn't arrived yet (a gap at the most recent row, which
must stay ``NaN`` rather than be reported as today's level).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Longest interior gap (in trading rows) that a local-market closure is
#: allowed to span before it's treated as missing data rather than a
#: holiday. Covers multi-day closures (e.g. a national holiday cluster)
#: without masking a genuine, longer-running data outage. Shared default
#: for both instrument prices and FX rates -- the same domain judgment call
#: applies to either.
DEFAULT_MAX_GAP = 5


@dataclass(frozen=True)
class AlignedLevels:
    """An aligned level frame plus the bookkeeping behind the alignment."""

    frame: Any  # pandas.DataFrame
    filled_cells: int
    trailing_gap_cells: int


def align_levels(frame: Any, *, max_gap: int) -> AlignedLevels:
    """Forward-fill interior gaps up to ``max_gap`` consecutive rows.

    A gap that reaches all the way to the last row (the live edge) is left
    ``NaN`` regardless of ``max_gap`` -- it is missing-because-not-arrived,
    not missing-because-market-closed, and must never be read as "flat
    since the last observation."
    """
    missing = frame.isna()
    trailing_gap = missing[::-1].cumprod().astype(bool)[::-1]
    aligned = frame.ffill(limit=max_gap).mask(trailing_gap)
    filled_cells = int((missing.to_numpy() & ~aligned.isna().to_numpy()).sum())
    trailing_gap_cells = int(trailing_gap.to_numpy().sum())
    return AlignedLevels(
        frame=aligned, filled_cells=filled_cells, trailing_gap_cells=trailing_gap_cells
    )


__all__ = ["DEFAULT_MAX_GAP", "AlignedLevels", "align_levels"]
