"""Daily simple-return seam for the optimizer; no observations cross an LLM boundary.

The production seam deliberately obtains price *levels* from Market Data Hub,
aligns them on their shared trading-date grid, and only then derives returns.
This makes a local-market holiday an explicit zero return rather than silently
removing the entire date from a multi-market portfolio's history.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class OptimizationDataset:
    """Internal-only return matrix and its bounded provenance metadata."""

    returns: Any  # pandas.DataFrame, deliberately optional at import time
    metadata: dict[str, Any]


class OptimizationDataBackend(Protocol):
    """Loads a complete canonical *daily simple-return* matrix privately."""

    def load_returns(
        self,
        instruments: list[str],
        *,
        start: str = "",
        end: str = "",
        frequency: str = "D",
    ) -> OptimizationDataset: ...


class MarketDataHubOptimizationBackend:
    """market-data-hub implementation of :class:`OptimizationDataBackend`."""

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path

    def load_returns(
        self,
        instruments: list[str],
        *,
        start: str = "",
        end: str = "",
        frequency: str = "D",
    ) -> OptimizationDataset:
        try:
            from market_data_hub import extract
            from market_data_hub.lazydatacore import Domain, InstrumentId
        except ImportError as exc:  # pragma: no cover - optional integration
            raise ImportError(
                "portfolio optimization requires market-data-hub: pip install "
                "'market-data-hub @ git+https://github.com/selvaz/market-data-hub.git'"
            ) from exc

        if frequency != "D":
            raise ValueError(
                "optimizer backend returns canonical daily data only; "
                "use spec.frequency for estimation"
            )
        # Match the hub-facing UX used by the statistical and regime tools:
        # bare symbols are a convenience spelling of ticker:<symbol>.
        requested = [item if ":" in item else f"ticker:{item}" for item in instruments]
        parsed = [InstrumentId.parse(item) for item in requested]
        unsupported = [str(item) for item in parsed if item.domain is not Domain.TICKER]
        if unsupported:
            raise ValueError(
                "portfolio optimization currently supports ticker: instruments only; "
                f"unsupported: {', '.join(unsupported)}"
            )
        if len({str(item) for item in parsed}) != len(parsed):
            raise ValueError("instruments must be unique after canonicalisation")

        symbols = [item.key for item in parsed]
        prices, metadata = extract.extract_series(
            symbols,
            start=start or None,
            end=end or None,
            domain="prices",
            field="adj_close",
            transform="level",
            frequency="D",
            fillna="none",
            db_path=self._db_path,
        )
        labels = [str(item) for item in parsed]
        prices = prices.rename(columns=dict(zip(symbols, labels, strict=True)))[labels]

        # The hub's wide price frame is the union of observed trading dates.
        # Forward-fill only across that already-observed grid: a component that
        # did not trade on a date on which another market did is held at its
        # latest tradable price.  ``fill_method=None`` prevents pandas from
        # applying a second, implicit fill while deriving returns.
        aligned_prices = prices.ffill()
        frame = aligned_prices.pct_change(fill_method=None)
        return OptimizationDataset(
            returns=frame,
            metadata={
                "source": "market-data-hub",
                "instruments": labels,
                "requested_start": start or None,
                "requested_end": end or None,
                "frequency": "D",
                "return_type": "simple",
                "source_field": "adj_close",
                "price_alignment": "forward_fill_shared_trading_grid",
                "price_rows": len(prices),
                "aligned_price_rows": len(aligned_prices),
                "n_rows": len(frame),
                **metadata,
            },
        )
