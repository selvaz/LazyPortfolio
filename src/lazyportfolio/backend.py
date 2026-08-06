"""Daily simple-return seam for the optimizer; no observations cross an LLM boundary.

The production seam deliberately obtains price *levels* from Market Data Hub,
aligns them on their shared trading-date grid, and only then derives returns.
This makes a local-market holiday an explicit zero return rather than silently
removing the entire date from a multi-market portfolio's history.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from lazyportfolio.price_alignment import DEFAULT_MAX_GAP, align_levels


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
        currency: str | None = None,
    ) -> OptimizationDataset: ...


class MarketDataHubOptimizationBackend:
    """market-data-hub implementation of :class:`OptimizationDataBackend`."""

    #: Longest interior gap (in trading rows) that a local-market closure is
    #: allowed to span before it's treated as missing data rather than a
    #: holiday. Covers multi-day closures (e.g. a national holiday cluster)
    #: without masking a genuine, longer-running data outage.
    _MAX_HOLIDAY_GAP = DEFAULT_MAX_GAP

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path

    def load_returns(
        self,
        instruments: list[str],
        *,
        start: str = "",
        end: str = "",
        frequency: str = "D",
        currency: str | None = None,
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
        # Forward-fill only across that already-observed grid, and only for
        # interior gaps up to `_MAX_HOLIDAY_GAP` rows: a component that did
        # not trade on a date on which another market did is held at its
        # latest tradable price. A gap at the *live edge* — the instrument's
        # most recent price hasn't arrived yet — is left as NaN rather than
        # reported as today's price, so it reads as missing (and gets
        # dropped downstream) instead of a fabricated flat/zero return.
        aligned = align_levels(prices, max_gap=self._MAX_HOLIDAY_GAP)
        aligned_prices = aligned.frame

        instrument_currencies: dict[str, str] | None = None
        if currency is not None:
            # Convert price *levels* into the portfolio's reference currency
            # before deriving returns -- a return isn't linear in the FX
            # rate, so converting returns directly would be wrong.
            from lazyportfolio import fx

            instrument_currencies = fx.instrument_currencies(symbols, db_path=self._db_path)
            fx_rates = fx.load_fx_rates(start or None, end or None, db_path=self._db_path)
            native_by_label = dict(
                zip(labels, (instrument_currencies[s] for s in symbols), strict=True)
            )
            aligned_prices = fx.convert_prices(aligned_prices, native_by_label, currency, fx_rates)

        # ``fill_method=None`` prevents pandas from applying a second,
        # implicit fill while deriving returns.
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
                "price_alignment_max_gap_days": self._MAX_HOLIDAY_GAP,
                "price_rows": len(prices),
                "filled_price_cells": aligned.filled_cells,
                "trailing_gap_cells": aligned.trailing_gap_cells,
                "n_rows": len(frame),
                "reference_currency": currency,
                "instrument_currencies": instrument_currencies,
                **metadata,
            },
        )
