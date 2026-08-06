"""Portfolio reference-currency conversion: USD, EUR, GBP, JPY only.

A tree's ``currency`` (validated in :mod:`lazyportfolio.v2.validation`) is
the currency every instrument -- including benchmark constituents -- must be
converted into before any return/covariance/optimization computation. This
module is the conversion itself; :mod:`lazyportfolio.backend` is the only
caller (see ``load_returns``'s ``currency`` parameter).

Market Data Hub has no dedicated FX-rates table: the three pairs below are
ordinary ``prices_daily`` rows, fetched the exact same way instrument prices
are (:func:`market_data_hub.extract.extract_series`). They give full
USD-pivoted coverage of exactly the four supported currencies -- EUR/GBP are
quoted directly against USD, JPY is USD-quoted (``USDJPY=X`` prices 1 USD in
JPY, so its reciprocal is USD-per-JPY) -- so no other pair is needed as long
as :data:`SUPPORTED_CURRENCIES` stays at four.

Instrument currency comes from ``listings.currency`` in the same hub
database. It is never guessed: a symbol with a ``NULL`` or unsupported
currency fails loudly, naming the symbol, rather than silently assuming a
default -- the same posture Market Data Hub's own ``currency_for_symbol``
already takes for symbols outside its curated universe.
"""

from __future__ import annotations

from typing import Any

from lazyportfolio.price_alignment import DEFAULT_MAX_GAP, align_levels
from lazyportfolio.v2.validation import SUPPORTED_CURRENCIES

#: Yahoo pair symbol quoting 1 USD in the given currency (i.e. the pair's
#: level is "units of this currency per 1 USD") for every non-USD supported
#: currency -- see module docstring for why these three are sufficient.
_USD_QUOTE_PAIRS = {
    "EUR": "EURUSD=X",  # prices 1 EUR in USD directly -> already USD-per-EUR
    "GBP": "GBPUSD=X",  # prices 1 GBP in USD directly -> already USD-per-GBP
    "JPY": "USDJPY=X",  # prices 1 USD in JPY -> USD-per-JPY is the reciprocal
}


def instrument_currencies(
    symbols: list[str], *, db_path: str | None = None
) -> dict[str, str]:
    """Native trading currency for each bare symbol, from ``listings.currency``.

    Raises ``ValueError`` naming any symbol whose currency is ``NULL`` or
    outside :data:`SUPPORTED_CURRENCIES` -- never guesses.
    """
    normalized = list(dict.fromkeys(symbol.strip().upper() for symbol in symbols if symbol))
    if not normalized:
        return {}
    try:
        from market_data_hub.db.connection import get_conn
    except ImportError as exc:  # pragma: no cover - optional integration
        raise ImportError(
            "currency conversion requires market-data-hub: pip install "
            "'market-data-hub @ git+https://github.com/selvaz/market-data-hub.git'"
        ) from exc

    placeholders = ", ".join("?" for _ in normalized)
    con = get_conn(read_only=True)
    try:
        rows = con.execute(
            f"""
            SELECT l.symbol, l.currency
            FROM listings l
            WHERE l.active_to IS NULL AND upper(l.symbol) IN ({placeholders})
            """,
            normalized,
        ).fetchall()
    finally:
        con.close()

    by_symbol = {str(symbol).upper(): currency for symbol, currency in rows}
    result: dict[str, str] = {}
    unknown: list[str] = []
    for symbol in normalized:
        currency = by_symbol.get(symbol)
        if currency is None or currency not in SUPPORTED_CURRENCIES:
            unknown.append(symbol)
            continue
        result[symbol] = currency
    if unknown:
        raise ValueError(
            "unknown or unsupported currency for instrument(s): "
            + ", ".join(sorted(unknown))
            + f" (supported: {sorted(SUPPORTED_CURRENCIES)})"
        )
    return result


def load_fx_rates(
    start: str | None,
    end: str | None,
    *,
    db_path: str | None = None,
    max_gap: int = DEFAULT_MAX_GAP,
) -> Any:
    """USD-pivoted FX levels: one column per supported currency, "USD per 1 unit".

    ``USD`` is a constant ``1.0`` column. Aligned the same way instrument
    prices are (:func:`lazyportfolio.price_alignment.align_levels`) -- an FX
    holiday or a live-edge FX quote gap deserves the identical treatment as
    an instrument price gap.
    """
    import pandas as pd

    try:
        from market_data_hub import extract
    except ImportError as exc:  # pragma: no cover - optional integration
        raise ImportError(
            "currency conversion requires market-data-hub: pip install "
            "'market-data-hub @ git+https://github.com/selvaz/market-data-hub.git'"
        ) from exc

    pair_symbols = list(_USD_QUOTE_PAIRS.values())
    levels, _ = extract.extract_series(
        pair_symbols,
        start=start or None,
        end=end or None,
        domain="prices",
        field="adj_close",
        transform="level",
        frequency="D",
        fillna="none",
        db_path=db_path,
    )
    levels = levels.rename(columns={pair: currency for currency, pair in _USD_QUOTE_PAIRS.items()})[
        list(_USD_QUOTE_PAIRS)
    ]
    aligned = align_levels(levels, max_gap=max_gap).frame

    usd_per_unit = pd.DataFrame(index=aligned.index)
    usd_per_unit["USD"] = 1.0
    usd_per_unit["EUR"] = aligned["EUR"]
    usd_per_unit["GBP"] = aligned["GBP"]
    usd_per_unit["JPY"] = 1.0 / aligned["JPY"]
    return usd_per_unit


def convert_prices(
    prices: Any,
    native_currencies: dict[str, str],
    target_currency: str,
    fx_usd_per_unit: Any,
) -> Any:
    """Convert each instrument column of ``prices`` into ``target_currency``.

    ``native_currencies`` maps each ``prices`` column name to its native
    currency (already validated by :func:`instrument_currencies`).
    Multiplies by ``fx_usd_per_unit[native] / fx_usd_per_unit[target]``,
    date-aligned -- a same-currency instrument (``native == target``)
    multiplies by exactly ``1.0``, a no-op.
    """
    if target_currency not in SUPPORTED_CURRENCIES:
        raise ValueError(f"unsupported target currency {target_currency!r}")
    # Re-align onto the instrument price frame's own date index (FX and
    # instrument prices are fetched independently and can have slightly
    # different trading calendars) using the same bounded-gap policy as
    # everywhere else -- an unbounded ffill here would reintroduce exactly
    # the stale-fill risk price_alignment.align_levels exists to prevent.
    fx = align_levels(fx_usd_per_unit.reindex(prices.index), max_gap=DEFAULT_MAX_GAP).frame
    target_usd_per_unit = fx[target_currency]
    converted = prices.copy()
    for column in prices.columns:
        native = native_currencies[column]
        multiplier = fx[native] / target_usd_per_unit
        converted[column] = prices[column] * multiplier
    return converted


__all__ = [
    "SUPPORTED_CURRENCIES",
    "convert_prices",
    "instrument_currencies",
    "load_fx_rates",
]
