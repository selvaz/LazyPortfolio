"""Portfolio reference-currency conversion: instrument_currencies, load_fx_rates,
convert_prices. Market Data Hub is mocked via sys.modules, mirroring
test_backend.py's pattern -- no live database needed.
"""

from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

from lazyportfolio.fx import (
    SUPPORTED_CURRENCIES,
    convert_prices,
    instrument_currencies,
    load_fx_rates,
)


def _stub_market_data_hub_db(monkeypatch, rows: list[tuple[str, str | None]]) -> None:
    """Stub market_data_hub.db.connection.get_conn for instrument_currencies."""

    class _FakeCursor:
        def __init__(self, rows: list[tuple[str, str | None]]) -> None:
            self._rows = rows

        def execute(self, sql: str, params: list[str]) -> _FakeCursor:
            wanted = {p.upper() for p in params}
            self._result = [row for row in self._rows if row[0].upper() in wanted]
            return self

        def fetchall(self) -> list[tuple[str, str | None]]:
            return self._result

        def close(self) -> None:
            pass

    connection_module = types.ModuleType("market_data_hub.db.connection")
    connection_module.get_conn = lambda read_only=True: _FakeCursor(rows)  # type: ignore[attr-defined]
    db_module = types.ModuleType("market_data_hub.db")
    db_module.connection = connection_module  # type: ignore[attr-defined]
    root = types.ModuleType("market_data_hub")
    root.db = db_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "market_data_hub", root)
    monkeypatch.setitem(sys.modules, "market_data_hub.db", db_module)
    monkeypatch.setitem(sys.modules, "market_data_hub.db.connection", connection_module)


def _stub_market_data_hub_extract(monkeypatch, levels: pd.DataFrame) -> dict[str, object]:
    """Stub market_data_hub.extract.extract_series for load_fx_rates."""
    calls: dict[str, object] = {}

    extract_module = types.ModuleType("market_data_hub.extract")

    def extract_series(symbols, *args, **kwargs):
        calls["symbols"] = symbols
        calls.update(kwargs)
        return levels[symbols], {"source": "market-data-hub"}

    extract_module.extract_series = extract_series  # type: ignore[attr-defined]
    root = types.ModuleType("market_data_hub")
    root.extract = extract_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "market_data_hub", root)
    monkeypatch.setitem(sys.modules, "market_data_hub.extract", extract_module)
    return calls


# --------------------------------------------------------------------------- #
# instrument_currencies
# --------------------------------------------------------------------------- #


def test_instrument_currencies_returns_supported_currencies(monkeypatch) -> None:
    _stub_market_data_hub_db(
        monkeypatch, [("SPY", "USD"), ("VGK", "EUR"), ("VOD.L", "GBP")]
    )
    result = instrument_currencies(["SPY", "VGK", "VOD.L"])
    assert result == {"SPY": "USD", "VGK": "EUR", "VOD.L": "GBP"}


def test_instrument_currencies_raises_naming_null_currency_symbol(monkeypatch) -> None:
    _stub_market_data_hub_db(monkeypatch, [("SPY", "USD"), ("EEM", None)])
    with pytest.raises(ValueError, match="EEM"):
        instrument_currencies(["SPY", "EEM"])


def test_instrument_currencies_raises_naming_unsupported_currency(monkeypatch) -> None:
    _stub_market_data_hub_db(monkeypatch, [("SPY", "USD"), ("EXH1.DE", "CHF")])
    with pytest.raises(ValueError, match="EXH1.DE"):
        instrument_currencies(["SPY", "EXH1.DE"])


def test_instrument_currencies_empty_input_returns_empty(monkeypatch) -> None:
    _stub_market_data_hub_db(monkeypatch, [])
    assert instrument_currencies([]) == {}


# --------------------------------------------------------------------------- #
# load_fx_rates
# --------------------------------------------------------------------------- #


def test_load_fx_rates_builds_usd_pivoted_frame(monkeypatch) -> None:
    index = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
    levels = pd.DataFrame(
        {
            "EURUSD=X": [1.10, 1.11, 1.09],
            "GBPUSD=X": [1.27, 1.28, 1.26],
            "USDJPY=X": [150.0, 151.0, 149.0],
        },
        index=index,
    )
    calls = _stub_market_data_hub_extract(monkeypatch, levels)

    rates = load_fx_rates("2026-01-01", "2026-01-10")

    assert calls["transform"] == "level"
    assert (rates["USD"] == 1.0).all()
    assert rates["EUR"].tolist() == pytest.approx([1.10, 1.11, 1.09])
    assert rates["GBP"].tolist() == pytest.approx([1.27, 1.28, 1.26])
    assert rates["JPY"].tolist() == pytest.approx([1 / 150.0, 1 / 151.0, 1 / 149.0])


def test_load_fx_rates_leaves_live_edge_gap_unfilled(monkeypatch) -> None:
    index = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
    levels = pd.DataFrame(
        {
            "EURUSD=X": [1.10, 1.11, None],
            "GBPUSD=X": [1.27, 1.28, 1.26],
            "USDJPY=X": [150.0, 151.0, 149.0],
        },
        index=index,
    )
    _stub_market_data_hub_extract(monkeypatch, levels)

    rates = load_fx_rates("2026-01-01", "2026-01-10")

    assert pd.isna(rates["EUR"].iloc[-1])


# --------------------------------------------------------------------------- #
# convert_prices
# --------------------------------------------------------------------------- #


def _fx_frame() -> pd.DataFrame:
    index = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
    return pd.DataFrame(
        {
            "USD": [1.0, 1.0, 1.0],
            "EUR": [1.10, 1.11, 1.09],
            "GBP": [1.27, 1.28, 1.26],
            "JPY": [1 / 150.0, 1 / 151.0, 1 / 149.0],
        },
        index=index,
    )


def test_convert_prices_same_currency_is_a_no_op() -> None:
    index = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
    prices = pd.DataFrame({"ticker:SPY": [100.0, 101.0, 102.0]}, index=index)
    converted = convert_prices(prices, {"ticker:SPY": "USD"}, "USD", _fx_frame())
    assert converted["ticker:SPY"].tolist() == pytest.approx([100.0, 101.0, 102.0])


def test_convert_prices_converts_eur_instrument_into_usd_portfolio() -> None:
    index = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
    prices = pd.DataFrame({"ticker:VGK": [50.0, 51.0, 52.0]}, index=index)
    converted = convert_prices(prices, {"ticker:VGK": "EUR"}, "USD", _fx_frame())
    assert converted["ticker:VGK"].tolist() == pytest.approx([55.0, 56.61, 56.68])


def test_convert_prices_into_a_non_usd_target_still_no_ops_the_matching_currency() -> None:
    index = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
    prices = pd.DataFrame(
        {"ticker:SPY": [100.0, 101.0, 102.0], "ticker:VGK": [50.0, 51.0, 52.0]}, index=index
    )
    converted = convert_prices(
        prices, {"ticker:SPY": "USD", "ticker:VGK": "EUR"}, "EUR", _fx_frame()
    )
    assert converted["ticker:VGK"].tolist() == pytest.approx([50.0, 51.0, 52.0])
    assert converted["ticker:SPY"].tolist() == pytest.approx([90.909091, 90.990991, 93.577982])


def test_convert_prices_rejects_unsupported_target_currency() -> None:
    index = pd.to_datetime(["2026-01-02"])
    prices = pd.DataFrame({"ticker:SPY": [100.0]}, index=index)
    with pytest.raises(ValueError, match="CHF"):
        convert_prices(prices, {"ticker:SPY": "USD"}, "CHF", _fx_frame())


def test_supported_currencies_is_exactly_four() -> None:
    assert SUPPORTED_CURRENCIES == {"USD", "EUR", "GBP", "JPY"}
