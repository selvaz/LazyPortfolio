"""Tests for the Market Data Hub price-to-return production seam."""

from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

from lazyportfolio.backend import MarketDataHubOptimizationBackend


def _stub_market_data_hub(monkeypatch, prices: pd.DataFrame) -> dict[str, object]:
    """Patch ``market_data_hub`` so ``extract_series`` returns ``prices`` verbatim."""
    calls: dict[str, object] = {}

    extract = types.ModuleType("market_data_hub.extract")

    def extract_series(*args, **kwargs):
        calls.update(kwargs)
        return prices, {"source": "market-data-hub"}

    extract.extract_series = extract_series  # type: ignore[attr-defined]

    root = types.ModuleType("market_data_hub")
    root.extract = extract  # type: ignore[attr-defined]
    identities = types.ModuleType("market_data_hub.lazydatacore")

    class _Domain:
        TICKER = "ticker"

    class _InstrumentId:
        def __init__(self, value: str) -> None:
            self.key = value.split(":", 1)[1]
            self.domain = _Domain.TICKER
            self._value = value

        @classmethod
        def parse(cls, value: str) -> _InstrumentId:
            return cls(value)

        def __str__(self) -> str:
            return self._value

    identities.Domain = _Domain  # type: ignore[attr-defined]
    identities.InstrumentId = _InstrumentId  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "market_data_hub", root)
    monkeypatch.setitem(sys.modules, "market_data_hub.extract", extract)
    monkeypatch.setitem(sys.modules, "market_data_hub.lazydatacore", identities)
    return calls


def test_backend_aligns_price_levels_before_calculating_returns(monkeypatch) -> None:
    """A local-market closure remains a zero-return observation, not a dropped day."""
    index = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
    prices = pd.DataFrame(
        {
            "US": [100.0, 101.0, 102.0],
            "LOCAL": [200.0, None, 202.0],
        },
        index=index,
    )
    calls = _stub_market_data_hub(monkeypatch, prices)

    dataset = MarketDataHubOptimizationBackend().load_returns(["US", "LOCAL"])

    assert calls["transform"] == "level"
    assert dataset.metadata["price_alignment"] == "forward_fill_shared_trading_grid"
    assert dataset.returns.index.tolist() == index.tolist()
    assert pd.isna(dataset.returns.iloc[0]).all()
    assert dataset.returns.loc[index[1], "ticker:LOCAL"] == 0.0
    assert dataset.returns.loc[index[2], "ticker:LOCAL"] == pytest.approx(0.01)


def test_backend_leaves_live_edge_gap_unfilled(monkeypatch) -> None:
    """A missing *most recent* price is left NaN, not reported as today's price."""
    index = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
    prices = pd.DataFrame(
        {
            "US": [100.0, 101.0, 102.0],
            "LOCAL": [200.0, 201.0, None],
        },
        index=index,
    )
    _stub_market_data_hub(monkeypatch, prices)

    dataset = MarketDataHubOptimizationBackend().load_returns(["US", "LOCAL"])

    assert dataset.metadata["filled_price_cells"] == 0
    assert dataset.metadata["trailing_gap_cells"] == 1
    assert dataset.returns.loc[index[1], "ticker:LOCAL"] == pytest.approx(0.005)
    assert pd.isna(dataset.returns.loc[index[2], "ticker:LOCAL"])


def test_backend_caps_interior_gap_fill_at_max_holiday_gap(monkeypatch) -> None:
    """A gap longer than the holiday allowance is left as missing, not stale-filled."""
    index = pd.to_datetime(
        [
            "2026-02-02",
            "2026-02-03",
            "2026-02-04",
            "2026-02-05",
            "2026-02-06",
            "2026-02-07",
            "2026-02-08",
            "2026-02-09",
        ]
    )
    prices = pd.DataFrame(
        {
            "US": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
            # LOCAL goes dark for 6 rows (2026-02-03..2026-02-08), one more than
            # `_MAX_HOLIDAY_GAP`, then resumes trading on 2026-02-09.
            "LOCAL": [200.0, None, None, None, None, None, None, 214.0],
        },
        index=index,
    )
    _stub_market_data_hub(monkeypatch, prices)

    dataset = MarketDataHubOptimizationBackend().load_returns(["US", "LOCAL"])

    assert dataset.metadata["price_alignment_max_gap_days"] == 5
    assert dataset.metadata["filled_price_cells"] == 5
    assert dataset.metadata["trailing_gap_cells"] == 0
    # Filled within the allowance: held flat at the last tradable price.
    assert dataset.returns.loc[index[1], "ticker:LOCAL"] == 0.0
    assert dataset.returns.loc[index[5], "ticker:LOCAL"] == 0.0
    # Beyond the allowance: neither the gap row nor the row that needs it as
    # a "previous price" can produce a real return.
    assert pd.isna(dataset.returns.loc[index[6], "ticker:LOCAL"])
    assert pd.isna(dataset.returns.loc[index[7], "ticker:LOCAL"])


# --------------------------------------------------------------------------- #
# Portfolio reference currency
# --------------------------------------------------------------------------- #


def _stub_market_data_hub_with_currency(
    monkeypatch,
    prices: pd.DataFrame,
    fx_pair_levels: pd.DataFrame,
    currencies: dict[str, str],
) -> None:
    """Like ``_stub_market_data_hub``, but ``extract_series`` dispatches on the
    requested symbols (instrument prices vs FX pairs) and
    ``market_data_hub.db.connection.get_conn`` serves ``listings.currency``."""

    extract = types.ModuleType("market_data_hub.extract")

    def extract_series(symbols, *args, **kwargs):
        if set(symbols) <= set(fx_pair_levels.columns):
            return fx_pair_levels[symbols], {"source": "market-data-hub"}
        return prices[symbols], {"source": "market-data-hub"}

    extract.extract_series = extract_series  # type: ignore[attr-defined]

    identities = types.ModuleType("market_data_hub.lazydatacore")

    class _Domain:
        TICKER = "ticker"

    class _InstrumentId:
        def __init__(self, value: str) -> None:
            self.key = value.split(":", 1)[1]
            self.domain = _Domain.TICKER
            self._value = value

        @classmethod
        def parse(cls, value: str) -> _InstrumentId:
            return cls(value)

        def __str__(self) -> str:
            return self._value

    identities.Domain = _Domain  # type: ignore[attr-defined]
    identities.InstrumentId = _InstrumentId  # type: ignore[attr-defined]

    class _FakeCursor:
        def execute(self, sql: str, params: list[str]) -> _FakeCursor:
            wanted = {p.upper() for p in params}
            self._result = [
                (symbol, currency)
                for symbol, currency in currencies.items()
                if symbol.upper() in wanted
            ]
            return self

        def fetchall(self) -> list[tuple[str, str]]:
            return self._result

        def close(self) -> None:
            pass

    connection = types.ModuleType("market_data_hub.db.connection")
    connection.get_conn = lambda read_only=True: _FakeCursor()  # type: ignore[attr-defined]
    db_module = types.ModuleType("market_data_hub.db")
    db_module.connection = connection  # type: ignore[attr-defined]

    root = types.ModuleType("market_data_hub")
    root.extract = extract  # type: ignore[attr-defined]
    root.db = db_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "market_data_hub", root)
    monkeypatch.setitem(sys.modules, "market_data_hub.extract", extract)
    monkeypatch.setitem(sys.modules, "market_data_hub.lazydatacore", identities)
    monkeypatch.setitem(sys.modules, "market_data_hub.db", db_module)
    monkeypatch.setitem(sys.modules, "market_data_hub.db.connection", connection)


def test_load_returns_without_currency_matches_today_behavior(monkeypatch) -> None:
    index = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
    prices = pd.DataFrame({"SPY": [100.0, 101.0, 102.0]}, index=index)
    _stub_market_data_hub(monkeypatch, prices)

    dataset = MarketDataHubOptimizationBackend().load_returns(["SPY"])

    assert dataset.metadata["reference_currency"] is None
    assert dataset.metadata["instrument_currencies"] is None


def test_load_returns_usd_native_instrument_in_usd_portfolio_is_a_no_op(monkeypatch) -> None:
    """Regression check: a USD-reference portfolio holding only USD-native
    instruments must produce byte-identical returns whether or not currency
    conversion runs."""
    index = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
    prices = pd.DataFrame({"SPY": [100.0, 101.0, 102.0]}, index=index)
    fx_levels = pd.DataFrame(
        {
            "EURUSD=X": [1.10, 1.11, 1.09],
            "GBPUSD=X": [1.27, 1.28, 1.26],
            "USDJPY=X": [150.0, 151.0, 149.0],
        },
        index=index,
    )
    _stub_market_data_hub_with_currency(monkeypatch, prices, fx_levels, {"SPY": "USD"})

    without_currency = MarketDataHubOptimizationBackend().load_returns(["SPY"])
    with_currency = MarketDataHubOptimizationBackend().load_returns(["SPY"], currency="USD")

    assert with_currency.returns["ticker:SPY"].tolist() == pytest.approx(
        without_currency.returns["ticker:SPY"].tolist(), nan_ok=True
    )
    assert with_currency.metadata["reference_currency"] == "USD"
    assert with_currency.metadata["instrument_currencies"] == {"SPY": "USD"}


def test_load_returns_converts_eur_instrument_into_usd_portfolio(monkeypatch) -> None:
    index = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
    prices = pd.DataFrame({"VGK": [50.0, 51.0, 52.0]}, index=index)
    fx_levels = pd.DataFrame(
        {
            "EURUSD=X": [1.10, 1.11, 1.09],
            "GBPUSD=X": [1.27, 1.28, 1.26],
            "USDJPY=X": [150.0, 151.0, 149.0],
        },
        index=index,
    )
    _stub_market_data_hub_with_currency(monkeypatch, prices, fx_levels, {"VGK": "EUR"})

    native = MarketDataHubOptimizationBackend().load_returns(["VGK"])
    converted = MarketDataHubOptimizationBackend().load_returns(["VGK"], currency="USD")

    # 51/50 - 1 = 2% native; once EUR/USD itself moves, the USD-converted
    # return is not the same 2% -- proof conversion actually ran.
    assert native.returns["ticker:VGK"].iloc[1] == pytest.approx(0.02)
    assert converted.returns["ticker:VGK"].iloc[1] != pytest.approx(0.02)
    assert converted.metadata["instrument_currencies"] == {"VGK": "EUR"}


def test_load_returns_raises_naming_instrument_with_unknown_currency(monkeypatch) -> None:
    index = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
    prices = pd.DataFrame({"EEM": [40.0, 41.0, 42.0]}, index=index)
    fx_levels = pd.DataFrame(
        {
            "EURUSD=X": [1.10, 1.11, 1.09],
            "GBPUSD=X": [1.27, 1.28, 1.26],
            "USDJPY=X": [150.0, 151.0, 149.0],
        },
        index=index,
    )
    _stub_market_data_hub_with_currency(monkeypatch, prices, fx_levels, {})

    with pytest.raises(ValueError, match="EEM"):
        MarketDataHubOptimizationBackend().load_returns(["EEM"], currency="USD")
