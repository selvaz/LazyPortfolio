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
