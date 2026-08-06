"""Tests for the Market Data Hub price-to-return production seam."""

from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

from lazyportfolio.backend import MarketDataHubOptimizationBackend


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

    dataset = MarketDataHubOptimizationBackend().load_returns(["US", "LOCAL"])

    assert calls["transform"] == "level"
    assert dataset.metadata["price_alignment"] == "forward_fill_shared_trading_grid"
    assert dataset.returns.index.tolist() == index.tolist()
    assert pd.isna(dataset.returns.iloc[0]).all()
    assert dataset.returns.loc[index[1], "ticker:LOCAL"] == 0.0
    assert dataset.returns.loc[index[2], "ticker:LOCAL"] == pytest.approx(0.01)
