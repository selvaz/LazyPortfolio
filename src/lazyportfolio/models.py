"""Public, serialisable protocol contract for the V2 walk-forward backtester.

The contract intentionally identifies a return dataset by its universe and
window; it never contains observations.  Prices and returns stay in
market-data-hub and enter the process only through an internal data backend.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

SUPPORTED_FREQUENCIES = frozenset({"D", "W", "M", "Q"})


class _PortfolioModel(BaseModel):
    """Base for LazyPortfolio's own pydantic contracts.

    ``extra="forbid"`` turns silent typos into validation errors and
    ``validate_default`` makes sure constrained defaults are checked too —
    mirrors ``lazyfin.model.common.LazyFinModel`` without depending on the
    LazyFin package.
    """

    model_config = ConfigDict(extra="forbid", validate_default=True)


class BacktestSpec(_PortfolioModel):
    """Walk-forward protocol with independent estimation and rebalance grids.

    ``train_size`` is measured in estimation-frequency return observations.
    Out-of-sample performance is always valued from canonical daily returns;
    ``rebalance_frequency`` controls how long each fitted allocation is held.
    """

    id: str
    train_size: int = Field(default=252, ge=2)
    rebalance_frequency: str = "W"
    include_partial_last_period: bool = False

    @model_validator(mode="after")
    def _validate_protocol(self) -> BacktestSpec:
        if self.rebalance_frequency not in SUPPORTED_FREQUENCIES:
            raise ValueError("rebalance_frequency must be one of D, W, M, Q")
        return self
