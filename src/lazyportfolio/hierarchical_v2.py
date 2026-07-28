"""Compatibility facade for the split hierarchical optimizer V2.

The implementation lives in ``lazyportfolio.v2``. This module preserves
established import paths without applying runtime patches or replacing engine
class objects during package initialization.
"""

from __future__ import annotations

from typing import Any

from lazyportfolio.v2 import api as _api

CASH_BORROW = _api.CASH_BORROW
CASH_LEND = _api.CASH_LEND
HierarchicalV2Backtester = _api.HierarchicalV2Backtester
Mode = _api.Mode
V2Audit = _api.V2Audit
V2BacktestReport = _api.V2BacktestReport
V2Benchmark = _api.V2Benchmark
V2Constraints = _api.V2Constraints
V2Estimate = _api.V2Estimate
V2Fold = _api.V2Fold
V2LocalOptimizer = _api.V2LocalOptimizer
V2Model = _api.V2Model
V2Node = _api.V2Node
V2NodeResult = _api.V2NodeResult
V2OptimizationError = _api.V2OptimizationError
V2View = _api.V2View
_RECOGNIZED_OBJECTIVES = _api._RECOGNIZED_OBJECTIVES
_V2Ledger = _api._V2Ledger
_effective_setting = _api.effective_setting
_ticker = _api._ticker


class HierarchicalV2Estimator:
    """Compatibility delegate around the canonical split estimator.

    Only historical private helper signatures are adapted here. Numerical
    behavior, hierarchy traversal and state all remain owned by the canonical
    ``lazyportfolio.v2.hierarchy`` implementation.
    """

    def __init__(self, optimiser: V2LocalOptimizer | None = None) -> None:
        self._delegate = _api.HierarchicalV2Estimator(optimiser)

    def estimate(
        self,
        model: V2Model,
        returns: Any,
        *,
        mode: Mode,
        periods_per_year: float,
    ) -> V2Estimate:
        return self._delegate.estimate(
            model,
            returns,
            mode=mode,
            periods_per_year=periods_per_year,
        )

    def estimate_direct_bottom_up(
        self,
        model: V2Model,
        returns: Any,
        periods_per_year: float,
    ) -> V2Estimate:
        return self._delegate.estimate_direct_bottom_up(
            model, returns, periods_per_year,
        )

    @staticmethod
    def _reference(
        node: V2Node,
        model: V2Model,
        returns: Any,
        frame: Any,
        child_columns: dict[str, str],
        synthetic_children: bool,
        reference_kind: str,
        root_reference: Any | None,
    ) -> tuple[Any | None, dict[str, float] | None]:
        del child_columns, synthetic_children
        return _api.HierarchicalV2Estimator._risk_reference(
            node,
            model,
            returns,
            frame,
            reference_kind,
            root_reference,
        )

    @staticmethod
    def _local_series(
        returns: Any,
        local: dict[str, float],
        child_results: dict[str, V2NodeResult],
        child_columns: dict[str, str],
    ) -> Any:
        return _api.HierarchicalV2Estimator._local_series(
            returns,
            local,
            child_results,
            child_columns,
        )

    @staticmethod
    def _compose(
        node: V2Node,
        returns: Any,
        local: dict[str, float],
        audit: V2Audit,
        child_results: dict[str, V2NodeResult],
        child_columns: dict[str, str],
    ) -> V2NodeResult:
        return _api.HierarchicalV2Estimator._compose(
            node,
            returns,
            local,
            audit,
            child_results,
            child_columns,
        )


__all__ = [
    "CASH_BORROW",
    "CASH_LEND",
    "HierarchicalV2Backtester",
    "HierarchicalV2Estimator",
    "Mode",
    "V2Audit",
    "V2BacktestReport",
    "V2Benchmark",
    "V2Constraints",
    "V2Estimate",
    "V2Fold",
    "V2LocalOptimizer",
    "V2Model",
    "V2Node",
    "V2NodeResult",
    "V2OptimizationError",
    "V2View",
    "_RECOGNIZED_OBJECTIVES",
    "_V2Ledger",
    "_effective_setting",
    "_ticker",
]
