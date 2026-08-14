"""Public assembly point for the split hierarchical optimizer V2."""

from lazyportfolio.v2.adaptive_pruning import (
    AdaptivePruningPolicy,
    AdaptivePruningResult,
    accumulated_node_metrics,
    run_adaptive_pruning,
    summarize_pruning_decisions,
)
from lazyportfolio.v2.backtest import (
    HierarchicalV2Backtester,
    _V2Ledger,
)
from lazyportfolio.v2.contracts import (
    RECOGNIZED_OBJECTIVES,
    Mode,
    V2Audit,
    V2BacktestReport,
    V2Benchmark,
    V2Constraints,
    V2Estimate,
    V2Fold,
    V2Node,
    V2NodeResult,
    V2OptimizationError,
    V2View,
    effective_setting,
    ticker,
)
from lazyportfolio.v2.hierarchy import HierarchicalV2Estimator
from lazyportfolio.v2.model import V2Model
from lazyportfolio.v2.moments import CASH_BORROW, CASH_LEND
from lazyportfolio.v2.solver import V2LocalOptimizer

_RECOGNIZED_OBJECTIVES = RECOGNIZED_OBJECTIVES
_ticker = ticker

__all__ = [
    "AdaptivePruningPolicy",
    "AdaptivePruningResult",
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
    "_ticker",
    "effective_setting",
    "accumulated_node_metrics",
    "run_adaptive_pruning",
    "summarize_pruning_decisions",
]
