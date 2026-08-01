"""Hierarchy traversal and composition for optimizer V2."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal

from lazyportfolio.v2.contracts import (
    RESERVED_ITERATIVE_REFERENCES,
    Mode,
    V2Audit,
    V2Component,
    V2Estimate,
    V2Node,
    V2NodeResult,
    V2OptimizationError,
    V2SolveContext,
    effective_setting,
)
from lazyportfolio.v2.model import V2Model
from lazyportfolio.v2.moments import (
    CASH_LEND,
    CASH_NAMES,
    financing_instrument,
    is_financing_instrument,
)
from lazyportfolio.v2.solver import V2LocalOptimizer
from lazyportfolio.v2.validation import setting_source


class HierarchicalV2Estimator:
    """Estimate flat, forward and forward-backward allocations on one window."""

    def __init__(self, optimiser: V2LocalOptimizer | None = None) -> None:
        self.optimiser = optimiser or V2LocalOptimizer()

    def estimate(
        self,
        model: V2Model,
        returns: Any,
        *,
        mode: Mode,
        periods_per_year: float,
    ) -> V2Estimate:
        if mode == "flat":
            return self._estimate_flat(model, returns, periods_per_year)

        forward: dict[str, V2NodeResult] = {}
        forward_root = self._solve_forward_root_first(
            model,
            returns,
            periods_per_year,
            forward,
        )
        self._finalize_audits(model, forward, forward_root.terminal_weights)

        if mode == "forward":
            return V2Estimate(mode, forward_root.terminal_weights, forward)
        if mode != "forward_backward":
            raise V2OptimizationError(f"unsupported hierarchy mode {mode!r}")

        backward: dict[str, V2NodeResult] = {}
        backward_root = self._solve_backward(
            model.root,
            model,
            returns,
            periods_per_year,
            forward_root.synthetic_returns,
            forward,
            backward,
        )
        self._finalize_audits(model, backward, backward_root.terminal_weights)
        return V2Estimate(
            mode,
            backward_root.terminal_weights,
            backward,
            forward_node_results=forward,
            synthetic_benchmark_weights=self._implemented_benchmark_weights(
                model,
                backward,
            ),
        )

    def _estimate_flat(
        self,
        model: V2Model,
        returns: Any,
        periods_per_year: float,
    ) -> V2Estimate:
        """Flat is a single independent solve over the terminal instruments.

        It has no father/root reference concept of its own (barred from
        ``forward_root_reference`` the same way the root node is) and must
        never depend on the recursive Forward pass succeeding anywhere else
        in the tree. Per-node Forward diagnostics are attached best-effort,
        purely for audit/backtest convenience — if the Forward pass fails for
        any reason, flat's own result is unaffected and those diagnostic
        entries are simply absent.
        """

        forward: dict[str, V2NodeResult] = {}
        try:
            forward_root = self._solve_forward_root_first(
                model, returns, periods_per_year, forward,
            )
            self._finalize_audits(model, forward, forward_root.terminal_weights)
        except V2OptimizationError:
            forward = {}

        node_results = dict(forward)
        flat = V2Node(
            id="__global_flat__",
            name="Global flat terminal allocation",
            instruments=model.root.terminal_instruments(),
            children=[],
            proxy=None,
            objective=model.root.objective,
            constraints=model.root.constraints,
        )
        local, audit, child_columns = self._solve_local(
            flat,
            model,
            returns,
            periods_per_year,
            forward_root_reference=None,
            child_results={},
            synthetic_children=False,
        )
        result = self._compose(
            flat,
            returns,
            local,
            audit,
            {},
            child_columns,
            periods_per_year=periods_per_year,
        )
        self._finalize_single_audit(result, result.terminal_weights)
        node_results[flat.name] = result
        return V2Estimate("flat", result.terminal_weights, node_results)

    def estimate_direct_bottom_up(
        self,
        model: V2Model,
        returns: Any,
        periods_per_year: float,
    ) -> V2Estimate:
        """Solve every leaf directly, then compose bottom-up - no Forward pass.

        This is the "direct bottom-up" resolver the methodology docs require
        as an independent proof that the Forward pass is a diagnostic, never
        a hidden dependency of the final Backward result: it never calls
        ``_solve_forward_root_first`` and never uses a
        ``forward_root_reference``. It must equal ``forward_backward``'s own
        result within numerical tolerance for any tree whose nodes do not
        request ``forward_root_reference`` (which the standard, non-iterative
        contract does not support at internal/leaf nodes in the first place).
        """

        leaves: dict[str, V2NodeResult] = {}

        def solve_leaf(node: V2Node) -> None:
            if not node.children:
                local, audit, child_columns = self._solve_local(
                    node,
                    model,
                    returns,
                    periods_per_year,
                    forward_root_reference=None,
                    child_results={},
                    synthetic_children=False,
                )
                leaves[node.name] = self._compose(
                    node, returns, local, audit, {}, child_columns,
                    periods_per_year=periods_per_year,
                )
                return
            for child in node.children:
                solve_leaf(child)

        solve_leaf(model.root)
        backward: dict[str, V2NodeResult] = {}
        root_result = self._solve_backward(
            model.root, model, returns, periods_per_year, None, leaves, backward,
        )
        self._finalize_audits(model, backward, root_result.terminal_weights)
        return V2Estimate("forward_backward", root_result.terminal_weights, backward)

    @staticmethod
    def _implemented_benchmark_weights(
        model: V2Model,
        node_results: dict[str, V2NodeResult],
    ) -> dict[str, float]:
        children_by_proxy = {
            child.proxy: node_results[child.name].terminal_weights
            for child in model.root.children
            if child.proxy is not None
        }
        terminal: dict[str, float] = {}
        for instrument, benchmark_weight in model.benchmark.weights.items():
            child_weights = children_by_proxy.get(instrument)
            if child_weights is None:
                terminal[instrument] = terminal.get(instrument, 0.0) + benchmark_weight
                continue
            for child_instrument, child_weight in child_weights.items():
                terminal[child_instrument] = (
                    terminal.get(child_instrument, 0.0)
                    + benchmark_weight * child_weight
                )
        return {
            name: weight
            for name, weight in terminal.items()
            if abs(weight) > 1e-12
        }

    def _solve_forward_root_first(
        self,
        model: V2Model,
        returns: Any,
        periods_per_year: float,
        output: dict[str, V2NodeResult],
    ) -> V2NodeResult:
        root = model.root
        local, audit, child_columns = self._solve_local(
            root,
            model,
            returns,
            periods_per_year,
            forward_root_reference=None,
            child_results={},
            synthetic_children=False,
        )
        root_proxy_series = self._local_series(
            returns,
            local,
            {},
            child_columns,
            audit=audit,
            periods_per_year=periods_per_year,
        )
        children = {
            child.name: self._solve_forward_child(
                child,
                model,
                returns,
                periods_per_year,
                root_proxy_series,
                output,
            )
            for child in root.children
        }
        result = self._compose(
            root,
            returns,
            local,
            audit,
            children,
            child_columns,
            periods_per_year=periods_per_year,
        )
        output[root.name] = result
        return result

    def _solve_forward_child(
        self,
        node: V2Node,
        model: V2Model,
        returns: Any,
        periods_per_year: float,
        forward_root_reference: Any,
        output: dict[str, V2NodeResult],
    ) -> V2NodeResult:
        local, audit, child_columns = self._solve_local(
            node,
            model,
            returns,
            periods_per_year,
            forward_root_reference=forward_root_reference,
            child_results={},
            synthetic_children=False,
        )
        children = {
            child.name: self._solve_forward_child(
                child,
                model,
                returns,
                periods_per_year,
                forward_root_reference,
                output,
            )
            for child in node.children
        }
        result = self._compose(
            node,
            returns,
            local,
            audit,
            children,
            child_columns,
            periods_per_year=periods_per_year,
        )
        output[node.name] = result
        return result

    def _solve_backward(
        self,
        node: V2Node,
        model: V2Model,
        returns: Any,
        periods_per_year: float,
        forward_root_reference: Any,
        forward: dict[str, V2NodeResult],
        output: dict[str, V2NodeResult],
    ) -> V2NodeResult:
        if not node.children:
            source = forward[node.name]
            result = V2NodeResult(
                dict(source.local_weights),
                dict(source.terminal_weights),
                source.synthetic_returns,
                source.audit,
            )
        else:
            children = {
                child.name: self._solve_backward(
                    child,
                    model,
                    returns,
                    periods_per_year,
                    forward_root_reference,
                    forward,
                    output,
                )
                for child in node.children
            }
            local, audit, child_columns = self._solve_local(
                node,
                model,
                returns,
                periods_per_year,
                forward_root_reference=forward_root_reference,
                child_results=children,
                synthetic_children=True,
            )
            result = self._compose(
                node,
                returns,
                local,
                audit,
                children,
                child_columns,
                periods_per_year=periods_per_year,
            )
        output[node.name] = result
        return result

    def _solve_local(
        self,
        node: V2Node,
        model: V2Model,
        returns: Any,
        periods_per_year: float,
        *,
        forward_root_reference: Any | None,
        child_results: dict[str, V2NodeResult],
        synthetic_children: bool,
    ) -> tuple[dict[str, float], V2Audit, dict[str, str]]:
        import pandas as pd

        pass_kind: Literal["forward", "backward", "flat"] = (
            "flat"
            if node.id == "__global_flat__"
            else "backward"
            if synthetic_children
            else "forward"
        )
        context = V2SolveContext(pass_kind=pass_kind)
        aliases: dict[str, str] = {}
        child_columns: dict[str, str] = {}
        columns: dict[str, Any] = {}
        for instrument in node.instruments:
            component_id = f"direct:{instrument}"
            context.components[component_id] = V2Component(
                id=component_id, kind="direct", raw_series_key=instrument
            )
            raw_series = returns[instrument]
            context.raw_proxy_series_by_component[component_id] = raw_series
            context.candidate_series_by_component[component_id] = raw_series
            context.component_to_solver_column[component_id] = instrument
            context.solver_column_to_component[instrument] = component_id
            columns[instrument] = raw_series
        for child in node.children:
            if child.proxy is None:
                raise V2OptimizationError(f"{child.name}: child proxy is required")
            if child.name in child_columns:
                raise V2OptimizationError(
                    f"{node.name}: two children share the name {child.name!r}; "
                    "sibling children under the same parent must have distinct "
                    "names, they are used as result keys"
                )
            component_id = f"child:{child.name}"
            context.components[component_id] = V2Component(
                id=component_id,
                kind="child",
                raw_series_key=child.proxy,
                child_id=child.name,
            )
            context.raw_proxy_series_by_component[component_id] = returns[child.proxy]
            if synthetic_children:
                column = f"{child.proxy}_SYNTH"
                synthetic_series = child_results[child.name].synthetic_returns
                context.synthetic_series_by_component[component_id] = synthetic_series
                context.candidate_series_by_component[component_id] = synthetic_series
            else:
                column = child.proxy
                context.candidate_series_by_component[component_id] = returns[child.proxy]
            if column in columns:
                raise V2OptimizationError(
                    f"{node.name}: child {child.name!r}'s solver column {column!r} "
                    "collides with another candidate already in this node's local "
                    "frame (a direct instrument, or a sibling child sharing the "
                    "same proxy); each candidate column in one node's local solve "
                    "must be unique, or weights silently double up"
                )
            columns[column] = context.candidate_series_by_component[component_id]
            if synthetic_children:
                aliases[column] = child.proxy
            context.component_to_solver_column[component_id] = column
            context.solver_column_to_component[column] = component_id
            child_columns[child.name] = column
        frame = pd.DataFrame(columns).dropna(how="any")

        target_reference, target_weights = self._risk_reference(
            node,
            model,
            returns,
            frame,
            node.constraints.volatility_reference,
            forward_root_reference,
        )
        cap_reference, cap_weights = self._risk_reference(
            node,
            model,
            returns,
            frame,
            node.constraints.max_volatility_reference,
            forward_root_reference,
        )
        tracking_reference, tracking_weights = self._risk_reference(
            node,
            model,
            returns,
            frame,
            node.constraints.tracking_error_reference,
            forward_root_reference,
        )
        mean_reference_weights, mean_reference_source = self._mean_reference(
            node, model, context,
        )
        risk_reference_source, risk_reference_weights = next(
            (
                (label, weights)
                for label, weights in (
                    ("volatility_reference", target_weights),
                    ("max_volatility_reference", cap_weights),
                    ("tracking_error_reference", tracking_weights),
                )
                if weights
            ),
            ("none", None),
        )
        # A risk reference (target/cap/TEV) is never reused as the mean
        # reference: configuring TEV, a volatility target or a volatility cap
        # must never implicitly select the equilibrium expected-return prior.
        # An explicit mean_reference_kind is the only way to populate one.
        reference_weights = mean_reference_weights
        risk_free_rate = effective_setting(
            node.constraints.risk_free_rate,
            model.root.constraints.risk_free_rate,
            0.0,
        )
        risk_aversion = effective_setting(
            node.constraints.risk_aversion,
            model.root.constraints.risk_aversion,
            1.0,
        )
        try:
            local, audit = self.optimiser.solve(
                frame,
                objective=node.objective,
                constraints=node.constraints,
                periods_per_year=periods_per_year,
                target_reference_series=target_reference,
                cap_reference_series=cap_reference,
                tracking_reference_series=tracking_reference,
                reference_weights=reference_weights,
                bound_aliases=aliases,
                risk_aversion=risk_aversion,
                risk_free_rate=risk_free_rate,
            )
        except V2OptimizationError as exc:
            raise V2OptimizationError(
                f"{node.name} ({pass_kind}): {exc} "
                f"[tracking_error_policy={node.constraints.tracking_error_policy!r}, "
                f"volatility_target_policy={node.constraints.volatility_target_policy!r}, "
                f"max_tracking_error={node.constraints.max_tracking_error!r}, "
                f"objective={node.objective!r}]"
            ) from exc
        cash_name = next((name for name in CASH_NAMES if name in local), "")
        cash_weight = float(local.get(cash_name, 0.0)) if cash_name else 0.0
        risky_gross = float(
            sum(weight for name, weight in local.items() if name not in CASH_NAMES)
        )
        financing_regime = audit.financing_regime
        if abs(cash_weight) <= 1e-10:
            financing_regime = "fully_invested"
            if cash_name:
                local.pop(cash_name, None)
            cash_name = ""
            cash_weight = 0.0
        return (
            local,
            replace(
                audit,
                risk_aversion_source=setting_source(
                    node.constraints.risk_aversion,
                    model.root.constraints.risk_aversion,
                ),
                risk_free_rate_source=setting_source(
                    node.constraints.risk_free_rate,
                    model.root.constraints.risk_free_rate,
                ),
                cash_enabled=node.constraints.cash_enabled,
                cash_instrument=(
                    financing_instrument(
                        cash_name,
                        node.id,
                        is_root=node.proxy is None,
                    )
                    if cash_name
                    else ""
                ),
                cash_weight=cash_weight,
                risky_gross_exposure=risky_gross,
                max_leverage=node.constraints.max_leverage,
                cash_lending_rate=risk_free_rate,
                cash_borrowing_rate=(
                    risk_free_rate + node.constraints.borrow_spread_bps / 10_000.0
                ),
                borrow_spread_bps=node.constraints.borrow_spread_bps,
                financing_regime=financing_regime,
                cash_enabled_source=node.constraints.cash_enabled_source,
                max_leverage_source=node.constraints.max_leverage_source,
                borrow_spread_bps_source=node.constraints.borrow_spread_bps_source,
                component_id=f"node:{node.name}",
                pass_kind=pass_kind,
                candidate_frame_composition={
                    component_id: (
                        "synthetic"
                        if component_id in context.synthetic_series_by_component
                        else "raw"
                    )
                    for component_id in context.components
                },
                mean_reference_source=mean_reference_source,
                risk_reference_source=risk_reference_source,
            ),
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
        *,
        periods_per_year: float = 252.0,
    ) -> V2NodeResult:
        local_cash = [name for name in CASH_NAMES if name in local]
        if local_cash and not audit.cash_enabled:
            raise V2OptimizationError(
                "cannot compose terminal series; missing returns: financing audit"
            )
        terminal = {instrument: local[instrument] for instrument in node.instruments}
        for child in node.children:
            parent_weight = local[child_columns[child.name]]
            for instrument, child_weight in child_results[child.name].terminal_weights.items():
                terminal[instrument] = (
                    terminal.get(instrument, 0.0) + parent_weight * child_weight
                )
        for cash_name in CASH_NAMES:
            if cash_name in local:
                ledger_name = financing_instrument(
                    cash_name,
                    node.id,
                    is_root=node.proxy is None,
                )
                terminal[ledger_name] = terminal.get(ledger_name, 0.0) + local[cash_name]
        terminal = {
            key: value for key, value in terminal.items() if abs(value) > 1e-12
        }
        missing = sorted(
            name
            for name in terminal
            if not is_financing_instrument(name) and name not in returns.columns
        )
        if missing:
            raise V2OptimizationError(
                f"cannot compose terminal series; missing returns: {missing}"
            )
        synthetic = HierarchicalV2Estimator._local_series(
            returns,
            local,
            child_results,
            child_columns,
            audit=audit,
            periods_per_year=periods_per_year,
        )
        return V2NodeResult(local, terminal, synthetic, audit)

    @staticmethod
    def _local_series(
        returns: Any,
        local: dict[str, float],
        child_results: dict[str, V2NodeResult],
        child_columns: dict[str, str],
        *,
        audit: V2Audit | None = None,
        periods_per_year: float = 252.0,
    ) -> Any:
        import pandas as pd

        result = None
        for column, weight in local.items():
            source = returns[column] if column in returns.columns else None
            if source is None:
                for name, child_column in child_columns.items():
                    if child_column == column:
                        source = child_results[name].synthetic_returns
                        break
            if source is None and column in CASH_NAMES and audit is not None:
                annual_rate = (
                    audit.cash_lending_rate
                    if column == CASH_LEND
                    else audit.cash_borrowing_rate
                )
                source = pd.Series(
                    annual_rate / periods_per_year,
                    index=returns.index,
                    dtype=float,
                )
            if source is None:
                raise V2OptimizationError(f"cannot resolve local series {column}")
            contribution = source * weight
            result = contribution if result is None else result + contribution
        if result is None:
            raise V2OptimizationError("cannot compose an empty local portfolio")
        return result

    @staticmethod
    def _risk_reference(
        node: V2Node,
        model: V2Model,
        returns: Any,
        frame: Any,
        reference_kind: str,
        forward_root_reference: Any | None,
    ) -> tuple[Any | None, dict[str, float] | None]:
        """Resolve a risk reference (volatility target/cap or TEV).

        Independent of :meth:`_mean_reference` — a risk reference must never
        be reused as a mean reference (or vice versa) by sharing a fallback
        value between them.
        """
        if reference_kind in RESERVED_ITERATIVE_REFERENCES:
            raise V2OptimizationError(
                f"reference {reference_kind!r} is reserved for a future iterative "
                "hierarchy mode and is not supported in flat, forward or "
                "forward_backward mode"
            )
        if reference_kind in {"none", "manual", "declared"}:
            return None, None
        if reference_kind == "forward_root_reference":
            if node is model.root or node.id == "__global_flat__":
                raise V2OptimizationError(
                    "forward_root_reference is invalid on the root node"
                )
            if forward_root_reference is None:
                raise V2OptimizationError(
                    "forward_root_reference is not available in this pass"
                )
            return forward_root_reference.reindex(frame.index), None
        if reference_kind == "benchmark":
            if set(model.benchmark.weights).issubset(frame.columns):
                series = (
                    frame.loc[:, list(model.benchmark.weights)]
                    .mul(model.benchmark.weights, axis="columns")
                    .sum(axis="columns")
                )
                return series, dict(model.benchmark.weights)
            if not set(model.benchmark.weights).issubset(returns.columns):
                missing = sorted(set(model.benchmark.weights) - set(returns.columns))
                raise V2OptimizationError(
                    f"benchmark reference series missing: {missing}"
                )
            series = (
                returns.loc[:, list(model.benchmark.weights)]
                .mul(model.benchmark.weights, axis="columns")
                .sum(axis="columns")
            )
            return series.reindex(frame.index), None
        if reference_kind == "father_proxy":
            if node.proxy is None:
                raise V2OptimizationError(
                    f"{node.name}: father reference requires a proxy"
                )
            series = returns[node.proxy].reindex(frame.index)
            weights = {node.proxy: 1.0} if node.proxy in frame.columns else None
            return series, weights
        raise V2OptimizationError(f"unsupported reference {reference_kind!r}")

    @staticmethod
    def _mean_reference(
        node: V2Node,
        model: V2Model,
        context: V2SolveContext,
    ) -> tuple[dict[str, float] | None, str]:
        """Resolve the strategic-weight reference for equilibrium expected
        returns, independent of any risk reference (:meth:`_risk_reference`).

        Never invents a weight for a component absent from the declared
        reference, and never falls back to a risk reference: an incomplete or
        absent mean reference simply resolves to ``None`` (letting the caller
        fall back to a non-equilibrium mean estimator), it does not borrow a
        risk-side weight.
        """

        kind = node.constraints.mean_reference_kind
        if kind == "none":
            return None, "none"
        if kind == "benchmark":
            resolved = HierarchicalV2Estimator._resolve_component_weights_from_raw_map(
                node, context, dict(model.benchmark.weights), "benchmark"
            )
            return resolved, "benchmark"
        if kind == "local_weights":
            declared = node.constraints.mean_reference_weights or {}
            resolved = HierarchicalV2Estimator._resolve_component_weights_from_raw_map(
                node, context, declared, "local_weights"
            )
            return resolved, "local_weights"
        raise V2OptimizationError(f"unsupported mean_reference_kind {kind!r}")

    @staticmethod
    def _resolve_component_weights_from_raw_map(
        node: V2Node,
        context: V2SolveContext,
        declared: dict[str, float],
        kind_label: str,
    ) -> dict[str, float]:
        """Map a raw-ticker-keyed weight map onto this node's *current*
        solver columns, via component identity — never via column-name
        matching against ``frame.columns`` directly.

        This is what makes ``mean_reference_kind="benchmark"`` (declared with
        raw benchmark tickers) resolve correctly in *both* Forward (raw proxy
        columns) and Backward (``_SYNTH`` columns): the raw ticker is matched
        to the component's stable identity, then mapped to whatever solver
        column currently represents that component in this pass — not to the
        raw ticker string itself, which does not exist as a column in
        Backward. Requires full coverage of the node's actual solved universe
        and rejects unknown extra keys; never invents a residual weight.
        """

        components = list(context.components.values())
        resolved: dict[str, float] = {}
        for component in components:
            if component.raw_series_key not in declared:
                raise V2OptimizationError(
                    f"{node.name}: mean_reference_kind={kind_label!r} is missing an "
                    f"entry for {component.raw_series_key!r}; a complete mean "
                    "reference must cover every component in this node's solved "
                    "universe, with no invented weight for an absent sleeve"
                )
            solver_column = context.component_to_solver_column[component.id]
            resolved[solver_column] = declared[component.raw_series_key]
        unknown = sorted(
            set(declared) - {component.raw_series_key for component in components}
        )
        if unknown:
            raise V2OptimizationError(
                f"{node.name}: mean_reference_kind={kind_label!r} declares "
                f"component(s) not in this node's solved universe: {unknown}"
            )
        return resolved

    @classmethod
    def _finalize_audits(
        cls,
        model: V2Model,
        results: dict[str, V2NodeResult],
        terminal_weights: dict[str, float],
    ) -> None:
        portfolio_risky, portfolio_cash, portfolio_net = cls._portfolio_exposure(
            terminal_weights
        )

        def walk(node: V2Node, global_weight: float, parent_weight: float) -> None:
            result = results[node.name]
            result.audit = replace(
                result.audit,
                parent_weight=parent_weight,
                global_node_weight=global_weight,
                global_risky_gross_exposure=(
                    global_weight * result.audit.risky_gross_exposure
                ),
                global_cash_weight=global_weight * result.audit.cash_weight,
                portfolio_risky_gross_exposure=portfolio_risky,
                portfolio_cash_weight=portfolio_cash,
                portfolio_net_exposure=portfolio_net,
            )
            for child in node.children:
                child_weight = cls._child_weight(result, child)
                walk(child, global_weight * child_weight, child_weight)

        walk(model.root, 1.0, 1.0)

    @classmethod
    def _finalize_single_audit(
        cls,
        result: V2NodeResult,
        terminal_weights: dict[str, float],
    ) -> None:
        portfolio_risky, portfolio_cash, portfolio_net = cls._portfolio_exposure(
            terminal_weights
        )
        result.audit = replace(
            result.audit,
            parent_weight=1.0,
            global_node_weight=1.0,
            global_risky_gross_exposure=result.audit.risky_gross_exposure,
            global_cash_weight=result.audit.cash_weight,
            portfolio_risky_gross_exposure=portfolio_risky,
            portfolio_cash_weight=portfolio_cash,
            portfolio_net_exposure=portfolio_net,
        )

    @staticmethod
    def _portfolio_exposure(
        terminal_weights: dict[str, float],
    ) -> tuple[float, float, float]:
        risky = float(
            sum(
                weight
                for name, weight in terminal_weights.items()
                if not is_financing_instrument(name)
            )
        )
        cash = float(
            sum(
                weight
                for name, weight in terminal_weights.items()
                if is_financing_instrument(name)
            )
        )
        return risky, cash, risky + cash

    @staticmethod
    def _child_weight(parent_result: V2NodeResult, child: V2Node) -> float:
        if child.proxy is None:
            raise V2OptimizationError(f"{child.name}: child proxy is required")
        for key in (child.proxy, f"{child.proxy}_SYNTH"):
            if key in parent_result.local_weights:
                return float(parent_result.local_weights[key])
        raise V2OptimizationError(
            f"{child.name}: parent allocation does not contain child proxy"
        )


__all__ = ["HierarchicalV2Estimator"]
