"""Audit ZIP and standalone HTML report builders for Tree Studio V2."""

# ruff: noqa: E501

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from html import escape
from numbers import Real
from typing import Any


def _safe(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _safe(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def _redact(value: Any) -> Any:
    sensitive = ("token", "secret", "password", "api_key", "apikey")
    if isinstance(value, dict):
        return {
            str(key): "<redacted>"
            if any(part in str(key).lower() for part in sensitive)
            else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(_safe(value), indent=2, sort_keys=True) + "\n").encode("utf-8")


def _csv_bytes(header: list[str], rows: Any) -> bytes:
    def cell(value: Any) -> Any:
        if isinstance(value, Real) and not isinstance(value, bool):
            return format(float(value), ".15g")
        return value

    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(header)
    writer.writerows([cell(value) for value in row] for row in rows)
    return buffer.getvalue().encode("utf-8")


def _estimate_payload(estimate: Any) -> dict[str, Any]:
    def nodes(results: dict[str, Any]) -> dict[str, Any]:
        return {
            name: {
                "local_weights": result.local_weights,
                "terminal_weights": result.terminal_weights,
                "audit": asdict(result.audit),
                "synthetic_series_name": f"RESULT_OUTPUT:{name}",
            }
            for name, result in results.items()
        }

    return {
        "mode": estimate.mode,
        "terminal_weights": estimate.terminal_weights,
        "synthetic_benchmark_weights": estimate.synthetic_benchmark_weights,
        "nodes": nodes(estimate.node_results),
        "forward_nodes": nodes(estimate.forward_node_results),
    }


def build_audit_bundle(
    *,
    config: dict[str, Any],
    data_metadata: dict[str, Any],
    daily_returns: Any,
    estimate: Any,
    report: Any,
    scientific_study: Any | None = None,
) -> bytes:
    """Return a compressed, reconstruction-ready audit bundle."""
    files: dict[str, bytes] = {}

    def add(name: str, body: bytes) -> None:
        files[name] = body

    add("configuration.json", _json_bytes(_redact(config)))
    add("data_metadata.json", _json_bytes(_redact(data_metadata)))
    add("point_estimate.json", _json_bytes(_estimate_payload(estimate)))
    add(
        "series/raw_daily_returns.csv",
        daily_returns.to_csv(index_label="date", float_format="%.15g").encode("utf-8"),
    )
    add(
        "backtest/metrics.csv",
        _csv_bytes(
            ["arm", "cagr", "annualized_volatility", "annualized_sharpe", "max_drawdown", "n_obs"],
            ([arm, *[metrics.get(key) for key in (
                "cagr", "annualized_volatility", "annualized_sharpe", "max_drawdown", "n_obs"
            )]] for arm, metrics in report.metrics.items()),
        ),
    )
    add(
        "backtest/curves.csv",
        _csv_bytes(
            ["arm", "date", "return"],
            (
                [arm, day, value]
                for arm, curve in report.curves.items()
                for day, value in curve.items()
            ),
        ),
    )
    add(
        "backtest/folds.csv",
        _csv_bytes(
            ["signal", "training_start", "training_end", "holding_start", "holding_end"],
            (
                [fold.signal, fold.training_start, fold.training_end, fold.holding_start, fold.holding_end]
                for fold in report.folds
            ),
        ),
    )
    add(
        "backtest/weights.csv",
        _csv_bytes(
            ["signal", "target", "instrument", "weight"],
            (
                [fold.signal, target, instrument, weight]
                for fold in report.folds
                for target, weights in fold.targets.items()
                for instrument, weight in weights.items()
            ),
        ),
    )
    audit_rows = []
    for fold in report.folds:
        for stage, audits in (("result", fold.audits), ("forward", fold.forward_audits)):
            for node, audit in audits.items():
                body = asdict(audit)
                audit_rows.append(
                    [
                        fold.signal,
                        stage,
                        node,
                        body["configured_objective"],
                        body["effective_objective"],
                        body["expected_return_annualized"],
                        body["objective_value"],
                        body["target_reference"],
                        body["target_volatility"],
                        body["actual_volatility"],
                        body["target_status"],
                        body["cap_reference"],
                        body["volatility_cap"],
                        body["tracking_error_limit"],
                        body["actual_tracking_error"],
                        body["tracking_error_status"],
                        body["soft_constraint_violation"],
                        body["sum_weights"],
                        body["solver_message"],
                        body["configured_mean_estimator"],
                        body["resolved_mean_estimator"],
                        body["views_applied"],
                        json.dumps(_safe(body["view_details"]), sort_keys=True),
                        body["risk_aversion"],
                        body["risk_aversion_source"],
                        body["risk_free_rate"],
                        body["risk_free_rate_source"],
                        json.dumps(_safe(body["minimum_slack"]), sort_keys=True),
                        json.dumps(_safe(body["maximum_slack"]), sort_keys=True),
                        body["component_id"],
                        body["pass_kind"],
                        json.dumps(_safe(body["candidate_frame_composition"]), sort_keys=True),
                        body["mean_reference_source"],
                        body["risk_reference_source"],
                        json.dumps(_safe(body["constraint_stage_results"]), sort_keys=True),
                        body["financing_regime"],
                        body["cash_enabled"],
                        body["cash_enabled_source"],
                        body["max_leverage"],
                        body["max_leverage_source"],
                        body["borrow_spread_bps"],
                        body["borrow_spread_bps_source"],
                        body["cash_lending_rate"],
                        body["cash_borrowing_rate"],
                    ]
                )
    add(
        "backtest/audits.csv",
        _csv_bytes(
            [
                "signal", "stage", "node", "configured_objective", "effective_objective",
                "expected_return_annualized", "objective_value", "target_reference",
                "target_volatility", "actual_volatility", "target_status", "cap_reference",
                "volatility_cap", "tracking_error_limit", "actual_tracking_error",
                "tracking_error_status", "soft_constraint_violation", "sum_weights",
                "solver_message", "configured_mean_estimator", "resolved_mean_estimator",
                "views_applied", "view_details_json",
                "risk_aversion", "risk_aversion_source",
                "risk_free_rate", "risk_free_rate_source",
                "minimum_slack_json", "maximum_slack_json",
                "component_id", "pass_kind", "candidate_frame_composition_json",
                "mean_reference_source", "risk_reference_source",
                "constraint_stage_results_json",
                "financing_regime", "cash_enabled", "cash_enabled_source",
                "max_leverage", "max_leverage_source",
                "borrow_spread_bps", "borrow_spread_bps_source",
                "cash_lending_rate", "cash_borrowing_rate",
            ],
            audit_rows,
        ),
    )
    add(
        "backtest/candidate_series.csv",
        _csv_bytes(
            ["signal", "solve", "position", "series"],
            (
                [fold.signal, solve, position, series]
                for fold in report.folds
                for solve, names in fold.candidate_series.items()
                for position, series in enumerate(names)
            ),
        ),
    )
    add(
        "series/fold_estimation_series.csv",
        _csv_bytes(
            ["signal", "series", "date", "return"],
            (
                [fold.signal, name, day, value]
                for fold in report.folds
                for name, series in fold.estimation_series.items()
                for day, value in series.items()
            ),
        ),
    )
    add(
        "backtest/transaction_costs.csv",
        _csv_bytes(
            ["arm", "total_cost_fraction"],
            ([arm, value] for arm, value in report.transaction_cost_paid.items()),
        ),
    )
    scientific_study_manifest: dict[str, Any] | None = None
    if scientific_study is not None:
        add(
            "scientific_study/comparisons.csv",
            _csv_bytes(
                [
                    "candidate", "baseline", "annualized_mean_difference",
                    "ci_low", "ci_high", "p_value", "holm_adjusted_p_value",
                ],
                (
                    [
                        item.candidate, item.baseline, item.annualized_mean_difference,
                        item.confidence_interval_low, item.confidence_interval_high,
                        item.p_value, item.holm_adjusted_p_value,
                    ]
                    for item in scientific_study.comparisons
                ),
            ),
        )
        add(
            "scientific_study/metrics.csv",
            _csv_bytes(
                ["arm", "cagr", "annualized_volatility", "annualized_sharpe", "max_drawdown", "n_obs"],
                ([arm, *[metrics.get(key) for key in (
                    "cagr", "annualized_volatility", "annualized_sharpe", "max_drawdown", "n_obs"
                )]] for arm, metrics in scientific_study.metrics.items()),
            ),
        )
        add(
            "scientific_study/curves.csv",
            _csv_bytes(
                ["arm", "date", "return"],
                (
                    [arm, day, value]
                    for arm, curve in scientific_study.curves.items()
                    for day, value in curve.items()
                ),
            ),
        )
        scientific_study_manifest = {
            "fold_count": scientific_study.fold_count,
            "common_oos_start": _safe(scientific_study.common_oos_start),
            "common_oos_end": _safe(scientific_study.common_oos_end),
            "protocol": _safe(scientific_study.protocol),
            "dropped_observations": scientific_study.dropped_observations,
        }
    add(
        "README.md",
        (
            b"# LazyFin V2 audit bundle\n\n"
            b"This bundle is sufficient to reconstruct every V2 decision.\n\n"
            b"- `configuration.json`: redacted Tree Studio model.\n"
            b"- `series/raw_daily_returns.csv`: original daily return matrix.\n"
            b"- `series/fold_estimation_series.csv`: every raw, reference, synthetic and diagnostic series by fold.\n"
            b"- `backtest/candidate_series.csv`: ordered input columns of every local solve.\n"
            b"- `backtest/weights.csv`: local, composed, benchmark and final weights per fold "
            b"-- rows with target `LOCAL:<node>` are that node's final (backward) local weight "
            b"history, `FORWARD_LOCAL:<node>` its forward-pass-only local weight history (a "
            b"`_SYNTH` suffix on the instrument marks a child's synthetic series).\n"
            b"- `backtest/audits.csv`: objectives, constraints, solver result and independent audit.\n"
            b"- `backtest/curves.csv`: all OOS arm returns on the common ledger grid.\n"
            b"- `scientific_study/comparisons.csv`: block-bootstrap paired comparison vs each required baseline (CI + Holm-adjusted p-value); only present when the study was enabled.\n"
            b"- `scientific_study/metrics.csv`, `scientific_study/curves.csv`: point metrics and OOS curves for V2_FINAL, every baseline and the representation/bottom-up ablations.\n"
            b"- `manifest.json`: file hashes and immutable reference policy.\n"
        ),
    )
    manifest = {
        "schema": "lazyfin-hierarchical-v2-audit-1",
        "created_at": datetime.now(UTC).isoformat(),
        "mode": report.mode,
        "reference_policy": "immutable_raw",
        "folds": len(report.folds),
        "scientific_study": scientific_study_manifest,
        "files": {
            name: {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
            for name, body in sorted(files.items())
        },
    }
    add("manifest.json", _json_bytes(manifest))
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, body in sorted(files.items()):
            archive.writestr(name, body)
    return output.getvalue()


def _pct(value: Any) -> str:
    return "-" if value is None else f"{float(value):.2%}"


def _num(value: Any) -> str:
    return "-" if value is None else f"{float(value):.3f}"


def _tree_html(config: dict[str, Any]) -> str:
    nodes = {str(node["id"]): node for node in config["nodes"]}

    def draw(node_id: str) -> str:
        node = nodes[node_id]
        proxy = f"<span>Proxy: {escape(str(node.get('proxy')))}</span>" if node.get("proxy") else ""
        instruments = ", ".join(map(str, node.get("instruments") or [])) or "Nessun ticker diretto"
        children = "".join(draw(str(child)) for child in node.get("children") or [])
        return (
            '<li><div class="tree-node"><strong>' + escape(str(node.get("name") or node_id))
            + "</strong>" + proxy + "<small>" + escape(instruments) + "</small></div>"
            + (f"<ul>{children}</ul>" if children else "") + "</li>"
        )

    return f'<div class="tree"><ul>{draw(str(config["root_id"]))}</ul></div>'


def _setting(value: Any, *, percentage: bool = False) -> str:
    if value in (None, "", {}, []):
        return "-"
    if percentage:
        return _pct(value)
    return escape(str(value))


def _weight_limits(constraints: dict[str, Any]) -> str:
    parts = []
    if constraints.get("per_asset_cap") not in (None, ""):
        parts.append(f"cap {_pct(constraints['per_asset_cap'])}")
    for label, key in (("min", "min_weights"), ("max", "max_weights")):
        values = constraints.get(key) or {}
        if values:
            rendered = ", ".join(
                f"{escape(str(instrument))} {_pct(weight)}"
                for instrument, weight in sorted(values.items())
            )
            parts.append(f"{label}: {rendered}")
    return "; ".join(parts) or "-"


def _node_settings_html(config: dict[str, Any]) -> str:
    rows = []
    for node in config["nodes"]:
        goal = node.get("goal") or {}
        constraints = node.get("constraints") or {}
        target_reference = constraints.get("volatility_reference") or "none"
        target_value = (
            _setting(constraints.get("vol_target"), percentage=True)
            if target_reference == "manual"
            else "calcolato per fold" if target_reference != "none" else "-"
        )
        cap_reference = constraints.get("max_volatility_reference") or "none"
        cap_value = (
            _setting(constraints.get("max_volatility"), percentage=True)
            if cap_reference == "manual"
            else "calcolato per fold" if cap_reference != "none" else "-"
        )
        tev = constraints.get("max_tracking_error")
        tev_text = "-" if tev in (None, "") else (
            f"{_pct(tev)} vs {escape(str(constraints.get('tracking_error_reference') or 'declared'))}"
        )
        rows.append(
            "<tr><td>" + escape(str(node.get("name") or node.get("id")))
            + "</td><td>" + escape(str(goal.get("objective") or "min_risk"))
            + "</td><td>" + escape(str(goal.get("risk_measure") or "variance"))
            + "</td><td>" + escape(str(target_reference)) + " / " + target_value
            + "</td><td>" + escape(str(cap_reference)) + " / " + cap_value
            + "</td><td>" + tev_text
            + "</td><td>" + escape(str(constraints.get("mean_estimator") or "auto"))
            + "</td><td>" + _weight_limits(constraints) + "</td></tr>"
        )
    return "".join(rows)


def _curve_svg(report: Any, arms: list[str]) -> str:
    selected = [arm for arm in arms if arm in report.curves]
    if not selected:
        return ""
    width, height, left, top = 920, 300, 52, 20
    plot_width, plot_height = 844, 235
    wealth = {}
    for arm in selected:
        values = (1.0 + report.curves[arm]).cumprod() * 100.0
        wealth[arm] = values
    minimum = min(float(values.min()) for values in wealth.values())
    maximum = max(float(values.max()) for values in wealth.values())
    span = max(maximum - minimum, 1e-9)
    colors = ["#126782", "#c28416", "#15765b", "#9b3f4a"]
    paths = []
    for index, (_arm, values) in enumerate(wealth.items()):
        points = []
        for position, value in enumerate(values):
            x = left + position * plot_width / max(len(values) - 1, 1)
            y = top + (maximum - float(value)) * plot_height / span
            points.append(f"{x:.2f},{y:.2f}")
        paths.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[index % len(colors)]}" stroke-width="2"/>'
        )
    legend = "".join(
        f'<span><i style="background:{colors[index % len(colors)]}"></i>{escape(arm)}</span>'
        for index, arm in enumerate(selected)
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Performance cumulata">'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" class="axis"/>'
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" class="axis"/>'
        + "".join(paths) + "</svg><div class=\"legend\">" + legend + "</div>"
    )


def _weight_lines_svg(dates: list[str], series: dict[str, list[float]], *, y_max: float | None = None) -> str:
    """Line chart of weight fractions over time, one line per instrument."""
    if not series or not dates:
        return ""
    width, height, left, top = 920, 260, 52, 20
    plot_width, plot_height = 844, 195
    n = len(dates)
    resolved_max = max((max(values) if values else 0.0) for values in series.values())
    resolved_max = max(y_max or resolved_max, resolved_max, 0.01) * 1.08
    colors = ["#126782", "#c28416", "#15765b", "#9b3f4a", "#6a4c93", "#1f8a70", "#c74e4e", "#4c6a92", "#a37c27", "#3f7f5f", "#7a5230", "#2f6690"]
    paths = []
    for index, values in enumerate(series.values()):
        points = []
        for position, value in enumerate(values):
            x = left + position * plot_width / max(n - 1, 1)
            y = top + (resolved_max - float(value)) * plot_height / resolved_max
            points.append(f"{x:.2f},{y:.2f}")
        paths.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[index % len(colors)]}" stroke-width="1.6"/>'
        )
    legend = "".join(
        f'<span><i style="background:{colors[index % len(colors)]}"></i>{escape(label)}</span>'
        for index, label in enumerate(series.keys())
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Pesi nel tempo">'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" class="axis"/>'
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" class="axis"/>'
        + "".join(paths) + '</svg><div class="legend">' + legend + "</div>"
    )


def _weight_history_html(config: dict[str, Any], report: Any) -> str:
    """Per-branch-node weight history: forward (on the raw proxy) vs backward
    (final, on the child's _SYNTH series when that child was expanded).

    Both are already recorded by the engine every fold as
    ``fold.targets["FORWARD_LOCAL:<name>"]`` / ``fold.targets["LOCAL:<name>"]``
    -- this only reads and charts them, it computes nothing new.
    """

    by_id = {str(node["id"]): node for node in config["nodes"]}
    blocks = []
    for node in config["nodes"]:
        node_id = str(node["id"])
        children_ids = node.get("children") or []
        if not children_ids:
            continue
        name = str(node.get("name") or node_id)
        forward_key = f"FORWARD_LOCAL:{name}"
        backward_key = f"LOCAL:{name}"
        if not any(backward_key in fold.targets for fold in report.folds):
            continue
        has_forward = any(forward_key in fold.targets for fold in report.folds)
        dates = [
            str(fold.signal.date()) if hasattr(fold.signal, "date") else str(fold.signal)
            for fold in report.folds
        ]
        candidate_tickers = [str(t) for t in (node.get("instruments") or [])]
        for child_id in children_ids:
            child = by_id.get(str(child_id))
            if child and child.get("proxy"):
                candidate_tickers.append(str(child["proxy"]))

        def series_for(key: str) -> dict[str, list[float]]:
            out: dict[str, list[float]] = {}
            for ticker in candidate_tickers:
                bare = ticker.replace("ticker:", "").upper()
                values: list[float] = []
                for fold in report.folds:
                    weights = fold.targets.get(key, {})
                    match = None
                    for raw_key, raw_value in weights.items():
                        candidate = str(raw_key).replace("ticker:", "").upper()
                        if candidate in (bare, f"{bare}_SYNTH"):
                            match = raw_value
                            break
                    values.append(float(match) if match is not None else 0.0)
                if any(value != 0.0 for value in values):
                    out[ticker] = values
            return out

        backward_series = series_for(backward_key)
        if not backward_series:
            continue
        forward_series = series_for(forward_key) if has_forward else {}
        y_max = max(
            [max(values) for values in backward_series.values()]
            + [max(values) for values in forward_series.values()]
            + [1.0]
        )
        backward_chart = _weight_lines_svg(dates, backward_series, y_max=y_max)
        forward_block = ""
        if forward_series:
            forward_chart = _weight_lines_svg(dates, forward_series, y_max=y_max)
            forward_block = f'<h4 style="margin-top:14px">Forward (sul proxy raw)</h4>{forward_chart}'
        blocks.append(
            f"<details class=\"arm-block\"><summary>{escape(name)}</summary><div>"
            f'<p class="hint">{len(dates)} fold, {escape(dates[0] if dates else "-")} - {escape(dates[-1] if dates else "-")}.</p>'
            f"<h4>Backward (composizione finale, serie _SYNTH per i figli espansi)</h4>{backward_chart}"
            f"{forward_block}</div></details>"
        )
    if not blocks:
        return ""
    return (
        "<h2>Pesi storici per nodo</h2>"
        '<p class="note">Per ogni nodo con figli, il peso locale assegnato a ogni componente '
        "(ticker diretto, o proxy/sintetico di una sleeve figlia) a ogni ribilanciamento. "
        '"Backward" e la composizione finale (usa le serie _SYNTH per i figli espansi); '
        '"Forward" e il passaggio diagnostico sul solo proxy raw, disponibile solo in modalita '
        "forward_backward.</p>" + "".join(blocks)
    )


def _node_value_comparison_html(config: dict[str, Any], report: Any) -> str:
    """Per-node proxy vs forward vs backward comparison, with a chart each.

    Uses the arms the backtest engine already produces for every node with a
    proxy (``FATHER:<name>`` = buy-and-hold the proxy ticker,
    ``FORWARD_NODE:<name>`` = that node's own forward-pass optimization,
    ``NODE:<name>`` = its final backward-reconstructed composition) -- no new
    computation, just a report-side view onto what the engine already ran.
    ``FORWARD_NODE:`` only exists in forward_backward mode; a plain forward
    backtest still shows proxy vs the single (forward) result.
    """

    root_id = str(config["root_id"])
    blocks = []
    for node in config["nodes"]:
        if str(node["id"]) == root_id:
            continue
        proxy = node.get("proxy")
        if not proxy:
            continue
        name = str(node.get("name") or node["id"])
        arms = [
            (f"FATHER:{name}", f"Proxy ({escape(str(proxy))})"),
            (f"FORWARD_NODE:{name}", "Forward (ottimizzazione locale)"),
            (f"NODE:{name}", "Backward (composizione finale)"),
        ]
        available = [(arm, label) for arm, label in arms if arm in report.metrics]
        if len(available) < 2:
            continue
        rows = "".join(
            "<tr><td>" + escape(label) + "</td><td>" + _pct(report.metrics[arm].get("cagr"))
            + "</td><td>" + _pct(report.metrics[arm].get("annualized_volatility")) + "</td><td>"
            + _num(report.metrics[arm].get("annualized_sharpe")) + "</td><td>"
            + _pct(report.metrics[arm].get("max_drawdown")) + "</td></tr>"
            for arm, label in available
        )
        chart = _curve_svg(report, [arm for arm, _label in available])
        blocks.append(
            f'<details class="arm-block"><summary>{escape(name)}</summary><div>'
            f'<table><thead><tr><th>Serie</th><th>CAGR</th><th>Vol</th><th>Sharpe</th><th>Max DD</th></tr></thead><tbody>{rows}</tbody></table>'
            f"{chart}</div></details>"
        )
    if not blocks:
        return ""
    return (
        "<h2>Valore per nodo — proxy vs forward vs backward</h2>"
        '<p class="note">Ogni sleeve confrontata con il proprio proxy (solo il ticker dichiarato, buy-and-hold), '
        "con la propria ottimizzazione forward e con la composizione backward finale: misura dove "
        "l'ottimizzazione aggiunge (o toglie) valore rispetto a tenere semplicemente il proxy.</p>"
        + "".join(blocks)
    )


def _scientific_study_html(scientific_study: Any | None) -> str:
    if scientific_study is None:
        return ""
    rows = "".join(
        "<tr><td>" + escape(item.candidate) + "</td><td>" + escape(item.baseline)
        + "</td><td>" + _pct(item.annualized_mean_difference) + "</td><td>"
        + _pct(item.confidence_interval_low) + " / " + _pct(item.confidence_interval_high)
        + "</td><td>" + f"{item.p_value:.4f}" + "</td><td>" + f"{item.holm_adjusted_p_value:.4f}" + "</td></tr>"
        for item in scientific_study.comparisons
    )
    metric_rows = "".join(
        "<tr><td>" + escape(arm) + "</td><td>" + _pct(metrics.get("cagr"))
        + "</td><td>" + _pct(metrics.get("annualized_volatility")) + "</td><td>"
        + _num(metrics.get("annualized_sharpe")) + "</td><td>"
        + _pct(metrics.get("max_drawdown")) + "</td></tr>"
        for arm, metrics in scientific_study.metrics.items()
    )
    return f"""<h2>Studio scientifico - block-bootstrap vs baseline</h2>
<p class="note">{scientific_study.fold_count} fold comuni, {escape(str(scientific_study.common_oos_start))} - {escape(str(scientific_study.common_oos_end))}. Ogni riga confronta il portafoglio finale (V2_FINAL) con una baseline obbligatoria sugli stessi fold OOS: differenza media annualizzata, intervallo di confidenza al 95% e p-value da block-bootstrap, con correzione di Holm per confronti multipli. Un p-value Holm sotto 0.05 indica un vantaggio statisticamente significativo dopo la correzione.</p>
<table><thead><tr><th>Candidato</th><th>Baseline</th><th>Diff. media ann.</th><th>CI 95%</th><th>p-value</th><th>p-value Holm</th></tr></thead><tbody>{rows}</tbody></table>
<h3 style="margin-top:20px">Metriche per arm (V2, baseline, ablation)</h3>
<table><thead><tr><th>Arm</th><th>CAGR</th><th>Vol</th><th>Sharpe</th><th>Max DD</th></tr></thead><tbody>{metric_rows}</tbody></table>"""


def build_client_report(
    *, config: dict[str, Any], data_metadata: dict[str, Any], estimate: Any, report: Any,
    scientific_study: Any | None = None,
) -> bytes:
    """Return a self-contained client-facing HTML report."""
    root = next(node for node in config["nodes"] if str(node["id"]) == str(config["root_id"]))
    title = str(root.get("name") or "Hierarchical allocation")
    backtest = config.get("backtest") or {}
    benchmark = backtest.get("benchmark") or {}
    benchmark_text = " + ".join(
        f"{float(weight):.0%} {ticker}" for ticker, weight in (benchmark.get("weights") or {}).items()
        if float(weight) != 0.0
    )
    metric_rows = "".join(
        "<tr><td>" + escape(arm) + "</td><td>" + _pct(metrics.get("cagr"))
        + "</td><td>" + _pct(metrics.get("annualized_volatility")) + "</td><td>"
        + _num(metrics.get("annualized_sharpe")) + "</td><td>"
        + _pct(metrics.get("max_drawdown")) + "</td></tr>"
        for arm, metrics in report.metrics.items()
        if arm in {"B0", "B0_SYNTH", "FORWARD_FINAL", "FINAL"}
    )
    settings_rows = _node_settings_html(config)
    node_rows = "".join(
        "<tr><td>" + escape(name) + "</td><td>" + escape(audit.configured_objective)
        + "</td><td>" + _pct(audit.expected_return_annualized) + "</td><td>"
        + _pct(audit.actual_volatility) + "</td><td>" + _pct(audit.actual_tracking_error)
        + "</td><td>" + escape(audit.target_status) + " / "
        + escape(audit.tracking_error_status) + "</td></tr>"
        for name, audit in report.folds[-1].audits.items()
    )
    final_weights = "".join(
        f"<tr><td>{escape(name)}</td><td>{_pct(weight)}</td></tr>"
        for name, weight in sorted(estimate.terminal_weights.items(), key=lambda item: -item[1])
    )
    nearest = sum(
        audit.target_status == "nearest_feasible" or audit.tracking_error_status == "nearest_feasible"
        for fold in report.folds for audit in fold.audits.values()
    )
    configuration_json = escape(
        json.dumps(_safe(_redact(config)), indent=2, sort_keys=True)
    )
    node_value_comparison_html = _node_value_comparison_html(config, report)
    weight_history_html = _weight_history_html(config, report)
    scientific_study_html = _scientific_study_html(scientific_study)
    created = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    html = f"""<!doctype html>
<html lang="it"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)} - LazyFin report</title><style>
:root{{--ink:#17212b;--muted:#62727c;--line:#d9e0e3;--blue:#126782;--gold:#c28416;--green:#15765b}}
*{{box-sizing:border-box}} body{{margin:0;color:var(--ink);font:14px/1.5 Inter,Arial,sans-serif;background:#f4f7f8}}
header{{padding:36px 7vw 30px;color:white;background:#102a34;border-bottom:5px solid var(--gold)}}
header small{{text-transform:uppercase;letter-spacing:1.2px;color:#b9ccd3}} h1{{margin:8px 0 5px;font-size:32px}}
main{{max-width:1180px;margin:auto;background:white;padding:34px 5vw 60px}} h2{{margin:32px 0 14px;border-bottom:2px solid var(--blue);padding-bottom:7px}}
.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}} .kpi{{padding:14px;border:1px solid var(--line);border-top:3px solid var(--blue)}}
.kpi b{{display:block;font-size:22px}} table{{width:100%;border-collapse:collapse;margin:12px 0 20px}} th,td{{padding:8px;border-bottom:1px solid var(--line);text-align:right}} th:first-child,td:first-child{{text-align:left}}
.tree ul{{list-style:none;padding-left:28px}} .tree>ul{{padding-left:0}} .tree-node{{display:inline-grid;gap:2px;min-width:230px;margin:5px;padding:9px 12px;border:1px solid #a9c3ca;border-left:4px solid var(--blue)}} .tree-node span,.tree-node small{{color:var(--muted)}}
svg{{width:100%;height:auto;border:1px solid var(--line);background:white}} .axis{{stroke:#9aaab1}} .legend{{display:flex;gap:16px;flex-wrap:wrap}} .legend i{{display:inline-block;width:10px;height:10px;margin-right:5px}}
.note{{padding:12px;border-left:4px solid var(--gold);background:#fff8e8}} footer{{margin-top:38px;color:var(--muted);font-size:12px}}
details{{margin-top:16px;border:1px solid var(--line);padding:10px 14px}} summary{{cursor:pointer;font-weight:700}} pre{{max-height:520px;overflow:auto;padding:12px;background:#f4f7f8;font:12px/1.45 Consolas,monospace;white-space:pre-wrap}}
@media(max-width:760px){{.kpis{{grid-template-columns:1fr 1fr}} main{{padding:24px 18px}}}}
@media print{{body{{background:white}} main{{max-width:none}}}}
</style></head><body><header><small>LazyFin Hierarchical Allocation</small><h1>{escape(title)}</h1>
<div>{escape(str(benchmark.get('name') or 'B0'))}: {escape(benchmark_text)} | {escape(report.mode)}</div></header><main>
<section class="kpis"><div class="kpi">CAGR finale<b>{_pct(report.metrics['FINAL']['cagr'])}</b></div>
<div class="kpi">Volatilita<b>{_pct(report.metrics['FINAL']['annualized_volatility'])}</b></div>
<div class="kpi">Sharpe<b>{_num(report.metrics['FINAL']['annualized_sharpe'])}</b></div>
<div class="kpi">Max drawdown<b>{_pct(report.metrics['FINAL']['max_drawdown'])}</b></div></section>
<h2>Mandato e protocollo</h2><table><tbody><tr><td>Benchmark</td><td>{escape(benchmark_text)}</td></tr>
<tr><td>Politica riferimenti</td><td>B0 e father raw immutabili</td></tr><tr><td>Finestra</td><td>{escape(str(backtest.get('train_size')))} osservazioni {escape(str(backtest.get('estimation_frequency')))}</td></tr>
<tr><td>Ribilanciamento</td><td>{escape(str(backtest.get('rebalance_frequency')))}</td></tr><tr><td>Periodo OOS</td><td>{escape(str(report.curves['FINAL'].index.min().date()))} - {escape(str(report.curves['FINAL'].index.max().date()))}</td></tr></tbody></table>
<h2>Albero di allocazione</h2>{_tree_html(config)}
<h2>Impostazioni dei nodi</h2><table><thead><tr><th>Nodo</th><th>Obiettivo</th><th>Rischio</th><th>Target vol ann.</th><th>Cap vol ann.</th><th>TEV ann.</th><th>Stima media</th><th>Limiti locali</th></tr></thead><tbody>{settings_rows}</tbody></table>
<h2>Risultati walk-forward</h2><table><thead><tr><th>Strategia</th><th>CAGR</th><th>Vol</th><th>Sharpe</th><th>Max DD</th></tr></thead><tbody>{metric_rows}</tbody></table>
{_curve_svg(report, ['B0','B0_SYNTH','FORWARD_FINAL','FINAL'])}
<p class="note"><b>B0_SYNTH e diagnostico.</b> Mantiene i pesi strategici del benchmark sulle sleeve ottimizzate. Target-vol, cap e TEV usano sempre B0 raw.</p>
<h2>Ultima decisione per nodo</h2><table><thead><tr><th>Nodo</th><th>Obiettivo</th><th>Rendimento atteso</th><th>Vol</th><th>TEV</th><th>Esito</th></tr></thead><tbody>{node_rows}</tbody></table>
{node_value_comparison_html}
{weight_history_html}
<h2>Pesi terminali correnti</h2><table><thead><tr><th>Strumento</th><th>Peso</th></tr></thead><tbody>{final_weights}</tbody></table>
<h2>Qualita dell'esecuzione</h2><p>{len(report.folds)} fold comuni; {nearest} decisioni nearest-feasible. Tutte le curve usano lo stesso calendario OOS e ledger con drift.</p>
{scientific_study_html}
<details><summary>Appendice - configurazione completa redatta</summary><pre>{configuration_json}</pre></details>
<footer>Generato {created}. Dati: {escape(str(data_metadata.get('date_start') or '-'))} - {escape(str(data_metadata.get('date_end') or '-'))}. Questo report descrive un backtest e non costituisce consulenza finanziaria.</footer>
</main></body></html>"""
    return html.encode("utf-8")
