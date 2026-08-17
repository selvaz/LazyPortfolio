"""Shared deterministic-pruning evaluation.

Used by both the on-demand CLI (``scripts/prune_tree_by_father.py``) and the
nightly rolling-vs-expanding job (``scripts/rolling_vs_expanding_backtest.py``),
so a tree gets its pruning evidence refreshed as part of the regular
unattended cycle instead of only when someone remembers to run the CLI by
hand.  ``run_variant`` is injected rather than imported from either caller,
so this module never has to know which one is running it and the two
scripts never import each other.
"""

# ruff: noqa: E501 -- build_report_html embeds the same CSS-in-f-string
# client report style as tree_studio_v2/exports.py (which carries the same
# suppression for the same reason: a styled HTML template's lines are not
# meaningfully improved by wrapping at 100 columns).

from __future__ import annotations

import html
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

import numpy as np  # noqa: E402
from project import tree_studio  # noqa: E402
from project.tree_studio_v2.exports import _curve_svg, _num, _pct  # noqa: E402

from lazyportfolio.calendar import _annualization_factor, _resample_simple_returns  # noqa: E402
from lazyportfolio.scientific_study import _holm_adjust, paired_block_bootstrap  # noqa: E402
from lazyportfolio.v2 import run_history, store  # noqa: E402
from lazyportfolio.v2.tree_pruning import PruningRule, prune_config, rule_payload  # noqa: E402

RunVariant = Callable[..., dict[str, Any]]


def _global_guard(
    baseline: dict[str, Any], candidate: dict[str, Any], rule: PruningRule
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for protocol in rule.required_protocols:
        before = baseline[protocol]["metrics"].get("FINAL", {})
        after = candidate[protocol]["metrics"].get("FINAL", {})
        sharpe_delta = float(after.get("annualized_sharpe", 0.0)) - float(before.get("annualized_sharpe", 0.0))
        before_vol = float(before.get("annualized_volatility", 0.0))
        after_vol = float(after.get("annualized_volatility", 0.0))
        before_dd_per_vol = abs(float(before.get("max_drawdown", 0.0))) / before_vol if before_vol > 0 else float("inf")
        after_dd_per_vol = abs(float(after.get("max_drawdown", 0.0))) / after_vol if after_vol > 0 else float("inf")
        if sharpe_delta < 0.0 or after_dd_per_vol > before_dd_per_vol * rule.max_drawdown_per_vol_ratio:
            reasons.append(
                f"{protocol}: FINAL ΔSharpe {sharpe_delta:.3f}, "
                f"MDD/vol {after_dd_per_vol:.3f} vs {before_dd_per_vol:.3f}"
            )
    return not reasons, reasons


def _metrics_table_html(rows: list[tuple[str, dict[str, Any]]]) -> str:
    body = "".join(
        "<tr><td>" + html.escape(label) + "</td><td>" + _pct(metrics.get("cagr"))
        + "</td><td>" + _pct(metrics.get("annualized_volatility")) + "</td><td>"
        + _num(metrics.get("annualized_sharpe")) + "</td><td>"
        + _pct(metrics.get("max_drawdown")) + "</td></tr>"
        for label, metrics in rows
    )
    return (
        "<table><thead><tr><th>Arm</th><th>CAGR</th><th>Vol</th><th>Sharpe</th>"
        f"<th>Max DD</th></tr></thead><tbody>{body}</tbody></table>"
    )


def _describe_action(action: dict[str, Any] | None, id_to_name: dict[str, str]) -> str:
    if not action:
        return "unchanged"
    if "contracted_into_parent" in action:
        parent = id_to_name.get(action["contracted_into_parent"], action["contracted_into_parent"])
        proxy = action.get("promoted_proxy")
        return f"contracted into {parent}" + (f"; proxy '{proxy}' promoted" if proxy else "")
    if "lifted_from_pruned_ancestor" in action:
        raw_ancestor = action["lifted_from_pruned_ancestor"]
        ancestor = id_to_name.get(raw_ancestor, raw_ancestor)
        return f"lifted past pruned parent {ancestor}"
    if "removed_with_ancestor" in action:
        ancestor = id_to_name.get(action["removed_with_ancestor"], action["removed_with_ancestor"])
        return f"removed together with {ancestor}"
    return html.escape(json.dumps(action))


def _protocol_curve_charts(
    source: dict[str, Any], candidate: dict[str, Any], rule: PruningRule
) -> dict[str, str]:
    """Baseline-vs-candidate FINAL curve chart per required protocol, reusing
    ``_run_full_backtest``'s in-memory cache (keyed on config content +
    ``expanding``, not on ``max_workers`` or the tree's store name -- see its
    docstring in ``tree_studio.py``) so this is a cache *hit* off
    ``run_variant``'s own calls above, not a second walk-forward run.  Never
    fatal: a chart is a nice-to-have on top of the pruning decision itself.
    """
    charts: dict[str, str] = {}
    for protocol in rule.required_protocols:
        expanding = protocol == "expanding"
        try:
            _, _, baseline_report = tree_studio._run_full_backtest(
                source, capture_audit_series=False, expanding=expanding
            )
            _, _, candidate_report = tree_studio._run_full_backtest(
                candidate, capture_audit_series=False, expanding=expanding
            )
        except Exception:  # noqa: BLE001 -- chart is a bonus, never fatal
            continue
        reports_by_label = (("Baseline", baseline_report), ("Candidate (pruned)", candidate_report))
        curves = {
            label: report.curves["FINAL"]
            for label, report in reports_by_label
            if "FINAL" in report.curves
        }
        if len(curves) < 2:
            continue
        svg = _curve_svg(SimpleNamespace(curves=curves), list(curves.keys()))
        if svg:
            charts[protocol] = svg
    return charts


def build_report_html(
    payload: dict[str, Any],
    id_to_name: dict[str, str] | None = None,
    curve_charts: dict[str, str] | None = None,
) -> bytes:
    """Self-contained pruning report, styled like ``build_client_report`` in
    ``tree_studio_v2.exports`` (same CSS variables/layout, KPI cards, curve
    chart) so it reads as part of the same family of reports rather than a
    bare debug dump -- but kept its own separate document, sent as its own
    Telegram attachment, since the pruning evidence is conceptually a gate
    decision on top of a tree, not a replacement for that tree's own client
    report.

    ``curve_charts`` (protocol -> pre-rendered ``_curve_svg`` HTML, built by
    the caller from the exact baseline/candidate FINAL curves) is kept out of
    ``payload`` deliberately: ``payload`` is also what gets persisted via
    ``run_history.record_run``, which must stay JSON-safe, while the chart
    HTML is already a plain string with no such constraint.
    """
    id_to_name = id_to_name or {}
    curve_charts = curve_charts or {}
    rows = []
    for decision in payload["decisions"]:
        badge = "prune" if decision["decision"] == "prune" else "retain"
        cells = "".join(
            f"<td>{html.escape(str(decision.get(key, '')))}</td>"
            for key in ("node_name", "proxy")
        )
        rows.append(
            f"<tr>{cells}"
            f"<td class='badge-{badge}'>{html.escape(decision['decision'])}</td>"
            f"<td>{html.escape(str(decision.get('reason', '')))}</td>"
            f"<td>{html.escape(_describe_action(decision.get('action'), id_to_name))}</td></tr>"
        )
    status = "PROMOTED" if payload["global_guard_passed"] else "NOT PROMOTED"
    pruned_count = sum(1 for item in payload["decisions"] if item["decision"] == "prune")
    protocol_sections = []
    for protocol in sorted(payload["rule"].get("required_protocols", ())):
        before = payload["baseline_final"].get(protocol)
        after = payload["candidate_final"].get(protocol)
        rows_for_protocol = [
            (label, metrics)
            for label, metrics in (("Baseline", before), ("Candidate (pruned)", after))
            if metrics
        ]
        if not rows_for_protocol:
            continue
        protocol_sections.append(
            f"<h3>{html.escape(protocol.capitalize())} window</h3>"
            f"{_metrics_table_html(rows_for_protocol)}"
            f"{curve_charts.get(protocol, '')}"
        )
    guard_reasons = payload["global_guard_reasons"]
    guard_items = "".join(f"<li>{html.escape(reason)}</li>" for reason in guard_reasons)
    guard_html = (
        "<p>No issues -- every required protocol cleared the guard.</p>"
        if not guard_reasons
        else f"<ul>{guard_items}</ul>"
    )
    min_sharpe = payload["rule"].get("min_sharpe_improvement")
    max_dd_ratio = payload["rule"].get("max_drawdown_per_vol_ratio")
    protocols = ", ".join(payload["rule"].get("required_protocols", ()))
    created = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(payload['source_tree'])} -- pruning report</title><style>
:root{{--ink:#17212b;--muted:#62727c;--line:#d9e0e3;--blue:#126782;--gold:#c28416;--green:#15765b;--red:#a4262c}}
*{{box-sizing:border-box}} body{{margin:0;color:var(--ink);font:14px/1.5 Inter,Arial,sans-serif;background:#f4f7f8}}
header{{padding:36px 7vw 30px;color:white;background:#102a34;border-bottom:5px solid var(--gold)}}
header small{{text-transform:uppercase;letter-spacing:1.2px;color:#b9ccd3}} h1{{margin:8px 0 5px;font-size:32px}}
main{{max-width:1180px;margin:auto;background:white;padding:34px 5vw 60px}} h2{{margin:32px 0 14px;border-bottom:2px solid var(--blue);padding-bottom:7px}}
h3{{margin:22px 0 8px}}
.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}} .kpi{{padding:14px;border:1px solid var(--line);border-top:3px solid var(--blue)}}
.kpi b{{display:block;font-size:22px}} table{{width:100%;border-collapse:collapse;margin:12px 0 20px}} th,td{{padding:8px;border-bottom:1px solid var(--line);text-align:right}} th:first-child,td:first-child{{text-align:left}}
svg{{width:100%;height:auto;border:1px solid var(--line);background:white}} .axis{{stroke:#9aaab1}} .legend{{display:flex;gap:16px;flex-wrap:wrap;margin:6px 0 18px}} .legend i{{display:inline-block;width:10px;height:10px;margin-right:5px}}
.badge-prune{{color:var(--red);font-weight:700}} .badge-retain{{color:var(--green);font-weight:700}}
footer{{margin-top:38px;color:var(--muted);font-size:12px}}
@media(max-width:760px){{.kpis{{grid-template-columns:1fr 1fr}} main{{padding:24px 18px}}}}
@media print{{body{{background:white}} main{{max-width:none}}}}
</style></head><body><header><small>LazyFin Hierarchical Allocation</small>
<h1>{html.escape(payload['source_tree'])}</h1>
<div>Deterministic pruning: {html.escape(status)}</div></header><main>
<section class="kpis">
<div class="kpi">Guard<b>{html.escape(status)}</b></div>
<div class="kpi">Branches cut<b>{pruned_count} / {len(payload['decisions'])}</b></div>
<div class="kpi">Min &Delta;Sharpe<b>{min_sharpe}</b></div>
<div class="kpi">Max MDD/vol ratio<b>{max_dd_ratio}</b></div>
</section>
<h2>Mandato e regola</h2>
<table><tbody>
<tr><td>Candidate tree</td><td>{html.escape(payload['candidate_tree'])}</td></tr>
<tr><td>Required protocols</td><td>{html.escape(protocols)}</td></tr>
<tr><td>Persisted</td><td>{html.escape(str(payload.get('persisted_tree') or '-'))}</td></tr>
</tbody></table>
<h2>Baseline vs candidate (FINAL)</h2>
{''.join(protocol_sections)}
<h2>Branch decisions</h2>
<table><thead><tr><th>Branch</th><th>Proxy</th><th>Decision</th><th>Evidence</th><th>Action</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<h2>Whole-tree guard</h2>
{guard_html}
<footer>Generated {created}.</footer>
</main></body></html>""".encode("utf-8")


def evaluate_pruning(
    tree_name: str,
    run_variant: RunVariant,
    *,
    rule: PruningRule = PruningRule(),
    write_tree: bool = False,
    output_name: str | None = None,
    baseline: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Classify every branch of ``tree_name`` and, if requested, persist the
    guarded candidate under a new name.  The source tree is never touched.

    ``baseline`` lets a caller that already ran both protocols for this
    exact tree this cycle (the nightly job) hand over that work instead of
    paying for it twice; the CLI leaves it unset and computes it here.
    """
    source = store.read_model(tree_name)
    if baseline is None:
        # Not user-facing on its own -- only the final pruning summary
        # (built below) is worth a Telegram message.
        baseline = {
            label: run_variant(tree_name, expanding=(label == "expanding"), send_telegram=False)
            for label in rule.required_protocols
        }
    candidate, decisions = prune_config(
        source, {label: item["metrics"] for label, item in baseline.items()}, rule
    )
    output_name = output_name or f"{tree_name} - Pruned"
    # A temporary distinct name lets the normal runner exercise precisely the proposed tree.
    temp_name = f"{output_name} - Candidate"
    store.write_model(temp_name, candidate)
    try:
        candidate_runs = {
            label: run_variant(temp_name, expanding=(label == "expanding"), send_telegram=False)
            for label in rule.required_protocols
        }
    finally:
        # It exists only to reuse the production runner.  The source and the
        # promoted final tree are never overwritten; a failed candidate is not
        # left visible in Tree Studio either.
        store.delete_model(temp_name)
    passed, guard_reasons = _global_guard(baseline, candidate_runs, rule)
    persisted_name = store.write_model(output_name, candidate) if write_tree and passed else None
    data_as_of, fingerprint = tree_studio._data_fingerprint(source)
    payload = {
        "created_at": datetime.now(UTC).isoformat(), "source_tree": tree_name,
        "candidate_tree": output_name, "persisted_tree": persisted_name,
        "rule": rule_payload(rule), "decisions": decisions,
        "global_guard_passed": passed, "global_guard_reasons": guard_reasons,
        "baseline_final": {k: v["metrics"].get("FINAL", {}) for k, v in baseline.items()},
        "candidate_final": {k: v["metrics"].get("FINAL", {}) for k, v in candidate_runs.items()},
    }
    run_id = run_history.record_run(
        cache_key=f"tree-pruning:{store.sanitize_model_name(tree_name)}:{datetime.now(UTC).isoformat()}",
        path="/scripts/pruning_runner", kind="tree_pruning", tree_id=store.sanitize_model_name(tree_name),
        config_hash=tree_studio._config_hash(source), data_as_of=data_as_of, data_fingerprint=fingerprint,
        weights=None, metrics={"global_guard_passed": passed}, payload=payload,
    )
    id_to_name = {node["id"]: node.get("name", node["id"]) for node in source["nodes"]}
    curve_charts = _protocol_curve_charts(source, candidate, rule)
    report_html = build_report_html(payload, id_to_name, curve_charts)
    run_history.attach_artifact(
        run_id, kind="report", content_type="text/html",
        filename="tree_pruning_report.html", blob=report_html,
    )
    payload["run_id"] = run_id
    payload["report_html"] = report_html
    return payload


def significance_report(
    curves: dict[str, Any],
    comparisons: list[tuple[str, str]],
    *,
    samples: int = 2_000,
    block_size: int | None = None,
    random_seed: int = 7,
    resample_frequency: str | None = None,
) -> list[dict[str, Any]]:
    """Block-bootstrap + Holm-adjusted significance for a set of candidate-
    vs-baseline curve pairs, reusing the same machinery already validated in
    ``lazyportfolio.scientific_study`` (used there to compare V2_FINAL
    against baselines).  Resamples *blocks* of the already-realized,
    already-causal return series -- it never needs to swap past and future,
    which is exactly why it fits a walk-forward mechanism where the decision
    at any date is a function of everything strictly before it: there is no
    symmetric "what if the second half came first" to test here, only
    whether the realized gap is distinguishable from noise.

    ``resample_frequency`` (e.g. ``"W"`` when rebalancing weekly) compounds
    each curve onto that grid *before* differencing, so one observation is
    exactly one holding period regardless of cadence -- a fixed day-count
    block size tuned for one rebalance frequency silently mixes several
    holding periods into one block under a different, finer frequency,
    diluting the very autocorrelation the block bootstrap is meant to
    respect. ``block_size=None`` picks 20 (~1 trading month) for daily
    curves, or 4 periods once resampled -- override either explicitly if
    a different residual-correlation length is expected.
    """
    if resample_frequency:
        curves = {name: _resample_simple_returns(curve, resample_frequency) for name, curve in curves.items()}
        periods_per_year = _annualization_factor(resample_frequency)
        default_block_size = 4
    else:
        periods_per_year = 252.0
        default_block_size = 20
    block_size = block_size if block_size is not None else default_block_size

    common_index = None
    for candidate, baseline in comparisons:
        idx = curves[candidate].index.intersection(curves[baseline].index)
        common_index = idx if common_index is None else common_index.intersection(idx)
    # Below this many observations there are no whole blocks left to
    # resample: every draw reproduces the original series verbatim, so the CI
    # collapses onto the point estimate and the p-value bottoms out at its
    # bootstrap floor (1/(samples+1)) regardless of whether the difference is
    # real -- less data reads as *more* certainty. Reported instead as a
    # point difference with no CI/p-value, rather than a number that looks
    # like a real test and is not one (D23 in
    # ecosystem-cleanup/docs/deferred-fixes.md).
    insufficient = len(common_index) <= block_size
    raw = []
    for candidate, baseline in comparisons:
        differences = (
            curves[candidate].loc[common_index].to_numpy(dtype=float)
            - curves[baseline].loc[common_index].to_numpy(dtype=float)
        )
        mean_diff = float(np.mean(differences) * periods_per_year)
        if insufficient:
            raw.append((candidate, baseline, mean_diff, None, None, None))
            continue
        low, high, p_value = paired_block_bootstrap(
            differences, samples=samples, block_size=block_size, random_seed=random_seed,
        )
        raw.append((candidate, baseline, mean_diff, low * periods_per_year, high * periods_per_year, p_value))
    testable_indexes = [i for i, item in enumerate(raw) if item[-1] is not None]
    holm_by_index = dict(zip(
        testable_indexes, _holm_adjust([raw[i][-1] for i in testable_indexes]),
    ))
    return [
        {
            "candidate": candidate, "baseline": baseline,
            "annualized_mean_difference": mean_diff, "ci_low": low, "ci_high": high,
            "p_value": p_value,
            "holm_adjusted_p_value": holm_by_index.get(i),
            **({"note": "sample too small for the block bootstrap "
                        f"(n_obs={len(common_index)} <= block_size={block_size}); "
                        "showing the point difference only"} if insufficient else {}),
        }
        for i, (candidate, baseline, mean_diff, low, high, p_value) in enumerate(raw)
    ]


def _sharpe(values: np.ndarray, periods_per_year: float) -> float:
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return float(np.mean(values) / std * np.sqrt(periods_per_year)) if std else 0.0


def sharpe_significance_report(
    curves: dict[str, Any],
    comparisons: list[tuple[str, str]],
    *,
    samples: int = 2_000,
    block_size: int | None = None,
    random_seed: int = 7,
    resample_frequency: str | None = None,
) -> list[dict[str, Any]]:
    """Block-bootstrap test on the SHARPE RATIO gap itself, not the mean
    return gap ``significance_report`` tests.  A candidate can beat a
    baseline's Sharpe purely by cutting volatility with unchanged (or even
    slightly lower) mean return -- ``significance_report``'s test on the
    mean of paired differences gives that case no credit at all, since it
    never looks at variance.  This resamples *paired* blocks (same block of
    dates drawn for both curves at once, preserving their joint day-to-day
    relationship) and recomputes each curve's own Sharpe ratio on every
    resampled path, so the statistic actually being tested is the one
    everyone has been eyeballing all day.

    The p-value here is a standard percentile bootstrap test (does the
    resampled Sharpe-gap distribution straddle zero) rather than a
    null-centered one like ``paired_block_bootstrap`` uses for the mean --
    Sharpe is a nonlinear ratio, so there is no single well-defined way to
    "recenter" a joint return series onto a shared null Sharpe the way you
    can just subtract a mean. This is the standard, honest tradeoff for
    testing a ratio statistic instead of a linear one.
    """
    if resample_frequency:
        curves = {name: _resample_simple_returns(curve, resample_frequency) for name, curve in curves.items()}
        periods_per_year = _annualization_factor(resample_frequency)
        default_block_size = 4
    else:
        periods_per_year = 252.0
        default_block_size = 20
    block_size = block_size if block_size is not None else default_block_size

    common_index = None
    for candidate, baseline in comparisons:
        idx = curves[candidate].index.intersection(curves[baseline].index)
        common_index = idx if common_index is None else common_index.intersection(idx)

    rng = np.random.default_rng(random_seed)
    n_obs = len(common_index)
    # Same collapse as significance_report's block bootstrap, and the same
    # reason: at or below one block there is nothing to resample but the
    # original series, so the CI would pin to the point estimate and the
    # p-value to its floor regardless of whether the gap is real (D23 in
    # ecosystem-cleanup/docs/deferred-fixes.md).
    insufficient = n_obs <= block_size
    blocks_needed = int(np.ceil(n_obs / block_size)) if not insufficient else 0
    offsets = np.arange(block_size)

    results = []
    for candidate, baseline in comparisons:
        c = curves[candidate].loc[common_index].to_numpy(dtype=float)
        b = curves[baseline].loc[common_index].to_numpy(dtype=float)
        observed = _sharpe(c, periods_per_year) - _sharpe(b, periods_per_year)
        if insufficient:
            results.append({
                "candidate": candidate, "baseline": baseline,
                "sharpe_difference": observed, "ci_low": None, "ci_high": None,
                "p_value": None,
                "note": "sample too small for the block bootstrap "
                        f"(n_obs={n_obs} <= block_size={block_size}); "
                        "showing the point difference only",
            })
            continue
        diffs = np.empty(samples)
        for sample in range(samples):
            starts = rng.integers(0, n_obs, size=blocks_needed)
            indexes = ((starts[:, None] + offsets[None, :]) % n_obs).reshape(-1)[:n_obs]
            diffs[sample] = _sharpe(c[indexes], periods_per_year) - _sharpe(b[indexes], periods_per_year)
        low, high = np.quantile(diffs, [0.025, 0.975])
        p_value = min(1.0, 2.0 * min(float(np.mean(diffs <= 0)), float(np.mean(diffs >= 0))))
        results.append({
            "candidate": candidate, "baseline": baseline,
            "sharpe_difference": observed, "ci_low": float(low), "ci_high": float(high),
            "p_value": p_value,
        })
    testable_indexes = [i for i, item in enumerate(results) if item["p_value"] is not None]
    holm_by_index = dict(zip(
        testable_indexes, _holm_adjust([results[i]["p_value"] for i in testable_indexes]),
    ))
    for i, item in enumerate(results):
        item["holm_adjusted_p_value"] = holm_by_index.get(i)
    return results


__all__ = ["evaluate_pruning", "build_report_html", "significance_report", "sharpe_significance_report"]
