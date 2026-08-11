"""External gate 4: validate the V2 audit ZIP and client HTML report."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

from lazyportfolio.hierarchical_v2 import (  # noqa: E402
    HierarchicalV2Backtester,
    HierarchicalV2Estimator,
    V2Model,
)
from project.tree_studio_v2.exports import build_audit_bundle, build_client_report  # noqa: E402
from project.tree_studio_v2.validate_backtests import load_daily  # noqa: E402

MODEL_PATH = (
    ROOT
    / "reports"
    / "tree_studio"
    / "models"
    / "Global allocation_3pct_TEV_father.json"
)
OUTPUT_DIR = ROOT / "reports" / "tree_studio" / "v2"


def _rows(archive: zipfile.ZipFile, name: str) -> list[dict[str, str]]:
    text = archive.read(name).decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def _assert_manifest(archive: zipfile.ZipFile) -> None:
    manifest = json.loads(archive.read("manifest.json"))
    if manifest["schema"] != "lazyfin-hierarchical-v2-audit-1":
        raise AssertionError("unexpected audit schema")
    if manifest["reference_policy"] != "immutable_raw":
        raise AssertionError("audit does not declare immutable raw references")
    for name, metadata in manifest["files"].items():
        body = archive.read(name)
        if len(body) != metadata["bytes"]:
            raise AssertionError(f"{name}: byte count differs from manifest")
        if hashlib.sha256(body).hexdigest() != metadata["sha256"]:
            raise AssertionError(f"{name}: SHA-256 differs from manifest")


def _assert_audit_content(archive: zipfile.ZipFile) -> None:
    required = {
        "configuration.json",
        "data_metadata.json",
        "point_estimate.json",
        "series/raw_daily_returns.csv",
        "series/fold_estimation_series.csv",
        "backtest/metrics.csv",
        "backtest/curves.csv",
        "backtest/folds.csv",
        "backtest/weights.csv",
        "backtest/audits.csv",
        "backtest/candidate_series.csv",
        "backtest/transaction_costs.csv",
        "README.md",
        "manifest.json",
    }
    missing = required - set(archive.namelist())
    if missing:
        raise AssertionError(f"audit ZIP is missing {sorted(missing)}")

    configuration = archive.read("configuration.json").decode("utf-8")
    if "DO-NOT-EXPORT" in configuration or "<redacted>" not in configuration:
        raise AssertionError("sensitive configuration value was not redacted")

    series_names = {row["series"] for row in _rows(archive, "series/fold_estimation_series.csv")}
    expected_series = {
        "RAW:ticker:ACWI",
        "REFERENCE:B0_RAW",
        "BACKWARD_INPUT:ticker:SPY_SYNTH",
        "DIAGNOSTIC:B0_SYNTH",
        "FORWARD_OUTPUT:Equity",
        "RESULT_OUTPUT:Equity",
    }
    if not expected_series <= series_names:
        raise AssertionError(f"audit series missing {sorted(expected_series - series_names)}")

    candidates = _rows(archive, "backtest/candidate_series.csv")
    result_equity = {
        row["series"] for row in candidates if row["solve"] == "RESULT:Equity"
    }
    forward_equity = {
        row["series"] for row in candidates if row["solve"] == "FORWARD:Equity"
    }
    if "ticker:SPY_SYNTH" not in result_equity:
        raise AssertionError("backward Equity solve does not expose SPY_SYNTH")
    if "ticker:SPY" not in forward_equity:
        raise AssertionError("forward Equity solve does not expose raw SPY")
    if any(name.startswith("ticker:XL") for name in result_equity):
        raise AssertionError("Equity local solve was flattened to sector constituents")

    targets = {row["target"] for row in _rows(archive, "backtest/weights.csv")}
    for expected in ("B0", "B0_SYNTH", "FINAL", "LOCAL:Equity"):
        if expected not in targets:
            raise AssertionError(f"weight history is missing {expected}")

    stages = {row["stage"] for row in _rows(archive, "backtest/audits.csv")}
    if stages != {"forward", "result"}:
        raise AssertionError(f"unexpected audit stages {stages}")
    arms = {row["arm"] for row in _rows(archive, "backtest/curves.csv")}
    if not {"B0", "B0_SYNTH", "FORWARD_FINAL", "FINAL"} <= arms:
        raise AssertionError("OOS curves do not contain all hierarchical comparison arms")


def main() -> None:
    config = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    config["fred_api_key"] = "DO-NOT-EXPORT"
    config["telegram"] = {"bot_token": "DO-NOT-EXPORT"}
    model = V2Model.from_config(config)
    daily = load_daily(config, model, "2022-12-31")
    backtest = config["backtest"]
    train_size = int(backtest.get("train_size") or 104)
    weekly = daily.resample("W-FRI").apply(lambda values: (1.0 + values).prod() - 1.0)
    estimate = HierarchicalV2Estimator().estimate(
        model,
        weekly.tail(train_size),
        mode="forward_backward",
        periods_per_year=52.0,
    )
    report = HierarchicalV2Backtester().run(
        model,
        daily,
        mode="forward_backward",
        train_size=train_size,
        estimation_frequency=str(backtest.get("estimation_frequency") or "W"),
        rebalance_frequency=str(backtest.get("rebalance_frequency") or "M"),
        transaction_cost_bps=float(backtest.get("transaction_cost_bps") or 0),
        capture_audit_series=True,
    )
    metadata = {
        "date_start": str(daily.index.min().date()),
        "date_end": str(daily.index.max().date()),
        "observations": len(daily),
    }
    audit = build_audit_bundle(
        config=config,
        data_metadata=metadata,
        daily_returns=daily,
        estimate=estimate,
        report=report,
    )
    client = build_client_report(
        config=config,
        data_metadata=metadata,
        estimate=estimate,
        report=report,
    )
    with zipfile.ZipFile(io.BytesIO(audit)) as archive:
        _assert_manifest(archive)
        _assert_audit_content(archive)

    report_html = client.decode("utf-8")
    for expected in (
        "Albero di allocazione",
        "Impostazioni dei nodi",
        "Risultati walk-forward",
        "B0_SYNTH",
        "Global allocation",
        "Equity",
        "<svg",
    ):
        if expected not in report_html:
            raise AssertionError(f"client report is missing {expected!r}")
    if "DO-NOT-EXPORT" in report_html or "https://" in report_html or "http://" in report_html:
        raise AssertionError("client report contains a secret or external dependency")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "audit_export_validation.zip").write_bytes(audit)
    (OUTPUT_DIR / "client_report_validation.html").write_bytes(client)
    print(
        f"PASS exports: {len(report.folds)} folds, "
        f"audit={len(audit):,} bytes, report={len(client):,} bytes"
    )


if __name__ == "__main__":
    main()
