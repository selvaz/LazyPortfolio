from pathlib import Path

HTML = (Path(__file__).resolve().parents[1] / "project" / "tree_studio.html").read_text(
    encoding="utf-8"
)


def test_tree_studio_exposes_per_node_financing_controls() -> None:
    assert 'id="c-cash-enabled"' in HTML
    assert 'id="c-leverage"' in HTML
    assert 'id="c-borrow-spread"' in HTML
    assert 'id="c-risk-free-rate"' in HTML
    assert "constraints:{cash_enabled:false,max_leverage:'1',borrow_spread_bps:''}" in HTML


def test_tree_studio_serializes_financing_without_forcing_it() -> None:
    assert "cash_enabled:n.goal.objective==='hrp'?false" in HTML
    assert "max_leverage:n.goal.objective==='hrp'?'1'" in HTML
    assert "borrow_spread_bps:n.goal.objective==='hrp'?'':" in HTML
    assert "financingEnabled=cashEnabled||leverage>1" in HTML
    assert "Il solver puo comunque scegliere cash zero" in HTML


def test_tree_studio_keeps_hrp_and_global_defaults_unambiguous() -> None:
    assert "HRP non supporta cash o leva" in HTML
    assert "Default root risk-free annuo" in HTML
    assert "Default root borrow spread (bps)" in HTML
    assert "Non abilitano cash o leva da soli" in HTML


def test_tree_studio_renders_financing_audit_fields() -> None:
    for field in (
        "cash_enabled_source",
        "max_leverage_source",
        "borrow_spread_bps_source",
        "global_risky_gross_exposure",
        "global_cash_weight",
        "portfolio_net_exposure",
    ):
        assert field in HTML
    assert "Financing locale e aggregato" in HTML
    assert "Provenienza C/L/S" in HTML


def test_tree_studio_embedded_javascript_is_syntax_valid(tmp_path: Path) -> None:
    import re
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        return
    match = re.search(r"<script>(.*)</script>", HTML, flags=re.DOTALL)
    assert match is not None
    script = tmp_path / "tree_studio.js"
    script.write_text(match.group(1), encoding="utf-8")
    result = subprocess.run(
        [node, "--check", str(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
