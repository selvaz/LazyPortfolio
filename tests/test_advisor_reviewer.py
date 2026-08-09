"""Node Advisor Fase 5 optional reviewer (docs/node-advisor-operational-plan.md
§7.3/§13): disabled by default ("reviewer assente per il primo vertical
slice"), never able to write anything, and must degrade to
``reviewed=False`` rather than raise on a malformed or unavailable
``claude_code`` response.

``project/advisor/reviewer.py`` is a script module, not an installed
package -- same sys.path pattern as ``tests/test_advisor_agent.py``. The
disabled-path tests need no ``lazytools`` import at all (the module only
imports ``claude_code`` once ``enabled=True``); the parsing tests mock
``lazytools.connectors.code_support.claude_code`` directly, so they still
need lazytools importable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_DIR = REPO_ROOT / "project"


@pytest.fixture()
def reviewer_module():
    sys.path.insert(0, str(PROJECT_DIR))
    try:
        import advisor.reviewer as module

        yield module
    finally:
        sys.path.remove(str(PROJECT_DIR))
        sys.modules.pop("advisor.reviewer", None)


_PROPOSAL = {
    "rationale": "SPY should outperform TLT given the current regime.",
    "proposed_views": [{"instruments": {"ticker:SPY": 1.0, "ticker:TLT": -1.0}}],
    "counterfactual": {"delta": {"ticker:SPY": 0.05, "ticker:TLT": -0.05}},
}


# --------------------------------------------------------------------- #
# Disabled by default -- no lazytools import, no subprocess, ever.
# --------------------------------------------------------------------- #
def test_review_proposal_is_disabled_by_default(reviewer_module) -> None:
    assert reviewer_module.review_proposal(_PROPOSAL) is None


def test_review_proposal_disabled_never_imports_lazytools(reviewer_module, monkeypatch) -> None:
    """A hard guarantee that the disabled path cannot reach the connector
    -- if it tried, importing a poisoned/missing lazytools would raise
    instead of the clean ``None`` this test asserts."""

    monkeypatch.setitem(sys.modules, "lazytools", None)  # any import raises ImportError
    assert reviewer_module.review_proposal(_PROPOSAL, enabled=False) is None


# --------------------------------------------------------------------- #
# Enabled path -- claude_code mocked, never a live subprocess call.
# --------------------------------------------------------------------- #
def test_review_proposal_enabled_parses_a_well_formed_response(
    reviewer_module, monkeypatch
) -> None:
    pytest.importorskip("lazytools", reason="reviewer enabled path needs lazytools installed")
    import lazytools.connectors.code_support as code_support

    def _fake_claude_code(task: str, *, mode: str = "read", **kwargs):
        assert mode == "read"
        return {
            "result": (
                '{"findings": [{"severity": "warning", "claim": "outperform", '
                '"evidence_mismatch": false, "detail": "delta is small"}], '
                '"recommendation": "acceptable but weak evidence"}'
            ),
            "content_is_untrusted": True,
        }

    monkeypatch.setattr(code_support, "claude_code", _fake_claude_code)

    result = reviewer_module.review_proposal(_PROPOSAL, enabled=True)

    assert result.reviewed is True
    assert result.recommendation == "acceptable but weak evidence"
    assert len(result.findings) == 1
    assert result.findings[0].severity == "warning"


def test_review_proposal_never_raises_on_a_connector_level_failure_string(
    reviewer_module, monkeypatch
) -> None:
    """``claude_code`` returns a plain "[claude_code] ..." string (not a
    dict) on a connector-level failure -- the reviewer must degrade to
    ``reviewed=False``, never propagate an exception up into the proposal
    flow just because the optional reviewer was unavailable."""

    pytest.importorskip("lazytools", reason="reviewer enabled path needs lazytools installed")
    import lazytools.connectors.code_support as code_support

    monkeypatch.setattr(
        code_support, "claude_code", lambda task, **kwargs: "[claude_code] CLI not found"
    )

    result = reviewer_module.review_proposal(_PROPOSAL, enabled=True)

    assert result.reviewed is False
    assert "unavailable" in result.recommendation


def test_review_proposal_never_raises_on_unparseable_model_output(
    reviewer_module, monkeypatch
) -> None:
    """Simulates a fully-compromised or just-broken model response (not
    valid JSON, or JSON that doesn't match ReviewResult's schema) -- must
    degrade gracefully, matching the same never-trust-tool-output posture
    ``advisor.agent`` uses for its own LLM's structured output."""

    pytest.importorskip("lazytools", reason="reviewer enabled path needs lazytools installed")
    import lazytools.connectors.code_support as code_support

    monkeypatch.setattr(
        code_support,
        "claude_code",
        lambda task, **kwargs: {"result": "not json at all", "content_is_untrusted": True},
    )

    result = reviewer_module.review_proposal(_PROPOSAL, enabled=True)

    assert result.reviewed is False
    assert result.findings == []


def test_review_proposal_never_exposes_a_write_shaped_tool_or_action(reviewer_module) -> None:
    """Structural guard, matching test_advisor_agent.py's analogous check:
    ReviewResult has no field through which a review could mutate a
    proposal or promote its status -- it is pure advisory output."""

    field_names = set(reviewer_module.ReviewResult.model_fields.keys())
    forbidden_substrings = ("status", "approve", "apply", "promote", "save", "delete")
    offenders = [
        f for f in field_names if any(bad in f.lower() for bad in forbidden_substrings)
    ]
    assert offenders == [], f"ReviewResult exposes a write-shaped field: {offenders}"
