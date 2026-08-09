"""Optional, read-only second-opinion reviewer for a ``ChangeProposal``.

docs/node-advisor-operational-plan.md §7.3/§13 Fase 5: "reviewer assente per
il primo vertical slice" is the plan's own recommended default -- this
module is wired in but :func:`review_proposal` defaults to disabled
(``enabled=False``) and returns ``None`` rather than running automatically.

The reviewer NEVER rewrites a view and NEVER promotes a proposal to
``pending_approval`` on its own (§7.3: "agent_review non deve da solo
sostenere una view") -- it is read-only evaluation surfaced to the human
next to the proposal card, not a second authority in the state machine. It
runs via ``lazytools``'s ``claude_code(mode="read")`` connector (a
sandboxed, read-only Claude Code CLI subprocess -- Read/Grep/Glob only, no
Bash, no write tools), the same connector already used elsewhere in the
ecosystem for read-only second opinions.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field


class ReviewFinding(BaseModel):
    severity: Literal["info", "warning", "critical"]
    claim: str
    evidence_mismatch: bool
    detail: str


class ReviewResult(BaseModel):
    """``reviewed=False`` means the reviewer did not produce usable output
    (disabled, connector failure, or unparseable response) -- callers must
    treat that the same as "no opinion", never as "reviewed and clean"."""

    reviewed: bool
    findings: list[ReviewFinding] = Field(default_factory=list)
    recommendation: str = ""


_REVIEW_PROMPT_TEMPLATE = (
    "You are reviewing a proposed change to a portfolio tree node's "
    "Black-Litterman views. You are READ-ONLY: you cannot and must not "
    "modify anything, approve anything, or promote this proposal -- your "
    "output is advisory only, read by a human alongside the proposal card.\n\n"
    "Proposal rationale: {rationale}\n"
    "Proposed views: {views_json}\n"
    "Counterfactual delta: {delta_json}\n\n"
    "Identify any claim in the rationale that is not supported by the "
    "counterfactual numbers actually shown (a claim/evidence mismatch), any "
    "confidence value that looks miscalibrated given the stated evidence, "
    "and any other concern a careful human reviewer would flag. Reply as "
    'compact JSON: {{"findings": [{{"severity": "info"|"warning"|'
    '"critical", "claim": str, "evidence_mismatch": bool, "detail": str}}], '
    '"recommendation": str}}. No prose, no markdown fences.'
)


def review_proposal(
    proposal: dict[str, Any],
    *,
    enabled: bool = False,
    mode: Literal["claude_code"] = "claude_code",
) -> ReviewResult | None:
    """Run the optional reviewer, or return ``None`` if disabled (the default).

    ``proposal`` is a ``ChangeProposal`` serialized as a dict
    (``.model_dump(mode="json")``) -- this module has no dependency on
    ``lazyportfolio.advisor.contracts`` beyond that shape, so it never needs
    to import it just to read a few fields.
    """

    if not enabled:
        return None

    from lazytools.connectors.code_support import claude_code

    prompt = _REVIEW_PROMPT_TEMPLATE.format(
        rationale=proposal.get("rationale", ""),
        views_json=json.dumps(proposal.get("proposed_views", [])),
        delta_json=json.dumps(proposal.get("counterfactual", {}).get("delta", {})),
    )
    if mode == "claude_code":
        raw = claude_code(prompt, mode="read")
    else:  # pragma: no cover - only one mode implemented in the MVP
        raise ValueError(f"unsupported reviewer mode: {mode!r}")

    return _parse_review_output(raw)


def _parse_review_output(raw: dict[str, Any] | str) -> ReviewResult:
    """``raw`` is either ``{"result": <text>, "content_is_untrusted": True}``
    on a successful connector call, or a plain ``"[claude_code] ..."``
    string on a connector-level failure (per ``claude_code``'s contract).

    The model's own JSON reply is untrusted content, same as any other tool
    result -- it is only ever parsed into the fixed ``ReviewResult`` schema
    below, never executed or interpolated anywhere.
    """

    if isinstance(raw, str):
        return ReviewResult(reviewed=False, recommendation=f"reviewer unavailable: {raw}")

    try:
        parsed = json.loads(raw["result"])
        return ReviewResult(reviewed=True, **parsed)
    except (KeyError, TypeError, ValueError):
        return ReviewResult(reviewed=False, recommendation="reviewer output could not be parsed")


__all__ = ["ReviewFinding", "ReviewResult", "review_proposal"]
