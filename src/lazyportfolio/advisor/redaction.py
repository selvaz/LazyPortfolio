"""Custom PII redaction for Session logs (docs/node-advisor-operational-plan.md §11/§13 Fase 5).

LazyBridge's ``Session`` already redacts common secret formats (API keys,
tokens) by default -- this module adds PII redaction on top: emails, phone
numbers, and long digit runs shaped like a government id/account number.

Apply this to text about to be written to a ``Session``/log only, never to
a canonical ``ChangeProposal`` payload itself -- redacting that would
change its ``content_hash``, the exact silent-corruption failure mode
§4.3's immutability invariant exists to prevent.
"""

from __future__ import annotations

import re

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
#: The optional country-code prefix requires either a literal "+" or a
#: trailing separator -- a bare digit prefix with no separator is never
#: allowed to "borrow" digits from a longer contiguous run (e.g. a 12-digit
#: account number), which would otherwise let this pattern shadow
#: _LONG_DIGIT_RUN below and mislabel a non-phone number as a phone number.
_PHONE = re.compile(
    r"(?<!\d)(?:\+\d{1,3}[ .\-]?|\d{1,3}[ .\-])?\(?\d{3}\)?[ .\-]?\d{3}[ .\-]?\d{4}(?!\d)"
)
_LONG_DIGIT_RUN = re.compile(r"(?<!\d)\d{9,}(?!\d)")


def redact_pii(text: str) -> str:
    """Return ``text`` with emails, phone numbers, and long digit runs
    (SSN/account-number-shaped) replaced by fixed placeholders.

    Order matters: emails are redacted first, so the digits inside an
    email address are never independently re-matched by the phone/long-run
    patterns afterward.
    """

    redacted = _EMAIL.sub("[redacted-email]", text)
    redacted = _PHONE.sub("[redacted-phone]", redacted)
    redacted = _LONG_DIGIT_RUN.sub("[redacted-number]", redacted)
    return redacted


__all__ = ["redact_pii"]
