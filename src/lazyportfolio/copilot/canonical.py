"""Canonical serialization and content hashing (docs/node-copilot-operational-plan.md §4.3).

``json.dumps(sort_keys=True)`` is not enough on its own: different producers
can serialize the same float differently (``1.0`` vs ``1``, trailing zeros,
locale-dependent formatting), which would make two byte-identical proposals
hash differently depending on which producer wrote them. :func:`canonicalize`
normalizes floats and dict key order before serialization; it deliberately
does **not** reorder lists -- array order is semantically meaningful to the
caller (e.g. patch operations must apply in order) and is not this module's
concern to guess at.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

_JSON_SEPARATORS = (",", ":")


class NonCanonicalValueError(ValueError):
    """Raised when a value cannot be represented in the canonical form.

    ``NaN``/``Infinity`` are valid Python floats but have no canonical JSON
    representation and must never silently become the string ``"NaN"`` or a
    non-standard JSON token.
    """


def canonicalize(value: Any) -> Any:
    """Recursively normalize ``value`` into its canonical form.

    Dict keys are sorted (and coerced to ``str``, matching ``json.dumps``'s
    own coercion, so the hash reflects what would actually be serialized).
    Floats that are whole numbers keep their float-ness (``1.0`` stays
    ``1.0``, never becomes the int ``1``) so two producers who both wrote a
    float still hash identically regardless of language/library-level
    float-vs-int coercion quirks upstream of this function.
    """

    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise NonCanonicalValueError(f"{value!r} has no canonical JSON representation")
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return value
    if value is None:
        return None
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda kv: str(kv[0]))
        return {str(key): canonicalize(item) for key, item in items}
    if isinstance(value, (list, tuple)):
        return [canonicalize(item) for item in value]
    raise NonCanonicalValueError(f"cannot canonicalize value of type {type(value).__name__!r}")


def canonical_json(value: Any) -> str:
    """Serialize ``value`` to a single canonical JSON string.

    Sorted keys, no extraneous whitespace, UTF-8-safe (``ensure_ascii=False``
    -- non-ASCII characters are encoded as themselves, not ``\\uXXXX``
    escapes, so the same Unicode string always produces the same bytes
    regardless of which library wrote it).
    """

    return json.dumps(
        canonicalize(value),
        sort_keys=True,
        separators=_JSON_SEPARATORS,
        ensure_ascii=False,
    )


def content_hash(value: Any) -> str:
    """``"sha256:" + hex digest`` of ``value``'s canonical JSON encoding.

    Same content hashes identically regardless of producer, key insertion
    order, or which library serialized it first -- the invariant a
    ``ChangeProposal.content_hash`` (and its future committee-produced
    siblings) depends on.
    """

    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


__all__ = ["NonCanonicalValueError", "canonical_json", "canonicalize", "content_hash"]
