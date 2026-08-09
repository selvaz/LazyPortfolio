import hashlib
import math

import pytest

from lazyportfolio.advisor.canonical import (
    NonCanonicalValueError,
    canonical_json,
    content_hash,
)

#: Golden vector: fixed, hand-computed once. If this ever changes, the
#: canonicalization scheme changed -- every ChangeProposal.content_hash
#: computed under the old scheme is no longer reproducible, which is exactly
#: the kind of silent break this test exists to catch.
_GOLDEN_PAYLOAD = {"b": 2, "a": 1.5, "c": ["x", "y"]}
_GOLDEN_JSON = '{"a":1.5,"b":2,"c":["x","y"]}'
_GOLDEN_HASH = "sha256:" + hashlib.sha256(_GOLDEN_JSON.encode("utf-8")).hexdigest()


def test_canonical_json_sorts_keys_and_uses_compact_separators() -> None:
    assert canonical_json(_GOLDEN_PAYLOAD) == _GOLDEN_JSON


def test_content_hash_matches_golden_vector() -> None:
    assert content_hash(_GOLDEN_PAYLOAD) == _GOLDEN_HASH
    assert content_hash(_GOLDEN_PAYLOAD).startswith("sha256:")


def test_content_hash_is_independent_of_dict_insertion_order() -> None:
    ordered_a = {"a": 1, "b": 2, "c": 3}
    ordered_b = {"c": 3, "a": 1, "b": 2}
    assert content_hash(ordered_a) == content_hash(ordered_b)


def test_content_hash_is_independent_of_nested_dict_insertion_order() -> None:
    nested_a = {"outer": {"x": 1, "y": 2}}
    nested_b = {"outer": {"y": 2, "x": 1}}
    assert content_hash(nested_a) == content_hash(nested_b)


def test_content_hash_does_not_reorder_lists() -> None:
    """Array order is semantically meaningful to the caller (e.g. patch
    operations apply in order) -- canonicalize() must never sort it away."""

    assert content_hash({"items": [1, 2, 3]}) != content_hash({"items": [3, 2, 1]})


def test_nan_and_infinity_are_rejected_not_silently_stringified() -> None:
    with pytest.raises(NonCanonicalValueError):
        canonical_json({"value": math.nan})
    with pytest.raises(NonCanonicalValueError):
        canonical_json({"value": math.inf})
    with pytest.raises(NonCanonicalValueError):
        canonical_json({"value": -math.inf})


def test_two_different_producers_writing_the_same_content_hash_identically() -> None:
    """Simulates the Node Advisor and a future batch producer independently
    building the same logical payload with different key orders -- the
    invariant docs/adr/0001-node-advisor-architecture.md Decision 3 depends
    on for comparing proposals across producers."""

    node_advisor_payload = {
        "kind": "replace_node_views",
        "node_id": "equity",
        "views": [{"confidence": 0.6, "expected_return": 0.02}],
    }
    committee_payload = {
        "views": [{"expected_return": 0.02, "confidence": 0.6}],
        "node_id": "equity",
        "kind": "replace_node_views",
    }
    assert content_hash(node_advisor_payload) == content_hash(committee_payload)
