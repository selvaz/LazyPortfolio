"""Node Advisor Fase 5 red-team coverage (docs/node-advisor-operational-plan.md
§12.4/§13).

Scenarios already proven by earlier phases are NOT re-tested here (that
would be redundant, not additional coverage):

* approval replay / idempotent retry -- tests/advisor/test_approval_service.py
* tampered payload / hash mismatch on approve -- test_tree_studio_advisor_vertical_slice.py
  (``test_approving_with_the_wrong_hash_is_rejected_with_409``)
* orphan job recovery after a worker crash -- same file
  (``test_worker_crash_is_recovered_by_the_heartbeat_reaper``)
* prompt injection escalating an LLM-proposed view's privileges --
  tests/test_advisor_agent.py

This file covers what those don't: an ``EvidenceRef.locator`` path-traversal
attempt (structural -- no evidence-fetching pipeline is wired yet, this
proves that gap is real and contained, not silently exploitable),
``content_hash`` sensitivity to single-field tampering, and a real
concurrent-writer SQLite lock (the second writer retries via
``busy_timeout`` instead of raising immediately).
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from lazyportfolio.advisor.canonical import content_hash
from lazyportfolio.advisor.contracts import EvidenceRef
from lazyportfolio.advisor.repository import create_tree, get_head, save_revision
from lazyportfolio.v2 import db as v2_db


# --------------------------------------------------------------------- #
# EvidenceRef.locator path traversal
# --------------------------------------------------------------------- #
def test_evidence_ref_accepts_a_path_traversal_shaped_locator_at_the_contract_level() -> None:
    """The contract itself imposes no filesystem semantics on ``locator``
    (it is a plain ``str``) -- this is fine ONLY because nothing in the
    advisor package ever opens a file using it (see the structural test
    below). This test documents that the contract layer is not, and was
    never meant to be, the enforcement point."""

    evidence = EvidenceRef(
        id=uuid4(),
        kind="artifact",
        locator="../../../../etc/passwd",
        title="malicious locator",
        retrieved_at=datetime.now(UTC),
        excerpt="irrelevant",
    )
    assert evidence.locator == "../../../../etc/passwd"


def test_no_code_in_the_advisor_package_dereferences_evidence_locator_as_a_filesystem_path() -> (
    None
):
    """Structural guard: proves the path-traversal surface above is inert
    right now (no evidence-fetching pipeline exists yet -- a disclosed gap,
    see docs/node-advisor-operational-plan.md), not just assumed inert.
    A future PR that wires evidence fetching MUST validate/sandbox
    ``locator`` before this test's premise (nothing reads it) still holds
    -- if it doesn't, this test should be replaced with a real allowlist
    test, not deleted silently."""

    advisor_src = Path(__file__).resolve().parents[2] / "src" / "lazyportfolio" / "advisor"
    advisor_project = Path(__file__).resolve().parents[2] / "project" / "advisor"
    offenders = []
    for root in (advisor_src, advisor_project):
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if ".locator" in text:
                offenders.append(str(path))
    assert offenders == [], (
        f"found code referencing EvidenceRef.locator: {offenders} -- if this is a new "
        f"evidence-fetching pipeline, it must validate/sandbox the locator before this "
        f"test is updated to allow it"
    )


# --------------------------------------------------------------------- #
# content_hash sensitivity -- every field must participate
# --------------------------------------------------------------------- #
def _base_payload() -> dict[str, object]:
    return {
        "kind": "replace_node_views",
        "node_id": "equity",
        "base_revision_id": "22222222-2222-4222-8222-222222222222",
        "proposed_views": [
            {
                "instruments": {"ticker:VTI": 1.0, "ticker:VXUS": -1.0},
                "expected_return": 0.02,
                "confidence": 0.6,
            }
        ],
        "rationale": "test",
        "expires_at": "2026-12-31T00:00:00+00:00",
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.__setitem__("node_id", "bond"),
        lambda p: p.__setitem__("rationale", "tampered"),
        lambda p: p.__setitem__("base_revision_id", "99999999-9999-4999-8999-999999999999"),
        lambda p: p["proposed_views"][0].__setitem__("expected_return", 0.99),
        lambda p: p["proposed_views"][0].__setitem__("confidence", 0.01),
        lambda p: p["proposed_views"][0]["instruments"].__setitem__("ticker:VTI", 0.5),
        lambda p: p.__setitem__("expires_at", "2099-01-01T00:00:00+00:00"),
    ],
    ids=[
        "node_id",
        "rationale",
        "base_revision_id",
        "expected_return",
        "confidence",
        "instrument_weight",
        "expires_at",
    ],
)
def test_tampering_any_single_field_changes_the_content_hash(mutate) -> None:
    """A proposal displayed to a human and then tampered with before
    approval (§8.3 step 2's threat model) must never keep the same hash --
    proven per-field here, complementing tests/advisor/test_canonical_hash.py's
    key-order-independence coverage (a different axis: same content,
    different serialization, MUST match; here: different content, MUST NOT
    match)."""

    import copy

    original = _base_payload()
    tampered = copy.deepcopy(original)
    mutate(tampered)

    assert content_hash(original) != content_hash(tampered)


# --------------------------------------------------------------------- #
# Concurrent writers -- busy_timeout retries, doesn't raise immediately
# --------------------------------------------------------------------- #
def test_a_second_writer_is_blocked_and_retried_not_immediately_rejected(tmp_path: Path) -> None:
    """Holds a real write lock on the shared sqlite db from one connection
    while a second thread calls save_revision on a completely separate
    connection to the same file -- proves busy_timeout (§11: "SQLite apre
    PRAGMA busy_timeout") makes the second writer wait and then succeed,
    rather than raising sqlite3.OperationalError('database is locked')
    immediately, which is what happens with busy_timeout=0."""

    db_path = tmp_path / "db.sqlite3"
    tree = create_tree(
        {
            "root_id": "root",
            "currency": "USD",
            "nodes": [
                {
                    "id": "root",
                    "name": "Root",
                    "children": [],
                    "instruments": ["ticker:VTI"],
                    "proxy": "ticker:VTI",
                    "goal": {"objective": "min_risk"},
                    "constraints": {},
                }
            ],
            "backtest": {"benchmark": {"name": "B0", "weights": {"ticker:VTI": 1.0}}},
        },
        actor_type="human",
        actor_id="test",
        db_path=db_path,
    )

    lock_holder = v2_db.connect(db_path)
    lock_holder.execute("BEGIN IMMEDIATE")
    # A real write under the reserved lock, matching what a genuine
    # concurrent writer would be doing (not just a bare BEGIN).
    lock_holder.execute(
        "UPDATE tree_heads SET head_revision_id = head_revision_id WHERE tree_id = ?",
        (tree.tree_id,),
    )

    result: dict[str, object] = {}

    def _second_writer() -> None:
        try:
            new_rev = save_revision(
                tree.tree_id,
                tree.config,
                actor_type="human",
                actor_id="second-writer",
                db_path=db_path,
            )
            result["revision_id"] = new_rev.revision_id
        except sqlite3.OperationalError as exc:
            result["error"] = exc

    thread = threading.Thread(target=_second_writer)
    thread.start()
    thread.join(timeout=0.5)
    assert thread.is_alive(), "second writer returned before the lock was ever released"

    lock_holder.commit()
    lock_holder.close()
    thread.join(timeout=5.0)

    assert not thread.is_alive(), "second writer never completed after the lock was released"
    assert "error" not in result, f"second writer raised instead of retrying: {result.get('error')}"
    assert result["revision_id"]
    head = get_head(tree.tree_id, db_path=db_path)
    assert head is not None
    assert head.revision_id == result["revision_id"]
