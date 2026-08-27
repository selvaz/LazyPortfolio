"""The confirmation run: applied is not the same as verified.

The state machine has declared ``applied -> confirmation_pending ->
confirmed | confirmation_failed`` since the contracts were written and nothing
drove those transitions, so a tree could change on the strength of a preview
nobody ever checked. These tests pin down what "confirmed" now means, and --
more importantly -- that a mismatch is reported rather than smoothed over.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from project.advisor import services

from lazyportfolio.advisor import confirmation, proposal_repository
from lazyportfolio.advisor.repository import create_tree
from lazyportfolio.backend import OptimizationDataset


def _config() -> dict[str, Any]:
    return {
        "root_id": "root",
        "currency": "USD",
        "nodes": [
            {
                "id": "root",
                "name": "Root",
                "children": ["equity", "bond"],
                "instruments": [],
                "goal": {"objective": "min_risk"},
                "constraints": {},
            },
            {
                "id": "equity",
                "name": "Equity",
                "children": [],
                "instruments": ["ticker:SPY", "ticker:TLT"],
                "proxy": "ticker:SPY",
                "goal": {"objective": "max_ratio"},
                "constraints": {},
            },
            {
                "id": "bond",
                "name": "Bond",
                "children": [],
                "instruments": ["ticker:AGG"],
                "proxy": "ticker:AGG",
                "goal": {"objective": "min_risk"},
                "constraints": {},
            },
        ],
        "backtest": {
            "benchmark": {
                "name": "B0",
                "weights": {"ticker:SPY": 0.4, "ticker:TLT": 0.3, "ticker:AGG": 0.3},
            }
        },
    }


class _FakeBackend:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def load_returns(self, instruments, *, start="", end="", frequency="D", currency=None):
        return OptimizationDataset(
            returns=self.frame.loc[:, instruments],
            metadata={"source": "fake-hub", "database_identity": "fake-hub"},
        )


@pytest.fixture()
def frame() -> pd.DataFrame:
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(20260827)
    index = pd.bdate_range("2020-01-01", periods=300)
    return pd.DataFrame(
        {
            "ticker:SPY": rng.normal(0.0005, 0.01, len(index)),
            "ticker:TLT": rng.normal(0.0002, 0.006, len(index)),
            "ticker:AGG": rng.normal(0.0001, 0.003, len(index)),
        },
        index=index,
    )


@pytest.fixture()
def applied(tmp_path, frame):
    """A proposal taken all the way through approval, ready to confirm."""

    store_path = str(tmp_path / "store.sqlite3")
    revision = create_tree(_config(), actor_type="human", actor_id="test", db_path=store_path)
    backend = _FakeBackend(frame)
    proposal = services.create_proposal(
        revision.tree_id,
        "equity",
        [
            {
                "instruments": {"ticker:SPY": 1.0, "ticker:TLT": -1.0},
                "expected_return": 0.03,
                "confidence": 0.6,
                "source": "test",
                "rationale": "test",
            }
        ],
        caller_id="test",
        backend=backend,
        db_path=store_path,
    )
    services.approve_proposal(
        proposal.id,
        proposal_hash=proposal.content_hash,
        approved_by="test",
        idempotency_key="test-key",
        db_path=store_path,
    )
    return proposal, store_path, backend


def test_an_applied_proposal_confirms_against_what_it_promised(applied) -> None:
    """The proposal showed a set of weights; the applied tree must produce
    them. This is the whole point -- approving on a preview nobody checks is
    what the missing confirmation run left standing."""

    proposal, store_path, backend = applied

    result = confirmation.confirm_applied_proposal(
        proposal.id, backend=backend, db_path=store_path
    )

    assert result.confirmed
    assert result.status == "confirmed"
    assert result.max_abs_deviation is not None
    assert result.max_abs_deviation <= confirmation.DEFAULT_TOLERANCE
    assert result.deviations == {}
    record = proposal_repository.get(proposal.id, db_path=store_path)
    assert record is not None
    assert record.status == "confirmed"


def test_weights_that_do_not_match_the_preview_fail_the_confirmation(applied) -> None:
    """A tolerance so tight that nothing can meet it stands in for "the
    applied tree does not produce what was shown". The verdict must be
    recorded as a failure, not rounded away."""

    proposal, store_path, backend = applied

    result = confirmation.confirm_applied_proposal(
        proposal.id, backend=backend, tolerance=-1.0, db_path=store_path
    )

    assert not result.confirmed
    assert result.status == "confirmation_failed"
    record = proposal_repository.get(proposal.id, db_path=store_path)
    assert record is not None
    assert record.status == "confirmation_failed"


def test_the_verdict_is_persisted_for_both_outcomes(applied) -> None:
    """A failed confirmation is a finding worth keeping, so it is written the
    same way a passing one is."""

    proposal, store_path, backend = applied
    assert confirmation.get_confirmation(proposal.id, db_path=store_path) is None

    confirmation.confirm_applied_proposal(
        proposal.id, backend=backend, tolerance=-1.0, db_path=store_path
    )
    stored = confirmation.get_confirmation(proposal.id, db_path=store_path)

    assert stored is not None
    assert stored["status"] == "confirmation_failed"
    assert stored["max_abs_deviation"] is not None
    assert stored["detail"]["tolerance"] == -1.0


def test_only_an_applied_proposal_can_be_confirmed(tmp_path, frame) -> None:
    """Confirming a proposal still waiting for a human would report on a tree
    that was never changed."""

    store_path = str(tmp_path / "store.sqlite3")
    revision = create_tree(_config(), actor_type="human", actor_id="test", db_path=store_path)
    proposal = services.create_proposal(
        revision.tree_id,
        "equity",
        [
            {
                "instruments": {"ticker:SPY": 1.0, "ticker:TLT": -1.0},
                "expected_return": 0.03,
                "confidence": 0.6,
                "source": "test",
                "rationale": "test",
            }
        ],
        caller_id="test",
        backend=_FakeBackend(frame),
        db_path=store_path,
    )

    with pytest.raises(confirmation.ProposalNotApplied, match="pending_approval"):
        confirmation.confirm_applied_proposal(proposal.id, db_path=store_path)


def test_a_confirmed_proposal_is_not_confirmed_again(applied) -> None:
    """``confirmed`` is terminal in the state machine, so a second run is
    refused at the door rather than rewriting a settled verdict."""

    proposal, store_path, backend = applied
    confirmation.confirm_applied_proposal(proposal.id, backend=backend, db_path=store_path)

    with pytest.raises(confirmation.ProposalNotApplied, match="confirmed"):
        confirmation.confirm_applied_proposal(proposal.id, backend=backend, db_path=store_path)
