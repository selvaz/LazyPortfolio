import pytest

from lazyportfolio.copilot.contracts import ProposalStatus
from lazyportfolio.copilot.state_machine import (
    TERMINAL_STATUSES,
    IllegalProposalTransition,
    legal_next_statuses,
    validate_transition,
)

_LEGAL_PAIRS: list[tuple[ProposalStatus, ProposalStatus]] = [
    ("drafting", "failed"),
    ("drafting", "pending_approval"),
    ("pending_approval", "rejected"),
    ("pending_approval", "expired"),
    ("pending_approval", "superseded"),
    ("pending_approval", "applying"),
    ("applying", "apply_failed"),
    ("applying", "applied"),
    ("applied", "confirmation_pending"),
    ("confirmation_pending", "confirmed"),
    ("confirmation_pending", "confirmation_failed"),
]

_ILLEGAL_PAIRS: list[tuple[ProposalStatus, ProposalStatus]] = [
    ("applied", "drafting"),
    ("confirmed", "drafting"),
    ("confirmed", "pending_approval"),
    ("drafting", "applied"),
    ("rejected", "pending_approval"),
    ("expired", "pending_approval"),
    ("superseded", "pending_approval"),
    ("apply_failed", "applying"),
    ("confirmation_failed", "confirmation_pending"),
]


@pytest.mark.parametrize("current,target", _LEGAL_PAIRS)
def test_legal_transitions_pass(current: ProposalStatus, target: ProposalStatus) -> None:
    validate_transition(current, target)
    assert target in legal_next_statuses(current)


@pytest.mark.parametrize("current,target", _ILLEGAL_PAIRS)
def test_illegal_transitions_raise(current: ProposalStatus, target: ProposalStatus) -> None:
    with pytest.raises(IllegalProposalTransition) as excinfo:
        validate_transition(current, target)
    assert excinfo.value.current == current
    assert excinfo.value.target == target
    assert target not in legal_next_statuses(current)


def test_a_status_never_legally_transitions_to_itself() -> None:
    for status in (
        "drafting",
        "pending_approval",
        "applying",
        "applied",
        "confirmation_pending",
    ):
        with pytest.raises(IllegalProposalTransition):
            validate_transition(status, status)  # type: ignore[arg-type]


def test_terminal_statuses_have_no_legal_successor() -> None:
    expected_terminal = {
        "failed",
        "rejected",
        "expired",
        "superseded",
        "apply_failed",
        "confirmed",
        "confirmation_failed",
    }
    assert TERMINAL_STATUSES == frozenset(expected_terminal)
    for status in TERMINAL_STATUSES:
        assert legal_next_statuses(status) == frozenset()
