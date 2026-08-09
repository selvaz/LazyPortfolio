"""ChangeProposal state machine (docs/node-advisor-operational-plan.md §4.5).

Transitions are enforced here so the rule exists and is testable
independently of any database -- Fase 1's ``ProposalRepository`` wraps
:func:`validate_transition` around a ``UPDATE ... WHERE status = ?`` CAS,
but the legality of a transition is not itself a database concern.
"""

from __future__ import annotations

from lazyportfolio.advisor.contracts import ProposalStatus

#: §4.5's transition diagram, encoded as legal-successor sets. A status with
#: an empty set is terminal: no further transition is ever legal from it.
_LEGAL_TRANSITIONS: dict[ProposalStatus, frozenset[ProposalStatus]] = {
    "drafting": frozenset({"failed", "pending_approval"}),
    "failed": frozenset(),
    "pending_approval": frozenset({"rejected", "expired", "superseded", "applying"}),
    "rejected": frozenset(),
    "expired": frozenset(),
    "superseded": frozenset(),
    "applying": frozenset({"apply_failed", "applied"}),
    "apply_failed": frozenset(),
    "applied": frozenset({"confirmation_pending"}),
    "confirmation_pending": frozenset({"confirmed", "confirmation_failed"}),
    "confirmed": frozenset(),
    "confirmation_failed": frozenset(),
}

#: Statuses with no legal outgoing transition -- a proposal in one of these
#: can never change status again; a revision/modification always creates a
#: *new* proposal (§8.4) instead.
TERMINAL_STATUSES: frozenset[ProposalStatus] = frozenset(
    status for status, successors in _LEGAL_TRANSITIONS.items() if not successors
)


class IllegalProposalTransition(ValueError):
    """Raised when a status transition is not in §4.5's diagram."""

    def __init__(self, current: ProposalStatus, target: ProposalStatus) -> None:
        self.current = current
        self.target = target
        super().__init__(f"cannot transition a proposal from {current!r} to {target!r}")


def legal_next_statuses(current: ProposalStatus) -> frozenset[ProposalStatus]:
    """The set of statuses reachable in one legal transition from ``current``."""

    return _LEGAL_TRANSITIONS[current]


def validate_transition(current: ProposalStatus, target: ProposalStatus) -> None:
    """Raise :class:`IllegalProposalTransition` unless ``target`` is a legal
    successor of ``current``. A proposal never transitions to its own
    current status through this function -- that is a no-op the caller
    should short-circuit, not a transition."""

    if target not in _LEGAL_TRANSITIONS[current]:
        raise IllegalProposalTransition(current, target)


__all__ = [
    "TERMINAL_STATUSES",
    "IllegalProposalTransition",
    "legal_next_statuses",
    "validate_transition",
]
