"""The confirmation run: did the applied tree produce what the proposal showed?

The state machine has declared ``applied -> confirmation_pending ->
confirmed | confirmation_failed`` since the contracts were written, and
nothing ever drove those transitions -- an applied proposal simply stopped at
``applied``. So the tree could be changed on the strength of a preview nobody
ever checked against the result.

What this checks, precisely: a proposal carries
``counterfactual.variant["terminal_weights"]`` -- the weights the human was
shown when they approved. After the new revision is the head, this re-solves
that revision and compares. A mismatch means the applied config does not
produce the previewed allocation, which is a defect in the patch or the
apply path, not market news.

It is deliberately *not* a fresh-data re-run. Solving on today's data would
mostly measure how far the market has moved since the proposal, which says
nothing about whether the change was applied faithfully, and would make
"confirmed" a statement about market stability rather than about the system.
The comparison uses the same window the proposal's own snapshot describes.
"""

from __future__ import annotations

import json
import os
from contextlib import closing
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from lazyportfolio.advisor import proposal_repository as _proposals
from lazyportfolio.advisor import repository as _trees
from lazyportfolio.advisor import snapshot as _snapshot
from lazyportfolio.advisor.contracts import ProposalStatus
from lazyportfolio.hierarchical_v2 import HierarchicalV2Estimator
from lazyportfolio.v2 import db as _db
from lazyportfolio.v2.mode import mode_from_config

if TYPE_CHECKING:
    from lazyportfolio.backend import OptimizationDataBackend

#: How far a re-solved weight may sit from the previewed one and still count
#: as the same allocation. Solvers are iterative and floating point is not
#: associative, so an exact match is not a reasonable bar; anything above this
#: is a difference a reader would see in a rendered percentage.
DEFAULT_TOLERANCE = 1e-6


class ProposalNotApplied(ValueError):
    """Only an ``applied`` proposal can be confirmed."""


class NothingToConfirm(ValueError):
    """The proposal recorded no previewed weights to compare against."""


@dataclass(frozen=True)
class ConfirmationResult:
    proposal_id: UUID
    revision_id: str
    status: str
    max_abs_deviation: float | None
    deviations: dict[str, float] = field(default_factory=dict)

    @property
    def confirmed(self) -> bool:
        return self.status == "confirmed"


def confirm_applied_proposal(
    proposal_id: UUID,
    *,
    backend: OptimizationDataBackend | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    db_path: str | os.PathLike[str] | None = None,
) -> ConfirmationResult:
    """Re-solve the applied revision and compare it to what was previewed.

    Moves the proposal ``applied -> confirmation_pending`` first, so a crash
    midway leaves it visibly unfinished rather than silently still ``applied``,
    and then to ``confirmed`` or ``confirmation_failed``. Both outcomes are
    recorded in ``proposal_confirmations``: a failure is a finding worth
    keeping, not an error to discard.
    """

    record = _proposals.get(proposal_id, db_path=db_path)
    if record is None:
        raise _proposals.ConcurrentProposalWrite(f"proposal {proposal_id} not found")
    if record.status != "applied":
        raise ProposalNotApplied(
            f"proposal {proposal_id} is {record.status!r}, not 'applied'"
        )

    previewed = (record.proposal.counterfactual.variant or {}).get("terminal_weights")
    if not isinstance(previewed, dict) or not previewed:
        raise NothingToConfirm(
            f"proposal {proposal_id} carries no previewed terminal weights"
        )

    head = _trees.get_head(str(record.proposal.tree_id), db_path=db_path)
    if head is None:
        raise NothingToConfirm(f"tree {record.proposal.tree_id} has no head revision")

    _proposals.transition(proposal_id, "applied", "confirmation_pending", db_path=db_path)

    try:
        model, dataset, _ = _snapshot.load_snapshot(head.config, backend=backend)
        estimate = HierarchicalV2Estimator().estimate(
            model,
            dataset.returns,
            mode=mode_from_config(head.config),
            periods_per_year=252.0,
        )
        actual = estimate.terminal_weights
    except Exception as exc:
        _record(
            proposal_id,
            head.revision_id,
            "confirmation_failed",
            None,
            {"error": str(exc)},
            db_path=db_path,
        )
        _proposals.transition(
            proposal_id, "confirmation_pending", "confirmation_failed", db_path=db_path
        )
        raise

    deviations = {
        instrument: float(actual.get(instrument, 0.0)) - float(previewed.get(instrument, 0.0))
        for instrument in {*actual, *previewed}
    }
    max_abs = max((abs(value) for value in deviations.values()), default=0.0)
    status: ProposalStatus = "confirmed" if max_abs <= tolerance else "confirmation_failed"

    _record(
        proposal_id,
        head.revision_id,
        status,
        max_abs,
        # Only what differs: a tree of 100 instruments that matched everywhere
        # would otherwise store 100 zeroes and bury the one that did not.
        {
            "tolerance": tolerance,
            "deviations": {k: v for k, v in deviations.items() if abs(v) > tolerance},
        },
        db_path=db_path,
    )
    _proposals.transition(proposal_id, "confirmation_pending", status, db_path=db_path)
    return ConfirmationResult(
        proposal_id=proposal_id,
        revision_id=head.revision_id,
        status=status,
        max_abs_deviation=max_abs,
        deviations={k: v for k, v in deviations.items() if abs(v) > tolerance},
    )


def _record(
    proposal_id: UUID,
    revision_id: str,
    status: str,
    max_abs_deviation: float | None,
    detail: dict[str, Any],
    *,
    db_path: str | os.PathLike[str] | None,
) -> None:
    with closing(_db.connect(db_path)) as conn:
        conn.execute(
            "INSERT INTO proposal_confirmations (proposal_id, confirmed_at, revision_id, "
            "status, max_abs_deviation, detail_json) VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(proposal_id) DO UPDATE SET "
            "confirmed_at = excluded.confirmed_at, revision_id = excluded.revision_id, "
            "status = excluded.status, max_abs_deviation = excluded.max_abs_deviation, "
            "detail_json = excluded.detail_json",
            (
                str(proposal_id),
                datetime.now(UTC).isoformat(),
                revision_id,
                status,
                max_abs_deviation,
                json.dumps(detail, sort_keys=True, default=str),
            ),
        )
        conn.commit()


def get_confirmation(
    proposal_id: UUID, *, db_path: str | os.PathLike[str] | None = None
) -> dict[str, Any] | None:
    """The recorded verdict for one proposal, or ``None`` if never confirmed."""

    with closing(_db.connect(db_path)) as conn:
        row = conn.execute(
            "SELECT confirmed_at, revision_id, status, max_abs_deviation, detail_json "
            "FROM proposal_confirmations WHERE proposal_id = ?",
            (str(proposal_id),),
        ).fetchone()
    if row is None:
        return None
    return {
        "confirmed_at": row[0],
        "revision_id": row[1],
        "status": row[2],
        "max_abs_deviation": row[3],
        "detail": json.loads(row[4]),
    }


__all__ = [
    "DEFAULT_TOLERANCE",
    "ConfirmationResult",
    "NothingToConfirm",
    "ProposalNotApplied",
    "confirm_applied_proposal",
    "get_confirmation",
]
