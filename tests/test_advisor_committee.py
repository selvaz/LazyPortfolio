"""Fase 6 exit criterion (docs/node-advisor-operational-plan.md §13): "il
committee produce proposte valide sugli stessi contratti senza modifiche a
schema/state machine -- la prova finale che la scelta producer-agnostic di
Fase 0 ha retto". No LLM here (see project/advisor/committee.py's module
docstring for why real committee reasoning is out of scope) -- every test
uses caller-supplied views, same as Fase 3's fixture path.

``project/advisor/committee.py`` is a script module, not an installed
package -- same sys.path pattern as ``tests/test_advisor_agent.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import pytest

from lazyportfolio.advisor.proposal_repository import get as get_proposal_record
from lazyportfolio.advisor.repository import create_tree
from lazyportfolio.backend import OptimizationDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_DIR = REPO_ROOT / "project"


class _FakeBackend:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def load_returns(self, instruments, *, start="", end="", frequency="D", currency=None):
        return OptimizationDataset(
            returns=self.frame.loc[:, instruments],
            metadata={"source": "fake-hub", "database_identity": "fake-hub"},
        )


@pytest.fixture()
def backend() -> _FakeBackend:
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(20260809)
    index = pd.bdate_range("2020-01-01", periods=300)
    frame = pd.DataFrame(
        {
            "ticker:VTI": rng.normal(0.0005, 0.01, len(index)),
            "ticker:VXUS": rng.normal(0.0003, 0.008, len(index)),
            "ticker:AGG": rng.normal(0.0001, 0.003, len(index)),
        },
        index=index,
    )
    return _FakeBackend(frame)


@pytest.fixture()
def committee_module():
    sys.path.insert(0, str(PROJECT_DIR))
    try:
        import advisor.committee as module

        yield module
    finally:
        sys.path.remove(str(PROJECT_DIR))
        sys.modules.pop("advisor.committee", None)


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
                "instruments": ["ticker:VTI", "ticker:VXUS"],
                "proxy": "ticker:VTI",
                "goal": {"objective": "min_risk"},
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
                "weights": {"ticker:VTI": 0.4, "ticker:VXUS": 0.2, "ticker:AGG": 0.4},
            }
        },
    }


@pytest.fixture()
def tree(tmp_path):
    store_path = str(tmp_path / "store.sqlite3")
    revision = create_tree(_config(), actor_type="human", actor_id="test", db_path=store_path)
    return revision, store_path


def _valid_equity_view() -> dict[str, Any]:
    return {
        "instruments": {"ticker:VTI": 1.0, "ticker:VXUS": -1.0},
        "expected_return": 0.02,
        "confidence": 0.6,
        "rationale": "committee test",
    }


def test_a_batch_across_multiple_nodes_shares_one_batch_id_and_producer_identity(
    committee_module, tree, backend
) -> None:
    revision, store_path = tree
    result = committee_module.run_committee_batch(
        revision.tree_id,
        {
            "equity": [_valid_equity_view()],
            "bond": [
                {
                    "instruments": {"ticker:AGG": 1.0},
                    "expected_return": 0.01,
                    "confidence": 0.4,
                    "rationale": "committee test",
                }
            ],
        },
        backend=backend,
        db_path=store_path,
    )

    assert result.errors == {}
    assert len(result.proposals) == 2
    node_ids = {p.node_id for p in result.proposals}
    assert node_ids == {"equity", "bond"}
    for proposal in result.proposals:
        assert proposal.batch_id == result.batch_id
        assert proposal.model_provenance.producer_kind == "scheduled_batch"
        assert proposal.model_provenance.producer_id == committee_module.PRODUCER_ID

        record = get_proposal_record(proposal.id, db_path=store_path)
        assert record is not None
        assert record.status == "pending_approval"


def test_a_caller_supplied_batch_id_is_honored_not_overwritten(
    committee_module, tree, backend
) -> None:
    revision, store_path = tree
    explicit_batch_id = uuid4()

    result = committee_module.run_committee_batch(
        revision.tree_id,
        {"equity": [_valid_equity_view()]},
        batch_id=explicit_batch_id,
        backend=backend,
        db_path=store_path,
    )

    assert result.batch_id == explicit_batch_id
    assert result.proposals[0].batch_id == explicit_batch_id


def test_the_node_advisors_own_conversational_proposals_still_leave_batch_id_none(
    tmp_path, backend
) -> None:
    """Regression guard for §3.4 point 2: adding batch_id support for the
    committee must never change the interactive Node Advisor's own
    proposals, which always leave it None."""

    from lazyportfolio.advisor import proposal_repository as proposals

    sys.path.insert(0, str(PROJECT_DIR))
    try:
        from advisor import services
    finally:
        sys.path.remove(str(PROJECT_DIR))
        sys.modules.pop("advisor.services", None)

    store_path = str(tmp_path / "store.sqlite3")
    revision = create_tree(_config(), actor_type="human", actor_id="test", db_path=store_path)

    proposal = services.create_proposal(
        revision.tree_id,
        "equity",
        [_valid_equity_view()],
        caller_id="test",
        backend=backend,
        db_path=store_path,
    )

    assert proposal.batch_id is None
    record = proposals.get(proposal.id, db_path=store_path)
    assert record is not None
    assert record.proposal.batch_id is None


def test_an_invalid_node_in_the_batch_is_recorded_as_an_error_and_does_not_block_the_rest(
    committee_module, tree, backend
) -> None:
    """Same validation as the interactive path -- an out-of-universe
    instrument is rejected exactly the same way (node_universe.validate_view_set
    has no branch on producer_kind) -- but one bad node must not sink the
    whole batch."""

    revision, store_path = tree
    result = committee_module.run_committee_batch(
        revision.tree_id,
        {
            "equity": [_valid_equity_view()],
            "bond": [
                {
                    # AGG belongs to bond's universe -- ticker:VTI does not.
                    "instruments": {"ticker:VTI": 1.0},
                    "expected_return": 0.02,
                    "confidence": 0.6,
                    "rationale": "invalid on purpose",
                }
            ],
        },
        backend=backend,
        db_path=store_path,
    )

    assert len(result.proposals) == 1
    assert result.proposals[0].node_id == "equity"
    assert "bond" in result.errors
    assert "instrument_outside_universe" in result.errors["bond"]


def test_a_financing_instrument_view_from_the_committee_is_rejected_like_any_other_producer(
    committee_module, tree
) -> None:
    """§7.2/§11's financing-instrument prohibition has no producer_kind
    branch -- a batch producer gets no more privilege than the interactive
    Node Advisor."""

    revision, store_path = tree
    result = committee_module.run_committee_batch(
        revision.tree_id,
        {
            "equity": [
                {
                    "instruments": {"cash:rf": 1.0},
                    "expected_return": 0.5,
                    "confidence": 0.9,
                    "rationale": "invalid on purpose",
                }
            ]
        },
        db_path=store_path,
    )

    assert result.proposals == []
    assert "financing_instrument_forbidden" in result.errors["equity"]


def test_an_unknown_tree_id_is_recorded_as_an_error_not_raised(committee_module, tmp_path) -> None:
    result = committee_module.run_committee_batch(
        "does-not-exist",
        {"equity": [_valid_equity_view()]},
        db_path=str(tmp_path / "store.sqlite3"),
    )

    assert result.proposals == []
    assert "equity" in result.errors
