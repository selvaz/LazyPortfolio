"""Contract tests for the reusable scheduled batch proposal producer."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pandas as pd
import pytest
from project.advisor import batch_producer, services

from lazyportfolio.advisor.proposal_repository import get as get_proposal_record
from lazyportfolio.advisor.repository import create_tree
from lazyportfolio.backend import OptimizationDataset

PRODUCER_ID = "scheduled-research"
RATIONALE = "Scheduled research batch proposal."
MODEL = "test-model"


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
        "rationale": "batch producer test",
    }


def _run(tree_id: str, node_views: dict[str, list[dict[str, Any]]], **kwargs):
    return batch_producer.run_proposal_batch(
        tree_id,
        node_views,
        producer_id=PRODUCER_ID,
        rationale=RATIONALE,
        model=MODEL,
        **kwargs,
    )


def test_batch_shares_identity_and_persists_pending_proposals(tree, backend) -> None:
    revision, store_path = tree
    result = _run(
        revision.tree_id,
        {
            "equity": [_valid_equity_view()],
            "bond": [
                {
                    "instruments": {"ticker:AGG": 1.0},
                    "expected_return": 0.01,
                    "confidence": 0.4,
                    "rationale": "batch producer test",
                }
            ],
        },
        backend=backend,
        db_path=store_path,
    )

    assert result.errors == {}
    assert {proposal.node_id for proposal in result.proposals} == {"equity", "bond"}
    for proposal in result.proposals:
        assert proposal.batch_id == result.batch_id
        assert proposal.model_provenance.producer_kind == "scheduled_batch"
        assert proposal.model_provenance.producer_id == PRODUCER_ID
        record = get_proposal_record(proposal.id, db_path=store_path)
        assert record is not None
        assert record.status == "pending_approval"


def test_explicit_batch_id_is_preserved(tree, backend) -> None:
    revision, store_path = tree
    explicit_batch_id = uuid4()
    result = _run(
        revision.tree_id,
        {"equity": [_valid_equity_view()]},
        batch_id=explicit_batch_id,
        backend=backend,
        db_path=store_path,
    )
    assert result.batch_id == explicit_batch_id
    assert result.proposals[0].batch_id == explicit_batch_id


def test_interactive_proposal_still_has_no_batch_id(tree, backend) -> None:
    revision, store_path = tree
    proposal = services.create_proposal(
        revision.tree_id,
        "equity",
        [_valid_equity_view()],
        caller_id="test",
        backend=backend,
        db_path=store_path,
    )
    assert proposal.batch_id is None


def test_invalid_node_does_not_block_valid_sibling(tree, backend) -> None:
    revision, store_path = tree
    result = _run(
        revision.tree_id,
        {
            "equity": [_valid_equity_view()],
            "bond": [
                {
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
    assert [proposal.node_id for proposal in result.proposals] == ["equity"]
    assert "instrument_outside_universe" in result.errors["bond"]


def test_financing_instrument_is_rejected_for_batch_producer(tree) -> None:
    revision, store_path = tree
    result = _run(
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


def test_unknown_tree_is_recorded_as_error(tmp_path) -> None:
    result = _run(
        "does-not-exist",
        {"equity": [_valid_equity_view()]},
        db_path=str(tmp_path / "store.sqlite3"),
    )
    assert result.proposals == []
    assert "equity" in result.errors


@pytest.mark.parametrize("field", ["producer_id", "rationale", "model"])
def test_provenance_inputs_are_required(field) -> None:
    values = {"producer_id": PRODUCER_ID, "rationale": RATIONALE, "model": MODEL}
    values[field] = "  "
    with pytest.raises(ValueError, match=field):
        batch_producer.run_proposal_batch("tree", {}, **values)
