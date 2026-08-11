from datetime import UTC, date, datetime
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from lazyportfolio.advisor.contracts import (
    ChangeProposal,
    CounterfactualResult,
    CoverageEntry,
    EvidenceRef,
    JsonPatchOperation,
    ModelProvenance,
    NodeComponent,
    NodeContext,
    ProposedView,
    SnapshotDescriptor,
    ValidationResult,
)

_NOW = datetime(2026, 8, 9, tzinfo=UTC)
_TODAY = date(2026, 8, 9)


def _snapshot() -> SnapshotDescriptor:
    return SnapshotDescriptor(
        schema_version="1.0",
        source="market-data-hub",
        database_identity="test",
        universe=["ticker:VTI"],
        start=_TODAY,
        end=_TODAY,
        data_as_of=_TODAY,
        field="close",
        currency="USD",
        frequency="D",
        coverage=[CoverageEntry(instrument="ticker:VTI", observation_count=100)],
        source_run_ids=[],
        fingerprint="sha256:abc",
    )


def test_node_context_accepts_a_minimal_valid_payload() -> None:
    context = NodeContext(
        schema_version="1.0",
        tree_id=uuid4(),
        revision_id=uuid4(),
        node_id="equity",
        node_name="Equity",
        objective="min_risk",
        mode="forward_backward",
        solved_components=[
            NodeComponent(
                component_id="equity_us",
                kind="child",
                label="Equity US",
                candidate_instrument="ticker:VTI",
                child_node_id="equity_us",
            )
        ],
        allowed_view_instruments=["ticker:VTI"],
        direct_instruments=[],
        child_node_ids=["equity_us"],
        parent_node_id="root",
        parent_candidate_instrument="ticker:ACWI",
        snapshot=_snapshot(),
    )
    assert context.node_id == "equity"
    assert context.solved_components[0].kind == "child"


def test_node_context_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        NodeContext(
            schema_version="1.0",
            tree_id=uuid4(),
            revision_id=uuid4(),
            node_id="equity",
            node_name="Equity",
            objective="min_risk",
            mode="forward_backward",
            not_a_real_field="oops",  # type: ignore[call-arg]
        )


def test_proposed_view_confidence_must_be_in_zero_one_range() -> None:
    with pytest.raises(ValidationError):
        ProposedView(
            instruments={"ticker:VTI": 1.0},
            expected_return=0.02,
            confidence=0.0,
            rationale="test",
        )
    with pytest.raises(ValidationError):
        ProposedView(
            instruments={"ticker:VTI": 1.0},
            expected_return=0.02,
            confidence=1.5,
            rationale="test",
        )
    view = ProposedView(
        instruments={"ticker:VTI": 1.0},
        expected_return=0.02,
        confidence=0.6,
        rationale="test",
    )
    assert view.confidence == 0.6


def test_proposed_view_requires_at_least_one_instrument() -> None:
    with pytest.raises(ValidationError):
        ProposedView(instruments={}, expected_return=0.02, confidence=0.5, rationale="test")


def test_change_proposal_kind_is_an_open_string_not_a_closed_literal() -> None:
    """docs/adr/0001-node-advisor-architecture.md Decision 3, point 1: a
    second producer's kind must be accepted without a schema change here."""

    proposal = _minimal_proposal(kind="a_future_batch_kind_nobody_registered_yet")
    assert proposal.kind == "a_future_batch_kind_nobody_registered_yet"


def test_change_proposal_kind_cannot_be_blank() -> None:
    with pytest.raises(ValidationError):
        _minimal_proposal(kind="")


def test_change_proposal_batch_id_defaults_to_none_and_accepts_a_value() -> None:
    proposal = _minimal_proposal()
    assert proposal.batch_id is None
    batched = _minimal_proposal(batch_id=uuid4())
    assert batched.batch_id is not None


def test_model_provenance_distinguishes_producer_kind() -> None:
    interactive = ModelProvenance(
        producer_kind="interactive_chat", producer_id="node-advisor", model="test-model"
    )
    batch = ModelProvenance(
        producer_kind="scheduled_batch", producer_id="scheduled-research", model="test-model"
    )
    assert interactive.producer_kind != batch.producer_kind


def test_evidence_ref_minimal_payload() -> None:
    ref = EvidenceRef(
        id=uuid4(),
        kind="web",
        locator="https://example.invalid/article",
        title="An article",
        retrieved_at=_NOW,
        excerpt="short excerpt",
    )
    assert ref.publisher is None
    assert ref.supports_claims == []


def _minimal_proposal(
    *, kind: str = "replace_node_views", batch_id: UUID | None = None
) -> ChangeProposal:
    return ChangeProposal(
        id=uuid4(),
        schema_version="1.0",
        kind=kind,
        batch_id=batch_id,
        tree_id=uuid4(),
        base_revision_id=uuid4(),
        node_id="equity",
        snapshot=_snapshot(),
        information_cutoff=_NOW,
        patch=[
            JsonPatchOperation(
                op="replace", path="/nodes/equity/constraints/views", value=[]
            )
        ],
        proposed_views=[],
        rationale="test",
        model_provenance=ModelProvenance(
            producer_kind="interactive_chat", producer_id="node-advisor", model="test-model"
        ),
        validation=ValidationResult(valid=True),
        counterfactual=CounterfactualResult(),
        expires_at=_NOW,
        content_hash="sha256:0000",
    )
