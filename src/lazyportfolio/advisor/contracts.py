"""Node Advisor canonical contracts.

Every model here mirrors a section of ``docs/node-advisor-operational-plan.md``
one field at a time -- these are not free-form data classes, they are the
contract the rest of the Node Advisor's services (Fase 1+) validate against.
Reuses :class:`lazyportfolio.models._PortfolioModel` (``extra="forbid"``,
``validate_default=True``) as the shared base, instead of a second private
base class, so a typo in a field name fails validation the same way it
already does for :class:`~lazyportfolio.models.BacktestSpec`.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import Field

from lazyportfolio.models import _PortfolioModel

#: docs/node-advisor-operational-plan.md §4.5 -- the full set of states a
#: ChangeProposal can be in. Transition legality lives in state_machine.py,
#: not here -- this is only the vocabulary.
ProposalStatus = Literal[
    "drafting",
    "failed",
    "pending_approval",
    "rejected",
    "expired",
    "superseded",
    "applying",
    "apply_failed",
    "applied",
    "confirmation_pending",
    "confirmed",
    "confirmation_failed",
]

#: §3.4 point 3 -- who produced a proposal: a human conversation (the Node
#: Advisor MVP), or a scheduled batch job (a future non-interactive producer).
ProducerKind = Literal["interactive_chat", "scheduled_batch"]

#: Mirrors lazyportfolio.v2.contracts.Mode without importing it: the advisor
#: package only ever carries mode as a descriptive string on NodeContext, it
#: never feeds it back into the V2 solver directly.
Mode = Literal["flat", "forward", "forward_backward"]


class CoverageEntry(_PortfolioModel):
    """One instrument's observed data coverage inside a snapshot (§4.2)."""

    instrument: str
    start: date | None = None
    end: date | None = None
    observation_count: int = Field(ge=0)
    source_run_id: str | None = None


class SnapshotDescriptor(_PortfolioModel):
    """§4.2 -- an opaque hash is not enough for audit/freshness; this is the
    full descriptor a fingerprint is computed over."""

    schema_version: Literal["1.0"]
    source: Literal["market-data-hub"]
    database_identity: str
    universe: list[str] = Field(default_factory=list)
    start: date | None = None
    end: date | None = None
    data_as_of: date | None = None
    field: str
    currency: str
    frequency: str
    coverage: list[CoverageEntry] = Field(default_factory=list)
    source_run_ids: list[str] = Field(default_factory=list)
    fingerprint: str


class NodeComponent(_PortfolioModel):
    """§4.1 -- one solved candidate column inside a node's local frame."""

    component_id: str
    kind: Literal["direct", "child"]
    label: str
    candidate_instrument: str
    child_node_id: str | None = None


class RunSummary(_PortfolioModel):
    """Bounded summary of a prior estimate/backtest run referenced by NodeContext.

    Deliberately excludes raw series/weights payloads -- NodeContext is
    consumed by an LLM prompt, not a report renderer.
    """

    run_id: str
    kind: Literal["estimate", "backtest"]
    created_at: datetime
    metrics: dict[str, float] = Field(default_factory=dict)


class NodeContext(_PortfolioModel):
    """§4.1 -- the canonical, LazyPortfolio-produced context for one node.

    ``allowed_view_instruments`` is the authoritative list for views: for a
    child seen from its parent it carries the child's proxy/candidate, never
    its internal terminal tickers. Produced by NodeUniverseResolver
    (Fase 2) -- this module only defines the shape.
    """

    schema_version: Literal["1.0"]
    tree_id: UUID
    revision_id: UUID
    node_id: str
    node_name: str
    objective: str
    mode: Mode
    solved_components: list[NodeComponent] = Field(default_factory=list)
    allowed_view_instruments: list[str] = Field(default_factory=list)
    direct_instruments: list[str] = Field(default_factory=list)
    child_node_ids: list[str] = Field(default_factory=list)
    parent_node_id: str | None = None
    parent_candidate_instrument: str | None = None
    constraints: dict[str, Any] = Field(default_factory=dict)
    current_views: list[dict[str, Any]] = Field(default_factory=list)
    snapshot: SnapshotDescriptor | None = None
    recent_run: RunSummary | None = None


class ModelProvenance(_PortfolioModel):
    """§3.4 point 3 / §4.3 -- distinguishes an interactive Node Advisor
    conversation from a scheduled batch producer."""

    producer_kind: ProducerKind
    producer_id: str
    model: str
    model_version: str | None = None
    prompt_version: str | None = None


class ProposedView(_PortfolioModel):
    """A single candidate Black-Litterman view, mirroring
    ``lazyportfolio.v2.contracts.V2View`` field-for-field (same shape as
    LazyTools' ``lazytools.skills.macro_views.MacroView``, so mapping either
    producer's output into a ChangeProposal is a straight field copy)."""

    instruments: dict[str, float] = Field(min_length=1)
    expected_return: float
    confidence: float = Field(gt=0.0, le=1.0)
    source: str = "node-advisor"
    rationale: str


class JsonPatchOperation(_PortfolioModel):
    """One RFC 6902 operation. patch.py's allowlist restricts which
    op/path combinations are ever accepted for a real proposal -- this
    model only defines the wire shape."""

    op: Literal["add", "replace", "remove"]
    path: str
    value: Any | None = None


class EvidenceRef(_PortfolioModel):
    """§4.4 -- untrusted data: distinguishes origin, retrieved content and
    temporal validity. Never a source of routing/privilege decisions."""

    id: UUID
    kind: Literal["artifact", "web", "datahub", "crawler", "agent_review"]
    locator: str
    title: str
    publisher: str | None = None
    retrieved_at: datetime
    published_at: datetime | None = None
    as_of: date | None = None
    content_hash: str | None = None
    excerpt: str
    supports_claims: list[str] = Field(default_factory=list)


class ValidationIssue(_PortfolioModel):
    """One machine-readable validation error or warning (§6.1)."""

    code: str
    message: str
    path: str | None = None


class ValidationResult(_PortfolioModel):
    valid: bool
    errors: list[ValidationIssue] = Field(default_factory=list)
    warnings: list[ValidationIssue] = Field(default_factory=list)


class CounterfactualResult(_PortfolioModel):
    """§6.3 -- baseline/variant computed on the same in-memory dataset."""

    baseline: dict[str, Any] = Field(default_factory=dict)
    variant: dict[str, Any] = Field(default_factory=dict)
    delta: dict[str, Any] = Field(default_factory=dict)
    turnover_one_way: float | None = None
    solver_versions: dict[str, str] = Field(default_factory=dict)
    seed: int | None = None


class ChangeProposal(_PortfolioModel):
    """§4.3 -- immutable. A revision request ("lower confidence to 0.25")
    creates a new proposal with ``supersedes_proposal_id`` set; it never
    updates this one in place.

    ``kind`` is a plain string, not a closed ``Literal``, because it is
    validated at runtime against a validator registry (§3.4 point 1): the
    MVP registers only ``"replace_node_views"``, and the type must not
    change when another producer kind is added.
    ``batch_id`` is nullable and groups proposals from the same producer
    run (§3.4 point 2); the Node Advisor's own conversational flow always
    leaves it ``None``.
    """

    id: UUID
    schema_version: Literal["1.0"]
    kind: str = Field(min_length=1)
    batch_id: UUID | None = None
    supersedes_proposal_id: UUID | None = None
    tree_id: UUID
    base_revision_id: UUID
    #: ``None`` only on a compound proposal, which covers several nodes and
    #: has no single one. Never read for authority even when set: resolve the
    #: scope through :func:`proposal_node_views`, which dispatches on ``kind``.
    node_id: str | None = None
    snapshot: SnapshotDescriptor
    information_cutoff: datetime
    patch: list[JsonPatchOperation] = Field(default_factory=list)
    proposed_views: list[ProposedView] = Field(default_factory=list)
    #: Set only on ``kind == COMPOUND_KIND``, where it is the authoritative
    #: scope. A node may map to an empty list, meaning "clear this node's
    #: views" -- so emptiness of a *value* is meaningful, while an empty
    #: mapping is not a valid compound proposal.
    node_views: dict[str, list[ProposedView]] | None = None
    rationale: str
    caveats: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    model_provenance: ModelProvenance
    validation: ValidationResult
    counterfactual: CounterfactualResult
    expires_at: datetime
    content_hash: str


#: ``kind`` of a proposal covering exactly one node -- the only kind that
#: existed before compound proposals, and the shape of every stored proposal
#: written until then.
SINGLE_KIND = "replace_node_views"

#: ``kind`` of a proposal covering several nodes, applied as one unit.
COMPOUND_KIND = "replace_nodes_views"


class MalformedProposalScope(ValueError):
    """The proposal's ``kind`` and its scope fields disagree."""


def proposal_node_views(proposal: ChangeProposal) -> dict[str, list[ProposedView]]:
    """The nodes this proposal changes, and the views to put on each.

    The single place any consumer may learn a proposal's scope. Reading
    ``node_id`` or ``proposed_views`` directly is what would let a human be
    shown one node while approval applies several -- so this dispatches on
    ``kind`` and refuses a payload whose fields contradict it, rather than
    silently preferring one representation over the other.
    """

    if proposal.kind == COMPOUND_KIND:
        if not proposal.node_views:
            raise MalformedProposalScope(
                f"proposal {proposal.id} is {COMPOUND_KIND!r} but declares no node_views"
            )
        if proposal.node_id is not None or proposal.proposed_views:
            raise MalformedProposalScope(
                f"proposal {proposal.id} is {COMPOUND_KIND!r} but also carries "
                "single-node fields; the two representations must not both be set"
            )
        return dict(proposal.node_views)

    if proposal.node_views is not None:
        raise MalformedProposalScope(
            f"proposal {proposal.id} is {proposal.kind!r} but carries node_views; "
            f"only {COMPOUND_KIND!r} may"
        )
    if proposal.node_id is None:
        raise MalformedProposalScope(
            f"proposal {proposal.id} is {proposal.kind!r} but has no node_id"
        )
    return {proposal.node_id: list(proposal.proposed_views)}


__all__ = [
    "COMPOUND_KIND",
    "SINGLE_KIND",
    "ChangeProposal",
    "CounterfactualResult",
    "CoverageEntry",
    "MalformedProposalScope",
    "proposal_node_views",
    "EvidenceRef",
    "JsonPatchOperation",
    "Mode",
    "ModelProvenance",
    "NodeComponent",
    "NodeContext",
    "ProducerKind",
    "ProposalStatus",
    "ProposedView",
    "RunSummary",
    "SnapshotDescriptor",
    "ValidationIssue",
    "ValidationResult",
]
