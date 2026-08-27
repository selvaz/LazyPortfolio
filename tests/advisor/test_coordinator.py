"""Tree Coordinator: the invariants that only exist because delegation is recursive.

A single-node advisor turn never had to worry about a tree moving underneath
it, about the same node being reached twice, or about a budget shared across
branches -- so those are what this file pins down, alongside the standing
structural guarantee that no write-shaped tool is ever on an LLM's surface.

The LLM is stubbed everywhere here: what is under test is the deterministic
Python around the model, which is where all the authority actually lives.
"""

from __future__ import annotations

import threading
from typing import Any
from uuid import uuid4

import pandas as pd
import pytest

pytest.importorskip("lazybridge", reason="the coordinator requires lazybridge")
pytest.importorskip("lazytools", reason="the coordinator requires lazytools")

from project.advisor import agent as advisor_agent  # noqa: E402
from project.advisor import coordinator, services  # noqa: E402

from lazyportfolio.advisor.repository import create_tree, get_head, save_revision  # noqa: E402
from lazyportfolio.backend import OptimizationDataset  # noqa: E402


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
def tree(tmp_path):
    store_path = str(tmp_path / "store.sqlite3")
    revision = create_tree(_config(), actor_type="human", actor_id="test", db_path=store_path)
    return revision, store_path


@pytest.fixture()
def conversation(tree):
    revision, store_path = tree
    return services.create_conversation(
        revision.tree_id,
        services.COORDINATOR_SCOPE,
        caller_id="test",
        db_path=store_path,
    )


CHILDREN = {"root": ["equity", "bond"], "equity": [], "bond": []}


def _propose_result(instruments: dict[str, float] | None = None):
    return advisor_agent.AdvisorTurnResult(
        route="propose",
        message="SPY over TLT.",
        proposed_views=[
            advisor_agent.CandidateView(
                instruments=instruments or {"ticker:SPY": 1.0, "ticker:TLT": -1.0},
                expected_return=0.03,
                confidence=0.6,
                rationale="test",
            )
        ],
    )


def _explain_result():
    return advisor_agent.AdvisorTurnResult(route="explain", message="Here is why.")


def _build(
    tree,
    conversation,
    *,
    node_id: str = "equity",
    budget: int = 6,
    seen_nodes: set[str] | None = None,
    backend: Any = None,
    pinned_revision_id: str | None = None,
):
    revision, store_path = tree
    return coordinator._build_consult_tool(
        revision.tree_id,
        node_id,
        CHILDREN,
        pinned_revision_id=pinned_revision_id or revision.revision_id,
        batch_id=uuid4(),
        budget=coordinator.ConsultationBudget(budget),
        seen_nodes=seen_nodes if seen_nodes is not None else set(),
        caller_id="test",
        conversation_id=conversation.conversation_id,
        model="test-model",
        backend=backend,
        db_path=store_path,
        base_tools_factory=list,
    )


# --------------------------------------------------------------------- #
# Structural: no write-shaped tool anywhere on a coordinator-built surface
# --------------------------------------------------------------------- #
def test_builder_tools_never_include_a_write_tool() -> None:
    """The tree builder transforms drafts in memory; it must not hold a tool
    that could reach a store, a revision, or a proposal."""

    names = {t.name for t in coordinator._builder_tools()}
    forbidden = ("save", "delete", "apply", "write", "approve", "reject", "onboard")
    offenders = [n for n in names if any(bad in n.lower() for bad in forbidden)]
    assert offenders == [], f"write-shaped tool(s) exposed to the LLM: {offenders}"
    assert "check_name_available" in names


def test_each_node_gets_a_uniquely_named_tool() -> None:
    """Two node ids that sanitize to the same string must still produce
    distinct tool names -- a tool list with duplicates is not something to
    rely on behaving predictably."""

    assert coordinator._consult_tool_name("a-b") != coordinator._consult_tool_name("a_b")
    assert coordinator._consult_tool_name("us/equity").startswith("consult_node__us_equity__")


# --------------------------------------------------------------------- #
# Budget
# --------------------------------------------------------------------- #
def test_budget_of_one_admits_exactly_one_of_two_concurrent_reservations() -> None:
    """Whether one model response's tool calls run concurrently is a
    LazyBridge detail, so the reservation is atomic rather than assuming
    they do not."""

    budget = coordinator.ConsultationBudget(1)
    barrier = threading.Barrier(8)
    granted: list[int] = []
    denied: list[int] = []
    lock = threading.Lock()

    def attempt() -> None:
        barrier.wait()
        try:
            budget.reserve()
        except coordinator.ConsultationBudgetExceeded:
            with lock:
                denied.append(1)
        else:
            with lock:
                granted.append(1)

    threads = [threading.Thread(target=attempt) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(granted) == 1
    assert len(denied) == 7
    assert budget.remaining == 0


def test_an_exhausted_budget_refuses_without_calling_the_model(
    monkeypatch, tree, conversation
) -> None:
    calls: list[str] = []

    def _never(*args: Any, **kwargs: Any):
        calls.append("called")
        raise AssertionError("the model must not be called once the budget is spent")

    monkeypatch.setattr(advisor_agent, "run_node_turn", _never)
    tool = _build(tree, conversation, budget=0)

    result = tool.func("look at this node")

    assert calls == []
    assert "budget" in result["error"]


# --------------------------------------------------------------------- #
# Revision pinning and per-turn dedup
# --------------------------------------------------------------------- #
def test_create_proposal_refuses_a_head_that_moved_after_its_own_check(
    monkeypatch, tree, frame
) -> None:
    """The coordinator's own check happens before ``create_proposal``, which
    then re-reads the head and does slow snapshot/counterfactual work. That
    second window is inside ``create_proposal``, so the pin has to be enforced
    there too -- an outer check alone cannot see it."""

    revision, store_path = tree
    moved: dict[str, str] = {}
    real_load = services.snapshot_service.load_snapshot

    def _move_head_mid_evaluation(*args: Any, **kwargs: Any):
        if not moved:
            new = save_revision(
                revision.tree_id,
                _config(),
                actor_type="human",
                actor_id="concurrent-editor",
                db_path=store_path,
            )
            moved["revision_id"] = new.revision_id
        return real_load(*args, **kwargs)

    monkeypatch.setattr(services.snapshot_service, "load_snapshot", _move_head_mid_evaluation)

    with pytest.raises(services.StaleBaseRevision, match="while this proposal was being"):
        services.create_proposal(
            revision.tree_id,
            "equity",
            [{"instruments": {"ticker:SPY": 1.0, "ticker:TLT": -1.0},
              "expected_return": 0.03, "confidence": 0.6, "source": "test",
              "rationale": "test"}],
            caller_id="test",
            expected_revision_id=revision.revision_id,
            backend=_FakeBackend(frame),
            db_path=store_path,
        )
    assert moved, "the test did not actually move the head"
    assert services.list_proposals(revision.tree_id, db_path=store_path) == []


def test_a_failed_persist_releases_the_node_for_a_retry(monkeypatch, tree, conversation) -> None:
    """Dedup must not punish a failure: a node nothing was ever filed for is
    still a node worth asking again."""

    revision, store_path = tree
    monkeypatch.setattr(advisor_agent, "run_node_turn", lambda *a, **k: _propose_result())

    def _fail(*args: Any, **kwargs: Any):
        raise RuntimeError("market data unavailable")

    monkeypatch.setattr(services, "create_proposal", _fail)
    seen: set[str] = set()
    tool = _build(tree, conversation, seen_nodes=seen)

    result = tool.func("propose something")

    assert "market data unavailable" in result["error"]
    assert "equity" not in seen, "a node with no proposal stayed marked as done"


def test_a_tree_that_moved_mid_turn_refuses_to_persist(
    monkeypatch, tree, conversation, frame
) -> None:
    """The coordinator reasons against a pinned revision; if the head moved
    while it was thinking, the proposal it would file no longer describes the
    tree it reasoned about."""

    revision, store_path = tree
    monkeypatch.setattr(advisor_agent, "run_node_turn", lambda *a, **k: _propose_result())
    tool = _build(tree, conversation, backend=_FakeBackend(frame))

    save_revision(
        revision.tree_id,
        _config(),
        actor_type="human",
        actor_id="someone-else",
        db_path=store_path,
    )
    result = tool.func("propose something")

    assert result["route"] == "refused"
    assert "changed since this turn started" in result["message"]
    assert services.list_proposals(revision.tree_id, db_path=store_path) == []


def test_the_same_node_cannot_produce_two_proposals_in_one_turn(
    monkeypatch, tree, conversation, frame
) -> None:
    """A node is reachable both directly and through an ancestor's
    delegation. Two proposals sharing one base revision are not both
    approvable, so the second is refused rather than filed."""

    revision, store_path = tree
    monkeypatch.setattr(advisor_agent, "run_node_turn", lambda *a, **k: _propose_result())
    seen: set[str] = set()
    tool = _build(tree, conversation, seen_nodes=seen, backend=_FakeBackend(frame))

    first = tool.func("propose something")
    second = tool.func("propose something else")

    assert first["route"] == "propose"
    assert first["proposal_id"]
    assert second["route"] == "refused"
    assert "already produced a proposal" in second["message"]
    assert len(services.list_proposals(revision.tree_id, db_path=store_path)) == 1


def test_an_explain_turn_files_nothing_but_is_still_audited(
    monkeypatch, tree, conversation
) -> None:
    """"List proposals by batch" cannot see an explain-only consultation --
    the conversation event is what keeps it reconstructable."""

    revision, store_path = tree
    monkeypatch.setattr(advisor_agent, "run_node_turn", lambda *a, **k: _explain_result())
    tool = _build(tree, conversation)

    result = tool.func("why is this node weighted like that?")

    assert result["route"] == "explain"
    assert result["proposal_id"] is None
    assert services.list_proposals(revision.tree_id, db_path=store_path) == []
    events = [
        m.content
        for m in services.list_messages(conversation.conversation_id, db_path=store_path)
        if m.content.get("kind") == "coordinator_consultation"
    ]
    assert [e["status"] for e in events] == ["reasoned"]
    assert events[0]["route"] == "explain"


def test_a_failure_inside_a_node_comes_back_as_a_tool_result_not_a_crash(
    monkeypatch, tree, conversation
) -> None:
    """One node blowing up must not abort the whole turn and discard the
    consultations that already succeeded."""

    revision, store_path = tree

    def _boom(*args: Any, **kwargs: Any):
        raise RuntimeError("the model provider is down")

    monkeypatch.setattr(advisor_agent, "run_node_turn", _boom)
    tool = _build(tree, conversation)

    result = tool.func("propose something")

    assert "the model provider is down" in result["error"]
    statuses = [
        m.content.get("status")
        for m in services.list_messages(conversation.conversation_id, db_path=store_path)
        if m.content.get("kind") == "coordinator_consultation"
    ]
    assert statuses == ["error"]


# --------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------- #
def test_a_coordinator_proposal_carries_its_batch_and_producer(
    monkeypatch, tree, conversation, frame
) -> None:
    revision, store_path = tree
    monkeypatch.setattr(advisor_agent, "run_node_turn", lambda *a, **k: _propose_result())
    batch_id = uuid4()
    tool = coordinator._build_consult_tool(
        revision.tree_id,
        "equity",
        CHILDREN,
        pinned_revision_id=revision.revision_id,
        batch_id=batch_id,
        budget=coordinator.ConsultationBudget(6),
        seen_nodes=set(),
        caller_id="test",
        conversation_id=conversation.conversation_id,
        model="test-model",
        backend=_FakeBackend(frame),
        db_path=store_path,
        base_tools_factory=list,
    )

    tool.func("propose something")

    records = services.list_proposals(revision.tree_id, batch_id=batch_id, db_path=store_path)
    assert len(records) == 1
    provenance = records[0].proposal.model_provenance
    assert provenance.producer_id == "tree-coordinator-agent"
    assert provenance.producer_kind == "interactive_chat"
    assert records[0].proposal.batch_id == batch_id
    # Scoping is by tree as well as batch: a batch id alone is unowned.
    assert services.list_proposals(revision.tree_id, batch_id=uuid4(), db_path=store_path) == []
    head = get_head(revision.tree_id, db_path=store_path)
    assert head is not None
    assert head.revision_id == revision.revision_id, "proposing must not move the head"


# --------------------------------------------------------------------- #
# Tree builder: pure transforms, and the onboarded-name guard
# --------------------------------------------------------------------- #
def _builder_tool(name: str, *, db_path: str | None = None):
    return next(t for t in coordinator._builder_tools(db_path=db_path) if t.name == name)


def test_builder_transforms_never_mutate_the_input_config() -> None:
    config = _config()
    before = str(config)

    updated = _builder_tool("update_node").func(config, "equity", {"name": "Equity Sleeve"})

    assert str(config) == before, "the input draft was mutated"
    node = next(n for n in updated["nodes"] if n["id"] == "equity")
    assert node["name"] == "Equity Sleeve"


def test_builder_refuses_to_return_an_invalid_draft() -> None:
    """The same gate ``write_model`` applies, applied one step earlier so the
    model gets an actionable error while it can still act on it."""

    with pytest.raises(ValueError, match="invalid V2 draft"):
        _builder_tool("update_node").func(_config(), "equity", {"instruments": "not-a-list"})


def test_builder_removes_a_whole_subtree_leaving_no_dangling_child() -> None:
    updated = _builder_tool("remove_node").func(_config(), "equity")

    ids = {n["id"] for n in updated["nodes"]}
    assert "equity" not in ids
    assert all("equity" not in (n.get("children") or []) for n in updated["nodes"])


def test_builder_cannot_rename_a_node_or_edit_constraints_by_side_effect() -> None:
    with pytest.raises(ValueError, match="cannot change a node id"):
        _builder_tool("update_node").func(_config(), "equity", {"id": "renamed"})
    with pytest.raises(ValueError, match="use set_constraints"):
        _builder_tool("update_node").func(_config(), "equity", {"constraints": {}})


def test_an_invented_draft_never_reaches_the_caller(monkeypatch, tree, conversation) -> None:
    """The builder's tools validate what they return, but nothing forces the
    model to have used them -- it can put an invented config straight into its
    structured output, so the boundary validates too."""

    revision, store_path = tree

    class _FakeCoordinator:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __call__(self, prompt: str) -> Any:
            from types import SimpleNamespace

            return SimpleNamespace(
                payload=coordinator.CoordinatorTurnResult(
                    message="here is your tree",
                    draft_config={"root_id": "nope", "nodes": [{"id": "nope"}]},
                ),
                error=None,
            )

    import lazybridge

    monkeypatch.setattr(lazybridge, "Agent", _FakeCoordinator)
    monkeypatch.setattr(lazybridge, "LLMEngine", lambda *a, **k: None)

    result = coordinator.run_coordinator_turn(
        revision.tree_id,
        "build me a tree",
        caller_id="test",
        conversation_id=conversation.conversation_id,
        db_path=store_path,
    )

    assert result["draft_config"] is None
    assert "not a valid tree" in result["draft_error"]


def test_builder_reports_an_onboarded_name_as_unavailable(tmp_path) -> None:
    """The named-config row and the advisor's revision history are separate
    stores: editing the named row after onboarding changes nothing about the
    live tree while looking like it did."""

    from lazyportfolio.advisor.migration import migrate_legacy_trees
    from lazyportfolio.v2.store import write_model

    store_path = str(tmp_path / "store.sqlite3")
    write_model("my-tree", _config(), store_path=store_path)
    tool = _builder_tool("check_name_available", db_path=store_path)

    assert tool.func("my-tree")["available"] is True

    migrate_legacy_trees(db_path=store_path)
    after = tool.func("my-tree")

    assert after["available"] is False
    assert "already onboarded" in after["reason"]


# --------------------------------------------------------------------- #
# API scoping
# --------------------------------------------------------------------- #
def test_a_node_conversation_cannot_be_driven_through_the_coordinator_route(
    tree, monkeypatch
) -> None:
    """Posting a node conversation here would run a whole tree-wide turn under
    a conversation scoped to one node."""

    from project.advisor import api

    revision, store_path = tree
    node_conversation = services.create_conversation(
        revision.tree_id, "equity", caller_id="test", db_path=store_path
    )

    with pytest.raises(api.ApiError, match="scoped to a node"):
        api.handle_post(
            f"/api/advisor/coordinator/conversations/{node_conversation.conversation_id}/messages",
            {"text": "review the whole tree"},
            db_path=store_path,
        )


def test_proposals_are_listed_by_tree_and_batch_together(tree, conversation, monkeypatch, frame):
    """A batch id is an unowned identifier: querying by it alone would let a
    caller who guessed one read across trees."""

    from project.advisor import api

    revision, store_path = tree
    monkeypatch.setattr(advisor_agent, "run_node_turn", lambda *a, **k: _propose_result())
    batch_id = uuid4()
    coordinator._build_consult_tool(
        revision.tree_id,
        "equity",
        CHILDREN,
        pinned_revision_id=revision.revision_id,
        batch_id=batch_id,
        budget=coordinator.ConsultationBudget(6),
        seen_nodes=set(),
        caller_id="test",
        conversation_id=conversation.conversation_id,
        model="test-model",
        backend=_FakeBackend(frame),
        db_path=store_path,
        base_tools_factory=list,
    ).func("propose something")

    other_tree = create_tree(_config(), actor_type="human", actor_id="other", db_path=store_path)
    status, payload = api.handle_get(
        f"/api/trees/{other_tree.tree_id}/proposals",
        {"batch_id": [str(batch_id)]},
        db_path=store_path,
    )

    assert status == 200
    assert payload["proposals"] == [], "a batch id leaked a proposal from another tree"

    with pytest.raises(api.ApiError, match="batch_id must be a UUID"):
        api.handle_get(
            f"/api/trees/{revision.tree_id}/proposals",
            {"batch_id": ["not-a-uuid"]},
            db_path=store_path,
        )
