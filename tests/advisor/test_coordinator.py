"""Tree Coordinator: the invariants that only exist because delegation is recursive.

A single-node advisor turn never had to worry about a tree moving underneath
it, about a second node also wanting to propose, or about a budget shared
across branches -- so those are what this file pins down.

No LLM runs here. The node-reasoning step is a parameter (``_TurnScope.run_turn``),
so a test hands in a function that returns the result it wants to exercise;
the deterministic Python around it, which is where all the authority lives, is
the real subject.
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
from project.advisor import api, coordinator, services  # noqa: E402

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
                "instruments": ["ticker:AGG", "ticker:TLT"],
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


CHILDREN = {"root": ["equity", "bond"], "equity": [], "bond": []}


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


def _proposes(*args: Any, **kwargs: Any) -> advisor_agent.AdvisorTurnResult:
    return advisor_agent.AdvisorTurnResult(
        route="propose",
        message="SPY over TLT.",
        proposed_views=[
            advisor_agent.CandidateView(
                instruments={"ticker:SPY": 1.0, "ticker:TLT": -1.0},
                expected_return=0.03,
                confidence=0.6,
                rationale="test",
            )
        ],
    )


def _explains(*args: Any, **kwargs: Any) -> advisor_agent.AdvisorTurnResult:
    return advisor_agent.AdvisorTurnResult(route="explain", message="Here is why.")


def _scope(tree, conversation, *, run_turn, budget=6, backend=None, **overrides) -> Any:
    revision, store_path = tree
    return coordinator._TurnScope(
        tree_id=revision.tree_id,
        pinned_revision_id=overrides.get("pinned_revision_id", revision.revision_id),
        batch_id=overrides.get("batch_id", uuid4()),
        budget=coordinator.ConsultationBudget(budget),
        proposed_nodes=overrides.get("proposed_nodes", set()),
        caller_id="test",
        conversation_id=conversation.conversation_id,
        model="test-model",
        base_tools=list,
        run_turn=run_turn,
        backend=backend,
        db_path=store_path,
    )


def _events(conversation, store_path) -> list[dict[str, Any]]:
    return [
        m.content
        for m in services.list_messages(conversation.conversation_id, db_path=store_path)
        if m.content.get("kind") == "coordinator_consultation"
    ]


# --------------------------------------------------------------------- #
# Structural
# --------------------------------------------------------------------- #
def test_the_builder_holds_no_write_shaped_tool() -> None:
    """It drafts in memory; nothing on its surface can reach a store, a
    revision, or a proposal."""

    names = {t.name for t in coordinator._builder_tools()}
    forbidden = ("save", "delete", "apply", "write", "approve", "reject", "onboard")
    assert [n for n in names if any(bad in n.lower() for bad in forbidden)] == []
    assert names == {"check_name_available"}


def test_tool_names_are_distinct_and_short_enough_to_send() -> None:
    """Two ids that sanitize alike must not collide, and function-tool APIs
    commonly cap a name at 64 characters."""

    assert coordinator._consult_tool_name("a-b") != coordinator._consult_tool_name("a_b")
    assert coordinator._consult_tool_name("us/equity").startswith("consult_node__us_equity__")
    assert len(coordinator._consult_tool_name("x" * 300)) <= 64


def test_a_node_is_given_its_children_and_no_siblings(tree, conversation) -> None:
    """The whole point of the recursion: the tools a node can reach are its own
    children's, so a branch cannot reach across into another one."""

    handed: dict[str, list[str]] = {}

    def _capture(instruction, *, context, tools, model, agent_name):
        handed[context.node_id] = [t.name for t in tools]
        return _explains()

    scope = _scope(tree, conversation, run_turn=_capture)
    coordinator._build_consult_tool("root", CHILDREN, scope).func("look around")
    coordinator._build_consult_tool("equity", CHILDREN, scope).func("look around")

    assert handed["root"] == [
        coordinator._consult_tool_name("equity"),
        coordinator._consult_tool_name("bond"),
    ]
    assert handed["equity"] == [], "a leaf was handed tools it has no children for"


def test_a_negative_budget_is_rejected_rather_than_read_as_exhausted() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        coordinator.ConsultationBudget(-1)


# --------------------------------------------------------------------- #
# Budget
# --------------------------------------------------------------------- #
def test_a_budget_of_one_admits_exactly_one_of_eight_concurrent_reservations() -> None:
    """Whether one model response's tool calls run concurrently is a LazyBridge
    detail, so the reservation is atomic rather than assuming they do not."""

    budget = coordinator.ConsultationBudget(1)
    barrier = threading.Barrier(8)
    outcomes: list[int | None] = []
    lock = threading.Lock()

    def attempt() -> None:
        barrier.wait()
        reserved = budget.reserve()
        with lock:
            outcomes.append(reserved)

    threads = [threading.Thread(target=attempt) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert [o for o in outcomes if o is not None] == [0]
    assert outcomes.count(None) == 7
    assert budget.remaining == 0


def test_an_exhausted_budget_refuses_without_reasoning_at_all(tree, conversation) -> None:
    def _never(*args: Any, **kwargs: Any):
        raise AssertionError("the model must not be called once the budget is spent")

    tool = coordinator._build_consult_tool(
        "equity", CHILDREN, _scope(tree, conversation, run_turn=_never, budget=0)
    )

    assert "budget" in tool.func("look at this node")["message"]


# --------------------------------------------------------------------- #
# Revision pinning
# --------------------------------------------------------------------- #
def test_a_tree_that_moved_refuses_before_reasoning_even_to_explain(
    tree, conversation
) -> None:
    """An explain path files nothing, so it is tempting to let it through --
    but an answer drawn from a revision this turn never saw would still be
    presented, and audited, as an answer about the pinned one."""

    revision, store_path = tree
    tool = coordinator._build_consult_tool(
        "equity", CHILDREN, _scope(tree, conversation, run_turn=_explains)
    )
    save_revision(
        revision.tree_id, _config(), actor_type="human", actor_id="other", db_path=store_path
    )

    result = tool.func("why is this node weighted like that?")

    assert result["route"] == "refused"
    assert "changed since this turn started" in result["message"]


def test_the_insert_itself_refuses_a_head_that_moved(tree, frame) -> None:
    """The last window is between the final check and the insert, so the
    database evaluates the head in the same statement -- a caller cannot lose
    that race however carefully it checks first."""

    revision, store_path = tree
    views = [
        {
            "instruments": {"ticker:SPY": 1.0, "ticker:TLT": -1.0},
            "expected_return": 0.03,
            "confidence": 0.6,
            "source": "test",
            "rationale": "test",
        }
    ]
    stale_revision_id = revision.revision_id
    save_revision(
        revision.tree_id, _config(), actor_type="human", actor_id="other", db_path=store_path
    )

    with pytest.raises(services.StaleBaseRevision):
        services.create_proposal(
            revision.tree_id,
            "equity",
            views,
            caller_id="test",
            expected_revision_id=stale_revision_id,
            backend=_FakeBackend(frame),
            db_path=store_path,
        )

    assert services.list_proposals(revision.tree_id, db_path=store_path) == []


# --------------------------------------------------------------------- #
# One proposal per turn
# --------------------------------------------------------------------- #
def test_only_one_proposal_is_created_per_turn_even_across_nodes(
    tree, conversation, frame
) -> None:
    """Proposals from one turn share a base revision, so approving one makes
    the rest impossible to apply. The second is refused rather than filed as a
    card that could never be approved -- and that holds for a *different* node,
    not only a repeat of the same one."""

    revision, store_path = tree
    scope = _scope(tree, conversation, run_turn=_proposes, backend=_FakeBackend(frame))
    equity = coordinator._build_consult_tool("equity", CHILDREN, scope)
    bond = coordinator._build_consult_tool("bond", CHILDREN, scope)

    first = equity.func("propose something")
    same_again = equity.func("propose something else")
    other_node = bond.func("propose something here too")

    assert first["route"] == "propose" and first["proposal_id"]
    assert same_again["route"] == "refused"
    assert other_node["route"] == "refused"
    assert "already created for node equity" in other_node["message"]
    assert len(services.list_proposals(revision.tree_id, db_path=store_path)) == 1


def test_a_proposal_is_written_in_one_commit_at_its_final_status(tree, frame) -> None:
    """Written as "drafting" and transitioned afterwards, a failed transition
    left a proposal that existed while its caller believed nothing had been
    written -- and the caller then retries. One commit removes the window."""

    revision, store_path = tree
    services.create_proposal(
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

    (record,) = services.list_proposals(revision.tree_id, db_path=store_path)
    assert record.status == "pending_approval", "a proposal was left mid-write"


def test_a_failed_persist_releases_the_node_for_a_retry(tree, conversation, frame) -> None:
    """Dedup must not punish a failure: a node nothing was filed for is still
    a node worth asking again -- so the retry has to actually succeed."""

    revision, store_path = tree

    class _FailsOnce(_FakeBackend):
        calls = 0

        def load_returns(self, *args: Any, **kwargs: Any):
            _FailsOnce.calls += 1
            if _FailsOnce.calls == 1:
                raise RuntimeError("market data unavailable")
            return super().load_returns(*args, **kwargs)

    scope = _scope(tree, conversation, run_turn=_proposes, backend=_FailsOnce(frame))
    tool = coordinator._build_consult_tool("equity", CHILDREN, scope)

    failed = tool.func("propose something")
    retried = tool.func("propose something")

    assert "market data unavailable" in failed["error"]
    assert retried["route"] == "propose" and retried["proposal_id"]
    assert len(services.list_proposals(revision.tree_id, db_path=store_path)) == 1


# --------------------------------------------------------------------- #
# Audit trail
# --------------------------------------------------------------------- #
def test_an_explain_turn_files_nothing_but_is_still_audited(tree, conversation) -> None:
    """"List proposals by batch" cannot see an explain-only consultation --
    the conversation event is what keeps it reconstructable."""

    revision, store_path = tree
    tool = coordinator._build_consult_tool(
        "equity", CHILDREN, _scope(tree, conversation, run_turn=_explains)
    )

    result = tool.func("why is this node weighted like that?")

    assert result["route"] == "explain" and result["proposal_id"] is None
    assert services.list_proposals(revision.tree_id, db_path=store_path) == []
    events = _events(conversation, store_path)
    assert [e["status"] for e in events] == ["reasoned"]
    assert events[0]["route"] == "explain"


def test_a_failure_inside_a_node_comes_back_as_a_tool_result_not_a_crash(
    tree, conversation
) -> None:
    """One node blowing up must not abort the whole turn and discard the
    consultations that already succeeded."""

    def _boom(*args: Any, **kwargs: Any):
        raise RuntimeError("the model provider is down")

    tool = coordinator._build_consult_tool(
        "equity", CHILDREN, _scope(tree, conversation, run_turn=_boom)
    )

    assert "the model provider is down" in tool.func("propose something")["error"]
    assert [e["status"] for e in _events(conversation, tree[1])] == ["error"]


def test_a_coordinator_proposal_carries_its_batch_and_producer(
    tree, conversation, frame
) -> None:
    revision, store_path = tree
    batch_id = uuid4()
    tool = coordinator._build_consult_tool(
        "equity",
        CHILDREN,
        _scope(
            tree,
            conversation,
            run_turn=_proposes,
            backend=_FakeBackend(frame),
            batch_id=batch_id,
        ),
    )

    tool.func("propose something")

    records = services.list_proposals(revision.tree_id, batch_id=batch_id, db_path=store_path)
    assert len(records) == 1
    provenance = records[0].proposal.model_provenance
    assert provenance.producer_id == "tree-coordinator-agent"
    assert provenance.producer_kind == "interactive_chat"
    assert records[0].proposal.batch_id == batch_id
    head = get_head(revision.tree_id, db_path=store_path)
    assert head is not None
    assert head.revision_id == revision.revision_id, "proposing must not move the head"


# --------------------------------------------------------------------- #
# Draft validation and conversation history -- pure, so tested directly
# --------------------------------------------------------------------- #
def test_the_draft_gate_accepts_a_real_tree_and_rejects_anything_else() -> None:
    """The builder is asked for a config but nothing forces it to return a
    valid one, so this gate -- not the builder's own care -- is what keeps an
    unusable draft out of the editor. ``run_coordinator_turn`` calls it on the
    way out; this covers the gate itself."""

    valid, error = coordinator._validate_draft_config(_config())
    assert valid == _config() and error is None

    rejected, error = coordinator._validate_draft_config({"root_id": "x", "nodes": [{"id": "x"}]})
    assert rejected is None
    assert "not a valid tree" in error

    assert coordinator._validate_draft_config(None) == (None, None)


def test_history_replays_the_exchange_but_not_the_current_message(tree, conversation) -> None:
    """The panel reuses one conversation, so "the other change you mentioned"
    has to reach the model with the exchange it refers to -- once."""

    from lazyportfolio.advisor import conversation_repository

    _, store_path = tree
    add = conversation_repository.add_message
    add(conversation.conversation_id, "user", {"text": "look at equity"}, db_path=store_path)
    add(
        conversation.conversation_id,
        "assistant",
        {"kind": "coordinator_turn_result", "message": "equity looks fine, bond does not"},
        db_path=store_path,
    )
    add(
        conversation.conversation_id,
        "assistant",
        {"kind": "coordinator_consultation", "node_id": "equity", "status": "reasoned"},
        db_path=store_path,
    )
    current = add(
        conversation.conversation_id, "user", {"text": "then do the other one"}, db_path=store_path
    )

    history = coordinator._recent_history(
        conversation.conversation_id,
        current_message_id=current.message_id,
        db_path=store_path,
    )

    assert "Human: look at equity" in history
    assert "You: equity looks fine, bond does not" in history
    assert "then do the other one" not in history
    assert "coordinator_consultation" not in history


def test_flatten_summary_reads_the_shape_off_a_summary(tree) -> None:
    revision, store_path = tree
    summary = services.get_tree_summary(revision.tree_id, db_path=store_path)

    assert coordinator._flatten_summary(summary["root"]) == ("root", CHILDREN)


# --------------------------------------------------------------------- #
# Builder guard
# --------------------------------------------------------------------- #
def test_the_builder_reports_an_onboarded_name_as_unavailable(tmp_path) -> None:
    """The named-config row and the advisor's revision history are separate
    stores: editing the named row after onboarding changes nothing about the
    live tree while looking like it did."""

    from lazyportfolio.advisor.migration import migrate_legacy_trees
    from lazyportfolio.v2.store import write_model

    store_path = str(tmp_path / "store.sqlite3")
    write_model("my-tree", _config(), store_path=store_path)
    (check,) = coordinator._builder_tools(store_path)

    assert check.func("my-tree")["available"] is True

    migrate_legacy_trees(db_path=store_path)
    after = check.func("my-tree")

    assert after["available"] is False
    assert "already onboarded" in after["reason"]


# --------------------------------------------------------------------- #
# API scoping
# --------------------------------------------------------------------- #
def test_a_node_conversation_cannot_be_driven_through_the_coordinator_route(tree) -> None:
    """Posting a node conversation here would run a whole tree-wide turn under
    a conversation scoped to one node."""

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


def test_proposals_are_listed_by_tree_and_batch_together(tree, conversation, frame) -> None:
    """A batch id is an unowned identifier: querying by it alone would let a
    caller who guessed one read across trees."""

    revision, store_path = tree
    batch_id = uuid4()
    coordinator._build_consult_tool(
        "equity",
        CHILDREN,
        _scope(
            tree,
            conversation,
            run_turn=_proposes,
            backend=_FakeBackend(frame),
            batch_id=batch_id,
        ),
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


def test_trees_are_listed_by_a_name_a_person_can_recognise(tmp_path) -> None:
    """A tree_id is a UUID nobody can pick out of a list, so the picker needs
    the root's display name -- read from the head config, so a tree created
    directly (never migrated from a named model) still shows something."""

    store_path = str(tmp_path / "store.sqlite3")
    first = create_tree(_config(), actor_type="human", actor_id="test", db_path=store_path)
    renamed = _config()
    renamed["nodes"][0]["name"] = "Portafoglio Globale"
    second = create_tree(renamed, actor_type="human", actor_id="test", db_path=store_path)

    status, payload = api.handle_get("/api/trees", db_path=store_path)

    assert status == 200
    by_id = {t["tree_id"]: t for t in payload["trees"]}
    assert by_id[first.tree_id]["name"] == "Root"
    assert by_id[second.tree_id]["name"] == "Portafoglio Globale"
    assert by_id[second.tree_id]["revision_id"] == second.revision_id


def test_an_oversized_tree_is_an_http_error_not_a_dropped_request(tmp_path) -> None:
    """``TreeTooLargeError`` is a supported outcome for a valid tree, and only
    ``ApiError`` reaches the HTTP layer's error translation -- without this the
    caller gets a dropped connection instead of an answer. Built past the real
    cap rather than lowering it, so the path under test is the real one."""

    from lazyportfolio.advisor.node_universe import MAX_SUMMARY_NODES

    # Siblings need distinct proxies, so each child gets its own ticker.
    children = [f"n{i}" for i in range(MAX_SUMMARY_NODES)]
    oversized = {
        "root_id": "root",
        "currency": "USD",
        "nodes": [
            {
                "id": "root",
                "name": "Root",
                "children": children,
                "instruments": [],
                "goal": {"objective": "min_risk"},
                "constraints": {},
            },
            *(
                {
                    "id": child,
                    "name": child,
                    "children": [],
                    "instruments": [f"ticker:{child.upper()}"],
                    "proxy": f"ticker:{child.upper()}",
                    "goal": {"objective": "min_risk"},
                    "constraints": {},
                }
                for child in children
            ),
        ],
        "backtest": {
            "benchmark": {
                "name": "B0",
                "weights": {
                    f"ticker:{child.upper()}": 1.0 / len(children) for child in children
                },
            }
        },
    }
    store_path = str(tmp_path / "store.sqlite3")
    revision = create_tree(oversized, actor_type="human", actor_id="test", db_path=store_path)

    with pytest.raises(api.ApiError) as caught:
        api.handle_get(f"/api/trees/{revision.tree_id}/summary", db_path=store_path)

    assert caught.value.status == 413
    assert "too large" in caught.value.message
