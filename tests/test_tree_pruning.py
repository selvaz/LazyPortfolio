from lazyportfolio.v2.tree_pruning import PruningRule, prune_config


def _metrics(sharpe_delta: float, drawdown_delta: float = 0.0, node_vol: float = 0.10):
    father = {"annualized_sharpe": 0.50, "max_drawdown": -0.20, "annualized_volatility": 0.10}
    node = {
        "annualized_sharpe": 0.50 + sharpe_delta,
        "max_drawdown": -0.20 + drawdown_delta,
        "annualized_volatility": node_vol,
    }
    return {"NODE:Child": node, "FATHER:Child": father}


def test_pruning_promotes_proxy_once_and_preserves_source():
    source = {
        "root_id": "root",
        "nodes": [
            {"id": "root", "name": "Root", "children": ["child"], "instruments": ["ABC"]},
            {"id": "child", "name": "Child", "children": [], "instruments": ["X"], "proxy": "ABC"},
        ],
    }
    candidate, decisions = prune_config(
        source, {"rolling": _metrics(-0.10), "expanding": _metrics(-0.10)}
    )
    assert source["nodes"][0]["children"] == ["child"]
    assert candidate["nodes"] == [
        {"id": "root", "name": "Root", "children": [], "instruments": ["ABC"]}
    ]
    assert decisions[0]["decision"] == "prune"
    assert decisions[0]["action"]["promoted_proxy"] == "ABC"


def test_branch_requires_improvement_in_every_protocol_and_drawdown_guard():
    source = {
        "root_id": "root",
        "nodes": [
            {"id": "root", "name": "Root", "children": ["child"], "instruments": []},
            {"id": "child", "name": "Child", "children": [], "instruments": ["X"], "proxy": "P"},
        ],
    }
    candidate, decisions = prune_config(
        source,
        {"rolling": _metrics(0.04), "expanding": _metrics(0.04, -0.06)},
        PruningRule(min_sharpe_improvement=0.03, max_drawdown_per_vol_ratio=1.0),
    )
    assert decisions[0]["decision"] == "prune"
    assert candidate["nodes"][0]["instruments"] == ["P"]


def test_pruning_parent_lifts_retained_descendant_without_orphaning_it():
    source = {
        "root_id": "root",
        "nodes": [
            {"id": "root", "name": "Root", "children": ["parent"], "instruments": []},
            {
                "id": "parent",
                "name": "Parent",
                "children": ["child"],
                "instruments": [],
                "proxy": "PP",
            },
            {"id": "child", "name": "Child", "children": [], "instruments": ["X"], "proxy": "CP"},
        ],
    }
    parent_arm = {"annualized_sharpe": 0.40, "max_drawdown": -0.20, "annualized_volatility": 0.10}
    father_of_parent = {
        "annualized_sharpe": 0.50,
        "max_drawdown": -0.20,
        "annualized_volatility": 0.10,
    }
    child_arm = {"annualized_sharpe": 0.60, "max_drawdown": -0.20, "annualized_volatility": 0.10}
    father_of_child = {
        "annualized_sharpe": 0.50,
        "max_drawdown": -0.20,
        "annualized_volatility": 0.10,
    }
    metrics = {}
    for protocol in ("rolling", "expanding"):
        metrics[protocol] = {
            "NODE:Parent": parent_arm,
            "FATHER:Parent": father_of_parent,
            "NODE:Child": child_arm,
            "FATHER:Child": father_of_child,
        }
    candidate, decisions = prune_config(source, metrics)
    assert [node["id"] for node in candidate["nodes"]] == ["root", "child"]
    assert candidate["nodes"][0]["instruments"] == ["PP"]
    assert candidate["nodes"][0]["children"] == ["child"]
    child = next(item for item in decisions if item["node_id"] == "child")
    assert child["decision"] == "retain"
    assert child["action"] == {"lifted_from_pruned_ancestor": "parent"}


def test_two_consecutive_pruned_levels_cut_the_whole_branch_without_orphaning():
    source = {
        "root_id": "root",
        "nodes": [
            {"id": "root", "name": "Root", "children": ["parent"], "instruments": []},
            {
                "id": "parent", "name": "Parent", "children": ["child"],
                "instruments": [], "proxy": "PP",
            },
            {
                "id": "child", "name": "Child", "children": ["grandchild"],
                "instruments": ["X"], "proxy": "CP",
            },
            {
                "id": "grandchild", "name": "Grandchild", "children": [],
                "instruments": ["Y"], "proxy": "GP",
            },
        ],
    }
    fails = {"annualized_sharpe": 0.40, "max_drawdown": -0.20, "annualized_volatility": 0.10}
    father = {"annualized_sharpe": 0.50, "max_drawdown": -0.20, "annualized_volatility": 0.10}
    # Grandchild individually beats its father, but sits under two
    # consecutive pruned levels (parent and child) -- the whole branch must
    # be cut rather than lifting it out on its own.
    beats = {"annualized_sharpe": 0.60, "max_drawdown": -0.20, "annualized_volatility": 0.10}
    metrics = {}
    for protocol in ("rolling", "expanding"):
        metrics[protocol] = {
            "NODE:Parent": fails, "FATHER:Parent": father,
            "NODE:Child": fails, "FATHER:Child": father,
            "NODE:Grandchild": beats, "FATHER:Grandchild": father,
        }
    candidate, decisions = prune_config(source, metrics)
    expected_root = {"id": "root", "name": "Root", "children": [], "instruments": ["PP"]}
    assert candidate["nodes"] == [expected_root]
    by_id = {node["id"]: node for node in candidate["nodes"]}
    reachable: set[str] = set()
    frontier = [candidate["root_id"]]
    while frontier:
        node_id = frontier.pop()
        reachable.add(node_id)
        frontier.extend(by_id.get(node_id, {}).get("children", []))
    assert reachable == {node["id"] for node in candidate["nodes"]}
    grandchild = next(item for item in decisions if item["node_id"] == "grandchild")
    assert grandchild["decision"] == "retain"
    assert grandchild["action"] == {"removed_with_ancestor": "parent"}
