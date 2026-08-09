from typing import Literal

import pytest

from lazyportfolio.advisor.contracts import JsonPatchOperation
from lazyportfolio.advisor.patch import DisallowedPatchError, validate_patch, views_patch_path


def _replace_views(node_id: str) -> JsonPatchOperation:
    return JsonPatchOperation(op="replace", path=views_patch_path(node_id), value=[])


def test_the_exact_allowed_path_and_op_is_accepted() -> None:
    validate_patch([_replace_views("equity")], "equity")


def test_empty_patch_is_rejected() -> None:
    with pytest.raises(DisallowedPatchError):
        validate_patch([], "equity")


def test_wrong_node_id_path_is_rejected() -> None:
    with pytest.raises(DisallowedPatchError):
        validate_patch([_replace_views("bond")], "equity")


@pytest.mark.parametrize("op", ["add", "remove"])
def test_ops_other_than_replace_are_rejected(op: Literal["add", "remove"]) -> None:
    operation = JsonPatchOperation(op=op, path="/nodes/equity/constraints/views", value=[])
    with pytest.raises(DisallowedPatchError):
        validate_patch([operation], "equity")


@pytest.mark.parametrize(
    "path",
    [
        "/nodes/equity/constraints/min_weights",
        "/nodes/equity/proxy",
        "/nodes/equity/children",
        "/nodes/equity/constraints/views/0",
        "/root_id",
    ],
)
def test_paths_outside_constraints_views_are_rejected(path: str) -> None:
    operation = JsonPatchOperation(op="replace", path=path, value=[])
    with pytest.raises(DisallowedPatchError):
        validate_patch([operation], "equity")


def test_a_second_operation_on_a_different_node_is_rejected() -> None:
    """MVP invariant (plan §11): a patch may touch exactly one node."""

    with pytest.raises(DisallowedPatchError):
        validate_patch([_replace_views("equity"), _replace_views("bond")], "equity")
