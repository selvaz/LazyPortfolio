"""JSON Patch allowlist for the Node Advisor MVP (docs/node-advisor-operational-plan.md §11).

Security invariant: "the MVP patch may touch exactly one node and only
``constraints.views``". This module is where that invariant becomes code
instead of a convention -- the approval service (Fase 1) must call
:func:`validate_patch` on the server-reconstructed patch, never trust a
client-supplied one.
"""

from __future__ import annotations

from collections.abc import Iterable

from lazyportfolio.advisor.contracts import JsonPatchOperation


class DisallowedPatchError(ValueError):
    """A patch operation is outside the MVP's ``constraints.views``-only allowlist."""


def views_patch_path(node_id: str) -> str:
    """The single path the MVP allowlist accepts, for a given node."""

    return f"/nodes/{node_id}/constraints/views"


def validate_patch(patch: list[JsonPatchOperation], node_ids: str | Iterable[str]) -> None:
    """Raise unless ``patch`` is exactly one ``replace`` per declared node.

    Set equality, not merely "every operation is permitted": a patch that
    touches only some of the declared nodes would apply a change the proposal
    was not approved for, and a patch with a duplicate path would silently
    let one operation override another. So the operation paths and the
    declared nodes must correspond one to one.

    Accepts a bare ``node_id`` for the single-node case, which is every
    proposal written before compound ones existed.
    """

    declared = {node_ids} if isinstance(node_ids, str) else set(node_ids)
    if not declared:
        raise DisallowedPatchError("a proposal must declare at least one node")
    if not patch:
        raise DisallowedPatchError("patch must contain at least one operation")

    allowed = {views_patch_path(node_id): node_id for node_id in declared}
    seen: set[str] = set()
    for operation in patch:
        if operation.op != "replace":
            raise DisallowedPatchError(
                f"op {operation.op!r} is not allowed in the MVP; only 'replace' is"
            )
        if operation.path not in allowed:
            raise DisallowedPatchError(
                f"path {operation.path!r} is not allowed; only the declared nodes' "
                f"own views are: {sorted(allowed)}"
            )
        if operation.path in seen:
            raise DisallowedPatchError(
                f"path {operation.path!r} appears twice; one operation would "
                "silently override the other"
            )
        seen.add(operation.path)

    missing = set(allowed) - seen
    if missing:
        raise DisallowedPatchError(
            f"patch does not touch every declared node; missing {sorted(missing)}"
        )


__all__ = ["DisallowedPatchError", "validate_patch", "views_patch_path"]
