"""JSON Patch allowlist for the Node Copilot MVP (docs/node-copilot-operational-plan.md §11).

Security invariant: "the MVP patch may touch exactly one node and only
``constraints.views``". This module is where that invariant becomes code
instead of a convention -- the approval service (Fase 1) must call
:func:`validate_patch` on the server-reconstructed patch, never trust a
client-supplied one.
"""

from __future__ import annotations

from lazyportfolio.copilot.contracts import JsonPatchOperation


class DisallowedPatchError(ValueError):
    """A patch operation is outside the MVP's ``constraints.views``-only allowlist."""


def views_patch_path(node_id: str) -> str:
    """The single path the MVP allowlist accepts, for a given node."""

    return f"/nodes/{node_id}/constraints/views"


def validate_patch(patch: list[JsonPatchOperation], node_id: str) -> None:
    """Raise :class:`DisallowedPatchError` unless ``patch`` is exactly one
    ``replace`` operation on ``node_id``'s ``constraints/views`` path.

    Rejects empty patches too: a proposal with no patch operations changes
    nothing and should never have reached the approval step.
    """

    if not patch:
        raise DisallowedPatchError("patch must contain at least one operation")
    allowed_path = views_patch_path(node_id)
    for operation in patch:
        if operation.op != "replace":
            raise DisallowedPatchError(
                f"op {operation.op!r} is not allowed in the MVP; only 'replace' is"
            )
        if operation.path != allowed_path:
            raise DisallowedPatchError(
                f"path {operation.path!r} is not allowed in the MVP; only "
                f"{allowed_path!r} (node {node_id!r}'s own views) is"
            )


__all__ = ["DisallowedPatchError", "validate_patch", "views_patch_path"]
