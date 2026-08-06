"""artifact_registry.py — best-effort cataloging of Tree Studio HTML reports
into the shared LazyTools artifact registry (``lazytools.registry``).

This is entirely optional plumbing: Tree Studio's own report/audit export
flow (``project/tree_studio.py``) works identically whether or not it fires.
Three independent reasons it may be a no-op, all of them fine:

1. ``lazytools`` is not installed in this environment at all (it is not a
   declared dependency of this package -- see pyproject.toml -- only
   imported opportunistically, same pattern as
   ``market_data_hub/artifact_registry.py`` in the sibling market-data-hub
   repo).
2. ``lazytools.registry.resolve_db("lazyportfolio_artifacts")`` returns
   ``None`` because ``LAZYPORTFOLIO_ARTIFACTS_DB`` is unset -- this DB is
   declared ``required=False`` in LazyTools' ``KNOWN_DBS``, i.e. opt-in per
   deployment.
3. Anything else goes wrong while registering (a locked/corrupt sqlite file,
   an unexpected exception inside lazytools, ...).

In every case Tree Studio's HTTP handler must keep serving the report to the
browser: registering it as an artifact is a nice-to-have index entry, never
a condition for the request's success.

Unlike market-data-hub's/LazyCrawler's reports (written to disk and
registered via ``content_uri``), Tree Studio's client report is never
written to a file -- it only ever exists as an in-memory/DB-persisted blob
(see ``lazyportfolio.v2.run_history``). So this module registers the report
with ``content=`` (the actual HTML text), not ``content_uri=``.

Whether ``lazytools`` actually resolves to mypy varies by environment (it
resolves to concrete types locally, where the sibling LazyTools checkout is
installed, but CI never installs this optional dependency at all). Either
way makes the *other* environment flag the fallback assignments below as
"unused ignore" or "incompatible assignment" -- there is no single
`# type: ignore` that is simultaneously correct in both, so this file opts
out of `warn_unused_ignores` instead of chasing the two cases.
"""
# mypy: warn-unused-ignores=false

from __future__ import annotations

import sys

try:
    from lazytools.registry import register_artifact, resolve_db
except ImportError:
    resolve_db = None  # type: ignore[assignment]
    register_artifact = None  # type: ignore[assignment]


def register_report_artifact(
    *, title: str, summary: str, tags: list[str], content: str
) -> str | None:
    """Catalog one Tree Studio HTML report as a ``lazyportfolio``/``report`` artifact.

    Best-effort only: swallows every exception (import errors, missing/unset
    ``LAZYPORTFOLIO_ARTIFACTS_DB``, sqlite errors, ...) and prints a warning
    to stderr instead of raising, so callers never need to guard this call.

    Args:
        title: Short human-readable title (derived from the tree config's
            root node name -- there is no session/date concept for an
            on-demand report).
        summary: Cheap-to-read summary of the tree structure/instruments
            involved. Keyword-dense: ``search_artifacts`` only ever matches
            against ``title``/``summary``/``tags``, never ``content``.
        tags: Free-text tags (e.g. ``["tree-studio"]``).
        content: The full report HTML, stored inline (no file exists for
            this report to point a ``content_uri`` at).

    Returns:
        The new artifact's id, or ``None`` if registration was skipped or
        failed.
    """
    if resolve_db is None or register_artifact is None:
        return None
    try:
        db_path = resolve_db("lazyportfolio_artifacts")
        if not db_path:
            return None  # LAZYPORTFOLIO_ARTIFACTS_DB unset -- optional, skip silently
        return register_artifact(  # type: ignore[no-any-return]
            db_path,
            repo="lazyportfolio",
            kind="report",
            title=title,
            summary=summary,
            tags=tags,
            content=content,
        )
    except Exception as exc:
        print(
            f"[tree-studio] artifact registration failed (non-fatal): {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return None
