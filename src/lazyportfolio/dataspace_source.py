"""DataSpace adapter for this repository's Tree Studio store.

Makes LazyPortfolio registrable in a :class:`lazydataspace.DataSpace` so a
workflow spanning several repositories can verify every source's readiness
together, before its first write.

Deliberately thin: no second path resolver and no second read API. The path
comes from :func:`lazyportfolio.v2.db.resolve_db_path` — the one place that
decides which sqlite file Tree Studio's persistence lives in — and callers
reach trees and run history through the v2 API exactly as they do today.
Registering this Source changes nothing about how the repo works standalone.

``lazydataspace`` is an optional dependency (``pip install
lazyportfolio[lazydataspace]``). Nothing else in this package imports this
module, so the repo installs and runs without it.

Example:
    from lazydataspace import DataSpace
    from lazyportfolio.dataspace_source import PortfolioSource

    space = DataSpace(PortfolioSource())
    space.require_ready()
"""

from __future__ import annotations

import os
import sqlite3

from lazydataspace import Health, SourceInfo

from lazyportfolio.v2.db import resolve_db_path

#: What this endpoint offers, mirroring what the store actually holds:
#: saved tree configurations with their revision history, and the structured
#: record of every estimate/backtest/report run.
CAPABILITIES = (
    "portfolio.trees",
    "portfolio.runs",
    "artifacts",
)

#: Presence of this table distinguishes "a readable SQLite file" from
#: "actually this repository's Tree Studio store".
_SENTINEL_TABLE = "trees"


class PortfolioSource:
    """This repository's Tree Studio store, as a DataSpace ``Source``.

    Satisfies the ``lazydataspace.Source`` protocol structurally — no base
    class to inherit.

    Args:
        db_path: Explicit store path. Omit to use this repo's own
            resolution order (``LAZYPORTFOLIO_TREE_DB``, then the
            repo-local default).
    """

    def __init__(self, db_path: str | os.PathLike[str] | None = None) -> None:
        self._db_path = db_path

    @property
    def name(self) -> str:
        return "portfolio"

    @property
    def owner(self) -> str:
        return "lazyportfolio"

    @property
    def capabilities(self) -> tuple[str, ...]:
        return CAPABILITIES

    def describe(self) -> SourceInfo:
        """Return the non-sensitive self-description.

        Carries no path: ``SourceInfo`` has no field for one, and the
        description is written to be safe in a log.
        """
        return SourceInfo(
            name=self.name,
            owner=self.owner,
            capabilities=self.capabilities,
            description=(
                "Tree Studio store: saved portfolio tree configurations with "
                "their revision history, plus structured run history "
                "(weights, metrics, data-as-of, config hash) for every "
                "estimate, backtest and report run. Read via the "
                "lazyportfolio.v2 API."
            ),
        )

    def health(self) -> Health:
        """Resolve the store path, open it read-only and confirm it is ours.

        A real check: it resolves, opens and queries.

        Unlike the other adapters in this ecosystem there is no "nothing
        configured" state to report — this repo's resolver always yields a
        path, falling back to a repo-local default when
        ``LAZYPORTFOLIO_TREE_DB`` is unset. An unset env var therefore shows
        up as "the default store does not exist yet", which is the honest
        answer: the workflow would write to a store nobody configured.

        Opens with ``mode=ro`` so a readiness probe can never create an
        empty store and then report it ready.

        Failure details name the configuration knob but never its value:
        this report is logged, and SQLite errors quote the full path.
        """
        try:
            path = str(resolve_db_path(self._db_path))
        except Exception as exc:
            return Health(ready=False, detail=f"path resolution raised {type(exc).__name__}")

        if not os.path.exists(path):
            return Health(
                ready=False,
                detail="store does not exist (set LAZYPORTFOLIO_TREE_DB or create it)",
            )

        try:
            con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            try:
                row = con.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                    (_SENTINEL_TABLE,),
                ).fetchone()
            finally:
                con.close()
        except Exception as exc:
            # Type only: SQLite errors quote the full file path.
            return Health(ready=False, detail=f"cannot open store: {type(exc).__name__}")

        if row is None:
            return Health(
                ready=False,
                detail=f"database is readable but has no {_SENTINEL_TABLE} table (wrong file?)",
            )
        return Health(ready=True)


__all__ = ["CAPABILITIES", "PortfolioSource"]
