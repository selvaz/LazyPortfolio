"""Shared SQLite persistence for Tree Studio and the Node Advisor domain.

Saved tree configurations (:mod:`lazyportfolio.v2.store`) and run history/
artifacts (:mod:`lazyportfolio.v2.run_history`) used to live in different
places -- one directory of loose JSON files per tree, one opaque cache blob
table keyed by an unstructured hash. Both are single-user, personal data that
needs to survive a process restart, so they share one on-disk database and
one connection helper instead of three copies of the "resolve path / create
parent dir / open sqlite3 connection" boilerplate. Stdlib-only.

The Node Advisor domain tables (revisions, conversations, jobs, proposals,
approvals, outbox -- :mod:`lazyportfolio.advisor`) share the same file and
connection helper, added purely additively (``CREATE TABLE IF NOT EXISTS``)
in ``_ADVISOR_SCHEMA`` below: ``trees``/``runs``/``run_artifacts`` and their
existing callers are untouched by this module
(docs/node-advisor-operational-plan.md §5.1; see
docs/node-advisor-schema-migration-draft.md for the migration this schema
supports).
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

#: Environment variable overriding where the shared database file lives.
_ENV_VAR = "LAZYPORTFOLIO_TREE_DB"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trees (
    name TEXT PRIMARY KEY,
    config TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cache_key TEXT NOT NULL UNIQUE,
    path TEXT NOT NULL,
    kind TEXT NOT NULL,
    tree_id TEXT,
    config_hash TEXT NOT NULL,
    data_as_of TEXT,
    data_fingerprint TEXT NOT NULL,
    created_at TEXT NOT NULL,
    weights TEXT,
    metrics TEXT,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_tree_id ON runs(tree_id);
CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at);

CREATE TABLE IF NOT EXISTS run_artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    kind TEXT NOT NULL,
    content_type TEXT NOT NULL,
    filename TEXT NOT NULL,
    blob BLOB,
    external_artifact_id TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_run_artifacts_run_id ON run_artifacts(run_id);
"""

#: Node Advisor domain tables (docs/node-advisor-operational-plan.md §5.1).
#: All ids are TEXT (str(uuid4()) at the Python layer) -- sqlite3 has no
#: native UUID type, and this matches the existing TEXT-keyed convention
#: (``trees.name``, ``runs.cache_key``) rather than introducing a second id
#: representation.
#:
#: ``tree_heads`` is deliberately a new table, not a ``head_revision_id``
#: column added to the legacy ``trees`` table the plan's §5.1 draft
#: originally sketched: keeping ``trees`` byte-for-byte untouched (see the
#: module docstring) means every existing caller of
#: ``list_saved_models``/``read_model``/``write_model`` needs zero review
#: for this migration, at the cost of one extra table. See
#: docs/node-advisor-schema-migration-draft.md for the reasoning and
#: docs/node-advisor-operational-plan.md §5.1 for the synced invariant.
_ADVISOR_SCHEMA = """
CREATE TABLE IF NOT EXISTS tree_revisions (
    revision_id TEXT PRIMARY KEY,
    tree_id TEXT NOT NULL,
    parent_revision_id TEXT,
    config_json TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    actor_type TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_tree_revisions_tree_id ON tree_revisions(tree_id);

CREATE TABLE IF NOT EXISTS tree_heads (
    tree_id TEXT PRIMARY KEY,
    head_revision_id TEXT NOT NULL REFERENCES tree_revisions(revision_id)
);

CREATE TABLE IF NOT EXISTS legacy_tree_names (
    tree_id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS agent_conversations (
    conversation_id TEXT PRIMARY KEY,
    tree_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_agent_conversations_tree_node
    ON agent_conversations(tree_id, node_id);

CREATE TABLE IF NOT EXISTS agent_messages (
    message_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES agent_conversations(conversation_id),
    role TEXT NOT NULL,
    content_json TEXT NOT NULL,
    revision_id TEXT,
    data_fingerprint TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_agent_messages_conversation
    ON agent_messages(conversation_id, created_at);

CREATE TABLE IF NOT EXISTS agent_jobs (
    job_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES agent_conversations(conversation_id),
    request_message_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    checkpoint_key TEXT NOT NULL,
    session_db_path TEXT,
    budget_json TEXT NOT NULL,
    started_at TEXT,
    heartbeat_at TEXT,
    finished_at TEXT,
    error_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_agent_jobs_status ON agent_jobs(status);

CREATE TABLE IF NOT EXISTS change_proposals (
    proposal_id TEXT PRIMARY KEY,
    batch_id TEXT,
    supersedes_proposal_id TEXT,
    tree_id TEXT NOT NULL,
    base_revision_id TEXT NOT NULL,
    -- Nullable since schema version 1: a compound proposal covers several
    -- nodes and has no single one. Denormalized for display and indexing
    -- only; the authoritative scope is in payload_json, resolved by
    -- lazyportfolio.advisor.contracts.proposal_node_views.
    node_id TEXT,
    kind TEXT NOT NULL,
    producer_kind TEXT NOT NULL,
    producer_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_change_proposals_tree_status ON change_proposals(tree_id, status);
CREATE INDEX IF NOT EXISTS idx_change_proposals_batch ON change_proposals(batch_id);

CREATE TABLE IF NOT EXISTS proposal_approvals (
    approval_id TEXT PRIMARY KEY,
    proposal_id TEXT NOT NULL UNIQUE REFERENCES change_proposals(proposal_id),
    approved_by TEXT NOT NULL,
    approved_at TEXT NOT NULL,
    approved_hash TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    applied_revision_id TEXT,
    result_json TEXT
);

-- The confirmation run's verdict, one row per applied proposal. Separate from
-- proposal_approvals because approving and confirming answer different
-- questions: approval records that a human agreed, confirmation records
-- whether the applied tree actually produces the weights the proposal showed
-- them. A proposal can be approved and then fail confirmation.
CREATE TABLE IF NOT EXISTS proposal_confirmations (
    proposal_id TEXT PRIMARY KEY REFERENCES change_proposals(proposal_id),
    confirmed_at TEXT NOT NULL,
    revision_id TEXT NOT NULL,
    status TEXT NOT NULL,
    max_abs_deviation REAL,
    detail_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS proposal_evidence (
    proposal_id TEXT NOT NULL REFERENCES change_proposals(proposal_id),
    evidence_id TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    excerpt TEXT NOT NULL,
    content_hash TEXT,
    PRIMARY KEY (proposal_id, evidence_id)
);

CREATE TABLE IF NOT EXISTS outbox_events (
    event_id TEXT PRIMARY KEY,
    aggregate_type TEXT NOT NULL,
    aggregate_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    delivered_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_outbox_undelivered ON outbox_events(delivered_at);
"""


def resolve_db_path(db_path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the one shared sqlite file Tree Studio's persistence lives in.

    Precedence: an explicit ``db_path`` argument, then the
    ``LAZYPORTFOLIO_TREE_DB`` env var, then the default --
    ``<repo>/reports/tree_studio/tree_studio.sqlite3``.
    """
    if db_path:
        return Path(db_path).resolve()
    env = os.environ.get(_ENV_VAR)
    if env:
        return Path(env).resolve()
    # .../src/lazyportfolio/v2/db.py -> v2 -> lazyportfolio -> src -> repo root
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "reports" / "tree_studio" / "tree_studio.sqlite3"


#: Bumped when a migration below is added. A database at this version needs
#: no further work; one below it is brought forward on the next connect.
_SCHEMA_VERSION = 1


def _migration_001_nullable_proposal_node(conn: sqlite3.Connection) -> None:
    """Make ``change_proposals.node_id`` nullable, for compound proposals.

    ``CREATE TABLE IF NOT EXISTS`` cannot relax a constraint on a table that
    already exists, and SQLite cannot drop NOT NULL in place, so this is the
    documented rebuild: create the new shape, copy, drop, rename. Existing
    rows are copied byte for byte -- ``payload_json`` and ``content_hash`` are
    never rewritten, so no stored proposal's hash changes.

    Three tables carry foreign keys *into* this one, which is why foreign key
    enforcement is off for the rebuild and ``foreign_key_check`` runs before
    the commit rather than trusting that it went well.
    """

    columns = {row[1] for row in conn.execute("PRAGMA table_info(change_proposals)")}
    if not columns:
        return  # fresh database: the schema above already has the new shape
    notnull = {
        row[1]: row[3] for row in conn.execute("PRAGMA table_info(change_proposals)")
    }
    if not notnull.get("node_id"):
        return  # already rebuilt, or created fresh at the current shape

    conn.execute("PRAGMA foreign_keys = OFF")
    # Without this, RENAME would rewrite references to the renamed table in
    # other tables' definitions -- exactly what must not happen here.
    conn.execute("PRAGMA legacy_alter_table = ON")
    try:
        conn.execute("BEGIN")
        conn.execute(
            "CREATE TABLE change_proposals_migrated ("
            "proposal_id TEXT PRIMARY KEY, batch_id TEXT, supersedes_proposal_id TEXT, "
            "tree_id TEXT NOT NULL, base_revision_id TEXT NOT NULL, node_id TEXT, "
            "kind TEXT NOT NULL, producer_kind TEXT NOT NULL, producer_id TEXT NOT NULL, "
            "payload_json TEXT NOT NULL, content_hash TEXT NOT NULL, status TEXT NOT NULL, "
            "expires_at TEXT NOT NULL, created_at TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO change_proposals_migrated SELECT proposal_id, batch_id, "
            "supersedes_proposal_id, tree_id, base_revision_id, node_id, kind, "
            "producer_kind, producer_id, payload_json, content_hash, status, "
            "expires_at, created_at FROM change_proposals"
        )
        conn.execute("DROP TABLE change_proposals")
        conn.execute("ALTER TABLE change_proposals_migrated RENAME TO change_proposals")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_change_proposals_tree_status "
            "ON change_proposals(tree_id, status)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_change_proposals_batch "
            "ON change_proposals(batch_id)"
        )
        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise sqlite3.IntegrityError(
                f"migration 001 would leave dangling references: {violations[:5]}"
            )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.execute("PRAGMA legacy_alter_table = OFF")
        conn.execute("PRAGMA foreign_keys = ON")


def _migrate(conn: sqlite3.Connection) -> None:
    version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if version >= _SCHEMA_VERSION:
        return
    if version < 1:
        _migration_001_nullable_proposal_node(conn)
    conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")


def connect(db_path: str | os.PathLike[str] | None = None) -> sqlite3.Connection:
    """Open a connection to the shared database, creating the schema if needed.

    Enables foreign key enforcement, WAL (so a long-running reader never
    blocks a writer, needed once the advisor worker/job tables are in use),
    and a busy timeout (retries on ``SQLITE_BUSY`` instead of raising
    immediately) -- required by the compare-and-swap writes
    ``lazyportfolio.advisor``'s repositories rely on
    (docs/node-advisor-operational-plan.md §11: "SQLite apre PRAGMA
    foreign_keys=ON, WAL e busy timeout").
    """
    path = resolve_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.executescript(_SCHEMA)
    conn.executescript(_ADVISOR_SCHEMA)
    _migrate(conn)
    return conn


__all__ = ["connect", "resolve_db_path"]
