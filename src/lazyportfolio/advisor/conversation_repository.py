"""Conversation/message persistence (docs/node-advisor-operational-plan.md §5.1/§13 Fase 3).

``agent_messages`` is the append-only conversation log Fase 3's job worker
reads its "fixture" request from -- no LLM in this phase, so a message's
``content_json`` is a plain structured payload (node_id + views), not free
text an LLM would otherwise parse.
"""

from __future__ import annotations

import json
import os
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from lazyportfolio.v2 import db as _db


@dataclass(frozen=True)
class Conversation:
    conversation_id: str
    tree_id: str
    node_id: str
    user_id: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class Message:
    message_id: str
    conversation_id: str
    role: str
    content: dict[str, Any]
    revision_id: str | None
    data_fingerprint: str | None
    created_at: str


def create_conversation(
    tree_id: str,
    node_id: str,
    *,
    user_id: str,
    db_path: str | os.PathLike[str] | None = None,
) -> Conversation:
    conversation_id = str(uuid4())
    now = datetime.now(UTC).isoformat()
    with closing(_db.connect(db_path)) as conn:
        conn.execute(
            "INSERT INTO agent_conversations "
            "(conversation_id, tree_id, node_id, user_id, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (conversation_id, tree_id, node_id, user_id, now, now),
        )
        conn.commit()
    return Conversation(
        conversation_id=conversation_id,
        tree_id=tree_id,
        node_id=node_id,
        user_id=user_id,
        created_at=now,
        updated_at=now,
    )


def get_conversation(
    conversation_id: str, *, db_path: str | os.PathLike[str] | None = None
) -> Conversation | None:
    with closing(_db.connect(db_path)) as conn:
        row = conn.execute(
            "SELECT conversation_id, tree_id, node_id, user_id, created_at, updated_at "
            "FROM agent_conversations WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
    return None if row is None else Conversation(*row)


def list_conversations(
    tree_id: str, *, db_path: str | os.PathLike[str] | None = None
) -> list[Conversation]:
    with closing(_db.connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT conversation_id, tree_id, node_id, user_id, created_at, updated_at "
            "FROM agent_conversations WHERE tree_id = ? ORDER BY created_at DESC",
            (tree_id,),
        ).fetchall()
    return [Conversation(*row) for row in rows]


def add_message(
    conversation_id: str,
    role: str,
    content: dict[str, Any],
    *,
    revision_id: str | None = None,
    data_fingerprint: str | None = None,
    db_path: str | os.PathLike[str] | None = None,
) -> Message:
    message_id = str(uuid4())
    now = datetime.now(UTC).isoformat()
    content_json = json.dumps(content, sort_keys=True, default=str)
    with closing(_db.connect(db_path)) as conn:
        conn.execute(
            "INSERT INTO agent_messages "
            "(message_id, conversation_id, role, content_json, revision_id, "
            "data_fingerprint, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (message_id, conversation_id, role, content_json, revision_id, data_fingerprint, now),
        )
        conn.execute(
            "UPDATE agent_conversations SET updated_at = ? WHERE conversation_id = ?",
            (now, conversation_id),
        )
        conn.commit()
    return Message(
        message_id=message_id,
        conversation_id=conversation_id,
        role=role,
        content=content,
        revision_id=revision_id,
        data_fingerprint=data_fingerprint,
        created_at=now,
    )


def list_messages(
    conversation_id: str, *, db_path: str | os.PathLike[str] | None = None
) -> list[Message]:
    with closing(_db.connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT message_id, conversation_id, role, content_json, revision_id, "
            "data_fingerprint, created_at FROM agent_messages "
            "WHERE conversation_id = ? ORDER BY created_at ASC",
            (conversation_id,),
        ).fetchall()
    return [
        Message(
            message_id=row[0],
            conversation_id=row[1],
            role=row[2],
            content=json.loads(row[3]),
            revision_id=row[4],
            data_fingerprint=row[5],
            created_at=row[6],
        )
        for row in rows
    ]


__all__ = [
    "Conversation",
    "Message",
    "add_message",
    "create_conversation",
    "get_conversation",
    "list_conversations",
    "list_messages",
]
