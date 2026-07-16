"""
Message persistence repository.

All operations use short-lived connections (get_connection()) and are
safe to call from any thread simultaneously.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional

from models.message import NormalizedMessage
from storage.database import get_connection

logger = logging.getLogger("yap.storage.messages")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def store_message(msg: NormalizedMessage) -> bool:
    """
    Insert a NormalizedMessage into MESSAGES.
    Returns True if inserted, False if the message_id already existed (idempotent).
    Does NOT increment profile message_count — caller handles that via profiles.py.
    """
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO MESSAGES
                (message_id, thread_id, sender_id, text, timestamp, item_type,
                 is_sent_by_viewer, reply_to_message_id, reply_to_user_id,
                 memory_processed, stored_at, conversation_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                msg.message_id,
                msg.thread_id,
                msg.sender_id,
                msg.text,
                msg.timestamp.isoformat(),
                msg.item_type,
                1 if msg.is_sent_by_viewer else 0,
                msg.reply_to_message_id,
                msg.reply_to_user_id,
                1 if msg.is_historical else 0,  # bootstrap messages skip memory extraction
                _now_iso(),
                msg.conversation_id,
            )
        )
        conn.commit()
        inserted = cursor.rowcount == 1

    if not inserted:
        logger.debug("[STORE] duplicate message_id=%s", msg.message_id)
        return False

    logger.info("[STORE] message inserted message_id=%s sender=%s text=%r",
                msg.message_id, msg.sender_id, (msg.text or "")[:60])
    return True


_SELECT_WITH_USERNAME = """
    SELECT MESSAGES.*, USERS.username AS sender_username
    FROM MESSAGES
    LEFT JOIN USERS ON USERS.user_id = MESSAGES.sender_id
"""


def get_message_by_id(message_id: str) -> Optional[dict]:
    """Fetch a single message dict by ID (with resolved sender_username), or None if not found."""
    with get_connection() as conn:
        row = conn.execute(
            _SELECT_WITH_USERNAME + " WHERE MESSAGES.message_id = ?",
            (message_id,)
        ).fetchone()
        return dict(row) if row else None


def get_messages_for_thread(thread_id: str, limit: int = 40, since_ts: Optional[str] = None) -> List[dict]:
    """
    Fetch recent messages for a thread, oldest-first, with sender_username
    resolved via USERS (falls back to sender_id if no profile exists yet —
    e.g. the bot's own messages, or a user seen for the first time this turn).
    Optionally filtered to messages after since_ts (ISO string).
    """
    with get_connection() as conn:
        if since_ts:
            rows = conn.execute(
                _SELECT_WITH_USERNAME + """
                WHERE MESSAGES.thread_id = ? AND MESSAGES.timestamp > ?
                ORDER BY MESSAGES.timestamp ASC
                LIMIT ?
                """,
                (thread_id, since_ts, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                _SELECT_WITH_USERNAME + """
                WHERE MESSAGES.thread_id = ?
                ORDER BY MESSAGES.timestamp DESC
                LIMIT ?
                """,
                (thread_id, limit)
            ).fetchall()
            rows = list(reversed(rows))
        result = [dict(r) for r in rows]
        for r in result:
            if not r.get("sender_username"):
                r["sender_username"] = r["sender_id"]
        return result


def get_reply_target(message_id: str) -> Optional[dict]:
    """
    Given a message_id, follow its reply_to_message_id chain one level
    and return the target message dict, or None.
    """
    msg = get_message_by_id(message_id)
    if not msg or not msg.get("reply_to_message_id"):
        return None
    return get_message_by_id(msg["reply_to_message_id"])


def is_bot_message(message_id: str, bot_user_id: str) -> bool:
    """Return True if the message was sent by the bot's own IG account (Eve)."""
    msg = get_message_by_id(message_id)
    if not msg:
        return False
    return msg.get("sender_id") == bot_user_id or bool(msg.get("is_sent_by_viewer"))


def get_unprocessed_for_memory(limit: int = 20) -> List[dict]:
    """
    Return messages that have not yet been considered for memory extraction.
    Excludes the bot's own messages, very short/empty messages, and any
    messages currently being processed by another worker tick (memory_in_progress=1).
    """
    with get_connection() as conn:
        rows = conn.execute(
            _SELECT_WITH_USERNAME + """
            WHERE MESSAGES.memory_processed = 0
              AND MESSAGES.memory_in_progress = 0
              AND MESSAGES.is_sent_by_viewer = 0
              AND MESSAGES.text IS NOT NULL
              AND LENGTH(TRIM(MESSAGES.text)) > 3
            ORDER BY MESSAGES.timestamp ASC
            LIMIT ?
            """,
            (limit,)
        ).fetchall()
        result = [dict(r) for r in rows]
        for r in result:
            if not r.get("sender_username"):
                r["sender_username"] = r["sender_id"]
        return result


def get_unprocessed_eve_messages(limit: int = 10) -> List[dict]:
    """
    Return Eve's OWN sent messages not yet considered for self-continuity
    extraction (see intelligence.memory_extractor.extract_eve_self_state).
    Excludes any messages currently being processed by another worker tick
    (memory_in_progress=1) to prevent double-extraction.
    """
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM MESSAGES
            WHERE memory_processed = 0
              AND memory_in_progress = 0
              AND is_sent_by_viewer = 1
              AND text IS NOT NULL
              AND LENGTH(TRIM(text)) > 3
            ORDER BY timestamp ASC
            LIMIT ?
            """,
            (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def mark_memory_in_progress(message_ids: List[str]) -> None:
    """
    Phase 1 of two-phase memory commit.
    Claims the batch atomically: sets memory_in_progress=1 so no other
    worker tick can pick up the same messages concurrently.
    """
    if not message_ids:
        return
    with get_connection() as conn:
        placeholders = ",".join("?" * len(message_ids))
        conn.execute(
            f"UPDATE MESSAGES SET memory_in_progress = 1 WHERE message_id IN ({placeholders})",
            message_ids
        )
        conn.commit()


def mark_memory_processed(message_ids: List[str]) -> None:
    """
    Phase 2 (success path) of two-phase memory commit.
    Marks the batch fully processed: memory_processed=1, memory_in_progress=0.
    Call only AFTER extraction has successfully completed.
    """
    if not message_ids:
        return
    with get_connection() as conn:
        placeholders = ",".join("?" * len(message_ids))
        conn.execute(
            f"""UPDATE MESSAGES
                SET memory_processed = 1, memory_in_progress = 0
                WHERE message_id IN ({placeholders})""",
            message_ids
        )
        conn.commit()


def mark_memory_failed(message_ids: List[str]) -> None:
    """
    Phase 2 (failure path) of two-phase memory commit.
    Rolls back in-progress claim so messages are retried next worker tick.
    Call when extraction raises an exception.
    """
    if not message_ids:
        return
    with get_connection() as conn:
        placeholders = ",".join("?" * len(message_ids))
        conn.execute(
            f"UPDATE MESSAGES SET memory_in_progress = 0 WHERE message_id IN ({placeholders})",
            message_ids
        )
        conn.commit()


def get_lane_messages(thread_id: str, participant_ids: List[str], limit: int = 15) -> List[dict]:
    """
    Fetch recent messages in a thread that involve specific participants.
    Used to build lane-specific context for reply generation (focused
    context, AFTER targeting has already decided Eve should reply — see
    PART 7: lanes must not be used to filter what the social router sees).
    """
    if not participant_ids:
        return get_messages_for_thread(thread_id, limit=limit)
    with get_connection() as conn:
        placeholders = ",".join("?" * len(participant_ids))
        rows = conn.execute(
            _SELECT_WITH_USERNAME + f"""
            WHERE MESSAGES.thread_id = ?
              AND MESSAGES.sender_id IN ({placeholders})
            ORDER BY MESSAGES.timestamp DESC
            LIMIT ?
            """,
            [thread_id] + participant_ids + [limit]
        ).fetchall()
        result = list(reversed([dict(r) for r in rows]))
        for r in result:
            if not r.get("sender_username"):
                r["sender_username"] = r["sender_id"]
        return result


def get_recent_scene(thread_id: str, limit: int = 15) -> List[dict]:
    """
    Fetch the raw, unfiltered recent GC scene — every message regardless of
    lane — with reply metadata and resolved usernames. This is what the
    social targeting layer (attention gate escalation + social judge) must
    see so it can reconstruct who is talking to whom (PART 7). Lanes are a
    downstream focusing tool for reply generation, not a targeting filter.
    """
    return get_messages_for_thread(thread_id, limit=limit)


def get_recent_bot_texts(bot_user_id: str, limit: int = 10) -> List[str]:
    """
    Return the bot's own most recent message texts (any modality), oldest
    first. Used for anti-repetition and by the mode selector's voice-recency
    signal.
    """
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT text FROM MESSAGES
            WHERE sender_id = ? AND text IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (bot_user_id, limit),
        ).fetchall()
        return list(reversed([r["text"] for r in rows if r["text"]]))


def get_messages_for_conversation(conversation_id: str, limit: int = 15) -> List[dict]:
    """Retrieve messages in a conversation thread, oldest first."""
    with get_connection() as conn:
        rows = conn.execute(
            _SELECT_WITH_USERNAME + """
            WHERE MESSAGES.conversation_id = ?
            ORDER BY MESSAGES.timestamp DESC
            LIMIT ?
            """,
            (conversation_id, limit)
        ).fetchall()
        result = list(reversed([dict(r) for r in rows]))
        for r in result:
            if not r.get("sender_username"):
                r["sender_username"] = r["sender_id"]
        return result


def count_newer_messages_in_conversation(conversation_id: str, trigger_message_id: str) -> int:
    """Count how many messages have been added to this conversation since the trigger message."""
    with get_connection() as conn:
        trig = conn.execute("SELECT timestamp FROM MESSAGES WHERE message_id = ?", (trigger_message_id,)).fetchone()
        if not trig:
            return 0
        trig_ts = trig[0]
        row = conn.execute(
            """
            SELECT COUNT(*) FROM MESSAGES
            WHERE conversation_id = ? AND timestamp > ?
            """,
            (conversation_id, trig_ts)
        ).fetchone()
        return row[0] if row else 0

