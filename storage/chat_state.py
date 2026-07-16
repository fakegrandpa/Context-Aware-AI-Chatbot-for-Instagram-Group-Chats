"""
Chat State Storage Manager — V6.
Handles monotonic room versioning per thread_id.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from models.message import NormalizedMessage
from storage.database import get_connection

logger = logging.getLogger("yap.storage.chat_state")


def get_room_version(thread_id: str) -> int:
    """Retrieve the current room_version for a thread, default to 0 if not set."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT room_version FROM CHAT_STATE WHERE thread_id = ?",
            (str(thread_id),)
        ).fetchone()
        return row[0] if row else 0


def accept_and_persist_message(msg: NormalizedMessage) -> tuple[bool, int]:
    """
    Atomically:
    1. Detect durable duplicate message by message ID.
    2. Persist the new message in the MESSAGES table if accepted.
    3. Increment the thread's room_version exactly once.
    4. Return (is_accepted, room_version).
    """
    thread_id_str = str(msg.thread_id)
    msg_id_str = str(msg.message_id)
    
    with get_connection() as conn:
        # 1. Persist message if not already present (atomic dedup on message_id)
        now_iso = datetime.now(timezone.utc).isoformat()
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO MESSAGES
                (message_id, thread_id, sender_id, text, timestamp, item_type,
                 is_sent_by_viewer, reply_to_message_id, reply_to_user_id,
                 memory_processed, stored_at, conversation_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                msg_id_str,
                thread_id_str,
                str(msg.sender_id),
                msg.text,
                msg.timestamp.isoformat(),
                msg.item_type,
                1 if msg.is_sent_by_viewer else 0,
                msg.reply_to_message_id,
                msg.reply_to_user_id,
                1 if msg.is_historical else 0,
                now_iso,
                msg.conversation_id,
            )
        )

        if cursor.rowcount == 0:
            # Duplicate transport event - fetch version without increment
            row = conn.execute(
                "SELECT room_version FROM CHAT_STATE WHERE thread_id = ?",
                (thread_id_str,)
            ).fetchone()
            current_version = row[0] if row else 0
            conn.commit()
            logger.debug("[CHAT_STATE] duplicate message_id=%s version=%d (no-op)", msg_id_str, current_version)
            return False, current_version

        # 2. Increment room_version
        row = conn.execute(
            "SELECT room_version FROM CHAT_STATE WHERE thread_id = ?",
            (thread_id_str,)
        ).fetchone()
        
        msg_ts_iso = msg.timestamp.isoformat()
        if row is None:
            new_version = 1
            conn.execute(
                "INSERT INTO CHAT_STATE (thread_id, room_version, last_message_timestamp, updated_at) VALUES (?, 1, ?, ?)",
                (thread_id_str, msg_ts_iso, now_iso)
            )
        else:
            new_version = row[0] + 1
            conn.execute(
                "UPDATE CHAT_STATE SET room_version = ?, last_message_timestamp = ?, updated_at = ? WHERE thread_id = ?",
                (new_version, msg_ts_iso, now_iso, thread_id_str)
            )
            
        conn.commit()
        logger.info("[CHAT_STATE] accepted msg_id=%s in thread=%s version=%d", msg_id_str, thread_id_str, new_version)
        return True, new_version
