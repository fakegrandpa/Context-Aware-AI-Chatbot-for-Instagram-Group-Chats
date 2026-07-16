"""
Unified Eve Turn Ledger storage manager — V5.5.
Stores and retrieves Eve's sent turns across both TEXT and VOICE modalities.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional

from storage.database import get_connection

logger = logging.getLogger("yap.storage.eve_turns")


def store_eve_turn(
    conversation_id: str,
    trigger_message_id: str,
    target_user_id: Optional[str],
    modality: str,
    semantic_summary: str,
    exact_text: Optional[str],
    voice_transcript: Optional[str],
    conversation_version: int,
    session_id: Optional[str] = None,
    snapshot_version: Optional[int] = None,
    speech_act: Optional[str] = None,
    intent_tag: Optional[str] = None,
    stance: Optional[str] = None,
    anchor_message_id: Optional[str] = None,
) -> None:
    """Insert a new Eve turn into the EVE_TURNS ledger."""
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO EVE_TURNS
                (conversation_id, trigger_message_id, target_user_id, modality,
                 semantic_summary, exact_text, voice_transcript, created_at, conversation_version,
                 session_id, snapshot_version, speech_act, intent_tag, stance, anchor_message_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                trigger_message_id,
                target_user_id,
                modality,
                semantic_summary,
                exact_text,
                voice_transcript,
                now,
                conversation_version,
                session_id,
                snapshot_version,
                speech_act,
                intent_tag,
                stance,
                anchor_message_id,
            ),
        )
        conn.commit()
    logger.info(
        "[EVE_TURNS] recorded Eve turn conversation_id=%s modality=%s summary=%r",
        conversation_id,
        modality,
        semantic_summary[:60],
    )


def get_recent_eve_turns(limit: int = 5) -> List[dict]:
    """Retrieve the most recent Eve turns, newest first."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM EVE_TURNS
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_recent_eve_turns_for_thread(thread_id: str, limit: int = 5) -> List[dict]:
    """Retrieve the most recent Eve turns for a specific Instagram thread, newest first."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM EVE_TURNS
            WHERE conversation_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (str(thread_id), limit),
        ).fetchall()
        return [dict(r) for r in rows]

