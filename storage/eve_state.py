"""
Eve character-state repository — V5.

Ownership-distinct from storage/memories.py, which stores facts ABOUT GC
users. This module stores facts ABOUT Eve herself:

- stable facts: identity that does not casually change (name, age, region).
  Seeded once; not meant to be rewritten by the memory extractor.
- dynamic facts: short-term character continuity (a fictional event Eve
  mentioned, current mood/plan) so she doesn't contradict herself later
  ("kal mera viva hai" -> "viva kaisa gaya" should still connect).

Deliberately lightweight: no cron-driven life generation, no autonomous
event invention. Dynamic entries are only written when the memory
extractor observes Eve's own message establishing something durable.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional

from storage.database import get_connection

logger = logging.getLogger("yap.storage.eve_state")

# Dynamic life-state entries older than this are no longer surfaced as
# "current" context — they still exist in the table but stop being injected
# into prompts. Keeps continuity short-term, per PART 9 ("not a life simulator").
DYNAMIC_STATE_MAX_AGE_HOURS = 72

STABLE_SEED_FACTS = {
    "name": "Eve",
    "age": "20",
    "gender": "female",
    "background": "Maharashtra, India",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_stable_facts_seeded() -> None:
    """
    Idempotently seed Eve's stable core identity facts if not already present.
    Safe to call on every startup.
    """
    now = _now_iso()
    with get_connection() as conn:
        existing = conn.execute(
            "SELECT slot FROM EVE_STATE WHERE state_type = 'stable' AND active = 1"
        ).fetchall()
        existing_slots = {row[0] for row in existing}

        for slot, value in STABLE_SEED_FACTS.items():
            if slot in existing_slots:
                continue
            conn.execute(
                """
                INSERT INTO EVE_STATE
                    (state_type, slot, value, confidence, source_message_id, created_at, updated_at, active)
                VALUES ('stable', ?, ?, 1.0, NULL, ?, ?, 1)
                """,
                (slot, value, now, now),
            )
        conn.commit()


def get_stable_facts() -> List[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM EVE_STATE WHERE state_type = 'stable' AND active = 1 ORDER BY slot"
        ).fetchall()
        return [dict(r) for r in rows]


def add_dynamic_state(
    slot: str,
    value: str,
    confidence: float = 0.7,
    source_message_id: Optional[str] = None,
) -> Optional[int]:
    """
    Record a durable-ish piece of Eve's own life continuity (e.g. slot='event',
    value='has a viva tomorrow'). Not deduplicated aggressively — the extractor
    is expected to only call this for genuinely new claims.
    """
    if not value or not value.strip():
        return None
    now = _now_iso()
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO EVE_STATE
                (state_type, slot, value, confidence, source_message_id, created_at, updated_at, active)
            VALUES ('dynamic', ?, ?, ?, ?, ?, ?, 1)
            """,
            (slot, value.strip(), confidence, source_message_id, now, now),
        )
        conn.commit()
        state_id = cursor.lastrowid
    logger.info("[EVE_STATE] dynamic state added slot=%s value=%r id=%d", slot, value[:60], state_id)
    return state_id


def get_recent_dynamic_state(limit: int = 5) -> List[dict]:
    """Return recent, still-fresh dynamic life-state entries, newest first."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM EVE_STATE
            WHERE state_type = 'dynamic' AND active = 1
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    now = datetime.now(timezone.utc)
    fresh = []
    for r in rows:
        d = dict(r)
        try:
            updated = datetime.fromisoformat(d["updated_at"])
            if updated.tzinfo is None:
                updated = updated.astimezone(timezone.utc)
            age_hours = (now - updated).total_seconds() / 3600.0
        except (ValueError, TypeError):
            age_hours = 0.0
        if age_hours <= DYNAMIC_STATE_MAX_AGE_HOURS:
            fresh.append(d)
    return fresh


def deactivate_dynamic_state(state_id: int) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE EVE_STATE SET active = 0, updated_at = ? WHERE id = ?",
            (_now_iso(), state_id),
        )
        conn.commit()
