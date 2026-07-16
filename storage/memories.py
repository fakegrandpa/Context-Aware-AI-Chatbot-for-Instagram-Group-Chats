"""
Memory repository — V4 Claim/Belief Quality.

Implements contradiction-aware memory tracking with status transitions:
candidate, active, conflicted, superseded, rejected.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import List, Optional

from storage.database import get_connection

logger = logging.getLogger("yap.storage.memories")

VALID_MEMORY_TYPES = {"identity", "preference", "personal_fact", "relationship", "episodic"}

# Messages too short/trivial to store memory about
TRIVIAL_TEXTS = {
    "hi", "hello", "hey", "ok", "okay", "lol", "lmao", "bruh", "bro", "fr",
    "😭", "💀", "haha", "yeah", "nah", "yep", "nope", "k", "kk", "hm", "hmm",
    "idk", "ig", "true", "facts", "bet", "alr", "wait", "wtf", "omg", "stfu",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_fact(fact: str) -> str:
    """Normalize for duplicate detection: lowercase, collapse whitespace."""
    return re.sub(r"\s+", " ", fact.lower().strip())


def add_claim_memory(
    user_id: str,
    memory_type: str,
    slot: str,
    value: str,
    claim_type: str,
    confidence: float,
    source_message_id: Optional[str] = None,
) -> Optional[int]:
    """
    Store an extracted memory claim. Applies state transitions and checks for contradictions
    for mutually exclusive identity slots (e.g. name, age, city, college, course).
    """
    if memory_type not in VALID_MEMORY_TYPES:
        logger.warning("[MEMORY] invalid memory_type=%r", memory_type)
        return None
    if not value or not value.strip():
        return None

    normalized = _normalize_fact(value)
    if normalized in TRIVIAL_TEXTS:
        return None

    now = _now_iso()
    claim_type = claim_type.upper() if claim_type else "NEW"

    # Define mutually exclusive slots
    exclusive_slots = {"name", "age", "city", "college", "course"}
    is_exclusive = (memory_type == "identity" and slot in exclusive_slots)

    with get_connection() as conn:
        # 1. Fetch any existing active/candidate memories for this user and slot
        existing_rows = conn.execute(
            """
            SELECT * FROM MEMORIES
            WHERE user_id = ? AND memory_type = ? AND slot = ? AND status IN ('active', 'candidate', 'conflicted')
            """,
            (user_id, memory_type, slot)
        ).fetchall()

        existing_list = [dict(r) for r in existing_rows]

        # 2. Check for exact normalized duplicate first (unless it is an explicit correction)
        if claim_type != "CORRECTION":
            for ext in existing_list:
                if ext["normalized_fact"] == normalized:
                    # This is a support claim for an existing memory
                    new_status = ext["status"]
                    # If it was candidate, maybe promote it to active if confidence or support is high
                    new_support = ext["support_count"] + 1
                    if ext["status"] == "candidate" and (new_support >= 2 or confidence >= 0.6):
                        new_status = "active"

                    conn.execute(
                        """
                        UPDATE MEMORIES
                        SET support_count = ?, confidence = ?, status = ?, updated_at = ?, active = ?
                        WHERE id = ?
                        """,
                        (
                            new_support,
                            max(confidence, ext["confidence"]),
                            new_status,
                            now,
                            1 if new_status == "active" else 0,
                            ext["id"]
                        )
                    )
                    conn.commit()
                    logger.info(
                        "[MEMORY] supported existing memory_id=%d slot=%s new_support=%d status=%s",
                        ext["id"], slot, new_support, new_status
                    )
                    return None

        # 3. Handle mutually exclusive slot conflict or corrections
        if is_exclusive and existing_list:
            if claim_type == "CORRECTION":
                logger.info("[MEMORY] CORRECTION received. Superseding old memories for slot=%s", slot)
                # Supersede all old active/candidate/conflicted memories for this slot
                for old_mem in existing_list:
                    conn.execute(
                        "UPDATE MEMORIES SET status = 'superseded', active = 0, updated_at = ? WHERE id = ?",
                        (now, old_mem["id"])
                    )
                
                # Insert the new correction as active
                cursor = conn.execute(
                    """
                    INSERT INTO MEMORIES
                        (user_id, memory_type, slot, value, normalized_fact, status, claim_type,
                         support_count, contradiction_count, confidence, source_message_id,
                         created_at, updated_at, active)
                    VALUES (?, ?, ?, ?, ?, 'active', ?, 1, 0, ?, ?, ?, ?, 1)
                    """,
                    (user_id, memory_type, slot, value.strip(), normalized, claim_type,
                     confidence, source_message_id, now, now)
                )
                conn.commit()
                return cursor.lastrowid
            
            else:
                active_mems = [m for m in existing_list if m["status"] == "active"]
                if active_mems:
                    # Contradiction: Mark the conflict, do NOT overwrite the active memory
                    logger.info("[MEMORY] CONTRADICTION detected for slot=%s: new=%r vs existing=%r",
                                slot, value, [m["value"] for m in active_mems])
                    
                    # Increment contradiction count on the active memory
                    for active_mem in active_mems:
                        conn.execute(
                            """
                            UPDATE MEMORIES
                            SET contradiction_count = contradiction_count + 1, status = 'conflicted', updated_at = ?
                            WHERE id = ?
                            """,
                            (now, active_mem["id"])
                        )

                    # Insert the new claim as status = 'conflicted' and inactive (active = 0)
                    cursor = conn.execute(
                        """
                        INSERT INTO MEMORIES
                            (user_id, memory_type, slot, value, normalized_fact, status, claim_type,
                             support_count, contradiction_count, confidence, source_message_id,
                             created_at, updated_at, active)
                        VALUES (?, ?, ?, ?, ?, 'conflicted', ?, 1, 1, ?, ?, ?, ?, 0)
                        """,
                        (user_id, memory_type, slot, value.strip(), normalized, claim_type,
                         confidence, source_message_id, now, now)
                    )
                    conn.commit()
                    return cursor.lastrowid

        # 4. Standard insertion (no conflicts or corrections)
        # Determine status: JOKE_OR_UNCERTAIN -> candidate. Low confidence -> candidate. Otherwise active.
        if claim_type == "JOKE_OR_UNCERTAIN" or confidence < 0.5:
            initial_status = "candidate"
            is_active = 0
        else:
            initial_status = "active"
            is_active = 1

        cursor = conn.execute(
            """
            INSERT INTO MEMORIES
                (user_id, memory_type, slot, value, normalized_fact, status, claim_type,
                 support_count, contradiction_count, confidence, source_message_id,
                 created_at, updated_at, active)
            VALUES (?, ?, ?, ?, ?, ?, ?, 1, 0, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                memory_type,
                slot,
                value.strip(),
                normalized,
                initial_status,
                claim_type,
                confidence,
                source_message_id,
                now,
                now,
                is_active
            )
        )
        conn.commit()
        memory_id = cursor.lastrowid

    logger.info(
        "[MEMORY] stored user_id=%s type=%s slot=%s value=%r status=%s id=%d",
        user_id, memory_type, slot, value[:50], initial_status, memory_id
    )
    return memory_id


def add_memory(
    user_id: str,
    memory_type: str,
    fact: str,
    confidence: float,
    source_message_id: Optional[str] = None,
) -> Optional[int]:
    """
    Backwards compatibility wrapper over add_claim_memory.
    Defaults claim_type to NEW and slot to 'general' or 'name'.
    """
    slot = "name" if (memory_type == "identity" and "name" in fact.lower()) else "general"
    return add_claim_memory(
        user_id=user_id,
        memory_type=memory_type,
        slot=slot,
        value=fact,
        claim_type="NEW",
        confidence=confidence,
        source_message_id=source_message_id
    )


def get_active_memories(user_id: str, limit: int = 10) -> List[dict]:
    """
    Retrieve active memories for a user.
    """
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM MEMORIES
            WHERE user_id = ? AND status = 'active'
            ORDER BY confidence DESC, updated_at DESC
            LIMIT ?
            """,
            (user_id, limit)
        ).fetchall()
        return [{**dict(r), "fact": r["value"]} for r in rows]


def get_episodic_memories(user_id: str, limit: int = 5) -> List[dict]:
    """Retrieve recent episodic memories for a user."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM MEMORIES
            WHERE user_id = ? AND memory_type = 'episodic' AND status = 'active'
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (user_id, limit)
        ).fetchall()
        return [{**dict(r), "fact": r["value"]} for r in rows]


def get_relevant_memories(user_id: str, topic_words: List[str], limit: int = 8) -> List[dict]:
    """
    Fetch active memories for a user, optionally filtered by topic word relevance.
    """
    if not topic_words:
        return get_active_memories(user_id, limit=limit)

    with get_connection() as conn:
        like_clauses = " OR ".join(["normalized_fact LIKE ?"] * len(topic_words))
        params = [user_id] + [f"%{w.lower()}%" for w in topic_words] + [limit]
        rows = conn.execute(
            f"""
            SELECT * FROM MEMORIES
            WHERE user_id = ? AND status = 'active' AND ({like_clauses})
            ORDER BY confidence DESC, updated_at DESC
            LIMIT ?
            """,
            params
        ).fetchall()
        result = [dict(r) for r in rows]
        if len(result) < limit:
            existing_ids = {r["id"] for r in result}
            extra = conn.execute(
                """
                SELECT * FROM MEMORIES
                WHERE user_id = ? AND status = 'active'
                ORDER BY confidence DESC, updated_at DESC
                LIMIT ?
                """,
                (user_id, limit)
            ).fetchall()
            for r in extra:
                if r["id"] not in existing_ids and len(result) < limit:
                    result.append(dict(r))
        return [{**dict(r), "fact": r["value"]} for r in result]


def get_unresolved_contradictions(user_id: str) -> List[dict]:
    """
    Retrieve any conflicting claims for this user.
    Returns memories with status = 'conflicted' so the bot can notice the discrepancies.
    """
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM MEMORIES
            WHERE user_id = ? AND status = 'conflicted'
            ORDER BY updated_at DESC
            """,
            (user_id,)
        ).fetchall()
        return [{**dict(r), "fact": r["value"]} for r in rows]


def deactivate_memory(memory_id: int) -> None:
    """Soft-delete a memory by setting status='rejected' or active=0."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE MEMORIES SET status = 'rejected', active = 0, updated_at = ? WHERE id = ?",
            (_now_iso(), memory_id)
        )
        conn.commit()
    logger.info("[MEMORY] deactivated id=%s", memory_id)


def update_memory_confidence(memory_id: int, confidence: float) -> None:
    """Update the confidence score for an existing memory."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE MEMORIES SET confidence = ?, updated_at = ? WHERE id = ?",
            (confidence, _now_iso(), memory_id)
        )
        conn.commit()


def is_trivial_text(text: str) -> bool:
    """Return True if a message text is too trivial to extract memory from."""
    if not text:
        return True
    normalized = text.strip().lower()
    if normalized in TRIVIAL_TEXTS:
        return True
    if len(normalized) <= 3:
        return True
    return False

