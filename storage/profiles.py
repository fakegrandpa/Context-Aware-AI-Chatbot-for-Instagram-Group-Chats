"""
User profile repository.

Identity key: user_id (TEXT, Instagram user pk). Never use username as primary key.
Username changes update the existing profile — do not create a second row.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Optional

import config
from storage.database import get_connection

logger = logging.getLogger("yap.storage.profiles")

FAMILIARITY_MAX = 1.0
FAMILIARITY_MIN = 0.0

# Familiarity deltas for interaction types. Direct interaction with Eve is
# stronger evidence than passive presence — see record_passive_activity()
# for the bounded, diminishing passive-exposure curve.
FAMILIARITY_DELTA_DIRECT_CONV = 0.01    # direct conversation with Yap
FAMILIARITY_DELTA_REPLY_TO_YAP = 0.008  # user replied to Yap
FAMILIARITY_DELTA_YAP_REPLY = 0.005     # Yap replied to user

# Relationship tiers auto-derived from familiarity_score. "serious" is
# intentionally not part of this ladder — it describes a tone, not a
# familiarity level, and nothing currently writes it automatically.
_RELATIONSHIP_TIERS = (
    (0.70, "playful"),
    (0.40, "friendly"),
    (0.15, "familiar"),
    (0.0, "new"),
)


def _relationship_for_score(score: float) -> str:
    for threshold, label in _RELATIONSHIP_TIERS:
        if score >= threshold:
            return label
    return "new"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_or_create_user(user_id: str, username: str) -> tuple[dict, bool]:
    """
    Return (user_dict, created).
    - If user_id is new: insert profile, return (profile, True).
    - If user_id exists with different username: update username, return (profile, False).
    - If user_id exists and username matches: just update last_seen, return (profile, False).
    """
    now = _now_iso()
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM USERS WHERE user_id = ?", (user_id,)
        ).fetchone()

        if row is None:
            conn.execute(
                """
                INSERT INTO USERS
                    (user_id, username, first_seen, last_seen, message_count,
                     relationship_to_yap, familiarity_score)
                VALUES (?, ?, ?, ?, 0, 'new', 0.0)
                """,
                (user_id, username, now, now)
            )
            conn.commit()
            profile = dict(conn.execute(
                "SELECT * FROM USERS WHERE user_id = ?", (user_id,)
            ).fetchone())
            logger.info("[PROFILE] created user_id=%s username=%s", user_id, username)
            return profile, True

        profile = dict(row)
        old_username = profile.get("username", "")
        if old_username != username:
            conn.execute(
                "UPDATE USERS SET username = ?, last_seen = ? WHERE user_id = ?",
                (username, now, user_id)
            )
            conn.commit()
            profile["username"] = username
            logger.info("[PROFILE] username changed user_id=%s old=%s new=%s",
                        user_id, old_username, username)
        else:
            conn.execute(
                "UPDATE USERS SET last_seen = ? WHERE user_id = ?",
                (now, user_id)
            )
            conn.commit()
        profile["last_seen"] = now
        return profile, False


def increment_message_count(user_id: str) -> None:
    """Increment message_count exactly once per new stored message."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE USERS SET message_count = message_count + 1 WHERE user_id = ?",
            (user_id,)
        )
        conn.commit()


def resolve_display_name(user_id: str) -> str:
    """
    Resolve a user_id to its best display name: Eve's own name if it's the
    bot's own account, else the profile's preferred_name/username, else the
    raw id as a last resort (e.g. a reply target we've never seen before).

    Used for reply-graph context (e.g. "X is replying to Y's message") where
    the stored MESSAGES row only has sender_id, not a display name.
    """
    if not user_id:
        return "?"
    if config.BOT_USER_ID and user_id == config.BOT_USER_ID:
        return config.BOT_NAME
    profile = get_user(user_id)
    if profile:
        return profile.get("preferred_name") or profile.get("username") or user_id
    return user_id


def get_user(user_id: str) -> Optional[dict]:
    """Fetch a user profile by user_id. Returns None if not found."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM USERS WHERE user_id = ?", (user_id,)
        ).fetchone()
        return dict(row) if row else None


def update_familiarity(user_id: str, delta: float) -> None:
    """
    Increment familiarity_score by delta, clamped to [0.0, 1.0], and keep
    relationship_to_yap honestly derived from the resulting score. Used
    deterministically based on real Eve interactions only — this function
    never grants social permission to interrupt (see conversation/attention.py
    and intelligence/social_judge.py, which never consult familiarity for
    targeting), it only makes replies feel more textured once Eve is already
    talking to someone.
    """
    with get_connection() as conn:
        row = conn.execute(
            "SELECT familiarity_score FROM USERS WHERE user_id = ?", (user_id,)
        ).fetchone()
        if row is None:
            return
        current = row[0] or 0.0
        new_score = max(FAMILIARITY_MIN, min(FAMILIARITY_MAX, current + delta))
        conn.execute(
            "UPDATE USERS SET familiarity_score = ?, relationship_to_yap = ? WHERE user_id = ?",
            (new_score, _relationship_for_score(new_score), user_id)
        )
        conn.commit()


def record_passive_activity(user_id: str) -> None:
    """
    Bounded, diminishing familiarity growth from simply being present and
    participating in the GC (not from Eve replying to them). Evidence
    strength decreases as message_count grows, so someone who has sent 40
    messages gradually becomes familiar without a single high-volume burst
    making them Eve's best friend, and without staying near-zero forever
    just because Eve never happened to reply to them (PART 5).
    """
    with get_connection() as conn:
        row = conn.execute(
            "SELECT familiarity_score, message_count FROM USERS WHERE user_id = ?",
            (user_id,)
        ).fetchone()
        if row is None:
            return
        current = row[0] or 0.0
        message_count = row[1] or 0
        # Base rate stays below FAMILIARITY_DELTA_REPLY_TO_YAP so a single
        # direct interaction with Eve is always stronger evidence than a
        # single passive appearance, even before the diminishing curve.
        delta = 0.005 / (1.0 + message_count * 0.05)
        new_score = max(FAMILIARITY_MIN, min(FAMILIARITY_MAX, current + delta))
        conn.execute(
            "UPDATE USERS SET familiarity_score = ?, relationship_to_yap = ? WHERE user_id = ?",
            (new_score, _relationship_for_score(new_score), user_id)
        )
        conn.commit()


def update_preferred_name(user_id: str, name: str) -> None:
    """
    Set preferred_name from a confirmed identity/name memory (see
    intelligence/memory_extractor.py). Only overwrites with a non-trivial
    value so a low-confidence/joke extraction can't blank out a good name.
    """
    name = (name or "").strip()
    if not name or len(name) > 50:
        return
    with get_connection() as conn:
        conn.execute(
            "UPDATE USERS SET preferred_name = ? WHERE user_id = ?",
            (name, user_id)
        )
        conn.commit()


def update_relationship(user_id: str, relationship: str) -> None:
    """
    Update relationship_to_yap field.
    Allowed: 'new', 'familiar', 'friendly', 'playful', 'serious'
    """
    allowed = {"new", "familiar", "friendly", "playful", "serious"}
    if relationship not in allowed:
        logger.warning("[PROFILE] invalid relationship %r for user_id=%s", relationship, user_id)
        return
    with get_connection() as conn:
        conn.execute(
            "UPDATE USERS SET relationship_to_yap = ? WHERE user_id = ?",
            (relationship, user_id)
        )
        conn.commit()


def build_profile_summary(
    user_id: str,
    memories: list,
    episodic_memories: Optional[list] = None,
    contradictions: Optional[list] = None
) -> dict:
    """
    Build a compact profile dict for use in Gemini prompts.
    Includes user fields + active memories + episodic memories + contradictions.
    """
    profile = get_user(user_id)
    if not profile:
        return {"user_id": user_id, "username": "unknown", "known": False}

    return {
        "user_id": user_id,
        "username": profile.get("username", "unknown"),
        "preferred_name": profile.get("preferred_name"),
        "relationship_to_yap": profile.get("relationship_to_yap", "new"),
        "familiarity_score": profile.get("familiarity_score", 0.0),
        "message_count": profile.get("message_count", 0),
        "language_style": profile.get("language_style"),
        "memories": memories,
        "episodic_memories": episodic_memories or [],
        "contradictions": contradictions or [],
        "known": True,
    }


def detect_language_style(text: str) -> str:
    """
    Heuristically detect the language style (English, Devanagari Hindi, or Hinglish)
    of a message text.
    """
    if not text:
        return "English"
    
    # Devanagari check (\u0900 to \u097F)
    if any("\u0900" <= c <= "\u097F" for c in text):
        return "Devanagari Hindi"
        
    # Roman Hindi / Hinglish stopwords
    hinglish_keywords = {
        "yaar", "kya", "bhai", "hai", "ko", "nhi", "nahi", "tha", "hota", "gaya",
        "aur", "kar", "se", "ho", "re", "tu", "ab", "kab", "aaj", "kal", "ek",
        "kam", "naam", "hua", "hoga", "sath", "saath", "gaye", "raha", "rahi"
    }
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return "English"
        
    hinglish_count = sum(1 for w in words if w in hinglish_keywords)
    if hinglish_count >= 1 or (hinglish_count / len(words) >= 0.12):
        return "Hinglish"
    return "English"


def update_language_style(user_id: str, language_style: str) -> None:
    """
    Update the language_style preference column for a user.
    """
    with get_connection() as conn:
        conn.execute(
            "UPDATE USERS SET language_style = ? WHERE user_id = ?",
            (language_style, user_id)
        )
        conn.commit()

