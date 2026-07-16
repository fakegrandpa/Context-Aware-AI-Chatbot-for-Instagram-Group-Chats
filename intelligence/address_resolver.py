"""
Address Resolver Subsystem — V6.
Performs deterministic turn-ownership (WHO) resolution based on SceneSnapshots.
This resolver implements strict priority rules without querying Gemini or doing semantic speculation.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from models.message import NormalizedMessage
from models.scene import SceneSnapshot

logger = logging.getLogger("yap.intelligence.address_resolver")

# Excluded greetings for address parsing (not to be treated as explicit addresses)
EXCLUDED_GREETINGS = {
    "hi", "hello", "hey", "hii", "helloo", "heyy", "bro", "bruh", "oye",
    "bhai", "bhaiya", "bhe", "yaar", "yaara", "siri", "alexa", "google"
}

# Ownership categories
OWNERSHIP_EVE = "EVE"
OWNERSHIP_SPECIFIC_USER = "SPECIFIC_USER"
OWNERSHIP_OPEN_GROUP = "OPEN_GROUP"
OWNERSHIP_AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True)
class AddressResolution:
    ownership: str  # EVE, SPECIFIC_USER, OPEN_GROUP, AMBIGUOUS
    target_user_id: Optional[str]
    anchor_message_id: str
    session_id: Optional[str]
    confidence: str  # HARD, STRONG, PROVISIONAL
    evidence: Dict[str, any] = field(default_factory=dict)
    continuation_of_eve_interaction: bool = False


class AddressResolver:
    """Resolves WHO a room moment/burst is directed toward using SceneSnapshot and trigger evidence."""

    def __init__(self, bot_user_id: str, bot_username: str):
        self.bot_user_id = str(bot_user_id)
        self.bot_username = str(bot_username).lower()

    def resolve_explicit_target(self, text: str, recent_events: List[NormalizedMessage]) -> Optional[str]:
        """Resolve target user ID if an explicit @mention or vocative is present in the text."""
        if not text:
            return None
        text_clean = re.sub(r'[^\w\s@_.]', ' ', text).strip()
        words = text_clean.split()
        if not words:
            return None

        # 1. Check @mentions
        for w in words:
            if w.startswith("@") and len(w) > 1:
                target_uname = w[1:].lower()
                if target_uname in ("eve", self.bot_username):
                    return self.bot_user_id
                for m in recent_events:
                    if m.sender_username and m.sender_username.lower() == target_uname:
                        return m.sender_id

        # 2. Check first-word vocative
        first_word = words[0].lower()
        if first_word in EXCLUDED_GREETINGS:
            return None

        if first_word in ("eve", self.bot_username):
            return self.bot_user_id

        for m in recent_events:
            if m.sender_username and m.sender_username.lower() == first_word:
                if m.sender_id != self.bot_user_id:
                    return m.sender_id

        return None

    def resolve(self, snapshot: SceneSnapshot, burst_msgs: List[NormalizedMessage]) -> AddressResolution:
        """Evaluate multi-party timeline snapshot and trigger messages to determine WHO owns the turn."""
        if not burst_msgs:
            raise ValueError("burst_msgs must not be empty")

        trigger = burst_msgs[-1]
        text = (trigger.text or "").strip()
        text_lower = text.lower()
        session_id = trigger.conversation_id

        # 1. Native Reply directly to Eve
        if trigger.reply_to_message_id and trigger.reply_to_user_id == self.bot_user_id:
            logger.info("[ADDRESS_RESOLVER] resolved to EVE (origin: native reply)")
            return AddressResolution(
                ownership=OWNERSHIP_EVE,
                target_user_id=self.bot_user_id,
                anchor_message_id=trigger.message_id,
                session_id=session_id,
                confidence="HARD",
                evidence={"native_reply_to_eve": True},
                continuation_of_eve_interaction=True,
            )

        # 2. Explicit deterministic Eve address
        target_uid = self.resolve_explicit_target(text, snapshot.recent_events)
        if target_uid == self.bot_user_id:
            logger.info("[ADDRESS_RESOLVER] resolved to EVE (origin: explicit address)")
            return AddressResolution(
                ownership=OWNERSHIP_EVE,
                target_user_id=self.bot_user_id,
                anchor_message_id=trigger.message_id,
                session_id=session_id,
                confidence="STRONG",
                evidence={"explicit_address_to_eve": True},
                continuation_of_eve_interaction=True,
            )

        # 3. Native Reply to another human
        if trigger.reply_to_message_id and trigger.reply_to_user_id and trigger.reply_to_user_id != self.bot_user_id:
            logger.info("[ADDRESS_RESOLVER] resolved to SPECIFIC_USER=%s (origin: native reply to human)", trigger.reply_to_user_id)
            return AddressResolution(
                ownership=OWNERSHIP_SPECIFIC_USER,
                target_user_id=trigger.reply_to_user_id,
                anchor_message_id=trigger.message_id,
                session_id=session_id,
                confidence="HARD",
                evidence={"native_reply_to_human": True},
                continuation_of_eve_interaction=False,
            )

        # 4. Explicit deterministic other-user address
        if target_uid and target_uid != self.bot_user_id:
            logger.info("[ADDRESS_RESOLVER] resolved to SPECIFIC_USER=%s (origin: explicit address to human)", target_uid)
            return AddressResolution(
                ownership=OWNERSHIP_SPECIFIC_USER,
                target_user_id=target_uid,
                anchor_message_id=trigger.message_id,
                session_id=session_id,
                confidence="STRONG",
                evidence={"explicit_address_to_human": True},
                continuation_of_eve_interaction=False,
            )

        # 5. Active Eve Engagement Session Continuation
        # Check if the trigger belongs to the session in which Eve is currently active
        if session_id and snapshot.eve_engagement.active_session_id == session_id:
            logger.info("[ADDRESS_RESOLVER] resolved to EVE (origin: active session continuation)")
            return AddressResolution(
                ownership=OWNERSHIP_EVE,
                target_user_id=self.bot_user_id,
                anchor_message_id=trigger.message_id,
                session_id=session_id,
                confidence="STRONG",
                evidence={"active_session_continuation": True},
                continuation_of_eve_interaction=True,
            )

        # 6. Dialogue-Session Continuation (Eve previously involved in the session)
        if session_id:
            session_view = next((s for s in snapshot.active_sessions if s.session_id == session_id), None)
            if session_view and session_view.eve_involved:
                logger.info("[ADDRESS_RESOLVER] resolved to EVE (origin: session involving Eve)")
                return AddressResolution(
                    ownership=OWNERSHIP_EVE,
                    target_user_id=self.bot_user_id,
                    anchor_message_id=trigger.message_id,
                    session_id=session_id,
                    confidence="STRONG",
                    evidence={"eve_session_continuation": True},
                    continuation_of_eve_interaction=True,
                )
            elif session_view:
                # Active session with no Eve involvement belongs to the human exchange
                logger.info("[ADDRESS_RESOLVER] resolved to SPECIFIC_USER (origin: active human session continuation)")
                return AddressResolution(
                    ownership=OWNERSHIP_SPECIFIC_USER,
                    target_user_id=None,
                    anchor_message_id=trigger.message_id,
                    session_id=session_id,
                    confidence="STRONG",
                    evidence={"human_session_continuation": True},
                    continuation_of_eve_interaction=False,
                )

        # 7. Open Group broadcast (keywords or ending in "?")
        is_question = text.endswith("?")
        group_keywords = {"anyone", "everyone", "guys", "you guys", "who is", "bro what", "what do you", "what do u"}
        is_group_broadcast = any(k in text_lower for k in group_keywords)
        
        if is_group_broadcast or (is_question and not trigger.reply_to_message_id):
            logger.info("[ADDRESS_RESOLVER] resolved to OPEN_GROUP")
            return AddressResolution(
                ownership=OWNERSHIP_OPEN_GROUP,
                target_user_id=None,
                anchor_message_id=trigger.message_id,
                session_id=session_id,
                confidence="STRONG",
                evidence={"group_broadcast_or_question": True},
                continuation_of_eve_interaction=False,
            )

        # 8. Ambiguous Fallback
        logger.info("[ADDRESS_RESOLVER] resolved to AMBIGUOUS")
        return AddressResolution(
            ownership=OWNERSHIP_AMBIGUOUS,
            target_user_id=None,
            anchor_message_id=trigger.message_id,
            session_id=session_id,
            confidence="PROVISIONAL",
            evidence={"ambiguous_default": True},
            continuation_of_eve_interaction=False,
        )
