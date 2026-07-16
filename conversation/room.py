"""
Room State and Conversation Linker Subsystem — V5.5.
Manages room states, speaker interactions, online thread linking,
turn ownership, intervention routing, and pending turns.
"""
from __future__ import annotations

import logging
import math
import random
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field
from instagrapi import Client

import config
from models.message import NormalizedMessage
from storage import messages as msg_store
from storage.database import get_connection
from intelligence import gemini_pool

logger = logging.getLogger("yap.conversation.room")

# Regexes matching targeting patterns
_AT_EVE = re.compile(r'@eve\b', re.IGNORECASE)
_EVE_PREFIX = re.compile(r'^@?eve\b', re.IGNORECASE)
_EVE_STANDALONE = re.compile(r'(?:^|[\s,!?])@?eve(?:[\s,!?]|$)', re.IGNORECASE)
_AT_OTHER_USER = re.compile(r'@(\w+)\b', re.IGNORECASE)
_FIRST_WORD = re.compile(r'^@?([A-Za-z][A-Za-z0-9_.]{1,20})\b[,:]?\s')

DISENTANGLE_INSTRUCTION = """You are a conversation disentanglement assistant for a busy group chat.
Your task is to identify which previous message the CURRENT MESSAGE is responding to.

Candidates are listed with their [msg_id], sender, and text.
Analyze the CURRENT MESSAGE content, context, and candidates.
Return the [msg_id] of the message it is replying/responding to.
If it is starting a new conversation or does not respond to any of the candidates, return null.

Output ONLY JSON matching this schema:
{
  "parent_message_id": "string or null",
  "reason": "short explanation"
}
"""


class _DisentanglementSchema(BaseModel):
    parent_message_id: Optional[str] = Field(default=None, description="The msg_id of the candidate message, or null/None if none of the candidates match.")
    reason: str = Field(description="A short internal reason for the choice.")


def _vocative_other_user(
    text: str,
    current_sender_id: str,
    scene_before: List[dict],
    bot_user_id: str,
    bot_username: str,
) -> Optional[str]:
    """
    If the message opens by naming another user who is actually a known
    recent participant in this GC, return that username.
    """
    match = _FIRST_WORD.match(text)
    if not match:
        return None
    candidate = match.group(1).lower()
    if candidate in ("eve", (bot_username or "").lower()):
        return None

    known_usernames = {
        (m.get("sender_username") or "").lower()
        for m in scene_before
        if m.get("sender_id") not in (current_sender_id, bot_user_id)
        and not m.get("is_sent_by_viewer")
        and m.get("sender_username")
    }
    if candidate in known_usernames:
        return candidate
    return None


def build_disentangle_prompt(current_msg: NormalizedMessage, candidates: List[dict]) -> str:
    lines = ["CANDIDATES:"]
    for c in candidates:
        lines.append(f"- [msg_id={c['message_id']}] {c.get('sender_username') or c.get('sender_id')}: {c.get('text') or ''}")
    lines.append("")
    lines.append(f"CURRENT MESSAGE: {current_msg.sender_username}: {current_msg.text or ''}")
    return "\n".join(lines)


class SpeakerInteractionMap:
    """Tracks and decays pairwise interaction strength in memory."""

    def __init__(self, decay_rate: float = 0.005):
        self.decay_rate = decay_rate
        self.interactions: dict[tuple[str, str], float] = {}
        self.last_update = time.time()
        self.lock = threading.Lock()

    def _decay(self):
        now = time.time()
        elapsed = now - self.last_update
        self.last_update = now
        if elapsed <= 0:
            return
        decay_factor = math.exp(-self.decay_rate * elapsed)
        for pair in list(self.interactions.keys()):
            self.interactions[pair] *= decay_factor
            if self.interactions[pair] < 0.05:
                del self.interactions[pair]

    def record_interaction(self, u1: str, u2: str, increment: float):
        if not u1 or not u2 or u1 == u2:
            return
        pair = (min(u1, u2), max(u1, u2))
        with self.lock:
            self._decay()
            current = self.interactions.get(pair, 0.0)
            self.interactions[pair] = max(0.0, min(1.0, current + increment))

    def get_interaction(self, u1: str, u2: str) -> float:
        if not u1 or not u2 or u1 == u2:
            return 0.0
        pair = (min(u1, u2), max(u1, u2))
        with self.lock:
            self._decay()
            return self.interactions.get(pair, 0.0)


class ThreadLinker:
    """Links incoming messages to parent conversations using hierarchical evidence."""

    def __init__(self, bot_user_id: str, bot_username: str):
        self.bot_user_id = bot_user_id
        self.bot_username = bot_username

    def link_message(self, msg: NormalizedMessage, cl: Client, interaction_map: SpeakerInteractionMap) -> str:
        """Assign conversation_id using hierarchical evidence."""
        # 1. Instagram Native Reply
        if msg.reply_to_message_id:
            parent = msg_store.get_message_by_id(msg.reply_to_message_id)
            if parent and parent.get("conversation_id"):
                logger.info("[LINKER] Tier 1: linked to parent conversation=%s via native reply", parent["conversation_id"])
                return parent["conversation_id"]

        # Fetch recent thread messages to check mentions & vocatives
        recent = msg_store.get_messages_for_thread(msg.thread_id, limit=25)
        recent = [m for m in recent if m["message_id"] != msg.message_id]

        text = (msg.text or "").strip()
        text_lower = text.lower()

        # Check explicit Eve summons
        is_at_eve = bool(
            _AT_EVE.search(text) or
            _EVE_PREFIX.search(text) or
            _EVE_STANDALONE.search(text) or
            (self.bot_username and re.search(r'\b' + re.escape(self.bot_username.lower()) + r'\b', text_lower))
        )
        if is_at_eve:
            for r in reversed(recent):
                if r.get("conversation_id") and (r.get("sender_id") == self.bot_user_id or r.get("is_sent_by_viewer")):
                    logger.info("[LINKER] Tier 1: linked to Eve conversation=%s via Eve mention", r["conversation_id"])
                    return r["conversation_id"]

        # Check vocative other human
        vocative = _vocative_other_user(text, msg.sender_id, recent, self.bot_user_id, self.bot_username)
        if vocative:
            for r in reversed(recent):
                if r.get("sender_username", "").lower() == vocative.lower() and r.get("conversation_id"):
                    logger.info("[LINKER] Tier 1: linked to vocative user's conversation=%s", r["conversation_id"])
                    return r["conversation_id"]

        # TIER 2: Structural Link
        # Same sender continuation within short window
        if recent and recent[-1]["sender_id"] == msg.sender_id and recent[-1].get("conversation_id"):
            logger.info("[LINKER] Tier 2: same sender continuation conversation=%s", recent[-1]["conversation_id"])
            return recent[-1]["conversation_id"]

        # TIER 3: Content & Interaction mapping
        active_convs = {}
        now = datetime.now(timezone.utc)
        for r in recent:
            conv_id = r.get("conversation_id")
            if not conv_id:
                continue
            try:
                ts = datetime.fromisoformat(r["timestamp"])
                if ts.tzinfo is None:
                    ts = ts.astimezone(timezone.utc)
                age = (now - ts).total_seconds()
                if age < 300:  # 5 minutes
                    active_convs[conv_id] = r
            except Exception:
                pass

        if len(active_convs) == 1:
            conv_id = list(active_convs.keys())[0]
            logger.info("[LINKER] Tier 3: linked to sole active conversation=%s", conv_id)
            return conv_id

        if len(active_convs) > 1:
            best_conv = None
            max_strength = 0.0
            for conv_id, last_msg in active_convs.items():
                strength = interaction_map.get_interaction(msg.sender_id, last_msg["sender_id"])
                if strength > max_strength:
                    max_strength = strength
                    best_conv = conv_id
            if best_conv and max_strength > 0.4:
                logger.info("[LINKER] Tier 3: linked to conversation=%s via interaction strength=%.2f", best_conv, max_strength)
                return best_conv

            # TIER 4: Gemini Disentanglement Ambiguity resolver
            candidates = list(active_convs.values())
            logger.info("[LINKER] Ambiguity detected between %d conversations. Invoking Gemini...", len(candidates))
            parent_id = self.resolve_ambiguity_via_gemini(msg, candidates)
            if parent_id:
                for c in candidates:
                    if c["message_id"] == parent_id:
                        logger.info("[LINKER] Gemini resolved parent_id=%s -> conversation=%s", parent_id, c["conversation_id"])
                        return c["conversation_id"]

        # Default: New thread ID
        new_conv_id = str(uuid.uuid4())[:8]
        logger.info("[LINKER] Started new conversation thread=%s", new_conv_id)
        return new_conv_id

    def resolve_ambiguity_via_gemini(self, msg: NormalizedMessage, candidates: List[dict]) -> Optional[str]:
        prompt = build_disentangle_prompt(msg, candidates)
        try:
            response = gemini_pool.generate_content(
                contents=prompt,
                config_opts=types.GenerateContentConfig(
                    system_instruction=DISENTANGLE_INSTRUCTION,
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=_DisentanglementSchema,
                    thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
                ),
            )
            text = (response.text or "").strip()
            if not text:
                return None
            obj = _DisentanglementSchema.model_validate_json(text)
            return obj.parent_message_id
        except Exception as e:
            logger.warning("[LINKER] Gemini ambiguity resolution failed: %s", e)
            return None


class TurnOwnershipResolver:
    """Enforces strict turn-ownership rules to prevent Eve from answering human conversations."""

    def __init__(self, bot_user_id: str, bot_username: str):
        self.bot_user_id = bot_user_id
        self.bot_username = bot_username

    def resolve_ownership(self, msg: NormalizedMessage, thread_msgs: List[dict]) -> tuple[str, Optional[str]]:
        text = (msg.text or "").strip()
        text_lower = text.lower()

        # Check Tier 1
        is_reply_to_eve = bool(
            (msg.reply_to_message_id and msg.reply_to_user_id == self.bot_user_id)
        )
        is_at_eve = bool(
            _AT_EVE.search(text) or
            _EVE_PREFIX.search(text) or
            _EVE_STANDALONE.search(text) or
            (self.bot_username and re.search(r'\b' + re.escape(self.bot_username.lower()) + r'\b', text_lower))
        )
        is_reply_to_other_human = bool(
            msg.reply_to_message_id and msg.reply_to_user_id and msg.reply_to_user_id != self.bot_user_id
        )

        at_other = False
        other_user_id = None
        at_matches = _AT_OTHER_USER.findall(text)
        for match in at_matches:
            if match.lower() not in ("eve", self.bot_username.lower() if self.bot_username else ""):
                for tm in thread_msgs:
                    if tm.get("sender_username", "").lower() == match.lower():
                        at_other = True
                        other_user_id = tm.get("sender_id")
                        break
                if at_other:
                    break

        vocative = _vocative_other_user(text, msg.sender_id, thread_msgs, self.bot_user_id, self.bot_username)
        if vocative:
            for tm in thread_msgs:
                if tm.get("sender_username", "").lower() == vocative.lower():
                    other_user_id = tm.get("sender_id")
                    break

        if is_reply_to_eve:
            logger.info("[OWNERSHIP] resolved to EVE via native reply to Eve")
            return "EVE", self.bot_user_id
        if is_at_eve:
            logger.info("[OWNERSHIP] resolved to EVE via Eve mention/address")
            return "EVE", self.bot_user_id
        if is_reply_to_other_human:
            logger.info("[OWNERSHIP] resolved to SPECIFIC_USER=%s via native reply to human", msg.reply_to_user_id)
            return "SPECIFIC_USER", msg.reply_to_user_id
        if at_other or other_user_id:
            logger.info("[OWNERSHIP] resolved to SPECIFIC_USER=%s via human mention", other_user_id)
            return "SPECIFIC_USER", other_user_id

        # Check Tier 2: Structural sequence in thread
        if thread_msgs:
            prior_msgs = [m for m in thread_msgs if m["message_id"] != msg.message_id]
            if prior_msgs:
                last_msg = prior_msgs[-1]
                # If last turn in conversation thread belonged to Eve, it's a tight continuation
                if last_msg.get("sender_id") == self.bot_user_id or last_msg.get("is_sent_by_viewer"):
                    logger.info("[OWNERSHIP] resolved to EVE via thread continuation after Eve turn")
                    return "EVE", self.bot_user_id

                # Human-human alternating exchange
                distinct_senders = {m["sender_id"] for m in prior_msgs if m["sender_id"] != self.bot_user_id}
                if len(distinct_senders) >= 2:
                    logger.info("[OWNERSHIP] resolved to SPECIFIC_USER via human-human exchange thread")
                    return "SPECIFIC_USER", last_msg["sender_id"]

        # Check Tier 3: Open Group broadcast
        is_question = text.strip().endswith("?")
        group_keywords = {"anyone", "everyone", "guys", "you guys", "who is", "bro what", "what do you", "what do u"}
        is_group_chat_broadcast = any(k in text_lower for k in group_keywords)
        
        if is_group_chat_broadcast or (is_question and not is_reply_to_other_human and not is_reply_to_eve):
            logger.info("[OWNERSHIP] resolved to OPEN_GROUP")
            return "OPEN_GROUP", None

        logger.info("[OWNERSHIP] resolved to UNCLEAR")
        return "UNCLEAR", None


class InterventionRouter:
    """Manages intervention limits and eligibility checks for OPEN_GROUP turns."""

    def __init__(self, bot_user_id: str):
        self.bot_user_id = bot_user_id

    def should_intervene(self, msg: NormalizedMessage) -> bool:
        # 1. Room Velocity: count recent messages in SQLite in the last 60s
        now = datetime.now(timezone.utc)
        cutoff = now.timestamp() - 60
        with get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM MESSAGES WHERE thread_id = ? AND timestamp > ?",
                (msg.thread_id, datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat())
            ).fetchone()
            velocity = row[0] if row else 0

        logger.info("[ROUTER] room velocity: %d messages in last 60s", velocity)

        # 2. Check recent bot frequency (last 5 minutes)
        cutoff_5m = now.timestamp() - 300
        with get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM MESSAGES WHERE thread_id = ? AND sender_id = ? AND timestamp > ?",
                (msg.thread_id, self.bot_user_id, datetime.fromtimestamp(cutoff_5m, tz=timezone.utc).isoformat())
            ).fetchone()
            bot_count_5m = row[0] if row else 0

        if bot_count_5m >= 3:
            logger.info("[ROUTER] intervention rejected: Eve already active in last 5m (count=%d)", bot_count_5m)
            return False

        # In a busy GC, Eve is MORE conservative
        if velocity > 8:
            probability = 0.05
        elif velocity > 4:
            probability = 0.15
        else:
            probability = 0.35

        # Check consecutive turns
        with get_connection() as conn:
            last = conn.execute(
                "SELECT sender_id FROM MESSAGES WHERE thread_id = ? ORDER BY timestamp DESC LIMIT 1",
                (msg.thread_id,)
            ).fetchone()
            if last and (last[0] == self.bot_user_id):
                logger.info("[ROUTER] intervention rejected: Eve sent the last message")
                return False

        chosen = random.random() < probability
        logger.info("[ROUTER] intervention check probability=%.2f -> chosen=%s", probability, chosen)
        return chosen


class PendingTurn:
    def __init__(self, turn_id: str, conversation_id: str, trigger_message_id: str, target_user_id: Optional[str], plan: Any, conversation_version: int):
        self.turn_id = turn_id
        self.conversation_id = conversation_id
        self.trigger_message_id = trigger_message_id
        self.target_user_id = target_user_id
        self.plan = plan
        self.conversation_version = conversation_version
        self.status = "PENDING"
        self.created_at = time.time()


class PendingTurnCoordinator:
    """Manages active generating turns to coordinate and cancel stale responses."""

    def __init__(self):
        self._lock = threading.Lock()
        self._pending: dict[str, PendingTurn] = {}

    def register_turn(self, turn: PendingTurn):
        with self._lock:
            # Any existing pending turns in the same conversation are superseded/cancelled
            for t in list(self._pending.values()):
                if t.conversation_id == turn.conversation_id and t.status in ("PENDING", "GENERATING", "READY"):
                    t.status = "SUPERSEDED"
                    logger.info("[PENDING] turn_id=%s superseded by turn_id=%s in conversation=%s", t.turn_id, turn.turn_id, turn.conversation_id)
            self._pending[turn.turn_id] = turn

    def get_turn(self, turn_id: str) -> Optional[PendingTurn]:
        with self._lock:
            return self._pending.get(turn_id)

    def set_status(self, turn_id: str, status: str):
        with self._lock:
            if turn_id in self._pending:
                self._pending[turn_id].status = status


class RoomStateEngine:
    """Room State Engine that coordinates linking, ownership, and intervention."""

    def __init__(self, bot_user_id: str, bot_username: str):
        self.bot_user_id = bot_user_id
        self.bot_username = bot_username
        self.interaction_map = SpeakerInteractionMap()
        self.linker = ThreadLinker(bot_user_id, bot_username)
        self.ownership_resolver = TurnOwnershipResolver(bot_user_id, bot_username)
        self.router = InterventionRouter(bot_user_id)
        self.coordinator = PendingTurnCoordinator()
        self.lock = threading.Lock()
        
        self.currently_engaged_conv_id: Optional[str] = None
        self.last_activity_time: float = time.time()

    def process_incoming_message(self, msg: NormalizedMessage, cl: Client) -> str:
        """Resolve message thread, update interactions, and return conversation ID."""
        conv_id = self.linker.link_message(msg, cl, self.interaction_map)
        msg.conversation_id = conv_id
        
        # Update SpeakerInteractionMap
        # If native reply
        if msg.reply_to_message_id:
            parent = msg_store.get_message_by_id(msg.reply_to_message_id)
            if parent:
                self.interaction_map.record_interaction(msg.sender_id, parent["sender_id"], 1.0)
        
        # Check mentions
        text = (msg.text or "").strip()
        at_matches = _AT_OTHER_USER.findall(text)
        for match in at_matches:
            # Try to resolve user ID in recent scene
            recent = msg_store.get_messages_for_thread(msg.thread_id, limit=25)
            for r in recent:
                if r.get("sender_username", "").lower() == match.lower():
                    self.interaction_map.record_interaction(msg.sender_id, r["sender_id"], 0.8)
                    break
        
        # Record general conversation participation
        recent = msg_store.get_messages_for_thread(msg.thread_id, limit=10)
        if len(recent) > 1:
            prev = recent[-1]
            self.interaction_map.record_interaction(msg.sender_id, prev["sender_id"], 0.4)

        with self.lock:
            self.last_activity_time = time.time()
            
        return conv_id
