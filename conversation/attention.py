"""
Local Attention Gate — evaluates whether Eve should reply, ignore, or
escalate to the Gemini social judge, using only local signals.

Produces one of:
- LOCAL_REPLY  : confident enough locally → no Gemini social call
- LOCAL_IGNORE : confident enough locally → no Gemini social call
- GEMINI_REQUIRED : ambiguous → forward to Gemini social judge

Score is signed: positive = reply tendency, negative = ignore tendency.
Thresholds:
  score >= REPLY_THRESHOLD  → LOCAL_REPLY
  score <= IGNORE_THRESHOLD → LOCAL_IGNORE
  otherwise                 → GEMINI_REQUIRED

Fatigue influence:
- Fatigue adjusts REPLY_THRESHOLD for GEMINI_REQUIRED cases only.
- It never blocks LOCAL_REPLY from strong positive signals.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional

from models.decision import AttentionResult
from models.message import NormalizedMessage
from conversation.lanes import LaneState

logger = logging.getLogger("yap.conversation.attention")

REPLY_THRESHOLD = 0.65
IGNORE_THRESHOLD = -0.50

# Signal weights
W_NATIVE_REPLY_TO_EVE = 1.0
W_DIRECT_EVE_START = 1.0        # message starts with "eve"
W_AT_EVE = 1.0                  # @eve anywhere in text
W_EVE_ACTIVE_LANE = 0.5         # Eve is active in same lane
W_CONTINUATION_AFTER_EVE = 0.4  # follows directly after Eve spoke

W_NATIVE_REPLY_TO_HUMAN = -0.9  # reply chain targeting another human
W_AT_OTHER_USER = -0.7          # @mention of someone else
W_VOCATIVE_OTHER_USER = -0.7    # message opens by naming another known GC member
W_ACTIVE_HUMAN_EXCHANGE = -0.6  # recent scene shows a 2-party human exchange, not Eve
W_HUMAN_HUMAN_LANE = -0.6       # active human-to-human lane, no Eve involved
W_GROUP_QUESTION = 0.25         # question to whole group
W_WHOLE_GC_MESSAGE = 0.0        # neutral — general GC chatter

# How many recent (Eve-filtered) scene messages to consider when looking for
# an active 2-party human exchange the current sender is already part of.
_ACTIVE_EXCHANGE_WINDOW = 6

# Eve name patterns. Old "yap" address is intentionally NOT recognized —
# the V5 spec requires a clean rename, not dual-identity compatibility.
_EVE_STANDALONE = re.compile(r'(?:^|[\s,!?])@?eve(?:[\s,!?]|$)', re.IGNORECASE)
_AT_EVE = re.compile(r'@eve\b', re.IGNORECASE)
_EVE_PREFIX = re.compile(r'^@?eve\b', re.IGNORECASE)
_AT_OTHER_USER = re.compile(r'@(\w+)\b', re.IGNORECASE)
_QUESTION = re.compile(r'\?')
_FIRST_WORD = re.compile(r'^@?([A-Za-z][A-Za-z0-9_.]{1,20})\b[,:]?\s')


def _is_eve_related(m: dict, bot_user_id: str, bot_username: str) -> bool:
    """True if a scene message dict is Eve's own message OR explicitly addresses Eve."""
    if m.get("is_sent_by_viewer") or (bot_user_id and m.get("sender_id") == bot_user_id):
        return True
    text = (m.get("text") or "")
    text_lower = text.lower()
    if _AT_EVE.search(text) or _EVE_PREFIX.search(text) or _EVE_STANDALONE.search(text):
        return True
    if bot_username and re.search(r'\b' + re.escape(bot_username.lower()) + r'\b', text_lower):
        return True
    return False


def _vocative_other_user(
    text: str,
    current_sender_id: str,
    scene_before: List[dict],
    bot_user_id: str,
    bot_username: str,
) -> Optional[str]:
    """
    If the message opens by naming another user who is actually a known
    recent participant in this GC (evidence-backed, not a guess against the
    dictionary), return that username. This catches plain-text vocative
    address ("ved where are u from") that @-mention regexes miss.
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


def _active_human_exchange(
    scene_before: List[dict],
    current_sender_id: str,
    bot_user_id: str,
    bot_username: str,
) -> bool:
    """
    Turn-taking evidence (independent of lanes): does the recent scene show
    the current sender already engaged in a 2-party back-and-forth with one
    other (non-Eve) human, where that other human spoke most recently?

    Filters out Eve-sent/Eve-addressed messages first so interleaved Eve
    exchanges (e.g. Rahul saying hi to Eve in between) don't break the
    thread. Requires exactly <=2 distinct senders in the filtered window
    (including the current sender) — broader group chatter with 3+ voices
    is left to the Gemini judge rather than auto-ignored.
    """
    filtered = [
        m for m in scene_before
        if not _is_eve_related(m, bot_user_id, bot_username)
    ][-_ACTIVE_EXCHANGE_WINDOW:]
    if not filtered:
        return False

    senders = [m.get("sender_id") for m in filtered]
    distinct = set(senders)
    if current_sender_id not in distinct:
        return False
    if len(distinct) > 2:
        return False
    return senders[-1] != current_sender_id


def _addressed_by_prior_human(
    scene_before: List[dict],
    sender_username: str,
    bot_user_id: str,
    bot_username: str,
) -> bool:
    """
    Covers the case _active_human_exchange's "already part of the thread"
    requirement misses: someone is vocatively named by another human ("ved
    send me that file") and then themselves speaks for the first time in the
    window ("wait"). That reply is still just answering the human who named
    them, not talking to Eve — known profile/memory about the responder must
    not override this (PART 5/6: familiarity is not permission to interrupt).
    """
    if not scene_before or not sender_username:
        return False
    last = scene_before[-1]
    if _is_eve_related(last, bot_user_id, bot_username):
        return False
    match = _FIRST_WORD.match(last.get("text") or "")
    if not match:
        return False
    return match.group(1).lower() == sender_username.lower()


def evaluate(
    msg: NormalizedMessage,
    lane: Optional[LaneState],
    bot_user_id: str,
    bot_username: str,
    fatigue_multiplier: float = 0.0,
    recent_yap_message_ids: Optional[List[str]] = None,
    recent_scene: Optional[List[dict]] = None,
) -> AttentionResult:
    """
    Evaluate whether Eve should respond to this message locally (deterministic check)
    or forward to Gemini social judge.

    recent_scene: the same canonical raw recent-GC-scene dicts used by the
    social judge (storage.messages.get_recent_scene), oldest-first. Used for
    turn-taking / active-exchange evidence (WHO IS THIS MESSAGE TALKING TO)
    independent of the lane system, since lanes only merge same-sender
    continuations or explicit native-reply chains, not organic back-and-forth
    between two different humans.
    """
    reasons: List[str] = []
    text = (msg.text or "").strip()
    text_lower = text.lower()
    recent_ids = set(recent_yap_message_ids or [])
    scene_before = [
        m for m in (recent_scene or [])
        if m.get("message_id") != msg.message_id
    ]

    # Helpers to check targeting
    is_at_eve = bool(
        _AT_EVE.search(text) or
        _EVE_PREFIX.search(text) or
        _EVE_STANDALONE.search(text) or
        (bot_username and re.search(r'\b' + re.escape(bot_username.lower()) + r'\b', text_lower))
    )

    is_reply_to_eve = bool(
        (msg.reply_to_message_id and msg.reply_to_user_id == bot_user_id) or
        (msg.reply_to_message_id and msg.reply_to_message_id in recent_ids)
    )

    is_reply_to_other_human = bool(
        msg.reply_to_message_id and msg.reply_to_user_id and msg.reply_to_user_id != bot_user_id
    )

    # Check for @mentions of other users
    at_other = False
    first_other_at = ""
    at_matches = _AT_OTHER_USER.findall(text)
    for match in at_matches:
        if match.lower() not in ("eve", bot_username.lower() if bot_username else ""):
            at_other = True
            first_other_at = match
            break

    # Plain-text vocative address ("ved where are u from") — @-mention
    # regexes miss this, but it's strong away-from-Eve evidence when the
    # named user is an actual known recent GC participant.
    vocative_other = _vocative_other_user(text, msg.sender_id, scene_before, bot_user_id, bot_username)

    # Turn-taking evidence from the raw recent scene, independent of lanes.
    is_continuation_after_eve = bool(scene_before) and _is_eve_related(scene_before[-1], bot_user_id, bot_username)
    is_active_human_exchange = (
        _active_human_exchange(scene_before, msg.sender_id, bot_user_id, bot_username)
        or _addressed_by_prior_human(scene_before, msg.sender_username, bot_user_id, bot_username)
    )

    # Lane checks (supporting evidence — see conversation/lanes.py)
    is_strong_eve_lane = bool(
        lane is not None and lane.is_yap_lane() and lane.strength == "strong"
    )

    is_strong_human_lane = bool(
        lane is not None and not lane.is_yap_lane() and len(lane.participants) >= 2 and lane.strength == "strong"
    )

    # === DETERMINISTIC LOCAL_REPLY (strongest evidence first) ===
    if is_reply_to_eve:
        reasons.append("native_reply_to_eve")
        return AttentionResult(decision="LOCAL_REPLY", score=1.0, reasons=reasons)

    if is_at_eve:
        if _AT_EVE.search(text):
            reasons.append("at_eve_mention")
        elif _EVE_PREFIX.search(text):
            reasons.append("eve_prefix")
        elif _EVE_STANDALONE.search(text):
            reasons.append("eve_standalone")
        else:
            reasons.append("bot_username_mention")
        return AttentionResult(decision="LOCAL_REPLY", score=1.0, reasons=reasons)

    # === DETERMINISTIC LOCAL_IGNORE (explicit evidence away from Eve) ===
    if is_reply_to_other_human and not is_at_eve:
        reasons.append("native_reply_to_human")
        return AttentionResult(decision="LOCAL_IGNORE", score=-1.0, reasons=reasons)

    if at_other and not is_at_eve:
        reasons.append(f"at_other_user:{first_other_at}")
        return AttentionResult(decision="LOCAL_IGNORE", score=-1.0, reasons=reasons)

    if vocative_other and not is_at_eve:
        reasons.append(f"vocative_other_user:{vocative_other}")
        return AttentionResult(decision="LOCAL_IGNORE", score=W_VOCATIVE_OTHER_USER, reasons=reasons)

    # === TURN-TAKING EVIDENCE (recent scene, no Gemini call) ===
    # Checked in this order deliberately: an established human-to-human
    # thread the current sender is already part of is more specific evidence
    # than "Eve's message happens to be the last one in the scene" — e.g. an
    # interleaved "Rahul: hii eve / Eve: hii" shouldn't make Eve think she
    # owns the very next Atharv->Ved continuation just because her message
    # is chronologically last.
    if is_active_human_exchange and not is_at_eve:
        reasons.append("active_human_exchange")
        return AttentionResult(decision="LOCAL_IGNORE", score=W_ACTIVE_HUMAN_EXCHANGE, reasons=reasons)

    if is_continuation_after_eve and not (at_other or vocative_other):
        reasons.append("continuation_after_eve")
        return AttentionResult(decision="LOCAL_REPLY", score=W_CONTINUATION_AFTER_EVE, reasons=reasons)

    # === LANE SUPPORTING EVIDENCE ===
    if is_strong_eve_lane:
        reasons.append("eve_active_lane")
        return AttentionResult(decision="LOCAL_REPLY", score=1.0, reasons=reasons)

    if is_strong_human_lane and not is_at_eve:
        reasons.append("human_lane")
        return AttentionResult(decision="LOCAL_IGNORE", score=-1.0, reasons=reasons)

    # === GEMINI_REQUIRED ===
    reasons.append("ambiguous_targeting")
    return AttentionResult(decision="GEMINI_REQUIRED", score=0.0, reasons=reasons)
