"""
Conversation lanes — lightweight in-memory tracking of active conversational
threads within the GC.

A lane represents a currently active conversation between a set of participants.
Lanes are identified deterministically from message evidence (native replies,
turn-taking, temporal proximity) — not from Gemini calls.

Lane state is in-memory only; it does not need to survive restarts since lanes
are short-lived social phenomena (typically minutes). Persistence for message
content lives in SQLite via storage/messages.py.
"""
from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

from models.message import NormalizedMessage

logger = logging.getLogger("yap.conversation.lanes")

# Lane continuity: a gap larger than this weakens/closes a lane
LANE_IDLE_SECONDS = 300     # 5 minutes
LANE_WEAK_SECONDS = 120     # 2 minutes = decay grace period before strength erodes

# Maximum active lanes to track in memory (prevents unbounded growth)
MAX_ACTIVE_LANES = 20

# Minimum turn-taking exchanges to count as a MEDIUM-strength lane
MIN_TURNS_FOR_MEDIUM_LANE = 2

# Evidence-strength scores fed into a lane's running score on each assignment.
# "strong" = native reply chain, "medium" = same-sender turn-taking
# continuation, "weak" = general temporal-proximity GC chatter.
_EVIDENCE_SCORE = {"strong": 1.0, "medium": 0.5, "weak": 0.2}

# Score thresholds that translate the running score into the honest
# "strong"/"medium"/"weak" label read by conversation/attention.py. A lane's
# score is a blend of decayed prior evidence and each new message's evidence
# (see LaneState.record_evidence), so it can rise to "strong" OR fall back
# down to "weak" as real conversational evidence changes — it is not a
# one-way ratchet.
STRONG_SCORE_THRESHOLD = 0.75
MEDIUM_SCORE_THRESHOLD = 0.35


def _score_to_strength(score: float) -> str:
    if score >= STRONG_SCORE_THRESHOLD:
        return "strong"
    if score >= MEDIUM_SCORE_THRESHOLD:
        return "medium"
    return "weak"


@dataclass
class LaneState:
    lane_id: str
    participants: Set[str]          # user_ids (may include bot_user_id if Yap is in lane)
    message_ids: List[str]          # ordered list of message_ids in this lane
    last_activity: datetime         # timezone-aware UTC
    strength: str                   # "strong", "medium", "weak" — kept as a plain
                                     # field (not a property) so callers/tests can
                                     # still construct a LaneState with an explicit
                                     # value; LaneManager is the only writer during
                                     # normal operation, via record_evidence().
    contains_yap: bool = False      # True if Yap is an active participant
    score: Optional[float] = None   # running evidence score backing `strength`

    def __post_init__(self) -> None:
        if self.score is None:
            # Seed the score from whatever strength label was passed in, so
            # directly-constructed LaneState objects (tests, or the initial
            # creation call) stay consistent with the score model.
            self.score = _EVIDENCE_SCORE.get(self.strength, 0.2)

    def seconds_since_activity(self) -> float:
        return (datetime.now(timezone.utc) - self.last_activity).total_seconds()

    def is_stale(self) -> bool:
        return self.seconds_since_activity() > LANE_IDLE_SECONDS

    def is_yap_lane(self) -> bool:
        return self.contains_yap

    def has_participant(self, user_id: str) -> bool:
        return user_id in self.participants

    def record_evidence(self, evidence_strength: str) -> None:
        """
        Update this lane's running score (and derived strength label) from a
        new message's evidence strength ("strong"/"medium"/"weak").

        Honest, bidirectional: a strong signal can raise a weak/medium lane
        all the way to "strong"; a run of weak evidence — or simply time
        passing without reinforcement — lets a lane's score decay back down,
        so "weak" and "medium" are reachable outcomes, not just one-way
        upgrades.
        """
        gap = self.seconds_since_activity()
        if gap <= LANE_WEAK_SECONDS:
            decay = 1.0
        elif gap >= LANE_IDLE_SECONDS:
            decay = 0.0
        else:
            decay = 1.0 - (gap - LANE_WEAK_SECONDS) / (LANE_IDLE_SECONDS - LANE_WEAK_SECONDS)

        decayed_prior = self.score * decay
        evidence_score = _EVIDENCE_SCORE.get(evidence_strength, 0.2)
        # New evidence is weighted more heavily than decayed prior state so
        # a single strong signal (e.g. a native reply landing in an existing
        # lane) can reliably cross into "strong" from any starting point —
        # this is the actual bug being fixed: evidence used to only ever
        # blend to "medium" at best, capping the human-lane IGNORE fast path
        # from ever firing for organic reply chains.
        self.score = max(0.0, min(1.0, decayed_prior * 0.2 + evidence_score * 0.8))
        self.strength = _score_to_strength(self.score)


class LaneManager:
    """
    Manages conversation lanes for a single thread.
    Thread-safe: all mutations are under a lock.
    """

    def __init__(self, bot_user_id: str):
        self._bot_user_id = bot_user_id
        self._lock = threading.Lock()
        # lane_id → LaneState
        self._lanes: Dict[str, LaneState] = {}
        # user_id → lane_id (which active lane is this user currently in)
        self._user_lane: Dict[str, str] = {}

    def assign_lane(self, msg: NormalizedMessage) -> Optional[LaneState]:
        """
        Assign a message to a lane and return the lane.

        Priority:
        1. STRONG: native reply → follow reply chain to find/create lane
        2. STRONG: explicit @mention or direct address known at call time (caller passes flag)
        3. MEDIUM: turn-taking continuation with same user as last message
        4. WEAK: temporal proximity → add to most recent active lane, or create new one
        """
        with self._lock:
            self._evict_stale()

            sender = msg.sender_id
            is_yap = (sender == self._bot_user_id) or msg.is_sent_by_viewer

            # --- STRONG: native reply ---
            if msg.reply_to_message_id:
                lane = self._find_lane_containing(msg.reply_to_message_id)
                if lane:
                    self._add_to_lane(lane, msg, strength="strong")
                    if not is_yap and msg.reply_to_user_id:
                        lane.participants.add(msg.reply_to_user_id)
                    lane.participants.add(sender)
                    lane.contains_yap = lane.contains_yap or is_yap or (
                        msg.reply_to_user_id == self._bot_user_id
                    )
                    self._user_lane[sender] = lane.lane_id
                    return lane
                else:
                    # Build a new lane from the reply pair
                    participants = {sender}
                    if msg.reply_to_user_id:
                        participants.add(msg.reply_to_user_id)
                    contains_yap = (
                        self._bot_user_id in participants or is_yap
                    )
                    lane = self._create_lane(participants, msg, strength="strong",
                                            contains_yap=contains_yap)
                    return lane

            # --- MEDIUM: turn-taking continuation ---
            # If sender was recently in a lane and it's not stale, continue it
            existing_lane_id = self._user_lane.get(sender)
            if existing_lane_id and existing_lane_id in self._lanes:
                lane = self._lanes[existing_lane_id]
                if not lane.is_stale() and lane.seconds_since_activity() < LANE_WEAK_SECONDS:
                    self._add_to_lane(lane, msg, strength="medium")
                    return lane

            # --- WEAK: general GC message ---
            # Check if there is a recent active lane involving this sender's known contacts
            recent_lane = self._most_recent_active_lane_for(sender)
            if recent_lane and recent_lane.seconds_since_activity() < LANE_IDLE_SECONDS:
                self._add_to_lane(recent_lane, msg, strength="weak")
                self._user_lane[sender] = recent_lane.lane_id
                return recent_lane

            # No applicable lane — create a new singleton lane
            contains_yap = is_yap
            lane = self._create_lane({sender}, msg, strength="weak", contains_yap=contains_yap)
            return lane

    def get_lane_for_message(self, message_id: str) -> Optional[LaneState]:
        """Find which lane a given message_id belongs to."""
        with self._lock:
            return self._find_lane_containing(message_id)

    def get_yap_lane(self) -> Optional[LaneState]:
        """Return the most recent active Yap-containing lane, if any."""
        with self._lock:
            yap_lanes = [
                l for l in self._lanes.values()
                if l.contains_yap and not l.is_stale()
            ]
            if not yap_lanes:
                return None
            return max(yap_lanes, key=lambda l: l.last_activity)

    def get_lane(self, lane_id: str) -> Optional[LaneState]:
        with self._lock:
            return self._lanes.get(lane_id)

    def get_all_active_lanes(self) -> List[LaneState]:
        with self._lock:
            self._evict_stale()
            return list(self._lanes.values())

    # --- Private helpers (call under lock) ---

    def _create_lane(self, participants: Set[str], msg: NormalizedMessage,
                     strength: str, contains_yap: bool) -> LaneState:
        lane_id = str(uuid.uuid4())[:8]
        lane = LaneState(
            lane_id=lane_id,
            participants=participants,
            message_ids=[msg.message_id],
            last_activity=msg.timestamp,
            strength=strength,
            contains_yap=contains_yap,
        )
        self._lanes[lane_id] = lane
        for uid in participants:
            self._user_lane[uid] = lane_id
        logger.debug("[LANE] created lane=%s participants=%s strength=%s yap=%s",
                     lane_id, participants, strength, contains_yap)
        # Trim if over max
        if len(self._lanes) > MAX_ACTIVE_LANES:
            oldest_id = min(self._lanes, key=lambda k: self._lanes[k].last_activity)
            del self._lanes[oldest_id]
        return lane

    def _add_to_lane(self, lane: LaneState, msg: NormalizedMessage, strength: str) -> None:
        lane.record_evidence(strength)
        lane.message_ids.append(msg.message_id)
        lane.last_activity = msg.timestamp
        logger.debug("[LANE] assigned lane=%s msg=%s evidence=%s -> strength=%s score=%.2f",
                     lane.lane_id, msg.message_id, strength, lane.strength, lane.score)
        logger.info("[LANE] assigned lane=%s participants=%s",
                    lane.lane_id, lane.participants)

    def _find_lane_containing(self, message_id: str) -> Optional[LaneState]:
        """Find a lane that contains the given message_id."""
        for lane in self._lanes.values():
            if message_id in lane.message_ids:
                return lane
        return None

    def _most_recent_active_lane_for(self, sender_id: str) -> Optional[LaneState]:
        """Find the most recently active lane that this sender is in."""
        matching = [
            l for l in self._lanes.values()
            if sender_id in l.participants and not l.is_stale()
        ]
        if not matching:
            return None
        return max(matching, key=lambda l: l.last_activity)

    def _evict_stale(self) -> None:
        """Remove stale lanes from memory (call under lock)."""
        stale = [lid for lid, l in self._lanes.items() if l.is_stale()]
        for lid in stale:
            lane = self._lanes.pop(lid)
            logger.debug("[LANE] evicted stale lane=%s", lid)
            # Clean up user_lane references
            for uid in list(self._user_lane.keys()):
                if self._user_lane[uid] == lid:
                    del self._user_lane[uid]
