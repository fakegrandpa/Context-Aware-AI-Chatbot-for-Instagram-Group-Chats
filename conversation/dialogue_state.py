"""
Dialogue State Subsystem — V6.
Maintains lightweight, ephemeral live room state for one chat thread.
This module is strictly structural and performs no cognitive reasoning or turn-policy checks.
"""
from __future__ import annotations

import logging
import re
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

from models.message import NormalizedMessage
from models.scene import (
    DialogueSessionView,
    EveEngagementView,
    ParticipantActivityView,
    SceneSnapshot,
    StanceStateView,
)

logger = logging.getLogger("yap.conversation.dialogue_state")

# Simple confidence enums
CONF_HARD = "HARD"
CONF_STRONG = "STRONG"
CONF_PROVISIONAL = "PROVISIONAL"

# Session states
STATE_ACTIVE = "ACTIVE"
STATE_FADING = "FADING"
STATE_CLOSED = "CLOSED"

# Session origins
ORIGIN_NATIVE_REPLY = "NATIVE_REPLY"
ORIGIN_EXPLICIT_ADDRESS = "EXPLICIT_ADDRESS"
ORIGIN_CONTINUATION = "CONTINUATION"
ORIGIN_INTERACTION = "INTERACTION"
ORIGIN_PROVISIONAL = "PROVISIONAL"

# Excluded greetings for address parsing (not to be treated as explicit addresses)
EXCLUDED_GREETINGS = {
    "hi", "hello", "hey", "hii", "helloo", "heyy", "bro", "bruh", "oye",
    "bhai", "bhaiya", "bhe", "yaar", "yaara", "siri", "alexa", "google"
}


class DialogueSession:
    """Represents a lightweight structural interaction grouping (not a topic model)."""

    def __init__(
        self,
        session_id: str,
        origin: str,
        confidence: str,
        initial_msg_id: str,
        participant_ids: Set[str],
        start_version: int,
    ):
        self.session_id = session_id
        self.origin = origin
        self.confidence = confidence
        self.recent_message_ids = [initial_msg_id]
        self.participant_ids = set(participant_ids)
        self.last_activity_version = start_version
        self.last_activity_timestamp = datetime.now(timezone.utc)
        self.eve_involved = False
        self.state = STATE_ACTIVE

    def touch(self, message_id: str, sender_id: str, version: int, timestamp: datetime, eve_involved: bool = False):
        """Update session activity tracking."""
        if message_id not in self.recent_message_ids:
            self.recent_message_ids.append(message_id)
        self.participant_ids.add(sender_id)
        self.last_activity_version = version
        self.last_activity_timestamp = timestamp
        if eve_involved:
            self.eve_involved = True

    def to_view(self) -> DialogueSessionView:
        """Create an immutable copy of the session state."""
        return DialogueSessionView(
            session_id=self.session_id,
            participant_ids=set(self.participant_ids),
            recent_message_ids=list(self.recent_message_ids),
            last_activity_version=self.last_activity_version,
            last_activity_timestamp=self.last_activity_timestamp,
            eve_involved=self.eve_involved,
            confidence=self.confidence,
            state=self.state,
            origin=self.origin,
        )


class ParticipantActivity:
    """Tracks bounded interaction recency of a specific user."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.last_message_id = ""
        self.last_message_version = 0
        self.last_message_timestamp = datetime.min.replace(tzinfo=timezone.utc)
        self.recent_addressed_user_ids: List[str] = []
        self.recent_reply_target_user_ids: List[str] = []

    def record_activity(
        self,
        message_id: str,
        version: int,
        timestamp: datetime,
        target_id: Optional[str] = None,
        reply_target_id: Optional[str] = None,
    ):
        self.last_message_id = message_id
        self.last_message_version = version
        self.last_message_timestamp = timestamp
        
        if target_id:
            if target_id in self.recent_addressed_user_ids:
                self.recent_addressed_user_ids.remove(target_id)
            self.recent_addressed_user_ids.append(target_id)
            if len(self.recent_addressed_user_ids) > 5:
                self.recent_addressed_user_ids.pop(0)

        if reply_target_id:
            if reply_target_id in self.recent_reply_target_user_ids:
                self.recent_reply_target_user_ids.remove(reply_target_id)
            self.recent_reply_target_user_ids.append(reply_target_id)
            if len(self.recent_reply_target_user_ids) > 5:
                self.recent_reply_target_user_ids.pop(0)

    def to_view(self) -> ParticipantActivityView:
        return ParticipantActivityView(
            user_id=self.user_id,
            last_message_id=self.last_message_id,
            last_message_version=self.last_message_version,
            last_message_timestamp=self.last_message_timestamp,
            recent_addressed_user_ids=list(self.recent_addressed_user_ids),
            recent_reply_target_user_ids=list(self.recent_reply_target_user_ids),
        )


class EveEngagement:
    """Basic per-chat structural involvement metrics."""

    def __init__(self):
        self.active_session_id: Optional[str] = None
        self.target_user_ids: Set[str] = set()
        self.last_eve_turn_id: Optional[str] = None
        self.last_eve_turn_version: Optional[int] = None
        self.engagement_strength: str = "NONE"  # "NONE", "LOW", "MEDIUM", "HIGH"
        self.last_activity_version: int = 0
        self.expiry_time: Optional[float] = None

    def update_engagement(
        self,
        session_id: Optional[str],
        target_ids: Set[str],
        turn_id: str,
        version: int,
        strength: str = "HIGH",
    ):
        self.active_session_id = session_id
        self.target_user_ids = set(target_ids)
        self.last_eve_turn_id = turn_id
        self.last_eve_turn_version = version
        self.engagement_strength = strength
        self.last_activity_version = version

    def to_view(self) -> EveEngagementView:
        return EveEngagementView(
            active_session_id=self.active_session_id,
            target_user_ids=set(self.target_user_ids),
            last_eve_turn_id=self.last_eve_turn_id,
            last_eve_turn_version=self.last_eve_turn_version,
            engagement_strength=self.engagement_strength,
            last_activity_version=self.last_activity_version,
            expiry_time=self.expiry_time,
        )


class StanceState:
    """Tracks committed conversational continuity (stance, speech act, targets)."""
    def __init__(self):
        self.last_stance: Optional[str] = None
        self.last_speech_act: Optional[str] = None
        self.last_intent_tag: Optional[str] = None
        self.last_target_user_id: Optional[str] = None
        self.unresolved_question_target: Optional[str] = None
        self.active_engagement_session: Optional[str] = None
        self.originating_eve_turn_id: Optional[str] = None
        self.committed_version: Optional[int] = None

    def commit(
        self,
        stance: Optional[str],
        speech_act: Optional[str],
        intent_tag: Optional[str],
        target_user_id: Optional[str],
        session_id: Optional[str],
        turn_id: str,
        version: int,
    ):
        self.last_stance = stance
        self.last_speech_act = speech_act
        self.last_intent_tag = intent_tag
        self.last_target_user_id = target_user_id
        self.active_engagement_session = session_id
        self.originating_eve_turn_id = turn_id
        self.committed_version = version
        
        if speech_act == "QUESTION" or (intent_tag and "question" in intent_tag.lower()):
            self.unresolved_question_target = target_user_id
        else:
            self.unresolved_question_target = None

    def to_view(self) -> StanceStateView:
        return StanceStateView(
            last_stance=self.last_stance,
            last_speech_act=self.last_speech_act,
            last_intent_tag=self.last_intent_tag,
            last_target_user_id=self.last_target_user_id,
            unresolved_question_target=self.unresolved_question_target,
            active_engagement_session=self.active_engagement_session,
            originating_eve_turn_id=self.originating_eve_turn_id,
            committed_version=self.committed_version,
        )


class DialogueState:
    """Manages structural room timeline, session linking, and participant recency."""

    def __init__(self, thread_id: str, bot_user_id: str, bot_username: str):
        self.thread_id = str(thread_id)
        self.bot_user_id = str(bot_user_id)
        self.bot_username = str(bot_username)
        
        self.room_version = 0
        self.recent_events: deque[NormalizedMessage] = deque(maxlen=100)
        self.active_sessions: Dict[str, DialogueSession] = {}
        self.participant_activity: Dict[str, ParticipantActivity] = {}
        self.reply_graph: Dict[str, str] = {}  # message_id -> replied_to_message_id
        self.eve_engagement = EveEngagement()
        self.stance_state = StanceState()

    def get_session_for_message(self, message_id: str) -> Optional[str]:
        """Scans recent events to find a linked session ID."""
        for m in self.recent_events:
            if m.message_id == message_id:
                return m.conversation_id
        return None

    def get_message_by_id(self, message_id: str) -> Optional[NormalizedMessage]:
        for m in self.recent_events:
            if m.message_id == message_id:
                return m
        return None

    def _resolve_explicit_target(self, text: str, sender_username: str) -> Optional[str]:
        """
        Conservative vocative/mention parser.
        Detects explicit mentions that resolve to known participants.
        Excludes broad generic greetings (bro, oye).
        """
        if not text:
            return None
        text_clean = re.sub(r'[^\w\s@_.]', ' ', text).strip()
        words = text_clean.split()
        if not words:
            return None

        # Check @mentions first
        for w in words:
            if w.startswith("@") and len(w) > 1:
                target_uname = w[1:].lower()
                if target_uname in ("eve", self.bot_username.lower()):
                    return self.bot_user_id
                # Match against known usernames in recent history
                for m in self.recent_events:
                    if m.sender_username and m.sender_username.lower() == target_uname:
                        return m.sender_id

        # Check conservative first-word vocative
        first_word = words[0].lower()
        if first_word in EXCLUDED_GREETINGS:
            return None

        if first_word in ("eve", self.bot_username.lower()):
            return self.bot_user_id

        for m in self.recent_events:
            if m.sender_username and m.sender_username.lower() == first_word:
                if m.sender_id != self.bot_user_id:
                    return m.sender_id

        return None

    def apply_message(self, msg: NormalizedMessage, current_version: int):
        """
        Applies a message to the DialogueState.
        Increments internal version and mutates session/participant state.
        This must be called ONLY inside the actor mailbox sequence.
        """
        self.room_version = current_version
        
        # Ensure message has conversation_id reference updated
        msg.conversation_id = None
        self.recent_events.append(msg)

        # 1. Update Reply Graph
        if msg.reply_to_message_id:
            self.reply_graph[msg.message_id] = msg.reply_to_message_id

        # 2. Resolve target target_user_id if any (mentions/vocatives)
        target_user_id = self._resolve_explicit_target(msg.text, msg.sender_username)
        
        # 3. Dialogue Session Affiliation Rules
        session_id: Optional[str] = None
        origin: Optional[str] = None
        confidence: Optional[str] = None

        is_bot_involved = (msg.sender_id == self.bot_user_id or target_user_id == self.bot_user_id)
        if not is_bot_involved and msg.reply_to_message_id:
            if msg.reply_to_user_id == self.bot_user_id:
                is_bot_involved = True
            else:
                parent = self.get_message_by_id(msg.reply_to_message_id)
                if parent and parent.sender_id == self.bot_user_id:
                    is_bot_involved = True

        # Rule A: Native Reply Edge (Hard Evidence)
        if msg.reply_to_message_id:
            parent_session_id = self.get_session_for_message(msg.reply_to_message_id)
            if parent_session_id and parent_session_id in self.active_sessions:
                session = self.active_sessions[parent_session_id]
                if session.state != STATE_CLOSED:
                    session_id = parent_session_id
                    session.touch(msg.message_id, msg.sender_id, current_version, msg.timestamp, eve_involved=is_bot_involved)
                    origin = ORIGIN_NATIVE_REPLY
                    confidence = CONF_HARD
            
            if not session_id:
                # Parent has no active session, create a native reply session
                parent_msg = self.get_message_by_id(msg.reply_to_message_id)
                parent_sender = parent_msg.sender_id if parent_msg else (msg.reply_to_user_id or "unknown")
                session_id = str(uuid.uuid4())[:8]
                new_session = DialogueSession(
                    session_id=session_id,
                    origin=ORIGIN_NATIVE_REPLY,
                    confidence=CONF_HARD,
                    initial_msg_id=msg.reply_to_message_id,
                    participant_ids={parent_sender, msg.sender_id},
                    start_version=current_version,
                )
                new_session.touch(msg.message_id, msg.sender_id, current_version, msg.timestamp, eve_involved=is_bot_involved)
                self.active_sessions[session_id] = new_session
                origin = ORIGIN_NATIVE_REPLY
                confidence = CONF_HARD

        # Rule B: Explicit Known User Address
        if not session_id and target_user_id:
            # Search active sessions involving the sender and the target T within last 5 versions
            for s_id, s in self.active_sessions.items():
                if s.state != STATE_CLOSED and msg.sender_id in s.participant_ids and target_user_id in s.participant_ids:
                    if current_version - s.last_activity_version <= 5:
                        session_id = s_id
                        s.touch(msg.message_id, msg.sender_id, current_version, msg.timestamp, eve_involved=is_bot_involved)
                        origin = ORIGIN_EXPLICIT_ADDRESS
                        confidence = CONF_STRONG
                        break
            # Note: Do not auto-start a new STRONG session for isolated explicit mentions to keep links conservative

        # Rule C: Sender Continuation
        if not session_id:
            # Look for a session this sender was active in recently (last 2 minutes or 5 versions)
            best_session_id = None
            min_version_diff = 99999
            for s_id, s in self.active_sessions.items():
                if s.state != STATE_CLOSED and msg.sender_id in s.participant_ids:
                    v_diff = current_version - s.last_activity_version
                    if v_diff <= 5 and v_diff < min_version_diff:
                        # Check timestamp bounds (2 minutes)
                        time_diff = (msg.timestamp - s.last_activity_timestamp).total_seconds()
                        if time_diff <= 120:
                            best_session_id = s_id
                            min_version_diff = v_diff

            if best_session_id:
                session_id = best_session_id
                self.active_sessions[session_id].touch(
                    msg.message_id, msg.sender_id, current_version, msg.timestamp, eve_involved=is_bot_involved
                )
                origin = ORIGIN_CONTINUATION
                confidence = CONF_PROVISIONAL

        # Rule D: Interaction Continuity (fallback check)
        if not session_id:
            # Check if sender has active interactions in any session
            for s_id, s in self.active_sessions.items():
                if s.state != STATE_CLOSED:
                    # Does this session include users this sender recently replied to/addressed?
                    sender_act = self.participant_activity.get(msg.sender_id)
                    if sender_act:
                        interacted = set(sender_act.recent_reply_target_user_ids) | set(sender_act.recent_addressed_user_ids)
                        if s.participant_ids & interacted:
                            if current_version - s.last_activity_version <= 5:
                                session_id = s_id
                                s.touch(msg.message_id, msg.sender_id, current_version, msg.timestamp, eve_involved=is_bot_involved)
                                origin = ORIGIN_INTERACTION
                                confidence = CONF_PROVISIONAL
                                break

        # Rule E: Unknown / Unaffiliated
        if not session_id:
            # Leave unaffiliated
            msg.conversation_id = None
        else:
            msg.conversation_id = session_id

        # 4. Decay / Close old sessions
        for s_id in list(self.active_sessions.keys()):
            session = self.active_sessions[s_id]
            ver_diff = current_version - session.last_activity_version
            time_diff = (msg.timestamp - session.last_activity_timestamp).total_seconds()
            
            if ver_diff > 30 or time_diff > 900:  # 30 versions or 15 mins
                session.state = STATE_CLOSED
                self.active_sessions.pop(s_id)
            elif ver_diff > 10 or time_diff > 300:  # 10 versions or 5 mins
                session.state = STATE_FADING

        # 5. Record Participant Activity
        if msg.sender_id not in self.participant_activity:
            self.participant_activity[msg.sender_id] = ParticipantActivity(msg.sender_id)
        
        reply_target_uid = msg.reply_to_user_id
        if not reply_target_uid and msg.reply_to_message_id:
            parent_msg = self.get_message_by_id(msg.reply_to_message_id)
            if parent_msg:
                reply_target_uid = parent_msg.sender_id
                
        self.participant_activity[msg.sender_id].record_activity(
            message_id=msg.message_id,
            version=current_version,
            timestamp=msg.timestamp,
            target_id=target_user_id,
            reply_target_id=reply_target_uid,
        )

        # 6. Basic Eve Engagement Updates for Inbound Viewer Messages
        if msg.is_sent_by_viewer or msg.sender_id == self.bot_user_id:
            targets = {target_user_id} if target_user_id else set()
            if reply_target_uid:
                targets.add(reply_target_uid)
            self.eve_engagement.update_engagement(
                session_id=session_id,
                target_ids=targets,
                turn_id=msg.message_id,
                version=current_version,
                strength="HIGH",
            )

    def create_snapshot(self, burst_message_ids: List[str]) -> SceneSnapshot:
        """Generate a completely detached, immutable view of current room state."""
        recent_events_copy = [
            NormalizedMessage(
                message_id=m.message_id,
                thread_id=m.thread_id,
                sender_id=m.sender_id,
                sender_username=m.sender_username,
                text=m.text,
                timestamp=m.timestamp,
                item_type=m.item_type,
                is_sent_by_viewer=m.is_sent_by_viewer,
                reply_to_message_id=m.reply_to_message_id,
                reply_to_user_id=m.reply_to_user_id,
                is_historical=m.is_historical,
                raw_dm=m.raw_dm,
                conversation_id=m.conversation_id,
            )
            for m in self.recent_events
        ]
        
        session_views = [s.to_view() for s in self.active_sessions.values()]
        
        activity_views = {
            uid: act.to_view() for uid, act in self.participant_activity.items()
        }
        
        return SceneSnapshot(
            thread_id=self.thread_id,
            room_version=self.room_version,
            created_at=datetime.now(timezone.utc),
            recent_events=recent_events_copy,
            active_sessions=session_views,
            participant_activity=activity_views,
            reply_graph=dict(self.reply_graph),
            eve_engagement=self.eve_engagement.to_view(),
            stance_state=self.stance_state.to_view(),
            burst_message_ids=list(burst_message_ids),
        )
