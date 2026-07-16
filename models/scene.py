"""
Scene Snapshot Models — V6.
Immutable value objects representing the room's dialogue state at a specific version.
These models are safe to pass to background executors / generators.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Set

from models.message import NormalizedMessage


@dataclass(frozen=True)
class DialogueSessionView:
    session_id: str
    participant_ids: Set[str]
    recent_message_ids: List[str]
    last_activity_version: int
    last_activity_timestamp: datetime
    eve_involved: bool
    confidence: str  # "HARD", "STRONG", "PROVISIONAL"
    state: str  # "ACTIVE", "FADING", "CLOSED"
    origin: str  # "NATIVE_REPLY", "EXPLICIT_ADDRESS", "CONTINUATION", "INTERACTION", "PROVISIONAL"


@dataclass(frozen=True)
class ParticipantActivityView:
    user_id: str
    last_message_id: str
    last_message_version: int
    last_message_timestamp: datetime
    recent_addressed_user_ids: List[str]
    recent_reply_target_user_ids: List[str]


@dataclass(frozen=True)
class EveEngagementView:
    active_session_id: Optional[str]
    target_user_ids: Set[str]
    last_eve_turn_id: Optional[str]
    last_eve_turn_version: Optional[int]
    engagement_strength: str  # "NONE", "LOW", "MEDIUM", "HIGH"
    last_activity_version: int
    expiry_time: Optional[float]


@dataclass(frozen=True)
class StanceStateView:
    last_stance: Optional[str]
    last_speech_act: Optional[str]
    last_intent_tag: Optional[str]
    last_target_user_id: Optional[str]
    unresolved_question_target: Optional[str]
    active_engagement_session: Optional[str]
    originating_eve_turn_id: Optional[str]
    committed_version: Optional[int]


@dataclass(frozen=True)
class SceneSnapshot:
    thread_id: str
    room_version: int
    created_at: datetime
    recent_events: List[NormalizedMessage]
    active_sessions: List[DialogueSessionView]
    participant_activity: Dict[str, ParticipantActivityView]
    reply_graph: Dict[str, str]  # message_id -> replied_to_message_id
    eve_engagement: EveEngagementView
    stance_state: StanceStateView
    burst_message_ids: List[str]
