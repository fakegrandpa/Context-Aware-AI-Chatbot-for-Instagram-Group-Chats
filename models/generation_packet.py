"""
Generation Packet Schema — V6.
Immutable dataclasses containing the interpreted social situation for Gemini 3.1 Flash Lite.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Set, Dict

from models.message import NormalizedMessage


@dataclass(frozen=True)
class EveIdentity:
    bot_user_id: str
    bot_username: str
    core_instruction: str


@dataclass(frozen=True)
class TurnContract:
    participation_mode: str  # REQUIRED, ELIGIBLE
    target_user_id: Optional[str]
    anchor_message_id: str
    session_id: Optional[str]
    requires_native_reply: bool
    expected_response_length: str  # "SHORT", "MEDIUM", "LONG"


@dataclass(frozen=True)
class TargetProfile:
    user_id: str
    username: str
    display_name: str
    familiarity_band: str  # "STRANGER", "AQUAINTANCE", "FRIEND", "CLOSE"
    relationship_tier: int
    detected_style: Dict[str, any] = field(default_factory=dict)


@dataclass(frozen=True)
class DialogueSessionContext:
    session_id: str
    participant_usernames: List[str]
    recent_messages: List[NormalizedMessage]
    eve_involved: bool
    last_committed_stance: Optional[str]
    last_speech_act: Optional[str]


@dataclass(frozen=True)
class RelevantMemory:
    slot: str
    value: str
    memory_type: str
    relevance_reason: str


@dataclass(frozen=True)
class EveContinuity:
    last_stance: Optional[str]
    last_speech_act: Optional[str]
    last_intent_tag: Optional[str]
    recent_eve_turns: List[dict] = field(default_factory=list)


@dataclass(frozen=True)
class SocialSituation:
    summary: str
    is_eve_involved: bool
    target_name: str
    active_participants: List[str]


@dataclass(frozen=True)
class GenerationPacket:
    identity: EveIdentity
    contract: TurnContract
    situation: SocialSituation
    target: TargetProfile
    active_session: Optional[DialogueSessionContext]
    recent_room_scene: List[NormalizedMessage]
    memories: List[RelevantMemory]
    continuity: EveContinuity
