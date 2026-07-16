"""
Context Selector Subsystem — V6.
Selects and formats social context into a compact GenerationPacket for Gemini.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import List, Optional, Set

from models.message import NormalizedMessage
from models.scene import SceneSnapshot
from models.generation_packet import (
    GenerationPacket,
    EveIdentity,
    TurnContract,
    TargetProfile,
    DialogueSessionContext,
    RelevantMemory,
    EveContinuity,
    SocialSituation,
)
from storage import profiles as prof_store
from storage import memories as mem_store
from storage import eve_turns

logger = logging.getLogger("yap.intelligence.context_selector")

STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "of", "to", "in", "on", 
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "from", "up", "down", "in",
    "out", "on", "off", "over", "under", "again", "further", "then", "once",
    "hi", "hello", "hey", "bro", "bruh", "bhai", "yaar"
}


class ContextSelector:
    """Selects and packages relevant structural, memory, and continuity data for a turn opportunity."""

    def __init__(self, bot_user_id: str, bot_username: str, core_instruction: str):
        self.bot_user_id = str(bot_user_id)
        self.bot_username = str(bot_username)
        self.core_instruction = str(core_instruction)

    def _extract_topics(self, text: str) -> List[str]:
        """Extract alphanumeric words from trigger text for memory subject matching."""
        if not text:
            return []
        words = re.findall(r"\b\w{3,15}\b", text.lower())
        return [w for w in words if w not in STOP_WORDS]

    def select(self, snapshot: SceneSnapshot, opportunity: TurnOpportunity) -> GenerationPacket:
        """Construct the compact, clean, and interpreted social GenerationPacket."""
        # 1. Identity View
        identity = EveIdentity(
            bot_user_id=self.bot_user_id,
            bot_username=self.bot_username,
            core_instruction=self.core_instruction,
        )

        # 2. Turn Contract
        requires_reply = opportunity.address_resolution.evidence.get("native_reply_to_eve", False)
        contract = TurnContract(
            participation_mode=opportunity.participation_mode,
            target_user_id=opportunity.target_user_id,
            anchor_message_id=opportunity.anchor_message_id,
            session_id=opportunity.session_id,
            requires_native_reply=requires_reply,
            expected_response_length="SHORT",
        )

        # 3. Target Profile lookup
        target_uid = opportunity.target_user_id
        if not target_uid or target_uid == self.bot_user_id:
            # Fallback to the sender of the trigger message
            trigger = next((m for m in snapshot.recent_events if m.message_id == opportunity.anchor_message_id), None)
            if not trigger and snapshot.recent_events:
                trigger = snapshot.recent_events[-1]
            target_uid = trigger.sender_id if trigger else "unknown"

        # Resolve display name / username
        trigger_msg = next((m for m in snapshot.recent_events if m.message_id == opportunity.anchor_message_id), None)
        target_username = trigger_msg.sender_username if trigger_msg else "unknown"
        
        # Lazy create profile if we have a valid ID
        prof_dict, _ = prof_store.get_or_create_user(target_uid, target_username)
        
        fam_score = prof_dict.get("familiarity_score", 0.0)
        if fam_score >= 0.7:
            fam_band = "CLOSE"
        elif fam_score >= 0.4:
            fam_band = "FRIEND"
        elif fam_score >= 0.15:
            fam_band = "AQUAINTANCE"
        else:
            fam_band = "STRANGER"

        target_profile = TargetProfile(
            user_id=target_uid,
            username=target_username,
            display_name=prof_dict.get("preferred_name") or target_username,
            familiarity_band=fam_band,
            relationship_tier=int(fam_score * 10),
            detected_style={"detected_languages": prof_dict.get("detected_languages")} if prof_dict.get("detected_languages") else {},
        )

        # 4. Active Dialogue Session Context
        active_session_ctx = None
        if opportunity.session_id:
            session_view = next((s for s in snapshot.active_sessions if s.session_id == opportunity.session_id), None)
            if session_view:
                session_msgs = [
                    m for m in snapshot.recent_events 
                    if m.message_id in session_view.recent_message_ids
                ]
                
                # Retrieve participant usernames
                usernames = []
                for p_id in session_view.participant_ids:
                    # Look up username in events
                    p_uname = next((m.sender_username for m in snapshot.recent_events if m.sender_id == p_id), p_id)
                    usernames.append(p_uname)

                last_stance = None
                last_speech = None
                if snapshot.stance_state.active_engagement_session == opportunity.session_id:
                    last_stance = snapshot.stance_state.last_stance
                    last_speech = snapshot.stance_state.last_speech_act

                active_session_ctx = DialogueSessionContext(
                    session_id=opportunity.session_id,
                    participant_usernames=usernames,
                    recent_messages=session_msgs,
                    eve_involved=session_view.eve_involved,
                    last_committed_stance=last_stance,
                    last_speech_act=last_speech,
                )

        # 5. Memories (Relevance-first selection)
        memories = []
        if trigger_msg and trigger_msg.text:
            topics = self._extract_topics(trigger_msg.text)
            if topics:
                all_active = mem_store.get_active_memories(target_uid, limit=50)
                matched = []
                for m in all_active:
                    norm = m.get("normalized_fact", "").lower()
                    if any(t in norm for t in topics):
                        matched.append(m)
                
                for m in matched[:3]:
                    memories.append(
                        RelevantMemory(
                            slot=m.get("slot", "general"),
                            value=m.get("value"),
                            memory_type=m.get("memory_type", "personal_fact"),
                            relevance_reason="Entity/Subject match"
                        )
                    )

        # 6. Eve Continuity
        recent_turns_list = eve_turns.get_recent_eve_turns_for_thread(snapshot.thread_id, limit=3)
        continuity = EveContinuity(
            last_stance=snapshot.stance_state.last_stance,
            last_speech_act=snapshot.stance_state.last_speech_act,
            last_intent_tag=snapshot.stance_state.last_intent_tag,
            recent_eve_turns=recent_turns_list,
        )

        # 7. Recent Room Scene (Max 15 messages for a clean window view)
        recent_scene = snapshot.recent_events[-15:]

        # 8. Social Situation Interpretation
        is_involved = active_session_ctx.eve_involved if active_session_ctx else False
        p_names = active_session_ctx.participant_usernames if active_session_ctx else [target_username]
        
        summary = ""
        if opportunity.participation_mode == "REQUIRED":
            summary = f"Direct interaction target to Eve. Sender {target_username} is explicitly addressing Eve."
        elif is_involved:
            summary = f"Continuing ongoing interaction involving Eve with active participants {', '.join(p_names)}."
        else:
            summary = f"Eligible open group moment. Active human chat includes {', '.join(p_names)}."

        situation = SocialSituation(
            summary=summary,
            is_eve_involved=is_involved,
            target_name=target_profile.display_name,
            active_participants=p_names,
        )

        return GenerationPacket(
            identity=identity,
            contract=contract,
            situation=situation,
            target=target_profile,
            active_session=active_session_ctx,
            recent_room_scene=recent_scene,
            memories=memories,
            continuity=continuity,
        )
