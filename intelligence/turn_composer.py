"""
Turn Composer Subsystem — V6.
Performs a single-call structured generation using Gemini 3.1 Flash Lite.
Returns a TurnProposal.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Optional

from google.genai import types
from pydantic import BaseModel, Field

from models.generation_packet import GenerationPacket
from intelligence import gemini_pool

logger = logging.getLogger("yap.intelligence.turn_composer")


class TurnProposalModel(BaseModel):
    action: str = Field(description="Must be either 'REPLY' or 'IGNORE'")
    target_user_id: Optional[str] = Field(description="Stable user ID of the human participant being addressed")
    anchor_message_id: str = Field(description="Message ID of the trigger message being responded to")
    speech_act: str = Field(description="Speech act tag, e.g. GREET, STATEMENT, QUESTION, JOKE, AGREE, DISAGREE, REJECT, DISMISS")
    intent_tag: str = Field(description="Brief developer-oriented intent tag, e.g. greet_user, mock_cat_preference")
    stance: str = Field(description="Eve's conversational stance/vibe, e.g. tired, dismissive, playful, friendly, sarcastic")
    reply_text: Optional[str] = Field(description="The actual text reply Eve says. Must be null/empty if action is IGNORE")
    continuity_marker: Optional[str] = Field(description="Optional brief callback continuity reference")


@dataclass(frozen=True)
class TurnProposal:
    action: str  # REPLY, IGNORE
    target_user_id: Optional[str]
    anchor_message_id: str
    speech_act: str
    intent_tag: str
    stance: str
    reply_text: Optional[str]
    continuity_marker: Optional[str] = None


class TurnComposer:
    """Invokes Gemini 3.1 Flash Lite to generate a TurnProposal from a GenerationPacket."""

    def __init__(self, bot_user_id: str):
        self.bot_user_id = str(bot_user_id)

    def format_prompt(self, packet: GenerationPacket) -> str:
        """Render the structured packet context for the Gemini model."""
        parts = []
        
        # 1. Social Situation
        parts.append("### CURRENT SOCIAL SITUATION")
        parts.append(f"Summary: {packet.situation.summary}")
        parts.append(f"Target Participant: {packet.target.display_name} (Familiarity: {packet.target.familiarity_band})")
        parts.append(f"Active Participants: {', '.join(packet.situation.active_participants)}")
        parts.append("")
        
        # 2. Turn Contract
        parts.append("### TURN CONTRACT")
        parts.append(f"Participation Mode: {packet.contract.participation_mode}")
        parts.append(f"Target User ID: {packet.contract.target_user_id}")
        parts.append(f"Anchor Message ID: {packet.contract.anchor_message_id}")
        parts.append(f"Requires Direct Native Reply: {packet.contract.requires_native_reply}")
        parts.append(f"Expected Reply Length: {packet.contract.expected_response_length}")
        parts.append("")

        # 3. Active Session Context
        if packet.active_session:
            parts.append("### ACTIVE INTERACTION HISTORY")
            for m in packet.active_session.recent_messages:
                sender = "Eve (You)" if (m.sender_id == self.bot_user_id or m.is_sent_by_viewer) else m.sender_username
                parts.append(f"- {sender}: {m.text}")
            if packet.active_session.last_committed_stance:
                parts.append(f"Eve's Previous Stance: {packet.active_session.last_committed_stance}")
                parts.append(f"Eve's Previous Speech Act: {packet.active_session.last_speech_act}")
            parts.append("")
            
        # 4. Recent Room Scene
        parts.append("### RECENT ROOM TIMELINE (ordered recent GC history)")
        for m in packet.recent_room_scene:
            sender = "Eve (You)" if (m.sender_id == self.bot_user_id or m.is_sent_by_viewer) else m.sender_username
            parts.append(f"- {sender}: {m.text}")
        parts.append("")

        # 5. Relevant Memories
        if packet.memories:
            parts.append("### RELEVANT FACTS ESTABLISHED PREVIOUSLY")
            for m in packet.memories:
                parts.append(f"- {m.value} (Type: {m.memory_type})")
            parts.append("")

        # 6. Continuity
        if packet.continuity.last_stance:
            parts.append("### EVE'S RECENT CONTINUITY STATE")
            parts.append(f"Last Stance: {packet.continuity.last_stance}")
            parts.append(f"Last Speech Act: {packet.continuity.last_speech_act}")
            parts.append(f"Last Intent Tag: {packet.continuity.last_intent_tag}")
            parts.append("")

        parts.append("Evaluate this room moment. Decide whether to REPLY or IGNORE.")
        parts.append("If Participation Mode is REQUIRED, you MUST set action to REPLY.")
        parts.append("Reply naturally, like a human participant in a group chat. Keep the reply extremely short and conversational (no AI explanations or thoughts).")
        
        return "\n".join(parts)

    def compose(self, packet: GenerationPacket) -> TurnProposal:
        """Perform a single Gemini call to select action and generate reply."""
        prompt = self.format_prompt(packet)
        system_instruction = packet.identity.core_instruction

        start_time = time.perf_counter()
        try:
            response = gemini_pool.generate_content(
                contents=prompt,
                config_opts=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=TurnProposalModel,
                    temperature=0.7,
                    max_output_tokens=300,
                    thinking_config=types.ThinkingConfig(
                        thinking_level="MINIMAL"
                    ),
                ),
            )
            
            logger.info("[TURN_COMPOSER] Gemini call completed in %.2fs", time.perf_counter() - start_time)
            
            # Parse structured JSON response
            data = json.loads(response.text or "{}")
            
            action = data.get("action", "IGNORE").upper()
            target_user_id = data.get("target_user_id") or packet.contract.target_user_id
            anchor_message_id = data.get("anchor_message_id") or packet.contract.anchor_message_id
            speech_act = data.get("speech_act", "STATEMENT").upper()
            intent_tag = data.get("intent_tag", "chat_reply")
            stance = data.get("stance", "friendly").upper()
            reply_text = data.get("reply_text")
            continuity_marker = data.get("continuity_marker")

            # Deterministic override for invalid REQUIRED IGNORE cases
            if packet.contract.participation_mode == "REQUIRED" and action == "IGNORE":
                logger.warning("[TURN_COMPOSER] Overriding invalid IGNORE on REQUIRED opportunity.")
                action = "REPLY"
                if not reply_text:
                    reply_text = "hmm?"

            if action == "IGNORE":
                reply_text = None

            return TurnProposal(
                action=action,
                target_user_id=target_user_id,
                anchor_message_id=anchor_message_id,
                speech_act=speech_act,
                intent_tag=intent_tag,
                stance=stance,
                reply_text=reply_text,
                continuity_marker=continuity_marker,
            )

        except Exception as e:
            logger.exception("[TURN_COMPOSER] structured generation call failed: %s", e)
            # Safe default fallback proposal on Gemini error
            action = "REPLY" if packet.contract.participation_mode == "REQUIRED" else "IGNORE"
            return TurnProposal(
                action=action,
                target_user_id=packet.contract.target_user_id,
                anchor_message_id=packet.contract.anchor_message_id,
                speech_act="STATEMENT",
                intent_tag="fallback_error",
                stance="TIRED",
                reply_text="hmm?" if action == "REPLY" else None,
            )
