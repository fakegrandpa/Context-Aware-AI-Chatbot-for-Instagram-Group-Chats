"""
Turn Planner — V5.5.
Generates a TurnPlan to align TEXT and VOICE generation intent.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from google.genai import types
from pydantic import BaseModel, Field

from models.context import ResponseContext
from intelligence import gemini_pool
from intelligence.prompts import (
    _profile_block,
    _eve_life_block,
    _scene_block,
    _active_exchange_block,
)

logger = logging.getLogger("yap.intelligence.turn_planner")

TURN_PLANNER_INSTRUCTION = """You are the conversational planner for Eve, a 20-year-old female bot.
Your job is to determine Eve's INTENT, STANCE, and SPEECH ACT for her next turn in the conversation.
Do NOT generate the actual reply. Instead, define the structure, intent, and constraints of the reply.

Analyze the CURRENT MESSAGE and plan:
1. Speech Act (ANSWER, QUESTION, TEASE, LAUGH, AGREE, DISAGREE, HELLO, CHAT)
2. Intent (e.g. "answer where Eve is from", "react to joke", "ask for clarification")
3. Stance/Energy (CASUAL, PLAYFUL, SERIOUS, DRY, CONFUSED, EXCITED)
4. Facts to use (if any relevant facts about Eve or the user should be mentioned)
5. Continuity notes (how to connect to what was said earlier, keeping thread coherent)
6. Avoid topics (repetition or contradictions to avoid based on previous turns)

Output ONLY valid JSON matching this schema:
{
  "speech_act": "ANSWER" | "QUESTION" | "TEASE" | "LAUGH" | "AGREE" | "DISAGREE" | "HELLO" | "CHAT",
  "intent": "string",
  "stance": "CASUAL" | "PLAYFUL" | "SERIOUS" | "DRY" | "CONFUSED" | "EXCITED",
  "facts_to_use": ["string"],
  "continuity_notes": "string",
  "avoid_topics": ["string"]
}
"""


class _TurnPlanSchema(BaseModel):
    speech_act: str = Field(description="ANSWER, QUESTION, TEASE, LAUGH, AGREE, DISAGREE, HELLO, or CHAT")
    intent: str = Field(description="Short description of what Eve intends to say/do, e.g. 'answer where she is from' or 'tease Rahul about football'.")
    stance: str = Field(description="CASUAL, PLAYFUL, SERIOUS, DRY, CONFUSED, or EXCITED")
    facts_to_use: List[str] = Field(default_factory=list, description="Specific facts from Eve's stable facts or user memories to mention, if any.")
    continuity_notes: str = Field(description="Notes on how this connects to previous turns in the conversation.")
    avoid_topics: List[str] = Field(default_factory=list, description="Topics or phrases to avoid to prevent repetition/contradiction.")


@dataclass
class TurnPlan:
    conversation_id: str
    trigger_message_id: str
    target_user_id: Optional[str]
    speech_act: str
    intent: str
    stance: str
    facts_to_use: List[str]
    continuity_notes: str
    avoid_topics: List[str]
    conversation_version: int


def build_turn_plan_prompt(ctx: ResponseContext) -> str:
    parts = []
    
    profile_block = _profile_block(ctx)
    if profile_block:
        parts.append(profile_block + "\n")
        
    life_block = _eve_life_block(ctx)
    if life_block:
        parts.append(life_block + "\n")
        
    parts.append(f"Recent group chat messages:\n{_scene_block(ctx)}\n")
    
    exchange_block = _active_exchange_block(ctx)
    if exchange_block:
        parts.append(f"ACTIVE/RELEVANT EXCHANGE (focused highlight of the live thread):\n{exchange_block}\n")
        
    if ctx.reply_metadata and ctx.reply_metadata.reply_to_username and ctx.reply_metadata.reply_to_text:
        parts.append(
            f"REPLY CONTEXT: {ctx.sender_username} is replying to "
            f"{ctx.reply_metadata.reply_to_username}'s message: \"{ctx.reply_metadata.reply_to_text}\"\n"
        )
        
    parts.append(f"CURRENT MESSAGE — plan the response to this now:\n{ctx.sender_username}: {ctx.current_message}")
    
    return "\n".join(parts)


def generate_turn_plan(ctx: ResponseContext, conversation_version: int) -> TurnPlan:
    prompt = build_turn_plan_prompt(ctx)
    try:
        response = gemini_pool.generate_content(
            contents=prompt,
            config_opts=types.GenerateContentConfig(
                system_instruction=TURN_PLANNER_INSTRUCTION,
                temperature=0.2,
                response_mime_type="application/json",
                response_schema=_TurnPlanSchema,
                thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
            ),
        )
        text = (response.text or "").strip()
        if not text:
            raise ValueError("empty response from Gemini")
            
        obj = _TurnPlanSchema.model_validate_json(text)
        return TurnPlan(
            conversation_id=ctx.thread_id,
            trigger_message_id=ctx.current_message_id,
            target_user_id=ctx.sender_id,
            speech_act=obj.speech_act,
            intent=obj.intent,
            stance=obj.stance,
            facts_to_use=obj.facts_to_use,
            continuity_notes=obj.continuity_notes,
            avoid_topics=obj.avoid_topics,
            conversation_version=conversation_version,
        )
    except Exception as e:
        logger.error("[TURN_PLANNER] failed to generate plan: %s", e)
        # Safe fallback TurnPlan so we never crash the pipeline
        return TurnPlan(
            conversation_id=ctx.thread_id,
            trigger_message_id=ctx.current_message_id,
            target_user_id=ctx.sender_id,
            speech_act="CHAT",
            intent="respond casually to " + ctx.sender_username,
            stance="CASUAL",
            facts_to_use=[],
            continuity_notes="casually continuing",
            avoid_topics=[],
            conversation_version=conversation_version,
        )
