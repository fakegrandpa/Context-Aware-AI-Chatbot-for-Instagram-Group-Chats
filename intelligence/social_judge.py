"""
Upgraded Gemini social judge for GEMINI_REQUIRED cases.

Produces a structured SocialDecisionResult with target_type, target_user_id,
action, confidence, and reason. Separate from reply generation.

Safe IGNORE fallback on any API/schema error.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from google.genai import types
from pydantic import BaseModel, Field

import config
from models.decision import SocialDecisionResult
from intelligence.prompts import SOCIAL_DECISION_INSTRUCTION, build_decision_prompt
from intelligence import gemini_pool

logger = logging.getLogger("yap.intelligence.social_judge")


class _SocialDecisionSchema(BaseModel):
    target_type: str = Field(description="EVE, SPECIFIC_USER, GROUP, or UNKNOWN")
    target_user_id: Optional[str] = Field(default=None, description="user_id or null")
    action: str = Field(description="REPLY or IGNORE")
    confidence: float = Field(description="0.0 to 1.0")
    tone: str = Field(description="PLAYFUL, HOSTILE, SERIOUS, NEUTRAL, AFFECTIONATE, or UNCLEAR")
    reason: str = Field(description="short internal reason")


def judge(
    msg_text: str,
    sender_username: str,
    reply_to_username: Optional[str],
    reply_to_text: Optional[str],
    scene_messages: List[dict],
    profile_summary: Optional[dict],
    fatigue_multiplier: float = 0.0,
    trigger_message_id: Optional[str] = None,
) -> SocialDecisionResult:
    """
    Call Gemini to make a social decision for an ambiguous message.

    `scene_messages` must be the RAW recent GC scene (see
    storage.messages.get_recent_scene), NOT a lane-filtered list — PART 7 of
    the V5 spec requires the router to see the full picture so it can tell
    who is talking to whom, independent of lane assignment.

    Returns SocialDecisionResult. On any failure, returns safe IGNORE default.
    """
    prompt = build_decision_prompt(
        msg_text=msg_text,
        sender_username=sender_username,
        reply_to_username=reply_to_username,
        reply_to_text=reply_to_text,
        scene_messages=scene_messages,
        profile_summary=profile_summary,
        trigger_message_id=trigger_message_id,
    )

    try:
        response = gemini_pool.generate_content(
            contents=prompt,
            config_opts=types.GenerateContentConfig(
                system_instruction=SOCIAL_DECISION_INSTRUCTION,
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=_SocialDecisionSchema,
                thinking_config=types.ThinkingConfig(
                    thinking_level="MINIMAL"
                ),
            ),
        )
        text = (response.text or "").strip()
        if not text:
            raise ValueError("empty response from Gemini social judge")

        obj = _SocialDecisionSchema.model_validate_json(text)

        # Validate fields
        target_type = obj.target_type if obj.target_type in ("EVE", "SPECIFIC_USER", "GROUP", "UNKNOWN") else "UNKNOWN"
        action = obj.action if obj.action in ("REPLY", "IGNORE") else "IGNORE"

        # Apply action targeting rules
        if target_type == "EVE":
            action = "REPLY"
        elif target_type == "SPECIFIC_USER":
            action = "IGNORE"
        elif target_type == "GROUP":
            # Use action from model, but apply fatigue override if applicable
            if fatigue_multiplier > 0.5 and action == "REPLY" and obj.confidence < 0.7:
                action = "IGNORE"
                logger.info("[FATIGUE] overrode REPLY→IGNORE for group target, fatigue=%.2f", fatigue_multiplier)
        elif target_type == "UNKNOWN":
            # Use action from model, apply fatigue override if applicable
            if fatigue_multiplier > 0.5 and action == "REPLY" and obj.confidence < 0.7:
                action = "IGNORE"
                logger.info("[FATIGUE] overrode REPLY→IGNORE for unknown target, fatigue=%.2f", fatigue_multiplier)

        result = SocialDecisionResult(
            target_type=target_type,
            target_user_id=obj.target_user_id,
            action=action,
            confidence=obj.confidence,
            tone=obj.tone,
            reason=obj.reason or "",
        )
        logger.info("[SOCIAL] %s", result)
        return result

    except Exception as e:
        logger.warning("[SOCIAL] judge failed (%s), defaulting to IGNORE", e)
        return SocialDecisionResult(
            target_type="UNKNOWN",
            target_user_id=None,
            action="IGNORE",
            confidence=1.0,
            tone="UNCLEAR",
            reason=f"error: {e}",
        )
