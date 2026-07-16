"""
Background memory extractors — V5.

Two independent, lightweight extraction passes, both running from
workers/memory_worker.py on the same poll cadence (no extra cron jobs):

1. extract_batch()          — facts about GC users (unchanged V4 Claim/Belief
                               logic, stored in storage/memories.py, keyed by
                               user_id).
2. extract_eve_self_state() — Eve's own life-continuity, extracted ONLY from
                               Eve's own already-sent messages, stored in
                               storage/eve_state.py (ownership-distinct from
                               user memories — never merged with a real
                               user's profile).

Never blocks the realtime pipeline. Errors are caught and logged.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from google.genai import types
from pydantic import BaseModel, Field

import config
from intelligence.prompts import (
    MEMORY_EXTRACTION_INSTRUCTION,
    EVE_SELF_STATE_INSTRUCTION,
    build_memory_prompt,
    build_eve_self_state_prompt,
)
from storage import eve_state
from storage import memories as mem_store
from storage import profiles as prof_store
from intelligence import gemini_pool

logger = logging.getLogger("yap.intelligence.memory_extractor")


class _MemoryCandidate(BaseModel):
    user_id: str
    memory_type: str = Field(description="identity|preference|personal_fact|relationship|episodic")
    slot: str = Field(description="Specific semantic slot. For identity, use 'name', 'age', 'city', 'college', 'course'. Use 'general' for preference/episodic/relationship unless a clear slot exists.")
    value_fact: str = Field(description="The specific claim value or factual statement, e.g. 'Atharv', 'Likes cricket', 'Lost football match'.")
    claim_type: str = Field(description="NEW | SUPPORT | CONTRADICTION | CORRECTION | JOKE_OR_UNCERTAIN")
    confidence: float = Field(ge=0.0, le=1.0)
    source_message_id: Optional[str] = None


class _MemoryExtractionResult(BaseModel):
    memories: List[_MemoryCandidate] = Field(default_factory=list)


def extract_batch(messages: List[dict]) -> int:
    """
    Extract memories from a batch of message dicts (GC users only).
    Stores valid memories to SQLite using the V4 Claim/Belief logic.
    Returns the count of memories stored.

    messages: list of {message_id, sender_id, sender_username, text, ...} dicts
    """
    if not messages:
        return 0

    # Filter trivial messages before sending to Gemini
    meaningful = [
        m for m in messages
        if m.get("text") and not mem_store.is_trivial_text(m.get("text", ""))
    ]
    if not meaningful:
        logger.info("[MEMORY] batch skipped — all trivial (%d messages)", len(messages))
        return 0

    logger.info("[MEMORY] batch started messages=%d meaningful=%d",
                len(messages), len(meaningful))

    prompt = build_memory_prompt(meaningful)

    try:
        response = gemini_pool.generate_content(
            contents=prompt,
            config_opts=types.GenerateContentConfig(
                system_instruction=MEMORY_EXTRACTION_INSTRUCTION,
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=_MemoryExtractionResult,
            ),
        )
        text = (response.text or "").strip()
        if not text:
            logger.warning("[MEMORY] empty response from Gemini")
            return 0

        result = _MemoryExtractionResult.model_validate_json(text)
    except Exception as e:
        logger.error("[MEMORY] Gemini extraction failed: %s", e)
        return 0

    stored = 0
    for candidate in result.memories:
        if not candidate.user_id or not candidate.value_fact:
            continue
        if candidate.memory_type not in mem_store.VALID_MEMORY_TYPES:
            continue
        if candidate.confidence < 0.4:
            continue
        mem_id = mem_store.add_claim_memory(
            user_id=candidate.user_id,
            memory_type=candidate.memory_type,
            slot=candidate.slot,
            value=candidate.value_fact,
            claim_type=candidate.claim_type,
            confidence=candidate.confidence,
            source_message_id=candidate.source_message_id,
        )
        if mem_id is not None:
            stored += 1
        # A confirmed (non-joke, reasonably confident) identity/name claim is
        # useful profile data — make it actually reach response context via
        # preferred_name, instead of sitting inert in MEMORIES only (PART 5).
        if (
            candidate.memory_type == "identity"
            and candidate.slot == "name"
            and candidate.claim_type != "JOKE_OR_UNCERTAIN"
            and candidate.confidence >= 0.6
        ):
            prof_store.update_preferred_name(candidate.user_id, candidate.value_fact)

    logger.info("[MEMORY] batch complete stored=%d from %d candidates",
                stored, len(result.memories))
    return stored


class _EveLifeEventCandidate(BaseModel):
    slot: str
    value: str
    confidence: float = Field(ge=0.0, le=1.0)
    source_message_id: Optional[str] = None


class _EveLifeExtractionResult(BaseModel):
    life_events: List[_EveLifeEventCandidate] = Field(default_factory=list)


def extract_eve_self_state(messages: List[dict]) -> int:
    """
    Extract Eve's own life-continuity from a batch of Eve's OWN sent
    messages. Evidence-driven only — never invents events, only notices
    when Eve already said something durable about herself. Stores via
    storage/eve_state.py, never storage/memories.py.
    """
    if not messages:
        return 0

    meaningful = [
        m for m in messages
        if m.get("text") and not mem_store.is_trivial_text(m.get("text", ""))
    ]
    if not meaningful:
        return 0

    prompt = build_eve_self_state_prompt(meaningful)

    try:
        response = gemini_pool.generate_content(
            contents=prompt,
            config_opts=types.GenerateContentConfig(
                system_instruction=EVE_SELF_STATE_INSTRUCTION,
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=_EveLifeExtractionResult,
            ),
        )
        text = (response.text or "").strip()
        if not text:
            return 0
        result = _EveLifeExtractionResult.model_validate_json(text)
    except Exception as e:
        logger.error("[EVE_STATE] extraction failed: %s", e)
        return 0

    stored = 0
    for candidate in result.life_events:
        if not candidate.value or candidate.confidence < 0.5:
            continue
        state_id = eve_state.add_dynamic_state(
            slot=candidate.slot or "general",
            value=candidate.value,
            confidence=candidate.confidence,
            source_message_id=candidate.source_message_id,
        )
        if state_id is not None:
            stored += 1

    if stored:
        logger.info("[EVE_STATE] batch complete stored=%d from %d candidates", stored, len(result.life_events))
    return stored
