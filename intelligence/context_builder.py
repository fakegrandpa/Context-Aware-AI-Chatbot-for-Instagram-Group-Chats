"""
Canonical response context builder — V5, PART 10.

Called exactly once per accepted message (after the social routing layer
has already decided Eve should reply). Both intelligence/response_generator.py
(TEXT) and intelligence/voice_generator.py (VOICE) consume the SAME
ResponseContext produced here — neither path fetches profile/memory/scene
state independently, which is what prevents the two modalities from
drifting into different personalities or different facts.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from models.context import ReplyMetadata, ResponseContext
from storage import eve_state
from storage import memories as mem_store
from storage import profiles as prof_store

logger = logging.getLogger("yap.intelligence.context_builder")


def build_response_context(
    sender_id: str,
    sender_username: str,
    current_message: str,
    scene_messages: List[dict],
    recent_eve_replies: List[str],
    current_message_id: str = "",
    active_exchange_messages: Optional[List[dict]] = None,
    reply_to_username: Optional[str] = None,
    reply_to_text: Optional[str] = None,
    reply_to_message_id: Optional[str] = None,
    reply_to_user_id: Optional[str] = None,
    routing_context: Optional[object] = None,
    thread_id: str = "",
    bot_user_id: str = "",
    bot_username: str = "",
) -> ResponseContext:
    """Assemble the canonical ResponseContext. Pure DB reads — no Gemini calls."""
    topic_words = [w for w in current_message.lower().split() if len(w) > 3][:5]

    try:
        active_mems = mem_store.get_relevant_memories(sender_id, topic_words, limit=5)
        episodic_mems = mem_store.get_episodic_memories(sender_id, limit=3)
        contradictions = mem_store.get_unresolved_contradictions(sender_id)
        profile_summary = prof_store.build_profile_summary(
            user_id=sender_id,
            memories=active_mems,
            episodic_memories=episodic_mems,
            contradictions=contradictions,
        )
    except Exception as e:
        logger.warning("[CONTEXT] profile/memory retrieval failed: %s", e)
        active_mems, episodic_mems, contradictions = [], [], []
        profile_summary = None

    try:
        eve_stable = eve_state.get_stable_facts()
        eve_dynamic = eve_state.get_recent_dynamic_state(limit=5)
    except Exception as e:
        logger.warning("[CONTEXT] eve_state retrieval failed: %s", e)
        eve_stable, eve_dynamic = [], []

    relationship = None
    if profile_summary and profile_summary.get("known"):
        relationship = {
            "relationship_to_yap": profile_summary.get("relationship_to_yap"),
            "familiarity_score": profile_summary.get("familiarity_score"),
            "language_style": profile_summary.get("language_style"),
        }

    try:
        from storage import eve_turns
        recent_turns = eve_turns.get_recent_eve_turns(limit=5)
        formatted_replies = []
        for t in reversed(recent_turns):
            tgt = prof_store.resolve_display_name(t.get("target_user_id"))
            mod = t.get("modality")
            content = t.get("exact_text") or t.get("voice_transcript") or t.get("semantic_summary")
            formatted_replies.append(f"[{mod}] to {tgt}: {content}")
        if not formatted_replies:
            formatted_replies = list(recent_eve_replies[-5:])
    except Exception as e:
        logger.warning("[CONTEXT] eve_turns retrieval failed: %s", e)
        formatted_replies = list(recent_eve_replies[-5:])

    return ResponseContext(
        sender_id=sender_id,
        sender_username=sender_username,
        current_message=current_message,
        current_message_id=current_message_id,
        sender_profile=profile_summary,
        sender_memories=active_mems,
        relationship=relationship,
        recent_gc_messages=scene_messages,
        active_exchange_messages=active_exchange_messages or [],
        reply_metadata=ReplyMetadata(
            reply_to_message_id=reply_to_message_id,
            reply_to_user_id=reply_to_user_id,
            reply_to_username=reply_to_username,
            reply_to_text=reply_to_text,
        ),
        recent_eve_replies=formatted_replies,
        eve_stable_facts=eve_stable,
        eve_dynamic_state=eve_dynamic,
        routing_context=routing_context,
        thread_id=thread_id,
        bot_user_id=bot_user_id,
        bot_username=bot_username,
    )

