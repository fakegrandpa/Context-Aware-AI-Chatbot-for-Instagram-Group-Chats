"""
ResponseContext — the single canonical state bundle consumed by BOTH the
text and voice reply paths (V5, PART 10).

Built once per accepted message/burst by intelligence/context_builder.py.
Text formatting (intelligence/prompts.format_text_context) and voice
formatting (intelligence/prompts.format_voice_context) both read from this
same object — neither path fetches profile/memory/scene state independently,
so personality and factual state cannot drift between modalities.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class ReplyMetadata:
    """Native-reply graph info for the current message, if any."""
    reply_to_message_id: Optional[str] = None
    reply_to_user_id: Optional[str] = None
    reply_to_username: Optional[str] = None
    reply_to_text: Optional[str] = None


@dataclass
class ResponseContext:
    # Who is speaking to Eve right now
    sender_id: str
    sender_username: str

    # The message Eve is responding to. Kept explicitly separate from
    # historical context — see PART 10 / PART 18 (no-text-intermediary rule).
    current_message: str

    # message_id of current_message, if known — used to exclude it from
    # recent_gc_messages when rendering the scene (it's shown separately,
    # explicitly marked, not folded into "history").
    current_message_id: str = ""

    # storage.profiles.build_profile_summary(...) output for the sender.
    sender_profile: Optional[dict] = None

    # storage.memories active/episodic/contradiction lists for the sender.
    sender_memories: List[dict] = field(default_factory=list)

    # Relationship/familiarity snapshot, pulled out of sender_profile for
    # convenient access (relationship_to_yap, familiarity_score).
    relationship: Optional[dict] = None

    # The canonical raw recent GC scene (storage.messages.get_recent_scene) —
    # the SAME scene the attention gate / social judge used for targeting.
    # Reply relationships are preserved (see intelligence.prompts.format_raw_scene)
    # so response generation understands who was talking to whom, not just
    # a flattened "sender: text" log.
    recent_gc_messages: List[dict] = field(default_factory=list)

    # Optional focused highlight — the current lane's messages, when a lane
    # is known and adds useful focus beyond the raw scene above. This is an
    # ADDITION, not a replacement: recent_gc_messages must never be dropped
    # in favor of this narrower view (that was the participant-only tunnel
    # vision bug this field exists to avoid repeating).
    active_exchange_messages: List[dict] = field(default_factory=list)

    # Native reply graph for the current message.
    reply_metadata: ReplyMetadata = field(default_factory=ReplyMetadata)

    # Eve's own most recent reply texts (any modality) — anti-repetition.
    recent_eve_replies: List[str] = field(default_factory=list)

    # Eve's own character-side state: stable identity facts + recent dynamic
    # life-state. Ownership-distinct from sender_memories.
    eve_stable_facts: List[dict] = field(default_factory=list)
    eve_dynamic_state: List[dict] = field(default_factory=list)

    # Whatever the attention gate / social judge decided, for traceability.
    # Not necessarily rendered into the prompt verbatim.
    routing_context: Optional[Any] = None

    # thread/bot identity, useful to formatters without re-threading params
    thread_id: str = ""
    bot_user_id: str = ""
    bot_username: str = ""
