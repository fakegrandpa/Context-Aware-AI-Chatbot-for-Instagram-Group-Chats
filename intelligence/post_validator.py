"""
Post-Generation Validator Subsystem — V6.
Performs deterministic, non-cognitive checks on the generated turn proposals.
"""
from __future__ import annotations

import logging
from typing import Optional, List
from models.message import NormalizedMessage
from conversation.chat_actor import GenerationLease

logger = logging.getLogger("yap.intelligence.post_validator")


class PostValidationResult:
    def __init__(self, is_valid: bool, reason: Optional[str] = None, modified_text: Optional[str] = None):
        self.is_valid = is_valid
        self.reason = reason
        self.modified_text = modified_text


class PostValidator:
    def __init__(self, bot_user_id: str):
        self.bot_user_id = bot_user_id

    def validate(
        self,
        proposal_text: str,
        lease: GenerationLease,
        recent_sent_texts: List[str]
    ) -> PostValidationResult:
        """
        Validates the proposal text against a set of deterministic rules.
        """
        # 1. Check if proposal text is empty or only whitespace
        if not proposal_text or not proposal_text.strip():
            return PostValidationResult(False, "Empty reply text")

        # 2. Check length boundaries (max 1000 characters)
        if len(proposal_text) > 1000:
            return PostValidationResult(False, f"Reply text exceeds maximum length limit: {len(proposal_text)} chars")

        # 3. Duplicate check (prevent exact match repetition within recent turns)
        clean_proposal = proposal_text.strip().lower()
        for recent in recent_sent_texts:
            if clean_proposal == recent.strip().lower():
                return PostValidationResult(False, "Duplicate message detected within recent turns")

        # 4. Check lease status
        if lease.status != "ACTIVE":
            return PostValidationResult(False, f"Lease is not ACTIVE (status={lease.status})")

        return PostValidationResult(True)
