"""
Decision models — internal result types for the attention gate and social judge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

# Attention gate outcomes
AttentionDecision = Literal["LOCAL_REPLY", "LOCAL_IGNORE", "GEMINI_REQUIRED"]


@dataclass
class AttentionResult:
    """Output of the local attention gate."""
    decision: AttentionDecision
    score: float                    # Signed score: positive = reply tendency, negative = ignore
    reasons: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        reasons_str = ",".join(self.reasons) if self.reasons else "none"
        return f"{self.decision} score={self.score:.2f} reasons={reasons_str}"


# Social judge outcomes
TargetType = Literal["EVE", "SPECIFIC_USER", "GROUP", "UNKNOWN"]
SocialAction = Literal["REPLY", "IGNORE"]

# Reply modality outcomes (V5)
ReplyMode = Literal["TEXT", "VOICE"]


@dataclass
class SocialDecisionResult:
    """Output of the Gemini social judge (used for GEMINI_REQUIRED cases)."""
    target_type: TargetType
    target_user_id: Optional[str]  # user_id or None
    action: SocialAction
    confidence: float              # 0.0–1.0
    reason: str                    # Short internal reason string
    tone: Optional[str] = None     # PLAYFUL | HOSTILE | SERIOUS | NEUTRAL | AFFECTIONATE | UNCLEAR

    def __str__(self) -> str:
        return (
            f"action={self.action} target={self.target_type} "
            f"user={self.target_user_id} confidence={self.confidence:.2f} tone={self.tone} reason={self.reason}"
        )
