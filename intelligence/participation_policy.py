"""
Participation Policy Subsystem — V6.
Determines whether Eve should socially participate in the current room moment.
Exposes REQUIRED, ELIGIBLE, SUPPRESS, and DEFER decisions deterministically.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from intelligence.address_resolver import (
    AddressResolution,
    OWNERSHIP_EVE,
    OWNERSHIP_SPECIFIC_USER,
    OWNERSHIP_OPEN_GROUP,
    OWNERSHIP_AMBIGUOUS,
)
from models.message import NormalizedMessage
from models.scene import SceneSnapshot

logger = logging.getLogger("yap.intelligence.participation_policy")

# Participation Decisions
MODE_REQUIRED = "REQUIRED"
MODE_ELIGIBLE = "ELIGIBLE"
MODE_SUPPRESS = "SUPPRESS"
MODE_DEFER = "DEFER"


@dataclass(frozen=True)
class ParticipationDecision:
    mode: str  # REQUIRED, ELIGIBLE, SUPPRESS, DEFER
    reason: str
    pressure_score: int


@dataclass(frozen=True)
class TurnOpportunity:
    opportunity_id: str
    thread_id: str
    snapshot_version: int
    participation_mode: str  # REQUIRED, ELIGIBLE, SUPPRESS, DEFER
    address_resolution: AddressResolution
    target_user_id: Optional[str]
    anchor_message_id: str
    session_id: Optional[str]
    trigger_message_ids: List[str]
    priority: int  # 1 = HIGH (REQUIRED), 2 = MEDIUM (ELIGIBLE), 3 = LOW


class ParticipationPolicy:
    """Evaluates AddressResolution and SceneSnapshot to make a deterministic participation decision."""

    def __init__(self, bot_user_id: str):
        self.bot_user_id = str(bot_user_id)

    def get_participation_pressure(self, snapshot: SceneSnapshot) -> int:
        """Count how many turns Eve has sent in the last 60 seconds of the thread history."""
        now = datetime.now(timezone.utc)
        recent_bot_msgs = [
            m for m in snapshot.recent_events 
            if m.sender_id == self.bot_user_id or m.is_sent_by_viewer
        ]
        count_60s = sum(1 for m in recent_bot_msgs if (now - m.timestamp).total_seconds() <= 60)
        return count_60s

    def evaluate(self, address_res: AddressResolution, snapshot: SceneSnapshot) -> ParticipationDecision:
        """Apply deterministic social rules to decide if Eve should speak."""
        # 1. Compute Eve's recent participation pressure
        pressure = self.get_participation_pressure(snapshot)

        # 2. Check for Deferral due to high room velocity (rapid back-and-forth)
        # If the last two messages in the snapshot occurred within 300ms, and it's not direct
        if len(snapshot.recent_events) >= 2:
            last_msg = snapshot.recent_events[-1]
            prev_msg = snapshot.recent_events[-2]
            time_gap = (last_msg.timestamp - prev_msg.timestamp).total_seconds()
            
            if time_gap < 0.3 and address_res.ownership != OWNERSHIP_EVE:
                logger.info("[PARTICIPATION_POLICY] DEFER: room is highly active (gap=%.2fs)", time_gap)
                return ParticipationDecision(
                    mode=MODE_DEFER,
                    reason=f"High room activity velocity (gap={time_gap:.2f}s)",
                    pressure_score=pressure
                )

        # 3. Direct Native Reply or Explicit mentions (REQUIRED)
        if address_res.ownership == OWNERSHIP_EVE:
            if address_res.evidence.get("native_reply_to_eve") or address_res.evidence.get("explicit_address_to_eve"):
                logger.info("[PARTICIPATION_POLICY] REQUIRED: direct target to Eve")
                return ParticipationDecision(
                    mode=MODE_REQUIRED,
                    reason="Direct native reply or explicit address to Eve",
                    pressure_score=pressure
                )
            
            # Active interaction/session continuation
            if address_res.continuation_of_eve_interaction:
                # If pressure is high, we downgrade session continuation from REQUIRED to ELIGIBLE
                if pressure >= 3:
                    logger.info("[PARTICIPATION_POLICY] ELIGIBLE: session continuation, but high pressure (%d)", pressure)
                    return ParticipationDecision(
                        mode=MODE_ELIGIBLE,
                        reason=f"Eve session continuation with high pressure ({pressure} turns in 60s)",
                        pressure_score=pressure
                    )
                else:
                    logger.info("[PARTICIPATION_POLICY] REQUIRED: session continuation")
                    return ParticipationDecision(
                        mode=MODE_REQUIRED,
                        reason="Active Eve dialogue session continuation",
                        pressure_score=pressure
                    )

        # 4. Human-human exchange (SUPPRESS)
        if address_res.ownership == OWNERSHIP_SPECIFIC_USER:
            logger.info("[PARTICIPATION_POLICY] SUPPRESS: human-to-human conversation")
            return ParticipationDecision(
                mode=MODE_SUPPRESS,
                reason="Directed toward another human participant",
                pressure_score=pressure
            )

        # 5. Open Group broadcast (ELIGIBLE)
        if address_res.ownership == OWNERSHIP_OPEN_GROUP:
            if pressure >= 2:  # Bounded suppression for unsolicited group entries
                logger.info("[PARTICIPATION_POLICY] SUPPRESS: open group, but high pressure (%d)", pressure)
                return ParticipationDecision(
                    mode=MODE_SUPPRESS,
                    reason=f"Open group broadcast suppressed due to pressure ({pressure} turns in 60s)",
                    pressure_score=pressure
                )
            else:
                logger.info("[PARTICIPATION_POLICY] ELIGIBLE: open group broadcast")
                return ParticipationDecision(
                    mode=MODE_ELIGIBLE,
                    reason="Open group broadcast / general question",
                    pressure_score=pressure
                )

        # 6. Ambiguous / default (SUPPRESS)
        logger.info("[PARTICIPATION_POLICY] SUPPRESS: ambiguous/unclear target")
        return ParticipationDecision(
            mode=MODE_SUPPRESS,
            reason="Ambiguous ownership target",
            pressure_score=pressure
        )

    def create_opportunity(
        self,
        decision: ParticipationDecision,
        address_res: AddressResolution,
        snapshot: SceneSnapshot,
        trigger_message_ids: List[str]
    ) -> TurnOpportunity:
        """Factory helper to build a TurnOpportunity."""
        opp_id = str(uuid.uuid4())[:8]
        priority = 3
        if decision.mode == MODE_REQUIRED:
            priority = 1
        elif decision.mode == MODE_ELIGIBLE:
            priority = 2

        return TurnOpportunity(
            opportunity_id=opp_id,
            thread_id=snapshot.thread_id,
            snapshot_version=snapshot.room_version,
            participation_mode=decision.mode,
            address_resolution=address_res,
            target_user_id=address_res.target_user_id,
            anchor_message_id=address_res.anchor_message_id,
            session_id=address_res.session_id,
            trigger_message_ids=list(trigger_message_ids),
            priority=priority
        )
