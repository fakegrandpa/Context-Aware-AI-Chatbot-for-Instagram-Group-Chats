"""
EVE V6 Phase 3 ParticipationPolicy Tests.
"""
from __future__ import annotations

import os
import sys
import unittest
from datetime import datetime, timezone, timedelta
from typing import List, Optional

# Adjust path to import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.message import NormalizedMessage
from models.scene import SceneSnapshot
from conversation.dialogue_state import DialogueState
from intelligence.address_resolver import AddressResolver
from intelligence.participation_policy import (
    ParticipationPolicy,
    MODE_REQUIRED,
    MODE_ELIGIBLE,
    MODE_SUPPRESS,
    MODE_DEFER,
)

BOT_USER_ID = "99999"
BOT_USERNAME = "eve"


def _make_msg(
    message_id: str,
    sender_id: str,
    sender_username: str,
    text: str,
    is_viewer: bool = False,
    reply_to_msg_id: Optional[str] = None,
    reply_to_user_id: Optional[str] = None,
    ts_offset_seconds: int = 0,
    item_type: str = "text",
    thread_id: str = "test_thread",
) -> NormalizedMessage:
    ts = datetime.now(timezone.utc) + timedelta(seconds=ts_offset_seconds)
    return NormalizedMessage(
        message_id=str(message_id),
        thread_id=str(thread_id),
        sender_id=str(sender_id),
        sender_username=sender_username,
        text=text,
        timestamp=ts,
        item_type=item_type,
        is_sent_by_viewer=is_viewer,
        reply_to_message_id=reply_to_msg_id,
        reply_to_user_id=reply_to_user_id,
    )


class TestPhase3ParticipationPolicy(unittest.TestCase):

    def setUp(self):
        self.resolver = AddressResolver(BOT_USER_ID, BOT_USERNAME)
        self.policy = ParticipationPolicy(BOT_USER_ID)

    def test_01_direct_eve_reply_cannot_be_ignored(self):
        """Direct native reply to Eve is REQUIRED."""
        state = DialogueState("t_when", BOT_USER_ID, BOT_USERNAME)
        m1 = _make_msg("m1", "userA", "alice", "hey @eve", thread_id="t_when")
        state.apply_message(m1, 1)

        snapshot = state.create_snapshot(["m1"])
        res = self.resolver.resolve(snapshot, [m1])
        decision = self.policy.evaluate(res, snapshot)

        self.assertEqual(decision.mode, MODE_REQUIRED)

    def test_02_other_human_conversation_suppressed(self):
        """Conversations between other human users are SUPPRESSed."""
        state = DialogueState("t_when", BOT_USER_ID, BOT_USERNAME)
        m1 = _make_msg("m1", "userA", "alice", "msg 1", ts_offset_seconds=-10, thread_id="t_when")
        state.apply_message(m1, 1)
        m2 = _make_msg("m2", "userB", "bob", "reply to 1", reply_to_msg_id="m1", reply_to_user_id="userA", ts_offset_seconds=0, thread_id="t_when")
        state.apply_message(m2, 2)

        snapshot = state.create_snapshot(["m2"])
        res = self.resolver.resolve(snapshot, [m2])
        decision = self.policy.evaluate(res, snapshot)

        self.assertEqual(decision.mode, MODE_SUPPRESS)

    def test_03_open_group_eligibility(self):
        """Open group broadcast is ELIGIBLE under normal conditions."""
        state = DialogueState("t_when", BOT_USER_ID, BOT_USERNAME)
        m1 = _make_msg("m1", "userA", "alice", "anyone want to play guys?", thread_id="t_when")
        state.apply_message(m1, 1)

        snapshot = state.create_snapshot(["m1"])
        res = self.resolver.resolve(snapshot, [m1])
        decision = self.policy.evaluate(res, snapshot)

        self.assertEqual(decision.mode, MODE_ELIGIBLE)

    def test_04_over_participation_suppresses_open_group(self):
        """Eligible open group replies are suppressed if Eve spoke too recently."""
        state = DialogueState("t_when", BOT_USER_ID, BOT_USERNAME)
        
        # Inject three recent Eve messages spaced in time
        for i in range(3):
            m_bot = _make_msg(f"bot_{i}", BOT_USER_ID, BOT_USERNAME, f"bot chat {i}", is_viewer=True, ts_offset_seconds=-30 + i*5, thread_id="t_when")
            state.apply_message(m_bot, i + 1)
            
        m1 = _make_msg("m1", "userA", "alice", "anyone there guys?", ts_offset_seconds=0, thread_id="t_when")
        state.apply_message(m1, 4)

        snapshot = state.create_snapshot(["m1"])
        res = self.resolver.resolve(snapshot, [m1])
        decision = self.policy.evaluate(res, snapshot)

        self.assertEqual(decision.mode, MODE_SUPPRESS)
        self.assertEqual(decision.pressure_score, 3)

    def test_05_direct_eve_reply_never_suppressed_by_pressure(self):
        """REQUIRED direct turns are never suppressed by participation pressure."""
        state = DialogueState("t_when", BOT_USER_ID, BOT_USERNAME)
        
        # High pressure (4 messages in 60s)
        for i in range(4):
            m_bot = _make_msg(f"bot_{i}", BOT_USER_ID, BOT_USERNAME, "chat", is_viewer=True, thread_id="t_when")
            state.apply_message(m_bot, i + 1)
            
        m1 = _make_msg("m1", "userA", "alice", "@eve answer me immediately", thread_id="t_when")
        state.apply_message(m1, 5)

        snapshot = state.create_snapshot(["m1"])
        res = self.resolver.resolve(snapshot, [m1])
        decision = self.policy.evaluate(res, snapshot)

        # Still REQUIRED!
        self.assertEqual(decision.mode, MODE_REQUIRED)

    def test_06_rapid_room_velocity_defers(self):
        """If room activity is extremely rapid, eligible turns are DEFERred."""
        state = DialogueState("t_when", BOT_USER_ID, BOT_USERNAME)
        
        # Post two back-to-back messages within 100ms
        m1 = _make_msg("m1", "userA", "alice", "hi", ts_offset_seconds=-1, thread_id="t_when")
        state.apply_message(m1, 1)
        
        m2 = _make_msg("m2", "userB", "bob", "everyone check this", ts_offset_seconds=-1, thread_id="t_when")
        # Shift timestamp slightly to simulate 100ms gap
        m2.timestamp = m1.timestamp + timedelta(milliseconds=100)
        state.apply_message(m2, 2)

        snapshot = state.create_snapshot(["m2"])
        res = self.resolver.resolve(snapshot, [m2])
        decision = self.policy.evaluate(res, snapshot)

        self.assertEqual(decision.mode, MODE_DEFER)

    def test_07_opportunity_creation_values(self):
        """Opportunity creation translates priority and target fields properly."""
        state = DialogueState("t_when", BOT_USER_ID, BOT_USERNAME)
        m1 = _make_msg("m1", "userA", "alice", "@eve query", thread_id="t_when")
        state.apply_message(m1, 1)

        snapshot = state.create_snapshot(["m1"])
        res = self.resolver.resolve(snapshot, [m1])
        decision = self.policy.evaluate(res, snapshot)
        opportunity = self.policy.create_opportunity(decision, res, snapshot, ["m1"])

        self.assertEqual(opportunity.participation_mode, MODE_REQUIRED)
        self.assertEqual(opportunity.priority, 1)  # REQUIRED is priority 1
        self.assertEqual(opportunity.target_user_id, BOT_USER_ID)
        self.assertEqual(opportunity.trigger_message_ids, ["m1"])


if __name__ == "__main__":
    unittest.main()
