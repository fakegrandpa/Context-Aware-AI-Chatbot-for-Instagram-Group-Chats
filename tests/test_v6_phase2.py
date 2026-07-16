"""
EVE V6 Phase 2 AddressResolver and DialogueSession Tests.
"""
from __future__ import annotations

import os
import sys
import sqlite3
import unittest
from datetime import datetime, timezone, timedelta
from typing import List, Optional

# Adjust path to import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.message import NormalizedMessage
from models.scene import SceneSnapshot
from conversation.dialogue_state import DialogueState
from intelligence.address_resolver import (
    AddressResolver,
    OWNERSHIP_EVE,
    OWNERSHIP_SPECIFIC_USER,
    OWNERSHIP_OPEN_GROUP,
    OWNERSHIP_AMBIGUOUS,
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


class TestPhase2AddressResolver(unittest.TestCase):

    def setUp(self):
        self.resolver = AddressResolver(BOT_USER_ID, BOT_USERNAME)

    def test_01_direct_eve_native_reply(self):
        """Direct Eve native reply resolves to EVE with HARD confidence."""
        state = DialogueState("t_addr", BOT_USER_ID, BOT_USERNAME)
        m1 = _make_msg("m1", "userA", "alice", "hello", thread_id="t_addr")
        state.apply_message(m1, 1)

        m2 = _make_msg("m2", "userB", "bob", "reply to bot", reply_to_msg_id="m1", reply_to_user_id=BOT_USER_ID, thread_id="t_addr")
        state.apply_message(m2, 2)

        snapshot = state.create_snapshot(["m2"])
        resolution = self.resolver.resolve(snapshot, [m2])

        self.assertEqual(resolution.ownership, OWNERSHIP_EVE)
        self.assertEqual(resolution.target_user_id, BOT_USER_ID)
        self.assertEqual(resolution.confidence, "HARD")

    def test_02_direct_other_human_reply(self):
        """Direct other-human native reply resolves to SPECIFIC_USER."""
        state = DialogueState("t_addr", BOT_USER_ID, BOT_USERNAME)
        m1 = _make_msg("m1", "userA", "alice", "msg 1", thread_id="t_addr")
        state.apply_message(m1, 1)

        m2 = _make_msg("m2", "userB", "bob", "reply to alice", reply_to_msg_id="m1", reply_to_user_id="userA", thread_id="t_addr")
        state.apply_message(m2, 2)

        snapshot = state.create_snapshot(["m2"])
        resolution = self.resolver.resolve(snapshot, [m2])

        self.assertEqual(resolution.ownership, OWNERSHIP_SPECIFIC_USER)
        self.assertEqual(resolution.target_user_id, "userA")
        self.assertEqual(resolution.confidence, "HARD")

    def test_03_explicit_eve_address(self):
        """Explicit mention or vocative prefix of Eve resolves to EVE with STRONG confidence."""
        state = DialogueState("t_addr", BOT_USER_ID, BOT_USERNAME)
        m1 = _make_msg("m1", "userA", "alice", "@eve check this out", thread_id="t_addr")
        state.apply_message(m1, 1)

        snapshot = state.create_snapshot(["m1"])
        resolution = self.resolver.resolve(snapshot, [m1])

        self.assertEqual(resolution.ownership, OWNERSHIP_EVE)
        self.assertEqual(resolution.target_user_id, BOT_USER_ID)
        self.assertEqual(resolution.confidence, "STRONG")

    def test_04_explicit_other_user_address(self):
        """Explicit mention of another user resolves to SPECIFIC_USER with STRONG confidence."""
        state = DialogueState("t_addr", BOT_USER_ID, BOT_USERNAME)
        m_prev = _make_msg("m_p", "userB", "bob", "earlier", thread_id="t_addr")
        state.apply_message(m_prev, 1)

        m1 = _make_msg("m1", "userA", "alice", "@bob answer me", thread_id="t_addr")
        state.apply_message(m1, 2)

        snapshot = state.create_snapshot(["m1"])
        resolution = self.resolver.resolve(snapshot, [m1])

        self.assertEqual(resolution.ownership, OWNERSHIP_SPECIFIC_USER)
        self.assertEqual(resolution.target_user_id, "userB")
        self.assertEqual(resolution.confidence, "STRONG")

    def test_05_eve_interaction_continuation(self):
        """Messages affiliated with a session where Eve is active resolve to EVE ownership."""
        state = DialogueState("t_addr", BOT_USER_ID, BOT_USERNAME)
        
        # Start session involving Eve
        m1 = _make_msg("m1", "userA", "alice", "eve trigger", thread_id="t_addr")
        state.apply_message(m1, 1)
        
        m_eve = _make_msg("e1", BOT_USER_ID, BOT_USERNAME, "eve response", is_viewer=True, thread_id="t_addr")
        state.apply_message(m_eve, 2)
        
        # Human continuation in the same session without direct mention
        m2 = _make_msg("m2", "userA", "alice", "why dismissive?", reply_to_msg_id="e1", reply_to_user_id=BOT_USER_ID, thread_id="t_addr")
        state.apply_message(m2, 3)
        
        # Trigger message continuing the interaction (without native reply to human)
        # Using sender continuation or interaction continuity to link session
        m3 = _make_msg("m3", "userA", "alice", "haha yes", thread_id="t_addr")
        state.apply_message(m3, 4)

        snapshot = state.create_snapshot([m3.message_id])
        resolution = self.resolver.resolve(snapshot, [m3])

        self.assertEqual(resolution.ownership, OWNERSHIP_EVE)
        self.assertTrue(resolution.continuation_of_eve_interaction)

    def test_06_overlapping_sessions(self):
        """Active human-human session does not pollute or grant ownership to Eve."""
        state = DialogueState("t_addr", BOT_USER_ID, BOT_USERNAME)
        
        # Session 1: Human-only exchange
        m1 = _make_msg("m1", "userA", "alice", "exam tomorrow", thread_id="t_addr")
        state.apply_message(m1, 1)
        m2 = _make_msg("m2", "userB", "bob", "ready?", reply_to_msg_id="m1", reply_to_user_id="userA", thread_id="t_addr")
        state.apply_message(m2, 2)

        # Session 2: Eve session
        m3 = _make_msg("m3", "userC", "charlie", "@eve roast me", thread_id="t_addr")
        state.apply_message(m3, 3)

        # Trigger message continuing the human exchange
        m4 = _make_msg("m4", "userA", "alice", "not really", reply_to_msg_id="m2", reply_to_user_id="userB", thread_id="t_addr")
        state.apply_message(m4, 4)

        snapshot = state.create_snapshot(["m4"])
        resolution = self.resolver.resolve(snapshot, [m4])

        # Resolves to SPECIFIC_USER, not EVE, since it belongs to the human-only session
        self.assertEqual(resolution.ownership, OWNERSHIP_SPECIFIC_USER)
        self.assertFalse(resolution.continuation_of_eve_interaction)

    def test_07_unrelated_interleaved_chatter(self):
        """Interleaved unrelated chatter remains unaffiliated and resolves to AMBIGUOUS."""
        state = DialogueState("t_addr", BOT_USER_ID, BOT_USERNAME)
        
        m1 = _make_msg("m1", "userA", "alice", "weather is nice today", thread_id="t_addr")
        state.apply_message(m1, 1)

        snapshot = state.create_snapshot([m1.message_id])
        resolution = self.resolver.resolve(snapshot, [m1])

        self.assertEqual(resolution.ownership, OWNERSHIP_AMBIGUOUS)
        self.assertIsNone(resolution.session_id)

    def test_08_open_group_broadcast(self):
        """Group keywords or questions without replies resolve to OPEN_GROUP."""
        state = DialogueState("t_addr", BOT_USER_ID, BOT_USERNAME)
        m1 = _make_msg("m1", "userA", "alice", "anyone up for gaming guys?", thread_id="t_addr")
        state.apply_message(m1, 1)

        snapshot = state.create_snapshot(["m1"])
        resolution = self.resolver.resolve(snapshot, [m1])

        self.assertEqual(resolution.ownership, OWNERSHIP_OPEN_GROUP)

    def test_09_quoted_eve_text_ignored(self):
        """Quoted/referenced text of Eve in a reply to another human does not cause false Eve ownership."""
        state = DialogueState("t_addr", BOT_USER_ID, BOT_USERNAME)
        
        # Eve turn
        m_eve = _make_msg("e1", BOT_USER_ID, BOT_USERNAME, "i hate exams", is_viewer=True, thread_id="t_addr")
        state.apply_message(m_eve, 1)
        
        # Human replies to Alice, quoting Eve's words
        m1 = _make_msg("m1", "userA", "alice", "did you hear her say i hate exams?", thread_id="t_addr")
        state.apply_message(m1, 2)
        
        m2 = _make_msg("m2", "userB", "bob", "yeah, but she is lazy", reply_to_msg_id="m1", reply_to_user_id="userA", thread_id="t_addr")
        state.apply_message(m2, 3)

        snapshot = state.create_snapshot(["m2"])
        resolution = self.resolver.resolve(snapshot, [m2])

        # Bob replied to Alice, so it is SPECIFIC_USER, not EVE
        self.assertEqual(resolution.ownership, OWNERSHIP_SPECIFIC_USER)
        self.assertEqual(resolution.target_user_id, "userA")


if __name__ == "__main__":
    unittest.main()
