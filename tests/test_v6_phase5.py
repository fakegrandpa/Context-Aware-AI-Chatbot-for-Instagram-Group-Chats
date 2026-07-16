"""
EVE V6 Phase 5 ContextSelector and GenerationPacket Tests.
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
from storage import database as db_module

def _get_test_conn():
    conn = sqlite3.connect("file:testdb?mode=memory&cache=shared", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

db_module.get_connection = _get_test_conn

import storage.eve_turns
storage.eve_turns.get_connection = _get_test_conn
import storage.profiles
storage.profiles.get_connection = _get_test_conn
import storage.memories
storage.memories.get_connection = _get_test_conn
import storage.messages
storage.messages.get_connection = _get_test_conn
import storage.chat_state
storage.chat_state.get_connection = _get_test_conn

from storage import database, profiles, memories, eve_turns
from conversation.dialogue_state import DialogueState
from intelligence.address_resolver import AddressResolver
from intelligence.participation_policy import ParticipationPolicy
from intelligence.context_selector import ContextSelector
from models.generation_packet import GenerationPacket

BOT_USER_ID = "99999"
BOT_USERNAME = "eve"
CORE_PROMPT = "You are Eve, a chill GC participant."

_KEEP_ALIVE_CONN: Optional[sqlite3.Connection] = None


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


class TestPhase5ContextSelector(unittest.TestCase):

    def setUp(self):
        global _KEEP_ALIVE_CONN
        if _KEEP_ALIVE_CONN:
            _KEEP_ALIVE_CONN.close()
            _KEEP_ALIVE_CONN = None
        database.DB_PATH = "file:testdb?mode=memory&cache=shared"
        _KEEP_ALIVE_CONN = _get_test_conn()
        database.init_db()
        self.resolver = AddressResolver(BOT_USER_ID, BOT_USERNAME)
        self.policy = ParticipationPolicy(BOT_USER_ID)
        self.selector = ContextSelector(BOT_USER_ID, BOT_USERNAME, CORE_PROMPT)

    def tearDown(self):
        global _KEEP_ALIVE_CONN
        if _KEEP_ALIVE_CONN:
            _KEEP_ALIVE_CONN.close()
            _KEEP_ALIVE_CONN = None

    def test_01_target_familiarity_band(self):
        """Familiarity scores are mapped to correct TargetProfile familiarity bands."""
        user_id = "userA"
        # 1. Test Stranger
        profiles.get_or_create_user(user_id, "alice")
        state = DialogueState("t_ctx", BOT_USER_ID, BOT_USERNAME)
        m1 = _make_msg("m1", user_id, "alice", "@eve hello", thread_id="t_ctx")
        state.apply_message(m1, 1)

        snapshot = state.create_snapshot(["m1"])
        res = self.resolver.resolve(snapshot, [m1])
        dec = self.policy.evaluate(res, snapshot)
        opp = self.policy.create_opportunity(dec, res, snapshot, ["m1"])

        packet = self.selector.select(snapshot, opp)
        self.assertEqual(packet.target.familiarity_band, "STRANGER")

        # 2. Upgrade to Friend
        with database.get_connection() as conn:
            conn.execute("UPDATE USERS SET familiarity_score = 0.5 WHERE user_id = ?", (user_id,))
            conn.commit()

        snapshot2 = state.create_snapshot(["m1"])
        packet2 = self.selector.select(snapshot2, opp)
        self.assertEqual(packet2.target.familiarity_band, "FRIEND")

    def test_02_session_context_and_stance_continuity(self):
        """GenerationPacket carries session details and committed stance continuity."""
        state = DialogueState("t_ctx", BOT_USER_ID, BOT_USERNAME)
        
        # Setup session
        m1 = _make_msg("m1", "userA", "alice", "hey", thread_id="t_ctx")
        state.apply_message(m1, 1)
        m2 = _make_msg("m2", "userB", "bob", "reply to alice", reply_to_msg_id="m1", reply_to_user_id="userA", thread_id="t_ctx")
        state.apply_message(m2, 2)

        # Set stance continuity inside stance state
        state.stance_state.commit(
            stance="PLAYFUL",
            speech_act="TEASE",
            intent_tag="tease_user",
            target_user_id="userA",
            session_id=m2.conversation_id,
            turn_id="turn_t",
            version=2
        )

        snapshot = state.create_snapshot(["m2"])
        res = self.resolver.resolve(snapshot, [m2])
        # Force session match for ownership resolver compatibility in test
        res = res.__class__(
            ownership="EVE",
            target_user_id=BOT_USER_ID,
            anchor_message_id="m2",
            session_id=m2.conversation_id,
            confidence="STRONG",
            evidence=res.evidence,
            continuation_of_eve_interaction=True
        )

        dec = self.policy.evaluate(res, snapshot)
        opp = self.policy.create_opportunity(dec, res, snapshot, ["m2"])

        packet = self.selector.select(snapshot, opp)
        self.assertIsNotNone(packet.active_session)
        self.assertEqual(packet.active_session.session_id, m2.conversation_id)
        self.assertEqual(packet.active_session.last_committed_stance, "PLAYFUL")
        self.assertEqual(packet.active_session.last_speech_act, "TEASE")

    def test_03_relevance_first_memories_selection(self):
        """Memories are filtered by subject keyword overlap."""
        user_id = "userA"
        profiles.get_or_create_user(user_id, "alice")
        
        # Add memories: one about exams, one about cats
        memories.add_claim_memory(user_id, "personal_fact", "exams", "alice hates exams", "NEW", 0.9)
        memories.add_claim_memory(user_id, "preference", "pets", "alice loves cats", "NEW", 0.8)

        state = DialogueState("t_ctx", BOT_USER_ID, BOT_USERNAME)
        # Message mentions exams but not cats
        m1 = _make_msg("m1", user_id, "alice", "@eve i am stressed about my exams", thread_id="t_ctx")
        state.apply_message(m1, 1)

        snapshot = state.create_snapshot(["m1"])
        res = self.resolver.resolve(snapshot, [m1])
        dec = self.policy.evaluate(res, snapshot)
        opp = self.policy.create_opportunity(dec, res, snapshot, ["m1"])

        packet = self.selector.select(snapshot, opp)
        
        # Confirm that only the exam memory was selected based on keywords
        self.assertEqual(len(packet.memories), 1)
        self.assertEqual(packet.memories[0].value, "alice hates exams")
        self.assertEqual(packet.memories[0].slot, "exams")


if __name__ == "__main__":
    unittest.main()
