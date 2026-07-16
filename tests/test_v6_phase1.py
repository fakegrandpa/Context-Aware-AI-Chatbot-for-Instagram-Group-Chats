"""
EVE V6 Phase 1 Unit and Concurrency Tests.
Validates live-room core, dialogue state mutation, room versioning, and async generation boundaries.
"""
from __future__ import annotations

import os
import sys
import sqlite3
import threading
import time
import uuid
import unittest
from datetime import datetime, timezone, timedelta
from typing import List, Optional
from unittest.mock import MagicMock, patch

# Adjust path to import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
config.DB_PATH = type(config.DB_PATH)(":memory:")  # Use in-memory DB

# Setup shared test database keep-alive
_KEEP_ALIVE_CONN: Optional[sqlite3.Connection] = None

def _get_test_connection():
    conn = sqlite3.connect("file:testdb?mode=memory&cache=shared", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn

import storage.database as db_module
db_module.get_connection = _get_test_connection

from storage.database import init_db
from models.message import NormalizedMessage
from models.scene import SceneSnapshot
from conversation.chat_actor import ChatActor
from conversation.chat_actor_registry import ChatActorRegistry
from conversation.dialogue_state import (
    DialogueState,
    DialogueSession,
    CONF_HARD,
    CONF_STRONG,
    CONF_PROVISIONAL,
    STATE_ACTIVE,
    STATE_FADING,
    STATE_CLOSED,
)
from storage import chat_state
from storage import messages as msg_store
from storage import eve_turns
from conversation.lanes import LaneManager
from conversation.fatigue import FatigueTracker
from conversation.mode_selector import ModeSelector
from voice.health import VoiceHealth

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


class Phase1TestSetup(unittest.TestCase):
    def setUp(self):
        """Create fresh schema for each test."""
        global _KEEP_ALIVE_CONN
        if _KEEP_ALIVE_CONN:
            _KEEP_ALIVE_CONN.close()
            _KEEP_ALIVE_CONN = None
        _KEEP_ALIVE_CONN = sqlite3.connect("file:testdb?mode=memory&cache=shared", uri=True, check_same_thread=False)
        init_db()
        config.BURST_WINDOW_MS = 50  # Set fast burst window for tests
        
        # Test-specific namespace to prevent duplicate ID pollution
        self.test_prefix = str(uuid.uuid4())[:8]
        
        self.cl = MagicMock()
        self.lane_manager = LaneManager(BOT_USER_ID)
        self.fatigue_tracker = FatigueTracker()
        self.mode_selector = ModeSelector()
        self.voice_health = VoiceHealth()
        self.voice_health.disable_permanently("tested")

    def tearDown(self):
        global _KEEP_ALIVE_CONN
        if _KEEP_ALIVE_CONN:
            _KEEP_ALIVE_CONN.close()
            _KEEP_ALIVE_CONN = None

    def make_msg(
        self,
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
        uniq_id = f"{self.test_prefix}_{message_id}"
        uniq_reply_to = f"{self.test_prefix}_{reply_to_msg_id}" if reply_to_msg_id else None
        uniq_thread = f"{self.test_prefix}_{thread_id}" if thread_id else None
        
        return _make_msg(
            message_id=uniq_id,
            sender_id=sender_id,
            sender_username=sender_username,
            text=text,
            is_viewer=is_viewer,
            reply_to_msg_id=uniq_reply_to,
            reply_to_user_id=reply_to_user_id,
            ts_offset_seconds=ts_offset_seconds,
            item_type=item_type,
            thread_id=uniq_thread,
        )


class TestPhase1Core(Phase1TestSetup):

    def test_01_monotonic_room_version(self):
        """Test 2: Real monotonic room version advances beyond 15."""
        thread_id = "t_mono"
        for i in range(25):
            msg = self.make_msg(f"msg_{i}", "user1", "atharv", f"text {i}", thread_id=thread_id)
            is_accepted, version = chat_state.accept_and_persist_message(msg)
            self.assertTrue(is_accepted)
            self.assertEqual(version, i + 1)
            
        uniq_thread = f"{self.test_prefix}_{thread_id}"
        self.assertEqual(chat_state.get_room_version(uniq_thread), 25)

    def test_02_duplicate_transport_dedup(self):
        """Test 3: Duplicate transport messages are ignored and do not increment version."""
        thread_id = "t_dup"
        msg1 = self.make_msg("msg_u", "user1", "atharv", "hello", thread_id=thread_id)
        is_accepted1, version1 = chat_state.accept_and_persist_message(msg1)
        self.assertTrue(is_accepted1)
        self.assertEqual(version1, 1)

        is_accepted2, version2 = chat_state.accept_and_persist_message(msg1)
        self.assertFalse(is_accepted2)
        self.assertEqual(version2, 1)  # Version stays same

        # Ensure only 1 row in MESSAGES
        conn = _get_test_connection()
        uniq_thread = f"{self.test_prefix}_{thread_id}"
        count = conn.execute("SELECT COUNT(*) FROM MESSAGES WHERE thread_id=?", (uniq_thread,)).fetchone()[0]
        self.assertEqual(count, 1)

    def test_03_interleaved_room_order(self):
        """Test 4: SceneSnapshot preserves exact ordered recent room events."""
        uniq_thread = f"{self.test_prefix}_t_ord"
        state = DialogueState(uniq_thread, BOT_USER_ID, BOT_USERNAME)
        events = [
            self.make_msg("a1", "atharv", "atharv", "hey", ts_offset_seconds=-10, thread_id="t_ord"),
            self.make_msg("v1", "ved", "ved", "what's up", ts_offset_seconds=-8, thread_id="t_ord"),
            self.make_msg("a2", "atharv", "atharv", "nothing", ts_offset_seconds=-6, thread_id="t_ord"),
            self.make_msg("r1", "rahul", "rahul", "yo", ts_offset_seconds=-4, thread_id="t_ord"),
            self.make_msg("v2", "ved", "ved", "same", ts_offset_seconds=-2, thread_id="t_ord"),
        ]
        for idx, m in enumerate(events):
            state.apply_message(m, idx + 1)

        snapshot = state.create_snapshot([events[-1].message_id])
        snapshot_msg_ids = [m.message_id for m in snapshot.recent_events]
        self.assertEqual(snapshot_msg_ids, [e.message_id for e in events])

    def test_04_native_reply_session_continuity(self):
        """Test 5: Native replies create a cohesive session without forcing general chat into it."""
        uniq_thread = f"{self.test_prefix}_t_rep"
        state = DialogueState(uniq_thread, BOT_USER_ID, BOT_USERNAME)
        
        m1 = self.make_msg("m1", "userA", "alice", "msg 1", thread_id="t_rep")
        state.apply_message(m1, 1)
        
        m2 = self.make_msg("m2", "userB", "bob", "reply to 1", reply_to_msg_id="m1", reply_to_user_id="userA", thread_id="t_rep")
        state.apply_message(m2, 2)
        
        m3 = self.make_msg("m3", "userA", "alice", "reply to 2", reply_to_msg_id="m2", reply_to_user_id="userB", thread_id="t_rep")
        state.apply_message(m3, 3)

        # Unrelated interleaved chatter
        m_unrelated = self.make_msg("m_un", "userC", "charlie", "random chat", thread_id="t_rep")
        state.apply_message(m_unrelated, 4)

        # Check sessions
        self.assertEqual(len(state.active_sessions), 1)
        session = list(state.active_sessions.values())[0]
        self.assertEqual(session.origin, "NATIVE_REPLY")
        self.assertEqual(session.confidence, CONF_HARD)
        self.assertIn(m1.message_id, session.recent_message_ids)
        self.assertIn(m2.message_id, session.recent_message_ids)
        self.assertIn(m3.message_id, session.recent_message_ids)
        self.assertNotIn(m_unrelated.message_id, session.recent_message_ids)

        self.assertIsNone(m_unrelated.conversation_id)

    def test_05_unknown_affiliation(self):
        """Test 6: Ambiguous/unaffiliated messages default to session_id = None."""
        uniq_thread = f"{self.test_prefix}_t_amb"
        state = DialogueState(uniq_thread, BOT_USER_ID, BOT_USERNAME)
        m1 = self.make_msg("m1", "userA", "alice", "hi", thread_id="t_amb")
        state.apply_message(m1, 1)
        m2 = self.make_msg("m2", "userB", "bob", "hello", thread_id="t_amb")
        state.apply_message(m2, 2)

        self.assertIsNone(m1.conversation_id)
        self.assertIsNone(m2.conversation_id)

    def test_06_same_user_in_multiple_sessions(self):
        """Test 7: A user can participate in multiple active sessions concurrently."""
        uniq_thread = f"{self.test_prefix}_t_mult"
        state = DialogueState(uniq_thread, BOT_USER_ID, BOT_USERNAME)
        
        # Session 1: Alice <-> Bob
        m1 = self.make_msg("s1_1", "userA", "alice", "session 1", thread_id="t_mult")
        state.apply_message(m1, 1)
        m2 = self.make_msg("s1_2", "userB", "bob", "reply to 1", reply_to_msg_id="s1_1", reply_to_user_id="userA", thread_id="t_mult")
        state.apply_message(m2, 2)

        # Session 2: Alice <-> Charlie
        m3 = self.make_msg("s2_1", "userC", "charlie", "session 2", thread_id="t_mult")
        state.apply_message(m3, 3)
        m4 = self.make_msg("s2_2", "userA", "alice", "reply to charlie", reply_to_msg_id="s2_1", reply_to_user_id="userC", thread_id="t_mult")
        state.apply_message(m4, 4)

        self.assertEqual(len(state.active_sessions), 2)
        sessions = list(state.active_sessions.values())
        
        self.assertTrue(any("userA" in s.participant_ids for s in sessions))
        session1 = next(s for s in sessions if m1.message_id in s.recent_message_ids)
        session2 = next(s for s in sessions if m3.message_id in s.recent_message_ids)
        
        self.assertIn("userA", session1.participant_ids)
        self.assertIn("userA", session2.participant_ids)

    def test_07_eve_engagement_is_per_chat(self):
        """Test 8: Eve engagement properties are scoped per thread/actor."""
        uniq_thread_a = f"{self.test_prefix}_thread_a"
        uniq_thread_b = f"{self.test_prefix}_thread_b"
        state_a = DialogueState(uniq_thread_a, BOT_USER_ID, BOT_USERNAME)
        state_b = DialogueState(uniq_thread_b, BOT_USER_ID, BOT_USERNAME)

        m_eve = self.make_msg("e1", BOT_USER_ID, BOT_USERNAME, "reply from eve", is_viewer=True, thread_id="thread_a")
        state_a.apply_message(m_eve, 1)

        self.assertEqual(state_a.eve_engagement.last_eve_turn_id, m_eve.message_id)
        self.assertEqual(state_a.eve_engagement.last_eve_turn_version, 1)
        
        self.assertIsNone(state_b.eve_engagement.last_eve_turn_id)
        self.assertIsNone(state_b.eve_engagement.last_eve_turn_version)

    def test_08_snapshot_immutability(self):
        """Test 9: SceneSnapshot copies are frozen and detached from active mutations."""
        uniq_thread = f"{self.test_prefix}_t_snap"
        state = DialogueState(uniq_thread, BOT_USER_ID, BOT_USERNAME)
        m1 = self.make_msg("m1", "userA", "alice", "msg 1", thread_id="t_snap")
        state.apply_message(m1, 1)

        snapshot = state.create_snapshot([m1.message_id])
        self.assertEqual(snapshot.room_version, 1)
        self.assertEqual(len(snapshot.recent_events), 1)

        # Mutate the live state
        m2 = self.make_msg("m2", "userB", "bob", "msg 2", thread_id="t_snap")
        state.apply_message(m2, 2)

        # Snapshot at version 1 must remain untouched
        self.assertEqual(snapshot.room_version, 1)
        self.assertEqual(len(snapshot.recent_events), 1)

    def test_09_actor_uniqueness_in_registry(self):
        """Test 10: Concurrent registry requests resolve to exactly one actor instance per thread."""
        registry = ChatActorRegistry()
        msg = self.make_msg("m1", "userA", "alice", "hi", thread_id="t_uniq")

        actors = []
        def fetch_actor():
            act = registry.route_message(
                msg, self.cl, BOT_USER_ID, BOT_USERNAME,
                self.lane_manager, self.fatigue_tracker, self.mode_selector, self.voice_health
            )
            actors.append(act)

        threads = [threading.Thread(target=fetch_actor) for _ in range(5)]
        for t in threads: t.start()
        for t in threads: t.join()

        # All threads must fetch the same unique actor
        first = actors[0]
        for a in actors:
            self.assertIs(a, first)
        
        registry.shutdown_all()

    def test_10_burst_does_not_delay_state_mutation(self):
        """Test 11: Ephemeral state mutations occur immediately at ingress, not deferred by burst timer."""
        uniq_thread = f"{self.test_prefix}_t_burst"
        actor = ChatActor(
            uniq_thread, self.cl, BOT_USER_ID, BOT_USERNAME,
            self.lane_manager, self.fatigue_tracker, self.mode_selector, self.voice_health
        )
        actor.start()
        
        msg = self.make_msg("m1", "userA", "alice", "immediate updates", thread_id="t_burst")
        actor.post_message(msg)
        
        # State must update immediately in mailbox loop - wait deterministically
        success = False
        for _ in range(40):
            if actor.dialogue_state.room_version == 1:
                success = True
                break
            time.sleep(0.05)
        self.assertTrue(success, f"Room version did not update to 1 in time (current: {actor.dialogue_state.room_version})")
        self.assertEqual(len(actor.dialogue_state.recent_events), 1)
        actor.stop()

    def test_11_room_level_burst_view(self):
        """Test 12: Coalesced burst window observes ordered multi-party timeline rather than sender-only blocks."""
        uniq_thread = f"{self.test_prefix}_t_rburst"
        actor = ChatActor(
            uniq_thread, self.cl, BOT_USER_ID, BOT_USERNAME,
            self.lane_manager, self.fatigue_tracker, self.mode_selector, self.voice_health
        )
        actor.start()

        # Interleaved rapid messages from different users within micro-window
        m1 = self.make_msg("m1", "userA", "alice", "ready?", ts_offset_seconds=0, thread_id="t_rburst")
        m2 = self.make_msg("m2", "userB", "bob", "go", ts_offset_seconds=0, thread_id="t_rburst")
        
        actor.post_message(m1)
        actor.post_message(m2)
        
        # Wait deterministically for version to reach 2
        success = False
        for _ in range(40):
            if actor.dialogue_state.room_version == 2:
                success = True
                break
            time.sleep(0.05)
        self.assertTrue(success, f"Room version did not update to 2 in time (current: {actor.dialogue_state.room_version})")
        self.assertEqual(len(actor.dialogue_state.recent_events), 2)
        actor.stop()


class TestPhase1Concurrency(Phase1TestSetup):

    @patch("intelligence.turn_composer.TurnComposer.compose")
    @patch("intelligence.turn_planner.generate_turn_plan")
    @patch("intelligence.response_generator.generate_from_context")
    @patch("instagram.sender.send_reply")
    def test_concurrency_advances_room_during_blocked_generation(
        self, mock_send_reply, mock_generate_context, mock_generate_plan, mock_compose
    ):
        """
        Test 1: Concurrency Invariant.
        Room state A triggers generation and blocks in executor.
        While blocked, B, C, D are sent.
        Assert that version increases and DialogueState updates to include B/C/D
        before generation for A is released.
        """
        uniq_thread = f"{self.test_prefix}_t_concur"
        actor = ChatActor(
            uniq_thread, self.cl, BOT_USER_ID, BOT_USERNAME,
            self.lane_manager, self.fatigue_tracker, self.mode_selector, self.voice_health
        )
        
        # Synchronization primitives for test coordination
        block_generation_event = threading.Event()
        generation_reached_block = threading.Event()
        
        def blocking_generator(*args, **kwargs):
            generation_reached_block.set()
            block_generation_event.wait(timeout=5.0)  # Block until released
            return "response to A", 0.1

        def blocking_composer(*args, **kwargs):
            generation_reached_block.set()
            block_generation_event.wait(timeout=5.0)
            from intelligence.turn_composer import TurnProposal
            return TurnProposal(
                action="REPLY",
                target_user_id="userA",
                anchor_message_id=self.test_prefix + "_msg_a",
                speech_act="CHAT",
                intent_tag="respond",
                stance="CASUAL",
                reply_text="response to A"
            )

        mock_generate_plan.return_value = MagicMock(speech_act="CHAT", intent="respond", stance="CASUAL")
        mock_generate_context.side_effect = blocking_generator
        mock_compose.side_effect = blocking_composer
        mock_send_reply.return_value = MagicMock(id="eve_reply_1")

        actor.start()

        # 1. Post message A to start generation and block it (mention Eve to resolve ownership)
        msg_a = self.make_msg("msg_a", "userA", "alice", "eve trigger", thread_id="t_concur")
        actor.post_message(msg_a)

        # Wait for generation A to enter block
        success = generation_reached_block.wait(timeout=4.0)
        self.assertTrue(success, "Generation thread did not start or block in time")

        # 2. While generation is blocked, post B, C, D
        msg_b = self.make_msg("msg_b", "userB", "bob", "b", thread_id="t_concur")
        msg_c = self.make_msg("msg_c", "userC", "charlie", "c", thread_id="t_concur")
        msg_d = self.make_msg("msg_d", "userB", "bob", "d", thread_id="t_concur")

        actor.post_message(msg_b)
        actor.post_message(msg_c)
        actor.post_message(msg_d)

        # Let the mailbox handle events (B, C, D)
        time.sleep(0.1)

        # ASSERTIONS while generation is still BLOCKED
        self.assertEqual(actor.dialogue_state.room_version, 4)  # msg_a (1) + msg_b (2) + msg_c (3) + msg_d (4)
        
        recent_ids = [m.message_id for m in actor.dialogue_state.recent_events]
        self.assertIn(msg_a.message_id, recent_ids)
        self.assertIn(msg_b.message_id, recent_ids)
        self.assertIn(msg_c.message_id, recent_ids)
        self.assertIn(msg_d.message_id, recent_ids)
        self.assertEqual(recent_ids, [msg_a.message_id, msg_b.message_id, msg_c.message_id, msg_d.message_id])

        # Release generation thread
        block_generation_event.set()
        time.sleep(0.1)  # Let generation complete event handle in mailbox

        # Assert response registration
        self.assertFalse(actor.current_generation_active)
        actor.stop()


if __name__ == "__main__":
    unittest.main()
