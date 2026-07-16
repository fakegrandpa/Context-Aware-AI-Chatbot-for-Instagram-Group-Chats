"""
EVE V6 Phase 7 Generation Lease and Revalidation Tests.
"""
from __future__ import annotations

import os
import sys
import sqlite3
import unittest
from datetime import datetime, timezone, timedelta
from typing import List, Optional
from unittest.mock import MagicMock, patch

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

from storage import database
from conversation.dialogue_state import DialogueState
from conversation.chat_actor import ChatActor, GenerationLease

BOT_USER_ID = "99999"
BOT_USERNAME = "eve"

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


class TestPhase7GenerationLease(unittest.TestCase):

    def setUp(self):
        global _KEEP_ALIVE_CONN
        if _KEEP_ALIVE_CONN:
            _KEEP_ALIVE_CONN.close()
            _KEEP_ALIVE_CONN = None
        database.DB_PATH = "file:testdb?mode=memory&cache=shared"
        _KEEP_ALIVE_CONN = _get_test_conn()
        database.init_db()

        self.cl = MagicMock()
        self.lane_manager = MagicMock()
        self.fatigue_tracker = MagicMock()
        self.mode_selector = MagicMock()
        self.voice_health = MagicMock()
        self.voice_health.is_healthy.return_value = False

        self.actor = ChatActor(
            "t_lease", self.cl, BOT_USER_ID, BOT_USERNAME,
            self.lane_manager, self.fatigue_tracker, self.mode_selector, self.voice_health
        )

    def tearDown(self):
        global _KEEP_ALIVE_CONN
        if _KEEP_ALIVE_CONN:
            _KEEP_ALIVE_CONN.close()
            _KEEP_ALIVE_CONN = None
        self.actor.stop()

    def test_01_lease_creation(self):
        """Lease is registered correctly on creation."""
        lease = GenerationLease(
            lease_id="l1",
            start_version=1,
            session_id="s1",
            target_user_id="userA",
            anchor_message_id="m1"
        )
        self.assertEqual(lease.status, "ACTIVE")
        self.assertEqual(lease.lease_id, "l1")

    def test_02_classify_irrelevant_compatible(self):
        """New unrelated or sender-aligned messages are classified as IRRELEVANT or COMPATIBLE."""
        m1 = _make_msg("m1", "userA", "alice", "hey eve", thread_id="t_lease")
        self.actor.dialogue_state.apply_message(m1, 1)
        m1.conversation_id = "s1"

        lease = GenerationLease(
            lease_id="l1",
            start_version=1,
            session_id="s1",
            target_user_id="userA",
            anchor_message_id="m1"
        )

        # 1. Irrelevant message (different thread/session)
        m_irrel = _make_msg("m2", "userB", "bob", "random stuff", thread_id="other_thread")
        res = self.actor._classify_lease_delta(lease, m_irrel)
        self.assertEqual(res, "IRRELEVANT")

        # 2. Compatible message (same sender continuing thought in same session)
        m_comp = _make_msg("m3", "userA", "alice", "what are you doing?", thread_id="t_lease", reply_to_msg_id="m1", reply_to_user_id="userA")
        self.actor.dialogue_state.apply_message(m_comp, 2)
        m_comp.conversation_id = "s1"
        res2 = self.actor._classify_lease_delta(lease, m_comp)
        self.assertEqual(res2, "COMPATIBLE")

    def test_03_classify_cancelling_interrupt(self):
        """Cancelling keywords trigger CANCELLING, explicit bot address triggers HIGHER_PRIORITY."""
        m1 = _make_msg("m1", "userA", "alice", "hey eve", thread_id="t_lease")
        self.actor.dialogue_state.apply_message(m1, 1)
        m1.conversation_id = "s1"

        lease = GenerationLease(
            lease_id="l1",
            start_version=1,
            session_id="s1",
            target_user_id="userA",
            anchor_message_id="m1"
        )

        # 1. Cancelling message
        m_cancel = _make_msg("m2", "userA", "alice", "nevermind actually", thread_id="t_lease", reply_to_msg_id="m1", reply_to_user_id="userA")
        self.actor.dialogue_state.apply_message(m_cancel, 2)
        m_cancel.conversation_id = "s1"
        res = self.actor._classify_lease_delta(lease, m_cancel)
        self.assertEqual(res, "CANCELLING")

        # 2. Higher priority interrupt (direct address to Eve)
        m_interrupt = _make_msg("m3", "userB", "bob", "@eve answer me now", thread_id="t_lease", reply_to_msg_id="m1", reply_to_user_id="userA")
        m_interrupt.reply_to_user_id = BOT_USER_ID
        self.actor.dialogue_state.apply_message(m_interrupt, 3)
        m_interrupt.conversation_id = "s1"
        res2 = self.actor._classify_lease_delta(lease, m_interrupt)
        self.assertEqual(res2, "HIGHER_PRIORITY")


if __name__ == "__main__":
    unittest.main()
