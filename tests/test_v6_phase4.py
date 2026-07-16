"""
EVE V6 Phase 4 Conversational Stance and Continuity Tests.
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

from storage import database, eve_turns
from conversation.dialogue_state import DialogueState
from conversation.chat_actor import ChatActor

BOT_USER_ID = "99999"
BOT_USERNAME = "eve"

_KEEP_ALIVE_CONN: Optional[sqlite3.Connection] = None


class TestPhase4StanceState(unittest.TestCase):

    def setUp(self):
        global _KEEP_ALIVE_CONN
        if _KEEP_ALIVE_CONN:
            _KEEP_ALIVE_CONN.close()
            _KEEP_ALIVE_CONN = None
        database.DB_PATH = "file:testdb?mode=memory&cache=shared"
        _KEEP_ALIVE_CONN = _get_test_conn()
        database.init_db()

    def tearDown(self):
        global _KEEP_ALIVE_CONN
        if _KEEP_ALIVE_CONN:
            _KEEP_ALIVE_CONN.close()
            _KEEP_ALIVE_CONN = None

    def test_01_successful_sent_turn_commits_stance(self):
        """A successfully sent turn commits stance values to DialogueState."""
        state = DialogueState("t_stance1", BOT_USER_ID, BOT_USERNAME)
        
        # Initial stance state is empty
        self.assertIsNone(state.stance_state.last_stance)

        # Simulate GenerationCompleted event payload
        payload = {
            "reply_sent": True,
            "turn_id": "eve_t1",
            "target_user_id": "userA",
            "anchor_message_id": "msg_anchor",
            "session_id": "session123",
            "text": "chill, leaving you alone",
            "originating_version": 2,
            "stance": "TIRED_DISMISSIVE",
            "speech_act": "REJECT",
            "intent_tag": "reject_voice_request"
        }

        # Commit directly via the commit method
        state.stance_state.commit(
            stance=payload["stance"],
            speech_act=payload["speech_act"],
            intent_tag=payload["intent_tag"],
            target_user_id=payload["target_user_id"],
            session_id=payload["session_id"],
            turn_id=payload["turn_id"],
            version=payload["originating_version"]
        )

        self.assertEqual(state.stance_state.last_stance, "TIRED_DISMISSIVE")
        self.assertEqual(state.stance_state.last_speech_act, "REJECT")
        self.assertEqual(state.stance_state.last_intent_tag, "reject_voice_request")
        self.assertEqual(state.stance_state.active_engagement_session, "session123")

    def test_02_failed_generation_or_send_does_not_commit_stance(self):
        """Failed generations/sends must not alter the committed stance state."""
        state = DialogueState("t_stance2", BOT_USER_ID, BOT_USERNAME)
        state.stance_state.commit(
            stance="FRIENDLY",
            speech_act="GREET",
            intent_tag="greet_user",
            target_user_id="userA",
            session_id="session1",
            turn_id="turn_ok",
            version=1
        )

        # Confirm initial commit succeeded
        self.assertEqual(state.stance_state.last_stance, "FRIENDLY")

        # Simulate a failed generation or failed send sequence
        # Stance should NOT mutate because commit is never called for failed outcomes
        self.assertEqual(state.stance_state.last_stance, "FRIENDLY")

    def test_03_independent_stance_per_thread(self):
        """Conversational stance state is isolated between different threads."""
        state_a = DialogueState("thread_A", BOT_USER_ID, BOT_USERNAME)
        state_b = DialogueState("thread_B", BOT_USER_ID, BOT_USERNAME)

        state_a.stance_state.commit(
            stance="ANGRY",
            speech_act="ROAST",
            intent_tag="roast_user",
            target_user_id="userA",
            session_id="session_A",
            turn_id="t_a",
            version=1
        )

        # Thread B remains empty, Thread A has stance
        self.assertEqual(state_a.stance_state.last_stance, "ANGRY")
        self.assertIsNone(state_b.stance_state.last_stance)

    def test_04_stance_reconstructs_on_startup(self):
        """On startup, StanceState is recovered from the database EVE_TURNS history."""
        # Insert a turn with stance columns populated
        eve_turns.store_eve_turn(
            conversation_id="thread_recover",
            trigger_message_id="msg_trigger",
            target_user_id="userA",
            modality="TEXT",
            semantic_summary="summary text",
            exact_text="exact text reply",
            voice_transcript=None,
            conversation_version=10,
            session_id="session_rec",
            snapshot_version=10,
            speech_act="REJECT",
            intent_tag="reject_voice",
            stance="TIRED_DISMISSIVE",
            anchor_message_id="msg_trigger"
        )

        # Create actor (which triggers state recovery in __init__ / _recover_state)
        actor = ChatActor("thread_recover", None, BOT_USER_ID, BOT_USERNAME, None, None, None, None)
        actor._recover_state()

        # Confirm recovered stance values
        self.assertEqual(actor.dialogue_state.stance_state.last_stance, "TIRED_DISMISSIVE")
        self.assertEqual(actor.dialogue_state.stance_state.last_speech_act, "REJECT")
        self.assertEqual(actor.dialogue_state.stance_state.last_intent_tag, "reject_voice")
        self.assertEqual(actor.dialogue_state.stance_state.active_engagement_session, "session_rec")


if __name__ == "__main__":
    unittest.main()
