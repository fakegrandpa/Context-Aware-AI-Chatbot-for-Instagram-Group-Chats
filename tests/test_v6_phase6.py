"""
EVE V6 Phase 6 TurnComposer Tests.
"""
from __future__ import annotations

import os
import sys
import json
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
from intelligence.address_resolver import AddressResolver
from intelligence.participation_policy import ParticipationPolicy
from intelligence.context_selector import ContextSelector
from intelligence.turn_composer import TurnComposer, TurnProposal

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


class TestPhase6TurnComposer(unittest.TestCase):

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
        self.composer = TurnComposer(BOT_USER_ID)

    def tearDown(self):
        global _KEEP_ALIVE_CONN
        if _KEEP_ALIVE_CONN:
            _KEEP_ALIVE_CONN.close()
            _KEEP_ALIVE_CONN = None

    @patch("intelligence.gemini_pool.generate_content")
    def test_01_single_gemini_call_replays_reply(self, mock_generate):
        """TurnComposer executes one call and returns structured REPLY proposal."""
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "action": "REPLY",
            "target_user_id": "userA",
            "anchor_message_id": "m1",
            "speech_act": "STATEMENT",
            "intent_tag": "reply_greetings",
            "stance": "FRIENDLY",
            "reply_text": "hey alice, what's up?",
            "continuity_marker": None
        })
        mock_generate.return_value = mock_response

        state = DialogueState("t_comp", BOT_USER_ID, BOT_USERNAME)
        m1 = _make_msg("m1", "userA", "alice", "@eve hello", thread_id="t_comp")
        state.apply_message(m1, 1)

        snapshot = state.create_snapshot(["m1"])
        res = self.resolver.resolve(snapshot, [m1])
        dec = self.policy.evaluate(res, snapshot)
        opp = self.policy.create_opportunity(dec, res, snapshot, ["m1"])
        packet = self.selector.select(snapshot, opp)

        proposal = self.composer.compose(packet)

        mock_generate.assert_called_once()
        self.assertEqual(proposal.action, "REPLY")
        self.assertEqual(proposal.reply_text, "hey alice, what's up?")
        self.assertEqual(proposal.stance, "FRIENDLY")

    @patch("intelligence.gemini_pool.generate_content")
    def test_02_required_ignore_override(self, mock_generate):
        """If model returns IGNORE on REQUIRED opportunity, composer overrides to REPLY."""
        mock_response = MagicMock()
        # Model returns IGNORE but participation mode in trigger will be REQUIRED
        mock_response.text = json.dumps({
            "action": "IGNORE",
            "target_user_id": "userA",
            "anchor_message_id": "m1",
            "speech_act": "STATEMENT",
            "intent_tag": "ignore_turn",
            "stance": "TIRED",
            "reply_text": None,
            "continuity_marker": None
        })
        mock_generate.return_value = mock_response

        state = DialogueState("t_comp", BOT_USER_ID, BOT_USERNAME)
        m1 = _make_msg("m1", "userA", "alice", "@eve hello", thread_id="t_comp")
        state.apply_message(m1, 1)

        snapshot = state.create_snapshot(["m1"])
        res = self.resolver.resolve(snapshot, [m1])
        dec = self.policy.evaluate(res, snapshot)
        # Ensure it is REQUIRED
        self.assertEqual(dec.mode, "REQUIRED")
        
        opp = self.policy.create_opportunity(dec, res, snapshot, ["m1"])
        packet = self.selector.select(snapshot, opp)

        proposal = self.composer.compose(packet)

        # Overridden to REPLY and has fallback text
        self.assertEqual(proposal.action, "REPLY")
        self.assertEqual(proposal.reply_text, "hmm?")


if __name__ == "__main__":
    unittest.main()
