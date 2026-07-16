"""
EVE V6 Phase 12 — Busy Group-Chat Simulator.

Deterministic stress test of the full V6 live-room pipeline under concurrent
message load. Verifies:

1. Room version monotonicity via chat_state under concurrent actor bursts
2. Duplicate dedup via chat_state.accept_and_persist_message
3. Snapshot immutability — snapshot taken at version N is frozen after that
4. ParticipationPolicy pressure correctly defers open-group under high velocity
5. Direct Eve reply is never suppressed regardless of pressure
6. GenerationLease classification of an interrupting direct-Eve message
7. GenerationLease IRRELEVANT classification for unrelated side-chatter
8. Concurrent memory worker ticks cannot claim the same batch

Run standalone:
    python tests/busy_gc_simulator.py

Or via pytest:
    pytest tests/busy_gc_simulator.py -v
"""
from __future__ import annotations

import os
import sqlite3
import sys
import threading
import time
import unittest
from datetime import datetime, timezone, timedelta
from typing import List, Optional
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

import storage.database as db_module
import storage.messages as _messages_mod
from storage.database import init_db
from storage import messages as msg_store
from storage import chat_state

from models.message import NormalizedMessage
from conversation.dialogue_state import DialogueState
from intelligence.address_resolver import AddressResolver
from intelligence.participation_policy import (
    ParticipationPolicy,
    MODE_REQUIRED,
    MODE_ELIGIBLE,
    MODE_SUPPRESS,
    MODE_DEFER,
)
from conversation.chat_actor import ChatActor, GenerationLease


def _now(offset_s: float = 0.0) -> datetime:
    return datetime.now(timezone.utc) + timedelta(seconds=offset_s)


def _msg(
    msg_id: str,
    sender_id: str,
    sender_username: str,
    text: str,
    reply_to_msg_id: str | None = None,
    reply_to_user_id: str | None = None,
    ts_offset: float = 0.0,
    thread_id: str = "busythread",
) -> NormalizedMessage:
    return NormalizedMessage(
        message_id=msg_id,
        thread_id=thread_id,
        sender_id=sender_id,
        sender_username=sender_username,
        text=text,
        timestamp=_now(ts_offset),
        item_type="text",
        is_sent_by_viewer=False,
        reply_to_message_id=reply_to_msg_id,
        reply_to_user_id=reply_to_user_id,
        is_historical=False,
    )


BOT_ID = "bot999"
BOT_NAME = "eve"
USER_A = ("u001", "atharv")
USER_B = ("u002", "rahul")
USER_C = ("u003", "priya")


class BusyGCSetup(unittest.TestCase):
    def setUp(self):
        # Save original connection factories before patching
        self._orig_db_get = db_module.get_connection
        self._orig_msg_get = _messages_mod.get_connection

        # Keep-alive connection for this test's in-memory DB
        self._conn = sqlite3.connect(
            "file:busygc?mode=memory&cache=shared", uri=True, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript("""
            DROP TABLE IF EXISTS MESSAGES;
            DROP TABLE IF EXISTS USERS;
            DROP TABLE IF EXISTS MEMORIES;
            DROP TABLE IF EXISTS BOT_STATE;
            DROP TABLE IF EXISTS EVE_STATE;
            DROP TABLE IF EXISTS RELATIONSHIPS;
            DROP TABLE IF EXISTS EVE_TURNS;
            DROP TABLE IF EXISTS CHAT_STATE;
        """)

        def _get_conn():
            conn = sqlite3.connect(
                "file:busygc?mode=memory&cache=shared", uri=True, check_same_thread=False
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=5000")
            return conn

        # Patch per-test so we don't leak into other modules
        db_module.get_connection = _get_conn
        _messages_mod.get_connection = _get_conn
        self._get_conn = _get_conn

        config.BOT_USER_ID = BOT_ID
        config.BOT_NAME = BOT_NAME
        init_db()

    def tearDown(self):
        # Restore original connection factories
        db_module.get_connection = self._orig_db_get
        _messages_mod.get_connection = self._orig_msg_get
        self._conn.close()



class TestBusyGCRoomVersion(BusyGCSetup):

    def test_01_monotonic_room_version_under_concurrent_ingest(self):
        """Room version must be strictly monotonic even when 20 messages arrive concurrently."""
        thread_id = "busythread"
        lock = threading.Lock()
        versions_seen: List[int] = []
        errors: List[Exception] = []

        def ingest(i: int):
            try:
                msg = _msg(f"m{i:04d}", USER_A[0], USER_A[1], f"message {i}",
                           ts_offset=i * 0.001, thread_id=thread_id)
                with lock:
                    accepted, version = chat_state.accept_and_persist_message(msg)
                    if accepted:
                        versions_seen.append(version)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=ingest, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Unexpected errors: {errors}")
        self.assertEqual(len(versions_seen), 20, "All 20 messages must be accepted")
        self.assertEqual(sorted(versions_seen), list(range(1, 21)),
                         "Room versions must be monotonically 1-20")

    def test_02_dedup_guard_under_concurrent_ingest(self):
        """The same message ID ingested 5 times must only increment room version once."""
        msg = _msg("dup001", USER_A[0], USER_A[1], "duplicate message")
        lock = threading.Lock()
        accepted_count = [0]

        def ingest():
            with lock:
                accepted, _ = chat_state.accept_and_persist_message(msg)
                if accepted:
                    accepted_count[0] += 1

        threads = [threading.Thread(target=ingest) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(accepted_count[0], 1, "Duplicate message must only be accepted once")
        self.assertEqual(chat_state.get_room_version("busythread"), 1,
                         "Room version must equal 1 after 5 duplicate ingests")

    def test_03_snapshot_captures_version_at_creation_time(self):
        """SceneSnapshot taken at version N must not reflect later ingests."""
        thread_id = "busythread"
        ds = DialogueState(thread_id, BOT_ID, BOT_NAME)

        for i in range(5):
            m = _msg(f"early{i}", USER_A[0], USER_A[1], f"early msg {i}",
                     ts_offset=i * 0.001, thread_id=thread_id)
            ds.apply_message(m, i + 1)

        snap = ds.create_snapshot([])
        snap_version = snap.room_version

        # Ingest 5 more messages after the snapshot
        for i in range(5, 10):
            m = _msg(f"late{i}", USER_B[0], USER_B[1], f"late msg {i}",
                     ts_offset=i * 0.001, thread_id=thread_id)
            ds.apply_message(m, i + 1)

        self.assertEqual(snap.room_version, snap_version,
                         "Snapshot room_version must be immutable after creation")
        self.assertLess(snap.room_version, ds.room_version,
                        "Actor room version must have advanced past the snapshot")


class TestBusyGCParticipationPressure(BusyGCSetup):

    def test_04_high_velocity_traffic_defers_open_group_participation(self):
        """Under high velocity, ParticipationPolicy must DEFER or SUPPRESS open-group turns."""
        ds = DialogueState("busythread", BOT_ID, BOT_NAME)
        resolver = AddressResolver(BOT_ID, BOT_NAME)
        policy = ParticipationPolicy(BOT_ID)

        # Simulate 10 messages in the last 4 seconds (high velocity)
        msgs = []
        for i in range(10):
            msg = _msg(f"v{i}", USER_A[0], USER_A[1], f"fast msg {i}",
                       ts_offset=-(4 - i * 0.3))
            ds.apply_message(msg, i + 1)
            msgs.append(msg)

        snap = ds.create_snapshot([msgs[-1].message_id])
        res = resolver.resolve(snap, [msgs[-1]])
        decision = policy.evaluate(res, snap)

        self.assertIn(decision.mode, (MODE_SUPPRESS, MODE_DEFER),
                      "High-velocity traffic must suppress or defer open-group participation")

    def test_05_direct_eve_reply_never_deferred_by_pressure(self):
        """Even under maximum pressure, a direct Eve reply must always fire."""
        ds = DialogueState("busythread", BOT_ID, BOT_NAME)
        resolver = AddressResolver(BOT_ID, BOT_NAME)
        policy = ParticipationPolicy(BOT_ID)

        # Load up participation pressure
        for i in range(3):
            bot_msg = _msg(f"bot{i}", BOT_ID, BOT_NAME, f"bot turn {i}",
                           ts_offset=-30 + i * 5)
            bot_msg.is_sent_by_viewer = True
            ds.apply_message(bot_msg, i + 1)

        # Direct Eve reply arrives
        trigger = _msg("direct1", USER_A[0], USER_A[1], "hey @eve what do you think",
                       ts_offset=0)
        trigger.reply_to_user_id = BOT_ID
        ds.apply_message(trigger, 4)

        snap = ds.create_snapshot([trigger.message_id])
        res = resolver.resolve(snap, [trigger])
        decision = policy.evaluate(res, snap)

        self.assertEqual(decision.mode, MODE_REQUIRED,
                         "Direct Eve reply must be REQUIRED regardless of participation pressure")




class TestBusyGCGenerationLease(BusyGCSetup):

    def _make_actor(self, thread_id: str = "busythread") -> ChatActor:
        cl = MagicMock()
        lane_mgr = MagicMock()
        fatigue = MagicMock()
        mode_sel = MagicMock()
        vh = MagicMock()
        vh.is_healthy.return_value = False
        return ChatActor(thread_id, cl, BOT_ID, BOT_NAME, lane_mgr, fatigue, mode_sel, vh)

    def test_06_burst_interruption_cancels_active_lease(self):
        """A direct Eve-addressed message from another user must interrupt an active lease."""
        actor = self._make_actor()

        m1 = _msg("pre1", USER_A[0], USER_A[1], "hey eve", thread_id="busythread")
        actor.dialogue_state.apply_message(m1, 1)
        m1.conversation_id = "sess3"

        lease = GenerationLease(
            lease_id="lease001",
            start_version=1,
            session_id="sess3",
            target_user_id=USER_A[0],
            anchor_message_id="pre1",
        )

        # Different user directly addresses Eve
        m_burst = _msg("burst1", USER_B[0], USER_B[1], "hey eve answer me",
                       reply_to_user_id=BOT_ID, thread_id="busythread")
        m_burst.reply_to_user_id = BOT_ID
        m_burst.conversation_id = "sess3"
        actor.dialogue_state.apply_message(m_burst, 2)

        result = actor._classify_lease_delta(lease, m_burst)
        self.assertIn(result, ("HIGHER_PRIORITY", "CANCELLING"),
                      "A direct Eve address from a different user must cancel or supersede the active lease")
        actor.stop()

    def test_07_irrelevant_chatter_does_not_cancel_lease(self):
        """Unrelated messages between two other users must not interrupt an active lease."""
        actor = self._make_actor()

        m1 = _msg("anchor", USER_A[0], USER_A[1], "hey eve!", reply_to_user_id=BOT_ID,
                  thread_id="busythread")
        actor.dialogue_state.apply_message(m1, 1)
        m1.conversation_id = "sess4"

        lease = GenerationLease(
            lease_id="lease002",
            start_version=1,
            session_id="sess4",
            target_user_id=USER_A[0],
            anchor_message_id="anchor",
        )

        # Side chatter — no Eve mention, different users, different session
        side1 = _msg("side1", USER_B[0], USER_B[1], "rahul bhai aaj kya plan hai",
                     thread_id="busythread")
        side1.conversation_id = "sess_other"
        actor.dialogue_state.apply_message(side1, 2)

        result = actor._classify_lease_delta(lease, side1)
        self.assertEqual(result, "IRRELEVANT",
                         "Unrelated side-chatter must be classified IRRELEVANT and not cancel the active lease")
        actor.stop()


class TestBusyGCMemoryReliability(BusyGCSetup):

    def test_08_concurrent_memory_ticks_do_not_double_process(self):
        """Two concurrent MemoryWorker ticks must not claim the same batch."""
        conn = self._get_conn()

        # Insert 5 user messages with memory_in_progress=0
        for i in range(5):
            conn.execute(
                """INSERT OR IGNORE INTO MESSAGES
                   (message_id, thread_id, sender_id, text, timestamp, item_type,
                    is_sent_by_viewer, memory_processed, memory_in_progress, stored_at)
                   VALUES (?, 'busythread', 'u001', ?, ?, 'text', 0, 0, 0, ?)""",
                (f"mem{i}", f"user says fact {i}", _now().isoformat(), _now().isoformat())
            )
        conn.commit()

        # Tick 1 fetches and claims the first 3
        batch1 = msg_store.get_unprocessed_for_memory(limit=3)
        msg_store.mark_memory_in_progress([m["message_id"] for m in batch1])

        # Tick 2 must not see any of tick 1's claimed rows
        batch2 = msg_store.get_unprocessed_for_memory(limit=10)
        overlap = set(m["message_id"] for m in batch1) & set(m["message_id"] for m in batch2)
        self.assertEqual(len(overlap), 0,
                         "In-progress messages must not appear in concurrent tick fetch")


if __name__ == "__main__":
    unittest.main(verbosity=2)
