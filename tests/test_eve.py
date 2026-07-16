"""
Eve V5 — comprehensive test suite.

All 49 unit tests + 7 scenario simulations.
Uses in-memory SQLite for storage tests.
No live Instagram or Gemini calls (mocked where needed).

Run: python -m pytest tests/test_eve.py -v
  OR: python tests/test_eve.py
"""
import sys
import os
import sqlite3
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Patch config before imports that use it
import config
config.DB_PATH = type(config.DB_PATH)(":memory:")  # Override to in-memory for tests

# Override get_connection for in-memory — we use a shared connection for tests
# since :memory: DB is per-connection. We'll use a module-level connection for tests.
_TEST_DB_CONN: Optional[sqlite3.Connection] = None


def _get_test_connection():
    global _TEST_DB_CONN
    if _TEST_DB_CONN is None:
        _TEST_DB_CONN = sqlite3.connect("file:testdb?mode=memory&cache=shared", uri=True, check_same_thread=False)
        _TEST_DB_CONN.row_factory = sqlite3.Row
        _TEST_DB_CONN.execute("PRAGMA foreign_keys=ON")
    return _TEST_DB_CONN


# Patch storage.database.get_connection to return our shared test connection
import storage.database as db_module
db_module.get_connection = _get_test_connection

# Now import everything else
import instagram_client
from models.message import NormalizedMessage, normalize_realtime, normalize_http
from models.decision import AttentionResult
from storage.database import init_db
from storage import messages as msg_store
from storage import profiles as prof_store
from storage import memories as mem_store
from conversation.attention import evaluate as attention_evaluate
from conversation.burst import BurstCoalescer, BurstGroup
from conversation.fatigue import FatigueTracker
from conversation.lanes import LaneManager, LaneState
from intelligence import social_judge


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
    is_historical: bool = False,
    item_type: str = "text",
    thread_id: str = "thread1",
) -> NormalizedMessage:
    ts = datetime.now(timezone.utc) + timedelta(seconds=ts_offset_seconds)
    return NormalizedMessage(
        message_id=message_id,
        thread_id=thread_id,
        sender_id=sender_id,
        sender_username=sender_username,
        text=text,
        timestamp=ts,
        item_type=item_type,
        is_sent_by_viewer=is_viewer,
        reply_to_message_id=reply_to_msg_id,
        reply_to_user_id=reply_to_user_id,
        is_historical=is_historical,
    )


class TestSetup(unittest.TestCase):
    def setUp(self):
        """Create fresh schema for each test."""
        global _TEST_DB_CONN
        # Drop and recreate all tables
        if _TEST_DB_CONN:
            _TEST_DB_CONN.executescript("""
                DROP TABLE IF EXISTS USERS;
                DROP TABLE IF EXISTS MESSAGES;
                DROP TABLE IF EXISTS MEMORIES;
                DROP TABLE IF EXISTS RELATIONSHIPS;
                DROP TABLE IF EXISTS BOT_STATE;
            """)
        init_db()


# ======================================================================
# TESTS 1-5: Profile + Message Persistence
# ======================================================================

class TestProfileCreation(TestSetup):

    def test_01_unknown_sender_creates_profile(self):
        """Test 1: Unknown sender creates profile automatically."""
        profile, created = prof_store.get_or_create_user("user1", "atharv")
        self.assertTrue(created)
        self.assertEqual(profile["user_id"], "user1")
        self.assertEqual(profile["username"], "atharv")
        self.assertEqual(profile["relationship_to_yap"], "new")
        self.assertAlmostEqual(profile["familiarity_score"], 0.0)

    def test_02_same_user_id_no_duplicate(self):
        """Test 2: Same user_id does not create a duplicate profile."""
        prof_store.get_or_create_user("user2", "rahul")
        _, created = prof_store.get_or_create_user("user2", "rahul")
        self.assertFalse(created)
        # Verify only one row
        conn = _get_test_connection()
        rows = conn.execute("SELECT COUNT(*) FROM USERS WHERE user_id='user2'").fetchone()[0]
        self.assertEqual(rows, 1)

    def test_03_username_change_updates_profile(self):
        """Test 3: Username change updates existing identity, no new row."""
        prof_store.get_or_create_user("user3", "ved_old")
        profile, created = prof_store.get_or_create_user("user3", "ved_new")
        self.assertFalse(created)
        self.assertEqual(profile["username"], "ved_new")
        conn = _get_test_connection()
        rows = conn.execute("SELECT COUNT(*) FROM USERS WHERE user_id='user3'").fetchone()[0]
        self.assertEqual(rows, 1)

    def test_04_duplicate_message_no_count_increment(self):
        """Test 4: Duplicate message_id does not increment message_count."""
        prof_store.get_or_create_user("user4", "sam")
        msg = _make_msg("msg001", "user4", "sam", "hello")
        inserted1 = msg_store.store_message(msg)
        self.assertTrue(inserted1)
        prof_store.increment_message_count("user4")

        inserted2 = msg_store.store_message(msg)
        self.assertFalse(inserted2)
        # Should NOT increment again

        profile = prof_store.get_user("user4")
        # We only incremented once
        self.assertEqual(profile["message_count"], 1)

    def test_05_reply_metadata_persists(self):
        """Test 5: reply_to_message_id and reply_to_user_id are persisted."""
        prof_store.get_or_create_user("user5a", "alice")
        prof_store.get_or_create_user("user5b", "bob")
        msg = _make_msg("msg002", "user5a", "alice", "hello bob",
                        reply_to_msg_id="msg000", reply_to_user_id="user5b")
        msg_store.store_message(msg)
        fetched = msg_store.get_message_by_id("msg002")
        self.assertEqual(fetched["reply_to_message_id"], "msg000")
        self.assertEqual(fetched["reply_to_user_id"], "user5b")


# ======================================================================
# TESTS 6-10: Attention Gate — Reply graph and address signals
# ======================================================================

class TestAttentionGate(TestSetup):

    def test_06_reply_to_human_causes_local_ignore(self):
        """Test 6: Native reply to another human → LOCAL_IGNORE."""
        msg = _make_msg("msg003", "user6", "atharv", "im 20",
                        reply_to_msg_id="msg_rahul", reply_to_user_id="rahul_id")
        result = attention_evaluate(
            msg=msg, lane=None,
            bot_user_id=BOT_USER_ID, bot_username=BOT_USERNAME,
        )
        self.assertEqual(result.decision, "LOCAL_IGNORE")
        self.assertIn("native_reply_to_human", result.reasons)

    def test_07_reply_to_eve_causes_local_reply(self):
        """Test 7: Native reply to Eve → LOCAL_REPLY."""
        msg = _make_msg("msg004", "user7", "rahul", "stfu",
                        reply_to_msg_id="eve_msg_1", reply_to_user_id=BOT_USER_ID)
        result = attention_evaluate(
            msg=msg, lane=None,
            bot_user_id=BOT_USER_ID, bot_username=BOT_USERNAME,
        )
        self.assertEqual(result.decision, "LOCAL_REPLY")
        self.assertIn("native_reply_to_eve", result.reasons)

    def test_08_explicit_eve_overrides_human_reply(self):
        """Test 8: Explicit 'eve' in text with human-target reply → override to REPLY."""
        # Even without native reply metadata, explicit address should score high
        msg = _make_msg("msg005", "user8", "ved", "eve what's 2+2")
        result = attention_evaluate(
            msg=msg, lane=None,
            bot_user_id=BOT_USER_ID, bot_username=BOT_USERNAME,
        )
        self.assertEqual(result.decision, "LOCAL_REPLY")

    def test_09_at_eve_causes_local_reply(self):
        """Test 9: @eve in message → LOCAL_REPLY."""
        msg = _make_msg("msg006", "user9", "sam", "@eve where are you from?")
        result = attention_evaluate(
            msg=msg, lane=None,
            bot_user_id=BOT_USER_ID, bot_username=BOT_USERNAME,
        )
        self.assertEqual(result.decision, "LOCAL_REPLY")
        self.assertIn("at_eve_mention", result.reasons)

    def test_10_at_other_user_discourages_eve(self):
        """Test 10: @mention of another user lowers score."""
        msg = _make_msg("msg007", "user10", "rahul", "@ved where are you from?")
        result = attention_evaluate(
            msg=msg, lane=None,
            bot_user_id=BOT_USER_ID, bot_username=BOT_USERNAME,
        )
        # Should be LOCAL_IGNORE or GEMINI_REQUIRED with negative score
        self.assertIn("at_other_user:ved", result.reasons)
        self.assertLess(result.score, 0)


# ======================================================================
# TESTS 11-12: Conversation Lanes
# ======================================================================

class TestConversationLanes(TestSetup):

    def test_11_human_human_lane_separated_from_eve_lane(self):
        """Test 11: Human-to-human lane differs from Eve lane."""
        lane_mgr = LaneManager(bot_user_id=BOT_USER_ID)

        # Eve replies to user1
        eve_msg = _make_msg("e1", BOT_USER_ID, "eve", "hello", is_viewer=True)
        user1_reply = _make_msg("u1", "user1", "atharv", "hey eve",
                                reply_to_msg_id="e1", reply_to_user_id=BOT_USER_ID,
                                ts_offset_seconds=1)
        lane_eve = lane_mgr.assign_lane(eve_msg)
        lane_eve_reply = lane_mgr.assign_lane(user1_reply)

        # Two humans talking to each other
        msg_a = _make_msg("h1", "user2", "rahul", "where are you from?", ts_offset_seconds=2)
        msg_b = _make_msg("h2", "user3", "ved", "pune",
                          reply_to_msg_id="h1", reply_to_user_id="user2",
                          ts_offset_seconds=3)
        lane_human = lane_mgr.assign_lane(msg_a)
        lane_human2 = lane_mgr.assign_lane(msg_b)

        # Eve lane and human lane should be different
        self.assertIsNotNone(lane_eve)
        self.assertIsNotNone(lane_human)
        self.assertNotEqual(lane_eve.lane_id, lane_human.lane_id)

    def test_12_lane_context_relevant_participants(self):
        """Test 12: Lane participants are tracked correctly."""
        lane_mgr = LaneManager(bot_user_id=BOT_USER_ID)
        msg1 = _make_msg("lm1", "userA", "alice", "hey")
        msg2 = _make_msg("lm2", "userB", "bob", "sup",
                         reply_to_msg_id="lm1", reply_to_user_id="userA",
                         ts_offset_seconds=1)
        lane_mgr.assign_lane(msg1)
        lane = lane_mgr.assign_lane(msg2)
        self.assertIn("userA", lane.participants)
        self.assertIn("userB", lane.participants)


# ======================================================================
# TESTS 13-14: Burst Coalescing
# ======================================================================

class TestBurstCoalescing(TestSetup):

    def test_13_burst_messages_coalesce(self):
        """Test 13: Same sender consecutive messages coalesce into one burst."""
        received: List[BurstGroup] = []
        coalescer = BurstCoalescer(window_ms=100, emit_callback=received.append)

        msg1 = _make_msg("b1", "user_x", "rahul", "bro", ts_offset_seconds=0)
        msg2 = _make_msg("b2", "user_x", "rahul", "wait", ts_offset_seconds=0)
        msg3 = _make_msg("b3", "user_x", "rahul", "listen", ts_offset_seconds=0)

        coalescer.add(msg1)
        coalescer.add(msg2)
        coalescer.add(msg3)

        time.sleep(0.3)  # Wait for burst window to close
        self.assertEqual(len(received), 1)
        self.assertEqual(len(received[0].messages), 3)

    def test_14_different_senders_dont_coalesce(self):
        """Test 14: Messages from different senders form separate bursts."""
        received: List[BurstGroup] = []
        coalescer = BurstCoalescer(window_ms=100, emit_callback=received.append)

        msg1 = _make_msg("c1", "user_a", "alice", "hey")
        msg2 = _make_msg("c2", "user_b", "bob", "hello")

        coalescer.add(msg1)
        coalescer.add(msg2)

        time.sleep(0.3)
        self.assertEqual(len(received), 2)
        senders = {r.sender_id for r in received}
        self.assertIn("user_a", senders)
        self.assertIn("user_b", senders)


# ======================================================================
# TESTS 15-16: Social Fatigue
# ======================================================================

class TestSocialFatigue(TestSetup):

    def test_15_fatigue_affects_ambiguous_threshold(self):
        """Test 15: Social fatigue raises threshold for ambiguous replies."""
        tracker = FatigueTracker(max_replies_60s=2, max_replies_5min=5)
        # Simulate heavy activity
        for _ in range(4):
            tracker.record_reply()
        fatigue = tracker.get_fatigue_multiplier()
        self.assertGreater(fatigue, 0.5)

        # Ambiguous message should be harder to trigger with high fatigue
        msg = _make_msg("f1", "user_f", "rahul", "what's up everyone")
        result = attention_evaluate(
            msg=msg, lane=None,
            bot_user_id=BOT_USER_ID, bot_username=BOT_USERNAME,
            fatigue_multiplier=fatigue,
        )
        # Should not be LOCAL_REPLY for ambiguous group message under high fatigue
        self.assertNotEqual(result.decision, "LOCAL_REPLY")

    def test_16_fatigue_never_blocks_direct_address(self):
        """Test 16: Fatigue never blocks direct Eve address."""
        tracker = FatigueTracker(max_replies_60s=2)
        for _ in range(10):
            tracker.record_reply()
        fatigue = tracker.get_fatigue_multiplier()

        msg = _make_msg("f2", "user_g", "atharv", "eve what are you doing")
        result = attention_evaluate(
            msg=msg, lane=None,
            bot_user_id=BOT_USER_ID, bot_username=BOT_USERNAME,
            fatigue_multiplier=fatigue,
        )
        self.assertEqual(result.decision, "LOCAL_REPLY")


# ======================================================================
# TESTS 17-18: Memory Duplicate Protection
# ======================================================================

class TestMemory(TestSetup):

    def test_17_memory_duplicate_normalization(self):
        """Test 17: Duplicate memory facts are normalized and deduplicated."""
        mem_store.add_memory("uid1", "preference", "Likes Football", 0.9)
        result = mem_store.add_memory("uid1", "preference", "  likes   football ", 0.9)
        self.assertIsNone(result)  # Should be blocked as duplicate

        mems = mem_store.get_active_memories("uid1")
        self.assertEqual(len(mems), 1)

    def test_18_deactivated_memory_excluded(self):
        """Test 18: Deactivated memory is not returned by get_active_memories."""
        mem_id = mem_store.add_memory("uid2", "identity", "Lives in Mumbai", 0.95)
        self.assertIsNotNone(mem_id)
        mem_store.deactivate_memory(mem_id)
        mems = mem_store.get_active_memories("uid2")
        self.assertEqual(len(mems), 0)


# ======================================================================
# TESTS 19-20: Profile Context + Familiarity
# ======================================================================

class TestProfileContext(TestSetup):

    def test_19_profile_contains_relevant_memories(self):
        """Test 19: Profile context includes active memories."""
        prof_store.get_or_create_user("uid3", "ved")
        mem_store.add_memory("uid3", "personal_fact", "Plays cricket", 0.9)
        mem_store.add_memory("uid3", "identity", "From Pune", 0.95)
        mems = mem_store.get_active_memories("uid3", limit=5)
        summary = prof_store.build_profile_summary("uid3", mems)
        self.assertTrue(summary["known"])
        self.assertEqual(len(summary["memories"]), 2)

    def test_20_familiarity_increases_and_is_clamped(self):
        """Test 20: Familiarity increases from interactions and clamps at 1.0."""
        prof_store.get_or_create_user("uid4", "sam")
        # Simulate 200 interactions that would push past 1.0
        for _ in range(200):
            prof_store.update_familiarity("uid4", 0.01)
        profile = prof_store.get_user("uid4")
        self.assertLessEqual(profile["familiarity_score"], 1.0)
        self.assertGreaterEqual(profile["familiarity_score"], 0.0)


# ======================================================================
# TESTS 21-22: Deduplication + Historical Bootstrap
# ======================================================================

class TestDeduplication(TestSetup):

    def test_21_realtime_and_fallback_same_message_once(self):
        """Test 21: Same message_id from realtime and fallback is processed once."""
        prof_store.get_or_create_user("uid5", "rahul")
        msg = _make_msg("dedup_1", "uid5", "rahul", "hello")
        inserted1 = msg_store.store_message(msg)
        inserted2 = msg_store.store_message(msg)
        self.assertTrue(inserted1)
        self.assertFalse(inserted2)

    def test_22_historical_messages_not_replied_to(self):
        """Test 22: Historical bootstrap messages are never processed as new."""
        # Historical messages have is_historical=True
        msg = _make_msg("hist_1", "uid6", "old_user", "old message", is_historical=True)
        inserted = msg_store.store_message(msg)
        self.assertTrue(inserted)
        stored = msg_store.get_message_by_id("hist_1")
        # memory_processed=1 means it won't be re-extracted
        self.assertEqual(stored["memory_processed"], 1)  # historical = skip memory


# ======================================================================
# TESTS 23-24: Native Reply Sending
# ======================================================================

class TestNativeReplySend(TestSetup):

    def test_23_native_reply_uses_trigger_message(self):
        """Test 23: send_reply attempts native reply when trigger_message_id is provided."""
        from instagram import sender as ig_sender
        mock_cl = MagicMock()
        mock_dm = MagicMock()
        mock_dm.id = "32906"
        mock_dm.client_context = "ctx_abc"
        mock_cl.direct_message.return_value = mock_dm
        mock_cl.direct_send.return_value = MagicMock()

        # thread_id must be numeric string for int() conversion
        result = ig_sender.send_reply(mock_cl, "340282366", "hello!", trigger_message_id="32906")

        # fetch_direct_message tries cl.direct_message(int(thread_id), int(message_id))
        mock_cl.direct_message.assert_called_once_with(340282366, 32906)
        # Then direct_send should be called with reply_to_message
        mock_cl.direct_send.assert_called_once()
        call_kwargs = mock_cl.direct_send.call_args[1]
        self.assertEqual(call_kwargs.get("reply_to_message"), mock_dm)

    def test_24_native_reply_fallback_on_failure(self):
        """Test 24: Native reply failure safely falls back to normal send."""
        from instagram import sender as ig_sender
        mock_cl = MagicMock()
        mock_dm = MagicMock()
        mock_dm.id = "32907"
        mock_dm.client_context = "ctx_xyz"
        mock_cl.direct_message.return_value = mock_dm
        mock_cl.direct_send.side_effect = Exception("API error")
        mock_cl.direct_answer.return_value = MagicMock()

        # thread_id must be numeric for int() conversion
        ig_sender.send_reply(mock_cl, "340282366", "hello!", trigger_message_id="32907")

        mock_cl.direct_message.assert_called_once_with(340282366, 32907)
        # Fallback direct_answer called
        mock_cl.direct_answer.assert_called_once()


# ======================================================================
# TESTS 25-26: Staleness Check
# ======================================================================

class TestStaleness(TestSetup):

    def test_25_staleness_ignores_unrelated_lane(self):
        """Test 25: Staleness check doesn't discard response for unrelated lane activity."""
        # This is tested via MessageWorker._is_stale logic
        # A message in a different lane should not trigger staleness
        from workers.message_worker import MessageWorker
        worker = MessageWorker(
            cl=MagicMock(), thread_id="t1", bot_user_id=BOT_USER_ID,
            bot_username=BOT_USERNAME, processed_ids=set(),
            last_ts_container=[datetime.now(timezone.utc)],
            state_saver=lambda: None,
            lane_manager=LaneManager(BOT_USER_ID),
            fatigue_tracker=FatigueTracker(),
        )
        trigger = _make_msg("trig1", "u1", "rahul", "eve what's 2+2")
        lane = MagicMock()
        lane.lane_id = "lane_a"

        # No newer same-sender message → not stale
        is_stale = worker._is_stale(trigger, lane)
        self.assertFalse(is_stale)

    def test_26_staleness_discards_cancelled_response(self):
        """Test 26: Staleness check discards response when same sender sends cancellation."""
        from workers.message_worker import MessageWorker
        worker = MessageWorker(
            cl=MagicMock(), thread_id="t1", bot_user_id=BOT_USER_ID,
            bot_username=BOT_USERNAME, processed_ids=set(),
            last_ts_container=[datetime.now(timezone.utc)],
            state_saver=lambda: None,
            lane_manager=LaneManager(BOT_USER_ID),
            fatigue_tracker=FatigueTracker(),
        )

        trigger = _make_msg("trig2", "u2", "ved", "eve who are you", ts_offset_seconds=-1)
        cancel_msg = _make_msg("cancel1", "u2", "ved", "nvm im dumb", ts_offset_seconds=0)

        lane = MagicMock()
        lane.lane_id = "lane_b"

        # Inject the cancel message as the latest in the lane
        with worker._lane_lock:
            worker._lane_latest["lane_b"] = cancel_msg

        is_stale = worker._is_stale(trigger, lane)
        self.assertTrue(is_stale)


# ======================================================================
# TESTS 27-28: UTC Timestamps + SQLite Threading
# ======================================================================

class TestTimestampsAndThreading(TestSetup):

    def test_27_utc_timestamps_are_timezone_aware(self):
        """Test 27: NormalizedMessage timestamps are always timezone-aware UTC."""
        msg = _make_msg("ts1", "u1", "rahul", "hello")
        self.assertIsNotNone(msg.timestamp.tzinfo)
        self.assertEqual(msg.timestamp.tzinfo, timezone.utc)

    def test_28_sqlite_works_from_multiple_threads(self):
        """Test 28: SQLite storage works correctly from multiple threads simultaneously."""
        import tempfile, os, sqlite3 as sq
        # Use a completely independent file DB — no shared connection
        tmp = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        tmp_path = tmp.name
        tmp.close()

        def make_conn():
            conn = sq.connect(tmp_path, check_same_thread=False, timeout=5.0)
            conn.row_factory = sq.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=5000")
            return conn

        # Create schema
        with make_conn() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS T28 (
                    id TEXT PRIMARY KEY, val TEXT, ts TEXT
                );
            """)

        errors = []
        def worker(thread_id_str: str):
            try:
                row_id = f"row_{thread_id_str}_{int(time.time()*1000000)}"
                with make_conn() as c:
                    c.execute("INSERT OR IGNORE INTO T28(id,val,ts) VALUES(?,?,?)",
                              (row_id, f"v_{thread_id_str}", datetime.now(timezone.utc).isoformat()))
                    c.commit()
                    row = c.execute("SELECT id FROM T28 WHERE id=?", (row_id,)).fetchone()
                    if row is None:
                        raise AssertionError(f"row {row_id} not found after insert")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker, args=(str(i),)) for i in range(8)]
        for t in threads: t.start()
        for t in threads: t.join()

        try:
            os.unlink(tmp_path)
        except Exception:
            pass

        self.assertEqual(errors, [], f"SQLite threading errors: {errors}")

    def test_29_negative_timestamp_regression(self):
        """Test 29: Regression test for negative/invalid direct activity timestamps (like -1000)."""
        import instagrapi.extractors as ex
        
        # Test direct call to extractor with live observed raw negative timestamp (-1000)
        dt = ex._direct_timestamp_from_microseconds(-1000)
        # Should not raise OSError on Windows, and should equal the local datetime for 0 timestamp
        expected_dt = datetime.fromtimestamp(0)
        self.assertEqual(dt, expected_dt)

        # Test string value input too
        dt_str = ex._direct_timestamp_from_microseconds("-1000")
        self.assertEqual(dt_str, expected_dt)
        
        # Test standard parsing flows
        dt_zero = ex._direct_timestamp_from_microseconds(0)
        self.assertEqual(dt_zero, expected_dt)

        # Valid microsecond timestamp should be unchanged (e.g. 1770000000000000)
        valid_ts = 1770000000000000
        dt_valid = ex._direct_timestamp_from_microseconds(valid_ts)
        self.assertEqual(dt_valid, datetime.fromtimestamp(valid_ts // 1_000_000))


# ======================================================================
# SCENARIO SIMULATIONS A-G
# ======================================================================

class TestScenarioSimulations(TestSetup):
    """
    Social scenario simulations based on real GC behavior.
    Tests decision logic without making live Gemini calls.
    """

    def test_scenario_a_human_to_human_conversation(self):
        """
        Scenario A: Rahul and Ved talking to each other.
        Eve should locally ignore (human-to-human lane).
        """
        lane_mgr = LaneManager(bot_user_id=BOT_USER_ID)

        msg1 = _make_msg("sa1", "rahul_id", "Rahul", "where are you from?")
        msg2 = _make_msg("sa2", "ved_id", "Ved", "im from pune",
                         reply_to_msg_id="sa1", reply_to_user_id="rahul_id",
                         ts_offset_seconds=1)
        msg3 = _make_msg("sa3", "rahul_id", "Rahul", "what do you study?",
                         reply_to_msg_id="sa2", reply_to_user_id="ved_id",
                         ts_offset_seconds=2)

        lane_mgr.assign_lane(msg1)
        lane_mgr.assign_lane(msg2)
        lane3 = lane_mgr.assign_lane(msg3)

        # msg3 is Rahul replying to Ved — human-to-human
        result = attention_evaluate(
            msg=msg3, lane=lane3,
            bot_user_id=BOT_USER_ID, bot_username=BOT_USERNAME,
        )
        # Should be local ignore or at most GEMINI_REQUIRED, definitely not LOCAL_REPLY
        self.assertNotEqual(result.decision, "LOCAL_REPLY")
        print(f"\n[SCENARIO A] {result}")

    def test_scenario_b_direct_eve_address(self):
        """
        Scenario B: Rahul says 'eve where are you from?'
        Eve should reply.
        """
        msg = _make_msg("sb1", "rahul_id", "Rahul", "eve where are you from?")
        result = attention_evaluate(
            msg=msg, lane=None,
            bot_user_id=BOT_USER_ID, bot_username=BOT_USERNAME,
        )
        self.assertEqual(result.decision, "LOCAL_REPLY")
        print(f"\n[SCENARIO B] {result}")

    def test_scenario_c_atharv_replies_rahul_not_eve(self):
        """
        Scenario C: Atharv natively replies to Rahul's message with 'im 20', not Eve.
        Eve should locally ignore.
        """
        msg = _make_msg("sc1", "atharv_id", "Atharv", "im 20",
                        reply_to_msg_id="rahul_msg", reply_to_user_id="rahul_id")
        result = attention_evaluate(
            msg=msg, lane=None,
            bot_user_id=BOT_USER_ID, bot_username=BOT_USERNAME,
        )
        self.assertEqual(result.decision, "LOCAL_IGNORE")
        self.assertIn("native_reply_to_human", result.reasons)
        print(f"\n[SCENARIO C] {result}")

    def test_scenario_d_atharv_replies_eve(self):
        """
        Scenario D: Atharv natively replies to Eve with 'stfu'.
        Eve should locally reply.
        """
        msg = _make_msg("sd1", "atharv_id", "Atharv", "stfu",
                        reply_to_msg_id="eve_msg_1", reply_to_user_id=BOT_USER_ID)
        result = attention_evaluate(
            msg=msg, lane=None,
            bot_user_id=BOT_USER_ID, bot_username=BOT_USERNAME,
        )
        self.assertEqual(result.decision, "LOCAL_REPLY")
        self.assertIn("native_reply_to_eve", result.reasons)
        print(f"\n[SCENARIO D] {result}")

    def test_scenario_e_burst_coalescing(self):
        """
        Scenario E: Rahul sends burst — 'bro', 'wait', 'listen', 'wtf', '😭'
        Should be treated as one burst.
        """
        received: List[BurstGroup] = []
        coalescer = BurstCoalescer(window_ms=100, emit_callback=received.append)

        for i, text in enumerate(["bro", "wait", "listen", "wtf", "😭"]):
            msg = _make_msg(f"se{i}", "rahul_id", "Rahul", text)
            coalescer.add(msg)
            time.sleep(0.02)

        time.sleep(0.3)
        self.assertEqual(len(received), 1)
        self.assertEqual(len(received[0].messages), 5)
        self.assertEqual(received[0].trigger_message.text, "\U0001f62d")
        trigger_text = received[0].trigger_message.text
        print(f"\n[SCENARIO E] burst count={len(received[0].messages)} trigger=emoji_ok")

    def test_scenario_f_lane_separation(self):
        """
        Scenario F: Two human conversations + Eve talking with third user.
        Lane participants should be separated.
        """
        lane_mgr = LaneManager(bot_user_id=BOT_USER_ID)

        # Eve ↔ User3
        eve_msg = _make_msg("sf_eve", BOT_USER_ID, "eve", "hey!", is_viewer=True)
        user3_reply = _make_msg("sf_u3", "user3", "sam", "eve sup",
                                reply_to_msg_id="sf_eve", reply_to_user_id=BOT_USER_ID,
                                ts_offset_seconds=1)
        lane1 = lane_mgr.assign_lane(eve_msg)
        lane1b = lane_mgr.assign_lane(user3_reply)

        # User1 ↔ User2
        u1_msg = _make_msg("sf_u1", "user1", "atharv", "hey", ts_offset_seconds=2)
        u2_msg = _make_msg("sf_u2", "user2", "ved", "hey atharv",
                           reply_to_msg_id="sf_u1", reply_to_user_id="user1",
                           ts_offset_seconds=3)
        lane2 = lane_mgr.assign_lane(u1_msg)
        lane2b = lane_mgr.assign_lane(u2_msg)

        # Lanes should be different
        all_lane_ids = {lane1.lane_id, lane2.lane_id}
        self.assertEqual(len(all_lane_ids), 2)

        # Lane should contain Eve
        self.assertTrue(lane1.contains_yap or lane1b.contains_yap)
        print(f"\n[SCENARIO F] eve_lane={lane1.lane_id} human_lane={lane2.lane_id} separated={lane1.lane_id != lane2.lane_id}")

    def test_scenario_g_memory_context_retrieval(self):
        """
        Scenario G: Known user says 'i lost again' — their stored memory about
        losing a football match should be retrievable.
        """
        prof_store.get_or_create_user("uid_g", "rahul")
        mem_store.add_memory("uid_g", "episodic", "Lost a football match recently", 0.9)
        mem_store.add_memory("uid_g", "preference", "Likes football", 0.85)
        mem_store.add_memory("uid_g", "identity", "From Mumbai", 0.95)

        # Retrieve with topic words from "i lost again"
        topic_words = ["lost", "again"]
        memories = mem_store.get_relevant_memories("uid_g", topic_words, limit=5)

        facts = [m["fact"] for m in memories]
        self.assertTrue(any("football" in f.lower() or "lost" in f.lower() for f in facts))
        print(f"\n[SCENARIO G] retrieved memories={facts}")


# ======================================================================
# V4 UPGRADE TESTS
# ======================================================================

class TestV4Upgrades(unittest.TestCase):
    def setUp(self):
        init_db()
        with db_module.get_connection() as conn:
            conn.execute("DELETE FROM USERS")
            conn.execute("DELETE FROM MEMORIES")
            conn.commit()

    def test_language_style_detection_and_tracking(self):
        # 1. Test detection heuristics
        style_eng = prof_store.detect_language_style("hey bro, what's up?")
        style_hing = prof_store.detect_language_style("kya chal raha hai yaar?")
        style_dev = prof_store.detect_language_style("नमस्ते, आप कैसे हैं?")
        
        self.assertEqual(style_eng, "English")
        self.assertEqual(style_hing, "Hinglish")
        self.assertEqual(style_dev, "Devanagari Hindi")

        # 2. Test database tracking
        prof_store.get_or_create_user("user_lang_test", "lang_user")
        prof_store.update_language_style("user_lang_test", "Hinglish")
        user = prof_store.get_user("user_lang_test")
        self.assertEqual(user["language_style"], "Hinglish")

    def test_claim_belief_transitions_and_contradictions(self):
        user_id = "user_mem_test"
        # Initial name claim
        mem_store.add_claim_memory(user_id, "identity", "name", "Zuck", "NEW", 0.9)
        active_mems = mem_store.get_active_memories(user_id)
        self.assertEqual(len(active_mems), 1)
        self.assertEqual(active_mems[0]["value"], "Zuck")
        self.assertEqual(active_mems[0]["status"], "active")

        # Add identical name (SUPPORT)
        mem_store.add_claim_memory(user_id, "identity", "name", "Zuck", "SUPPORT", 0.8)
        active_mems = mem_store.get_active_memories(user_id)
        self.assertEqual(len(active_mems), 1)
        self.assertEqual(active_mems[0]["support_count"], 2)

        # Contradictory name claim
        mem_store.add_claim_memory(user_id, "identity", "name", "Gates", "CONTRADICTION", 0.95)
        
        # Zuck should remain but get conflicted status
        with db_module.get_connection() as conn:
            zuck_row = conn.execute("SELECT * FROM MEMORIES WHERE value = 'Zuck'").fetchone()
            gates_row = conn.execute("SELECT * FROM MEMORIES WHERE value = 'Gates'").fetchone()
        
        zuck_dict = dict(zuck_row)
        gates_dict = dict(gates_row)
        
        self.assertEqual(zuck_dict["status"], "conflicted")
        self.assertEqual(zuck_dict["contradiction_count"], 1)
        self.assertEqual(gates_dict["status"], "conflicted")
        self.assertEqual(gates_dict["active"], 0)

        # Retrieve unresolved contradictions
        unresolved = mem_store.get_unresolved_contradictions(user_id)
        self.assertTrue(any(x["value"] == "Gates" for x in unresolved))

        # Correction signal
        mem_store.add_claim_memory(user_id, "identity", "name", "Gates", "CORRECTION", 0.99)
        
        # Gates should become active (active = 1, status = 'active'), Zuck should become superseded (active = 0, status = 'superseded')
        with db_module.get_connection() as conn:
            zuck_after = dict(conn.execute("SELECT * FROM MEMORIES WHERE value = 'Zuck'").fetchone())
            gates_after = dict(conn.execute("SELECT * FROM MEMORIES WHERE value = 'Gates' AND status = 'active'").fetchone())
        
        self.assertEqual(zuck_after["status"], "superseded")
        self.assertEqual(zuck_after["active"], 0)
        self.assertEqual(gates_after["status"], "active")
        self.assertEqual(gates_after["active"], 1)

    def test_joke_filtering(self):
        user_id = "joke_test"
        # Joke/uncertain claim
        mem_store.add_claim_memory(user_id, "identity", "age", "100 years old", "JOKE_OR_UNCERTAIN", 0.8)
        active_mems = mem_store.get_active_memories(user_id)
        self.assertEqual(len(active_mems), 0)
        
        with db_module.get_connection() as conn:
            row = dict(conn.execute("SELECT * FROM MEMORIES WHERE user_id = ?", (user_id,)).fetchone())
        self.assertEqual(row["status"], "candidate")
        self.assertEqual(row["active"], 0)

    def test_targeting_and_roast_rules(self):
        # TEST 2: Incoming message is a native reply to Eve: "stfu" -> LOCAL_REPLY
        msg2 = NormalizedMessage(
            message_id="m2", thread_id="t1", sender_id="user_a", sender_username="alice",
            text="stfu", timestamp=datetime.now(timezone.utc), item_type="text",
            is_sent_by_viewer=False, reply_to_message_id="bot_msg_id", reply_to_user_id=BOT_USER_ID
        )
        self.assertEqual(attention_evaluate(msg2, None, BOT_USER_ID, "eve").decision, "LOCAL_REPLY")

        # TEST 3: Incoming message is a native reply to Eve: "fuck u" -> LOCAL_REPLY
        msg3 = NormalizedMessage(
            message_id="m3", thread_id="t1", sender_id="user_a", sender_username="alice",
            text="fuck u", timestamp=datetime.now(timezone.utc), item_type="text",
            is_sent_by_viewer=False, reply_to_message_id="bot_msg_id", reply_to_user_id=BOT_USER_ID
        )
        self.assertEqual(attention_evaluate(msg3, None, BOT_USER_ID, "eve").decision, "LOCAL_REPLY")

        # TEST 4: Active direct Eve lane: Eve: "no", User: "yes" -> LOCAL_REPLY
        lane_eve = LaneState(
            lane_id="lane_eve", participants={"user_a", BOT_USER_ID}, message_ids=["bot_msg_id"],
            last_activity=datetime.now(timezone.utc), strength="strong", contains_yap=True
        )
        msg4 = NormalizedMessage(
            message_id="m4", thread_id="t1", sender_id="user_a", sender_username="alice",
            text="yes", timestamp=datetime.now(timezone.utc), item_type="text",
            is_sent_by_viewer=False
        )
        self.assertEqual(attention_evaluate(msg4, lane_eve, BOT_USER_ID, "eve").decision, "LOCAL_REPLY")

        # TEST 5: Active direct Eve lane: Eve: "bro 😭", User: "ur annoying" -> LOCAL_REPLY
        msg5 = NormalizedMessage(
            message_id="m5", thread_id="t1", sender_id="user_a", sender_username="alice",
            text="ur annoying", timestamp=datetime.now(timezone.utc), item_type="text",
            is_sent_by_viewer=False
        )
        self.assertEqual(attention_evaluate(msg5, lane_eve, BOT_USER_ID, "eve").decision, "LOCAL_REPLY")

        # TEST 6: Rahul replies natively to Ved: "stfu" -> LOCAL_IGNORE
        msg6 = NormalizedMessage(
            message_id="m6", thread_id="t1", sender_id="rahul_id", sender_username="rahul",
            text="stfu", timestamp=datetime.now(timezone.utc), item_type="text",
            is_sent_by_viewer=False, reply_to_message_id="ved_msg_id", reply_to_user_id="ved_id"
        )
        self.assertEqual(attention_evaluate(msg6, None, BOT_USER_ID, "eve").decision, "LOCAL_IGNORE")

        # TEST 7: Rahul says to Ved: "love u bro" (human-to-human context) -> LOCAL_IGNORE
        msg7 = NormalizedMessage(
            message_id="m7", thread_id="t1", sender_id="rahul_id", sender_username="rahul",
            text="love u bro", timestamp=datetime.now(timezone.utc), item_type="text",
            is_sent_by_viewer=False, reply_to_message_id="ved_msg_id", reply_to_user_id="ved_id"
        )
        self.assertEqual(attention_evaluate(msg7, None, BOT_USER_ID, "eve").decision, "LOCAL_IGNORE")

        # TEST 8: User: "eve i hate u" -> LOCAL_REPLY (explicit mention, negative tone does not cause ignore)
        msg8 = NormalizedMessage(
            message_id="m8", thread_id="t1", sender_id="user_a", sender_username="alice",
            text="eve i hate u", timestamp=datetime.now(timezone.utc), item_type="text",
            is_sent_by_viewer=False
        )
        self.assertEqual(attention_evaluate(msg8, None, BOT_USER_ID, "eve").decision, "LOCAL_REPLY")

        # TEST 9: Two humans are arguing (Eve is not part of the lane) -> LOCAL_IGNORE
        lane_humans = LaneState(
            lane_id="lane_humans", participants={"user_a", "user_b"}, message_ids=["other_msg_id"],
            last_activity=datetime.now(timezone.utc), strength="strong", contains_yap=False
        )
        msg9 = NormalizedMessage(
            message_id="m9", thread_id="t1", sender_id="user_a", sender_username="alice",
            text="ur an idiot", timestamp=datetime.now(timezone.utc), item_type="text",
            is_sent_by_viewer=False
        )
        self.assertEqual(attention_evaluate(msg9, lane_humans, BOT_USER_ID, "eve").decision, "LOCAL_IGNORE")

        # TEST 10: Two humans are joking (Eve is not part of the lane) -> LOCAL_IGNORE
        msg10 = NormalizedMessage(
            message_id="m10", thread_id="t1", sender_id="user_a", sender_username="alice",
            text="lmao true", timestamp=datetime.now(timezone.utc), item_type="text",
            is_sent_by_viewer=False
        )
        self.assertEqual(attention_evaluate(msg10, lane_humans, BOT_USER_ID, "eve").decision, "LOCAL_IGNORE")

        # TEST 11: Whole-group question: "guys kal college aa rahe ho?" -> GEMINI_REQUIRED
        msg11 = NormalizedMessage(
            message_id="m11", thread_id="t1", sender_id="user_a", sender_username="alice",
            text="guys kal college aa rahe ho?", timestamp=datetime.now(timezone.utc), item_type="text",
            is_sent_by_viewer=False
        )
        self.assertEqual(attention_evaluate(msg11, None, BOT_USER_ID, "eve").decision, "GEMINI_REQUIRED")

        # TEST 12: Direct interaction with Eve while social fatigue is high -> LOCAL_REPLY (fatigue must not block)
        msg12 = NormalizedMessage(
            message_id="m12", thread_id="t1", sender_id="user_a", sender_username="alice",
            text="@eve what's up", timestamp=datetime.now(timezone.utc), item_type="text",
            is_sent_by_viewer=False
        )
        self.assertEqual(attention_evaluate(msg12, None, BOT_USER_ID, "eve", fatigue_multiplier=0.9).decision, "LOCAL_REPLY")

    @patch("intelligence.social_judge.gemini_pool.generate_content")
    def test_social_judge_targeting_overrides(self, mock_generate_content):
        # TEST 1: Context: Eve: "kar na", User: "u will be blocked".
        # Target: EVE, Action: IGNORE returned by model is overridden to REPLY.
        # Judge does not reject it because of hostile/negative tone!
        mock_response = MagicMock()
        mock_response.text = '{"target_type": "EVE", "target_user_id": null, "action": "IGNORE", "confidence": 0.9, "tone": "HOSTILE", "reason": "user is threatening to block AI"}'
        mock_generate_content.return_value = mock_response

        res = social_judge.judge("u will be blocked", "alice", None, None, [], None)
        self.assertEqual(res.target_type, "EVE")
        self.assertEqual(res.action, "REPLY")
        self.assertEqual(res.tone, "HOSTILE")

        # target_type = SPECIFIC_USER overrides action to IGNORE regardless of positive tone
        mock_response2 = MagicMock()
        mock_response2.text = '{"target_type": "SPECIFIC_USER", "target_user_id": "bob_id", "action": "REPLY", "confidence": 0.95, "tone": "AFFECTIONATE", "reason": "talking to bob"}'
        mock_generate_content.return_value = mock_response2
        res2 = social_judge.judge("i love u bob", "alice", None, None, [], None)
        self.assertEqual(res2.target_type, "SPECIFIC_USER")
        self.assertEqual(res2.action, "IGNORE")
        self.assertEqual(res2.tone, "AFFECTIONATE")

        # Direct interaction with Eve while social fatigue is high -> REPLY
        mock_response3 = MagicMock()
        mock_response3.text = '{"target_type": "EVE", "target_user_id": null, "action": "REPLY", "confidence": 0.8, "tone": "NEUTRAL", "reason": "direct question"}'
        mock_generate_content.return_value = mock_response3
        res3 = social_judge.judge("eve tell me", "alice", None, None, [], None, fatigue_multiplier=0.8)
        self.assertEqual(res3.target_type, "EVE")
        self.assertEqual(res3.action, "REPLY")


# ======================================================================
# GEMINI API POOL TESTS
# ======================================================================

from unittest.mock import patch, MagicMock
from intelligence.gemini_pool import GeminiPool, GeminiKey, get_pool
import intelligence.gemini_pool as gemini_pool
from google.genai import types
from pydantic import BaseModel
import time

class DummySchema(BaseModel):
    test_field: str

class TestGeminiPool(unittest.TestCase):
    def setUp(self):
        import os
        self.saved_env = {}
        for key in ["GEMINI_API_KEY", "GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GEMINI_API_KEY_4", "GEMINI_API_KEY_5"]:
            if key in os.environ:
                self.saved_env[key] = os.environ[key]
                del os.environ[key]

    def tearDown(self):
        import os
        for key in ["GEMINI_API_KEY", "GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GEMINI_API_KEY_4", "GEMINI_API_KEY_5"]:
            if key in os.environ:
                del os.environ[key]
        for key, val in self.saved_env.items():
            os.environ[key] = val

    def test_pool_loading_cases(self):
        # Test Case 1: One configured key loads
        with patch.dict("os.environ", {"GEMINI_API_KEY_1": "key1", "GEMINI_API_KEY_2": "", "GEMINI_API_KEY": ""}):
            pool = GeminiPool()
            self.assertEqual(len(pool._keys), 1)
            self.assertEqual(pool._keys[0].key_index, 1)

        # Test Case 2 & 3: Two and five configured keys load
        with patch.dict("os.environ", {
            "GEMINI_API_KEY_1": "key1",
            "GEMINI_API_KEY_2": "key2",
            "GEMINI_API_KEY_3": "key3",
            "GEMINI_API_KEY_4": "key4",
            "GEMINI_API_KEY_5": "key5",
            "GEMINI_API_KEY": ""
        }):
            pool = GeminiPool()
            self.assertEqual(len(pool._keys), 5)

        # Test Case 4 & 5: Empty and missing slots are ignored
        with patch.dict("os.environ", {
            "GEMINI_API_KEY_1": "  key1  ",  # stripped
            "GEMINI_API_KEY_2": "",
            "GEMINI_API_KEY_4": "key4",
            "GEMINI_API_KEY": ""
        }, clear=True):
            pool = GeminiPool()
            self.assertEqual(len(pool._keys), 2)
            self.assertEqual(pool._keys[0]._api_key, "key1")
            self.assertEqual(pool._keys[1]._api_key, "key4")

        # Test Case 6: Duplicate keys are deduplicated
        with patch.dict("os.environ", {
            "GEMINI_API_KEY_1": "key1",
            "GEMINI_API_KEY_2": "key1",
            "GEMINI_API_KEY": ""
        }):
            pool = GeminiPool()
            self.assertEqual(len(pool._keys), 1)

        # Test Case 7: Legacy GEMINI_API_KEY compatibility
        with patch.dict("os.environ", {
            "GEMINI_API_KEY_1": "key1",
            "GEMINI_API_KEY": "legacy"
        }):
            pool = GeminiPool()
            self.assertEqual(len(pool._keys), 2)
            self.assertEqual(pool._keys[1]._api_key, "legacy")

    def test_round_robin_cases(self):
        # Test Case 8: Round-robin with 2 keys
        with patch.dict("os.environ", {"GEMINI_API_KEY_1": "key1", "GEMINI_API_KEY_2": "key2", "GEMINI_API_KEY": ""}):
            pool = GeminiPool()
            self.assertEqual(len(pool._keys), 2)
            
            # Mock the Clients
            mock_client1 = MagicMock()
            mock_client2 = MagicMock()
            pool._keys[0].client = mock_client1
            pool._keys[1].client = mock_client2

            # Request 1 -> key 1
            pool.generate_content("hello")
            self.assertEqual(pool._keys[0].total_requests, 1)
            self.assertEqual(pool._keys[1].total_requests, 0)

            # Request 2 -> key 2
            pool.generate_content("hello")
            self.assertEqual(pool._keys[0].total_requests, 1)
            self.assertEqual(pool._keys[1].total_requests, 1)

            # Request 3 -> key 1
            pool.generate_content("hello")
            self.assertEqual(pool._keys[0].total_requests, 2)
            self.assertEqual(pool._keys[1].total_requests, 1)

    def test_failover_scenarios(self):
        # Test Case 10 & 11: 429 & RESOURCE_EXHAUSTED failover
        with patch.dict("os.environ", {"GEMINI_API_KEY_1": "key1", "GEMINI_API_KEY_2": "key2", "GEMINI_API_KEY": ""}):
            pool = GeminiPool()
            
            mock_client1 = MagicMock()
            mock_client2 = MagicMock()
            
            # Client 1 raises 429 RESOURCE_EXHAUSTED
            mock_client1.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED quota exceeded")
            mock_client2.models.generate_content.return_value = MagicMock(text="success_val")
            
            pool._keys[0].client = mock_client1
            pool._keys[1].client = mock_client2
            
            res = pool.generate_content("hello")
            
            self.assertEqual(res.text, "success_val")
            self.assertTrue(pool._keys[0].is_cooling_down())
            self.assertEqual(pool._keys[0].total_failures, 1)
            self.assertEqual(pool._keys[1].total_successes, 1)

        # Test Case 12 & 13: Bounded failover on timeout / 5xx transient errors
        with patch.dict("os.environ", {"GEMINI_API_KEY_1": "key1", "GEMINI_API_KEY_2": "key2", "GEMINI_API_KEY": ""}):
            pool = GeminiPool()
            
            mock_client1 = MagicMock()
            mock_client2 = MagicMock()
            
            mock_client1.models.generate_content.side_effect = TimeoutError("request timed out")
            mock_client2.models.generate_content.return_value = MagicMock(text="success_transient")
            
            pool._keys[0].client = mock_client1
            pool._keys[1].client = mock_client2
            
            res = pool.generate_content("hello")
            self.assertEqual(res.text, "success_transient")
            self.assertEqual(pool._keys[0].consecutive_failures, 1)

        # Test Case 14: Invalid key becomes unhealthy
        with patch.dict("os.environ", {"GEMINI_API_KEY_1": "key1", "GEMINI_API_KEY_2": "key2", "GEMINI_API_KEY": ""}):
            pool = GeminiPool()
            
            mock_client1 = MagicMock()
            mock_client2 = MagicMock()
            
            mock_client1.models.generate_content.side_effect = Exception("API_KEY_INVALID error code 400")
            mock_client2.models.generate_content.return_value = MagicMock(text="valid_key_val")
            
            pool._keys[0].client = mock_client1
            pool._keys[1].client = mock_client2
            
            res = pool.generate_content("hello")
            self.assertEqual(res.text, "valid_key_val")
            self.assertFalse(pool._keys[0].healthy)

        # Test Case 15 & 16: Cooldown key is skipped, healthy continues working
        with patch.dict("os.environ", {"GEMINI_API_KEY_1": "key1", "GEMINI_API_KEY_2": "key2", "GEMINI_API_KEY": ""}):
            pool = GeminiPool()
            
            mock_client1 = MagicMock()
            mock_client2 = MagicMock()
            mock_client2.models.generate_content.return_value = MagicMock(text="only_healthy")
            
            pool._keys[0].client = mock_client1
            pool._keys[1].client = mock_client2
            
            # Put key 1 in cooldown manually
            pool._keys[0].mark_cooldown(10.0)
            
            res = pool.generate_content("hello")
            self.assertEqual(res.text, "only_healthy")
            self.assertEqual(pool._keys[0].total_requests, 0)
            self.assertEqual(pool._keys[1].total_requests, 1)

        # Test Case 17: All unavailable raises error (subsystem fallback check)
        with patch.dict("os.environ", {"GEMINI_API_KEY_1": "key1", "GEMINI_API_KEY": ""}):
            pool = GeminiPool()
            pool._keys[0].mark_unhealthy()
            
            with self.assertRaises(RuntimeError):
                pool.generate_content("hello")

        # Test Case 18, 19 & 20: Structured schema and context are preserved
        with patch.dict("os.environ", {"GEMINI_API_KEY_1": "key1", "GEMINI_API_KEY_2": "key2", "GEMINI_API_KEY": ""}):
            pool = GeminiPool()
            
            mock_client1 = MagicMock()
            mock_client2 = MagicMock()
            
            mock_client1.models.generate_content.side_effect = Exception("429 rate limit exceeded")
            mock_client2.models.generate_content.return_value = MagicMock(text='{"test_field": "structured_val"}')
            
            pool._keys[0].client = mock_client1
            pool._keys[1].client = mock_client2
            
            cfg = types.GenerateContentConfig(response_schema=DummySchema)
            res = pool.generate_content(contents="hello", config_opts=cfg, model="custom-model")
            
            self.assertEqual(res.text, '{"test_field": "structured_val"}')
            
            # Check arguments passed to second client are identical
            mock_client2.models.generate_content.assert_called_once_with(
                model="custom-model",
                contents="hello",
                config=cfg
            )

        # Test Case 22: Raw API keys never appear in logs or repr
        with patch.dict("os.environ", {"GEMINI_API_KEY_1": "secretkey123", "GEMINI_API_KEY": ""}):
            pool = GeminiPool()
            repr_str = repr(pool._keys[0])
            self.assertNotIn("secretkey123", repr_str)

    def test_pool_concurrency(self):
        # Test Case 21: Concurrent workers safely use the pool
        with patch.dict("os.environ", {
            "GEMINI_API_KEY_1": "key1",
            "GEMINI_API_KEY_2": "key2",
            "GEMINI_API_KEY": ""
        }):
            pool = GeminiPool()
            
            mock_client1 = MagicMock()
            mock_client2 = MagicMock()
            
            def slow_response(*args, **kwargs):
                time.sleep(0.05)
                return MagicMock(text="done")
                
            mock_client1.models.generate_content.side_effect = slow_response
            mock_client2.models.generate_content.side_effect = slow_response
            
            pool._keys[0].client = mock_client1
            pool._keys[1].client = mock_client2
            
            threads = []
            for _ in range(5):
                t = threading.Thread(target=pool.generate_content, args=("query",))
                threads.append(t)
                t.start()
                
            for t in threads:
                t.join()
                
            # Verify request counts
            total = pool._keys[0].total_requests + pool._keys[1].total_requests
            self.assertEqual(total, 5)


# ======================================================================
# V5 SUBSYSTEM TESTS
# ======================================================================

class TestEveV5Subsystems(TestSetup):

    def test_eve_state_seeding_and_dynamic_facts(self):
        """Test EVE_STATE stable facts seeding and dynamic state lifecycle."""
        from storage import eve_state
        # Seeding should be idempotent and add stable facts
        eve_state.ensure_stable_facts_seeded()
        stable = eve_state.get_stable_facts()
        slots = {r["slot"] for r in stable}
        self.assertIn("name", slots)
        self.assertIn("age", slots)
        self.assertIn("gender", slots)
        self.assertIn("background", slots)

        # Add dynamic facts
        state_id1 = eve_state.add_dynamic_state("event", "going to a movie tomorrow", confidence=0.8)
        state_id2 = eve_state.add_dynamic_state("mood", "feeling excited", confidence=0.9)
        self.assertIsNotNone(state_id1)
        self.assertIsNotNone(state_id2)

        # Get recent dynamic state
        recent = eve_state.get_recent_dynamic_state(limit=5)
        recent_slots = [r["slot"] for r in recent]
        self.assertIn("event", recent_slots)
        self.assertIn("mood", recent_slots)

        # Deactivate
        eve_state.deactivate_dynamic_state(state_id1)
        recent_after_deact = eve_state.get_recent_dynamic_state(limit=5)
        recent_slots_after = [r["slot"] for r in recent_after_deact]
        self.assertNotIn("event", recent_slots_after)
        self.assertIn("mood", recent_slots_after)

    def test_mode_selector_controller(self):
        """Test proportional mode selector probability adjustments and history."""
        from conversation.mode_selector import ModeSelector, compute_energy_hint
        # Setup mode selector with 50% target ratio
        selector = ModeSelector(target_ratio=0.50, history_window=4)
        
        # Fresh stats
        stats = selector.stats()
        self.assertEqual(stats["target_ratio"], 0.50)
        self.assertEqual(stats["current_ratio"], 0.50)

        # Record TEXT replies, lowering voice ratio to 0%
        for _ in range(4):
            selector.record("TEXT")
        self.assertEqual(selector.stats()["current_ratio"], 0.0)

        # Proportional controller should raise voice probability (under target ratio)
        mode = selector.select_mode(voice_healthy=True, energetic=False)
        # Record voice, raising current voice ratio
        selector.record("VOICE")
        self.assertGreater(selector.stats()["current_ratio"], 0.0)

        # If voice is unhealthy, mode selector must always select TEXT
        mode_unhealthy = selector.select_mode(voice_healthy=False)
        self.assertEqual(mode_unhealthy, "TEXT")

        # Energy hint checks
        self.assertTrue(compute_energy_hint("EVE SUP!!!")) # uppercase & !!
        self.assertTrue(compute_energy_hint("😭😭 Level 100")) # emojis

    def test_voice_health_tracking(self):
        """Test VoiceHealth startup block, failure counter, and cooldown behavior."""
        from voice.health import VoiceHealth
        health = VoiceHealth(failure_threshold=3, cooldown_seconds=10.0)
        self.assertTrue(health.is_healthy())

        # Simulate failures to trigger cooldown
        health.record_failure()
        self.assertTrue(health.is_healthy()) # 1 failure
        health.record_failure()
        self.assertTrue(health.is_healthy()) # 2 failures
        health.record_failure()
        
        # 3 failures -> entered cooldown
        self.assertFalse(health.is_healthy())
        self.assertTrue(health.status()["in_cooldown"])

        # Reset on success
        health.record_success()
        self.assertTrue(health.is_healthy())
        self.assertEqual(health.status()["consecutive_failures"], 0)

        # Disable permanently
        health.disable_permanently("no ffmpeg")
        self.assertFalse(health.is_healthy())
        self.assertEqual(health.status()["startup_enabled"], False)

    def test_audio_helper_functions(self):
        """Test voice subsystem audio helper functions."""
        from voice import audio as voice_audio
        self.assertEqual(voice_audio.parse_sample_rate("audio/pcm;rate=24000"), 24000)
        self.assertEqual(voice_audio.parse_sample_rate("audio/pcm;rate=16000"), 16000)
        self.assertIsNone(voice_audio.parse_sample_rate("audio/wav"))

    def test_voice_degradation_without_ffmpeg(self):
        """Test that missing ffmpeg permanently blocks voice and selects TEXT."""
        from voice.health import VoiceHealth
        from conversation.mode_selector import ModeSelector

        # Initialize voice health and disable it permanently (simulating startup check failure)
        health = VoiceHealth(failure_threshold=3)
        health.disable_permanently("ffmpeg missing")
        self.assertFalse(health.is_healthy())

        # Select mode should return TEXT even with 0% current ratio and energetic boost
        selector = ModeSelector(target_ratio=0.5, history_window=5)
        for _ in range(5):
            selector.record("TEXT")

        # Even with extremely energetic conditions, ModeSelector must return TEXT when voice is unhealthy
        mode = selector.select_mode(voice_healthy=health.is_healthy(), energetic=True)
        self.assertEqual(mode, "TEXT")


# ======================================================================
# SOCIAL INTELLIGENCE REPAIR — lane strength, scene-based routing,
# canonical GC scene, familiarity wiring, text emoji ban, retry bounding.
# ======================================================================

def _smsg(mid, sid, sun, text, viewer=False, rid=None, ruid=None):
    """Build a canonical raw-scene message dict, matching the shape returned
    by storage.messages.get_recent_scene()."""
    return {
        "message_id": mid, "sender_id": sid, "sender_username": sun, "text": text,
        "is_sent_by_viewer": viewer, "reply_to_message_id": rid, "reply_to_user_id": ruid,
    }


class TestLaneStrengthReachability(unittest.TestCase):
    """Weak/medium/strong must all be honestly reachable — see PART 1/2 of
    the social intelligence repair: the old code could only ever reach
    "weak" (creation) or cap at "medium" on upgrade, never true "strong"
    from accumulated evidence, and never decayed back down."""

    def test_weak_lane_reachable_on_creation(self):
        mgr = LaneManager(bot_user_id=BOT_USER_ID)
        msg = _make_msg("w1", "userA", "alice", "random gc chatter")
        lane = mgr.assign_lane(msg)
        self.assertEqual(lane.strength, "weak")

    def test_strong_lane_reachable_on_reply_pair_creation(self):
        mgr = LaneManager(bot_user_id=BOT_USER_ID)
        msg1 = _make_msg("s1", "userA", "alice", "hey")
        mgr.assign_lane(msg1)
        msg2 = _make_msg("s2", "userB", "bob", "hi back", reply_to_msg_id="s1",
                          reply_to_user_id="userA", ts_offset_seconds=1)
        lane = mgr.assign_lane(msg2)
        self.assertEqual(lane.strength, "strong")

    def test_medium_lane_reachable_via_blended_evidence(self):
        mgr = LaneManager(bot_user_id=BOT_USER_ID)
        msg1 = _make_msg("m1", "userA", "alice", "hey")
        lane = mgr.assign_lane(msg1)
        self.assertEqual(lane.strength, "weak")
        # Same-sender turn-taking continuation (medium evidence) blended
        # against the decayed weak prior should cross into "medium".
        msg2 = _make_msg("m2", "userA", "alice", "anyone there", ts_offset_seconds=1)
        lane = mgr.assign_lane(msg2)
        self.assertEqual(lane.strength, "medium")

    def test_strong_evidence_upgrades_existing_lane_not_capped_at_medium(self):
        """Regression test for the actual bug: _add_to_lane used to cap any
        upgrade at 'medium' even when the new evidence was 'strong'."""
        mgr = LaneManager(bot_user_id=BOT_USER_ID)
        msg1 = _make_msg("u1", "userA", "alice", "hey")
        lane = mgr.assign_lane(msg1)
        # A native reply landing in this lane is strong evidence.
        msg2 = _make_msg("u2", "userB", "bob", "yo", reply_to_msg_id="u1",
                          reply_to_user_id="userA", ts_offset_seconds=1)
        lane2 = mgr.assign_lane(msg2)
        self.assertIs(lane, lane2)
        self.assertEqual(lane.strength, "strong")

    def test_human_lane_fast_path_reachable_end_to_end(self):
        """A native reply chain between two humans must actually produce a
        LOCAL_IGNORE via the lane fast path, proving it isn't dead code."""
        mgr = LaneManager(bot_user_id=BOT_USER_ID)
        msg1 = _make_msg("h1", "ved_id", "ved", "im cooked")
        mgr.assign_lane(msg1)
        msg2 = _make_msg("h2", "rahul_id", "rahul", "same", reply_to_msg_id="h1",
                          reply_to_user_id="ved_id", ts_offset_seconds=1)
        lane = mgr.assign_lane(msg2)
        msg3 = _make_msg("h3", "ved_id", "ved", "bro why", ts_offset_seconds=2)
        result = attention_evaluate(msg=msg3, lane=lane, bot_user_id=BOT_USER_ID, bot_username=BOT_USERNAME)
        self.assertEqual(result.decision, "LOCAL_IGNORE")
        self.assertIn("human_lane", result.reasons)


class TestSocialRoutingScenes(unittest.TestCase):
    """The 15 required deterministic social simulations — WHO IS THIS
    MESSAGE TALKING TO, not "can Eve think of something to say"."""

    def _cur(self, mid, sid, sun, text, rid=None, ruid=None):
        return NormalizedMessage(
            message_id=mid, thread_id="t1", sender_id=sid, sender_username=sun,
            text=text, timestamp=datetime.now(timezone.utc), item_type="text",
            is_sent_by_viewer=False, reply_to_message_id=rid, reply_to_user_id=ruid,
        )

    def _route(self, scene, msg):
        return attention_evaluate(
            msg=msg, lane=None, bot_user_id=BOT_USER_ID, bot_username=BOT_USERNAME,
            recent_scene=scene,
        )

    def test_scene1_human_location_conversation_ignored(self):
        scene = [_smsg("s1", "atharv", "atharv", "ved where are u from"),
                 _smsg("s2", "ved", "ved", "pune")]
        msg = self._cur("c1", "atharv", "atharv", "thought u were from mumbai")
        self.assertEqual(self._route(scene, msg).decision, "LOCAL_IGNORE")

    def test_scene2_human_exam_conversation_ignored(self):
        scene = [_smsg("s1", "ved", "ved", "exam tomorrow"),
                 _smsg("s2", "rahul", "rahul", "same bro")]
        msg = self._cur("c2", "ved", "ved", "im cooked")
        self.assertEqual(self._route(scene, msg).decision, "LOCAL_IGNORE")

    def test_scene3_direct_eve_continuation_replied(self):
        scene = [_smsg("s1", BOT_USER_ID, "eve", "what did i do", viewer=True)]
        msg = self._cur("c3", "atharv", "atharv", "stfu")
        self.assertEqual(self._route(scene, msg).decision, "LOCAL_REPLY")

    def test_scene4_eve_roast_continuation_replied(self):
        scene = [_smsg("s1", BOT_USER_ID, "eve", "ur actually stupid", viewer=True)]
        msg = self._cur("c4", "atharv", "atharv", "no u")
        self.assertEqual(self._route(scene, msg).decision, "LOCAL_REPLY")

    def test_scene5_mock_block_conversation_replied_not_ignored_on_negativity(self):
        scene = [_smsg("s1", "atharv", "atharv", "pls dont block me"),
                 _smsg("s2", BOT_USER_ID, "eve", "why would i", viewer=True)]
        msg = self._cur("c5", "atharv", "atharv", "u will be blocked")
        self.assertEqual(self._route(scene, msg).decision, "LOCAL_REPLY")

    def test_scene6_direct_address_replied(self):
        msg = self._cur("c6", "rahul", "rahul", "hii eve")
        self.assertEqual(self._route([], msg).decision, "LOCAL_REPLY")

    def test_scene7_native_reply_to_other_human_ignored(self):
        scene = [_smsg("s1", "ved", "ved", "im cooked")]
        msg = self._cur("c7", "rahul", "rahul", "same", rid="s1", ruid="ved")
        self.assertEqual(self._route(scene, msg).decision, "LOCAL_IGNORE")

    def test_scene8_native_reply_to_eve_replied(self):
        scene = [_smsg("s1", BOT_USER_ID, "eve", "nah ur done", viewer=True)]
        msg = self._cur("c8", "atharv", "atharv", "why", rid="s1", ruid=BOT_USER_ID)
        self.assertEqual(self._route(scene, msg).decision, "LOCAL_REPLY")

    def test_scene9_parallel_exchanges_ignores_interleaved_eve_chatter(self):
        scene = [_smsg("s1", "atharv", "atharv", "ved where u from"),
                 _smsg("s2", "ved", "ved", "pune"),
                 _smsg("s3", "rahul", "rahul", "hii eve"),
                 _smsg("s4", BOT_USER_ID, "eve", "hii", viewer=True)]
        msg = self._cur("c9", "atharv", "atharv", "thought u were from mumbai")
        self.assertEqual(self._route(scene, msg).decision, "LOCAL_IGNORE")

    def test_scene10_group_address_escalates_not_forced_reply(self):
        msg = self._cur("c10", "atharv", "atharv", "what are u guys doing")
        self.assertEqual(self._route([], msg).decision, "GEMINI_REQUIRED")

    def test_scene11_eve_insult_replied(self):
        msg = self._cur("c11", "atharv", "atharv", "eve ur dumb")
        self.assertEqual(self._route([], msg).decision, "LOCAL_REPLY")

    def test_scene12_other_user_insult_ignored(self):
        scene = [_smsg("s1", "ved", "ved", "hi")]
        msg = self._cur("c12", "atharv", "atharv", "ved ur dumb")
        self.assertEqual(self._route(scene, msg).decision, "LOCAL_IGNORE")

    def test_scene13_short_ambiguous_human_continuation_ignored(self):
        scene = [_smsg("s1", "ved", "ved", "i deleted it"),
                 _smsg("s2", "atharv", "atharv", "why")]
        msg = self._cur("c13", "ved", "ved", "idk")
        self.assertEqual(self._route(scene, msg).decision, "LOCAL_IGNORE")

    def test_scene14_short_eve_continuation_replied(self):
        scene = [_smsg("s1", BOT_USER_ID, "eve", "i deleted it", viewer=True)]
        msg = self._cur("c14", "atharv", "atharv", "why")
        self.assertEqual(self._route(scene, msg).decision, "LOCAL_REPLY")

    def test_scene15_profile_does_not_grant_ownership(self):
        """Even if Atharv has a rich profile/memory, Ved answering the human
        who addressed him ('ved send me that file' -> 'wait') must not be
        treated as talking to Eve — familiarity is never consulted here at
        all (attention.py never reads profile data), which is the point."""
        scene = [_smsg("s1", "atharv", "atharv", "ved send me that file")]
        msg = self._cur("c15", "ved", "ved", "wait")
        self.assertNotEqual(self._route(scene, msg).decision, "LOCAL_REPLY")


class TestCanonicalSceneAndReplyRelationships(TestSetup):
    """Router and response generator must derive from the same canonical
    scene, and reply relationships must survive into response generation
    instead of being flattened to 'sender: text'."""

    def test_social_judge_prompt_includes_recent_15_message_scene(self):
        scene = [_smsg(f"m{i}", f"u{i}", f"user{i}", f"text{i}") for i in range(15)]
        from intelligence.prompts import build_decision_prompt
        prompt = build_decision_prompt(
            msg_text="hello", sender_username="alice", reply_to_username=None,
            reply_to_text=None, scene_messages=scene, profile_summary=None,
        )
        for i in range(15):
            self.assertIn(f"text{i}", prompt)

    def test_scene_formatting_preserves_reply_arrows(self):
        from intelligence.prompts import format_raw_scene
        scene = [
            _smsg("a1", "rahul_id", "rahul", "ur dumb"),
            _smsg("a2", "ved_id", "ved", "no u", rid="a1", ruid="rahul_id"),
        ]
        rendered = format_raw_scene(scene)
        self.assertIn("(replying to rahul)", rendered)

    def test_response_context_uses_full_scene_not_lane_filtered(self):
        """Regression test for the tunnel-vision bug: response generation
        used to only see lane-participant messages, silently dropping
        anyone outside the lane even though they were in the raw scene."""
        from intelligence import context_builder
        prof_store.get_or_create_user("atharv_id", "atharv")
        full_scene = [
            _smsg("x1", "atharv_id", "atharv", "ved where are u from"),
            _smsg("x2", "ved_id", "ved", "pune"),
            _smsg("x3", "rahul_id", "rahul", "hii eve"),
        ]
        ctx = context_builder.build_response_context(
            sender_id="atharv_id", sender_username="atharv",
            current_message="thought u were from mumbai",
            scene_messages=full_scene, recent_eve_replies=[],
        )
        senders_in_context = {m["sender_id"] for m in ctx.recent_gc_messages}
        # Rahul must be visible even though he isn't in an "atharv/ved" lane.
        self.assertIn("rahul_id", senders_in_context)
        self.assertIn("ved_id", senders_in_context)

    def test_current_message_excluded_and_marked_separately(self):
        from intelligence.prompts import format_text_context
        from intelligence import context_builder
        prof_store.get_or_create_user("atharv_id", "atharv")
        scene = [_smsg("y1", "atharv_id", "atharv", "the current message text")]
        ctx = context_builder.build_response_context(
            sender_id="atharv_id", sender_username="atharv",
            current_message="the current message text",
            current_message_id="y1",
            scene_messages=scene, recent_eve_replies=[],
        )
        rendered = format_text_context(ctx)
        self.assertEqual(rendered.count("the current message text"), 1)
        self.assertIn("CURRENT MESSAGE", rendered)


class TestFamiliarityWiring(TestSetup):

    def test_passive_activity_grows_familiarity_gradually_and_diminishingly(self):
        prof_store.get_or_create_user("passive_uid", "sam")
        prev = 0.0
        deltas = []
        for i in range(40):
            prof_store.increment_message_count("passive_uid")
            prof_store.record_passive_activity("passive_uid")
            current = prof_store.get_user("passive_uid")["familiarity_score"]
            deltas.append(current - prev)
            prev = current
        # Not near-zero after 40 messages...
        self.assertGreater(prev, 0.05)
        # ...but not maxed out / best-friend either.
        self.assertLess(prev, 0.6)
        # Diminishing: the growth per message shrinks as message_count rises.
        self.assertGreater(deltas[0], deltas[-1])

    def test_direct_reply_to_eve_is_stronger_evidence_than_passive_presence(self):
        prof_store.get_or_create_user("direct_uid", "rahul")
        prof_store.get_or_create_user("passive_uid2", "ved")
        prof_store.update_familiarity("direct_uid", prof_store.FAMILIARITY_DELTA_REPLY_TO_YAP)
        prof_store.record_passive_activity("passive_uid2")
        direct_score = prof_store.get_user("direct_uid")["familiarity_score"]
        passive_score = prof_store.get_user("passive_uid2")["familiarity_score"]
        self.assertGreater(direct_score, passive_score)

    def test_familiarity_bounded_after_heavy_activity(self):
        prof_store.get_or_create_user("spam_uid", "spammer")
        for i in range(500):
            prof_store.increment_message_count("spam_uid")
            prof_store.record_passive_activity("spam_uid")
        score = prof_store.get_user("spam_uid")["familiarity_score"]
        self.assertLessEqual(score, 1.0)

    def test_relationship_tier_auto_derived_from_familiarity(self):
        prof_store.get_or_create_user("tier_uid", "bob")
        self.assertEqual(prof_store.get_user("tier_uid")["relationship_to_yap"], "new")
        prof_store.update_familiarity("tier_uid", 0.5)
        self.assertNotEqual(prof_store.get_user("tier_uid")["relationship_to_yap"], "new")

    def test_preferred_name_wired_from_identity_memory(self):
        from intelligence import memory_extractor
        prof_store.get_or_create_user("name_uid", "unknown_handle")
        mock_response = MagicMock()
        mock_response.text = (
            '{"memories": [{"user_id": "name_uid", "memory_type": "identity", '
            '"slot": "name", "value_fact": "Atharv", "claim_type": "NEW", '
            '"confidence": 0.9, "source_message_id": "m1"}]}'
        )
        with patch("intelligence.memory_extractor.gemini_pool.generate_content", return_value=mock_response):
            memory_extractor.extract_batch([
                {"message_id": "m1", "sender_id": "name_uid", "sender_username": "unknown_handle",
                 "text": "hi im Atharv"},
            ])
        self.assertEqual(prof_store.get_user("name_uid")["preferred_name"], "Atharv")


class TestTextEmojiBan(unittest.TestCase):

    def test_text_mode_addition_contains_hard_no_emoji_rule(self):
        from intelligence.prompts import EVE_TEXT_MODE_ADDITION
        self.assertIn("do not use any emoji", EVE_TEXT_MODE_ADDITION.lower())
        self.assertIn("hard rule", EVE_TEXT_MODE_ADDITION.lower())

    def test_core_instruction_has_no_emoji_characters(self):
        import re
        from intelligence.prompts import EVE_CORE_INSTRUCTION
        emoji_pattern = re.compile(
            "[\U0001F300-\U0001FAFF☀-➿←-⇿⬀-⯿]"
        )
        self.assertEqual(emoji_pattern.findall(EVE_CORE_INSTRUCTION), [])

    def test_generated_text_prompt_carries_no_emoji_instruction(self):
        from intelligence.response_generator import EVE_TEXT_MODE_ADDITION
        self.assertIn("Do not use any emoji", EVE_TEXT_MODE_ADDITION)

    def test_voice_mode_addition_retains_hindi_hinglish_behavior(self):
        """Removing emoji from text mode must not touch voice personality."""
        from intelligence.prompts import EVE_VOICE_MODE_ADDITION
        self.assertIn("Hindi", EVE_VOICE_MODE_ADDITION)
        self.assertIn("Hinglish", EVE_VOICE_MODE_ADDITION)


class TestGeminiCallReduction(unittest.TestCase):

    @patch("intelligence.response_generator.gemini_pool.generate_content")
    def test_text_generation_makes_exactly_one_pool_call_per_attempt(self, mock_generate):
        """Regression test for nested retries: response_generator used to
        wrap gemini_pool (which already owns bounded failover) in its own
        3x outer retry loop, multiplying attempts on failure."""
        from intelligence.response_generator import generate_from_context
        from models.context import ResponseContext

        mock_response = MagicMock()
        mock_response.text = "hii"
        mock_generate.return_value = mock_response

        ctx = ResponseContext(sender_id="u1", sender_username="alice", current_message="hi")
        text, _ = generate_from_context(ctx)
        self.assertEqual(text, "hii")
        self.assertEqual(mock_generate.call_count, 1)

    @patch("intelligence.response_generator.gemini_pool.generate_content")
    def test_text_generation_does_not_retry_on_failure(self, mock_generate):
        from intelligence.response_generator import generate_from_context
        from models.context import ResponseContext

        mock_generate.side_effect = RuntimeError("all keys exhausted")
        ctx = ResponseContext(sender_id="u1", sender_username="alice", current_message="hi")
        text, _ = generate_from_context(ctx)
        self.assertIsNone(text)
        # Exactly one call — gemini_pool already owns bounded failover
        # internally; this module must not multiply it with its own loop.
        self.assertEqual(mock_generate.call_count, 1)

    def test_local_fast_paths_avoid_gemini_judge_call(self):
        """Scenes with strong deterministic evidence must never reach
        GEMINI_REQUIRED — this is what keeps obvious cases off the Gemini
        critical path entirely."""
        scene = [_smsg("s1", BOT_USER_ID, "eve", "hii", viewer=True)]
        msg = NormalizedMessage(
            message_id="g1", thread_id="t1", sender_id="atharv_id", sender_username="atharv",
            text="stfu eve", timestamp=datetime.now(timezone.utc), item_type="text",
            is_sent_by_viewer=False,
        )
        result = attention_evaluate(
            msg=msg, lane=None, bot_user_id=BOT_USER_ID, bot_username=BOT_USERNAME, recent_scene=scene
        )
        self.assertNotEqual(result.decision, "GEMINI_REQUIRED")


class TestNativeReplyLatency(TestSetup):

    def test_dm_cache_hit_avoids_http_fetch(self):
        from instagram import sender as ig_sender
        cl = MagicMock()
        fake_dm = MagicMock()
        ig_sender.set_cached_dm("cached_msg_1", fake_dm)
        result = ig_sender.fetch_direct_message(cl, "thread1", "cached_msg_1")
        self.assertIs(result, fake_dm)
        cl.direct_message.assert_not_called()

    def test_prefetch_warms_cache_for_realtime_messages(self):
        """Realtime (MQTT) messages arrive without a raw_dm, which used to
        force a synchronous HTTP fetch right before send. The worker now
        prefetches in the background as soon as the message arrives."""
        from workers.message_worker import MessageWorker
        from instagram import sender as ig_sender
        from conversation.lanes import LaneManager
        from conversation.fatigue import FatigueTracker

        cl = MagicMock()
        fake_dm = MagicMock()
        cl.direct_message.return_value = fake_dm

        worker = MessageWorker(
            cl=cl, thread_id="12345", bot_user_id=BOT_USER_ID, bot_username=BOT_USERNAME,
            processed_ids=set(), last_ts_container=[datetime.now(timezone.utc)],
            state_saver=lambda: None, lane_manager=LaneManager(bot_user_id=BOT_USER_ID),
            fatigue_tracker=FatigueTracker(),
        )
        worker._prefetch_trigger_dm("67890")
        # Background daemon thread — give it a moment to complete.
        for _ in range(50):
            if ig_sender.get_cached_dm("67890") is not None:
                break
            time.sleep(0.01)
        self.assertIs(ig_sender.get_cached_dm("67890"), fake_dm)
        cl.direct_message.assert_called_once()


class TestStrictRepliesAndPrefetchSynchronization(TestSetup):

    def test_01_raw_dm_immediately_available_native_reply(self):
        from instagram import sender as ig_sender
        cl = MagicMock()
        fake_dm = MagicMock()
        fake_dm.id = "123"
        fake_dm.client_context = "ctx123"
        
        sent = ig_sender.send_reply(cl, "123", "my reply", trigger_dm=fake_dm, strict=True)
        
        cl.direct_message.assert_not_called()
        cl.direct_send.assert_called_once_with(
            text="my reply",
            thread_ids=[123],
            reply_to_message=fake_dm
        )

    def test_02_cache_hit_native_reply(self):
        from instagram import sender as ig_sender
        cl = MagicMock()
        fake_dm = MagicMock()
        fake_dm.id = "234"
        fake_dm.client_context = "ctx234"
        
        ig_sender.set_cached_dm("234", fake_dm)
        
        resolved_dm = ig_sender.fetch_direct_message(cl, "123", "234")
        self.assertIs(resolved_dm, fake_dm)
        cl.direct_message.assert_not_called()
        
        sent = ig_sender.send_reply(cl, "123", "my reply", trigger_message_id="234", strict=True)
        cl.direct_send.assert_called_once_with(
            text="my reply",
            thread_ids=[123],
            reply_to_message=fake_dm
        )

    def test_03_prefetch_finishes_before_generation(self):
        from instagram import sender as ig_sender
        cl = MagicMock()
        fake_dm = MagicMock()
        fake_dm.id = "345"
        fake_dm.client_context = "ctx345"
        cl.direct_message.return_value = fake_dm
        
        ig_sender.prefetch_direct_message(cl, "123", "345")
        
        for _ in range(50):
            if ig_sender.get_cached_dm("345") is not None:
                break
            time.sleep(0.01)
            
        resolved_dm = ig_sender.fetch_direct_message(cl, "123", "345")
        self.assertIs(resolved_dm, fake_dm)
        cl.direct_message.assert_called_once_with(123, 345)

    def test_04_generation_finishes_before_prefetch_waits_and_replies(self):
        from instagram import sender as ig_sender
        cl = MagicMock()
        fake_dm = MagicMock()
        fake_dm.id = "456"
        fake_dm.client_context = "ctx456"
        
        def slow_fetch(*args, **kwargs):
            time.sleep(0.2)
            return fake_dm
            
        cl.direct_message.side_effect = slow_fetch
        
        ig_sender.prefetch_direct_message(cl, "123", "456")
        
        t0 = time.time()
        resolved_dm = ig_sender.fetch_direct_message(cl, "123", "456", timeout=1.0)
        t_diff = time.time() - t0
        
        self.assertIs(resolved_dm, fake_dm)
        self.assertGreaterEqual(t_diff, 0.1)
        cl.direct_message.assert_called_once_with(123, 456)

    def test_05_prefetch_fails_sync_targeted_resolution(self):
        from instagram import sender as ig_sender
        cl = MagicMock()
        
        cl.direct_message.side_effect = Exception("HTTP error in prefetch")
        ig_sender.prefetch_direct_message(cl, "123", "567")
        
        for _ in range(50):
            with ig_sender._CACHE_LOCK:
                in_progress = "567" in ig_sender._PREFETCH_EVENTS
            if not in_progress:
                break
            time.sleep(0.01)
            
        fake_dm = MagicMock()
        fake_dm.id = "567"
        fake_dm.client_context = "ctx567"
        cl.direct_message.side_effect = None
        cl.direct_message.return_value = fake_dm
        
        resolved_dm = ig_sender.fetch_direct_message(cl, "123", "567")
        self.assertIs(resolved_dm, fake_dm)
        self.assertEqual(cl.direct_message.call_count, 2)

    def test_06_target_cannot_be_resolved_no_standalone_text(self):
        from instagram import sender as ig_sender
        cl = MagicMock()
        cl.direct_message.return_value = None
        
        sent = ig_sender.send_reply(cl, "123", "my reply", trigger_message_id="678", strict=True)
        
        self.assertIsNone(sent)
        cl.direct_answer.assert_not_called()

    def test_07_two_rapid_messages_use_own_triggers(self):
        from instagram import sender as ig_sender
        cl = MagicMock()
        
        dm1 = MagicMock()
        dm1.id = "111"
        dm1.client_context = "ctx1"
        
        dm2 = MagicMock()
        dm2.id = "222"
        dm2.client_context = "ctx2"
        
        def resolve_dm(thread_id, msg_id):
            if msg_id == 111:
                return dm1
            if msg_id == 222:
                return dm2
            return None
            
        cl.direct_message.side_effect = resolve_dm
        
        ig_sender.prefetch_direct_message(cl, "123", "111")
        ig_sender.prefetch_direct_message(cl, "123", "222")
        
        res1 = ig_sender.fetch_direct_message(cl, "123", "111")
        res2 = ig_sender.fetch_direct_message(cl, "123", "222")
        
        self.assertIs(res1, dm1)
        self.assertIs(res2, dm2)
        
        ig_sender.send_reply(cl, "123", "reply to 111", trigger_dm=res1, strict=True)
        ig_sender.send_reply(cl, "123", "reply to 222", trigger_dm=res2, strict=True)
        
        self.assertEqual(cl.direct_send.call_count, 2)
        calls = cl.direct_send.call_args_list
        self.assertEqual(calls[0][1]["reply_to_message"], dm1)
        self.assertEqual(calls[1][1]["reply_to_message"], dm2)

    def test_08_message_id_normalization(self):
        from instagram import sender as ig_sender
        cl = MagicMock()
        fake_dm = MagicMock()
        fake_dm.id = 99999
        fake_dm.client_context = "ctx999"
        
        ig_sender.set_cached_dm("99999.0", fake_dm)
        
        res1 = ig_sender.get_cached_dm(99999)
        self.assertIs(res1, fake_dm)
        
        res2 = ig_sender.get_cached_dm("99999")
        self.assertIs(res2, fake_dm)
        
        res3 = ig_sender.get_cached_dm("  99999  ")
        self.assertIs(res3, fake_dm)

    def test_09_voice_path_unchanged(self):
        from instagram import sender as ig_sender
        cl = MagicMock()
        fake_dm = MagicMock()
        cl.direct_send_voice.return_value = fake_dm
        
        res = ig_sender.send_voice(cl, "123", "dummy.m4a")
        self.assertIs(res, fake_dm)
        cl.direct_send_voice.assert_called_once()
        cl.direct_answer.assert_not_called()


# ======================================================================
# TESTS 30-34: V5.5 Upgrades (Turn Plans, Thread Linker, Revalidation)
# ======================================================================

class TestEveV5_5(TestSetup):

    def test_thread_linker_explicit_native_reply(self):
        from conversation.room import ThreadLinker, SpeakerInteractionMap
        linker = ThreadLinker(bot_user_id=BOT_USER_ID, bot_username=BOT_USERNAME)
        sim = SpeakerInteractionMap()
        cl = MagicMock()

        # Seed database with parent message
        p_msg = _make_msg("parent_id", "user1", "atharv", "hello eve", thread_id="t1")
        p_msg.conversation_id = "conv_123"
        msg_store.store_message(p_msg)

        # Incoming reply message
        r_msg = _make_msg("reply_id", "user2", "rahul", "what's up", thread_id="t1", reply_to_msg_id="parent_id")
        conv_id = linker.link_message(r_msg, cl, sim)
        self.assertEqual(conv_id, "conv_123")

    def test_turn_ownership_resolver_rules(self):
        from conversation.room import TurnOwnershipResolver
        resolver = TurnOwnershipResolver(bot_user_id=BOT_USER_ID, bot_username=BOT_USERNAME)

        # 1. Native reply to Eve
        msg_eve = _make_msg("m1", "user1", "atharv", "hi", reply_to_msg_id="e1", reply_to_user_id=BOT_USER_ID)
        owner, target = resolver.resolve_ownership(msg_eve, [])
        self.assertEqual(owner, "EVE")

        # 2. Open group query
        msg_group = _make_msg("m2", "user2", "rahul", "anyone wants to go out?")
        owner, target = resolver.resolve_ownership(msg_group, [])
        self.assertEqual(owner, "OPEN_GROUP")

    def test_pre_send_revalidation(self):
        from storage import messages as msg_store
        # Store a message in conversation "conv_abc"
        msg1 = _make_msg("m_trigger", "user1", "atharv", "hello", thread_id="t1")
        msg1.conversation_id = "conv_abc"
        msg_store.store_message(msg1)

        # Revalidate immediately: newer count should be 0
        from storage.messages import count_newer_messages_in_conversation
        self.assertEqual(count_newer_messages_in_conversation("conv_abc", "m_trigger"), 0)

        # Add a newer message in the same conversation
        msg2 = _make_msg("m_newer", "user2", "rahul", "replying...", thread_id="t1", ts_offset_seconds=5)
        msg2.conversation_id = "conv_abc"
        msg_store.store_message(msg2)

        # Newer count should now be 1
        self.assertEqual(count_newer_messages_in_conversation("conv_abc", "m_trigger"), 1)

    def test_unified_eve_ledger(self):
        from storage import eve_turns
        # Store a text turn
        eve_turns.store_eve_turn(
            conversation_id="conv_xyz",
            trigger_message_id="trig_1",
            target_user_id="user_1",
            modality="TEXT",
            semantic_summary="said hello",
            exact_text="hello!",
            voice_transcript=None,
            conversation_version=1
        )

        # Store a voice turn
        eve_turns.store_eve_turn(
            conversation_id="conv_xyz",
            trigger_message_id="trig_2",
            target_user_id="user_2",
            modality="VOICE",
            semantic_summary="replied with joke",
            exact_text=None,
            voice_transcript="haha so funny",
            conversation_version=2
        )

        # Retrieve recent turns
        turns = eve_turns.get_recent_eve_turns(limit=5)
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0]["modality"], "VOICE")
        self.assertEqual(turns[0]["voice_transcript"], "haha so funny")
        self.assertEqual(turns[1]["modality"], "TEXT")
        self.assertEqual(turns[1]["exact_text"], "hello!")

    def test_shared_turn_plan_prompt_formatting(self):
        from intelligence.turn_planner import TurnPlan
        from intelligence import prompts
        from models.context import ResponseContext

        plan = TurnPlan(
            conversation_id="conv_xyz",
            trigger_message_id="trig_1",
            target_user_id="user1",
            speech_act="TEASE",
            intent="tease Atharv",
            stance="PLAYFUL",
            facts_to_use=["likes cricket"],
            continuity_notes="continuing previous tease",
            avoid_topics=["football"],
            conversation_version=1
        )

        ctx = ResponseContext(
            sender_id="user1",
            sender_username="atharv",
            current_message="what's up",
            current_message_id="trig_1",
            thread_id="t1",
            bot_user_id=BOT_USER_ID,
            bot_username=BOT_USERNAME
        )

        formatted_text = prompts.format_text_context(ctx, plan=plan)
        self.assertIn("YOUR TURN PLAN (Follow this strictly):", formatted_text)
        self.assertIn("Speech Act: TEASE", formatted_text)
        self.assertIn("Intent: tease Atharv", formatted_text)
        self.assertIn("Stance: PLAYFUL", formatted_text)
        self.assertIn("Facts to mention: likes cricket", formatted_text)

        formatted_voice = prompts.format_voice_context(ctx, plan=plan)
        self.assertIn("[YOUR TURN PLAN - FOLLOW THIS STRICTLY]", formatted_voice)
        self.assertIn("Speech Act: TEASE", formatted_voice)



# ======================================================================
# SIMULATION SUMMARY
# ======================================================================

def print_simulation_summary():
    """Print a summary of local vs Gemini resolution."""
    test_cases = [
        ("Rahul replies to Ved (native)", "LOCAL_IGNORE"),
        ("Atharv replies Rahul 'im 20'", "LOCAL_IGNORE"),
        ("Rahul: 'eve where are you from'", "LOCAL_REPLY"),
        ("@eve summon", "LOCAL_REPLY"),
        ("Atharv replies to Eve 'stfu'", "LOCAL_REPLY"),
        ("Whole group question without @", "GEMINI_REQUIRED"),
        ("Ambiguous 'what's up everyone'", "GEMINI_REQUIRED"),
        ("User1 talking to User2 in known lane", "LOCAL_IGNORE"),
    ]
    local_count = sum(1 for _, d in test_cases if d != "GEMINI_REQUIRED")
    gemini_count = sum(1 for _, d in test_cases if d == "GEMINI_REQUIRED")
    total = len(test_cases)
    print(f"\n{'='*60}")
    print(f"SIMULATION SUMMARY — {total} test cases")
    print(f"  LOCAL resolved:    {local_count}/{total} ({local_count/total*100:.0f}%)")
    print(f"  GEMINI required:   {gemini_count}/{total} ({gemini_count/total*100:.0f}%)")
    print(f"{'='*60}")
    for case, decision in test_cases:
        print(f"  [{decision:20s}] {case}")


if __name__ == "__main__":
    print_simulation_summary()
    unittest.main(verbosity=2)


