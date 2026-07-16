"""
EVE V6 Phase 11 — Memory Reliability + Context Quality Tests.

Verifies the two-phase atomic memory commit:
- Phase 1 (claim): mark_memory_in_progress isolates a batch
- Phase 2 success: mark_memory_processed finalises it (processed=1, in_progress=0)
- Phase 2 failure: mark_memory_failed rolls it back (in_progress=0) for retry
- get_unprocessed_* excludes in-progress rows (concurrent-safe)
- MemoryWorker._tick_user_memories retries on extractor failure
- MemoryWorker._tick_user_memories commits on extractor success
"""
from __future__ import annotations

import os
import sqlite3
import sys
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

# ── in-memory test DB ─────────────────────────────────────────────────────────
_TEST_DB_CONN: sqlite3.Connection | None = None


def _get_test_connection():
    global _TEST_DB_CONN
    if _TEST_DB_CONN is None:
        _TEST_DB_CONN = sqlite3.connect(
            "file:p11testdb?mode=memory&cache=shared", uri=True, check_same_thread=False
        )
        _TEST_DB_CONN.row_factory = sqlite3.Row
        _TEST_DB_CONN.execute("PRAGMA foreign_keys=ON")
    return _TEST_DB_CONN


import storage.database as db_module
import storage.messages as _msg_mod_ref

from storage.database import init_db
from storage import messages as msg_store


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _insert_raw(conn, msg_id: str, *, is_viewer: int = 0, text: str = "hello world") -> None:
    """Insert a raw MESSAGES row bypassing the NormalizedMessage model."""
    conn.execute(
        """INSERT OR IGNORE INTO MESSAGES
           (message_id, thread_id, sender_id, text, timestamp, item_type,
            is_sent_by_viewer, memory_processed, memory_in_progress, stored_at)
           VALUES (?, 'thread1', 'user1', ?, ?, 'text', ?, 0, 0, ?)""",
        (msg_id, text, _now(), is_viewer, _now()),
    )
    conn.commit()


class Phase11TestSetup(unittest.TestCase):
    def setUp(self):
        # Save original connection factories before patching
        self._orig_db_get = db_module.get_connection
        self._orig_msg_get = _msg_mod_ref.get_connection

        # Create a dedicated keep-alive connection for this test
        self._conn = sqlite3.connect(
            "file:p11testdb?mode=memory&cache=shared", uri=True, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys=ON")

        # Clear all tables
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
                "file:p11testdb?mode=memory&cache=shared", uri=True, check_same_thread=False
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=5000")
            return conn

        # Patch connection factory in both modules for this test
        db_module.get_connection = _get_conn
        _msg_mod_ref.get_connection = _get_conn
        self._get_conn = _get_conn

        init_db()

    def tearDown(self):
        # Restore original connection factories
        db_module.get_connection = self._orig_db_get
        _msg_mod_ref.get_connection = self._orig_msg_get
        self._conn.close()


class TestPhase11MemoryReliability(Phase11TestSetup):

    def test_01_claim_isolates_batch(self):
        """mark_memory_in_progress=1 prevents those rows from being re-fetched."""
        conn = self._get_conn()
        _insert_raw(conn, "m1")
        _insert_raw(conn, "m2")

        # Claim m1
        msg_store.mark_memory_in_progress(["m1"])

        # get_unprocessed should only return m2
        unprocessed = msg_store.get_unprocessed_for_memory(limit=10)
        ids = [m["message_id"] for m in unprocessed]
        self.assertNotIn("m1", ids, "In-progress message must be excluded from unprocessed fetch")
        self.assertIn("m2", ids, "Unclaimed message must still be visible")

    def test_02_success_commit_marks_done(self):
        """mark_memory_processed sets processed=1 AND in_progress=0."""
        conn = self._get_conn()
        _insert_raw(conn, "m3")

        msg_store.mark_memory_in_progress(["m3"])
        msg_store.mark_memory_processed(["m3"])

        row = conn.execute("SELECT memory_processed, memory_in_progress FROM MESSAGES WHERE message_id='m3'").fetchone()
        self.assertEqual(row["memory_processed"], 1)
        self.assertEqual(row["memory_in_progress"], 0)

    def test_03_failure_rollback_releases_claim(self):
        """mark_memory_failed resets in_progress=0 so the worker can retry."""
        conn = self._get_conn()
        _insert_raw(conn, "m4")

        msg_store.mark_memory_in_progress(["m4"])
        msg_store.mark_memory_failed(["m4"])

        row = conn.execute("SELECT memory_processed, memory_in_progress FROM MESSAGES WHERE message_id='m4'").fetchone()
        self.assertEqual(row["memory_processed"], 0, "Failed batch must not be marked processed")
        self.assertEqual(row["memory_in_progress"], 0, "Failed batch must release its claim")

    def test_04_failed_batch_reappears_for_next_tick(self):
        """After mark_memory_failed, the messages are returned on the next unprocessed query."""
        conn = self._get_conn()
        _insert_raw(conn, "m5", text="something substantial here")

        msg_store.mark_memory_in_progress(["m5"])
        msg_store.mark_memory_failed(["m5"])

        # Should now show up again
        unprocessed = msg_store.get_unprocessed_for_memory(limit=10)
        ids = [m["message_id"] for m in unprocessed]
        self.assertIn("m5", ids, "Failed message must be retried on the next tick")

    def test_05_eve_messages_also_excluded_when_in_progress(self):
        """get_unprocessed_eve_messages also guards against in-progress rows."""
        conn = self._get_conn()
        _insert_raw(conn, "e1", is_viewer=1, text="i really love reading")

        msg_store.mark_memory_in_progress(["e1"])

        unprocessed = msg_store.get_unprocessed_eve_messages(limit=10)
        ids = [m["message_id"] for m in unprocessed]
        self.assertNotIn("e1", ids, "In-progress Eve message must be excluded")

    @patch("workers.memory_worker.memory_extractor")
    def test_06_worker_tick_retries_on_extractor_failure(self, mock_extractor):
        """MemoryWorker._tick_user_memories rolls back on extractor exception."""
        conn = self._get_conn()
        _insert_raw(conn, "w1", text="atharv likes cricket so much")

        mock_extractor.extract_batch.side_effect = RuntimeError("Gemini timeout")

        from workers.memory_worker import MemoryWorker
        worker = MemoryWorker(poll_seconds=999, batch_size=10)
        worker._tick_user_memories()

        row = conn.execute("SELECT memory_processed, memory_in_progress FROM MESSAGES WHERE message_id='w1'").fetchone()
        self.assertEqual(row["memory_processed"], 0, "Failed extraction must not mark message processed")
        self.assertEqual(row["memory_in_progress"], 0, "Failed extraction must release the claim")

    @patch("workers.memory_worker.memory_extractor")
    def test_07_worker_tick_commits_on_success(self, mock_extractor):
        """MemoryWorker._tick_user_memories commits permanently on extractor success."""
        conn = self._get_conn()
        _insert_raw(conn, "w2", text="atharv studies at NIT Hamirpur")

        mock_extractor.extract_batch.return_value = 1

        from workers.memory_worker import MemoryWorker
        worker = MemoryWorker(poll_seconds=999, batch_size=10)
        worker._tick_user_memories()

        row = conn.execute("SELECT memory_processed, memory_in_progress FROM MESSAGES WHERE message_id='w2'").fetchone()
        self.assertEqual(row["memory_processed"], 1, "Successful extraction must mark message processed")
        self.assertEqual(row["memory_in_progress"], 0, "In-progress flag must be cleared after success")


if __name__ == "__main__":
    unittest.main()
