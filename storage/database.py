"""
SQLite database initialization and connection factory.

Threading strategy: short-lived per-operation connections (check_same_thread=False,
WAL mode, busy_timeout). Each call to get_connection() opens a new connection, uses
it, and closes it. This is safe from all threads simultaneously because WAL mode
handles concurrent reads/writes at the SQLite level, and we never share a connection
object across threads.

Do NOT cache or share a single connection across threads.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import config

logger = logging.getLogger("yap.storage.database")

DB_PATH: Path = config.DB_PATH


def get_connection() -> sqlite3.Connection:
    """
    Open a new SQLite connection with WAL mode, foreign keys, and busy timeout.
    Caller is responsible for closing (use as context manager or explicit .close()).
    """
    conn = sqlite3.connect(
        str(DB_PATH),
        check_same_thread=False,
        timeout=5.0,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_db() -> None:
    """
    Create all tables and indexes if they do not already exist.
    Safe to call on every startup (idempotent) and handles schema migrations.
    """
    with get_connection() as conn:
        # Create user profiles, messages, relationships, and state tables
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS USERS (
                user_id         TEXT PRIMARY KEY,
                username        TEXT NOT NULL,
                display_name    TEXT,
                preferred_name  TEXT,
                first_seen      TEXT NOT NULL,
                last_seen       TEXT NOT NULL,
                message_count   INTEGER DEFAULT 0,
                language_style  TEXT,
                relationship_to_yap TEXT DEFAULT 'new',
                familiarity_score   REAL DEFAULT 0.0
            );

            CREATE TABLE IF NOT EXISTS MESSAGES (
                message_id          TEXT PRIMARY KEY,
                thread_id           TEXT NOT NULL,
                sender_id           TEXT NOT NULL,
                text                TEXT,
                timestamp           TEXT NOT NULL,
                item_type           TEXT NOT NULL,
                is_sent_by_viewer   INTEGER NOT NULL,
                reply_to_message_id TEXT,
                reply_to_user_id    TEXT,
                memory_processed    INTEGER DEFAULT 0,
                stored_at           TEXT NOT NULL,
                conversation_id     TEXT
            );

            CREATE TABLE IF NOT EXISTS EVE_STATE (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                state_type          TEXT NOT NULL CHECK(state_type IN ('stable','dynamic')),
                slot                TEXT NOT NULL,
                value               TEXT NOT NULL,
                confidence          REAL NOT NULL DEFAULT 1.0,
                source_message_id   TEXT,
                created_at          TEXT NOT NULL,
                updated_at          TEXT NOT NULL,
                active              INTEGER DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS RELATIONSHIPS (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                user_a_id           TEXT NOT NULL,
                user_b_id           TEXT NOT NULL,
                relationship_type   TEXT NOT NULL,
                summary             TEXT,
                confidence          REAL NOT NULL,
                updated_at          TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS BOT_STATE (
                key         TEXT PRIMARY KEY,
                value       TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS EVE_TURNS (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id     TEXT NOT NULL,
                trigger_message_id  TEXT NOT NULL,
                target_user_id      TEXT,
                modality            TEXT NOT NULL CHECK(modality IN ('TEXT','VOICE')),
                semantic_summary    TEXT NOT NULL,
                exact_text          TEXT,
                voice_transcript    TEXT,
                created_at          TEXT NOT NULL,
                conversation_version INTEGER NOT NULL,
                session_id          TEXT,
                snapshot_version    INTEGER,
                speech_act          TEXT,
                intent_tag          TEXT,
                stance              TEXT,
                anchor_message_id   TEXT
            );

            CREATE TABLE IF NOT EXISTS CHAT_STATE (
                thread_id               TEXT PRIMARY KEY,
                room_version            INTEGER NOT NULL DEFAULT 0,
                last_message_timestamp  TEXT,
                updated_at              TEXT NOT NULL
            );
        """)

        # Check schema of MEMORIES table for migration
        table_info = conn.execute("PRAGMA table_info(MEMORIES)").fetchall()
        if not table_info:
            # Table does not exist, create it with V4 Claim/Belief schema
            conn.execute("""
                CREATE TABLE MEMORIES (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id             TEXT NOT NULL,
                    memory_type         TEXT NOT NULL CHECK(memory_type IN ('identity','preference','personal_fact','relationship','episodic')),
                    slot                TEXT NOT NULL,
                    value               TEXT NOT NULL,
                    normalized_fact     TEXT NOT NULL,
                    status              TEXT NOT NULL CHECK(status IN ('candidate', 'active', 'conflicted', 'superseded', 'rejected')),
                    claim_type          TEXT,
                    support_count       INTEGER DEFAULT 1,
                    contradiction_count INTEGER DEFAULT 0,
                    confidence          REAL NOT NULL,
                    source_message_id   TEXT,
                    created_at          TEXT NOT NULL,
                    updated_at          TEXT NOT NULL,
                    active              INTEGER DEFAULT 1
                );
            """)
        else:
            cols = [row[1] for row in table_info]
            if "slot" not in cols or "status" not in cols:
                logger.info("[DB] Migrating MEMORIES table to V4 Claim/Belief schema...")
                conn.execute("ALTER TABLE MEMORIES RENAME TO MEMORIES_old")
                conn.execute("""
                    CREATE TABLE MEMORIES (
                        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id             TEXT NOT NULL,
                        memory_type         TEXT NOT NULL CHECK(memory_type IN ('identity','preference','personal_fact','relationship','episodic')),
                        slot                TEXT NOT NULL,
                        value               TEXT NOT NULL,
                        normalized_fact     TEXT NOT NULL,
                        status              TEXT NOT NULL CHECK(status IN ('candidate', 'active', 'conflicted', 'superseded', 'rejected')),
                        claim_type          TEXT,
                        support_count       INTEGER DEFAULT 1,
                        contradiction_count INTEGER DEFAULT 0,
                        confidence          REAL NOT NULL,
                        source_message_id   TEXT,
                        created_at          TEXT NOT NULL,
                        updated_at          TEXT NOT NULL,
                        active              INTEGER DEFAULT 1
                    );
                """)
                # Migrating the old table, map old 'fact' to 'value', parse/default 'slot'
                conn.execute("""
                    INSERT INTO MEMORIES (
                        id, user_id, memory_type, slot, value, normalized_fact, status,
                        claim_type, support_count, contradiction_count, confidence,
                        source_message_id, created_at, updated_at, active
                    )
                    SELECT 
                        id, user_id, memory_type, 
                        CASE WHEN memory_type = 'identity' THEN 'name' ELSE 'general' END as slot,
                        fact as value, normalized_fact,
                        CASE WHEN active = 1 THEN 'active' ELSE 'superseded' END as status,
                        'NEW' as claim_type, 1 as support_count, 0 as contradiction_count, confidence,
                        source_message_id, created_at, updated_at, active
                    FROM MEMORIES_old
                """)
                conn.execute("DROP TABLE MEMORIES_old")
                logger.info("[DB] Migration complete!")

        # Check schema of MESSAGES table for conversation_id column migration
        messages_info = conn.execute("PRAGMA table_info(MESSAGES)").fetchall()
        messages_cols = [row[1] for row in messages_info]
        if "conversation_id" not in messages_cols:
            logger.info("[DB] Migrating MESSAGES table to include conversation_id...")
            conn.execute("ALTER TABLE MESSAGES ADD COLUMN conversation_id TEXT")
            conn.commit()

        # Phase 11: add memory_in_progress sentinel column
        messages_info = conn.execute("PRAGMA table_info(MESSAGES)").fetchall()
        messages_cols = [row[1] for row in messages_info]
        if "memory_in_progress" not in messages_cols:
            logger.info("[DB] Migrating MESSAGES table to include memory_in_progress...")
            conn.execute("ALTER TABLE MESSAGES ADD COLUMN memory_in_progress INTEGER DEFAULT 0")
            conn.commit()

        # Check schema of EVE_TURNS table for V6 migration
        turns_info = conn.execute("PRAGMA table_info(EVE_TURNS)").fetchall()
        if turns_info:
            cols = [row[1] for row in turns_info]
            new_cols = [
                ("session_id", "TEXT"),
                ("snapshot_version", "INTEGER"),
                ("speech_act", "TEXT"),
                ("intent_tag", "TEXT"),
                ("stance", "TEXT"),
                ("anchor_message_id", "TEXT")
            ]
            for col_name, col_type in new_cols:
                if col_name not in cols:
                    logger.info("[DB] Migrating EVE_TURNS: adding column %s...", col_name)
                    conn.execute(f"ALTER TABLE EVE_TURNS ADD COLUMN {col_name} {col_type}")

        # Create indexes
        conn.executescript("""
            CREATE INDEX IF NOT EXISTS idx_messages_thread_ts
                ON MESSAGES(thread_id, timestamp);
            CREATE INDEX IF NOT EXISTS idx_messages_sender
                ON MESSAGES(sender_id);
            CREATE INDEX IF NOT EXISTS idx_messages_reply_to
                ON MESSAGES(reply_to_message_id);
            CREATE INDEX IF NOT EXISTS idx_messages_memory
                ON MESSAGES(memory_processed);
            CREATE INDEX IF NOT EXISTS idx_memories_user
                ON MEMORIES(user_id, status);
            CREATE INDEX IF NOT EXISTS idx_eve_state_active
                ON EVE_STATE(state_type, active);
            CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON MESSAGES(conversation_id);
            CREATE INDEX IF NOT EXISTS idx_eve_turns_conversation
                ON EVE_TURNS(conversation_id);
        """)
    logger.info("[DB] initialized at %s", DB_PATH)
