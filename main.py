"""
Eve V5 — main.py (orchestration)

Responsibilities:
- Initialize SQLite database
- Log in to Instagram (reuse session)
- Bootstrap state safely (no replies to historical messages)
- Start realtime MQTT subscriber thread
- Start message worker thread
- Start memory worker thread
- MQTT callback: normalize → burst coalescer → queue (no Gemini here)
- Fallback HTTP polling loop (when MQTT is down)
- Shutdown on KeyboardInterrupt

The MQTT callback is intentionally fast — it normalizes the message,
adds it to the burst coalescer, and returns. All social processing
happens in the message worker thread.
"""
import json
import logging
import socket
import ssl
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone

import config
import instagram_client
from conversation.burst import BurstCoalescer
from conversation.fatigue import FatigueTracker
from conversation.lanes import LaneManager
from conversation.mode_selector import ModeSelector
from models.message import normalize_http, normalize_realtime
from storage.database import init_db
from storage import eve_state
from storage import messages as msg_store
from storage import profiles as prof_store
from voice import audio as voice_audio
from voice.health import VoiceHealth
from workers.memory_worker import MemoryWorker
from conversation.chat_actor_registry import ChatActorRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("yap.main")

MAX_PROCESSED_IDS = 200

realtime_ready_event = threading.Event()
username_map_cache: dict = {}


def load_state():
    if config.STATE_PATH.exists():
        with open(config.STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_state(state):
    with open(config.STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def parse_ts(iso_str: str) -> datetime:
    ts = datetime.fromisoformat(iso_str)
    if ts.tzinfo is None:
        ts = ts.astimezone(timezone.utc)
    return ts


def bootstrap_state(thread_id: str, messages) -> dict:
    if messages:
        newest_ts = max(m.timestamp for m in messages)
        if newest_ts.tzinfo is None:
            newest_ts = newest_ts.astimezone(timezone.utc)
    else:
        newest_ts = datetime.now(timezone.utc)
    state = {
        "thread_id": str(thread_id),
        "last_timestamp": newest_ts.isoformat(),
        "processed_ids": [],
    }
    save_state(state)
    logger.info("First run for this thread: ignoring %d existing message(s)", len(messages))
    return state


def bootstrap_db_from_history(messages, username_map: dict, last_ts: datetime, bot_user_id: str) -> None:
    """
    Store historical messages in SQLite as bootstrap context.
    All are marked as historical (memory_processed=1, is_historical=True).
    No replies are sent.
    """
    stored = 0
    for m in messages:
        m_time = m.timestamp
        if m_time.tzinfo is None:
            m_time = m_time.astimezone(timezone.utc)
        if m_time > last_ts:
            continue  # Only bootstrap messages at or before last_ts

        uid = str(m.user_id) if hasattr(m, "user_id") else ""
        username = username_map.get(uid, uid)
        username_map_cache[uid] = username

        norm = normalize_http(m, config.TARGET_THREAD_ID, bot_user_id, username_map_cache)
        if norm is None:
            continue
        norm.is_historical = True

        inserted = msg_store.store_message(norm)
        if inserted and not norm.is_sent_by_viewer and norm.sender_id:
            prof_store.get_or_create_user(norm.sender_id, norm.sender_username)
            prof_store.increment_message_count(norm.sender_id)
            stored += 1

    logger.info("[BOOTSTRAP] stored %d historical messages in SQLite", stored)


def realtime_thread_loop(
    cl,
    thread_id: str,
    processed_ids: set,
    last_ts_container: list,
    state_saver,
    registry: ChatActorRegistry,
    lane_manager: LaneManager,
    fatigue_tracker: FatigueTracker,
    mode_selector: ModeSelector,
    voice_health: VoiceHealth,
):
    while True:
        try:
            realtime_ready_event.clear()
            logger.info("[REALTIME] connecting")
            realtime = cl.realtime_connect()
            logger.info("[REALTIME] connected")

            def on_message_event(payload):
                # Fast path: normalize + enqueue only. No Gemini here.
                if not payload or not isinstance(payload, dict):
                    return
                m_dict = payload.get("message")
                if not m_dict or not isinstance(m_dict, dict):
                    return

                event_thread_id = str(m_dict.get("thread_id", ""))
                if event_thread_id != str(thread_id):
                    return

                op = m_dict.get("op")
                if op != "add":
                    return

                logger.debug("[REALTIME] raw event received")

                norm = normalize_realtime(m_dict, config.BOT_USER_ID, username_map_cache)
                if norm is None:
                    return

                # Fast dedup check
                if norm.message_id in processed_ids:
                    logger.debug("[REALTIME] already processed message_id=%s", norm.message_id)
                    return

                m_time = norm.timestamp
                if m_time <= last_ts_container[0]:
                    logger.debug("[REALTIME] old message, skipping message_id=%s", norm.message_id)
                    return

                if not norm.is_sent_by_viewer:
                    logger.info("[REALTIME] target message sender=%s type=%s",
                                norm.sender_username, norm.item_type)

                # Route to registry
                registry.route_message(
                    norm,
                    cl,
                    config.BOT_USER_ID,
                    cl.username or config.IG_USERNAME,
                    lane_manager,
                    fatigue_tracker,
                    mode_selector,
                    voice_health,
                )

            realtime._handlers["message"].clear()
            realtime.on("message", on_message_event)

            logger.info("[REALTIME] subscribing to direct")
            try:
                realtime.direct_subscribe()
            except Exception as e:
                logger.error("[REALTIME] direct subscription failed: %s", e)
                realtime_ready_event.clear()
                raise e

            realtime_ready_event.set()

            last_ping = time.time()
            while realtime_ready_event.is_set():
                if time.time() - last_ping > 20:
                    try:
                        realtime.ping()
                        last_ping = time.time()
                    except Exception as e:
                        logger.error("[REALTIME] ping failed: %s", e)
                        realtime_ready_event.clear()
                        break

                try:
                    realtime.read_once()
                except (socket.timeout, TimeoutError, ssl.SSLError):
                    try:
                        realtime.ping()
                        last_ping = time.time()
                    except Exception as e_ping:
                        logger.error("[REALTIME] ping after timeout failed: %s", e_ping)
                        realtime_ready_event.clear()
                        break
                except Exception as e:
                    logger.error("[REALTIME] read_once error: %s", e)
                    realtime_ready_event.clear()
                    break

        except Exception as e:
            logger.error("[REALTIME] disconnected (error: %s)", e)
            realtime_ready_event.clear()

        if not realtime_ready_event.is_set():
            logger.info("[REALTIME] reconnecting in 5s")
            time.sleep(5)


def main():
    from intelligence import gemini_pool
    # Verify we have at least one valid Gemini key configured
    pool = gemini_pool.get_pool()
    if not pool._keys:
        logger.error("[GEMINI-POOL] Startup Error: No valid Gemini API keys found. Please set GEMINI_API_KEY_1 through GEMINI_API_KEY_5 or GEMINI_API_KEY in .env.")
        sys.exit(1)

    if not config.TARGET_THREAD_ID:
        logger.error(
            "TARGET_THREAD_ID is not set in .env. Run list_threads.py to find your thread ID."
        )
        sys.exit(1)

    thread_id = config.TARGET_THREAD_ID

    # Initialize SQLite
    init_db()
    eve_state.ensure_stable_facts_seeded()

    logger.info("Logging in to Instagram...")
    cl = instagram_client.build_client()
    logger.info("Logged in as %s (user_id=%s)", cl.username, cl.user_id)

    # Set runtime bot identity in config
    config.BOT_USER_ID = str(cl.user_id)
    config.BOT_NAME = "Eve"

    logger.info("Fetching thread %s...", thread_id)
    messages, username_map = instagram_client.fetch_thread(cl, thread_id, amount=40)

    # Populate username cache
    for uid, uname in username_map.items():
        username_map_cache[str(uid)] = uname

    # Load or bootstrap state.json (unchanged behavior)
    state = load_state()
    if not state or state.get("thread_id") != str(thread_id):
        state = bootstrap_state(thread_id, messages)

    last_ts = parse_ts(state["last_timestamp"])
    processed_ids: set = set(state.get("processed_ids", []))
    last_ts_container = [last_ts]

    # Bootstrap SQLite from historical messages (safe — no replies)
    bootstrap_db_from_history(messages, username_map, last_ts, config.BOT_USER_ID)

    # Shared state
    lane_manager = LaneManager(bot_user_id=config.BOT_USER_ID)
    fatigue_tracker = FatigueTracker(
        max_replies_60s=config.FATIGUE_MAX_REPLIES_60S,
        max_replies_5min=config.FATIGUE_MAX_REPLIES_5MIN,
        max_consecutive=config.FATIGUE_MAX_CONSECUTIVE,
    )

    def state_saver():
        nonlocal processed_ids
        ids_list = list(processed_ids)
        if len(ids_list) > MAX_PROCESSED_IDS:
            ids_list = ids_list[-MAX_PROCESSED_IDS:]
            processed_ids = set(ids_list)
        save_state({
            "thread_id": str(thread_id),
            "last_timestamp": last_ts_container[0].isoformat(),
            "processed_ids": ids_list,
        })

    # Initialize Voice Health and Mode Selector
    voice_health = VoiceHealth(
        failure_threshold=config.VOICE_FAILURE_THRESHOLD,
        cooldown_seconds=config.VOICE_COOLDOWN_SECONDS
    )
    mode_selector = ModeSelector(
        target_ratio=config.VOICE_TARGET_RATIO,
        history_window=config.VOICE_HISTORY_WINDOW
    )

    # Perform Voice health startup prerequisites checks
    if not config.VOICE_ENABLED:
        voice_health.disable_permanently("disabled in config")
    else:
        if not voice_audio.check_ffmpeg_available():
            voice_health.disable_permanently(f"ffmpeg not found on PATH or FFMPEG_PATH ({config.FFMPEG_PATH})")
        elif not hasattr(cl, "direct_send_voice"):
            voice_health.disable_permanently("instagrapi client does not support direct_send_voice")
        else:
            from intelligence.voice_generator import resolve_live_api_key
            if not resolve_live_api_key():
                voice_health.disable_permanently("no Gemini API key available for Live voice generation")



    # Initialize V6 ChatActorRegistry
    registry = ChatActorRegistry()

    # Start workers
    MemoryWorker().start()

    # Start realtime MQTT thread
    rt_thread = threading.Thread(
        target=realtime_thread_loop,
        args=(
            cl,
            thread_id,
            processed_ids,
            last_ts_container,
            state_saver,
            registry,
            lane_manager,
            fatigue_tracker,
            mode_selector,
            voice_health,
        ),
        daemon=True,
    )
    rt_thread.start()

    logger.info("Yap V4 is watching thread %s.", thread_id)

    fallback_logged_active = False

    while True:
        if not realtime_ready_event.is_set():
            if not fallback_logged_active:
                logger.info("[FALLBACK] polling active")
                fallback_logged_active = True

            try:
                polled_messages, polled_username_map = instagram_client.fetch_thread(
                    cl, thread_id, amount=40
                )
                for uid, uname in polled_username_map.items():
                    username_map_cache[str(uid)] = uname

                for m in polled_messages:
                    norm = normalize_http(
                        m, thread_id, config.BOT_USER_ID, username_map_cache
                    )
                    if norm is None:
                        continue
                    if norm.message_id in processed_ids:
                        continue
                    m_time = norm.timestamp
                    if m_time <= last_ts_container[0]:
                        continue
                    if norm.item_type != "text" or not norm.text:
                        continue

                    # Route to V6 registry
                    registry.route_message(
                        norm,
                        cl,
                        config.BOT_USER_ID,
                        cl.username or config.IG_USERNAME,
                        lane_manager,
                        fatigue_tracker,
                        mode_selector,
                        voice_health,
                    )

            except Exception as e:
                logger.error("[FALLBACK] poll failed (will retry): %s", e)

            time.sleep(config.POLL_INTERVAL_SECONDS)
        else:
            if fallback_logged_active:
                logger.info("[FALLBACK] polling stopped")
                fallback_logged_active = False
            time.sleep(10)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Stopped by user")
