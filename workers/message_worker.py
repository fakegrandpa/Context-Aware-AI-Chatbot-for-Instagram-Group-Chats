"""
Message worker — consumes BurstGroups from the event queue and
orchestrates the full processing pipeline:

  BurstGroup received
    -> persist all messages in burst
    -> auto-create/update user profiles
    -> assign conversation lane
    -> evaluate local attention gate
    -> if GEMINI_REQUIRED: call social judge on the RAW recent scene (not lane-filtered)
    -> if REPLY: build canonical ResponseContext (shared by text + voice)
    -> pre-send staleness check
    -> select reply mode (TEXT/VOICE)
    -> generate + send (voice failure falls back to exactly one text reply)
    -> record fatigue / familiarity / modality history
    -> save state

The MQTT callback only normalizes and enqueues — no Gemini calls there.
This worker consumes from the queue on its own thread.
"""
from __future__ import annotations

import logging
import queue
import re
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Set

from instagrapi import Client

import config
from conversation.attention import evaluate as attention_evaluate
from conversation.burst import BurstGroup
from conversation.fatigue import FatigueTracker
from conversation.lanes import LaneManager
from conversation.mode_selector import ModeSelector, compute_energy_hint
from conversation.room import RoomStateEngine, PendingTurn
from intelligence import context_builder, response_generator, social_judge, voice_generator
from intelligence import turn_planner
from instagram import sender as ig_sender
from models.message import NormalizedMessage
from storage import messages as msg_store
from storage import profiles as prof_store
from storage import memories as mem_store
from storage import eve_turns
from voice import audio as voice_audio
from voice.health import VoiceHealth

logger = logging.getLogger("yap.workers.message_worker")

# Staleness cancellation phrases (conservative list)
_CANCEL_PHRASES = re.compile(
    r'\b(nvm|nevermind|never mind|forget it|forget that|ignore that|my bad|mb)\b',
    re.IGNORECASE
)

VOICE_MESSAGE_PLACEHOLDER_TEXT = "[voice message]"


class MessageWorker:
    """
    Consumes BurstGroups and drives the full message pipeline.

    Args:
        cl: Authenticated instagrapi Client.
        thread_id: Target GC thread ID.
        bot_user_id: Eve's Instagram user_id.
        bot_username: Eve's Instagram username (the real IG handle, not the character name).
        processed_ids: Shared set of processed message IDs (for dedup with legacy state).
        last_ts_container: [datetime] wrapper for cross-thread last_ts mutation.
        state_saver: Callable to persist state.json.
        lane_manager: Shared LaneManager instance.
        fatigue_tracker: Shared FatigueTracker instance.
        mode_selector: Shared ModeSelector (TEXT/VOICE decision).
        voice_health: Shared VoiceHealth tracker (may be permanently disabled at startup).
    """

    def __init__(
        self,
        cl: Client,
        thread_id: str,
        bot_user_id: str,
        bot_username: str,
        processed_ids: Set[str],
        last_ts_container: List[datetime],
        state_saver: Callable,
        lane_manager: LaneManager,
        fatigue_tracker: FatigueTracker,
        mode_selector: Optional[ModeSelector] = None,
        voice_health: Optional[VoiceHealth] = None,
        room_engine: Optional[RoomStateEngine] = None,
    ):
        self._cl = cl
        self._thread_id = thread_id
        self._bot_user_id = bot_user_id
        self._bot_username = bot_username
        self._processed_ids = processed_ids
        self._last_ts_container = last_ts_container
        self._state_saver = state_saver
        self._lane_manager = lane_manager
        self._fatigue_tracker = fatigue_tracker
        self._mode_selector = mode_selector or ModeSelector()
        self._voice_health = voice_health or VoiceHealth()
        self._room_engine = room_engine or RoomStateEngine(bot_user_id, bot_username)

        # Thread-safe queue of BurstGroups
        self._queue: queue.Queue[BurstGroup] = queue.Queue()
        self._lock = threading.Lock()

        # Eve's own recent sent message IDs — used by attention gate
        self._recent_bot_msg_ids: deque[str] = deque(maxlen=50)
        # Eve's recent TEXT reply texts — for anti-repetition (voice replies
        # aren't transcribed, so they don't feed this — see PART 15/16).
        self._recent_bot_replies: deque[str] = deque(maxlen=10)

        # Track latest message per lane for staleness check
        # lane_id → latest NormalizedMessage processed
        self._lane_latest: Dict[str, NormalizedMessage] = {}
        self._lane_lock = threading.Lock()

    def enqueue(self, burst: BurstGroup) -> None:
        """Add a BurstGroup to the processing queue."""
        burst.enqueued_at = time.perf_counter()
        self._queue.put_nowait(burst)

    def start(self) -> threading.Thread:
        """Start the worker thread. Returns the thread."""
        t = threading.Thread(target=self._run, daemon=True, name="message-worker")
        t.start()
        return t

    def get_recent_bot_msg_ids(self) -> List[str]:
        with self._lock:
            return list(self._recent_bot_msg_ids)

    def _run(self) -> None:
        logger.info("[WORKER] message worker started")
        while True:
            try:
                burst: BurstGroup = self._queue.get(timeout=1.0)
                self._process_burst(burst)
            except queue.Empty:
                continue
            except Exception as e:
                logger.exception("[WORKER] unhandled exception in worker loop: %s", e)

    def _process_burst(self, burst: BurstGroup) -> None:
        """Full pipeline for one BurstGroup with precise latency timing and room context."""
        if not burst.messages:
            return

        trigger = burst.trigger_message
        if trigger is None:
            return

        start_processing = time.perf_counter()
        queue_delay = start_processing - getattr(burst, "enqueued_at", start_processing)
        burst_coalescing_wait = getattr(burst, "enqueued_at", start_processing) - getattr(burst, "created_at_perf", start_processing)

        # --- 1. Link conversation threads & Persist all messages + auto-create profiles ---
        start_persist = time.perf_counter()
        any_new = False
        for msg in burst.messages:
            if msg.raw_dm:
                ig_sender.set_cached_dm(msg.message_id, msg.raw_dm)
            elif not msg.is_sent_by_viewer and msg.item_type == "text":
                self._prefetch_trigger_dm(msg.message_id)

            with self._lock:
                if msg.message_id in self._processed_ids:
                    logger.debug("[WORKER] already processed message_id=%s", msg.message_id)
                    continue

            # Online thread linking: assign conversation_id BEFORE storing message
            conv_id = self._room_engine.process_incoming_message(msg, self._cl)
            msg.conversation_id = conv_id

            inserted = msg_store.store_message(msg)
            if inserted and not msg.is_sent_by_viewer:
                prof_store.get_or_create_user(msg.sender_id, msg.sender_username)
                prof_store.increment_message_count(msg.sender_id)
                prof_store.record_passive_activity(msg.sender_id)
                if msg.text:
                    lang_style = prof_store.detect_language_style(msg.text)
                    prof_store.update_language_style(msg.sender_id, lang_style)
                any_new = True

            with self._lock:
                self._processed_ids.add(msg.message_id)
                if msg.timestamp > self._last_ts_container[0]:
                    self._last_ts_container[0] = msg.timestamp

        # Always save state after processing
        self._state_saver()
        persist_duration = time.perf_counter() - start_persist

        # Skip further pipeline for Eve's own messages
        if trigger.is_sent_by_viewer:
            with self._lock:
                self._recent_bot_msg_ids.append(trigger.message_id)
                if trigger.text:
                    self._recent_bot_replies.append(trigger.text)
            return

        # Update fatigue tracking: human spoke
        self._fatigue_tracker.record_human_message()

        if not any_new:
            return

        # Only process text bursts through the social pipeline
        burst_text = burst.combined_text
        if not burst_text.strip():
            return

        if trigger.item_type != "text":
            return

        detection_delay = (datetime.now(timezone.utc) - trigger.timestamp).total_seconds()

        # Fetch conversation thread messages to check ownership
        thread_msgs = msg_store.get_messages_for_conversation(trigger.conversation_id)

        # --- 2. Turn Ownership Resolver ---
        ownership_class, target_user_id = self._room_engine.ownership_resolver.resolve_ownership(trigger, thread_msgs)
        if ownership_class == "UNCLEAR":
            logger.info("[WORKER] Turn ownership is UNCLEAR. Silently ignoring.")
            return
        if ownership_class == "SPECIFIC_USER":
            logger.info("[WORKER] Turn ownership is SPECIFIC_USER=%s (not Eve). Silently ignoring.", target_user_id)
            return

        # If OPEN_GROUP, run the InterventionRouter
        if ownership_class == "OPEN_GROUP":
            if not self._room_engine.router.should_intervene(trigger):
                logger.info("[WORKER] InterventionRouter decided not to intervene in OPEN_GROUP turn.")
                return

        # --- 3. Resolve reply graph context (usernames, not raw ids) ---
        reply_to_username: Optional[str] = None
        reply_to_text: Optional[str] = None
        if trigger.reply_to_message_id:
            target_msg = msg_store.get_message_by_id(trigger.reply_to_message_id)
            if target_msg:
                reply_to_username = prof_store.resolve_display_name(target_msg.get("sender_id", ""))
                reply_to_text = target_msg.get("text")
            if trigger.reply_to_user_id == self._bot_user_id:
                prof_store.update_familiarity(trigger.sender_id, prof_store.FAMILIARITY_DELTA_REPLY_TO_YAP)

        # Re-fetch scene messages for ResponseContext (PART 7/10)
        scene_msgs = msg_store.get_recent_scene(self._thread_id, limit=config.SOCIAL_SCENE_SIZE)

        # Bridged lane context manager for backward compatibility
        start_lane = time.perf_counter()
        lane = self._lane_manager.assign_lane(trigger)
        lane_duration = time.perf_counter() - start_lane

        with self._lane_lock:
            if lane:
                self._lane_latest[lane.lane_id] = trigger

        # Active exchange messages are the conversation thread messages
        active_exchange_msgs = thread_msgs
        recent_replies = list(self._recent_bot_replies)

        start_ctx = time.perf_counter()
        ctx = context_builder.build_response_context(
            sender_id=trigger.sender_id,
            sender_username=trigger.sender_username,
            current_message=burst_text,
            current_message_id=trigger.message_id,
            scene_messages=scene_msgs,
            active_exchange_messages=active_exchange_msgs,
            recent_eve_replies=recent_replies,
            reply_to_username=reply_to_username,
            reply_to_text=reply_to_text,
            reply_to_message_id=trigger.reply_to_message_id,
            reply_to_user_id=trigger.reply_to_user_id,
            routing_context={"ownership": ownership_class},
            thread_id=self._thread_id,
            bot_user_id=self._bot_user_id,
            bot_username=self._bot_username,
        )
        context_build_duration = time.perf_counter() - start_ctx

        # --- 4. Pre-send staleness check (before spending on generation) ---
        if self._is_stale(trigger, lane):
            logger.info("[STALE] discarded response trigger_message_id=%s before generation", trigger.message_id)
            return

        # --- 5. Generate shared TurnPlan (intent parity) ---
        conversation_version = len(thread_msgs)
        plan = turn_planner.generate_turn_plan(ctx, conversation_version)
        if plan is None:
            logger.error("[WORKER] failed to generate TurnPlan")
            return

        # Register pending turn in coordinator
        turn_id = str(uuid.uuid4())[:8]
        pending_turn = PendingTurn(
            turn_id=turn_id,
            conversation_id=trigger.conversation_id,
            trigger_message_id=trigger.message_id,
            target_user_id=trigger.sender_id,
            plan=plan,
            conversation_version=conversation_version
        )
        self._room_engine.coordinator.register_turn(pending_turn)
        self._room_engine.coordinator.set_status(turn_id, "GENERATING")

        # --- 6. Select reply mode ---
        from conversation.mode_selector import compute_energy_hint as local_energy_hint
        from models.decision import AttentionResult
        dummy_att = AttentionResult(decision="LOCAL_REPLY", reasons=["eve_active_lane"])
        energetic = local_energy_hint(burst_text, attention=dummy_att, tone=plan.stance)
        
        mode = self._mode_selector.select_mode(voice_healthy=self._voice_health.is_healthy(), energetic=energetic)

        sent = None
        sent_mode: Optional[str] = None
        reply_text: Optional[str] = None
        voice_timings = None
        gen_duration = 0.0
        fetch_duration = 0.0
        send_duration = 0.0

        if pending_turn.status == "SUPERSEDED":
            logger.info("[PENDING] turn_id=%s superseded before generation, aborting", turn_id)
            return

        # --- 7a. VOICE path ---
        if mode == "VOICE":
            start_voice = time.perf_counter()
            voice_result = voice_generator.generate_voice(ctx, plan=plan)
            voice_gen_total = time.perf_counter() - start_voice
            voice_timings = voice_result

            if voice_result.success and voice_result.audio_path:
                # Pre-send revalidation check
                newer_count = msg_store.count_newer_messages_in_conversation(trigger.conversation_id, trigger.message_id)
                if newer_count > 0:
                    logger.info("[STALE] pending turn_id=%s cancelled: conversation advanced by %d messages since trigger", turn_id, newer_count)
                    self._room_engine.coordinator.set_status(turn_id, "CANCELLED")
                    voice_audio.cleanup_audio_file(voice_result.audio_path)
                    return

                if pending_turn.status == "SUPERSEDED":
                    logger.info("[PENDING] turn_id=%s superseded before send, aborting", turn_id)
                    voice_audio.cleanup_audio_file(voice_result.audio_path)
                    return

                self._room_engine.coordinator.set_status(turn_id, "READY")
                start_send = time.perf_counter()
                sent = ig_sender.send_voice(self._cl, self._thread_id, voice_result.audio_path)
                send_duration = time.perf_counter() - start_send
                voice_audio.cleanup_audio_file(voice_result.audio_path)

                if sent:
                    sent_mode = "VOICE"
                    self._voice_health.record_success()
                    self._mode_selector.record("VOICE")
                    self._room_engine.coordinator.set_status(turn_id, "SENT")
                else:
                    logger.warning("[VOICE] instagram send failed, falling back to text")
                    self._voice_health.record_failure()
            else:
                logger.info("[VOICE] generation failed (%s), falling back to text", voice_result.failure_reason)
                self._voice_health.record_failure()

        # --- 7b. TEXT path ---
        if sent_mode is None:
            start_gen = time.perf_counter()
            reply_text, gen_duration = response_generator.generate_from_context(ctx, plan=plan)

            if not reply_text:
                logger.warning("[WORKER] no reply generated for burst from %s", trigger.sender_username)
                self._room_engine.coordinator.set_status(turn_id, "FAILED")
                return

            # Pre-send revalidation check
            newer_count = msg_store.count_newer_messages_in_conversation(trigger.conversation_id, trigger.message_id)
            if newer_count > 0:
                logger.info("[STALE] pending turn_id=%s cancelled: conversation advanced by %d messages since trigger", turn_id, newer_count)
                self._room_engine.coordinator.set_status(turn_id, "CANCELLED")
                return

            if pending_turn.status == "SUPERSEDED":
                logger.info("[PENDING] turn_id=%s superseded before send, aborting", turn_id)
                return

            self._room_engine.coordinator.set_status(turn_id, "READY")
            start_fetch = time.perf_counter()
            trigger_dm = trigger.raw_dm or ig_sender.fetch_direct_message(self._cl, self._thread_id, trigger.message_id)
            fetch_duration = time.perf_counter() - start_fetch

            start_send = time.perf_counter()
            sent = ig_sender.send_reply(
                cl=self._cl,
                thread_id=self._thread_id,
                text=reply_text,
                trigger_dm=trigger_dm,
                strict=True,
            )
            send_duration = time.perf_counter() - start_send

            if sent:
                sent_mode = "TEXT"
                self._mode_selector.record("TEXT")
                self._room_engine.coordinator.set_status(turn_id, "SENT")

        if not sent:
            logger.error("[WORKER] send failed for burst from %s (mode=%s)", trigger.sender_username, mode)
            self._room_engine.coordinator.set_status(turn_id, "FAILED")
            return

        # --- 8. Post-send bookkeeping & Unified Eve Turn Ledger ---
        stored_text = reply_text if sent_mode == "TEXT" else VOICE_MESSAGE_PLACEHOLDER_TEXT
        item_type = "text" if sent_mode == "TEXT" else "voice_media"

        with self._lock:
            if sent_mode == "TEXT" and reply_text:
                self._recent_bot_replies.append(reply_text)
            if hasattr(sent, "id") and sent.id:
                self._processed_ids.add(sent.id)
                self._recent_bot_msg_ids.append(sent.id)
                if hasattr(sent, "timestamp") and sent.timestamp:
                    ts = sent.timestamp
                    if ts.tzinfo is None:
                        ts = ts.astimezone(timezone.utc)
                    if ts > self._last_ts_container[0]:
                        self._last_ts_container[0] = ts

        # Store Eve's reply in MESSAGES table
        if hasattr(sent, "id") and sent.id:
            eve_msg = NormalizedMessage(
                message_id=str(sent.id),
                thread_id=self._thread_id,
                sender_id=self._bot_user_id,
                sender_username=self._bot_username,
                text=stored_text,
                timestamp=datetime.now(timezone.utc),
                item_type=item_type,
                is_sent_by_viewer=True,
                reply_to_message_id=trigger.message_id,
                reply_to_user_id=trigger.sender_id,
                is_historical=False,
                conversation_id=trigger.conversation_id,
            )
            msg_store.store_message(eve_msg)

        # Store Eve's reply in Unified Eve Turn Ledger (EVE_TURNS)
        turn_transcript = voice_result.transcript if (sent_mode == "VOICE" and voice_result) else None
        turn_summary = reply_text if sent_mode == "TEXT" else (turn_transcript or f"[Voice] Intent: {plan.intent}. Stance: {plan.stance}.")
        eve_turns.store_eve_turn(
            conversation_id=trigger.conversation_id,
            trigger_message_id=trigger.message_id,
            target_user_id=trigger.sender_id,
            modality=sent_mode,
            semantic_summary=turn_summary,
            exact_text=reply_text,
            voice_transcript=turn_transcript,
            conversation_version=conversation_version,
        )

        prof_store.update_familiarity(trigger.sender_id, prof_store.FAMILIARITY_DELTA_YAP_REPLY)
        self._fatigue_tracker.record_reply()
        self._state_saver()

        total_processing_time = time.perf_counter() - start_processing
        overall_confirm_duration = (datetime.now(timezone.utc) - trigger.timestamp).total_seconds()

        if sent_mode == "VOICE" and voice_timings is not None:
            logger.info(
                "[LATENCY:VOICE] msg_id=%s queue=%.3fs burst_wait=%.3fs persist=%.3fs lane=%.3fs "
                "context=%.3fs connect=%.3fs first_chunk=%.3fs voice_gen=%.3fs convert=%.3fs "
                "send=%.3fs total=%.3fs overall_confirm=%.3fs",
                trigger.message_id, queue_delay, burst_coalescing_wait, persist_duration, lane_duration,
                context_build_duration, voice_timings.connect_duration, voice_timings.first_chunk_duration,
                voice_timings.generation_duration, voice_timings.conversion_duration,
                send_duration, total_processing_time, overall_confirm_duration,
            )
            logger.info("Eve replied (voice)")
        else:
            logger.info(
                "[LATENCY:TEXT] msg_id=%s queue=%.3fs burst_wait=%.3fs persist=%.3fs lane=%.3fs "
                "context=%.3fs gen=%.3fs fetch=%.3fs send=%.3fs total=%.3fs overall_confirm=%.3fs",
                trigger.message_id, queue_delay, burst_coalescing_wait, persist_duration, lane_duration,
                context_build_duration, gen_duration, fetch_duration, send_duration, total_processing_time, overall_confirm_duration,
            )
            logger.info("Eve replied: %s", reply_text)

    def _get_active_exchange_messages(self, lane, scene_msgs: List[dict]) -> List[dict]:
        """
        Optional focused highlight of the current lane, ADDITIVE to the full
        raw scene already passed to response generation — never a
        replacement (that participant-only filtering was the tunnel-vision
        bug this repair removes). Skipped entirely when the lane is weak,
        Eve-less, or effectively identical to the full scene already shown.
        """
        if lane is None or lane.strength == "weak" or len(lane.participants) < 2:
            return []
        participants = list(lane.participants)
        lane_msgs = msg_store.get_lane_messages(
            self._thread_id, participants, limit=config.CONTEXT_LANE_SIZE
        )
        scene_ids = {m.get("message_id") for m in scene_msgs}
        lane_ids = {m.get("message_id") for m in lane_msgs}
        if lane_ids <= scene_ids:
            # Nothing the lane highlights isn't already visible in the raw
            # scene — skip the redundant section entirely.
            return []
        return lane_msgs

    def _prefetch_trigger_dm(self, message_id: str) -> None:
        """
        Delegate to ig_sender.prefetch_direct_message which handles
        background thread creation and synchronization events.
        """
        ig_sender.prefetch_direct_message(self._cl, self._thread_id, message_id)

    def _is_stale(self, trigger: NormalizedMessage, lane) -> bool:
        """
        Conservative staleness check. Only discards if same sender sent
        a clear cancellation after Eve started generating.
        Does NOT discard merely because unrelated GC messages arrived.
        """
        if lane is None:
            return False

        with self._lane_lock:
            latest = self._lane_latest.get(lane.lane_id)

        if latest is None:
            return False

        # Same sender, newer message in same lane after trigger
        if (latest.message_id != trigger.message_id
                and latest.sender_id == trigger.sender_id
                and latest.timestamp > trigger.timestamp):
            text = latest.text or ""
            if _CANCEL_PHRASES.search(text):
                return True

        return False
