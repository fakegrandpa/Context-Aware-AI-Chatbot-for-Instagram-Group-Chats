"""
Chat Actor Subsystem — V6.
Serial mailbox event loop and room-level burst coalescer for a single chat thread.
Mutates dialogue state inside the mailbox thread and executes generation asynchronously.
"""
from __future__ import annotations

import concurrent.futures
import logging
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

import config
from conversation.burst import BurstGroup
from conversation.dialogue_state import DialogueState
from conversation.mode_selector import compute_energy_hint
from intelligence import voice_generator
from instagram import sender as ig_sender
from models.message import NormalizedMessage
from models.scene import SceneSnapshot
from storage import chat_state
from storage import eve_turns
from storage import messages as msg_store
from storage import profiles as prof_store
from voice import audio as voice_audio

from dataclasses import dataclass

logger = logging.getLogger("yap.conversation.chat_actor")


@dataclass
class GenerationLease:
    lease_id: str
    start_version: int
    session_id: str
    target_user_id: Optional[str]
    anchor_message_id: str
    status: str = "ACTIVE"  # ACTIVE, CANCELLED, INTERRUPTED
    cancellation_reason: Optional[str] = None


class ChatActor:
    """
    Serial event-loop mailbox thread owning all mutable dialogue state for a specific chat.
    Does NOT block its mailbox for slow network I/O or Gemini generation.
    """

    def __init__(
        self,
        thread_id: str,
        cl,
        bot_user_id: str,
        bot_username: str,
        lane_manager,
        fatigue_tracker,
        mode_selector,
        voice_health,
    ):
        self.thread_id = str(thread_id)
        self.cl = cl
        self.bot_user_id = str(bot_user_id)
        self.bot_username = str(bot_username)
        self.lane_manager = lane_manager
        self.fatigue_tracker = fatigue_tracker
        self.mode_selector = mode_selector
        self.voice_health = voice_health

        # V6 Live Dialogue State
        self.dialogue_state = DialogueState(self.thread_id, self.bot_user_id, self.bot_username)


        # Mailbox primitives
        self.event_queue: queue.Queue = queue.Queue()
        self.running = False
        self.mailbox_thread: Optional[threading.Thread] = None

        # ThreadPoolExecutor for background generation
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"actor-gen-{self.thread_id}")

        # Room burst observation state
        self.burst_messages: List[NormalizedMessage] = []
        self.burst_timer: Optional[threading.Timer] = None
        self.burst_timer_lock = threading.Lock()

        # Ephemeral generation tracking
        self.current_generation_active = False
        self.generation_snapshots: Dict[int, SceneSnapshot] = {}
        self.active_lease: Optional[GenerationLease] = None

    def start(self):
        """Start the actor's mailbox thread."""
        self.running = True
        self.mailbox_thread = threading.Thread(
            target=self._loop, daemon=True, name=f"chat-actor-{self.thread_id}"
        )
        self.mailbox_thread.start()
        logger.info("[ACTOR-%s] started mailbox thread", self.thread_id)
        self.post_event(("recover_state", None))

    def stop(self):
        """Signal the actor mailbox thread to stop and clean up resources."""
        self.post_event(("stop_actor", None))
        self.executor.shutdown(wait=False)
        with self.burst_timer_lock:
            if self.burst_timer:
                self.burst_timer.cancel()
        if getattr(self, "mailbox_thread", None) is not None and self.mailbox_thread.is_alive():
            self.mailbox_thread.join(timeout=2.0)
        logger.info("[ACTOR-%s] stopped actor", self.thread_id)

    def post_message(self, msg: NormalizedMessage):
        """Inbound normalized message entry point."""
        self.post_event(("inbound_message", msg))

    def post_event(self, event: tuple[str, any]):
        """Post a raw event into the actor's mailbox queue."""
        self.event_queue.put(event)

    def _loop(self):
        while self.running:
            try:
                event = self.event_queue.get(timeout=1.0)
                self._handle_event(event)
                self.event_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.exception("[ACTOR-%s] exception in mailbox loop: %s", self.thread_id, e)

    def _handle_event(self, event: tuple[str, any]):
        event_type, payload = event
        if event_type == "inbound_message":
            self._handle_inbound_message(payload)
        elif event_type == "burst_deadline":
            self._handle_burst_deadline()
        elif event_type == "generation_completed":
            self._handle_generation_completed(payload)
        elif event_type == "generation_failed":
            self._handle_generation_failed(payload)
        elif event_type == "recover_state":
            self._recover_state()
        elif event_type == "stop_actor":
            self.running = False

    def _recover_state(self):
        """Recover initial DialogueState by replaying recent messages in the thread."""
        try:
            logger.info("[ACTOR-%s] recovering room state from SQLite...", self.thread_id)
            
            # 1. Restore version from CHAT_STATE
            db_version = chat_state.get_room_version(self.thread_id)
            self.dialogue_state.room_version = db_version
            
            # 2. Get recent messages in thread (oldest-first)
            recent_msgs = msg_store.get_messages_for_thread(self.thread_id, limit=50)
            
            # Map database dictionaries to NormalizedMessage
            norm_messages = []
            for m in recent_msgs:
                ts = datetime.fromisoformat(m["timestamp"])
                if ts.tzinfo is None:
                    ts = ts.astimezone(timezone.utc)
                norm = NormalizedMessage(
                    message_id=str(m["message_id"]),
                    thread_id=str(m["thread_id"]),
                    sender_id=str(m["sender_id"]),
                    sender_username=m.get("sender_username") or str(m["sender_id"]),
                    text=m.get("text"),
                    timestamp=ts,
                    item_type=m.get("item_type", "text"),
                    is_sent_by_viewer=bool(m.get("is_sent_by_viewer")),
                    reply_to_message_id=m.get("reply_to_message_id"),
                    reply_to_user_id=m.get("reply_to_user_id"),
                    is_historical=True,
                )
                norm.conversation_id = m.get("conversation_id")
                norm_messages.append(norm)

            # Apply messages sequentially
            total_msgs = len(norm_messages)
            for idx, msg in enumerate(norm_messages):
                # Calculate sequential dummy versions backwards from current version
                mock_version = max(1, db_version - total_msgs + idx + 1)
                self.dialogue_state.apply_message(msg, mock_version)

            # 3. Retrieve EVE_TURNS to reconstruct structural Eve engagement
            recent_turns = eve_turns.get_recent_eve_turns_for_thread(self.thread_id, limit=5)
            if recent_turns:
                latest_turn = recent_turns[0]
                targets = {latest_turn["target_user_id"]} if latest_turn.get("target_user_id") else set()
                self.dialogue_state.eve_engagement.update_engagement(
                    session_id=latest_turn.get("conversation_id"),
                    target_ids=targets,
                    turn_id=latest_turn["trigger_message_id"],
                    version=latest_turn.get("conversation_version", db_version),
                )
                
                # Reconstruct StanceState from the most recent turn containing stance data
                stance_turn = next((t for t in recent_turns if t.get("stance")), None)
                if stance_turn:
                    self.dialogue_state.stance_state.commit(
                        stance=stance_turn.get("stance"),
                        speech_act=stance_turn.get("speech_act"),
                        intent_tag=stance_turn.get("intent_tag"),
                        target_user_id=stance_turn.get("target_user_id"),
                        session_id=stance_turn.get("session_id"),
                        turn_id=stance_turn["trigger_message_id"],
                        version=stance_turn.get("conversation_version", db_version),
                    )
                
            logger.info("[ACTOR-%s] recovery complete. version=%d active_sessions=%d", 
                        self.thread_id, self.dialogue_state.room_version, len(self.dialogue_state.active_sessions))
        except Exception as e:
            logger.error("[ACTOR-%s] failed to recover room state: %s", self.thread_id, e)

    def _handle_inbound_message(self, msg: NormalizedMessage):
        """Process inbound messages: acceptance, persistence, state mutation, and room-level burst."""
        # 1. Atomic ingress acceptance & version increment
        is_accepted, version = chat_state.accept_and_persist_message(msg)
        
        if not is_accepted:
            return  # Duplicate message, discard completely

        # 2. Update user profiles and active familiarity
        if not msg.is_sent_by_viewer:
            prof_store.get_or_create_user(msg.sender_id, msg.sender_username)
            prof_store.increment_message_count(msg.sender_id)
            prof_store.record_passive_activity(msg.sender_id)
            if msg.text:
                lang_style = prof_store.detect_language_style(msg.text)
                prof_store.update_language_style(msg.sender_id, lang_style)

        # 3. Mutate live DialogueState inside mailbox thread
        self.dialogue_state.apply_message(msg, version)

        # Check active lease revalidation delta
        if self.active_lease and self.active_lease.status == "ACTIVE":
            delta = self._classify_lease_delta(self.active_lease, msg)
            if delta == "HIGHER_PRIORITY":
                logger.info("[ACTOR-%s] Lease %s interrupted by higher priority message", self.thread_id, self.active_lease.lease_id)
                self.active_lease = GenerationLease(
                    lease_id=self.active_lease.lease_id,
                    start_version=self.active_lease.start_version,
                    session_id=self.active_lease.session_id,
                    target_user_id=self.active_lease.target_user_id,
                    anchor_message_id=self.active_lease.anchor_message_id,
                    status="INTERRUPTED",
                    cancellation_reason="Interrupt by new explicit message"
                )
            elif delta == "CANCELLING":
                logger.info("[ACTOR-%s] Lease %s cancelled", self.thread_id, self.active_lease.lease_id)
                self.active_lease = GenerationLease(
                    lease_id=self.active_lease.lease_id,
                    start_version=self.active_lease.start_version,
                    session_id=self.active_lease.session_id,
                    target_user_id=self.active_lease.target_user_id,
                    anchor_message_id=self.active_lease.anchor_message_id,
                    status="CANCELLED",
                    cancellation_reason="User cancelled"
                )
            elif delta == "MATERIAL":
                logger.info("[ACTOR-%s] Lease %s marked with material change", self.thread_id, self.active_lease.lease_id)

        # 4. Filter out bot's own turns from triggering responses
        if msg.is_sent_by_viewer or msg.sender_id == self.bot_user_id:
            # Bot sent message: cancel active burst timers and clear burst queue
            with self.burst_timer_lock:
                if self.burst_timer:
                    self.burst_timer.cancel()
                    self.burst_timer = None
            self.burst_messages = []
            return

        # 5. Room-Level Burst Window Coalescing
        self.burst_messages.append(msg)
        
        # Check direct address to Eve for shortening window
        is_direct = False
        if msg.reply_to_message_id and msg.reply_to_user_id == self.bot_user_id:
            is_direct = True
        else:
            resolved_target = self.dialogue_state._resolve_explicit_target(msg.text, msg.sender_username)
            if resolved_target == self.bot_user_id:
                is_direct = True

        with self.burst_timer_lock:
            if self.burst_timer:
                self.burst_timer.cancel()
                
            delay = 0.05 if is_direct else (config.BURST_WINDOW_MS / 1000.0)
            self.burst_timer = threading.Timer(delay, self._fire_burst_deadline)
            self.burst_timer.daemon = True
            self.burst_timer.start()

    def _fire_burst_deadline(self):
        """Callback from Timer thread to post BurstDeadline into mailbox."""
        self.post_event(("burst_deadline", None))

    def _handle_burst_deadline(self):
        """Mailbox event: evaluate burst window, capture snapshot, and dispatch async generation."""
        with self.burst_timer_lock:
            self.burst_timer = None

        if not self.burst_messages:
            return

        msgs = list(self.burst_messages)
        self.burst_messages = []

        trigger = msgs[-1]
        
        # Capture SceneSnapshot
        snapshot = self.dialogue_state.create_snapshot([m.message_id for m in msgs])
        self.generation_snapshots[snapshot.room_version] = snapshot

        # Construct legacy V5.5 BurstGroup
        burst_group = BurstGroup(
            thread_id=self.thread_id,
            sender_id=trigger.sender_id,
            messages=msgs
        )

        # 1. Resolve address and ownership using V6 AddressResolver
        from intelligence.address_resolver import AddressResolver
        resolver = AddressResolver(self.bot_user_id, self.bot_username)
        address_res = resolver.resolve(snapshot, msgs)
        
        # 2. Evaluate using ParticipationPolicy
        from intelligence.participation_policy import ParticipationPolicy
        policy = ParticipationPolicy(self.bot_user_id)
        decision = policy.evaluate(address_res, snapshot)
        
        if decision.mode in ("SUPPRESS", "DEFER"):
            logger.info("[ACTOR-%s] Participation policy decided %s. Exiting.", self.thread_id, decision.mode)
            self.generation_snapshots.pop(snapshot.room_version, None)
            return

        # 3. Create ParticipationOpportunity
        opp = policy.create_opportunity(decision, address_res, snapshot, msgs)
        
        # 4. Build GenerationPacket
        from intelligence.context_selector import ContextSelector
        from intelligence.prompts import EVE_CORE_INSTRUCTION
        selector = ContextSelector(self.bot_user_id, self.bot_username, EVE_CORE_INSTRUCTION)
        packet = selector.select(snapshot, opp)

        # 5. Create and register the active lease
        import uuid
        lease_id = str(uuid.uuid4())[:8]
        self.active_lease = GenerationLease(
            lease_id=lease_id,
            start_version=snapshot.room_version,
            session_id=opp.session_id,
            target_user_id=opp.target_user_id,
            anchor_message_id=opp.anchor_message_id
        )

        logger.info("[ACTOR-%s] burst window closed. Dispatching V6 generation pipeline asynchronously. Lease: %s, version=%d",
                    self.thread_id, lease_id, snapshot.room_version)

        # Dispatch slow generation outside the actor mailbox thread
        self.current_generation_active = True
        self.executor.submit(self._run_v6_generation, snapshot, burst_group, packet, lease_id)

    def _run_v6_generation(self, snapshot: SceneSnapshot, burst: BurstGroup, packet: GenerationPacket, lease_id: str):
        """Runs the single-call TurnComposer V6 pipeline outside ChatActor thread."""
        trigger = burst.trigger_message
        if not trigger:
            self.post_event(("generation_failed", {"version": snapshot.room_version}))
            return

        try:
            # 5. Run TurnComposer to get TurnProposal
            from intelligence.turn_composer import TurnComposer
            composer = TurnComposer(self.bot_user_id)
            proposal = composer.compose(packet)
            
            # 6. Check TurnComposer action
            if proposal.action == "IGNORE":
                logger.info("[ACTOR-%s] TurnComposer decided IGNORE. Exiting.", self.thread_id)
                self.post_event((
                    "generation_completed",
                    {
                        "reply_sent": False,
                        "lease_id": lease_id,
                        "originating_version": snapshot.room_version,
                    }
                ))
                return
                
            # 7. Select reply mode
            energetic = compute_energy_hint(burst.combined_text, tone=proposal.stance)
            mode = self.mode_selector.select_mode(voice_healthy=self.voice_health.is_healthy(), energetic=energetic)
            
            sent = None
            sent_mode = None
            reply_text = None
            voice_result = None
            
            # Legacy conv id
            legacy_conv_id = trigger.conversation_id or "legacy_session"
            
            if mode == "VOICE":
                # Convert to legacy for voice generator
                from models.context import ResponseContext, ReplyMetadata
                from intelligence.turn_planner import TurnPlan
                from intelligence import voice_generator
                
                reply_to_text = next((m.text for m in packet.recent_room_scene if m.message_id == packet.contract.anchor_message_id), None)
                reply_meta = ReplyMetadata(
                    reply_to_message_id=packet.contract.anchor_message_id,
                    reply_to_user_id=packet.contract.target_user_id,
                    reply_to_username=packet.target.display_name,
                    reply_to_text=reply_to_text
                )
                
                # Fetch stable facts from database if possible
                from storage import eve_state
                stable_facts = eve_state.get_stable_facts()
                
                legacy_ctx = ResponseContext(
                    sender_id=packet.contract.target_user_id or "unknown",
                    sender_username=packet.target.display_name or "unknown",
                    current_message=trigger.text or "",
                    current_message_id=packet.contract.anchor_message_id,
                    thread_id=self.thread_id,
                    bot_user_id=self.bot_user_id,
                    bot_username=self.bot_username,
                    eve_stable_facts=stable_facts,
                    sender_memories=[],
                    recent_gc_messages=[],
                    active_exchange_messages=[],
                    recent_eve_replies=[],
                    reply_metadata=reply_meta
                )
                
                legacy_plan = TurnPlan(
                    conversation_id=self.thread_id,
                    trigger_message_id=packet.contract.anchor_message_id,
                    target_user_id=packet.contract.target_user_id,
                    speech_act=proposal.speech_act,
                    intent=proposal.intent_tag,
                    stance=proposal.stance,
                    facts_to_use=[m.value for m in packet.memories],
                    continuity_notes=proposal.continuity_marker or "None",
                    avoid_topics=[],
                    conversation_version=snapshot.room_version
                )
                
                voice_result = voice_generator.generate_voice(legacy_ctx, plan=legacy_plan)
                if voice_result.success and voice_result.audio_path:
                    sent = ig_sender.send_voice(self.cl, self.thread_id, voice_result.audio_path)
                    voice_audio.cleanup_audio_file(voice_result.audio_path)
                    if sent:
                        sent_mode = "VOICE"
                        self.voice_health.record_success()
                        self.mode_selector.record("VOICE")
                        reply_text = voice_result.transcript
                    else:
                        self.voice_health.record_failure()
                else:
                    self.voice_health.record_failure()
                    
            if sent_mode is None:
                # Text fallback / default Text path
                reply_text = proposal.reply_text
                if not reply_text:
                    logger.warning("[ACTOR-%s] TurnComposer returned empty reply text.", self.thread_id)
                    self.post_event(("generation_failed", {"version": snapshot.room_version}))
                    return
                    
                trigger_dm = trigger.raw_dm or ig_sender.fetch_direct_message(self.cl, self.thread_id, trigger.message_id)
                sent = ig_sender.send_reply(
                    cl=self.cl,
                    thread_id=self.thread_id,
                    text=reply_text,
                    trigger_dm=trigger_dm,
                    strict=True,
                )
                if sent:
                    sent_mode = "TEXT"
                    self.mode_selector.record("TEXT")
                    
            if not sent:
                logger.error("[ACTOR-%s] send failed", self.thread_id)
                self.post_event(("generation_failed", {"version": snapshot.room_version}))
                return
                
            # 8. Post-send Bookkeeping (SQL message persist + Turns ledger)
            stored_text = reply_text if sent_mode == "TEXT" else "[voice message]"
            item_type = "text" if sent_mode == "TEXT" else "voice_media"
            sent_id = getattr(sent, "id", None) or f"eve_{int(time.time()*1000)}"
            
            eve_msg = NormalizedMessage(
                message_id=str(sent_id),
                thread_id=self.thread_id,
                sender_id=self.bot_user_id,
                sender_username=self.bot_username,
                text=stored_text,
                timestamp=datetime.now(timezone.utc),
                item_type=item_type,
                is_sent_by_viewer=True,
                reply_to_message_id=trigger.message_id,
                reply_to_user_id=trigger.sender_id,
                is_historical=False,
                conversation_id=legacy_conv_id,
            )
            msg_store.store_message(eve_msg)
            
            turn_transcript = getattr(voice_result, "transcript", None) if voice_result else None
            turn_summary = reply_text if sent_mode == "TEXT" else (turn_transcript or f"[Voice] Intent: {proposal.intent_tag}.")
            eve_turns.store_eve_turn(
                conversation_id=self.thread_id,
                trigger_message_id=trigger.message_id,
                target_user_id=trigger.sender_id,
                modality=sent_mode,
                semantic_summary=turn_summary,
                exact_text=reply_text,
                voice_transcript=turn_transcript,
                conversation_version=snapshot.room_version,
                session_id=legacy_conv_id,
                snapshot_version=snapshot.room_version,
                speech_act=proposal.speech_act,
                intent_tag=proposal.intent_tag,
                stance=proposal.stance,
                anchor_message_id=trigger.message_id,
            )
            
            self.fatigue_tracker.record_reply()
            
            # Dispatch success event back to actor mailbox
            self.post_event((
                "generation_completed",
                {
                    "reply_sent": True,
                    "turn_id": sent_id,
                    "target_user_id": trigger.sender_id,
                    "anchor_message_id": trigger.message_id,
                    "session_id": legacy_conv_id,
                    "text": stored_text,
                    "originating_version": snapshot.room_version,
                    "speech_act": proposal.speech_act,
                    "intent_tag": proposal.intent_tag,
                    "stance": proposal.stance,
                    "lease_id": lease_id,
                }
            ))
            
        except Exception as e:
            logger.exception("[ACTOR-%s] error in background V6 generation: %s", self.thread_id, e)
    def _classify_lease_delta(self, lease: GenerationLease, msg: NormalizedMessage) -> str:
        """
        Classifies how a new incoming message msg affects the active generation lease.
        Returns: "IRRELEVANT", "COMPATIBLE", "MATERIAL", "CANCELLING", "HIGHER_PRIORITY"
        """
        # 1. If it's a direct explicit address to Eve, it is HIGHER_PRIORITY (interrupt)
        is_direct_to_eve = False
        if msg.reply_to_user_id == self.bot_user_id:
            is_direct_to_eve = True
        else:
            resolved_target = self.dialogue_state._resolve_explicit_target(msg.text, msg.sender_username)
            if resolved_target == self.bot_user_id:
                is_direct_to_eve = True
                
        if is_direct_to_eve:
            return "HIGHER_PRIORITY"
            
        # 2. Check if the message contains cancelling keywords
        text_lower = (msg.text or "").lower()
        if any(kw in text_lower for kw in ["nevermind", "ignore", "don't reply", "dont reply", "cancel"]):
            return "CANCELLING"
            
        # 3. Check if the message is in the same session
        msg_session_id = self.dialogue_state.get_session_for_message(msg.message_id)
        if msg_session_id != lease.session_id:
            return "IRRELEVANT"
            
        # 4. If in the same session, check if it's from the same target user or someone else
        if msg.sender_id == lease.target_user_id:
            return "COMPATIBLE"
            
        # 5. Someone else is talking in the same session
        return "MATERIAL"

    def _revalidate_generation(self, lease: GenerationLease, reply_text: str) -> bool:
        """
        Returns True if the generated reply_text is still socially valid given the current state.
        """
        if self.dialogue_state.room_version == lease.start_version:
            return True
            
        # Basic revalidation check: if another message in the session has the exact same content,
        # or if we got an interrupt/cancellation.
        if lease.status in ("CANCELLED", "INTERRUPTED"):
            return False
            
        return True

    def _handle_generation_completed(self, payload: dict):
        """Mailbox event: generation finished successfully. Apply results to DialogueState."""
        self.current_generation_active = False
        
        reply_sent = payload.get("reply_sent", False)
        turn_id = payload.get("turn_id")
        target_user_id = payload.get("target_user_id")
        anchor_message_id = payload.get("anchor_message_id")
        session_id = payload.get("session_id")
        text = payload.get("text", "")
        originating_version = payload.get("originating_version", self.dialogue_state.room_version)
        ret_lease_id = payload.get("lease_id")
        
        # 1. Lease Revalidation
        if ret_lease_id:
            if self.active_lease is None or self.active_lease.lease_id != ret_lease_id:
                logger.warning("[ACTOR-%s] Generation completed for obsolete or missing lease: %s", self.thread_id, ret_lease_id)
                return
                
            if not self._revalidate_generation(self.active_lease, text):
                logger.warning("[ACTOR-%s] Lease %s failed revalidation (status=%s). Discarding response.", 
                               self.thread_id, self.active_lease.lease_id, self.active_lease.status)
                self.active_lease = None
                self.generation_snapshots.pop(originating_version, None)
                return

            # Perform Post-Validation Check
            from intelligence.post_validator import PostValidator
            validator = PostValidator(self.bot_user_id)
            recent_sent = [m.text for m in self.dialogue_state.recent_events if (m.is_sent_by_viewer or m.sender_id == self.bot_user_id) and m.text]
            val_res = validator.validate(text, self.active_lease, recent_sent)
            if not val_res.is_valid:
                logger.warning("[ACTOR-%s] Lease %s failed post-validation: %s. Discarding.", 
                               self.thread_id, self.active_lease.lease_id, val_res.reason)
                self.active_lease = None
                self.generation_snapshots.pop(originating_version, None)
                return
                
            self.active_lease = None

        # Pop snapshot
        self.generation_snapshots.pop(originating_version, None)

        if not reply_sent:
            return

        if reply_sent and turn_id:
            # Reconstruct the Viewer message and apply it to DialogueState
            eve_msg = NormalizedMessage(
                message_id=str(turn_id),
                thread_id=self.thread_id,
                sender_id=self.bot_user_id,
                sender_username=self.bot_username,
                text=text,
                timestamp=datetime.now(timezone.utc),
                item_type="text",
                is_sent_by_viewer=True,
                reply_to_message_id=anchor_message_id,
                reply_to_user_id=target_user_id,
                is_historical=False,
                conversation_id=session_id,
            )
            # Inbound-only room_version increment in Phase 1: do not increment version for Eve turns
            self.dialogue_state.apply_message(eve_msg, self.dialogue_state.room_version)
            
            # Update basic structural Eve Engagement state
            targets = {target_user_id} if target_user_id else set()
            self.dialogue_state.eve_engagement.update_engagement(
                session_id=session_id,
                target_ids=targets,
                turn_id=turn_id,
                version=originating_version,
                strength="HIGH",
            )
            
            # Commit stance state
            self.dialogue_state.stance_state.commit(
                stance=payload.get("stance"),
                speech_act=payload.get("speech_act"),
                intent_tag=payload.get("intent_tag"),
                target_user_id=target_user_id,
                session_id=session_id,
                turn_id=turn_id,
                version=originating_version,
            )
            logger.info("[ACTOR-%s] registered Eve sent turn_id=%s version=%d", 
                        self.thread_id, turn_id, self.dialogue_state.room_version)

    def _handle_generation_failed(self, payload: dict):
        """Mailbox event: generation sequence failed."""
        self.current_generation_active = False
        self.active_lease = None
        version = payload.get("version", 0)
        self.generation_snapshots.pop(version, None)
        logger.warning("[ACTOR-%s] generation task failed for version=%d", self.thread_id, version)
