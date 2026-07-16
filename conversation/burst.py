"""
Burst coalescing — prevents rapid successive messages from the same sender
being processed as independent social events.

A burst window (default 600ms) collects consecutive messages from the same
sender in the same thread. When the window closes, they are emitted as a
BurstGroup for unified processing.

Thread-safe: all state is protected by a lock. The burst timer fires on a
background thread and enqueues a BurstGroup via a callback.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, List, Optional

from models.message import NormalizedMessage

logger = logging.getLogger("yap.conversation.burst")


@dataclass
class BurstGroup:
    """A group of coalesced messages from the same sender in the same thread."""
    thread_id: str
    sender_id: str
    messages: List[NormalizedMessage] = field(default_factory=list)

    @property
    def trigger_message(self) -> Optional[NormalizedMessage]:
        """The latest message in the burst — used as the native reply target."""
        return self.messages[-1] if self.messages else None

    @property
    def combined_text(self) -> str:
        """All non-empty texts joined — represents the full burst intent."""
        parts = [m.text for m in self.messages if m.text and m.text.strip()]
        return " ".join(parts)

    @property
    def earliest_timestamp(self) -> Optional[datetime]:
        if not self.messages:
            return None
        return min(m.timestamp for m in self.messages)


class BurstCoalescer:
    """
    Manages in-memory burst windows per (thread_id, sender_id) pair.
    When the window expires, calls emit_callback(BurstGroup).

    Args:
        window_ms: Burst window in milliseconds. Default 600ms.
        emit_callback: Called with the completed BurstGroup when window closes.
    """

    def __init__(self, window_ms: int = 600, emit_callback: Optional[Callable[[BurstGroup], None]] = None):
        self._window_ms = window_ms
        self._emit_callback = emit_callback
        self._lock = threading.Lock()
        # Key: (thread_id, sender_id) → BurstGroup
        self._pending: dict[tuple[str, str], BurstGroup] = {}
        # Key: (thread_id, sender_id) → Timer
        self._timers: dict[tuple[str, str], threading.Timer] = {}

    def add(self, msg: NormalizedMessage) -> None:
        """
        Add a message to its burst window. Resets the timer if already pending.
        Only coalesces text messages; other types pass through immediately.
        """
        key = (msg.thread_id, msg.sender_id)

        with self._lock:
            if msg.item_type != "text":
                # Non-text items are never coalesced — emit immediately as solo burst
                solo = BurstGroup(
                    thread_id=msg.thread_id,
                    sender_id=msg.sender_id,
                    messages=[msg],
                )
                solo.created_at_perf = time.perf_counter()
                self._schedule_emit(key, solo, immediately=True)
                return

            if key in self._pending:
                # Extend existing burst: cancel old timer, append message
                self._timers[key].cancel()
                self._pending[key].messages.append(msg)
                logger.debug("[BURST] extended key=%s count=%d", key, len(self._pending[key].messages))
            else:
                # Start new burst
                group = BurstGroup(
                    thread_id=msg.thread_id,
                    sender_id=msg.sender_id,
                    messages=[msg],
                )
                group.created_at_perf = time.perf_counter()
                self._pending[key] = group
                logger.debug("[BURST] started key=%s", key)

            # Schedule (or re-schedule) the emit timer
            timer = threading.Timer(
                self._window_ms / 1000.0,
                self._fire,
                args=(key,),
            )
            self._timers[key] = timer
            timer.daemon = True
            timer.start()

    def _fire(self, key: tuple[str, str]) -> None:
        """Called by the timer thread when the burst window expires."""
        with self._lock:
            group = self._pending.pop(key, None)
            self._timers.pop(key, None)

        if group and self._emit_callback:
            logger.debug("[BURST] emitting key=%s msgs=%d", key, len(group.messages))
            try:
                self._emit_callback(group)
            except Exception as e:
                logger.exception("[BURST] emit_callback raised: %s", e)

    def _schedule_emit(self, key: tuple[str, str], group: BurstGroup, immediately: bool = False) -> None:
        """Schedule an immediate or windowed emit (called under lock)."""
        if immediately:
            # Fire without waiting for the window
            thread = threading.Thread(target=self._emit_direct, args=(group,), daemon=True)
            thread.start()
        else:
            timer = threading.Timer(
                self._window_ms / 1000.0,
                self._fire,
                args=(key,),
            )
            self._timers[key] = timer
            timer.daemon = True
            timer.start()

    def _emit_direct(self, group: BurstGroup) -> None:
        if self._emit_callback:
            try:
                self._emit_callback(group)
            except Exception as e:
                logger.exception("[BURST] direct emit raised: %s", e)
