"""
Voice subsystem health tracking — V5.

Two independent gates:
1. Startup prerequisites (permanent for the process lifetime): missing
   GEMINI_LIVE_API_KEY, missing ffmpeg, or an installed instagrapi without
   direct_send_voice. If any are missing, voice is disabled for the whole
   run and the bot must still start in text-only mode (PART 26/22).
2. Runtime cooldown: repeated transient failures (Live timeout, empty
   audio, IG upload failure) put voice in a temporary cooldown so the bot
   doesn't hammer a broken path on every message, then it retries later.

Thread-safe (message_worker and main.py both touch this).
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger("yap.voice.health")


class VoiceHealth:
    def __init__(self, failure_threshold: int = 3, cooldown_seconds: float = 300.0):
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds

        self._lock = threading.Lock()
        self._startup_enabled = True
        self._disabled_reason: Optional[str] = None
        self._consecutive_failures = 0
        self._cooldown_until = 0.0

    def disable_permanently(self, reason: str) -> None:
        """Called at startup when a hard prerequisite is missing. Cannot be undone
        for this process — text-only operation continues normally."""
        with self._lock:
            self._startup_enabled = False
            self._disabled_reason = reason
        logger.warning("[VOICE] disabled for this run: %s", reason)

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._cooldown_until = 0.0

    def record_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._failure_threshold:
                self._cooldown_until = time.time() + self._cooldown_seconds
                logger.warning(
                    "[VOICE] entering cooldown for %.0fs after %d consecutive failures",
                    self._cooldown_seconds, self._consecutive_failures,
                )

    def is_healthy(self) -> bool:
        with self._lock:
            if not self._startup_enabled:
                return False
            return time.time() >= self._cooldown_until

    def status(self) -> dict:
        with self._lock:
            return {
                "startup_enabled": self._startup_enabled,
                "disabled_reason": self._disabled_reason,
                "consecutive_failures": self._consecutive_failures,
                "in_cooldown": time.time() < self._cooldown_until,
                "cooldown_remaining_s": max(0.0, self._cooldown_until - time.time()),
            }
