"""
Social fatigue tracker — prevents Yap from dominating active GCs.

Tracks Yap's recent participation and produces a fatigue multiplier that
raises the threshold for joining ambiguous/group conversations.

Fatigue NEVER blocks:
- LOCAL_REPLY (direct address, native reply to Yap)
- Explicit @yap summons

It only influences GEMINI_REQUIRED and borderline cases.
"""
from __future__ import annotations

import logging
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("yap.conversation.fatigue")


class FatigueTracker:
    """
    Tracks Yap's reply history and computes a fatigue multiplier.

    fatigue_multiplier returns a value in [0.0, 1.0]:
    - 0.0 = no fatigue (normal sensitivity)
    - 1.0 = maximum fatigue (very reluctant to join ambiguous conversations)

    Thread-safe.
    """

    def __init__(
        self,
        max_replies_60s: int = 4,
        max_replies_5min: int = 10,
        max_consecutive: int = 5,
    ):
        self._max_60s = max_replies_60s
        self._max_5min = max_replies_5min
        self._max_consecutive = max_consecutive

        self._lock = threading.Lock()
        # Rolling window of reply timestamps
        self._reply_times: deque[datetime] = deque(maxlen=200)
        self._consecutive_turns: int = 0
        self._last_reply_ts: Optional[datetime] = None

    def record_reply(self) -> None:
        """Call this whenever Yap sends a reply."""
        with self._lock:
            now = datetime.now(timezone.utc)
            self._reply_times.append(now)
            self._consecutive_turns += 1
            self._last_reply_ts = now

    def record_human_message(self) -> None:
        """
        Call when a non-Yap message is processed.
        Resets consecutive turn counter.
        """
        with self._lock:
            self._consecutive_turns = 0

    def get_fatigue_multiplier(self) -> float:
        """
        Returns a fatigue value in [0.0, 1.0].
        Higher = more fatigued = less willing to join ambiguous conversations.
        """
        with self._lock:
            return self._compute_fatigue()

    def get_stats(self) -> dict:
        """Return current fatigue stats for logging."""
        with self._lock:
            now = datetime.now(timezone.utc)
            r60 = self._count_in_window(now, 60)
            r5m = self._count_in_window(now, 300)
            return {
                "replies_60s": r60,
                "replies_5min": r5m,
                "consecutive_turns": self._consecutive_turns,
                "last_reply_ts": self._last_reply_ts.isoformat() if self._last_reply_ts else None,
                "fatigue": round(self._compute_fatigue(), 3),
            }

    # --- Private ---

    def _count_in_window(self, now: datetime, seconds: int) -> int:
        """Count replies in the last `seconds` seconds."""
        cutoff = now.timestamp() - seconds
        return sum(1 for t in self._reply_times if t.timestamp() >= cutoff)

    def _compute_fatigue(self) -> float:
        """Compute fatigue score. Call under lock."""
        now = datetime.now(timezone.utc)
        r60 = self._count_in_window(now, 60)
        r5m = self._count_in_window(now, 300)
        consec = self._consecutive_turns

        # Normalized sub-scores
        score_60s = min(1.0, r60 / max(self._max_60s, 1))
        score_5min = min(1.0, r5m / max(self._max_5min, 1))
        score_consec = min(1.0, consec / max(self._max_consecutive, 1))

        # Weighted average (60s activity is most impactful)
        fatigue = 0.5 * score_60s + 0.3 * score_5min + 0.2 * score_consec
        return round(min(1.0, fatigue), 3)

    @property
    def last_reply_ts(self) -> Optional[datetime]:
        with self._lock:
            return self._last_reply_ts
