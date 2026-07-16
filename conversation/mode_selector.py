"""
Reply mode selector — decides TEXT vs VOICE once the social routing layer
has already decided Eve should reply. V5, PARTS 11/19.

Purely local: no Gemini call, no fixed repeating pattern. Targets a
long-run ratio (~3/7 voice by default) via a proportional nudge against a
rolling window of recent modality history, with a small context-sensitive
boost for "energetic" moments. Voice is never selected when disabled,
unhealthy, or when the voice subsystem failed to start up (see voice/health.py).
"""
from __future__ import annotations

import logging
import random
import re
import threading
from collections import deque
from typing import Optional

import config
from models.decision import AttentionResult

logger = logging.getLogger("yap.conversation.mode_selector")

_ENERGETIC_EMOJI = set("😭💀😂🔥🗣️‼️⁉️🙏")


def compute_energy_hint(burst_text: str, attention: Optional[AttentionResult] = None, tone: Optional[str] = None) -> bool:
    """
    Cheap local heuristic for "this moment reads as energetic" — used only to
    nudge voice probability slightly, never to call Gemini. Any single signal
    is enough; this deliberately stays coarse.
    """
    text = burst_text or ""
    if text.count("!") >= 2:
        return True
    if len(text) > 3 and text.isupper():
        return True
    if sum(1 for ch in text if ch in _ENERGETIC_EMOJI) >= 2:
        return True
    if tone in ("PLAYFUL", "HOSTILE"):
        return True
    if attention is not None and attention.decision == "LOCAL_REPLY" and any(
        r in attention.reasons for r in ("native_reply_to_eve", "eve_active_lane")
    ):
        return True
    return False


class ModeSelector:
    """Thread-safe. One shared instance per running bot (see main.py)."""

    def __init__(self, target_ratio: Optional[float] = None, history_window: Optional[int] = None):
        self._target_ratio = target_ratio if target_ratio is not None else config.VOICE_TARGET_RATIO
        self._history_window = history_window or config.VOICE_HISTORY_WINDOW
        self._lock = threading.Lock()
        self._history: deque[str] = deque(maxlen=max(self._history_window * 3, 20))

    def _current_ratio(self) -> float:
        if not self._history:
            return self._target_ratio
        recent = list(self._history)[-self._history_window:]
        voice_count = sum(1 for m in recent if m == "VOICE")
        return voice_count / len(recent)

    def select_mode(self, voice_healthy: bool, energetic: bool = False) -> str:
        if not config.VOICE_ENABLED or not voice_healthy:
            return "TEXT"

        with self._lock:
            ratio = self._current_ratio()

        # Proportional controller: if we're under target, nudge probability
        # up; if over target, nudge down. Never a hard fixed pattern.
        delta = self._target_ratio - ratio
        probability = self._target_ratio + delta * 0.6
        if energetic:
            probability += 0.15
        probability = max(0.05, min(0.9, probability))

        mode = "VOICE" if random.random() < probability else "TEXT"
        return mode

    def record(self, mode: str) -> None:
        with self._lock:
            self._history.append(mode)

    def stats(self) -> dict:
        with self._lock:
            return {
                "history_len": len(self._history),
                "current_ratio": self._current_ratio(),
                "target_ratio": self._target_ratio,
            }
