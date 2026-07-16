"""
Instagram sender — wraps native reply and plain text send paths.

Native reply:
  cl.direct_send(text, thread_ids=[thread_id], reply_to_message=DirectMessage)

This requires a full DirectMessage object with .id and .client_context.
For MQTT-received triggers (where we have only message_id), we fetch the
DirectMessage via cl.direct_message() to get the full object.

Fallback:
  cl.direct_answer(thread_id, text) — plain text, no reply bubble.

Errors in native reply attempt fall back to normal send.
The main send pipeline is never broken by native reply failures.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

from instagrapi import Client
from instagrapi.types import DirectMessage

logger = logging.getLogger("yap.instagram.sender")

# Lock to synchronize cache access and prefetch registration
_CACHE_LOCK = threading.RLock()

# Bounded trigger DM cache by message_id (normalized string key)
_DM_CACHE: dict[str, tuple[float, DirectMessage]] = {}
CACHE_TTL = 300  # 5 minutes

# In-progress prefetch registry: message_id (normalized string) -> threading.Event
_PREFETCH_EVENTS: dict[str, threading.Event] = {}


def normalize_message_id(message_id: Any) -> str:
    """Normalize message IDs to a consistent clean string format."""
    if message_id is None:
        return ""
    s = str(message_id).strip()
    if s.endswith(".0"):
        try:
            return str(int(float(s)))
        except ValueError:
            pass
    return s


def get_cached_dm(message_id: str) -> Optional[DirectMessage]:
    """Retrieve a cached DirectMessage if not expired."""
    norm_id = normalize_message_id(message_id)
    now = time.time()
    with _CACHE_LOCK:
        if norm_id in _DM_CACHE:
            expiry, dm = _DM_CACHE[norm_id]
            if now < expiry:
                return dm
            else:
                del _DM_CACHE[norm_id]
    return None


def set_cached_dm(message_id: str, dm: DirectMessage) -> None:
    """Store a DirectMessage in the cache with a TTL."""
    if not message_id or dm is None:
        return
    norm_key = normalize_message_id(message_id)
    now = time.time()
    with _CACHE_LOCK:
        _DM_CACHE[norm_key] = (now + CACHE_TTL, dm)
        # Also cache under returned dm.id if available and different to prevent mismatch misses
        if hasattr(dm, "id") and dm.id:
            norm_dm_id = normalize_message_id(dm.id)
            if norm_dm_id != norm_key:
                _DM_CACHE[norm_dm_id] = (now + CACHE_TTL, dm)


def _fetch_and_cache(cl: Client, thread_id: str, message_id: str) -> Optional[DirectMessage]:
    """Perform the targeted synchronous HTTP fetch and cache the result."""
    norm_id = normalize_message_id(message_id)
    try:
        t_id = int(normalize_message_id(thread_id))
        m_id = int(norm_id)
        dm = cl.direct_message(t_id, m_id)
        if dm:
            set_cached_dm(norm_id, dm)
            return dm
    except Exception as e:
        logger.warning("[SEND] HTTP fetch failed for message_id=%s: %s", norm_id, e)
    return None


def prefetch_direct_message(cl: Client, thread_id: str, message_id: str) -> None:
    """
    Fire-and-forget background prefetch of the native DirectMessage object.
    Uses a threading.Event to signal other threads when the fetch completes.
    """
    norm_id = normalize_message_id(message_id)
    
    with _CACHE_LOCK:
        if get_cached_dm(norm_id) is not None:
            return
        if norm_id in _PREFETCH_EVENTS:
            return
        # Register a new prefetch event
        event = threading.Event()
        _PREFETCH_EVENTS[norm_id] = event

    def _run_prefetch():
        try:
            logger.info("[PREFETCH] Starting background prefetch for message_id=%s", norm_id)
            _fetch_and_cache(cl, thread_id, norm_id)
        except Exception as e:
            logger.warning("[PREFETCH] Background prefetch failed for message_id=%s: %s", norm_id, e)
        finally:
            with _CACHE_LOCK:
                event.set()
                _PREFETCH_EVENTS.pop(norm_id, None)

    threading.Thread(target=_run_prefetch, daemon=True, name=f"dm-prefetch-{norm_id}").start()


def fetch_direct_message(
    cl: Client,
    thread_id: str,
    message_id: str,
    timeout: float = 3.0,
) -> Optional[DirectMessage]:
    """
    Fetch a single DirectMessage object by thread + message_id.
    Checks local cache first, waits for background prefetch if in-progress,
    and falls back to synchronous HTTP targeted resolution if still missing.
    """
    norm_id = normalize_message_id(message_id)

    # 1. Cache HIT check
    cached = get_cached_dm(norm_id)
    if cached:
        logger.info("[SEND] Cache HIT for trigger message_id=%s", norm_id)
        return cached

    # 2. Wait for background prefetch if in progress
    event = None
    with _CACHE_LOCK:
        event = _PREFETCH_EVENTS.get(norm_id)

    if event is not None:
        logger.info("[SEND] Prefetch in progress for message_id=%s, waiting up to %s seconds...", norm_id, timeout)
        event.wait(timeout=timeout)
        cached = get_cached_dm(norm_id)
        if cached:
            logger.info("[SEND] Cache HIT after waiting for prefetch for message_id=%s", norm_id)
            return cached
        else:
            logger.warning("[SEND] Prefetch did not resolve message_id=%s within timeout", norm_id)

    # 3. Synchronous targeted resolution fallback
    logger.info("[SEND] Cache MISS for trigger message_id=%s, fetching over HTTP...", norm_id)
    return _fetch_and_cache(cl, thread_id, norm_id)


def send_reply(
    cl: Client,
    thread_id: str,
    text: str,
    trigger_message_id: Optional[str] = None,
    trigger_dm: Optional[DirectMessage] = None,
    strict: bool = False,
) -> Optional[DirectMessage]:
    """
    Send a reply to a thread, optionally as a native reply bubble.

    Attempts native reply if either trigger_dm (already fetched) or
    trigger_message_id (cached/will fetch) is provided.

    If strict=True, does not fall back to plain send if native reply cannot be sent.
    Returns the sent DirectMessage, or None on failure.
    """
    reply_dm: Optional[DirectMessage] = trigger_dm
    if reply_dm is None and trigger_message_id:
        reply_dm = fetch_direct_message(cl, thread_id, trigger_message_id)

    # Attempt native reply
    if reply_dm is not None and reply_dm.client_context:
        try:
            sent = cl.direct_send(
                text=text,
                thread_ids=[int(thread_id)],
                reply_to_message=reply_dm,
            )
            logger.info("[SEND] native reply message_id=%s", reply_dm.id)
            return sent
        except Exception as e:
            logger.warning("[SEND] native reply failed (%s)", e)
            if strict:
                logger.error("[SEND] strict mode active: skipping fallback plain send")
                return None

    if strict:
        logger.error("[SEND] strict mode active: reply target unavailable; skipping text reply")
        return None

    # Fallback: plain send (only if strict=False)
    try:
        sent = cl.direct_answer(int(thread_id), text)
        logger.info("[SEND] normal send thread_id=%s", thread_id)
        return sent
    except Exception as e:
        logger.error("[SEND] all send methods failed: %s", e)
        return None


def send_text(cl: Client, thread_id: str, text: str) -> Optional[DirectMessage]:
    """
    Simple plain text send (no reply bubble). Wrapper over direct_answer.
    """
    try:
        return cl.direct_answer(int(thread_id), text)
    except Exception as e:
        logger.error("[SEND] send_text failed: %s", e)
        return None


def send_voice(cl: Client, thread_id: str, m4a_path) -> Optional[DirectMessage]:
    """
    Send a native Instagram voice-message DM (AAC/M4A required).

    The installed instagrapi's direct_send_voice() has no reply-to-message
    parameter — voice notes cannot be sent as a native reply bubble the way
    text can (this is a real, currently-verified limitation of the library,
    not a decision we made). The voice note is simply sent to the thread;
    Eve's own routing/context already knows which message triggered it.

    Returns the sent DirectMessage, or None on failure (caller falls back to
    a single text reply — see workers/message_worker.py).
    """
    try:
        sent = cl.direct_send_voice(path=Path(m4a_path), thread_ids=[int(thread_id)])
        logger.info("[SEND] voice sent thread_id=%s", thread_id)
        return sent
    except Exception as e:
        logger.error("[SEND] voice send failed: %s", e)
        return None
