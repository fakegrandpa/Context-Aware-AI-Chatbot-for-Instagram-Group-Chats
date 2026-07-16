"""
Background memory worker — polls for unprocessed messages and runs
batched memory extraction without blocking the realtime pipeline.

Completely independent from the message pipeline. Errors are caught
at the batch level and never propagate to realtime.
"""
from __future__ import annotations

import logging
import threading
import time

import config
from intelligence import memory_extractor
from storage import messages as msg_store

logger = logging.getLogger("yap.workers.memory_worker")


class MemoryWorker:
    """
    Daemon thread that periodically extracts memories from unprocessed messages.

    Args:
        poll_seconds: How often to check for unprocessed messages (default 30s).
        batch_size: Max messages per extraction batch (default 15).
    """

    def __init__(
        self,
        poll_seconds: int = None,
        batch_size: int = None,
    ):
        self._poll_seconds = poll_seconds or config.MEMORY_WORKER_POLL_SECONDS
        self._batch_size = batch_size or config.MEMORY_BATCH_SIZE

    def start(self) -> threading.Thread:
        """Start the background thread. Returns the thread."""
        t = threading.Thread(target=self._run, daemon=True, name="memory-worker")
        t.start()
        logger.info("[MEMORY] worker started poll_seconds=%d batch_size=%d",
                    self._poll_seconds, self._batch_size)
        return t

    def _run(self) -> None:
        while True:
            try:
                self._tick()
            except Exception as e:
                logger.error("[MEMORY] worker tick error: %s", e)
            time.sleep(self._poll_seconds)

    def _tick(self) -> None:
        """Single extraction cycle. Two independent, ownership-distinct passes:
        user facts (storage/memories.py) and Eve's own life-continuity
        (storage/eve_state.py) — see PART 9 of the V5 spec."""
        self._tick_user_memories()
        self._tick_eve_self_state()

    def _tick_user_memories(self) -> None:
        messages = msg_store.get_unprocessed_for_memory(limit=self._batch_size)
        if not messages:
            return

        ids = [m["message_id"] for m in messages]
        # Phase 1: atomically claim the batch before starting Gemini work
        msg_store.mark_memory_in_progress(ids)

        try:
            stored = memory_extractor.extract_batch(messages)
            # Phase 2 (success): mark permanently processed
            msg_store.mark_memory_processed(ids)
            if stored > 0:
                logger.info("[MEMORY] extraction complete stored=%d batch=%d", stored, len(messages))
        except Exception as e:
            logger.error("[MEMORY] extraction failed for batch, will retry: %s", e)
            # Phase 2 (failure): roll back claim so worker retries next tick
            msg_store.mark_memory_failed(ids)

    def _tick_eve_self_state(self) -> None:
        messages = msg_store.get_unprocessed_eve_messages(limit=self._batch_size)
        if not messages:
            return

        ids = [m["message_id"] for m in messages]
        # Phase 1: atomically claim the batch
        msg_store.mark_memory_in_progress(ids)

        try:
            stored = memory_extractor.extract_eve_self_state(messages)
            # Phase 2 (success): mark permanently processed
            msg_store.mark_memory_processed(ids)
            if stored > 0:
                logger.info("[EVE_STATE] extraction complete stored=%d batch=%d", stored, len(messages))
        except Exception as e:
            logger.error("[EVE_STATE] extraction failed for batch, will retry: %s", e)
            # Phase 2 (failure): roll back claim so worker retries next tick
            msg_store.mark_memory_failed(ids)
