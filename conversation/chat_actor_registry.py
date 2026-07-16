"""
Chat Actor Registry — V6.
Thread-safe lazy manager for thread-scoped ChatActor instances.
Routes normalized messages to their correct actor mailbox.
"""
from __future__ import annotations

import logging
import threading
from typing import Dict

from conversation.chat_actor import ChatActor
from models.message import NormalizedMessage

logger = logging.getLogger("yap.conversation.chat_actor_registry")


class ChatActorRegistry:
    """Manages thread-scoped ChatActor instances and coordinates shutdown."""

    def __init__(self):
        self._actors: Dict[str, ChatActor] = {}
        self._lock = threading.Lock()

    def route_message(
        self,
        msg: NormalizedMessage,
        cl,
        bot_user_id: str,
        bot_username: str,
        lane_manager,
        fatigue_tracker,
        mode_selector,
        voice_health,
    ) -> ChatActor:
        """Retrieve or create the actor for this thread, and post the message to its queue."""
        thread_id_str = str(msg.thread_id)
        
        with self._lock:
            if thread_id_str not in self._actors:
                logger.info("[REGISTRY] lazily creating ChatActor for thread_id=%s", thread_id_str)
                actor = ChatActor(
                    thread_id=thread_id_str,
                    cl=cl,
                    bot_user_id=bot_user_id,
                    bot_username=bot_username,
                    lane_manager=lane_manager,
                    fatigue_tracker=fatigue_tracker,
                    mode_selector=mode_selector,
                    voice_health=voice_health,
                )
                actor.start()
                self._actors[thread_id_str] = actor
            
            actor = self._actors[thread_id_str]

        actor.post_message(msg)
        return actor

    def get_actor(self, thread_id: str) -> Optional[ChatActor]:
        """Fetch actor if it exists, without spawning a new one."""
        thread_id_str = str(thread_id)
        with self._lock:
            return self._actors.get(thread_id_str)

    def shutdown_all(self):
        """Cleanly stop all running actors."""
        with self._lock:
            for thread_id, actor in list(self._actors.items()):
                logger.info("[REGISTRY] stopping ChatActor for thread_id=%s", thread_id)
                actor.stop()
            self._actors.clear()
