"""
Instagram Session Lifecycle Manager — V6.
Dedicated owner of client login, session restoration, challenge/checkpoint handling,
and device fingerprint preservation.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional
from instagrapi import Client
import config

logger = logging.getLogger("yap.instagram.session_manager")


class InstagramSessionManager:
    """
    Manages the lifetime and authentication state of the instagrapi Client.
    Safely handles checkpoints, relogins, and device fingerprint persistence.
    """

    def __init__(self, session_path: Path):
        self.session_path = session_path
        self._lock = threading.RLock()
        self.cl: Optional[Client] = None

    def get_client(self) -> Client:
        """Returns the current Client, creating and authenticating it if necessary."""
        with self._lock:
            if self.cl is not None:
                return self.cl
            self.cl = self.initialize_client()
            return self.cl

    def initialize_client(self) -> Client:
        """Creates a fresh instagrapi Client and authenticates it."""
        cl = Client()
        cl.delay_range = [1, 3]

        # Load device settings if they exist to keep device fingerprint stable
        if self.session_path.exists():
            try:
                session = cl.load_settings(self.session_path)
                if session:
                    cl.set_settings(session)
                    logger.info("[SESSION-MANAGER] Persistent device settings loaded.")
            except Exception as e:
                logger.error("[SESSION-MANAGER] Failed to load settings: %s", e)

        # Attempt to reuse existing session
        if cl.user_id:
            try:
                # Lightweight check: get timeline feed to verify token
                cl.get_timeline_feed()
                if not cl.username and config.IG_USERNAME:
                    cl.username = config.IG_USERNAME
                logger.info("[SESSION-MANAGER] Successfully verified and resumed existing session for %s", cl.username)
                return cl
            except Exception as e:
                logger.info("[SESSION-MANAGER] Saved session is invalid (%s). Initiating relogin.", e)

        # Perform login
        try:
            if config.IG_SESSIONID:
                cl.login_by_sessionid(config.IG_SESSIONID)
                logger.info("[SESSION-MANAGER] Logged in via session ID as %s", cl.username)
            else:
                cl.login(config.IG_USERNAME, config.IG_PASSWORD)
                logger.info("[SESSION-MANAGER] Logged in via credentials as %s", cl.username)

            # Dump new settings/session to persist device settings
            cl.dump_settings(self.session_path)
        except Exception as e:
            logger.error("[SESSION-MANAGER] Authentication failed: %s", e)
            raise e

        return cl

    def handle_checkpoint(self, checkpoint_url: str):
        """Placeholder for challenge checkpoint resolution."""
        logger.warning("[SESSION-MANAGER] Checkpoint challenge required: %s", checkpoint_url)

    def invalidate_and_relogin(self) -> Client:
        """Force invalidates the active client and attempts a fresh credentials login."""
        with self._lock:
            logger.info("[SESSION-MANAGER] Force invalidating and logging in again...")
            if self.session_path.exists():
                try:
                    self.session_path.unlink()
                except Exception as e:
                    logger.warning("[SESSION-MANAGER] Failed to delete session file: %s", e)
            self.cl = None
            return self.get_client()
