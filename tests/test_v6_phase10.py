"""
EVE V6 Phase 10 Instagram Session Lifecycle Tests.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Adjust path to import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from instagram.session_manager import InstagramSessionManager


class TestPhase10SessionManager(unittest.TestCase):

    def setUp(self):
        self.temp_session_path = Path("temp_session_settings.json")
        if self.temp_session_path.exists():
            self.temp_session_path.unlink()
        self.manager = InstagramSessionManager(self.temp_session_path)

    def tearDown(self):
        if self.temp_session_path.exists():
            self.temp_session_path.unlink()

    @patch("instagram.session_manager.Client")
    def test_01_credential_login(self, mock_client_cls):
        """Manager authenticates using credentials if no session settings file exists."""
        mock_cl = MagicMock()
        mock_cl.user_id = None
        mock_client_cls.return_value = mock_cl

        config.IG_USERNAME = "test_user"
        config.IG_PASSWORD = "test_password"
        config.IG_SESSIONID = None

        cl = self.manager.get_client()
        self.assertIsNotNone(cl)
        mock_cl.login.assert_called_once_with("test_user", "test_password")
        mock_cl.dump_settings.assert_called_once_with(self.temp_session_path)

    @patch("instagram.session_manager.Client")
    def test_02_session_resume(self, mock_client_cls):
        """Manager resumes existing session if client has user_id and token is valid."""
        mock_cl = MagicMock()
        mock_cl.user_id = "12345"
        mock_client_cls.return_value = mock_cl

        # Simulate settings file existence
        self.temp_session_path.write_text("{}")

        cl = self.manager.get_client()
        self.assertIsNotNone(cl)
        mock_cl.load_settings.assert_called_once_with(self.temp_session_path)
        mock_cl.get_timeline_feed.assert_called_once()
        mock_cl.login.assert_not_called()

    @patch("instagram.session_manager.Client")
    def test_03_resume_failure_triggers_relogin(self, mock_client_cls):
        """If session verification fails, manager logs in again using credentials."""
        mock_cl = MagicMock()
        mock_cl.user_id = "12345"
        mock_cl.get_timeline_feed.side_effect = Exception("Expired token")
        mock_client_cls.return_value = mock_cl

        self.temp_session_path.write_text("{}")

        config.IG_USERNAME = "test_user"
        config.IG_PASSWORD = "test_password"
        config.IG_SESSIONID = None

        cl = self.manager.get_client()
        self.assertIsNotNone(cl)
        mock_cl.login.assert_called_once_with("test_user", "test_password")

    @patch("instagram.session_manager.Client")
    def test_04_invalidate_and_relogin(self, mock_client_cls):
        """invalidate_and_relogin discards settings file and forces a fresh client login."""
        mock_cl = MagicMock()
        mock_cl.user_id = None
        mock_client_cls.return_value = mock_cl

        self.temp_session_path.write_text("{}")

        config.IG_USERNAME = "test_user"
        config.IG_PASSWORD = "test_password"
        config.IG_SESSIONID = None

        cl = self.manager.invalidate_and_relogin()
        self.assertFalse(self.temp_session_path.exists())
        mock_cl.login.assert_called_once_with("test_user", "test_password")


if __name__ == "__main__":
    unittest.main()
