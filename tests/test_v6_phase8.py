"""
EVE V6 Phase 8 Post-Validator Tests.
"""
from __future__ import annotations

import unittest
from conversation.chat_actor import GenerationLease
from intelligence.post_validator import PostValidator, PostValidationResult


class TestPhase8PostValidator(unittest.TestCase):

    def setUp(self):
        self.validator = PostValidator("bot123")
        self.lease = GenerationLease(
            lease_id="l1",
            start_version=1,
            session_id="s1",
            target_user_id="userA",
            anchor_message_id="m1",
            status="ACTIVE"
        )

    def test_01_valid_proposal(self):
        res = self.validator.validate("Hello world!", self.lease, [])
        self.assertTrue(res.is_valid)
        self.assertIsNone(res.reason)

    def test_02_empty_proposal(self):
        res = self.validator.validate("  ", self.lease, [])
        self.assertFalse(res.is_valid)
        self.assertEqual(res.reason, "Empty reply text")

    def test_03_length_limit(self):
        long_text = "a" * 1001
        res = self.validator.validate(long_text, self.lease, [])
        self.assertFalse(res.is_valid)
        self.assertTrue("exceeds maximum length" in res.reason)

    def test_04_duplicate_repetition(self):
        res = self.validator.validate("hello", self.lease, ["hello", "how are you"])
        self.assertFalse(res.is_valid)
        self.assertEqual(res.reason, "Duplicate message detected within recent turns")

    def test_05_inactive_lease(self):
        inactive_lease = GenerationLease(
            lease_id="l1",
            start_version=1,
            session_id="s1",
            target_user_id="userA",
            anchor_message_id="m1",
            status="CANCELLED"
        )
        res = self.validator.validate("Hello", inactive_lease, [])
        self.assertFalse(res.is_valid)
        self.assertTrue("Lease is not ACTIVE" in res.reason)


if __name__ == "__main__":
    unittest.main()
