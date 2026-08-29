"""Tests for Grok-primary / OpenAI-backup failover."""

import unittest
from unittest.mock import patch

from flipper.llm.base import ChatResult
from flipper.llm.router import _is_retryable, complete


class RetryableTests(unittest.TestCase):
    def test_timeout_is_retryable(self):
        self.assertTrue(_is_retryable(TimeoutError("timed out")))

    def test_429_is_retryable(self):
        self.assertTrue(_is_retryable(RuntimeError("HTTP 429 rate limit")))

    def test_bad_json_is_not_retryable(self):
        self.assertFalse(_is_retryable(ValueError("invalid json")))


class RouterTests(unittest.TestCase):
    def test_primary_success(self):
        primary = ChatResult(text="{}", model="grok-4.6", provider="xai")
        with patch("flipper.llm.grok.complete", return_value=primary) as grok:
            with patch("flipper.llm.openai_backup.complete") as backup:
                result = complete("notes")
        self.assertEqual(result.provider, "xai")
        grok.assert_called_once()
        backup.assert_not_called()

    def test_fallback_on_timeout(self):
        backup = ChatResult(text="{}", model="gpt-5.5", provider="openai")
        with patch("flipper.llm.grok.complete", side_effect=TimeoutError("xai down")):
            with patch("flipper.llm.openai_backup.complete", return_value=backup) as openai:
                result = complete("notes")
        self.assertEqual(result.provider, "openai")
        openai.assert_called_once()

    def test_non_retryable_does_not_fallback(self):
        with patch("flipper.llm.grok.complete", side_effect=RuntimeError("XAI_API_KEY is missing")):
            with patch("flipper.llm.openai_backup.complete") as backup:
                with self.assertRaises(RuntimeError):
                    complete("notes")
        backup.assert_not_called()


if __name__ == "__main__":
    unittest.main()
