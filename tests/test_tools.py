"""Tests for tool registry and audio wrappers."""

import unittest

from flipper.tools.audio import ListAudioFormatsTool, ProcessAudioTool
from flipper.tools.registry import ToolRegistry


class RegistryTests(unittest.TestCase):
    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = ListAudioFormatsTool()
        registry.register(tool)
        self.assertIs(registry.get("list_audio_formats"), tool)
        self.assertEqual(registry.names(), ["list_audio_formats"])

    def test_missing_tool_raises(self):
        registry = ToolRegistry()
        with self.assertRaises(KeyError):
            registry.get("nope")


class AudioToolTests(unittest.TestCase):
    def test_list_formats(self):
        output = ListAudioFormatsTool().run()
        self.assertIn("wav", output)
        self.assertIn("mp3", output)

    def test_process_audio_identity(self):
        output = ProcessAudioTool().run(duration_seconds=0.05, sample_rate=8000)
        self.assertIn("8000 Hz", output)
        self.assertIn("identity processor", output)

    def test_process_audio_rejects_bad_duration(self):
        with self.assertRaises(ValueError):
            ProcessAudioTool().run(duration_seconds=0)


if __name__ == "__main__":
    unittest.main()
