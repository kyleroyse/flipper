"""Tests for the agent loop."""

import unittest

from flipper.agents.agent import Agent
from flipper.agents.types import ToolCall
from flipper.memory.store import MemoryStore
from flipper.tools.audio import ListAudioFormatsTool, ProcessAudioTool
from flipper.tools.base import BaseTool
from flipper.tools.registry import ToolRegistry


class StubTool(BaseTool):
    name = "stub"
    description = "Returns a fixed string."

    def run(self, **kwargs):
        return "stub-ok"


def build_agent(planner=None) -> Agent:
    registry = ToolRegistry()
    registry.register(ListAudioFormatsTool())
    registry.register(ProcessAudioTool())
    registry.register(StubTool())
    return Agent(registry=registry, memory=MemoryStore(), system_prompt="test", planner=planner)


class AgentTests(unittest.TestCase):
    def test_explicit_tool_name_in_task(self):
        agent = build_agent()
        result = agent.run("please run stub")
        self.assertIn("stub-ok", result)
        roles = [m.role for m in agent.memory.history()]
        self.assertIn("tool", roles)
        self.assertEqual(roles[-1], "assistant")

    def test_format_keyword_selects_list_formats(self):
        agent = build_agent()
        result = agent.run("what formats do you support?")
        self.assertIn("wav", result)

    def test_custom_planner_hook(self):
        def planner(task, tools):
            return [ToolCall(name="stub", arguments={})]

        agent = build_agent(planner=planner)
        result = agent.run("ignore matching")
        self.assertIn("stub-ok", result)

    def test_unknown_tool_from_planner(self):
        def planner(task, tools):
            return [ToolCall(name="missing", arguments={})]

        agent = build_agent(planner=planner)
        result = agent.run("go")
        self.assertIn("Unknown tool", result)


if __name__ == "__main__":
    unittest.main()
