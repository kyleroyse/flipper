from __future__ import annotations

"""Minimal agent loop: plan, call tools, observe, stop."""

from collections.abc import Callable, Sequence
from pathlib import Path

from flipper.agents.types import Message, ToolCall, ToolResult
from flipper.memory.store import MemoryStore
from flipper.tools.base import BaseTool
from flipper.tools.registry import ToolRegistry

Planner = Callable[[str, Sequence[BaseTool]], list[ToolCall]]


def load_system_prompt() -> str:
    """Load the default system prompt from the package."""
    path = Path(__file__).resolve().parent.parent / "prompts" / "system.md"
    return path.read_text(encoding="utf-8")


class Agent:
    """Run one user task against a tool registry.

    Planning is keyword matching by default. Pass ``planner`` to swap in a
    model later without changing the loop.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        memory: MemoryStore | None = None,
        system_prompt: str | None = None,
        planner: Planner | None = None,
    ) -> None:
        self.registry = registry
        self.memory = memory or MemoryStore()
        self.system_prompt = system_prompt if system_prompt is not None else load_system_prompt()
        self.planner = planner

    def run(self, task: str) -> str:
        """Execute one task and return the final assistant message."""
        self.memory.add(Message(role="system", content=self.system_prompt))
        self.memory.add(Message(role="user", content=task))

        calls = self._plan(task)
        results: list[ToolResult] = []

        if not calls:
            available = ", ".join(t.name for t in self.registry.all_tools()) or "(none)"
            final = f"No tool matched that task. Available tools: {available}"
            self.memory.add(Message(role="assistant", content=final))
            return final

        for call in calls:
            result = self._invoke(call)
            results.append(result)
            self.memory.add(Message(role="tool", content=result.output))

        final = self._summarize(task, results)
        self.memory.add(Message(role="assistant", content=final))
        return final

    def _plan(self, task: str) -> list[ToolCall]:
        tools = self.registry.all_tools()
        if self.planner is not None:
            return self.planner(task, tools)
        return self._match_tools(task, tools)

    def _match_tools(self, task: str, tools: Sequence[BaseTool]) -> list[ToolCall]:
        lowered = task.lower()
        by_name = {tool.name: tool for tool in tools}
        calls: list[ToolCall] = []

        for name in by_name:
            needle = name.replace("_", " ")
            if name in lowered or needle in lowered:
                calls.append(ToolCall(name=name, arguments={}))

        if calls:
            return calls

        if "format" in lowered and "list_audio_formats" in by_name:
            return [ToolCall(name="list_audio_formats", arguments={})]

        if any(word in lowered for word in ("process", "audio", "spectrogram", "clip")):
            if "process_audio" in by_name:
                return [ToolCall(name="process_audio", arguments={})]

        if "list_audio_formats" in by_name:
            return [ToolCall(name="list_audio_formats", arguments={})]

        return []

    def _invoke(self, call: ToolCall) -> ToolResult:
        try:
            tool = self.registry.get(call.name)
        except KeyError:
            return ToolResult(
                name=call.name,
                ok=False,
                output=f"Unknown tool: {call.name}",
            )

        try:
            output = tool.run(**call.arguments)
            return ToolResult(name=call.name, ok=True, output=str(output))
        except Exception as exc:
            return ToolResult(
                name=call.name,
                ok=False,
                output=f"{type(exc).__name__}: {exc}",
            )

    def _summarize(self, task: str, results: list[ToolResult]) -> str:
        lines = [f"Task: {task}", ""]
        for result in results:
            status = "ok" if result.ok else "error"
            lines.append(f"[{status}] {result.name}: {result.output}")
        return "\n".join(lines)
