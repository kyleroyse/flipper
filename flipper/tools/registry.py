"""Name to tool lookup."""

from __future__ import annotations

from flipper.tools.base import BaseTool


class ToolRegistry:
    """Register and resolve tools by name."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> ToolRegistry:
        """Add a tool. Later registers replace the same name."""
        if not getattr(tool, "name", None):
            raise ValueError("Tool must have a non-empty name")
        self._tools[tool.name] = tool
        return self

    def get(self, name: str) -> BaseTool:
        """Return a tool or raise KeyError."""
        if name not in self._tools:
            raise KeyError(name)
        return self._tools[name]

    def all_tools(self) -> list[BaseTool]:
        """Return tools in registration order."""
        return list(self._tools.values())

    def names(self) -> list[str]:
        """Return registered tool names."""
        return list(self._tools.keys())
