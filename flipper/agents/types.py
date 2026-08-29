"""Shared types for the agent loop."""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Message:
    """A single turn in the agent conversation."""

    role: str
    content: str


@dataclass
class ToolCall:
    """A request to run one registered tool."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Outcome of one tool invocation."""

    name: str
    ok: bool
    output: str
    data: Optional[Any] = None
