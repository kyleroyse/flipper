"""Tool protocol used by the agent."""

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """A named, described action the agent may invoke."""

    name: str
    description: str
    parameters: dict[str, Any] = {}

    @abstractmethod
    def run(self, **kwargs: Any) -> str:
        """Execute the tool and return a human-readable result."""
        raise NotImplementedError
