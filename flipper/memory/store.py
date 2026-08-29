"""In-memory conversation store."""

from flipper.agents.types import Message


class MemoryStore:
    """Append-only message history for a single run."""

    def __init__(self) -> None:
        self._messages: list[Message] = []

    def add(self, message: Message) -> None:
        """Append a message."""
        self._messages.append(message)

    def history(self) -> list[Message]:
        """Return a copy of the conversation."""
        return list(self._messages)

    def clear(self) -> None:
        """Drop all messages."""
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)
