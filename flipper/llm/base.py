"""Shared chat result type."""

from dataclasses import dataclass


@dataclass
class ChatResult:
    text: str
    model: str
    provider: str
