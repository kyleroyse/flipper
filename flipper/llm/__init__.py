"""Model clients. Graph nodes call router.complete only."""

from flipper.llm.base import ChatResult
from flipper.llm.router import complete

__all__ = ["ChatResult", "complete"]
