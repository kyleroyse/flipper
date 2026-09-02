"""Primary Grok, then OpenAI on timeout / 429 / 5xx. Nodes call complete()."""

import logging

from flipper.llm import grok, openai_backup
from flipper.llm.base import ChatResult

log = logging.getLogger(__name__)

_RETRYABLE = (
    TimeoutError,
    ConnectionError,
    OSError,
)


def complete(prompt: str, system: str | None = None) -> ChatResult:
    try:
        result = grok.complete(prompt, system=system)
        log.info("primary ok: %s", result.model)
        return result
    except Exception as exc:
        if not _is_retryable(exc):
            raise
        log.warning("Grok failed (%s); falling back to OpenAI", exc)
        result = openai_backup.complete(prompt, system=system)
        log.info("backup ok: %s", result.model)
        return result


def _is_retryable(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if isinstance(exc, _RETRYABLE):
        return True
    tokens = ("timeout", "rate limit", "429", "500", "502", "503", "unavailable")
    return any(t in name or t in text for t in tokens)
