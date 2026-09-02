"""Backup model: OpenAI ChatGPT. Same prompt contract as Grok."""

from flipper.config import settings
from flipper.llm.base import ChatResult
from flipper.llm.grok import EXTRACT_SYSTEM


def complete(prompt: str, system: str | None = None) -> ChatResult:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is missing")
    from openai import OpenAI

    instructions = system if system is not None else EXTRACT_SYSTEM
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.responses.create(
        model=settings.backup_model,
        instructions=instructions,
        input=prompt,
    )
    return ChatResult(
        text=response.output_text or "",
        model=settings.backup_model,
        provider="openai",
    )
