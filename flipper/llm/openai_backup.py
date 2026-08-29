"""Backup model: OpenAI ChatGPT. Same prompt and JSON contract as Grok."""

from flipper.config import settings
from flipper.llm.base import ChatResult
from flipper.llm.grok import SYSTEM


def complete(prompt: str) -> ChatResult:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is missing")
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.responses.create(
        model=settings.backup_model,
        instructions=SYSTEM,
        input=prompt,
    )
    return ChatResult(
        text=response.output_text or "",
        model=settings.backup_model,
        provider="openai",
    )
