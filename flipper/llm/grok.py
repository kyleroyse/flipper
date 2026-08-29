"""Primary model: xAI Grok 4.6."""

from flipper.config import settings
from flipper.llm.base import ChatResult

SYSTEM = """You extract structured scientific observations from lab notes.
Return JSON only: {"rows": [{"wav_id": str, "measurement": str, "value": number | null, "unit": str | null, "notes": str}]}.
Never invent a number that is not in the notes. Use null if missing.
"""


def complete(prompt: str) -> ChatResult:
    if not settings.xai_api_key:
        raise RuntimeError("XAI_API_KEY is missing")
    from xai_sdk import Client
    from xai_sdk.chat import system, user

    client = Client(api_key=settings.xai_api_key)
    chat = client.chat.create(model=settings.primary_model)
    chat.append(system(SYSTEM))
    chat.append(user(prompt))
    text = chat.sample().content or ""
    return ChatResult(text=text, model=settings.primary_model, provider="xai")
