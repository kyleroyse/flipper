"""Primary model: xAI Grok 4.6."""

from flipper.config import settings
from flipper.llm.base import ChatResult

EXTRACT_SYSTEM = """You extract structured scientific observations from lab notes.
Return JSON only: {"rows": [{"wav_id": str, "measurement": str, "value": number | null, "unit": str | null, "notes": str}]}.
Never invent a number that is not in the notes. Use null if missing.
"""

SUMMARY_SYSTEM = """You summarize dolphin research tables copied from Excel.
Write a short plain-language summary of what the table contains.
Do not invent numbers. Do not turn the table into measurement JSON rows.
"""

# Back-compat alias used by the extract graph.
SYSTEM = EXTRACT_SYSTEM


def complete(prompt: str, system: str | None = None) -> ChatResult:
    if not settings.xai_api_key:
        raise RuntimeError("XAI_API_KEY is missing")
    from xai_sdk import Client
    from xai_sdk.chat import system as sys_msg, user

    instructions = system if system is not None else EXTRACT_SYSTEM
    client = Client(api_key=settings.xai_api_key)
    chat = client.chat.create(model=settings.primary_model)
    chat.append(sys_msg(instructions))
    chat.append(user(prompt))
    text = chat.sample().content or ""
    return ChatResult(text=text, model=settings.primary_model, provider="xai")
