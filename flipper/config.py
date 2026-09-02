"""Keys, model names, and paths. Pin IDs in .env, not in the graph."""

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    xai_api_key: str = ""
    openai_api_key: str = ""
    primary_model: str = "grok-4.6"
    backup_model: str = "gpt-5.5"
    checkpoint_path: Path = ROOT / "runs" / "sessions.sqlite"
    processed_dir: Path = ROOT / "data" / "processed"
    dolphin_xlsx: Path | None = None

    @field_validator("dolphin_xlsx", mode="before")
    @classmethod
    def _empty_path(cls, value: object) -> object:
        if value is None or value == "":
            return None
        return value


settings = Settings()
settings.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
settings.processed_dir.mkdir(parents=True, exist_ok=True)
