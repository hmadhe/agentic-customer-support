from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env from the project root so it loads no matter which directory we run from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """App settings. Values come from environment variables first, then .env, then these defaults."""

    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    llm_temperature: float = 0.0
    # Hard limit on generated tokens. Without it, one runaway generation ran for
    # 8 minutes (see the Milestone 2 development log).
    max_output_tokens: int = 512
    ollama_embedding_model: str = "nomic-embed-text"
    retrieval_k: int = 4


def get_settings() -> Settings:
    return Settings()
