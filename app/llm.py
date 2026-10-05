from langchain_ollama import ChatOllama

from app.config import Settings, get_settings


def get_llm(settings: Settings | None = None) -> ChatOllama:
    """Build the chat model. The rest of the app depends on this function, not on Ollama directly."""
    settings = settings or get_settings()
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.llm_temperature,
    )
