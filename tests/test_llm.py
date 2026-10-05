from langchain_ollama import ChatOllama

from app.config import Settings
from app.llm import get_llm


def test_get_llm_uses_settings():
    # No network call: we only check the model is wired up from settings.
    settings = Settings(ollama_model="test-model", ollama_base_url="http://example:1234", llm_temperature=0.5)

    llm = get_llm(settings)

    assert isinstance(llm, ChatOllama)
    assert llm.model == "test-model"
    assert llm.base_url == "http://example:1234"
    assert llm.temperature == 0.5
