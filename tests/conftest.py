import pytest


@pytest.fixture(autouse=True)
def no_real_models_in_unit_tests(request, monkeypatch):
    """Unit tests must use fakes. Point Ollama at a closed port so an accidental real call fails fast.

    Added after a "fast" graph test silently called the real embedding model and LLM and took 58 seconds.
    """
    if request.node.get_closest_marker("llm") is None:
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9")


@pytest.fixture(scope="session")
def order_db(tmp_path_factory):
    """A seeded order database in a temporary folder, dated relative to today like the real one."""
    from app.orders import seed_db

    db_path = tmp_path_factory.mktemp("orders") / "voltcart.db"
    seed_db(db_path)
    return db_path


@pytest.fixture(scope="session")
def policy_store(tmp_path_factory):
    """For real-model tests: a vector store built once from the real policy documents.

    It lives in a temporary folder, so tests never depend on (or change) the local chroma_db/.
    """
    from app.ingest import ingest
    from app.vector_store import get_vector_store

    vector_store = get_vector_store(persist_directory=tmp_path_factory.mktemp("chroma"))
    ingest(vector_store)
    return vector_store
