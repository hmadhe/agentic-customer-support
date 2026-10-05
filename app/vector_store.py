from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from app.config import PROJECT_ROOT, Settings, get_settings

CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "voltcart_policies"


class NomicEmbeddings(OllamaEmbeddings):
    """nomic-embed-text was trained with task prefixes and retrieves better when they are used.

    Without them, the right chunk for "My laptop stopped working after 6 months" (Manufacturer warranty)
    was not in the top 4 results; with them it ranks 2nd. See the Milestone 2 development log.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents([f"search_document: {text}" for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(f"search_query: {text}")


def get_embeddings(settings: Settings | None = None) -> NomicEmbeddings:
    settings = settings or get_settings()
    return NomicEmbeddings(model=settings.ollama_embedding_model, base_url=settings.ollama_base_url)


def get_vector_store(embeddings: Embeddings | None = None, persist_directory: Path = CHROMA_DIR) -> Chroma:
    """The Chroma collection shared by ingestion and retrieval, saved to disk in chroma_db/."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings or get_embeddings(),
        persist_directory=str(persist_directory),
        # Cosine similarity gives relevance scores between 0 and 1, which are easier to inspect than L2 distance.
        collection_metadata={"hnsw:space": "cosine"},
    )
