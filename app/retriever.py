from langchain_chroma import Chroma

from app.config import get_settings
from app.schemas import RetrievedChunk
from app.vector_store import get_vector_store


def retrieve(question: str, vector_store: Chroma | None = None, k: int | None = None) -> list[RetrievedChunk]:
    """Return the k policy chunks most similar to the question, most relevant first."""
    vector_store = vector_store or get_vector_store()
    k = k or get_settings().retrieval_k
    results = vector_store.similarity_search_with_relevance_scores(question, k=k)
    return [
        RetrievedChunk(
            text=doc.page_content,
            source=doc.metadata["source"],
            section=doc.metadata["section"],
            score=round(score, 3),
        )
        for doc, score in results
    ]
