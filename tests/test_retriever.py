from langchain_core.embeddings import DeterministicFakeEmbedding

from app.ingest import ingest
from app.retriever import retrieve
from app.vector_store import get_vector_store


def test_retrieve_returns_chunks_with_source_and_section(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "returns.md").write_text("# Returns\n\n## Window\n\nReturn within 30 days.\n", encoding="utf-8")
    (docs / "shipping.md").write_text("# Shipping\n\n## Costs\n\nExpress costs $14.99.\n", encoding="utf-8")
    store = get_vector_store(DeterministicFakeEmbedding(size=32), persist_directory=tmp_path / "store")
    ingest(store, docs)

    # With fake embeddings only identical text matches, so query with a chunk's exact text.
    chunks = retrieve("Shipping > Costs\n\nExpress costs $14.99.", vector_store=store, k=2)

    assert len(chunks) == 2
    assert (chunks[0].source, chunks[0].section) == ("shipping.md", "Costs")
    assert chunks[0].score > chunks[1].score
    assert {chunk.source for chunk in chunks} == {"returns.md", "shipping.md"}
