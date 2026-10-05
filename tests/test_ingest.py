from langchain_core.embeddings import DeterministicFakeEmbedding

from app.ingest import CHUNK_SIZE, ingest, load_documents, split_documents
from app.vector_store import get_vector_store

POLICY_FILES = {"account.md", "payments.md", "returns.md", "shipping.md", "warranty.md"}


def write_policy(directory, name, text):
    (directory / name).write_text(text, encoding="utf-8")


def fake_store(path):
    # Fake embeddings: the same text always gives the same vector, and no Ollama is needed.
    return get_vector_store(DeterministicFakeEmbedding(size=32), persist_directory=path)


def test_load_documents_tags_each_file_with_its_source():
    documents = load_documents()

    assert {document.metadata["source"] for document in documents} == POLICY_FILES


def test_each_section_becomes_one_chunk_with_its_heading(tmp_path):
    write_policy(tmp_path, "test.md", "# Test Policy\n\n## Alpha\n\nAlpha rule.\n\n## Beta\n\nBeta rule.\n")

    chunks = split_documents(load_documents(tmp_path))

    assert [chunk.page_content for chunk in chunks] == ["Test Policy > Alpha\n\nAlpha rule.", "Test Policy > Beta\n\nBeta rule."]
    assert [chunk.metadata for chunk in chunks] == [
        {"source": "test.md", "section": "Alpha"},
        {"source": "test.md", "section": "Beta"},
    ]


def test_long_section_is_split_and_every_piece_keeps_its_heading(tmp_path):
    long_text = " ".join(f"Sentence number {i} of a very long rule." for i in range(40))
    write_policy(tmp_path, "test.md", f"# Test Policy\n\n## Long\n\n{long_text}\n")

    chunks = split_documents(load_documents(tmp_path))

    assert len(chunks) > 1
    assert all(chunk.page_content.startswith("Test Policy > Long\n\n") for chunk in chunks)
    assert all(chunk.metadata["section"] == "Long" for chunk in chunks)


def test_real_policy_chunks_have_metadata_and_fit_the_chunk_size():
    chunks = split_documents(load_documents())

    assert {chunk.metadata["source"] for chunk in chunks} == POLICY_FILES
    assert all(chunk.metadata["section"] for chunk in chunks)
    # The "<title> > <section>" prefix is added on top of the CHUNK_SIZE body.
    assert all(len(chunk.page_content) <= CHUNK_SIZE + 100 for chunk in chunks)


def test_reingesting_does_not_create_duplicates(tmp_path):
    docs, store_path = tmp_path / "docs", tmp_path / "store"
    docs.mkdir()
    write_policy(docs, "a.md", "# A\n\n## One\n\nFirst rule.\n\n## Two\n\nSecond rule.\n")
    store = fake_store(store_path)

    first = ingest(store, docs)
    second = ingest(store, docs)

    assert first == second == 2
    assert len(store.get()["ids"]) == 2


def test_reingesting_removes_chunks_of_deleted_documents(tmp_path):
    docs, store_path = tmp_path / "docs", tmp_path / "store"
    docs.mkdir()
    write_policy(docs, "keep.md", "# Keep\n\n## Rule\n\nKept.\n")
    write_policy(docs, "old.md", "# Old\n\n## Rule\n\nRemoved later.\n")
    store = fake_store(store_path)
    ingest(store, docs)

    (docs / "old.md").unlink()
    ingest(store, docs)

    assert {metadata["source"] for metadata in store.get()["metadatas"]} == {"keep.md"}
