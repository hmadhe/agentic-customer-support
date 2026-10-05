from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app.config import PROJECT_ROOT
from app.vector_store import get_vector_store

POLICIES_DIR = PROJECT_ROOT / "data" / "policies"
CHUNK_SIZE = 500  # characters
CHUNK_OVERLAP = 50


def load_documents(directory: Path = POLICIES_DIR) -> list[Document]:
    """Read every Markdown file, tagging each with its file name as the source."""
    return [
        Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": path.name})
        for path in sorted(directory.glob("*.md"))
    ]


def split_documents(documents: list[Document]) -> list[Document]:
    """Split each document into one chunk per "##" section.

    Each chunk starts with "<document title> > <section>" so it makes sense on its own.
    Sections longer than CHUNK_SIZE are split further, with overlap.
    """
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "title"), ("##", "section")])
    size_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    chunks = []
    for document in documents:
        for section in header_splitter.split_text(document.page_content):
            title = section.metadata.get("title", "")
            heading = section.metadata.get("section", "")
            for piece in size_splitter.split_text(section.page_content):
                chunks.append(
                    Document(
                        page_content=f"{title} > {heading}\n\n{piece}",
                        metadata={"source": document.metadata["source"], "section": heading},
                    )
                )
    return chunks


def ingest(vector_store: Chroma | None = None, directory: Path = POLICIES_DIR) -> int:
    """Load, split, embed and store the policy documents. Returns the number of chunks stored.

    The collection is emptied first, so re-running ingestion never creates duplicates
    and chunks from edited or deleted documents don't linger.
    """
    chunks = split_documents(load_documents(directory))
    vector_store = vector_store or get_vector_store()
    vector_store.reset_collection()
    vector_store.add_documents(chunks)
    return len(chunks)
