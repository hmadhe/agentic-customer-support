"""Build the policy vector store: python -m scripts.ingest"""

from app.ingest import ingest
from app.vector_store import get_vector_store


def main() -> None:
    vector_store = get_vector_store()
    stored = ingest(vector_store)
    print(f"Stored {stored} chunks. Collection now holds {len(vector_store.get()['ids'])} chunks.")


if __name__ == "__main__":
    main()
