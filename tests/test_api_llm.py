"""The API with the real graph and models. Slow and needs Ollama: `pytest -m llm`."""

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.graph import build_graph
from app.memory import make_checkpointer
from app.retriever import retrieve

pytestmark = pytest.mark.llm


def test_policy_question_through_the_api(policy_store, order_db, tmp_path):
    tickets_db = tmp_path / "tickets.db"
    app = create_app(
        graph_factory=lambda: build_graph(
            retriever=lambda question: retrieve(question, vector_store=policy_store),
            db_path=order_db, tickets_db_path=tickets_db, checkpointer=make_checkpointer(),
        ),
        tickets_db_path=tickets_db,
    )

    with TestClient(app) as client:
        body = client.post("/chat", json={"message": "How much is express shipping?"}).json()

    assert body["intent"] == "policy_question"
    assert "14.99" in body["reply"]
    assert body["sources"] == ["shipping.md"]
