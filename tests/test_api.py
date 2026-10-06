"""API tests with FastAPI's TestClient and a graph made of fakes (no models, temporary databases)."""

import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from app.api import create_app
from app.graph import build_graph
from app.memory import make_checkpointer
from app.schemas import Intent, IntentClassification, PolicyAnswer, RetrievedChunk, Sentiment

CHUNK = RetrievedChunk(text="Shipping > Costs\n\nExpress costs $14.99.", source="shipping.md", section="Costs", score=0.7)
INTENTS = {
    "How much is express shipping?": Intent.POLICY_QUESTION,
    "I want a human": Intent.HUMAN_REQUEST,
    "Where is order 1042?": Intent.ORDER_ISSUE,
    "Hi": Intent.GREETING,
}


def classify(inputs):
    if inputs["message"] == "slow":
        time.sleep(0.5)
        return IntentClassification(intent=Intent.GREETING, sentiment=Sentiment.NEUTRAL)
    if inputs["message"] == "Ollama is down":
        raise httpx.ConnectError("[WinError 10061] No connection could be made")
    return IntentClassification(intent=INTENTS[inputs["message"]], sentiment=Sentiment.NEUTRAL, order_id="1042")


def order_agent(messages):
    # First call: look up the order. After the tool result: answer.
    if isinstance(messages[-1], HumanMessage):
        return AIMessage("", tool_calls=[{"name": "get_order_status", "args": {"order_id": "1042"}, "id": "c1"}])
    return AIMessage("Your order 1042 has shipped.")


@pytest.fixture
def tickets_db(tmp_path):
    return tmp_path / "tickets.db"


@pytest.fixture
def client(order_db, tickets_db):
    answered = PolicyAnswer(answered=True, answer="Express shipping costs $14.99.", sources=["shipping.md"])
    app = create_app(
        graph_factory=lambda: build_graph(
            RunnableLambda(classify), retriever=lambda _: [CHUNK], answerer=lambda *_: answered,
            agent=RunnableLambda(order_agent), db_path=order_db, tickets_db_path=tickets_db,
            checkpointer=make_checkpointer(),
        ),
        tickets_db_path=tickets_db,
    )
    with TestClient(app) as test_client:  # `with` runs the startup (lifespan) code
        yield test_client


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_policy_question_returns_answer_and_sources(client):
    body = client.post("/chat", json={"message": "How much is express shipping?"}).json()

    assert body["reply"] == "Express shipping costs $14.99."
    assert (body["intent"], body["sources"], body["tools_used"]) == ("policy_question", ["shipping.md"], [])
    assert (body["escalated"], body["escalation_reason"], body["ticket_id"]) == (False, None, None)
    assert len(body["thread_id"]) == 36  # a new conversation gets a UUID


def test_order_question_reports_the_tools_used(client):
    body = client.post("/chat", json={"message": "Where is order 1042?"}).json()

    assert body["reply"] == "Your order 1042 has shipped."
    assert body["tools_used"] == ["get_order_status"]


def test_same_thread_id_continues_the_conversation(client):
    first = client.post("/chat", json={"message": "How much is express shipping?"}).json()
    client.post("/chat", json={"message": "Hi", "thread_id": first["thread_id"]})

    saved = client.app.state.graph.get_state({"configurable": {"thread_id": first["thread_id"]}}).values["messages"]
    assert [m.content for m in saved if isinstance(m, HumanMessage)] == ["How much is express shipping?", "Hi"]


def test_escalation_creates_a_ticket_that_the_tickets_endpoints_return(client):
    body = client.post("/chat", json={"message": "I want a human"}).json()

    assert (body["escalated"], body["escalation_reason"], body["ticket_id"]) == (True, "customer_request", 1)
    assert [t["ticket_id"] for t in client.get("/tickets").json()] == [1]
    assert client.get("/tickets/1").json()["customer_message"] == "I want a human"


def test_unknown_ticket_is_404(client):
    assert client.get("/tickets/99").status_code == 404


@pytest.mark.parametrize("payload", [{"message": ""}, {}, {"message": "x" * 2001}])
def test_invalid_requests_are_rejected_with_422(client, payload):
    assert client.post("/chat", json=payload).status_code == 422


def test_unreachable_model_returns_503_not_500(client):
    response = client.post("/chat", json={"message": "Ollama is down"})

    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"]


def test_simultaneous_messages_on_one_conversation_are_both_saved(client):
    # Without the per-conversation lock, one of these two messages was silently lost.
    sends = [
        threading.Thread(target=lambda: client.post("/chat", json={"message": "slow", "thread_id": "same"}))
        for _ in range(2)
    ]
    for t in sends:
        t.start()
    for t in sends:
        t.join()

    saved = client.app.state.graph.get_state({"configurable": {"thread_id": "same"}}).values["messages"]
    assert sum(isinstance(m, HumanMessage) for m in saved) == 2


def test_server_refuses_to_start_when_setup_is_missing(tickets_db):
    def missing_database():
        raise FileNotFoundError("Order database not found. Create it with: python -m scripts.seed_orders")

    app = create_app(graph_factory=missing_database, tickets_db_path=tickets_db)

    with pytest.raises(FileNotFoundError, match="seed_orders"):
        with TestClient(app):
            pass
