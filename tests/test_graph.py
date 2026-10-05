import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from app.agent import ASK_FOR_ORDER_ID, MAX_TOOL_ROUNDS
from app.answer import INSUFFICIENT_ANSWER, AnswerGenerationError
from app.graph import RESPONSES, build_graph
from app.schemas import Intent, IntentClassification, PolicyAnswer, RetrievedChunk, Sentiment
from app.tickets import TICKET_FAILED_REPLY, EscalationReason, list_tickets

CHUNK = RetrievedChunk(text="Shipping > Costs\n\nExpress costs $14.99.", source="shipping.md", section="Costs", score=0.7)


def fake_classifier(intent: Intent, sentiment=Sentiment.NEUTRAL, order_id=None) -> RunnableLambda:
    """Stands in for the LLM classifier and always returns the given classification."""
    result = IntentClassification(intent=intent, sentiment=sentiment, order_id=order_id)
    return RunnableLambda(lambda _: result)


def must_not_be_called(*_):
    raise AssertionError("this dependency should not be used for this intent")


def scripted_agent(*replies: AIMessage) -> RunnableLambda:
    """Stands in for the tool-calling LLM: returns the given replies in order."""
    remaining = iter(replies)
    return RunnableLambda(lambda _: next(remaining))


def tool_call(name: str, **args) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": f"call_{name}"}])


@pytest.fixture
def tickets_db(tmp_path):
    return tmp_path / "tickets.db"


@pytest.fixture
def make_graph(order_db, tickets_db):
    def make(classifier, retriever=must_not_be_called, answerer=must_not_be_called, agent=None, tickets_db_path=tickets_db):
        return build_graph(
            classifier, retriever=retriever, answerer=answerer, agent=agent or RunnableLambda(must_not_be_called),
            db_path=order_db, tickets_db_path=tickets_db_path,
        )

    return make


# --- Routes that need no human ---

@pytest.mark.parametrize("intent", [Intent.GREETING, Intent.OUT_OF_SCOPE])
def test_greeting_and_out_of_scope_get_a_fixed_reply(make_graph, tickets_db, intent):
    result = make_graph(fake_classifier(intent)).invoke({"message": "anything"})

    assert result["response"] == RESPONSES[intent]
    # Every turn resets the per-turn fields, so they are always present in the result.
    assert result["chunks"] == []
    assert [type(m).__name__ for m in result["messages"]] == ["HumanMessage", "AIMessage"]
    assert list_tickets(tickets_db) == []


def test_answered_policy_question_is_not_escalated(make_graph, tickets_db):
    calls = []

    def retriever(question):
        calls.append(("retrieve", question))
        return [CHUNK]

    def answerer(question, chunks):
        calls.append(("answer", question, chunks))
        return PolicyAnswer(answered=True, answer="Express shipping costs $14.99.", sources=["shipping.md"])

    graph = make_graph(fake_classifier(Intent.POLICY_QUESTION), retriever=retriever, answerer=answerer)
    result = graph.invoke({"message": "How much is express shipping?"})

    assert calls == [("retrieve", "How much is express shipping?"), ("answer", "How much is express shipping?", [CHUNK])]
    assert result["response"] == "Express shipping costs $14.99."
    assert result["policy_answer"].sources == ["shipping.md"]
    assert list_tickets(tickets_db) == []


def test_order_issue_runs_the_requested_tool_and_returns_the_agents_answer(make_graph, tickets_db):
    agent = scripted_agent(tool_call("get_order_status", order_id="1042"), AIMessage("Your order 1042 has shipped."))

    result = make_graph(fake_classifier(Intent.ORDER_ISSUE), agent=agent).invoke({"message": "Where is my order #1042?"})

    tool_results = [m.content for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_results) == 1
    assert tool_results[0].startswith("Order 1042: Bose QuietComfort headphones. Status: shipped.")
    assert result["response"] == "Your order 1042 has shipped."
    assert list_tickets(tickets_db) == []


def test_invented_order_id_is_not_looked_up(make_graph):
    agent = scripted_agent(tool_call("get_order_status", order_id="123456"))

    result = make_graph(fake_classifier(Intent.ORDER_ISSUE), agent=agent).invoke({"message": "Can I return my laptop?"})

    assert result["response"] == ASK_FOR_ORDER_ID
    assert not any(isinstance(m, ToolMessage) for m in result["messages"])


# --- Escalations: each one must create a ticket with the right reason ---

def test_human_request_creates_a_ticket(make_graph, tickets_db):
    classifier = fake_classifier(Intent.HUMAN_REQUEST, Sentiment.NEGATIVE, order_id="1042")

    result = make_graph(classifier).invoke({"message": "I want to talk to a person about order 1042"})

    [ticket] = list_tickets(tickets_db)
    assert ticket.reason == EscalationReason.CUSTOMER_REQUEST
    assert (ticket.customer_message, ticket.intent, ticket.sentiment, ticket.order_id) == (
        "I want to talk to a person about order 1042", "human_request", "negative", "1042")
    assert result["ticket"] == ticket
    assert "ticket #1" in result["response"]


def test_angry_customer_is_escalated_before_the_agent_runs(make_graph, tickets_db):
    classifier = fake_classifier(Intent.ORDER_ISSUE, Sentiment.ANGRY, order_id="1002")

    result = make_graph(classifier).invoke({"message": "I want a refund NOW for order 1002"})  # agent would raise

    assert [t.reason for t in list_tickets(tickets_db)] == [EscalationReason.ANGRY_CUSTOMER]
    assert "ticket #1" in result["response"]


def test_merely_negative_customer_is_still_helped_by_the_agent(make_graph, tickets_db):
    agent = scripted_agent(AIMessage("You can return order 1006 until the deadline."))
    classifier = fake_classifier(Intent.ORDER_ISSUE, Sentiment.NEGATIVE, order_id="1006")

    result = make_graph(classifier, agent=agent).invoke({"message": "A bit disappointed, can I return order 1006?"})

    assert result["response"] == "You can return order 1006 until the deadline."
    assert list_tickets(tickets_db) == []


def test_unanswered_policy_question_creates_a_policy_not_found_ticket(make_graph, tickets_db):
    unanswered = PolicyAnswer(answered=False, answer=INSUFFICIENT_ANSWER, sources=[])
    graph = make_graph(fake_classifier(Intent.POLICY_QUESTION), retriever=lambda _: [CHUNK], answerer=lambda *_: unanswered)

    result = graph.invoke({"message": "Do you offer price matching?"})

    assert [t.reason for t in list_tickets(tickets_db)] == [EscalationReason.POLICY_NOT_FOUND]
    assert result["response"].startswith("I don't have VoltCart policy information")
    assert "ticket #1" in result["response"]


def test_unusable_model_output_creates_a_model_error_ticket(make_graph, tickets_db):
    def broken_answerer(*_):
        raise AnswerGenerationError("could not parse")

    graph = make_graph(fake_classifier(Intent.POLICY_QUESTION), retriever=lambda _: [CHUNK], answerer=broken_answerer)
    result = graph.invoke({"message": "How much is express shipping?"})

    assert [t.reason for t in list_tickets(tickets_db)] == [EscalationReason.MODEL_ERROR]
    assert "ticket #1" in result["response"]


def test_agent_that_keeps_calling_tools_is_escalated(make_graph, tickets_db):
    agent = RunnableLambda(lambda _: tool_call("get_order_status", order_id="1042"))  # never answers

    result = make_graph(fake_classifier(Intent.ORDER_ISSUE), agent=agent).invoke({"message": "Where is order 1042?"})

    assert [t.reason for t in list_tickets(tickets_db)] == [EscalationReason.AGENT_GAVE_UP]
    assert sum(isinstance(m, ToolMessage) for m in result["messages"]) == MAX_TOOL_ROUNDS
    # Every stored tool call has its result: no unanswered tool-call request is left in the history.
    assert len([c for m in result["messages"] for c in getattr(m, "tool_calls", [])]) == MAX_TOOL_ROUNDS


def test_failing_tool_creates_a_tool_error_ticket(make_graph, tickets_db):
    def broken_retriever(_):
        raise ConnectionError("Ollama is not running")

    agent = scripted_agent(tool_call("search_policies", query="damaged item"))
    graph = make_graph(fake_classifier(Intent.ORDER_ISSUE), retriever=broken_retriever, agent=agent)

    result = graph.invoke({"message": "My headphones arrived broken"})

    assert next(m for m in result["messages"] if isinstance(m, ToolMessage)).status == "error"
    assert [t.reason for t in list_tickets(tickets_db)] == [EscalationReason.TOOL_ERROR]
    assert "ticket #1" in result["response"]


def test_no_ticket_promise_when_the_ticket_cannot_be_saved(make_graph, tmp_path):
    # A directory is not a valid SQLite file, so saving the ticket fails.
    result = make_graph(fake_classifier(Intent.HUMAN_REQUEST), tickets_db_path=tmp_path).invoke({"message": "Human please"})

    assert result["response"] == TICKET_FAILED_REPLY
    assert result["ticket"] is None


# --- Setup and contracts ---

def test_missing_order_database_stops_the_graph_at_startup(tmp_path):
    with pytest.raises(FileNotFoundError, match="seed_orders"):
        build_graph(fake_classifier(Intent.GREETING), retriever=must_not_be_called, answerer=must_not_be_called,
                    agent=RunnableLambda(must_not_be_called), db_path=tmp_path / "missing.db")


def test_classifier_receives_the_customer_message(make_graph):
    received = []

    def classify(inputs):
        received.append(inputs)
        return IntentClassification(intent=Intent.GREETING, sentiment=Sentiment.NEUTRAL)

    make_graph(RunnableLambda(classify)).invoke({"message": "Hi"})

    assert received == [{"message": "Hi", "history": ""}]


def test_classification_rejects_unknown_intent():
    with pytest.raises(ValidationError):
        IntentClassification(intent="refund_request", sentiment="neutral")


@pytest.mark.parametrize(
    "raw, cleaned",
    [("1042", "1042"), ("#1042", "1042"), (" 1042 ", "1042"), ("[order number]", None), ("XX", None), ("", None), (None, None)],
)
def test_classification_keeps_only_real_order_numbers(raw, cleaned):
    # "[order number]" and "XX" are real outputs from qwen2.5:3b.
    assert IntentClassification(intent="order_issue", sentiment="neutral", order_id=raw).order_id == cleaned
