import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from app.agent import ASK_FOR_ORDER_ID, FALLBACK_REPLY, MAX_TOOL_ROUNDS
from app.answer import INSUFFICIENT_ANSWER
from app.graph import RESPONSES, build_graph
from app.schemas import Intent, IntentClassification, PolicyAnswer, RetrievedChunk, Sentiment

CHUNK = RetrievedChunk(text="Shipping > Costs\n\nExpress costs $14.99.", source="shipping.md", section="Costs", score=0.7)


def fake_classifier(intent: Intent) -> RunnableLambda:
    """Stands in for the LLM classifier and always returns the given intent."""
    result = IntentClassification(intent=intent, sentiment=Sentiment.NEUTRAL)
    return RunnableLambda(lambda _: result)


def must_not_be_called(*_):
    raise AssertionError("this dependency should not be used for this intent")


def scripted_agent(*replies: AIMessage) -> RunnableLambda:
    """Stands in for the tool-calling LLM: returns the given replies in order."""
    remaining = iter(replies)
    return RunnableLambda(lambda _: next(remaining))


def tool_call(name: str, **args) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": f"call_{name}"}])


def make_graph(order_db, intent, retriever=must_not_be_called, answerer=must_not_be_called, agent=None):
    agent = agent or RunnableLambda(must_not_be_called)
    return build_graph(fake_classifier(intent), retriever=retriever, answerer=answerer, agent=agent, db_path=order_db)


@pytest.mark.parametrize("intent", [Intent.HUMAN_REQUEST, Intent.GREETING, Intent.OUT_OF_SCOPE])
def test_other_intents_get_a_placeholder_and_skip_rag_and_tools(order_db, intent):
    result = make_graph(order_db, intent).invoke({"message": "anything"})

    assert result["response"] == RESPONSES[intent]
    # Plain fields appear in the result only once a node writes them. Fields with a reducer
    # (messages uses add_messages) always appear, starting empty.
    assert "chunks" not in result
    assert result["messages"] == []


def test_policy_question_goes_through_retrieve_and_answer(order_db):
    calls = []

    def retriever(question):
        calls.append(("retrieve", question))
        return [CHUNK]

    def answerer(question, chunks):
        calls.append(("answer", question, chunks))
        return PolicyAnswer(answered=True, answer="Express shipping costs $14.99.", sources=["shipping.md"])

    graph = make_graph(order_db, Intent.POLICY_QUESTION, retriever=retriever, answerer=answerer)
    result = graph.invoke({"message": "How much is express shipping?"})

    assert calls == [
        ("retrieve", "How much is express shipping?"),
        ("answer", "How much is express shipping?", [CHUNK]),
    ]
    assert result["response"] == "Express shipping costs $14.99."
    assert result["policy_answer"].sources == ["shipping.md"]


def test_unanswered_policy_question_returns_the_insufficient_reply(order_db):
    unanswered = PolicyAnswer(answered=False, answer=INSUFFICIENT_ANSWER, sources=[])
    graph = make_graph(order_db, Intent.POLICY_QUESTION, retriever=lambda _: [CHUNK], answerer=lambda *_: unanswered)

    assert graph.invoke({"message": "Do you offer price matching?"})["response"] == INSUFFICIENT_ANSWER


def test_order_issue_runs_the_requested_tool_and_returns_the_agents_answer(order_db):
    agent = scripted_agent(tool_call("get_order_status", order_id="1042"), AIMessage("Your order 1042 has shipped."))

    result = make_graph(order_db, Intent.ORDER_ISSUE, agent=agent).invoke({"message": "Where is my order #1042?"})

    tool_results = [m.content for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_results) == 1
    assert tool_results[0].startswith("Order 1042: Bose QuietComfort headphones. Status: shipped.")
    assert "Tracking number: VC104200." in tool_results[0]
    assert result["response"] == "Your order 1042 has shipped."


def test_invented_order_id_is_not_looked_up(order_db):
    agent = scripted_agent(tool_call("get_order_status", order_id="123456"))

    result = make_graph(order_db, Intent.ORDER_ISSUE, agent=agent).invoke({"message": "Can I return the laptop I bought?"})

    assert result["response"] == ASK_FOR_ORDER_ID
    assert not any(isinstance(m, ToolMessage) for m in result["messages"])


def test_agent_that_keeps_calling_tools_is_stopped(order_db):
    agent = RunnableLambda(lambda _: tool_call("get_order_status", order_id="1042"))  # never answers

    result = make_graph(order_db, Intent.ORDER_ISSUE, agent=agent).invoke({"message": "Where is order 1042?"})

    assert result["response"] == FALLBACK_REPLY
    assert sum(isinstance(m, ToolMessage) for m in result["messages"]) == MAX_TOOL_ROUNDS


def test_failing_tool_is_reported_to_the_agent_instead_of_crashing(order_db):
    def broken_retriever(_):
        raise ConnectionError("Ollama is not running")

    agent = scripted_agent(tool_call("search_policies", query="damaged item"), AIMessage("Sorry, please try again later."))
    graph = make_graph(order_db, Intent.ORDER_ISSUE, retriever=broken_retriever, agent=agent)

    result = graph.invoke({"message": "My headphones arrived broken"})

    tool_message = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert tool_message.status == "error"
    assert result["response"] == "Sorry, please try again later."


def test_missing_order_database_stops_the_graph_at_startup(tmp_path):
    with pytest.raises(FileNotFoundError, match="seed_orders"):
        make_graph(tmp_path / "missing.db", Intent.GREETING)


def test_classifier_receives_the_customer_message(order_db):
    received = []

    def classify(inputs):
        received.append(inputs)
        return IntentClassification(intent=Intent.GREETING, sentiment=Sentiment.NEUTRAL)

    build_graph(RunnableLambda(classify), retriever=must_not_be_called, answerer=must_not_be_called,
                agent=RunnableLambda(must_not_be_called), db_path=order_db).invoke({"message": "Hi"})

    assert received == [{"message": "Hi"}]


def test_classification_rejects_unknown_intent():
    with pytest.raises(ValidationError):
        IntentClassification(intent="refund_request", sentiment="neutral")
