"""Multi-turn conversations with a checkpointer, using fakes (no models)."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from app.agent import current_turn
from app.graph import RESPONSES, build_graph
from app.memory import format_history, make_checkpointer, recent_messages
from app.schemas import Intent, IntentClassification, PolicyAnswer, RetrievedChunk, Sentiment
from app.tickets import list_tickets

CHUNK = RetrievedChunk(text="Shipping > Costs\n\nExpress costs $14.99.", source="shipping.md", section="Costs", score=0.7)
INTENTS = {
    "I want a human": Intent.HUMAN_REQUEST,
    "Hi": Intent.GREETING,
    "How much is express shipping?": Intent.POLICY_QUESTION,
    "Where is order 1042?": Intent.ORDER_ISSUE,
    "And order 1005?": Intent.ORDER_ISSUE,
}


def classifier_by_message() -> RunnableLambda:
    """Fake classifier: the intent depends on the message, so one conversation can switch intents."""
    return RunnableLambda(
        lambda inputs: IntentClassification(intent=INTENTS[inputs["message"]], sentiment=Sentiment.NEUTRAL)
    )


@pytest.fixture
def tickets_db(tmp_path):
    return tmp_path / "tickets.db"


@pytest.fixture
def make_graph(order_db, tickets_db):
    def make(agent=None):
        answered = PolicyAnswer(answered=True, answer="Express shipping costs $14.99.", sources=["shipping.md"])
        return build_graph(
            classifier_by_message(), retriever=lambda _: [CHUNK], answerer=lambda *_: answered,
            agent=agent or RunnableLambda(lambda _: AIMessage("unused")), db_path=order_db,
            tickets_db_path=tickets_db, checkpointer=make_checkpointer(),
        )

    return make


def thread(thread_id="t1"):
    return {"configurable": {"thread_id": thread_id}}


def test_escalation_does_not_carry_over_to_the_next_turn(make_graph, tickets_db):
    graph = make_graph()

    graph.invoke({"message": "I want a human"}, thread())
    second = graph.invoke({"message": "Hi"}, thread())

    assert second["response"] == RESPONSES[Intent.GREETING]
    assert second.get("escalation_reason") is None
    assert len(list_tickets(tickets_db)) == 1


def test_policy_answer_does_not_carry_over_to_the_next_turn(make_graph):
    graph = make_graph()

    graph.invoke({"message": "How much is express shipping?"}, thread())
    second = graph.invoke({"message": "Hi"}, thread())

    assert second.get("policy_answer") is None
    assert second.get("chunks", []) == []


def test_agent_sees_the_new_customer_message_on_the_second_turn(make_graph):
    seen = []

    def agent(messages):
        seen.append(messages)
        return AIMessage(f"answer {len(seen)}")

    graph = make_graph(agent=RunnableLambda(agent))
    graph.invoke({"message": "Where is order 1042?"}, thread())
    second = graph.invoke({"message": "And order 1005?"}, thread())

    last_customer_message = [m.content for m in seen[-1] if isinstance(m, HumanMessage)][-1]
    assert last_customer_message == "And order 1005?"
    assert second["response"] == "answer 2"


def test_classifier_sees_the_earlier_conversation(order_db, tickets_db):
    received = []

    def classify(inputs):
        received.append(inputs)
        return IntentClassification(intent=INTENTS[inputs["message"]], sentiment=Sentiment.NEUTRAL)

    graph = build_graph(
        RunnableLambda(classify), retriever=lambda _: [CHUNK],
        answerer=lambda *_: PolicyAnswer(answered=True, answer="Express shipping costs $14.99.", sources=["shipping.md"]),
        agent=RunnableLambda(lambda _: AIMessage("unused")), db_path=order_db, tickets_db_path=tickets_db,
        checkpointer=make_checkpointer(),
    )
    graph.invoke({"message": "How much is express shipping?"}, thread())
    graph.invoke({"message": "Hi"}, thread())

    assert received[0]["history"] == ""
    assert received[1]["history"] == "Customer: How much is express shipping?\nAssistant: Express shipping costs $14.99."


def test_history_leaves_out_tool_steps_and_keeps_only_recent_messages():
    messages = [
        HumanMessage("Where is order 1042?"),
        AIMessage("", tool_calls=[{"name": "get_order_status", "args": {"order_id": "1042"}, "id": "1"}]),
        ToolMessage("Order 1042: shipped.", tool_call_id="1"),
        AIMessage("Your order has shipped."),
        HumanMessage("Thanks"),
    ]

    assert format_history(messages) == "Customer: Where is order 1042?\nAssistant: Your order has shipped.\nCustomer: Thanks"
    assert format_history(messages, max_messages=1) == "Customer: Thanks"


def order_turn(i: int) -> list:
    call = {"name": "get_order_status", "args": {"order_id": "1042"}, "id": f"call_{i}"}
    return [HumanMessage(f"question {i}"), AIMessage("", tool_calls=[call]), ToolMessage("shipped", tool_call_id=f"call_{i}"), AIMessage(f"answer {i}")]


def test_current_turn_without_a_customer_message_is_everything():
    messages = [AIMessage("Hello")]

    assert current_turn(messages) == messages


def test_short_conversation_is_not_trimmed():
    messages = order_turn(0) + [HumanMessage("question 1")]

    assert recent_messages(messages, max_earlier=20) == messages


def test_long_conversation_keeps_the_current_turn_and_recent_history_from_a_customer_message():
    messages = [m for i in range(10) for m in order_turn(i)] + [HumanMessage("current question")]

    kept = recent_messages(messages, max_earlier=6)

    assert kept[-1].content == "current question"
    assert isinstance(kept[0], HumanMessage)  # never starts with an orphaned tool result or reply
    assert [m.content for m in kept if isinstance(m, HumanMessage)] == ["question 9", "current question"]


def test_current_turn_is_never_cut_even_if_it_is_long():
    current = order_turn(99)[:3] + order_turn(98)[1:3] + order_turn(97)[1:3]  # one question, three tool rounds
    messages = order_turn(0) + current

    kept = recent_messages(messages, max_earlier=2)

    assert kept == current


def test_saved_state_reloads_without_unregistered_type_warnings(make_graph, caplog):
    graph = make_graph()

    graph.invoke({"message": "I want a human"}, thread())  # stores classification, reason and ticket
    graph.invoke({"message": "How much is express shipping?"}, thread())

    assert "Deserializing unregistered type" not in caplog.text


def test_ticket_includes_the_earlier_conversation(make_graph, tickets_db):
    graph = make_graph()

    graph.invoke({"message": "How much is express shipping?"}, thread())
    graph.invoke({"message": "I want a human"}, thread())

    [ticket] = list_tickets(tickets_db)
    assert ticket.conversation == (
        "Customer: How much is express shipping?\nAssistant: Express shipping costs $14.99.\nCustomer: I want a human"
    )


def test_sqlite_conversations_survive_a_restart(order_db, tickets_db, tmp_path):
    received = []

    def classify(inputs):
        received.append(inputs)
        return IntentClassification(intent=INTENTS[inputs["message"]], sentiment=Sentiment.NEUTRAL)

    def start_server():
        # A brand-new checkpointer each time, like a server process starting up.
        return build_graph(
            RunnableLambda(classify), retriever=lambda _: [CHUNK],
            answerer=lambda *_: PolicyAnswer(answered=True, answer="Express shipping costs $14.99.", sources=["shipping.md"]),
            agent=RunnableLambda(lambda _: AIMessage("unused")), db_path=order_db, tickets_db_path=tickets_db,
            checkpointer=make_checkpointer(tmp_path / "conversations.db"),
        )

    start_server().invoke({"message": "How much is express shipping?"}, thread())
    start_server().invoke({"message": "Hi"}, thread())

    assert received[1]["history"] == "Customer: How much is express shipping?\nAssistant: Express shipping costs $14.99."


def test_separate_threads_do_not_share_state(make_graph, tickets_db):
    graph = make_graph()

    graph.invoke({"message": "I want a human"}, thread("a"))
    other = graph.invoke({"message": "Hi"}, thread("b"))

    assert other["response"] == RESPONSES[Intent.GREETING]
    assert other.get("ticket") is None
