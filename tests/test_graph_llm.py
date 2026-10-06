"""End-to-end tests through the whole graph with real models. Slow and needs Ollama: `pytest -m llm`."""

import pytest
from langchain_core.messages import ToolMessage

from app.agent import current_turn
from app.graph import build_graph
from app.memory import make_checkpointer
from app.retriever import retrieve
from app.schemas import Intent
from app.tickets import EscalationReason

pytestmark = pytest.mark.llm


@pytest.fixture(scope="module")
def graph(policy_store, order_db, tmp_path_factory):
    # Real classifier, answer chain and agent; test copies of the policy store, order and ticket databases.
    return build_graph(
        retriever=lambda question: retrieve(question, vector_store=policy_store),
        db_path=order_db,
        tickets_db_path=tmp_path_factory.mktemp("tickets") / "tickets.db",
    )


@pytest.fixture(scope="module")
def chat_graph(policy_store, order_db, tmp_path_factory):
    # Same as `graph`, but it remembers conversations by thread_id.
    return build_graph(
        retriever=lambda question: retrieve(question, vector_store=policy_store),
        db_path=order_db,
        tickets_db_path=tmp_path_factory.mktemp("tickets") / "tickets.db",
        checkpointer=make_checkpointer(),
    )


def tools_called(result) -> list[str]:
    # Only this turn's tool calls: with memory, messages holds the whole conversation.
    return [call["name"] for message in current_turn(result["messages"]) for call in getattr(message, "tool_calls", [])]


def chat(graph, thread_id: str, *messages: str):
    config = {"configurable": {"thread_id": thread_id}}
    return [graph.invoke({"message": message}, config) for message in messages]


def test_follow_up_uses_the_order_from_the_previous_turn(chat_graph):
    _, second = chat(chat_graph, "order-follow-up", "Where is my order 1001?", "Can I return it?")

    calls = [call for m in current_turn(second["messages"]) for call in getattr(m, "tool_calls", [])]
    assert [(c["name"], c["args"]["order_id"]) for c in calls] == [("check_return_eligibility", "1001")]
    assert "15%" in second["response"]


def test_escalation_in_one_turn_does_not_affect_the_next(chat_graph):
    first, second = chat(chat_graph, "after-escalation", "I want to talk to a real person.", "How much is express shipping?")

    assert first["escalation_reason"] == EscalationReason.CUSTOMER_REQUEST
    assert second["escalation_reason"] is None
    assert "14.99" in second["response"]


@pytest.mark.xfail(
    reason="Known limitation: with qwen2.5:3b, 'And how long does it take?' after a shipping question is classified "
    "order_issue even with the conversation as context, so the agent asks for an order number.",
    strict=False,
)
def test_policy_follow_up_with_only_a_pronoun(chat_graph):
    _, second = chat(chat_graph, "policy-follow-up", "How much is express shipping?", "And how long does it take?")

    assert second["classification"].intent == Intent.POLICY_QUESTION


def test_policy_question_is_answered_from_the_policies(graph):
    result = graph.invoke({"message": "How much is express shipping?"})

    assert result["classification"].intent == Intent.POLICY_QUESTION
    assert "14.99" in result["response"]
    assert result["policy_answer"].sources == ["shipping.md"]


def test_unanswerable_policy_question_is_escalated(graph):
    result = graph.invoke({"message": "Do you offer price matching?"})

    assert result["escalation_reason"] == EscalationReason.POLICY_NOT_FOUND
    assert f"ticket #{result['ticket'].ticket_id}" in result["response"]


def test_request_for_a_human_is_escalated(graph):
    result = graph.invoke({"message": "I want to talk to a real person."})

    assert result["escalation_reason"] == EscalationReason.CUSTOMER_REQUEST
    assert result["ticket"].customer_message == "I want to talk to a real person."


def test_angry_customer_is_escalated(graph):
    result = graph.invoke({"message": "I'm so angry, I want a refund NOW for order 1002"})

    assert result["escalation_reason"] == EscalationReason.ANGRY_CUSTOMER
    assert result["ticket"].order_id == "1002"


def test_order_status_is_looked_up(graph):
    result = graph.invoke({"message": "Where is my order #1042?"})

    assert result["classification"].intent == Intent.ORDER_ISSUE
    assert tools_called(result) == ["get_order_status"]
    assert "VC104200" in result["response"]


def test_return_eligibility_is_checked_with_the_tool(graph):
    result = graph.invoke({"message": "Can I return the laptop from order 1001?"})

    assert "check_return_eligibility" in tools_called(result)
    assert "15%" in result["response"]


def test_non_returnable_order_is_refused(graph):
    result = graph.invoke({"message": "Can I return my earbuds? Order 1004."})

    assert "check_return_eligibility" in tools_called(result)
    assert "hygiene" in result["response"].lower()


def test_missing_order_number_is_asked_for_not_invented(graph):
    result = graph.invoke({"message": "Can I return the laptop I bought last week?"})

    assert not any(isinstance(message, ToolMessage) for message in result["messages"])
    # The model words this itself when it asks without trying a lookup ("order number" or "order ID").
    assert "order number" in result["response"].lower() or "order id" in result["response"].lower()


def test_damaged_item_question_uses_the_policy_tool(graph):
    # Milestone 3's gap: this own-purchase question skipped RAG. The agent can now search the policies.
    result = graph.invoke({"message": "My headphones from order 2231 arrived broken, what can I do?"})

    assert "search_policies" in tools_called(result)
    assert "48 hours" in result["response"]
