"""End-to-end tests through the whole graph with real models. Slow and needs Ollama: `pytest -m llm`."""

import pytest
from langchain_core.messages import ToolMessage

from app.answer import INSUFFICIENT_ANSWER
from app.graph import build_graph
from app.retriever import retrieve
from app.schemas import Intent

pytestmark = pytest.mark.llm


@pytest.fixture(scope="module")
def graph(policy_store, order_db):
    # Real classifier, answer chain and agent; test copies of the policy store and order database.
    return build_graph(retriever=lambda question: retrieve(question, vector_store=policy_store), db_path=order_db)


def tools_called(result) -> list[str]:
    return [call["name"] for message in result["messages"] for call in getattr(message, "tool_calls", [])]


def test_policy_question_is_answered_from_the_policies(graph):
    result = graph.invoke({"message": "How much is express shipping?"})

    assert result["classification"].intent == Intent.POLICY_QUESTION
    assert "14.99" in result["response"]
    assert result["policy_answer"].sources == ["shipping.md"]


def test_unanswerable_policy_question_gets_the_insufficient_reply(graph):
    result = graph.invoke({"message": "Do you offer price matching?"})

    assert result["response"] == INSUFFICIENT_ANSWER


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
    assert "order number" in result["response"].lower()


def test_damaged_item_question_uses_the_policy_tool(graph):
    # Milestone 3's gap: this own-purchase question skipped RAG. The agent can now search the policies.
    result = graph.invoke({"message": "My headphones from order 2231 arrived broken, what can I do?"})

    assert "search_policies" in tools_called(result)
    assert "48 hours" in result["response"]
