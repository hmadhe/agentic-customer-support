"""End-to-end tests through the whole graph with real models. Slow and needs Ollama: `pytest -m llm`."""

import pytest

from app.answer import INSUFFICIENT_ANSWER
from app.graph import RESPONSES, build_graph
from app.retriever import retrieve
from app.schemas import Intent

pytestmark = pytest.mark.llm


@pytest.fixture(scope="module")
def graph(policy_store):
    # Real classifier and answer chain; retrieval uses the test copy of the policy store.
    return build_graph(retriever=lambda question: retrieve(question, vector_store=policy_store))


def test_policy_question_is_answered_from_the_policies(graph):
    result = graph.invoke({"message": "How much is express shipping?"})

    assert result["classification"].intent == Intent.POLICY_QUESTION
    assert "14.99" in result["response"]
    assert result["policy_answer"].sources == ["shipping.md"]


def test_unanswerable_policy_question_gets_the_insufficient_reply(graph):
    result = graph.invoke({"message": "Do you offer price matching?"})

    assert result["classification"].intent == Intent.POLICY_QUESTION
    assert result["response"] == INSUFFICIENT_ANSWER


def test_order_question_skips_rag(graph):
    result = graph.invoke({"message": "Where is my order #1042?"})

    assert result["classification"].intent == Intent.ORDER_ISSUE
    assert result["response"] == RESPONSES[Intent.ORDER_ISSUE]
    assert "chunks" not in result


@pytest.mark.xfail(
    reason="Known gap: questions about the customer's own purchase are classified order_issue, so they skip RAG "
    "even when the answer is in the policies. Planned fix in Milestone 4: a search_policies tool for the order agent.",
    strict=False,
)
def test_own_purchase_return_question_gets_a_policy_answer(graph):
    result = graph.invoke({"message": "Can I return the laptop I bought last week?"})

    assert "policy_answer" in result
