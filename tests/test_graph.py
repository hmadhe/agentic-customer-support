import pytest
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from app.answer import INSUFFICIENT_ANSWER
from app.graph import RESPONSES, build_graph
from app.schemas import Intent, IntentClassification, PolicyAnswer, RetrievedChunk, Sentiment

CHUNK = RetrievedChunk(text="Shipping > Costs\n\nExpress costs $14.99.", source="shipping.md", section="Costs", score=0.7)


def fake_classifier(intent: Intent) -> RunnableLambda:
    """Stands in for the LLM classifier and always returns the given intent."""
    result = IntentClassification(intent=intent, sentiment=Sentiment.NEUTRAL)
    return RunnableLambda(lambda _: result)


def must_not_be_called(*_):
    raise AssertionError("RAG should not run for this intent")


@pytest.mark.parametrize("intent", [intent for intent in Intent if intent != Intent.POLICY_QUESTION])
def test_non_policy_intents_get_a_placeholder_and_skip_rag(intent):
    graph = build_graph(fake_classifier(intent), retriever=must_not_be_called, answerer=must_not_be_called)

    result = graph.invoke({"message": "anything"})

    assert result["response"] == RESPONSES[intent]
    # graph.invoke() only returns fields a node wrote; Pydantic defaults are not included.
    assert "chunks" not in result
    assert "policy_answer" not in result


def test_policy_question_goes_through_retrieve_and_answer():
    calls = []

    def retriever(question):
        calls.append(("retrieve", question))
        return [CHUNK]

    def answerer(question, chunks):
        calls.append(("answer", question, chunks))
        return PolicyAnswer(answered=True, answer="Express shipping costs $14.99.", sources=["shipping.md"])

    graph = build_graph(fake_classifier(Intent.POLICY_QUESTION), retriever=retriever, answerer=answerer)
    result = graph.invoke({"message": "How much is express shipping?"})

    assert calls == [
        ("retrieve", "How much is express shipping?"),
        ("answer", "How much is express shipping?", [CHUNK]),
    ]
    assert result["chunks"] == [CHUNK]
    assert result["response"] == "Express shipping costs $14.99."
    assert result["policy_answer"].sources == ["shipping.md"]


def test_unanswered_policy_question_returns_the_insufficient_reply():
    unanswered = PolicyAnswer(answered=False, answer=INSUFFICIENT_ANSWER, sources=[])
    graph = build_graph(
        fake_classifier(Intent.POLICY_QUESTION), retriever=lambda _: [CHUNK], answerer=lambda *_: unanswered
    )

    result = graph.invoke({"message": "Do you offer price matching?"})

    assert result["response"] == INSUFFICIENT_ANSWER
    assert result["policy_answer"].answered is False


def test_classifier_receives_the_customer_message():
    received = []

    def classify(inputs):
        received.append(inputs)
        return IntentClassification(intent=Intent.ORDER_ISSUE, sentiment=Sentiment.NEUTRAL, order_id="1042")

    graph = build_graph(RunnableLambda(classify), retriever=must_not_be_called, answerer=must_not_be_called)
    graph.invoke({"message": "Where is order 1042?"})

    assert received == [{"message": "Where is order 1042?"}]


def test_classification_rejects_unknown_intent():
    with pytest.raises(ValidationError):
        IntentClassification(intent="refund_request", sentiment="neutral")
