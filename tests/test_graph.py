import pytest
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from app.graph import RESPONSES, build_graph
from app.schemas import Intent, IntentClassification, Sentiment


def fake_classifier(intent: Intent) -> RunnableLambda:
    """Stands in for the LLM classifier and always returns the given intent."""
    result = IntentClassification(intent=intent, sentiment=Sentiment.NEUTRAL)
    return RunnableLambda(lambda _: result)


@pytest.mark.parametrize("intent", list(Intent))
def test_graph_replies_for_every_intent(intent):
    graph = build_graph(classifier=fake_classifier(intent))

    result = graph.invoke({"message": "anything"})

    assert result["classification"].intent == intent
    assert result["response"] == RESPONSES[intent]


def test_classifier_receives_the_customer_message():
    received = []

    def classify(inputs):
        received.append(inputs)
        return IntentClassification(intent=Intent.ORDER_ISSUE, sentiment=Sentiment.NEUTRAL, order_id="1042")

    build_graph(classifier=RunnableLambda(classify)).invoke({"message": "Where is order 1042?"})

    assert received == [{"message": "Where is order 1042?"}]


def test_classification_rejects_unknown_intent():
    with pytest.raises(ValidationError):
        IntentClassification(intent="refund_request", sentiment="neutral")
