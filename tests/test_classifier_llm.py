"""Checks the real model's classifications. Slow and needs Ollama running: `pytest -m llm`."""

import pytest

from app.classifier import build_classifier
from app.schemas import Intent, Sentiment

pytestmark = pytest.mark.llm

INTENT_CASES = [
    ("Hi there!", Intent.GREETING),
    ("What is your return policy?", Intent.POLICY_QUESTION),
    ("How long does shipping take?", Intent.POLICY_QUESTION),
    ("Where is my order #1042?", Intent.ORDER_ISSUE),
    ("The headphones in order 2231 arrived broken!", Intent.ORDER_ISSUE),
    ("I want to talk to a real person.", Intent.HUMAN_REQUEST),
    # Regression: these were classified as "greeting" before the prompt fix in Milestone 1.
    ("What's the capital of France?", Intent.OUT_OF_SCOPE),
    ("Can you write me a poem about cats?", Intent.OUT_OF_SCOPE),
    ("What's the weather tomorrow?", Intent.OUT_OF_SCOPE),
    # Regression: general shipping questions were classified as "order_issue" before the second prompt fix.
    ("What are your shipping times?", Intent.POLICY_QUESTION),
    ("How much does shipping cost?", Intent.POLICY_QUESTION),
    # ...while questions about the customer's own order must stay "order_issue", even without an order number.
    ("How long does shipping take for my order 1042?", Intent.ORDER_ISSUE),
    ("My package still hasn't arrived", Intent.ORDER_ISSUE),
    ("Can I return the laptop I bought last week?", Intent.ORDER_ISSUE),
]


@pytest.fixture(scope="module")
def classifier():
    return build_classifier()


@pytest.mark.parametrize("message, expected", INTENT_CASES)
def test_intent(classifier, message, expected):
    assert classifier.invoke({"message": message}).intent == expected


def test_extracts_order_id_without_hash(classifier):
    assert classifier.invoke({"message": "Where is my order #1042?"}).order_id == "1042"


def test_detects_negative_sentiment(classifier):
    result = classifier.invoke({"message": "This is the third time I'm asking, your service is terrible!"})
    assert result.sentiment == Sentiment.NEGATIVE
