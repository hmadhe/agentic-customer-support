from enum import StrEnum

from pydantic import BaseModel, Field


class Intent(StrEnum):
    POLICY_QUESTION = "policy_question"
    ORDER_ISSUE = "order_issue"
    HUMAN_REQUEST = "human_request"
    GREETING = "greeting"
    OUT_OF_SCOPE = "out_of_scope"


class Sentiment(StrEnum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class IntentClassification(BaseModel):
    """How a customer message should be routed. The LLM must return exactly this shape."""

    intent: Intent = Field(description="What the customer wants.")
    sentiment: Sentiment = Field(description="How the customer feels.")
    order_id: str | None = Field(
        default=None,
        description="The order number if the customer mentions one, digits only (e.g. '1042'), otherwise null.",
    )
