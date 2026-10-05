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
    NEGATIVE = "negative"  # unhappy, disappointed or frustrated: the bot should still help
    ANGRY = "angry"  # hostile or furious: escalated to a human (Milestone 5)


class IntentClassification(BaseModel):
    """How a customer message should be routed. The LLM must return exactly this shape."""

    intent: Intent = Field(description="What the customer wants.")
    sentiment: Sentiment = Field(description="How the customer feels.")
    order_id: str | None = Field(
        default=None,
        description="The order number if the customer mentions one, digits only (e.g. '1042'), otherwise null.",
    )


class RetrievedChunk(BaseModel):
    """A piece of a policy document returned by the retriever."""

    text: str
    source: str  # file name, e.g. "returns.md"
    section: str  # heading within the document, e.g. "Restocking fee"
    score: float  # relevance from 0 (unrelated) to 1 (identical meaning)


class PolicyAnswer(BaseModel):
    """An answer to a policy question, based only on retrieved policy text.

    `answered` comes first so the model decides whether the excerpts contain the answer before writing one.
    `sources` deliberately has no default: an optional field is one the model is allowed to skip, and it did.
    """

    answered: bool = Field(description="true only if the policy excerpts contain the answer to the question.")
    answer: str = Field(description="The answer for the customer, in 1-3 sentences.")
    sources: list[str] = Field(description="File names of the excerpts the answer is based on, e.g. 'returns.md'.")
