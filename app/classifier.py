from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from app.llm import get_llm
from app.schemas import IntentClassification

SYSTEM_PROMPT = """You classify messages sent to VoltCart customer support. VoltCart is an online electronics store.

Choose exactly one intent:
- policy_question: general questions about how VoltCart works (shipping times and costs, returns, refunds, warranty, payments, accounts).
- order_issue: the customer refers to THEIR OWN order (an order number, "my order", "my package", "the item I bought"), e.g. its status, a delay, a damaged or wrong item, or returning or cancelling it.
- human_request: the customer asks to talk to a human, an agent or a manager.
- greeting: ONLY a greeting, thanks or goodbye (e.g. "hi", "thanks!", "bye"), with no question or request.
- out_of_scope: any question or request not about VoltCart, its products, orders or policies (e.g. general knowledge, weather, writing poems, homework).

Rules:
- If the message contains a question or a request, it is never a greeting.
- A question about shipping, returns or refunds in general is policy_question. It is order_issue only if the customer refers to their own order.

Sentiment:
- angry: the customer is hostile or furious (insults, swearing, shouting, demands like "NOW", or complaining that they keep being ignored).
- negative: the customer is unhappy, disappointed or frustrated, but not angry.
- positive: the customer is happy or thankful.
- neutral: otherwise.

order_id: the order number if one is mentioned (digits only), otherwise null."""

# Used only for follow-up messages. The first message of a conversation keeps the original prompt: wrapping every
# message in this template made qwen2.5:3b fail 5 of 18 Milestone 1 tests (see the Milestone 6 log).
FOLLOW_UP_TEMPLATE = """Earlier conversation (for context only):
{history}

Classify ONLY this latest customer message. Use the earlier conversation just to understand what it refers to,
for example "it" or "and how long does it take?". If it refers to an order from earlier in the conversation,
use that order number as order_id.

Latest customer message:
{message}"""


def build_classifier(llm: BaseChatModel | None = None) -> Runnable:
    """Prompt + LLM that returns a validated IntentClassification.

    Input: {"message": str} or {"message": str, "history": str}. With an empty or missing history,
    the message is classified exactly as before conversation memory was added.
    """
    model = (llm or get_llm()).with_structured_output(IntentClassification)
    single = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{message}")]) | model
    follow_up = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", FOLLOW_UP_TEMPLATE)]) | model
    return RunnableLambda(lambda inputs: (follow_up if inputs.get("history") else single).invoke(inputs))
