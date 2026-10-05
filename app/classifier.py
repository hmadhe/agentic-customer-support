from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

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

Sentiment: negative if the customer is upset, angry or frustrated; positive if happy or thankful; otherwise neutral.

order_id: the order number if one is mentioned (digits only), otherwise null."""


def build_classifier(llm: BaseChatModel | None = None) -> Runnable:
    """Prompt + LLM that returns a validated IntentClassification. Input: {"message": str}."""
    llm = llm or get_llm()
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{message}")])
    return prompt | llm.with_structured_output(IntentClassification)
