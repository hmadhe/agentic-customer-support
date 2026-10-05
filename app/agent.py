import re

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from app.llm import get_llm

AGENT_PROMPT = """You are VoltCart's customer support assistant, helping a customer with their own order.

Rules:
- Use the tools to look up facts. Never guess order details, dates, fees or policies.
- get_order_status: the order's product, status, dates and tracking number.
- check_return_eligibility: whether the order can be returned, the deadline and any restocking fee.
- search_policies: general rules, for example what to do about a damaged or wrong item.
- If you need an order number and the customer did not give one, ask for it. Never make one up.
- When you have what you need, answer the customer in 1-3 short sentences."""

# The agent may request tools at most this many times per message, so it can never loop forever.
MAX_TOOL_ROUNDS = 3
FALLBACK_REPLY = "I'm sorry, I couldn't complete that request. A member of our support team will help you."
ASK_FOR_ORDER_ID = "Could you please give me your order number? You can find it in your order confirmation email."


def invented_order_ids(tool_calls: list[dict], customer_message: str) -> list[str]:
    """Order numbers the agent wants to look up that the customer never wrote.

    Prompting alone did not stop qwen2.5:3b from inventing one ("123456"), so this is enforced in code.
    """
    customer_numbers = set(re.findall(r"\d+", customer_message))
    requested = [str(call["args"].get("order_id", "")).strip().lstrip("#") for call in tool_calls if "order_id" in call["args"]]
    return [order_id for order_id in requested if order_id not in customer_numbers]


def build_agent(tools: list[BaseTool], llm: BaseChatModel | None = None) -> Runnable:
    """The chat model with the tools bound to it. Input: a list of messages. Output: an AIMessage."""
    return (llm or get_llm()).bind_tools(tools)
