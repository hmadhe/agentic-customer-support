from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from app.agent import current_turn

# How many earlier customer/assistant messages the classifier sees as context.
CLASSIFIER_HISTORY_MESSAGES = 6
# How many messages from earlier turns the agent sees (about 5 order turns, roughly 750 tokens).
# Without a limit, a 40-turn conversation filled the 4096-token window and Ollama silently dropped part of it.
AGENT_HISTORY_MESSAGES = 20
# How many recent customer/assistant messages are saved with a support ticket.
TICKET_HISTORY_MESSAGES = 10

# Our own types stored in the graph state. The checkpointer only reloads types that are explicitly allowed:
# without this list LangGraph warns "Deserializing unregistered type ... will be blocked in a future version".
STATE_TYPES = [
    ("app.schemas", "Intent"),
    ("app.schemas", "Sentiment"),
    ("app.schemas", "IntentClassification"),
    ("app.schemas", "RetrievedChunk"),
    ("app.schemas", "PolicyAnswer"),
    ("app.tickets", "EscalationReason"),
    ("app.tickets", "Ticket"),
]


def make_checkpointer() -> InMemorySaver:
    """Saves each conversation (by thread_id) in memory: kept between messages, lost when the process stops."""
    return InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=STATE_TYPES))


def recent_messages(messages: list[AnyMessage], max_earlier: int = AGENT_HISTORY_MESSAGES) -> list[AnyMessage]:
    """What the agent sees: the current turn in full, plus at most `max_earlier` messages from earlier turns.

    The kept history always starts at a customer message, so it never begins with a tool result whose
    tool call was cut off, or with a reply whose question is missing.
    """
    turn = current_turn(messages)
    earlier = messages[: len(messages) - len(turn)][-max_earlier:]
    while earlier and not isinstance(earlier[0], HumanMessage):
        earlier = earlier[1:]
    return earlier + turn


def format_history(messages: list[AnyMessage], max_messages: int = CLASSIFIER_HISTORY_MESSAGES) -> str:
    """The recent conversation as plain text: what the customer wrote and what the assistant replied.

    Tool calls and tool results are left out; they are internal steps, not part of the conversation.
    """
    lines = []
    for message in messages:
        if isinstance(message, HumanMessage):
            lines.append(f"Customer: {message.content}")
        elif isinstance(message, AIMessage) and not message.tool_calls and message.content:
            lines.append(f"Assistant: {message.content}")
    return "\n".join(lines[-max_messages:])  # "" on the first turn
