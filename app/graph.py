from collections.abc import Callable
from pathlib import Path
from typing import Annotated

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from app import orders
from app.agent import (
    AGENT_PROMPT,
    ASK_FOR_ORDER_ID,
    FALLBACK_REPLY,
    MAX_TOOL_ROUNDS,
    build_agent,
    invented_order_ids,
)
from app.answer import answer_question, build_answer_chain
from app.classifier import build_classifier
from app.retriever import retrieve
from app.schemas import Intent, IntentClassification, PolicyAnswer, RetrievedChunk
from app.tools import build_tools
from app.vector_store import get_vector_store

Retriever = Callable[[str], list[RetrievedChunk]]
Answerer = Callable[[str, list[RetrievedChunk]], PolicyAnswer]


class SupportState(BaseModel):
    """Data passed between graph nodes. Each node returns only the fields it changes."""

    message: str
    classification: IntentClassification | None = None
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    policy_answer: PolicyAnswer | None = None
    # The order agent's conversation with its tools. add_messages appends instead of replacing.
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    response: str | None = None


# Placeholder replies for intents that don't have a real branch yet (escalation comes in Milestone 5).
RESPONSES = {
    Intent.HUMAN_REQUEST: "I'll connect you with a member of our support team.",
    Intent.GREETING: "Hi! Welcome to VoltCart support. How can I help you today?",
    Intent.OUT_OF_SCOPE: "Sorry, I can only help with VoltCart orders and store policies.",
}


def route_by_intent(state: SupportState) -> str:
    intent = state.classification.intent
    if intent == Intent.POLICY_QUESTION:
        return "retrieve"
    if intent == Intent.ORDER_ISSUE:
        return "agent"
    return "respond"


def route_after_agent(state: SupportState) -> str:
    """Run the requested tools, stop if the agent answered, or give up after MAX_TOOL_ROUNDS."""
    if not state.messages[-1].tool_calls:
        return END
    tool_rounds = sum(1 for message in state.messages if isinstance(message, AIMessage) and message.tool_calls)
    return "tools" if tool_rounds <= MAX_TOOL_ROUNDS else "fallback"


def build_graph(
    classifier: Runnable | None = None,
    retriever: Retriever | None = None,
    answerer: Answerer | None = None,
    agent: Runnable | None = None,
    db_path: Path = orders.DB_PATH,
):
    """Build the support workflow.

    START -> classify_intent -> policy_question -> retrieve -> answer -> END
                             -> order_issue     -> agent <-> tools -> END (fallback after MAX_TOOL_ROUNDS)
                             -> anything else   -> respond -> END

    Any dependency can be swapped for a fake in tests. The real ones are created once here, not inside the nodes.
    """
    classifier = classifier or build_classifier()
    if retriever is None:
        vector_store = get_vector_store()
        retriever = lambda question: retrieve(question, vector_store=vector_store)  # noqa: E731
    if answerer is None:
        answer_chain = build_answer_chain()
        answerer = lambda question, chunks: answer_question(question, chunks, chain=answer_chain)  # noqa: E731
    orders.require_db(db_path)  # a missing database is a setup mistake: stop now, not mid-conversation
    tools = build_tools(retriever, db_path)
    agent = agent or build_agent(tools)

    def classify_intent(state: SupportState) -> dict:
        return {"classification": classifier.invoke({"message": state.message})}

    def retrieve_policies(state: SupportState) -> dict:
        return {"chunks": retriever(state.message)}

    def answer(state: SupportState) -> dict:
        policy_answer = answerer(state.message, state.chunks)
        return {"policy_answer": policy_answer, "response": policy_answer.answer}

    def run_agent(state: SupportState) -> dict:
        # On the first turn, start the agent's conversation with its instructions and the customer's message.
        new_messages = [] if state.messages else [SystemMessage(AGENT_PROMPT), HumanMessage(state.message)]
        reply = agent.invoke(state.messages + new_messages)
        if invented_order_ids(reply.tool_calls, state.message):
            # Don't run a lookup for an order number the customer never gave; ask for it instead.
            reply = AIMessage(ASK_FOR_ORDER_ID)
        update = {"messages": new_messages + [reply]}
        if not reply.tool_calls:
            update["response"] = reply.content
        return update

    def fallback(state: SupportState) -> dict:
        return {"response": FALLBACK_REPLY}

    def respond(state: SupportState) -> dict:
        return {"response": RESPONSES[state.classification.intent]}

    graph = StateGraph(SupportState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve", retrieve_policies)
    graph.add_node("answer", answer)
    graph.add_node("agent", run_agent)
    # handle_tool_errors: an unexpected tool failure goes back to the agent as a message instead of crashing the graph.
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    graph.add_node("fallback", fallback)
    graph.add_node("respond", respond)

    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges("classify_intent", route_by_intent, ["retrieve", "agent", "respond"])
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)
    graph.add_conditional_edges("agent", route_after_agent, ["tools", "fallback", END])
    graph.add_edge("tools", "agent")
    graph.add_edge("fallback", END)
    graph.add_edge("respond", END)
    return graph.compile()
