import sqlite3
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
    MAX_TOOL_ROUNDS,
    build_agent,
    invented_order_ids,
    latest_tools_failed,
    tool_rounds,
)
from app.answer import AnswerGenerationError, answer_question, build_answer_chain
from app.classifier import build_classifier
from app.retriever import retrieve
from app.schemas import Intent, IntentClassification, PolicyAnswer, RetrievedChunk, Sentiment
from app.tickets import (
    ESCALATION_REPLIES,
    TICKET_FAILED_REPLY,
    TICKETS_DB_PATH,
    EscalationReason,
    Ticket,
    create_ticket,
)
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
    # Set by the node that decides a human is needed; read by the escalate node.
    escalation_reason: EscalationReason | None = None
    ticket: Ticket | None = None
    response: str | None = None


# Replies for intents that need neither RAG, tools nor a human.
RESPONSES = {
    Intent.GREETING: "Hi! Welcome to VoltCart support. How can I help you today?",
    Intent.OUT_OF_SCOPE: "Sorry, I can only help with VoltCart orders and store policies.",
}


def needs_human_now(classification: IntentClassification) -> bool:
    """Escalate before trying to help: the customer asked for a person, or is angry.

    Only "angry", not "negative": mildly unhappy customers ("a little disappointed... can I return it?")
    are classified negative, and the bot can answer those (see the Milestone 5 log).
    """
    return classification.intent == Intent.HUMAN_REQUEST or classification.sentiment == Sentiment.ANGRY


def route_by_intent(state: SupportState) -> str:
    if needs_human_now(state.classification):
        return "escalate"
    intent = state.classification.intent
    if intent == Intent.POLICY_QUESTION:
        return "retrieve"
    if intent == Intent.ORDER_ISSUE:
        return "agent"
    return "respond"


def route_after_answer(state: SupportState) -> str:
    return "escalate" if state.escalation_reason else END


def route_after_agent(state: SupportState) -> str:
    if state.escalation_reason:
        return "escalate"
    return "tools" if state.messages[-1].tool_calls else END


def build_graph(
    classifier: Runnable | None = None,
    retriever: Retriever | None = None,
    answerer: Answerer | None = None,
    agent: Runnable | None = None,
    db_path: Path = orders.DB_PATH,
    tickets_db_path: Path = TICKETS_DB_PATH,
):
    """Build the support workflow.

    START -> classify_intent -> policy_question -> retrieve -> answer -> END, or escalate if not answered
                             -> order_issue     -> agent <-> tools -> END, or escalate on tool error / too many rounds
                             -> human_request   -> escalate -> END
                             -> greeting, out_of_scope -> respond -> END

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
        try:
            policy_answer = answerer(state.message, state.chunks)
        except AnswerGenerationError:
            return {"escalation_reason": EscalationReason.MODEL_ERROR}
        if not policy_answer.answered:
            return {"policy_answer": policy_answer, "escalation_reason": EscalationReason.POLICY_NOT_FOUND}
        return {"policy_answer": policy_answer, "response": policy_answer.answer}

    def run_agent(state: SupportState) -> dict:
        if latest_tools_failed(state.messages):
            return {"escalation_reason": EscalationReason.TOOL_ERROR}

        # On the first turn, start the agent's conversation with its instructions and the customer's message.
        new_messages = [] if state.messages else [SystemMessage(AGENT_PROMPT), HumanMessage(state.message)]
        reply = agent.invoke(state.messages + new_messages)

        if invented_order_ids(reply.tool_calls, state.message):
            # Don't run a lookup for an order number the customer never gave; ask for it instead.
            reply = AIMessage(ASK_FOR_ORDER_ID)
        elif reply.tool_calls and tool_rounds(state.messages) >= MAX_TOOL_ROUNDS:
            # Out of tool rounds. The unanswered tool-call request is not stored: a history with a tool call
            # but no tool result would break the next turn once conversations are remembered.
            return {"messages": new_messages, "escalation_reason": EscalationReason.AGENT_GAVE_UP}

        update = {"messages": new_messages + [reply]}
        if not reply.tool_calls:
            update["response"] = reply.content
        return update

    def escalate(state: SupportState) -> dict:
        c = state.classification
        # Escalations straight from classification have no reason set yet; later nodes set their own.
        if state.escalation_reason:
            reason = state.escalation_reason
        elif c.intent == Intent.HUMAN_REQUEST:
            reason = EscalationReason.CUSTOMER_REQUEST
        else:
            reason = EscalationReason.ANGRY_CUSTOMER
        try:
            ticket = create_ticket(reason, state.message, c.intent, c.sentiment, c.order_id, tickets_db_path)
        except sqlite3.Error:
            return {"escalation_reason": reason, "response": TICKET_FAILED_REPLY}
        reply = ESCALATION_REPLIES[reason].format(ticket_id=ticket.ticket_id)
        return {"escalation_reason": reason, "ticket": ticket, "response": reply}

    def respond(state: SupportState) -> dict:
        return {"response": RESPONSES[state.classification.intent]}

    graph = StateGraph(SupportState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve", retrieve_policies)
    graph.add_node("answer", answer)
    graph.add_node("agent", run_agent)
    # handle_tool_errors: a failing tool produces an error ToolMessage instead of crashing the graph;
    # the agent node then escalates.
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    graph.add_node("escalate", escalate)
    graph.add_node("respond", respond)

    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges("classify_intent", route_by_intent, ["retrieve", "agent", "escalate", "respond"])
    graph.add_edge("retrieve", "answer")
    graph.add_conditional_edges("answer", route_after_answer, ["escalate", END])
    graph.add_conditional_edges("agent", route_after_agent, ["tools", "escalate", END])
    graph.add_edge("tools", "agent")
    graph.add_edge("escalate", END)
    graph.add_edge("respond", END)
    return graph.compile()
