import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langgraph.checkpoint.base import BaseCheckpointSaver
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
    customer_text,
    invented_order_ids,
    latest_tools_failed,
    tool_rounds,
)
from app.answer import AnswerGenerationError, answer_question, build_answer_chain
from app.classifier import build_classifier
from app.memory import TICKET_HISTORY_MESSAGES, format_history, recent_messages
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
from app.vector_store import get_vector_store, require_ingested

Retriever = Callable[[str], list[RetrievedChunk]]
Answerer = Callable[[str, list[RetrievedChunk]], PolicyAnswer]


class SupportState(BaseModel):
    """Data passed between graph nodes. Each node returns only the fields it changes.

    Two lifetimes: `messages` is the whole conversation and is kept between turns (with a checkpointer).
    Every other field belongs to the current turn and is reset when a new message arrives.
    """

    # The conversation: customer messages, the agent's tool calls and results, and every reply.
    # add_messages appends instead of replacing.
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)

    # --- Current turn only ---
    message: str
    classification: IntentClassification | None = None
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    policy_answer: PolicyAnswer | None = None
    # Set by the node that decides a human is needed; read by the escalate node.
    escalation_reason: EscalationReason | None = None
    ticket: Ticket | None = None
    response: str | None = None


# Returned by the first node of every turn, so nothing from the previous turn leaks into this one.
NEW_TURN = {"chunks": [], "policy_answer": None, "escalation_reason": None, "ticket": None, "response": None}


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
    checkpointer: BaseCheckpointSaver | None = None,
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
        require_ingested(vector_store)  # like the order database: stop now if setup was skipped
        retriever = lambda question: retrieve(question, vector_store=vector_store)  # noqa: E731
    if answerer is None:
        answer_chain = build_answer_chain()
        answerer = lambda question, chunks: answer_question(question, chunks, chain=answer_chain)  # noqa: E731
    orders.require_db(db_path)  # a missing database is a setup mistake: stop now, not mid-conversation
    tools = build_tools(retriever, db_path)
    agent = agent or build_agent(tools)

    def classify_intent(state: SupportState) -> dict:
        # state.messages holds the earlier turns; this turn's message is added below.
        classification = classifier.invoke({"message": state.message, "history": format_history(state.messages)})
        return {**NEW_TURN, "messages": [HumanMessage(state.message)], "classification": classification}

    def retrieve_policies(state: SupportState) -> dict:
        return {"chunks": retriever(state.message)}

    def answer(state: SupportState) -> dict:
        try:
            policy_answer = answerer(state.message, state.chunks)
        except AnswerGenerationError:
            return {"escalation_reason": EscalationReason.MODEL_ERROR}
        if not policy_answer.answered:
            return {"policy_answer": policy_answer, "escalation_reason": EscalationReason.POLICY_NOT_FOUND}
        return {
            "policy_answer": policy_answer,
            "response": policy_answer.answer,
            "messages": [AIMessage(policy_answer.answer)],
        }

    def run_agent(state: SupportState) -> dict:
        if latest_tools_failed(state.messages):
            return {"escalation_reason": EscalationReason.TOOL_ERROR}

        # The instructions are added for each call rather than stored; the history is trimmed to fit the context.
        reply = agent.invoke([SystemMessage(AGENT_PROMPT), *recent_messages(state.messages)])

        if invented_order_ids(reply.tool_calls, customer_text(state.messages)):
            # Don't run a lookup for an order number the customer never gave; ask for it instead.
            reply = AIMessage(ASK_FOR_ORDER_ID)
        elif reply.tool_calls and tool_rounds(state.messages) >= MAX_TOOL_ROUNDS:
            # Out of tool rounds. The unanswered tool-call request is not stored: a history with a tool call
            # but no tool result breaks the next turn of the conversation.
            return {"escalation_reason": EscalationReason.AGENT_GAVE_UP}

        update = {"messages": [reply]}
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
            ticket = create_ticket(
                reason, state.message, c.intent, c.sentiment, c.order_id,
                conversation=format_history(state.messages, max_messages=TICKET_HISTORY_MESSAGES),
                db_path=tickets_db_path,
            )
        except sqlite3.Error:
            return {"escalation_reason": reason, "response": TICKET_FAILED_REPLY, "messages": [AIMessage(TICKET_FAILED_REPLY)]}
        reply = ESCALATION_REPLIES[reason].format(ticket_id=ticket.ticket_id)
        return {"escalation_reason": reason, "ticket": ticket, "response": reply, "messages": [AIMessage(reply)]}

    def respond(state: SupportState) -> dict:
        reply = RESPONSES[state.classification.intent]
        return {"response": reply, "messages": [AIMessage(reply)]}

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
    # With a checkpointer, each thread_id's state is saved after every turn and loaded on the next one.
    return graph.compile(checkpointer=checkpointer)
