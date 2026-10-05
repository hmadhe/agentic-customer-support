from collections.abc import Callable

from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.answer import answer_question, build_answer_chain
from app.classifier import build_classifier
from app.retriever import retrieve
from app.schemas import Intent, IntentClassification, PolicyAnswer, RetrievedChunk
from app.vector_store import get_vector_store

Retriever = Callable[[str], list[RetrievedChunk]]
Answerer = Callable[[str, list[RetrievedChunk]], PolicyAnswer]


class SupportState(BaseModel):
    """Data passed between graph nodes. Each node returns only the fields it changes."""

    message: str
    classification: IntentClassification | None = None
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    policy_answer: PolicyAnswer | None = None
    response: str | None = None


# Placeholder replies for intents that don't have a real branch yet (tools and escalation come later).
RESPONSES = {
    Intent.ORDER_ISSUE: "I can help with your order. Let me check its details.",
    Intent.HUMAN_REQUEST: "I'll connect you with a member of our support team.",
    Intent.GREETING: "Hi! Welcome to VoltCart support. How can I help you today?",
    Intent.OUT_OF_SCOPE: "Sorry, I can only help with VoltCart orders and store policies.",
}


def route_by_intent(state: SupportState) -> str:
    """Policy questions go to RAG; everything else gets a placeholder reply for now."""
    return "retrieve" if state.classification.intent == Intent.POLICY_QUESTION else "respond"


def build_graph(
    classifier: Runnable | None = None,
    retriever: Retriever | None = None,
    answerer: Answerer | None = None,
):
    """Build the support workflow.

    START -> classify_intent -> (policy_question) -> retrieve -> answer -> END
                             -> (anything else)   -> respond ----------> END

    Any dependency can be swapped for a fake in tests. The real ones are created once here,
    not inside the nodes, so each message doesn't open a new vector store or rebuild the chain.
    """
    classifier = classifier or build_classifier()
    if retriever is None:
        vector_store = get_vector_store()
        retriever = lambda question: retrieve(question, vector_store=vector_store)  # noqa: E731
    if answerer is None:
        answer_chain = build_answer_chain()
        answerer = lambda question, chunks: answer_question(question, chunks, chain=answer_chain)  # noqa: E731

    def classify_intent(state: SupportState) -> dict:
        return {"classification": classifier.invoke({"message": state.message})}

    def retrieve_policies(state: SupportState) -> dict:
        return {"chunks": retriever(state.message)}

    def answer(state: SupportState) -> dict:
        policy_answer = answerer(state.message, state.chunks)
        return {"policy_answer": policy_answer, "response": policy_answer.answer}

    def respond(state: SupportState) -> dict:
        return {"response": RESPONSES[state.classification.intent]}

    graph = StateGraph(SupportState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve", retrieve_policies)
    graph.add_node("answer", answer)
    graph.add_node("respond", respond)

    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges("classify_intent", route_by_intent, ["retrieve", "respond"])
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)
    graph.add_edge("respond", END)
    return graph.compile()
