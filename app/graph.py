from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from app.classifier import build_classifier
from app.schemas import Intent, IntentClassification


class SupportState(BaseModel):
    """Data passed between graph nodes. Each node returns only the fields it changes."""

    message: str
    classification: IntentClassification | None = None
    response: str | None = None


# Placeholder replies until later milestones add RAG, tools and escalation.
RESPONSES = {
    Intent.POLICY_QUESTION: "Good question about our policies. I'll look that up for you.",
    Intent.ORDER_ISSUE: "I can help with your order. Let me check its details.",
    Intent.HUMAN_REQUEST: "I'll connect you with a member of our support team.",
    Intent.GREETING: "Hi! Welcome to VoltCart support. How can I help you today?",
    Intent.OUT_OF_SCOPE: "Sorry, I can only help with VoltCart orders and store policies.",
}


def build_graph(classifier: Runnable | None = None):
    """Build the support workflow: START -> classify_intent -> respond -> END.

    `classifier` can be swapped for a fake in tests so no LLM is called.
    """
    classifier = classifier or build_classifier()

    def classify_intent(state: SupportState) -> dict:
        return {"classification": classifier.invoke({"message": state.message})}

    def respond(state: SupportState) -> dict:
        return {"response": RESPONSES[state.classification.intent]}

    graph = StateGraph(SupportState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("respond", respond)
    graph.add_edge(START, "classify_intent")
    graph.add_edge("classify_intent", "respond")
    graph.add_edge("respond", END)
    return graph.compile()
