"""HTTP API for the support assistant. Run with: uvicorn app.api:app"""

import threading
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from app.agent import current_turn
from app.graph import build_graph
from app.memory import CONVERSATIONS_DB_PATH, make_checkpointer
from app.schemas import Intent, Sentiment
from app.tickets import TICKETS_DB_PATH, EscalationReason, Ticket, list_tickets


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000, examples=["Can I return the laptop from order 1001?"])
    thread_id: str | None = Field(
        default=None, max_length=100, description="Continue a conversation. Leave empty to start a new one."
    )


class ChatResponse(BaseModel):
    thread_id: str
    reply: str
    intent: Intent
    sentiment: Sentiment
    order_id: str | None
    sources: list[str] = Field(description="Policy documents the answer is based on (policy questions).")
    tools_used: list[str] = Field(description="Tools the order agent called in this turn.")
    escalated: bool
    escalation_reason: EscalationReason | None
    ticket_id: int | None


def default_graph():
    # Conversations are saved in SQLite so they survive a server restart.
    return build_graph(checkpointer=make_checkpointer(CONVERSATIONS_DB_PATH))


def create_app(graph_factory: Callable = default_graph, tickets_db_path: Path = TICKETS_DB_PATH) -> FastAPI:
    """Build the API. Tests pass a graph made of fakes and a temporary tickets database."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Built once at startup. Fails here, before serving any request, if a database or the vector store is missing.
        app.state.graph = graph_factory()
        yield

    app = FastAPI(title="VoltCart Support Assistant", lifespan=lifespan)

    # One lock per conversation. Two simultaneous messages on the same thread_id both loaded the same saved state,
    # and the second save overwrote the first, so a customer message was silently lost. Different conversations
    # still run in parallel. (This protects one server process only.)
    conversation_locks: dict[str, threading.Lock] = {}
    locks_guard = threading.Lock()

    def conversation_lock(thread_id: str) -> threading.Lock:
        with locks_guard:
            return conversation_locks.setdefault(thread_id, threading.Lock())

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    # A plain `def`, not `async def`: the graph makes blocking calls (Ollama, SQLite). FastAPI runs `def`
    # endpoints on a thread pool. As `async def`, one 3 s chat blocked every other request, /health included.
    @app.post("/chat", response_model=ChatResponse)
    def chat(body: ChatRequest, request: Request) -> ChatResponse:
        thread_id = body.thread_id or str(uuid.uuid4())
        try:
            with conversation_lock(thread_id):
                result = request.app.state.graph.invoke(
                    {"message": body.message}, {"configurable": {"thread_id": thread_id}}
                )
        except (httpx.TransportError, ConnectionError) as error:
            # Ollama is down or unreachable. 503 tells the client to retry later, unlike a bare 500.
            raise HTTPException(status_code=503, detail="The language model is unavailable. Please try again later.") from error

        c = result["classification"]
        policy_answer = result.get("policy_answer")
        ticket = result.get("ticket")
        return ChatResponse(
            thread_id=thread_id,
            reply=result["response"],
            intent=c.intent,
            sentiment=c.sentiment,
            order_id=c.order_id,
            sources=policy_answer.sources if policy_answer else [],
            tools_used=[call["name"] for m in current_turn(result["messages"]) for call in getattr(m, "tool_calls", [])],
            escalated=result.get("escalation_reason") is not None,
            escalation_reason=result.get("escalation_reason"),
            ticket_id=ticket.ticket_id if ticket else None,
        )

    @app.get("/tickets", response_model=list[Ticket])
    def tickets() -> list[Ticket]:
        return list_tickets(tickets_db_path)

    @app.get("/tickets/{ticket_id}", response_model=Ticket)
    def ticket(ticket_id: int) -> Ticket:
        for t in list_tickets(tickets_db_path):
            if t.ticket_id == ticket_id:
                return t
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found.")

    return app


app = create_app()
