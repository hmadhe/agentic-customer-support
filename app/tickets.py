import sqlite3
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel

from app.config import PROJECT_ROOT

TICKETS_DB_PATH = PROJECT_ROOT / "data" / "tickets.db"


class EscalationReason(StrEnum):
    CUSTOMER_REQUEST = "customer_request"  # the customer asked for a human
    ANGRY_CUSTOMER = "angry_customer"  # classified as angry: the bot answered only 1 of 5 angry messages well
    POLICY_NOT_FOUND = "policy_not_found"  # the policies don't answer the question
    MODEL_ERROR = "model_error"  # the LLM's output could not be used
    AGENT_GAVE_UP = "agent_gave_up"  # the order agent hit its tool-round limit
    TOOL_ERROR = "tool_error"  # a tool crashed, e.g. the database or Ollama was unavailable


_HANDOFF = "so I've passed it to our support team (ticket #{ticket_id}). A team member will contact you soon."
ESCALATION_REPLIES = {
    EscalationReason.CUSTOMER_REQUEST: "I've passed your request to our support team (ticket #{ticket_id}). A team member will contact you soon.",
    EscalationReason.ANGRY_CUSTOMER: "I'm sorry about your experience. I've passed this to our support team (ticket #{ticket_id}) so a team member can help you personally.",
    EscalationReason.POLICY_NOT_FOUND: "I don't have VoltCart policy information that answers this, " + _HANDOFF,
    EscalationReason.MODEL_ERROR: "Sorry, I couldn't complete that myself, " + _HANDOFF,
    EscalationReason.AGENT_GAVE_UP: "Sorry, I couldn't complete that myself, " + _HANDOFF,
    EscalationReason.TOOL_ERROR: "Sorry, I couldn't complete that myself, " + _HANDOFF,
}
# Never promise a human when no ticket was saved.
TICKET_FAILED_REPLY = (
    "Sorry, something went wrong on our side and I couldn't pass this to our support team. "
    "Please try again in a few minutes."
)


class Ticket(BaseModel):
    ticket_id: int
    created_at: datetime
    reason: EscalationReason
    customer_message: str
    intent: str | None
    sentiment: str | None
    order_id: str | None


def _connect(db_path: Path) -> sqlite3.Connection:
    # Unlike the order database, tickets are written by the app, so the table is created on first use.
    connection = sqlite3.connect(db_path)
    connection.execute(
        """CREATE TABLE IF NOT EXISTS tickets (
            ticket_id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, reason TEXT,
            customer_message TEXT, intent TEXT, sentiment TEXT, order_id TEXT)"""
    )
    return connection


def create_ticket(
    reason: EscalationReason,
    customer_message: str,
    intent: str | None = None,
    sentiment: str | None = None,
    order_id: str | None = None,
    db_path: Path = TICKETS_DB_PATH,
) -> Ticket:
    created_at = datetime.now().replace(microsecond=0)
    with _connect(db_path) as connection:
        cursor = connection.execute(
            "INSERT INTO tickets (created_at, reason, customer_message, intent, sentiment, order_id) VALUES (?, ?, ?, ?, ?, ?)",
            (created_at.isoformat(), reason, customer_message, intent, sentiment, order_id),
        )
    return Ticket(
        ticket_id=cursor.lastrowid, created_at=created_at, reason=reason, customer_message=customer_message,
        intent=intent, sentiment=sentiment, order_id=order_id,
    )


def list_tickets(db_path: Path = TICKETS_DB_PATH) -> list[Ticket]:
    with _connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute("SELECT * FROM tickets ORDER BY ticket_id").fetchall()
    return [Ticket(**dict(row)) for row in rows]
