import sqlite3

from app.tickets import EscalationReason, create_ticket, list_tickets


def test_created_ticket_can_be_listed(tmp_path):
    db = tmp_path / "tickets.db"

    ticket = create_ticket(
        EscalationReason.CUSTOMER_REQUEST, "I want a person", intent="human_request", sentiment="negative", db_path=db
    )

    assert ticket.ticket_id == 1
    assert list_tickets(db) == [ticket]


def test_ticket_ids_increase(tmp_path):
    db = tmp_path / "tickets.db"

    first = create_ticket(EscalationReason.POLICY_NOT_FOUND, "Price matching?", db_path=db)
    second = create_ticket(EscalationReason.TOOL_ERROR, "Where is order 1042?", order_id="1042", db_path=db)

    assert (first.ticket_id, second.ticket_id) == (1, 2)
    assert [t.reason for t in list_tickets(db)] == [EscalationReason.POLICY_NOT_FOUND, EscalationReason.TOOL_ERROR]


def test_listing_an_empty_database_returns_no_tickets(tmp_path):
    assert list_tickets(tmp_path / "tickets.db") == []


def test_ticket_stores_the_conversation(tmp_path):
    db = tmp_path / "tickets.db"

    create_ticket(EscalationReason.ANGRY_CUSTOMER, "NOW!", conversation="Customer: Where is 1042?\nCustomer: NOW!", db_path=db)

    assert list_tickets(db)[0].conversation == "Customer: Where is 1042?\nCustomer: NOW!"


def test_database_from_before_milestone_6_is_migrated(tmp_path):
    db = tmp_path / "tickets.db"
    with sqlite3.connect(db) as connection:  # the Milestone 5 layout, without the conversation column
        connection.execute(
            """CREATE TABLE tickets (ticket_id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, reason TEXT,
               customer_message TEXT, intent TEXT, sentiment TEXT, order_id TEXT)"""
        )
        connection.execute(
            "INSERT INTO tickets (created_at, reason, customer_message) VALUES ('2026-10-05T17:29:32', 'angry_customer', 'old')"
        )

    new = create_ticket(EscalationReason.CUSTOMER_REQUEST, "new", conversation="Customer: new", db_path=db)

    old, saved = list_tickets(db)
    assert (old.customer_message, old.conversation) == ("old", None)
    assert saved == new
