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
