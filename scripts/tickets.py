"""List the support tickets created by escalations: python -m scripts.tickets"""

from app.tickets import TICKETS_DB_PATH, list_tickets


def main() -> None:
    tickets = list_tickets()
    if not tickets:
        print(f"No tickets yet ({TICKETS_DB_PATH}).")
    for t in tickets:
        order = f" order={t.order_id}" if t.order_id else ""
        print(f"#{t.ticket_id} {t.created_at} [{t.reason}] intent={t.intent} sentiment={t.sentiment}{order}\n    {t.customer_message!r}")
        if t.conversation:
            print("    conversation:\n" + "\n".join(f"      {line}" for line in t.conversation.splitlines()))


if __name__ == "__main__":
    main()
