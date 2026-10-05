"""Command-line chat with the support graph. The conversation is remembered until you quit."""

import uuid

from app.agent import current_turn
from app.graph import build_graph
from app.memory import make_checkpointer


def main() -> None:
    graph = build_graph(checkpointer=make_checkpointer())
    # One thread per chat session: every message in this session continues the same conversation.
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    print("VoltCart support (type 'quit' to exit)")

    while True:
        try:
            message = input("\nYou: ").strip()
        except EOFError:
            break
        if not message:
            continue
        if message.lower() in {"quit", "exit"}:
            break

        result = graph.invoke({"message": message}, config)
        c = result["classification"]
        print(f"  [intent={c.intent} sentiment={c.sentiment} order_id={c.order_id}]")
        print(f"Bot: {result['response']}")
        if result.get("policy_answer"):
            print(f"  [sources={result['policy_answer'].sources} answered={result['policy_answer'].answered}]")
        for tool_call in [call for m in current_turn(result["messages"]) for call in getattr(m, "tool_calls", [])]:
            print(f"  [tool: {tool_call['name']}({tool_call['args']})]")
        if result.get("escalation_reason"):
            print(f"  [escalated: {result['escalation_reason']}]")


if __name__ == "__main__":
    main()
