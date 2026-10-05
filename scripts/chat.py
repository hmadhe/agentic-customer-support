"""Command-line chat with the support graph. Type 'quit' to exit."""

from app.graph import build_graph


def main() -> None:
    graph = build_graph()
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

        result = graph.invoke({"message": message})
        c = result["classification"]
        print(f"  [intent={c.intent} sentiment={c.sentiment} order_id={c.order_id}]")
        print(f"Bot: {result['response']}")
        if result.get("policy_answer"):
            print(f"  [sources={result['policy_answer'].sources} answered={result['policy_answer'].answered}]")
        for tool_call in [call for m in result.get("messages", []) for call in getattr(m, "tool_calls", [])]:
            print(f"  [tool: {tool_call['name']}({tool_call['args']})]")


if __name__ == "__main__":
    main()
