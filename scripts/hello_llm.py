"""Milestone 0 smoke test: make one real call to the local LLM."""

import time

from app.llm import get_llm


def main() -> None:
    llm = get_llm()
    print(f"Model: {llm.model} @ {llm.base_url}")

    start = time.perf_counter()
    response = llm.invoke(
        "You are a support assistant for VoltCart, an online electronics store. "
        "Greet a customer in one short sentence."
    )
    elapsed = time.perf_counter() - start

    print(f"Response: {response.content}")
    print(f"Latency: {elapsed:.1f}s")
    print(f"Tokens: {response.usage_metadata}")


if __name__ == "__main__":
    main()
