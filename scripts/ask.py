"""Ask the policy RAG pipeline a question: python -m scripts.ask "How long do refunds take?" """

import sys

from app.answer import answer_question
from app.retriever import retrieve


def main() -> None:
    for question in sys.argv[1:] or [input("Question: ")]:
        chunks = retrieve(question)
        print(f"\nQ: {question}")
        for chunk in chunks:
            print(f"   retrieved {chunk.score:.3f}  {chunk.source} > {chunk.section}")
        result = answer_question(question, chunks)
        print(f"A: {result.answer}")
        print(f"   answered={result.answered} sources={result.sources}")


if __name__ == "__main__":
    main()
