from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from app.llm import get_llm
from app.schemas import PolicyAnswer, RetrievedChunk

SYSTEM_PROMPT = """You are VoltCart's customer support assistant. Answer the customer's question using ONLY the policy excerpts below.

Rules:
- Use only facts stated in the excerpts. Never add numbers, fees, time limits or rules that are not written in them.
- If the excerpts do not answer the question, set answered to false. Do not guess and do not use general knowledge.
- In sources, list the file names of the excerpts your answer is based on.
- Keep the answer to 1-3 sentences.

Policy excerpts:
{context}"""

INSUFFICIENT_ANSWER = (
    "I'm sorry, I don't have VoltCart policy information that answers this question. "
    "A member of our support team can help."
)


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render chunks for the prompt, each labelled with its source file."""
    return "\n\n".join(f"[source: {chunk.source}]\n{chunk.text}" for chunk in chunks)


def build_answer_chain(llm: BaseChatModel | None = None) -> Runnable:
    """Prompt + LLM that returns a PolicyAnswer. Input: {"question": str, "context": str}."""
    llm = llm or get_llm()
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{question}")])
    return prompt | llm.with_structured_output(PolicyAnswer)


def answer_question(question: str, chunks: list[RetrievedChunk], chain: Runnable | None = None) -> PolicyAnswer:
    """Answer from the retrieved chunks only, or say the policy information is insufficient."""
    if not chunks:
        return PolicyAnswer(answered=False, answer=INSUFFICIENT_ANSWER, sources=[])

    chain = chain or build_answer_chain()
    try:
        result = chain.invoke({"question": question, "context": format_context(chunks)})
    except OutputParserException:
        # The output was not a valid PolicyAnswer, e.g. a runaway answer cut off at max_output_tokens.
        # Showing nothing is safer than showing a broken or half-written answer.
        return PolicyAnswer(answered=False, answer=INSUFFICIENT_ANSWER, sources=[])

    if not result.answered:
        return PolicyAnswer(answered=False, answer=INSUFFICIENT_ANSWER, sources=[])

    # Keep only sources that were actually retrieved, so the model cannot cite a document it never saw.
    retrieved_sources = {chunk.source for chunk in chunks}
    sources = [source for source in dict.fromkeys(result.sources) if source in retrieved_sources]
    return PolicyAnswer(answered=True, answer=result.answer, sources=sources)
