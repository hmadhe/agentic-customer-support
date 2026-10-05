from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnableLambda

from app.answer import INSUFFICIENT_ANSWER, answer_question, format_context
from app.schemas import PolicyAnswer, RetrievedChunk

CHUNKS = [
    RetrievedChunk(text="Returns > Window\n\nReturn within 30 days.", source="returns.md", section="Window", score=0.8),
    RetrievedChunk(text="Payments > Gift cards\n\nGift cards never expire.", source="payments.md", section="Gift cards", score=0.6),
]


def fake_chain(result: PolicyAnswer) -> RunnableLambda:
    """Stands in for the LLM answer chain and always returns the given result."""
    return RunnableLambda(lambda _: result)


def test_format_context_labels_each_chunk_with_its_source():
    context = format_context(CHUNKS)

    assert context == (
        "[source: returns.md]\nReturns > Window\n\nReturn within 30 days.\n\n"
        "[source: payments.md]\nPayments > Gift cards\n\nGift cards never expire."
    )


def test_no_chunks_means_insufficient_without_calling_the_model():
    def must_not_be_called(_):
        raise AssertionError("the model should not be called when nothing was retrieved")

    result = answer_question("Anything?", [], chain=RunnableLambda(must_not_be_called))

    assert result == PolicyAnswer(answered=False, answer=INSUFFICIENT_ANSWER, sources=[])


def test_unanswered_result_is_replaced_by_the_fixed_message():
    # Real example from qwen: it marked this unanswered but still claimed a policy in its text.
    model_result = PolicyAnswer(answered=False, answer="Price matching is not offered by VoltCart.", sources=["payments.md"])

    result = answer_question("Do you offer price matching?", CHUNKS, chain=fake_chain(model_result))

    assert result == PolicyAnswer(answered=False, answer=INSUFFICIENT_ANSWER, sources=[])


def test_sources_that_were_not_retrieved_are_removed():
    model_result = PolicyAnswer(answered=True, answer="30 days.", sources=["returns.md", "made_up.md", "returns.md"])

    result = answer_question("How long can I return?", CHUNKS, chain=fake_chain(model_result))

    assert result.sources == ["returns.md"]
    assert result.answer == "30 days."


def test_unparseable_model_output_is_treated_as_unanswered():
    def runaway(_):
        raise OutputParserException("Failed to parse PolicyAnswer from completion")

    result = answer_question("How long can I return?", CHUNKS, chain=RunnableLambda(runaway))

    assert result == PolicyAnswer(answered=False, answer=INSUFFICIENT_ANSWER, sources=[])
