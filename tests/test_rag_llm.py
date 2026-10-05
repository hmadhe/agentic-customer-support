"""Real-model RAG tests: nomic-embed-text retrieval and qwen answers. Slow and needs Ollama: `pytest -m llm`."""

import pytest

from app.answer import INSUFFICIENT_ANSWER, answer_question
from app.ingest import ingest
from app.retriever import retrieve
from app.vector_store import get_vector_store

pytestmark = pytest.mark.llm

# (question, document that should rank first, section that must be in the top results)
RETRIEVAL_CASES = [
    ("What is the restocking fee for a laptop?", "returns.md", "Restocking fee"),
    ("Can I return opened earbuds?", "returns.md", "Items that cannot be returned"),
    ("How long does a refund take?", "returns.md", "Refunds"),
    ("My laptop arrived with a cracked screen", "returns.md", "Damaged or wrong items"),
    ("How much is express shipping?", "shipping.md", "Shipping options and costs"),
    ("Do you ship to the UK?", "shipping.md", "Where we ship"),
    ("Does the protection plan cover water damage?", "warranty.md", "VoltCart Protection Plan"),
    ("Can I pay in monthly installments?", "payments.md", "Paying in installments"),
    ("Do gift cards expire?", "payments.md", "Gift cards"),
    ("How do I reset my password?", "account.md", "Passwords and sign-in"),
    ("Can I change the delivery address after ordering?", "account.md", "Changing your details"),
]

# (question, text the answer must contain, source it must cite)
ANSWER_CASES = [
    ("How much is express shipping?", "14.99", "shipping.md"),
    ("What is the restocking fee for a laptop?", "15%", "returns.md"),
    ("Can I pay in monthly installments?", "150", "payments.md"),
]

# None of the policy documents answer these.
UNANSWERABLE = ["Do you offer price matching?", "Is there a student discount?", "Can I pick up my order in a store?"]


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    """A fresh vector store built from the real policy documents, independent of the local chroma_db/."""
    vector_store = get_vector_store(persist_directory=tmp_path_factory.mktemp("chroma"))
    ingest(vector_store)
    return vector_store


@pytest.mark.parametrize("question, source, section", RETRIEVAL_CASES)
def test_retrieves_the_right_policy_section(store, question, source, section):
    chunks = retrieve(question, vector_store=store)

    assert chunks[0].source == source
    assert section in [chunk.section for chunk in chunks]


def test_warranty_question_retrieves_manufacturer_warranty(store):
    chunks = retrieve("My laptop stopped working after 6 months", vector_store=store)

    assert "Manufacturer warranty" in [chunk.section for chunk in chunks]


@pytest.mark.xfail(reason="Known limitation: 'Return window' ranks first because it shares the word 'laptop'.", strict=False)
def test_warranty_question_ranks_warranty_first(store):
    assert retrieve("My laptop stopped working after 6 months", vector_store=store)[0].source == "warranty.md"


@pytest.mark.parametrize("question, expected_text, source", ANSWER_CASES)
def test_answers_from_the_policy_and_cites_it(store, question, expected_text, source):
    result = answer_question(question, retrieve(question, vector_store=store))

    assert result.answered
    assert expected_text in result.answer
    assert source in result.sources


@pytest.mark.parametrize("question", UNANSWERABLE)
def test_says_so_when_the_policies_do_not_answer(store, question):
    result = answer_question(question, retrieve(question, vector_store=store))

    assert result.answered is False
    assert result.answer == INSUFFICIENT_ANSWER
    assert result.sources == []


@pytest.mark.xfail(
    reason="Known limitation of qwen2.5:3b: mixes the 'Refurbished products' excerpt into this answer and claims "
    "VoltCart repairs for free. The policy says the manufacturer handles repairs.",
    strict=False,
)
def test_warranty_answer_does_not_mix_up_who_repairs(store):
    question = "My laptop stopped working after 6 months"
    result = answer_question(question, retrieve(question, vector_store=store))

    assert "manufacturer handles" in result.answer.lower()
