import pytest
from pydantic import ValidationError

from app.agent import invented_order_ids
from app.orders import get_order
from app.schemas import RetrievedChunk
from app.tools import SEARCH_TOOL_CHUNKS, build_tools

CHUNKS = [
    RetrievedChunk(text=f"Policy > Section {i}\n\nRule {i}.", source=f"doc{i}.md", section=f"Section {i}", score=0.9 - i / 10)
    for i in range(4)
]


@pytest.fixture
def tools(order_db):
    get_order_status, check_return_eligibility, search_policies = build_tools(lambda _: CHUNKS, order_db)
    return {"status": get_order_status, "eligibility": check_return_eligibility, "search": search_policies}


def test_order_status_accepts_a_leading_hash(tools):
    assert tools["status"].invoke({"order_id": "#1042"}).startswith("Order 1042: Bose QuietComfort headphones. Status: shipped.")


def test_order_status_only_mentions_facts_that_exist(tools):
    # Order 1007 is still processing: no delivery date and no tracking number yet.
    text = tools["status"].invoke({"order_id": "1007"})

    assert "Status: processing." in text
    assert "Delivered" not in text
    assert "Tracking" not in text


def test_delivered_order_status_includes_the_delivery_date(tools):
    # Found by the coverage report: no test covered a delivered order's status text.
    text = tools["status"].invoke({"order_id": "1001"})

    assert text.startswith("Order 1001: Lenovo ThinkPad X1 laptop. Status: delivered.")
    assert "Delivered on " in text and "Tracking number: VC100100." in text


def test_unknown_order_gives_a_clear_message(tools):
    assert tools["status"].invoke({"order_id": "9999"}) == "No order found with number 9999."
    # Found by the coverage report: the eligibility tool's unknown-order path was untested.
    assert tools["eligibility"].invoke({"order_id": "9999"}) == "No order found with number 9999."


def test_non_numeric_order_id_is_rejected(tools):
    with pytest.raises(ValidationError):
        tools["status"].invoke({"order_id": "ORD-12"})


def test_eligibility_for_a_returnable_order_includes_deadline_and_fee(tools):
    text = tools["eligibility"].invoke({"order_id": "1001"})

    assert text.startswith("Order 1001 can be returned until ")
    assert "A 15% restocking fee applies." in text


def test_eligibility_for_a_non_returnable_order_gives_only_the_reason(tools):
    text = tools["eligibility"].invoke({"order_id": "1004"})

    assert text == "Order 1004 cannot be returned. Reason: Opened in-ear headphones and earbuds cannot be returned, for hygiene reasons."


def test_policy_search_returns_only_the_top_chunks_with_sources(tools):
    text = tools["search"].invoke({"query": "anything"})

    assert text.count("[source: ") == SEARCH_TOOL_CHUNKS == 2
    assert "[source: doc0.md]" in text and "[source: doc2.md]" not in text


def test_missing_database_raises_clearly_and_creates_no_file(tmp_path):
    missing = tmp_path / "missing.db"

    with pytest.raises(FileNotFoundError, match="seed_orders"):
        get_order("1001", missing)
    assert not missing.exists()


@pytest.mark.parametrize(
    "order_ids, message, invented",
    [
        (["1042"], "Where is my order #1042?", []),
        (["#1042"], "Where is my order 1042?", []),
        (["123456"], "Can I return the laptop I bought last week?", ["123456"]),  # the real qwen output
        (["1001", "1002"], "Compare orders 1001 and 1003", ["1002"]),
    ],
)
def test_invented_order_ids_are_detected(order_ids, message, invented):
    tool_calls = [{"name": "get_order_status", "args": {"order_id": order_id}, "id": str(i)} for i, order_id in enumerate(order_ids)]

    assert invented_order_ids(tool_calls, message) == invented


def test_tools_without_an_order_id_are_never_flagged():
    tool_calls = [{"name": "search_policies", "args": {"query": "damaged item"}, "id": "1"}]

    assert invented_order_ids(tool_calls, "My headphones arrived broken") == []
