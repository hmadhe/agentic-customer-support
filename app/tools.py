from collections.abc import Callable
from pathlib import Path

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field, field_validator

from app import orders
from app.answer import format_context
from app.schemas import RetrievedChunk


# The agent gets only the top chunks: with 4, it mixed unrelated rules into answers (see the Milestone 4 log).
SEARCH_TOOL_CHUNKS = 2


class OrderLookup(BaseModel):
    order_id: str = Field(description="The customer's order number, digits only, e.g. '1042'.")

    @field_validator("order_id")
    @classmethod
    def digits_only(cls, value: str) -> str:
        value = value.strip().lstrip("#")
        if not value.isdigit():
            raise ValueError("order_id must be the order number in digits only, e.g. '1042'")
        return value


class PolicySearch(BaseModel):
    query: str = Field(description="What to look up in VoltCart's policies, e.g. 'damaged item on arrival'.")


def describe_order(order: orders.Order) -> str:
    # Short sentences with only the facts that apply: given a full JSON dump, the model read out empty fields.
    parts = [f"Order {order.order_id}: {order.product}.", f"Status: {order.status}.", f"Ordered on {order.order_date}."]
    if order.delivered_date:
        parts.append(f"Delivered on {order.delivered_date}.")
    if order.tracking_number:
        parts.append(f"Tracking number: {order.tracking_number}.")
    return " ".join(parts)


def describe_eligibility(order_id: str, eligibility: orders.ReturnEligibility) -> str:
    if not eligibility.eligible:
        return f"Order {order_id} cannot be returned. Reason: {eligibility.reason}"
    fee = f"A {eligibility.restocking_fee_percent}% restocking fee applies." if eligibility.restocking_fee_percent else "No restocking fee."
    return f"Order {order_id} can be returned until {eligibility.return_deadline}. {eligibility.reason} {fee}"


def build_tools(retriever: Callable[[str], list[RetrievedChunk]], db_path: Path = orders.DB_PATH) -> list[BaseTool]:
    """The tools the order agent can call. Dependencies are passed in so tests can use fakes."""

    @tool(args_schema=OrderLookup)
    def get_order_status(order_id: str) -> str:
        """Look up a customer's order: product, status, order and delivery dates, and tracking number."""
        order = orders.get_order(order_id, db_path)
        if order is None:
            return f"No order found with number {order_id}."
        return describe_order(order)

    @tool(args_schema=OrderLookup)
    def check_return_eligibility(order_id: str) -> str:
        """Check whether a customer's order can be returned, by when, and whether a restocking fee applies."""
        order = orders.get_order(order_id, db_path)
        if order is None:
            return f"No order found with number {order_id}."
        return describe_eligibility(order_id, orders.check_return_eligibility(order))

    @tool(args_schema=PolicySearch)
    def search_policies(query: str) -> str:
        """Search VoltCart's policies (shipping, returns, warranty, payments, account), e.g. what to do about a damaged item."""
        return format_context(retriever(query)[:SEARCH_TOOL_CHUNKS])

    return [get_order_status, check_return_eligibility, search_policies]
