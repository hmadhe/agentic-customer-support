from datetime import date, timedelta

import pytest

from app.orders import Order, check_return_eligibility, get_order, seed_db

TODAY = date(2026, 10, 5)


def make_order(category="headphones", delivered_days_ago=10, opened=False, final_sale=False, status="delivered"):
    delivered = TODAY - timedelta(days=delivered_days_ago) if status == "delivered" else None
    return Order(
        order_id="1", product="Test product", category=category, status=status, order_date=TODAY - timedelta(days=20),
        delivered_date=delivered, opened=opened, final_sale=final_sale, total=100.0, tracking_number=None,
    )


def test_seeded_order_can_be_read_back(tmp_path):
    db = tmp_path / "orders.db"
    seed_db(db, today=TODAY)

    order = get_order("1001", db)

    assert order.product == "Lenovo ThinkPad X1 laptop"
    assert order.delivered_date == date(2026, 9, 25)  # delivered 10 days before TODAY
    assert order.opened is True


def test_unknown_order_returns_none(tmp_path):
    db = tmp_path / "orders.db"
    seed_db(db, today=TODAY)

    assert get_order("9999", db) is None


def test_reseeding_does_not_duplicate_orders(tmp_path):
    db = tmp_path / "orders.db"

    assert seed_db(db, today=TODAY) == seed_db(db, today=TODAY) == 12


@pytest.mark.parametrize(
    "order, eligible, fee",
    [
        (make_order("headphones", delivered_days_ago=30), True, 0),  # last day of the 30-day window
        (make_order("headphones", delivered_days_ago=31), False, 0),
        (make_order("laptop", delivered_days_ago=15, opened=True), True, 15),  # opened laptop: 15 days + fee
        (make_order("laptop", delivered_days_ago=16, opened=True), False, 0),
        (make_order("laptop", delivered_days_ago=20, opened=False), True, 0),  # unopened laptop: 30 days, no fee
        (make_order("phone", delivered_days_ago=16, opened=True), False, 0),
        (make_order("drone", delivered_days_ago=20, opened=True), True, 15),  # 30 days, but fee applies
        (make_order("earbuds", delivered_days_ago=1, opened=True), False, 0),  # hygiene rule
        (make_order("earbuds", delivered_days_ago=1, opened=False), True, 0),
        (make_order("gift_card", delivered_days_ago=1), False, 0),
        (make_order("camera", delivered_days_ago=1, final_sale=True), False, 0),
        (make_order("tv", status="shipped"), False, 0),
    ],
)
def test_return_eligibility_follows_the_returns_policy(order, eligible, fee):
    result = check_return_eligibility(order, today=TODAY)

    assert result.eligible is eligible
    assert result.restocking_fee_percent == fee
