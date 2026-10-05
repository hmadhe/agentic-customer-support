import sqlite3
from datetime import date, timedelta
from pathlib import Path

from pydantic import BaseModel

from app.config import PROJECT_ROOT

DB_PATH = PROJECT_ROOT / "data" / "voltcart.db"


class Order(BaseModel):
    order_id: str
    product: str
    category: str  # laptop, tablet, phone, camera, drone, earbuds, headphones, tv, gift_card
    status: str  # processing, shipped, delivered
    order_date: date
    delivered_date: date | None
    opened: bool
    final_sale: bool
    total: float
    tracking_number: str | None


class ReturnEligibility(BaseModel):
    eligible: bool
    reason: str
    return_deadline: date | None = None
    restocking_fee_percent: int = 0


# (order_id, product, category, status, days since ordered, days since delivered or None, opened, final_sale, total, tracking)
SEED_ORDERS = [
    ("1001", "Lenovo ThinkPad X1 laptop", "laptop", "delivered", 13, 10, True, False, 1499.00, "VC100100"),
    ("1002", "Dell XPS 13 laptop", "laptop", "delivered", 24, 20, True, False, 1199.00, "VC100200"),
    ("1003", "Sony WH-1000XM5 headphones", "headphones", "delivered", 28, 25, False, False, 349.00, "VC100300"),
    ("1004", "Apple AirPods Pro earbuds", "earbuds", "delivered", 8, 5, True, False, 249.00, "VC100400"),
    ("1005", "Samsung 65-inch TV", "tv", "shipped", 3, None, False, False, 899.00, "VC100500"),
    ("1006", "DJI Mini 4 drone", "drone", "delivered", 6, 3, True, False, 759.00, "VC100600"),
    ("1007", "iPad Air tablet", "tablet", "processing", 0, None, False, False, 599.00, None),
    ("1008", "VoltCart $100 gift card", "gift_card", "delivered", 12, 11, False, False, 100.00, None),
    ("1009", "Canon EOS R50 camera", "camera", "delivered", 9, 7, False, True, 579.00, "VC100900"),
    ("1010", "Google Pixel 9 phone", "phone", "delivered", 45, 40, True, False, 799.00, "VC101000"),
    ("1042", "Bose QuietComfort headphones", "headphones", "shipped", 4, None, False, False, 279.00, "VC104200"),
    ("2231", "Sennheiser HD 660 headphones", "headphones", "delivered", 4, 1, True, False, 499.00, "VC223100"),
]

# Rules from data/policies/returns.md
SHORT_WINDOW_CATEGORIES = {"laptop", "tablet", "phone"}  # 15 days once opened
RESTOCKING_FEE_CATEGORIES = {"laptop", "camera", "drone"}  # 15% once opened
STANDARD_WINDOW_DAYS = 30
SHORT_WINDOW_DAYS = 15
RESTOCKING_FEE_PERCENT = 15


def seed_db(db_path: Path = DB_PATH, today: date | None = None) -> int:
    """(Re)create the orders table with the seed orders, dated relative to `today`. Returns the row count."""
    today = today or date.today()
    with sqlite3.connect(db_path) as connection:
        connection.execute("DROP TABLE IF EXISTS orders")
        connection.execute(
            """CREATE TABLE orders (
                order_id TEXT PRIMARY KEY, product TEXT, category TEXT, status TEXT,
                order_date TEXT, delivered_date TEXT, opened INTEGER, final_sale INTEGER,
                total REAL, tracking_number TEXT)"""
        )
        for order_id, product, category, status, ordered_ago, delivered_ago, opened, final_sale, total, tracking in SEED_ORDERS:
            delivered = (today - timedelta(days=delivered_ago)).isoformat() if delivered_ago is not None else None
            connection.execute(
                "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (order_id, product, category, status, (today - timedelta(days=ordered_ago)).isoformat(),
                 delivered, opened, final_sale, total, tracking),
            )
    return len(SEED_ORDERS)


def require_db(db_path: Path = DB_PATH) -> None:
    """Fail early with instructions if the database was never created."""
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Order database not found at {db_path}. Create it with: python -m scripts.seed_orders")


def get_order(order_id: str, db_path: Path = DB_PATH) -> Order | None:
    require_db(db_path)
    # Read-only: a lookup should never write. A normal connect() silently creates an empty file if it's missing.
    with sqlite3.connect(f"{Path(db_path).as_uri()}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone()
    return Order(**dict(row)) if row else None


def check_return_eligibility(order: Order, today: date | None = None) -> ReturnEligibility:
    """Apply the returns policy to one order. Pure Python, so the answer never depends on the LLM."""
    today = today or date.today()

    if order.status != "delivered" or order.delivered_date is None:
        return ReturnEligibility(eligible=False, reason=f"The order has not been delivered yet (status: {order.status}).")
    if order.category == "gift_card":
        return ReturnEligibility(eligible=False, reason="Gift cards cannot be returned.")
    if order.final_sale:
        return ReturnEligibility(eligible=False, reason="Clearance items marked 'Final sale' cannot be returned.")
    if order.category == "earbuds" and order.opened:
        return ReturnEligibility(eligible=False, reason="Opened in-ear headphones and earbuds cannot be returned, for hygiene reasons.")

    short_window = order.opened and order.category in SHORT_WINDOW_CATEGORIES
    window_days = SHORT_WINDOW_DAYS if short_window else STANDARD_WINDOW_DAYS
    deadline = order.delivered_date + timedelta(days=window_days)
    if today > deadline:
        return ReturnEligibility(
            eligible=False,
            reason=f"The {window_days}-day return window ended on {deadline.isoformat()}.",
            return_deadline=deadline,
        )

    fee = RESTOCKING_FEE_PERCENT if order.opened and order.category in RESTOCKING_FEE_CATEGORIES else 0
    return ReturnEligibility(
        eligible=True,
        reason=f"Within the {window_days}-day return window.",
        return_deadline=deadline,
        restocking_fee_percent=fee,
    )
