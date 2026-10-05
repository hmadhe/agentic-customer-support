"""(Re)create the mock order database: python -m scripts.seed_orders"""

from app.orders import DB_PATH, seed_db


def main() -> None:
    count = seed_db()
    print(f"Seeded {count} orders into {DB_PATH} (dates are relative to today).")


if __name__ == "__main__":
    main()
