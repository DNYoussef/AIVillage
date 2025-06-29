from chromadb import PersistentClient
import csv
import os

from ingestion import add_text


def run(user_id: str, chroma_client: PersistentClient) -> int:
    """Ingest Amazon order history CSV."""
    csv_path = os.getenv("AMAZON_CSV", os.path.expanduser("~/amazon_orders.csv"))
    if not os.path.exists(csv_path):
        return 0
    coll = chroma_client.get_or_create_collection(f"user:{user_id}")
    n = 0
    with open(csv_path, newline="", encoding="utf8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("Title") or row.get("Item Name")
            if not text:
                continue
            meta = {"order_id": row.get("Order ID"), "src": "amazon"}
            if add_text(coll, text, meta, f"amz:{row.get('Order ID', n)}"):
                n += 1
            if n >= 1000:
                break
    return n
