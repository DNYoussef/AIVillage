from chromadb import PersistentClient
import os
import sqlite3

from ingestion import add_text


def run(user_id: str, chroma_client: PersistentClient):
    """Parse local WhatsApp db -- returns # docs."""
    db = os.getenv("WA_DB_PATH", "/data/data/com.whatsapp/databases/msgstore.db")
    if not os.path.exists(db):
        return 0
    conn = sqlite3.connect(db)
    coll = chroma_client.get_or_create_collection(f"user:{user_id}")
    n = 0
    for text, ts in conn.execute("SELECT data, timestamp FROM messages LIMIT 1000"):
        if add_text(coll, text, {"ts": ts, "src": "wa"}, f"wa:{ts}"):
            n += 1
    return n
