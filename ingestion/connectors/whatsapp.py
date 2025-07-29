"""Secure WhatsApp database ingestion connector."""

from contextlib import contextmanager
import logging
import os
import sqlite3

from chromadb import PersistentClient

from ingestion import add_text

logger = logging.getLogger(__name__)


@contextmanager
def safe_db_connection(db_path: str):
    """Context manager that validates and safely opens a SQLite database."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = None
    try:
        conn = sqlite3.connect(db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        yield conn
    finally:
        if conn is not None:
            conn.close()


def validate_db_path(db_path: str) -> str:
    """Validate database path to prevent unsafe access."""
    abs_path = os.path.abspath(db_path)
    allowed_base = os.path.abspath("/data/data/com.whatsapp/databases/")

    if not abs_path.startswith(allowed_base):
        raise ValueError("Database path not allowed")

    if not abs_path.endswith(".db"):
        raise ValueError("Invalid database file extension")

    return abs_path


def sanitize_message_text(text: str) -> str:
    """Sanitize message text before indexing."""
    if not isinstance(text, str):
        return ""

    clean = "".join(ch for ch in text if ord(ch) >= 32 or ch in "\n\t")
    return clean[:10000]


def run(user_id: str, chroma_client: PersistentClient) -> int:
    """Parse local WhatsApp db -- returns # docs (secure version)."""
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("Invalid user_id")

    safe_user_id = "".join(c for c in user_id if c.isalnum() or c in "-_")[:50]

    db_env = os.getenv("WA_DB_PATH", "/data/data/com.whatsapp/databases/msgstore.db")
    try:
        db_path = validate_db_path(db_env)
    except (ValueError, FileNotFoundError) as e:
        logger.warning("Database validation failed: %s", e)
        return 0

    limit = min(int(os.getenv("MAX_MESSAGES", "1000")), 10000)

    try:
        with safe_db_connection(db_path) as conn:
            coll = chroma_client.get_or_create_collection(f"user:{safe_user_id}")
            n = 0
            query = (
                "SELECT data, timestamp FROM messages WHERE data IS NOT NULL "
                "ORDER BY timestamp DESC LIMIT ?"
            )
            for row in conn.execute(query, (limit,)):
                text = sanitize_message_text(row["data"])
                ts = row["timestamp"]

                if not text.strip() or not isinstance(ts, (int, float)) or ts < 0:
                    continue

                try:
                    if add_text(coll, text, {"ts": ts, "src": "wa"}, f"wa:{ts}"):
                        n += 1
                except Exception as exc:  # pragma: no cover - external lib calls
                    logger.warning("Failed to add message: %s", exc)
                    continue

                if n >= limit:
                    break

            logger.info("Successfully processed %s messages for user %s", n, safe_user_id)
            return n
    except Exception as exc:  # pragma: no cover - external lib calls
        logger.error("WhatsApp connector failed: %s", exc)
        return 0
