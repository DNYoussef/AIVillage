import mailbox
import os

from chromadb import PersistentClient
from langdetect import detect

from ingestion import add_text


def run(user_id: str, chroma_client: PersistentClient) -> int:
    """Ingest local Gmail mbox exported file."""
    mbox_path = os.getenv("GMAIL_MBOX", os.path.expanduser("~/gmail.mbox"))
    if not os.path.exists(mbox_path):
        return 0
    mbox = mailbox.mbox(mbox_path)
    coll = chroma_client.get_or_create_collection(f"user:{user_id}")
    n = 0
    for msg in mbox:
        body = msg.get_payload(decode=True)
        if not body:
            continue
        try:
            text = body.decode(errors="ignore")
        except Exception:
            continue
        try:
            if detect(text) != "en":
                continue
        except Exception:
            pass
        if add_text(coll, text, {"src": "gmail"}, f"gmail:{n}"):
            n += 1
        if n >= 1000:
            break
    return n
