"""Dynamic loader for connector plugins.

A connector = any module that exposes:
    def run(user_id:str, chroma_client) -> int   # returns # docs indexed
"""

import importlib
from pathlib import Path
import re

_BAD_WORDS = re.compile(r"(?:spam|unsubscribe|buy now)", re.I)

CONNECTOR_PATH = Path(__file__).parent / "connectors"


def _clean_text(text: str) -> str | None:
    if not text:
        return None
    if _BAD_WORDS.search(text):
        return None
    return text


def add_text(collection, text: str, meta: dict, doc_id: str):
    clean = _clean_text(text)
    if clean is None:
        return False
    collection.add(documents=[clean], metadatas=[meta], ids=[doc_id])
    return True


def available():
    return [p.stem for p in CONNECTOR_PATH.glob("*.py") if p.stem != "__init__"]


def run_all(user_id: str, chroma_client):
    total = 0
    for name in available():
        mod = importlib.import_module(f"ingestion.connectors.{name}")
        total += mod.run(user_id, chroma_client)
    return total
