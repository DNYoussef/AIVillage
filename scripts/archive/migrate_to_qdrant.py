"""Migrate existing FAISS vectors into a Qdrant collection.

Run ``python scripts/migrate_to_qdrant.py`` with optional ``--dry-run``
and ``--delete-existing`` flags.  The script expects a serialized
``VectorStore`` (``vector_store.json`` by default) and will upsert the
stored embeddings into the Qdrant collection defined by the environment
variables ``QDRANT_URL`` and ``COLLECTION_NAME``.

The output is a JSON summary showing ``total`` vectors scanned,
``migrated`` vectors written, whether the run was a dry run, and the
elapsed time in seconds.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

from rag_system.vector_store import FaissAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qdrant-migrator")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("COLLECTION_NAME", "ai_village_vectors")
BATCH = 100
DIMENSION = 768


def ensure_collection(client: QdrantClient, delete_existing: bool) -> None:
    """Create the destination collection if needed."""
    collections = {c.name for c in client.get_collections().collections}
    if delete_existing and COLLECTION in collections:
        logger.info("Deleting existing collection %s", COLLECTION)
        client.delete_collection(collection_name=COLLECTION)
        collections.remove(COLLECTION)

    if COLLECTION not in collections:
        logger.info("Creating collection %s", COLLECTION)
        client.recreate_collection(
            COLLECTION,
            vectors_config=models.VectorParams(size=DIMENSION, distance=models.Distance.COSINE),
        )


def migrate(dry_run: bool, delete_existing: bool) -> dict[str, Any]:
    """Perform the migration and return a report."""
    start = time.time()
    adapter = FaissAdapter()
    client = QdrantClient(url=QDRANT_URL, timeout=60)

    ensure_collection(client, delete_existing)

    total = 0
    written = 0
    for ids, vectors, payload in adapter.iter_embeddings(batch_size=BATCH):
        total += len(ids)
        if dry_run:
            continue

        client.upsert(
            collection_name=COLLECTION,
            points=[
                models.PointStruct(id=i, vector=v, payload=p) for i, v, p in zip(ids, vectors, payload, strict=False)
            ],
        )
        written += len(ids)

    elapsed = round(time.time() - start, 2)
    return {
        "total": total,
        "migrated": written,
        "dry_run": dry_run,
        "seconds": elapsed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="do not write")
    parser.add_argument(
        "--delete-existing",
        action="store_true",
        help="delete and recreate the destination collection",
    )
    args = parser.parse_args()

    report = migrate(args.dry_run, args.delete_existing)
    print(json.dumps(report, indent=2))
    if not args.dry_run and report["migrated"] != report["total"]:
        exit(1)
