from __future__ import annotations

from typing import Any

from core.evidence import Chunk, EvidencePack


def wrap_in_pack(result: dict[str, Any], query: str) -> EvidencePack | None:
    """Return EvidencePack built from retrieval result."""
    if "evidence_pack" in result:
        return None
    retrieved: list[Any] = result.get("retrieved_info", [])
    chunks: list[Chunk] = []
    for idx, r in enumerate(retrieved):
        try:
            chunks.append(
                Chunk(
                    id=str(getattr(r, "id", idx)),
                    text=getattr(r, "content", ""),
                    score=min(max(float(getattr(r, "score", 0.0)), 0.0), 1.0),
                    source_uri="https://example.com",
                )
            )
        except Exception:
            continue
    if not chunks:
        return None
    pack = EvidencePack(query=query, chunks=chunks)
    result["evidence_pack"] = pack
    return pack
