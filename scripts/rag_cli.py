#!/usr/bin/env python3
"""Command line interface for Unified RAG components.

Commands
--------
```
ingest <file>               # Chunk and store a document
query <text> --mode MODE    # Query the unified system in a given mode
detect-gaps                # Run missing node detection
```
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Ensure the src directory is on the import path so we can access unified_rag modules
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT / "src"))


async def ingest_file(file_path: str) -> None:
    """Ingest a document using AdvancedIngestionEngine."""
    try:
        from unified_rag.ingestion.advanced_ingestion_engine import (
            AdvancedIngestionEngine,
        )
    except Exception as exc:  # pragma: no cover - best effort import
        print(f"Failed to import AdvancedIngestionEngine: {exc}")
        return

    engine = AdvancedIngestionEngine()
    try:
        await engine.initialize()
        content = Path(file_path).read_text(encoding="utf-8")
        processed = await engine.process_document(content, Path(file_path).stem)
    except Exception as exc:  # pragma: no cover - runtime guard
        print(f"Ingestion failed: {exc}")
        return

    # Store chunks to a JSON file next to the original document
    out_file = Path(file_path).with_suffix(Path(file_path).suffix + ".chunks.json")
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump([chunk.content for chunk in processed.chunks], fh, indent=2)
    print(f"Stored {len(processed.chunks)} chunks in {out_file}")


async def query_system(text: str, mode: str) -> None:
    """Query the UnifiedRAGSystem in a specified retrieval mode."""
    try:
        from unified_rag.core.unified_rag_system import (
            UnifiedRAGSystem,
            QueryType,
            RetrievalMode,
            QueryContext,
        )
    except Exception as exc:  # pragma: no cover - best effort import
        print(f"Failed to import UnifiedRAGSystem: {exc}")
        return

    system = UnifiedRAGSystem()
    try:
        await system.initialize()
    except Exception as exc:  # pragma: no cover - initialization may require optional deps
        print(f"Initialization failed: {exc}")
        return

    mode_map = {
        "creative": RetrievalMode.CREATIVE,
        "balanced": RetrievalMode.BALANCED,
        "analytical": RetrievalMode.BALANCED,
    }
    query_type_map = {
        "creative": QueryType.CREATIVE,
        "analytical": QueryType.ANALYTICAL,
    }
    retrieval_mode = mode_map.get(mode, RetrievalMode.BALANCED)
    query_type = query_type_map.get(mode, QueryType.FACTUAL)

    retrieve = getattr(system, "retrieve", system.query)
    try:
        response = await retrieve(
            text,
            query_type=query_type,
            retrieval_mode=retrieval_mode,
            context=QueryContext(),
        )
    except Exception as exc:  # pragma: no cover
        print(f"Query failed: {exc}")
        return

    print(response.answer)


async def detect_gaps() -> None:
    """Run MissingNodeDetector and print recommendations."""
    try:
        import networkx as nx
        from unified_rag.graph.missing_node_detector import MissingNodeDetector
    except Exception as exc:  # pragma: no cover
        print(f"Failed to import MissingNodeDetector: {exc}")
        return

    graph = nx.Graph()
    detector = MissingNodeDetector(graph)
    try:
        analysis = await detector.detect_missing_nodes()
    except Exception as exc:  # pragma: no cover
        print(f"Gap detection failed: {exc}")
        return

    if analysis.recommendations:
        for rec in analysis.recommendations:
            print(f"- {rec}")
    else:
        print("No gap recommendations available.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified RAG CLI tool")
    sub = parser.add_subparsers(dest="command")

    p_ingest = sub.add_parser("ingest", help="Ingest a document file")
    p_ingest.add_argument("file", help="Path to document")

    p_query = sub.add_parser("query", help="Query the RAG system")
    p_query.add_argument("text", help="Text of the question")
    p_query.add_argument(
        "--mode",
        choices=["creative", "balanced", "analytical"],
        default="balanced",
        help="Retrieval mode to use",
    )

    sub.add_parser("detect-gaps", help="Detect missing knowledge graph nodes")

    args = parser.parse_args()

    if args.command == "ingest":
        asyncio.run(ingest_file(args.file))
    elif args.command == "query":
        asyncio.run(query_system(args.text, args.mode))
    elif args.command == "detect-gaps":
        asyncio.run(detect_gaps())
    else:
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
