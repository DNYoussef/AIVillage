# isort: skip_file
"""RAG System Entry Point."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Sequence

from src.cli.base import run_cli

try:  # pragma: no cover - import guard for optional dependencies
    from rag_system.core.pipeline import EnhancedRAGPipeline
except Exception:  # pragma: no cover - import guard
    EnhancedRAGPipeline = None


def query_system(args) -> int:
    """Query the RAG system."""
    if not args.question:
        print("Error: --question is required for query action")
        return 1

    if EnhancedRAGPipeline is None:
        print("EnhancedRAGPipeline is unavailable; CLI is experimental.")
        return 1

    print(f"Querying: {args.question}")

    async def _run_query(question: str):
        pipeline = EnhancedRAGPipeline()
        await pipeline.initialize()
        result = await pipeline.process(question)
        await pipeline.shutdown()
        return result

    try:
        result = asyncio.run(_run_query(args.question))
    except Exception as exc:  # pragma: no cover - runtime safety
        print(f"RAG query failed: {exc}")
        return 1

    print(result)
    return 0


def index_document(args) -> int:
    """Index a document."""
    if not args.document:
        print("Error: --document is required for index action")
        return 1

    print("Document indexing is experimental and not yet implemented.")
    return 1


def search_documents(args) -> int:
    """Search documents."""
    if not args.question:
        print("Error: --question is required for search action")
        return 1

    print("Document search is experimental and not yet implemented.")
    return 1


def get_status(args) -> int:
    """Get service status."""
    print("RAG system status: Running")
    return 0


def configure_service(args) -> int:
    """Configure service."""
    print("Service configuration is experimental and not yet implemented.")
    return 1


def _configure(parser) -> None:
    parser.add_argument("--question", help="Question to query")
    parser.add_argument("--document", help="Document to index")
    parser.add_argument("--config", "-c", help="Configuration file path")
    parser.add_argument("--input", help="Input file or directory")
    parser.add_argument("--output", help="Output file or directory")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )


actions = {
    "query": query_system,
    "index": index_document,
    "search": search_documents,
    "status": get_status,
    "config": configure_service,
}


def main(args: Sequence[str] | None = None) -> int:
    """Main entry point for RAG system."""
    return run_cli(
        "Experimental RAG System Service",
        actions,
        _configure,
        args,
    )


if __name__ == "__main__":
    sys.exit(main())
