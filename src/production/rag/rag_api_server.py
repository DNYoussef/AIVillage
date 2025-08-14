"""CODEX-compliant RAG API Server.

Provides REST API endpoints on port 8082 as specified in CODEX requirements.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from rag_system.core.codex_rag_integration import CODEXRAGPipeline, Document

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# CODEX-specified port
RAG_API_PORT = int(os.getenv("RAG_API_PORT", "8082"))

# Initialize FastAPI app
app = FastAPI(
    title="CODEX RAG Pipeline API",
    description="CODEX-compliant RAG system with <100ms retrieval target",
    version="1.0.0",
)

# Global pipeline instance
pipeline: CODEXRAGPipeline | None = None


class DocumentInput(BaseModel):
    """Input model for document indexing."""

    id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    source_type: str = Field("wikipedia", description="Source type (wikipedia, educational, etc.)")
    metadata: dict[str, Any] | None = Field(default_factory=dict, description="Additional metadata")


class QueryRequest(BaseModel):
    """Request model for RAG queries."""

    query: str = Field(..., description="Search query")
    k: int = Field(10, description="Number of results to retrieve", ge=1, le=100)
    use_cache: bool = Field(True, description="Whether to use cache")
    rerank: bool = Field(True, description="Whether to use cross-encoder reranking")


class QueryResponse(BaseModel):
    """Response model for RAG queries."""

    query: str
    results: list[dict[str, Any]]
    metrics: dict[str, Any]
    timestamp: float


class IndexRequest(BaseModel):
    """Request model for batch document indexing."""

    documents: list[DocumentInput]
    chunk_size: int | None = Field(512, description="Chunk size in tokens")
    chunk_overlap: int | None = Field(50, description="Chunk overlap in tokens")


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    pipeline_ready: bool
    index_size: int
    cache_enabled: bool
    performance_metrics: dict[str, Any]
    timestamp: float


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize the RAG pipeline on startup."""
    global pipeline
    logger.info("Starting CODEX RAG API server...")

    try:
        pipeline = CODEXRAGPipeline()
        logger.info("RAG pipeline initialized successfully")

        # Load existing Wikipedia data if available
        wikipedia_data_path = Path("./data/wikipedia_corpus.json")
        if wikipedia_data_path.exists():
            logger.info("Loading Wikipedia corpus...")
            with open(wikipedia_data_path, encoding="utf-8") as f:
                data = json.load(f)
                documents = [Document(**doc) for doc in data["documents"]]
                stats = pipeline.index_documents(documents)
                logger.info(f"Loaded {stats['documents_processed']} Wikipedia articles")

    except Exception as e:
        logger.exception(f"Failed to initialize pipeline: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Cleanup on shutdown."""
    logger.info("Shutting down RAG API server...")


@app.get("/health/rag", response_model=HealthResponse)
async def health_check():
    """CODEX-required health check endpoint."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    perf_metrics = pipeline.get_performance_metrics()

    return HealthResponse(
        status="healthy" if perf_metrics.get("meets_target", False) else "degraded",
        pipeline_ready=True,
        index_size=perf_metrics.get("index_size", 0),
        cache_enabled=pipeline.cache.enabled,
        performance_metrics=perf_metrics,
        timestamp=time.time(),
    )


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Retrieve relevant documents for a query.

    This endpoint performs hybrid retrieval with optional caching and reranking.
    Target latency: <100ms
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        start_time = time.perf_counter()

        # Perform retrieval
        results, metrics = await pipeline.retrieve(query=request.query, k=request.k, use_cache=request.use_cache)

        # Format results
        formatted_results = [
            {
                "chunk_id": r.chunk_id,
                "document_id": r.document_id,
                "text": r.text,
                "score": r.score,
                "retrieval_method": r.retrieval_method,
                "metadata": r.metadata,
            }
            for r in results
        ]

        # Add timing info
        total_latency = (time.perf_counter() - start_time) * 1000
        metrics["total_latency_ms"] = total_latency

        # Log slow queries
        if total_latency > 100:
            logger.warning(f"Slow query detected: '{request.query[:50]}...' " f"took {total_latency:.2f}ms")

        return QueryResponse(
            query=request.query,
            results=formatted_results,
            metrics=metrics,
            timestamp=time.time(),
        )

    except Exception as e:
        logger.exception(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index")
async def index_documents(request: IndexRequest):
    """Index new documents into the RAG system."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        # Convert to Document objects
        documents = [
            Document(
                id=doc.id,
                title=doc.title,
                content=doc.content,
                source_type=doc.source_type,
                metadata=doc.metadata,
            )
            for doc in request.documents
        ]

        # Index documents
        stats = pipeline.index_documents(documents)

        return JSONResponse(content={"status": "success", "stats": stats, "timestamp": time.time()})

    except Exception as e:
        logger.exception(f"Indexing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Get detailed performance metrics."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    metrics = pipeline.get_performance_metrics()

    return JSONResponse(
        content={
            "pipeline_metrics": metrics,
            "cache_metrics": pipeline.cache.get_metrics() if pipeline.cache else {},
            "timestamp": time.time(),
        }
    )


@app.post("/clear_cache")
async def clear_cache():
    """Clear all cache layers."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        # Clear cache layers
        if pipeline.cache:
            pipeline.cache.l1_cache.clear()
            if pipeline.cache.l2_cache:
                pipeline.cache.l2_cache.flushdb()
            if pipeline.cache.l3_cache:
                pipeline.cache.l3_cache.clear()

            # Reset metrics
            pipeline.cache.hits = {"l1": 0, "l2": 0, "l3": 0}
            pipeline.cache.misses = 0
            pipeline.cache.latencies = []

        return JSONResponse(
            content={
                "status": "success",
                "message": "Cache cleared successfully",
                "timestamp": time.time(),
            }
        )

    except Exception as e:
        logger.exception(f"Cache clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/warm_cache")
async def warm_cache(queries: list[str] = Query(...)):
    """Warm the cache with common queries."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        warmed = 0
        total_time = 0

        for query in queries:
            start = time.perf_counter()
            await pipeline.retrieve(query, k=10, use_cache=True)
            total_time += (time.perf_counter() - start) * 1000
            warmed += 1

        return JSONResponse(
            content={
                "status": "success",
                "queries_warmed": warmed,
                "total_time_ms": total_time,
                "avg_time_ms": total_time / warmed if warmed > 0 else 0,
                "timestamp": time.time(),
            }
        )

    except Exception as e:
        logger.exception(f"Cache warming error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "CODEX RAG Pipeline API",
        "version": "1.0.0",
        "port": RAG_API_PORT,
        "health_endpoint": "/health/rag",
        "documentation": "/docs",
        "requirements": "CODEX Integration Requirements compliant",
    }


def run_server() -> None:
    """Run the RAG API server."""
    logger.info(f"Starting RAG API server on port {RAG_API_PORT}...")

    uvicorn.run(app, host="0.0.0.0", port=RAG_API_PORT, log_level="info", access_log=True)


if __name__ == "__main__":
    run_server()
