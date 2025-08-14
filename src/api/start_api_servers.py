#!/usr/bin/env python3
"""Start all API servers for AIVillage integration with real implementations."""

import hashlib
import logging
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from concurrent.futures import ThreadPoolExecutor

import uvicorn
from fastapi import FastAPI, HTTPException

# Import CODEX-compliant RAG implementation
sys.path.insert(0, str(Path(__file__).parent.parent / "production" / "rag" / "rag_system" / "core"))
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.core.security.digital_twin_encryption import (
    DigitalTwinEncryption,
    DigitalTwinEncryptionError,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from codex_rag_integration import CODEXRAGPipeline, Document

    CODEX_RAG_AVAILABLE = True
    logger.info("CODEX-compliant RAG implementation loaded successfully")
except ImportError as e:
    logger.warning(f"CODEX RAG not available, using fallback: {e}")
    CODEXRAGPipeline = None
    Document = None
    CODEX_RAG_AVAILABLE = False


# ---------------------------------------------------------------------------
# Digital Twin API with encrypted persistence
# ---------------------------------------------------------------------------

digital_twin_app = FastAPI(title="Digital Twin API")

DIGITAL_TWIN_DB = Path("data/digital_twin_profiles.db")
DIGITAL_TWIN_DB.parent.mkdir(parents=True, exist_ok=True)


def init_digital_twin_db() -> None:
    """Create required tables for Digital Twin profiles."""
    with sqlite3.connect(DIGITAL_TWIN_DB) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id TEXT UNIQUE NOT NULL,
                user_id_hash TEXT NOT NULL,
                preferred_difficulty TEXT,
                learning_style_encrypted BLOB NOT NULL
            )
            """
        )


try:
    encryption = DigitalTwinEncryption()
except DigitalTwinEncryptionError:
    import base64

    raw_key = os.urandom(32)
    os.environ["DIGITAL_TWIN_ENCRYPTION_KEY"] = base64.b64encode(raw_key).decode()
    encryption = DigitalTwinEncryption()


@digital_twin_app.get("/health/twin")
async def health_twin() -> dict[str, Any]:
    try:
        with sqlite3.connect(DIGITAL_TWIN_DB) as conn:
            conn.execute("SELECT 1")
        return {"status": "healthy", "service": "digital_twin"}
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Digital Twin health check failed")
        raise HTTPException(status_code=500, detail=str(exc))


@digital_twin_app.post("/profile/create")
async def create_profile(data: dict[str, Any]) -> dict[str, Any]:
    user_id = data.get("user_id")
    learning_style = data.get("learning_style")
    preferred_difficulty = data.get("preferred_difficulty", "medium")
    if not user_id or not learning_style:
        raise HTTPException(status_code=400, detail="user_id and learning_style required")

    profile_id = hashlib.sha256(f"{user_id}-{os.urandom(4)}".encode()).hexdigest()[:16]
    user_hash = hashlib.sha256(user_id.encode()).hexdigest()

    try:
        encrypted_ls = encryption.encrypt_sensitive_field(learning_style, "learning_style")
        with sqlite3.connect(DIGITAL_TWIN_DB) as conn:
            conn.execute(
                "INSERT INTO profiles (profile_id, user_id_hash, preferred_difficulty, learning_style_encrypted)"
                " VALUES (?, ?, ?, ?)",
                (profile_id, user_hash, preferred_difficulty, encrypted_ls),
            )
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to create profile")
        raise HTTPException(status_code=500, detail=str(exc))

    return {"success": True, "profile_id": profile_id}


@digital_twin_app.get("/profile/{profile_id}")
async def get_profile(profile_id: str) -> dict[str, Any]:
    with sqlite3.connect(DIGITAL_TWIN_DB) as conn:
        row = conn.execute(
            "SELECT preferred_difficulty, learning_style_encrypted FROM profiles WHERE profile_id=?",
            (profile_id,),
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Profile not found")

    try:
        learning_style = encryption.decrypt_sensitive_field(row[1], "learning_style")
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to decrypt learning style")
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "profile_id": profile_id,
        "preferred_difficulty": row[0],
        "learning_style": learning_style,
    }


# ---------------------------------------------------------------------------
# Evolution Metrics API with persistent tracking
# ---------------------------------------------------------------------------

evolution_app = FastAPI(title="Evolution Metrics API")

EVOLUTION_DB = Path("data/evolution_metrics.db")
EVOLUTION_DB.parent.mkdir(parents=True, exist_ok=True)


def init_evolution_db() -> None:
    with sqlite3.connect(EVOLUTION_DB) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


@evolution_app.get("/health/evolution")
async def health_evolution() -> dict[str, Any]:
    try:
        with sqlite3.connect(EVOLUTION_DB) as conn:
            conn.execute("SELECT 1")
        return {"status": "healthy", "service": "evolution_metrics"}
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Evolution Metrics health check failed")
        raise HTTPException(status_code=500, detail=str(exc))


@evolution_app.post("/metrics/record")
async def record_metrics(data: dict[str, Any]) -> dict[str, Any]:
    metrics = data.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise HTTPException(status_code=400, detail="metrics dict required")

    try:
        with sqlite3.connect(EVOLUTION_DB) as conn:
            for name, value in metrics.items():
                conn.execute(
                    "INSERT INTO metrics (metric_name, metric_value) VALUES (?, ?)",
                    (name, float(value)),
                )
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to record metrics")
        raise HTTPException(status_code=500, detail=str(exc))

    return {"success": True, "metrics_recorded": len(metrics)}


@evolution_app.get("/metrics/latest")
async def get_latest_metrics() -> dict[str, Any]:
    try:
        with sqlite3.connect(EVOLUTION_DB) as conn:
            rows = conn.execute("SELECT metric_name, metric_value FROM metrics ORDER BY recorded_at DESC").fetchall()
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to fetch metrics")
        raise HTTPException(status_code=500, detail=str(exc))

    latest: dict[str, float] = {}
    for name, value in rows:
        if name not in latest:
            latest[name] = float(value)

    return latest


# ---------------------------------------------------------------------------
# RAG Pipeline API with CODEX-compliant FAISS + sentence-transformers
# ---------------------------------------------------------------------------

rag_app = FastAPI(title="RAG Pipeline API")

# Initialize CODEX-compliant RAG pipeline
rag_pipeline: CODEXRAGPipeline = None


def init_rag_pipeline() -> None:
    """Initialize the CODEX-compliant RAG pipeline."""
    global rag_pipeline
    if not CODEX_RAG_AVAILABLE or CODEXRAGPipeline is None:
        logger.error("CODEX RAG implementation not available")
        rag_pipeline = None
        return

    try:
        rag_pipeline = CODEXRAGPipeline()
        logger.info("CODEX-compliant RAG pipeline initialized successfully")
    except Exception as e:
        logger.exception(f"Failed to initialize RAG pipeline: {e}")
        # Fallback to None - will be handled in endpoints
        rag_pipeline = None


@rag_app.get("/health/rag")
async def health_rag() -> dict[str, Any]:
    if rag_pipeline is None:
        return {
            "status": "degraded",
            "service": "rag_pipeline",
            "error": "Pipeline not initialized",
        }

    performance_metrics = rag_pipeline.get_performance_metrics()
    return {
        "status": "healthy",
        "service": "rag_pipeline",
        "index_size": performance_metrics.get("index_size", 0),
        "avg_latency_ms": performance_metrics.get("avg_latency_ms", 0),
        "meets_target": performance_metrics.get("meets_target", True),
    }


@rag_app.post("/index/add")
async def add_to_index(data: dict[str, Any]) -> dict[str, Any]:
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")

    documents_data = data.get("documents")
    if not isinstance(documents_data, list):
        raise HTTPException(status_code=400, detail="documents list required")

    # Convert to Document objects
    documents = []
    for doc_data in documents_data:
        doc_id = doc_data.get("id")
        title = doc_data.get("title", f"Document {doc_id}")
        content = doc_data.get("content")
        source_type = doc_data.get("source_type", "api")
        metadata = doc_data.get("metadata", {})

        if doc_id and content:
            documents.append(
                Document(
                    id=str(doc_id),
                    title=title,
                    content=str(content),
                    source_type=source_type,
                    metadata=metadata,
                )
            )

    if not documents:
        return {
            "success": False,
            "documents_added": 0,
            "error": "No valid documents provided",
        }

    try:
        stats = rag_pipeline.index_documents(documents)
        return {
            "success": True,
            "documents_added": stats["documents_processed"],
            "chunks_created": stats["chunks_created"],
            "vectors_indexed": stats["vectors_indexed"],
            "processing_time_ms": stats["processing_time_ms"],
        }
    except Exception as e:
        logger.exception("Failed to index documents")
        raise HTTPException(status_code=500, detail=str(e))


@rag_app.post("/query")
async def query_rag(data: dict[str, Any]) -> dict[str, Any]:
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")

    query = data.get("query")
    k = data.get("k", 5)  # Number of results to return
    use_cache = data.get("use_cache", True)

    if not query:
        raise HTTPException(status_code=400, detail="query required")

    if not isinstance(k, int) or k < 1 or k > 50:
        raise HTTPException(status_code=400, detail="k must be integer between 1 and 50")

    try:
        results, metrics = await rag_pipeline.retrieve(query, k=k, use_cache=use_cache)

        # Format results for API response
        formatted_results = [
            {
                "chunk_id": result.chunk_id,
                "document_id": result.document_id,
                "text": result.text,
                "score": result.score,
                "retrieval_method": result.retrieval_method,
                "metadata": result.metadata,
            }
            for result in results
        ]

        return {
            "query": query,
            "results": formatted_results,
            "metrics": metrics,
            "total_results": len(formatted_results),
        }

    except Exception as e:
        logger.exception("RAG query failed")
        raise HTTPException(status_code=500, detail=str(e))


@rag_app.get("/metrics/performance")
async def get_rag_metrics() -> dict[str, Any]:
    """Get RAG pipeline performance metrics."""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")

    try:
        performance_metrics = rag_pipeline.get_performance_metrics()
        return performance_metrics
    except Exception as e:
        logger.exception("Failed to get RAG metrics")
        raise HTTPException(status_code=500, detail=str(e))


def run_server(app, host: str, port: int) -> None:
    """Run a FastAPI server."""
    uvicorn.run(app, host=host, port=port, log_level="info")


def main() -> None:
    """Start all API servers."""
    init_digital_twin_db()
    init_evolution_db()
    init_rag_pipeline()

    print("Starting AIVillage API servers...")

    servers = [
        (digital_twin_app, "0.0.0.0", 8080),
        (evolution_app, "0.0.0.0", 8081),
        (rag_app, "0.0.0.0", 8082),
    ]

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for app, host, port in servers:
            print(f"Starting {app.title} on {host}:{port}")
            future = executor.submit(run_server, app, host, port)
            futures.append(future)

        # Wait for all servers
        try:
            for future in futures:
                future.result()
        except KeyboardInterrupt:
            print("\nShutting down servers...")
            sys.exit(0)


if __name__ == "__main__":
    # Check if FastAPI is installed
    try:
        import fastapi
        import uvicorn
    except ImportError:
        print("Installing required packages...")
        import subprocess

        subprocess.run(
            [sys.executable, "-m", "pip", "install", "fastapi", "uvicorn[standard]"],
            check=False,
        )
        print("Packages installed. Please run the script again.")
        sys.exit(1)

    main()
