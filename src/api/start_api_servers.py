#!/usr/bin/env python3
"""Start all API servers for AIVillage integration with real implementations."""

import logging
import os
import sqlite3
import sys
import hashlib
from pathlib import Path
from typing import Any, Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, HTTPException
import uvicorn
from concurrent.futures import ThreadPoolExecutor

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

from core.security.digital_twin_encryption import (
    DigitalTwinEncryption,
    DigitalTwinEncryptionError,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
async def health_twin() -> Dict[str, Any]:
    try:
        with sqlite3.connect(DIGITAL_TWIN_DB) as conn:
            conn.execute("SELECT 1")
        return {"status": "healthy", "service": "digital_twin"}
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Digital Twin health check failed")
        raise HTTPException(status_code=500, detail=str(exc))


@digital_twin_app.post("/profile/create")
async def create_profile(data: Dict[str, Any]) -> Dict[str, Any]:
    user_id = data.get("user_id")
    learning_style = data.get("learning_style")
    preferred_difficulty = data.get("preferred_difficulty", "medium")
    if not user_id or not learning_style:
        raise HTTPException(status_code=400, detail="user_id and learning_style required")

    profile_id = hashlib.sha256(f"{user_id}-{os.urandom(4)}".encode()).hexdigest()[:16]
    user_hash = hashlib.sha256(user_id.encode()).hexdigest()

    try:
        encrypted_ls = encryption.encrypt_sensitive_field(
            learning_style, "learning_style"
        )
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
async def get_profile(profile_id: str) -> Dict[str, Any]:
    with sqlite3.connect(DIGITAL_TWIN_DB) as conn:
        row = conn.execute(
            "SELECT preferred_difficulty, learning_style_encrypted FROM profiles WHERE profile_id=?",
            (profile_id,),
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Profile not found")

    try:
        learning_style = encryption.decrypt_sensitive_field(
            row[1], "learning_style"
        )
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
async def health_evolution() -> Dict[str, Any]:
    try:
        with sqlite3.connect(EVOLUTION_DB) as conn:
            conn.execute("SELECT 1")
        return {"status": "healthy", "service": "evolution_metrics"}
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Evolution Metrics health check failed")
        raise HTTPException(status_code=500, detail=str(exc))


@evolution_app.post("/metrics/record")
async def record_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
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
async def get_latest_metrics() -> Dict[str, Any]:
    try:
        with sqlite3.connect(EVOLUTION_DB) as conn:
            rows = conn.execute(
                "SELECT metric_name, metric_value FROM metrics ORDER BY recorded_at DESC"
            ).fetchall()
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to fetch metrics")
        raise HTTPException(status_code=500, detail=str(exc))

    latest: Dict[str, float] = {}
    for name, value in rows:
        if name not in latest:
            latest[name] = float(value)

    return latest


# ---------------------------------------------------------------------------
# RAG Pipeline API with TF-IDF search
# ---------------------------------------------------------------------------

rag_app = FastAPI(title="RAG Pipeline API")

RAG_INDEX_PATH = Path("data/rag_index.pkl")
RAG_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

vectorizer = TfidfVectorizer()
doc_ids: List[str] = []
doc_texts: List[str] = []
doc_matrix = None


def load_rag_index() -> None:
    global vectorizer, doc_ids, doc_texts, doc_matrix
    if RAG_INDEX_PATH.exists():
        data = joblib.load(RAG_INDEX_PATH)
        vectorizer = data["vectorizer"]
        doc_ids = data["doc_ids"]
        doc_texts = data["doc_texts"]
        doc_matrix = data["doc_matrix"]


def save_rag_index() -> None:
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "doc_ids": doc_ids,
            "doc_texts": doc_texts,
            "doc_matrix": doc_matrix,
        },
        RAG_INDEX_PATH,
    )


@rag_app.get("/health/rag")
async def health_rag() -> Dict[str, Any]:
    return {"status": "healthy", "service": "rag_pipeline", "index_size": len(doc_ids)}


@rag_app.post("/index/add")
async def add_to_index(data: Dict[str, Any]) -> Dict[str, Any]:
    documents = data.get("documents")
    if not isinstance(documents, list):
        raise HTTPException(status_code=400, detail="documents list required")

    added = 0
    for doc in documents:
        doc_id = doc.get("id")
        content = doc.get("content")
        if doc_id and content:
            doc_ids.append(str(doc_id))
            doc_texts.append(str(content))
            added += 1

    if added:
        global doc_matrix
        doc_matrix = vectorizer.fit_transform(doc_texts)
        save_rag_index()

    return {"success": True, "documents_added": added}


@rag_app.post("/query")
async def query_rag(data: Dict[str, Any]) -> Dict[str, Any]:
    query = data.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="query required")
    if doc_matrix is None or not doc_texts:
        raise HTTPException(status_code=404, detail="index empty")

    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, doc_matrix)[0]
    top_idx = scores.argsort()[::-1][:5]
    results = [
        {
            "doc_id": doc_ids[i],
            "score": float(scores[i]),
            "content": doc_texts[i],
        }
        for i in top_idx
    ]

    return {"query": query, "results": results}

def run_server(app, host: str, port: int):
    """Run a FastAPI server"""
    uvicorn.run(app, host=host, port=port, log_level="info")

def main():
    """Start all API servers"""
    init_digital_twin_db()
    init_evolution_db()
    load_rag_index()

    print("Starting AIVillage API servers...")
    
    servers = [
        (digital_twin_app, "0.0.0.0", 8080),
        (evolution_app, "0.0.0.0", 8081),
        (rag_app, "0.0.0.0", 8082)
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
        subprocess.run([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn[standard]"])
        print("Packages installed. Please run the script again.")
        sys.exit(1)
    
    main()
