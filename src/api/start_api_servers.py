#!/usr/bin/env python3
"""Start all API servers for AIVillage integration"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Digital Twin API
digital_twin_app = FastAPI(title="Digital Twin API")

@digital_twin_app.get("/health/twin")
async def health_twin():
    return {"status": "healthy", "service": "digital_twin", "timestamp": str(Path.cwd())}

@digital_twin_app.post("/profile/create")
async def create_profile(data: Dict[str, Any]):
    return {"success": True, "profile_id": "test_profile_001", "data": data}

@digital_twin_app.get("/profile/{profile_id}")
async def get_profile(profile_id: str):
    return {"profile_id": profile_id, "learning_style": "visual", "knowledge_level": "intermediate"}

# Evolution Metrics API
evolution_app = FastAPI(title="Evolution Metrics API")

@evolution_app.get("/health/evolution")
async def health_evolution():
    return {"status": "healthy", "service": "evolution_metrics", "active_rounds": 4}

@evolution_app.post("/metrics/record")
async def record_metrics(data: Dict[str, Any]):
    return {"success": True, "metrics_recorded": len(data.get("metrics", []))}

@evolution_app.get("/metrics/latest")
async def get_latest_metrics():
    return {
        "round": 4,
        "avg_fitness": 0.75,
        "best_fitness": 0.92,
        "population_size": 50
    }

# RAG Pipeline API
rag_app = FastAPI(title="RAG Pipeline API")

@rag_app.get("/health/rag")
async def health_rag():
    return {"status": "healthy", "service": "rag_pipeline", "index_size": 0}

@rag_app.post("/query")
async def query_rag(data: Dict[str, Any]):
    query = data.get("query", "")
    return {
        "query": query,
        "results": [
            {"doc_id": "doc_001", "score": 0.95, "content": "Sample result"},
            {"doc_id": "doc_002", "score": 0.87, "content": "Another result"}
        ],
        "latency_ms": 12.5
    }

@rag_app.post("/index/add")
async def add_to_index(data: Dict[str, Any]):
    return {"success": True, "documents_added": 1}

def run_server(app, host: str, port: int):
    """Run a FastAPI server"""
    uvicorn.run(app, host=host, port=port, log_level="info")

def main():
    """Start all API servers"""
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
