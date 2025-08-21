"""Atlantis Twin Service – v0.2.0 (Sprint-2 fusion)
• Maintains per-user conversation state in-memory (proof-of-concept)
• Exposes `/v1/chat`, `/v1/embeddings` stub, `/healthz`, `/metrics`
• Prometheus counters + histograms
• Pydantic request/response models w/ user_id & conversation_id.
"""

from __future__ import annotations

import logging
import os

# Import unified error handling and configuration
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))
import time
import uuid
from typing import Any

import uvicorn
from cachetools import LRUCache
from core.chat_engine import ChatEngine
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from prometheus_client import REGISTRY, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

# Add the production RAG system to the path
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src" / "production" / "rag"))
from rag_system.graph_explain import MAX_HOPS, explain_path

from .schemas import ChatRequest, ChatResponse, HealthResponse

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from services.core.config import get_config
from services.core.service_error_handler import (
    ErrorCategory,
    ErrorSeverity,
    create_service_error,
    resource_error,
    twin_error_handler,
    validation_error,
)


class DummyModel:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    def infer(self, prompt: str) -> str:
        return f"Echo from {os.path.basename(self.model_path)} › {prompt}"


# Load unified configuration
config = get_config()
twin_config = config.twin


class TwinSettings:
    model_path: str = os.getenv("TWIN_MODEL_PATH", "models/small-llama.bin")
    max_context: int = config.ai.max_context_length
    log_level: str = twin_config.log_level


settings = TwinSettings()

logger = logging.getLogger(__name__)

CALIBRATION_ENABLED = os.getenv("CALIBRATION_ENABLED", "0") == "1"
if CALIBRATION_ENABLED:
    try:
        from calibration.conformal import ConformalCalibrator

        _calibrator = ConformalCalibrator.load_default()
        logger.info("Calibration enabled")
    except Exception:  # pragma: no cover - rarely triggered
        logger.exception("Failed to load calibrator")
        CALIBRATION_ENABLED = False
        _calibrator = None
else:  # pragma: no cover - disabled feature
    _calibrator = None


REQUESTS = Counter("twin_requests_total", "Total chat requests", registry=REGISTRY)
LATENCY = Histogram(
    "twin_chat_latency_seconds",
    "Chat latency",
    buckets=(0.1, 0.3, 0.5, 1, 2, 5),
    registry=REGISTRY,
)

# Metrics for the graph explainer
EXPLAIN_REQS = Counter("explain_requests_total", "Path explanation requests", ["status"], registry=REGISTRY)
EXPLAIN_LATENCY = Histogram(
    "explain_latency_seconds",
    "Path explanation latency",
    buckets=(0.1, 0.3, 0.5, 1, 2, 5),
    registry=REGISTRY,
)


class TwinAgent:
    def __init__(self, model_path: str) -> None:
        self._model = DummyModel(model_path)
        self._conversations: LRUCache = LRUCache(maxsize=1000)

    async def chat(self, req: ChatRequest) -> ChatResponse:
        start = time.perf_counter()

        conv_id = req.conversation_id or str(uuid.uuid4())
        history = self._conversations.get(conv_id)
        if history is None:
            history = []
            self._conversations[conv_id] = history

        history.append({"role": "user", "content": req.message, "ts": datetime.now(timezone.utc)})

        recent_msgs = "\n".join(m["content"] for m in history[-6:])
        prompt = f"Context:\n{recent_msgs}\nUser: {req.message}\nAssistant:"

        answer = self._model.infer(prompt)

        history.append({"role": "assistant", "content": answer, "ts": datetime.now(timezone.utc)})

        raw_prob = 0.5
        calibrated = None
        if CALIBRATION_ENABLED and _calibrator is not None:
            try:
                calibrated = _calibrator.calibrate(raw_prob)
            except Exception as exc:  # pragma: no cover - unexpected
                logger.warning("Calibration error: %s", exc)

        latency_ms = (time.perf_counter() - start) * 1000
        return ChatResponse(
            response=answer,
            conversation_id=conv_id,
            timestamp=datetime.now(timezone.utc),
            processing_time_ms=latency_ms,
            calibrated_prob=calibrated,
        )

    def delete_conversation(self, conv_id: str) -> None:
        """Remove a conversation from memory."""
        self._conversations.pop(conv_id, None)

    async def delete_user_data(self, user_id: str):
        """Erase all conversations for `user_id` – used by privacy tests."""
        to_del = [cid for cid, msgs in list(self._conversations.items()) if msgs and msgs[0].get("user_id") == user_id]
        for cid in to_del:
            del self._conversations[cid]
        return {"deleted_conversations": len(to_del)}


agent: TwinAgent | None = None


def get_agent() -> TwinAgent:
    """Return singleton TwinAgent instance."""
    global agent
    if agent is None:
        agent = TwinAgent(settings.model_path)
    return agent


app = FastAPI(title="Atlantis Twin", version="0.2.0")
_engine = ChatEngine()

# Import new architecture components
from services.core.business_logic import ServiceBusinessLogicFactory
from services.core.http_adapters import HTTPAdapterFactory

# Create service instances using clean architecture
business_logic_factory = ServiceBusinessLogicFactory(
    {
        "service_name": twin_config.name,
        "version": twin_config.version,
        "max_message_length": 5000,
        "max_file_size": twin_config.max_request_size,
        "dependencies": {"chat_engine": _engine},
    }
)

adapter_factory = HTTPAdapterFactory(business_logic_factory)
chat_adapter = adapter_factory.create_chat_adapter()
query_adapter = adapter_factory.create_query_adapter()
upload_adapter = adapter_factory.create_upload_adapter()
health_adapter = adapter_factory.create_health_adapter()

# Add global exception handlers
from fastapi.exceptions import RequestValidationError

app.add_exception_handler(Exception, twin_error_handler.http_exception_handler)
app.add_exception_handler(RequestValidationError, twin_error_handler.http_exception_handler)


@app.post("/v1/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    REQUESTS.inc()
    started = time.time()

    # Use the new architecture
    payload = await chat_adapter.handle_chat_request({"message": req.message, "conversation_id": req.conversation_id})

    payload["processing_time_ms"] = int((time.time() - started) * 1000)
    return ChatResponse(**payload)


@app.get("/v1/embeddings")
async def embeddings_stub():
    return {"message": "Embeddings endpoint – coming soon"}


@app.get("/healthz", response_model=HealthResponse)
async def health():
    # Use the new architecture
    payload = await health_adapter.handle_health_check()
    return HealthResponse(
        status=payload["status"],
        version=payload["version"],
        model_loaded=agent is not None,
        timestamp=datetime.fromisoformat(payload["timestamp"]),
    )


@app.get("/metrics")
async def metrics():
    return generate_latest(), 200, {"Content-Type": "text/plain; version=0.0.4"}


class ExplainRequest(BaseModel):
    src: str
    dst: str
    hops: int | None = None


class ExplainResponse(BaseModel):
    nodes: list
    edges: list
    hops: int
    found: bool
    processing_ms: float


@app.post("/explain", response_model=ExplainResponse)
async def explain_endpoint(req: ExplainRequest):
    started = time.perf_counter()
    try:
        # Validate request
        if not req.src or not req.dst:
            msg = "Source and destination nodes are required"
            raise validation_error(
                msg,
                details={"src_provided": bool(req.src), "dst_provided": bool(req.dst)},
            )

        if req.src == req.dst:
            msg = "Source and destination cannot be the same"
            raise validation_error(msg, details={"node": req.src})

        hops = req.hops if req.hops is not None else MAX_HOPS

        if hops <= 0 or hops > MAX_HOPS:
            msg = f"Hops must be between 1 and {MAX_HOPS}"
            raise validation_error(
                msg,
                details={"hops": hops, "max_hops": MAX_HOPS},
            )

        data = explain_path(req.src, req.dst, hops)

        if not data["found"]:
            EXPLAIN_REQS.labels(status="error").inc()
            msg = "Path not found between nodes"
            raise resource_error(
                msg,
                details={
                    "source": req.src,
                    "destination": req.dst,
                    "max_hops": hops,
                },
            )

        EXPLAIN_REQS.labels(status="success").inc()
        data["processing_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return ExplainResponse(**data)

    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise

        # Handle unexpected errors
        service_error = create_service_error(
            message=f"Path explanation failed: {exc!s}",
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.ERROR,
            operation="path_explanation",
            details={
                "source": req.src,
                "destination": req.dst,
                "hops": req.hops,
                "error_type": type(exc).__name__,
            },
        )

        logger.exception(
            "Path explanation failed",
            extra={
                "error": service_error.to_dict(),
                "request": {"src": req.src, "dst": req.dst, "hops": req.hops},
            },
        )

        raise service_error

    finally:
        EXPLAIN_LATENCY.observe(time.perf_counter() - started)


@app.post("/v1/evidence")
async def evidence(pack: dict[str, Any]):
    logger.debug("Evidence received %s", pack.get("id"))
    return {"status": "ok"}


@app.delete("/v1/user/{user_id}")
async def erase_user(user_id: str, _agent: TwinAgent = Depends(get_agent)):
    return await _agent.delete_user_data(user_id)


# New migrated routes from server.py


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000)


class QueryResponse(BaseModel):
    response: str
    chunks: list[dict[str, Any]] = []
    processing_time_ms: int


@app.post("/v1/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    """Process RAG query - migrated from server.py /query endpoint."""
    REQUESTS.inc()
    started = time.time()

    # Use the new architecture
    payload = await query_adapter.handle_query_request({"query": req.query, "limit": 10})

    LATENCY.observe(time.time() - started)
    return QueryResponse(
        response=payload.get("results", [{}])[0].get("text", ""),
        chunks=payload.get("results", []),
        processing_time_ms=payload.get("processing_time_ms", 0),
    )


class UploadResponse(BaseModel):
    status: str
    filename: str
    size: int
    message: str


@app.post("/v1/upload", response_model=UploadResponse)
async def upload_endpoint(file: UploadFile = File(...)):
    """Upload file to vector store - migrated from server.py /upload endpoint."""
    # Use the new architecture
    payload = await upload_adapter.handle_upload_request(file)
    return UploadResponse(**payload)


# Debug endpoints (non-production)


@app.get("/v1/debug/bayes")
async def debug_bayes_endpoint():
    """Get Bayes network snapshot - migrated from server.py /bayes endpoint."""
    # TODO: Integrate with actual Bayes network
    return {
        "message": "Bayes network debug endpoint",
        "nodes": [],
        "edges": [],
        "timestamp": datetime.now(timezone.utc),
    }


@app.get("/v1/debug/logs")
async def debug_logs_endpoint():
    """Get knowledge tracker logs - migrated from server.py /logs endpoint."""
    # TODO: Integrate with actual knowledge tracker
    return {
        "message": "Knowledge tracker logs endpoint",
        "logs": [],
        "timestamp": datetime.now(timezone.utc),
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=False)
