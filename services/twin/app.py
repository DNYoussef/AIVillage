"""Atlantis Twin Service – v0.2.0 (Sprint-2 fusion)
• Maintains per-user conversation state in-memory (proof-of-concept)
• Exposes `/v1/chat`, `/v1/embeddings` stub, `/healthz`, `/metrics`
• Prometheus counters + histograms
• Pydantic request/response models w/ user_id & conversation_id
"""

from __future__ import annotations

import os
import time
import uuid
from datetime import datetime
from typing import Dict, Optional, Any

from cachetools import LRUCache

from fastapi import FastAPI, Depends, HTTPException
from .schemas import ChatRequest, ChatResponse, HealthResponse
from rag_system.graph_explain import explain_path, MAX_HOPS
from pydantic import BaseModel
from core.chat_engine import ChatEngine
from prometheus_client import Counter, Histogram, generate_latest
import uvicorn
import logging


class DummyModel:
    def __init__(self, model_path: str):
        self.model_path = model_path

    def infer(self, prompt: str) -> str:  # noqa: D401
        return f"Echo from {os.path.basename(self.model_path)} › {prompt}"


class TwinSettings:
    _raw_path = os.getenv("TWIN_MODEL_PATH", "models/small-llama.bin")
    model_path: str = (
        _raw_path
        if _raw_path.startswith("/models/") and os.path.exists(_raw_path)
        else "models/small-llama.bin"
    )
    max_context: int = int(os.getenv("TWIN_MAX_CONTEXT", 4096))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


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


REQUESTS = Counter("twin_requests_total", "Total chat requests")
LATENCY = Histogram(
    "twin_chat_latency_seconds", "Chat latency", buckets=(0.1, 0.3, 0.5, 1, 2, 5)
)

# Metrics for the graph explainer
EXPLAIN_REQS = Counter(
    "explain_requests_total",
    "Path explanation requests",
    ["status"],
)
EXPLAIN_LATENCY = Histogram(
    "explain_latency_seconds",
    "Path explanation latency",
    buckets=(0.1, 0.3, 0.5, 1, 2, 5),
)


class TwinAgent:
    def __init__(self, model_path: str):
        self._model = DummyModel(model_path)
        self._conversations: LRUCache = LRUCache(maxsize=1000)

    async def chat(self, req: ChatRequest) -> ChatResponse:
        start = time.perf_counter()

        conv_id = req.conversation_id or str(uuid.uuid4())
        history = self._conversations.get(conv_id)
        if history is None:
            history = []
            self._conversations[conv_id] = history

        history.append(
            {"role": "user", "content": req.message, "ts": datetime.utcnow()}
        )

        recent_msgs = "\n".join(m["content"] for m in history[-6:])
        prompt = f"Context:\n{recent_msgs}\nUser: {req.message}\nAssistant:"

        answer = self._model.infer(prompt)

        history.append(
            {"role": "assistant", "content": answer, "ts": datetime.utcnow()}
        )

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
            timestamp=datetime.utcnow(),
            processing_time_ms=latency_ms,
            calibrated_prob=calibrated,
        )

    def delete_conversation(self, conv_id: str) -> None:
        """Remove a conversation from memory."""
        self._conversations.pop(conv_id, None)

    async def delete_user_data(self, user_id: str):
        """Erase all conversations for `user_id` – used by privacy tests."""
        to_del = [
            cid
            for cid, msgs in list(self._conversations.items())
            if msgs and msgs[0].get("user_id") == user_id
        ]
        for cid in to_del:
            del self._conversations[cid]
        return {"deleted_conversations": len(to_del)}


agent: Optional[TwinAgent] = None


def get_agent() -> TwinAgent:  # noqa: D401
    """Return singleton TwinAgent instance."""
    global agent
    if agent is None:
        agent = TwinAgent(settings.model_path)
    return agent


app = FastAPI(title="Atlantis Twin", version="0.2.0")
_engine = ChatEngine()


@app.post("/v1/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    REQUESTS.inc()
    started = time.time()
    try:
        payload = _engine.process_chat(req.message, req.conversation_id)
    except Exception:
        logger.exception("chat processing failed")
        raise HTTPException(status_code=500, detail="Internal error") from None

    payload["processing_time_ms"] = int((time.time() - started) * 1000)
    return ChatResponse(**payload)


@app.get("/v1/embeddings")
async def embeddings_stub():
    return {"message": "Embeddings endpoint – coming soon"}


@app.get("/healthz", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        version="0.2.0",
        model_loaded=agent is not None,
        timestamp=datetime.utcnow(),
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
        hops = req.hops if req.hops is not None else MAX_HOPS
        data = explain_path(req.src, req.dst, hops)
        if not data["found"]:
            EXPLAIN_REQS.labels(status="error").inc()
            raise HTTPException(status_code=404, detail="Path not found")
        EXPLAIN_REQS.labels(status="success").inc()
        data["processing_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return ExplainResponse(**data)
    finally:
        EXPLAIN_LATENCY.observe(time.perf_counter() - started)


@app.post("/v1/evidence")
async def evidence(pack: Dict[str, Any]):
    logger.debug("Evidence received %s", pack.get("id"))
    return {"status": "ok"}


@app.delete("/v1/user/{user_id}")
async def erase_user(user_id: str, _agent: TwinAgent = Depends(get_agent)):
    return await _agent.delete_user_data(user_id)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=False)
