#!/usr/bin/env python3
"""⚠️ DEVELOPMENT SERVER ONLY ⚠️.

This server is for development and testing purposes only.
- Set AIVILLAGE_DEV_MODE=true to suppress warnings
- For production, use Gateway and Twin microservices in services/
- See docs/architecture.md for production deployment guide

WARNING: This server is NOT production-ready despite being shown in quick start guides.
"""

# isort: skip_file

import html
import mimetypes
import os
from pathlib import Path
import re
import tempfile
import warnings

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware import Middleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

try:  # Optional rag_system components
    from rag_system.core.pipeline import EnhancedRAGPipeline
    from rag_system.utils.logging import setup_logger as get_logger
    from rag_system.graph_explain import MAX_HOPS, explain_path
    from rag_system.tracking.unified_knowledge_tracker import (
        UnifiedKnowledgeTracker,
    )
except ModuleNotFoundError:  # pragma: no cover - optional dependency

    class EnhancedRAGPipeline:  # type: ignore[too-few-public-methods]
        def __init__(self) -> None:
            class DummyRetriever:
                def __init__(self) -> None:
                    self.vector_store = None
                    self.graph_store = None

            self.hybrid_retriever = DummyRetriever()
            self.knowledge_tracker = None

        async def initialize(self) -> None:
            """Fallback pipeline initializer."""
            return

        async def shutdown(self) -> None:
            """Fallback pipeline shutdown."""
            return

        async def process(self, _query: str) -> dict[str, str]:
            """Fallback query processor."""
            return {"answer": ""}

    def get_logger(name: str):  # type: ignore[return-any]
        import logging

        return logging.getLogger(name)

    MAX_HOPS = 0

    def explain_path(*_args, **_kwargs):
        return []

    class UnifiedKnowledgeTracker:  # type: ignore[too-few-public-methods]
        def __init__(self, *_args, **_kwargs) -> None:
            return


from starlette.middleware.base import BaseHTTPMiddleware

# Try to import from servers module, with fallback for reorganized structure
try:
    from servers.common.config import load_config
    from servers.common.middleware import RateLimiter, SecurityMiddleware
except ImportError:
    # Fallback implementations for missing servers module
    def load_config():
        """Fallback config loader when servers module unavailable."""
        return {
            "API_KEY": os.getenv("API_KEY", "dev-key-12345"),
            "HOST": os.getenv("HOST", "localhost"),
            "PORT": int(os.getenv("PORT", "8080")),
            "DEBUG": os.getenv("DEBUG", "true").lower() == "true",
            "MAX_FILE_SIZE": int(os.getenv("MAX_FILE_SIZE", "10485760")),  # 10MB default
            "UPLOAD_DIR": os.getenv("UPLOAD_DIR", "/tmp/uploads"),
            "ALLOWED_EXTENSIONS": [".txt", ".md", ".pdf", ".docx", ".json"],
            "RATE_LIMIT": int(os.getenv("RATE_LIMIT", "100")),
            "CORS_ORIGINS": os.getenv("CORS_ORIGINS", "*").split(","),
            "CHUNK_SIZE": int(os.getenv("CHUNK_SIZE", "8192")),
            "MAX_CHUNKS": int(os.getenv("MAX_CHUNKS", "1000")),
            "ENABLE_RAG": os.getenv("ENABLE_RAG", "true").lower() == "true",
            "RAG_MAX_RESULTS": int(os.getenv("RAG_MAX_RESULTS", "10")),
            "SECURITY_ENABLED": os.getenv("SECURITY_ENABLED", "true").lower() == "true",
            "RATE_LIMIT_REQUESTS": int(os.getenv("RATE_LIMIT_REQUESTS", "100")),
            "RATE_LIMIT_WINDOW": int(os.getenv("RATE_LIMIT_WINDOW", "60")),
            "UPLOAD_PATH": os.getenv("UPLOAD_PATH", "/tmp/uploads"),
            "STATIC_FILES": os.getenv("STATIC_FILES", "true").lower() == "true",
        }

    class RateLimiter:
        """Fallback rate limiter."""

        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, request):
            return request

    class SecurityMiddleware:
        """Fallback security middleware."""

        def __init__(self, *args, **kwargs):
            pass


logger = get_logger(__name__)

# Check if running in dev mode
IS_DEV_MODE = os.getenv("AIVILLAGE_DEV_MODE", "false").lower() == "true"

if not IS_DEV_MODE:
    warnings.warn(
        "WARNING: server.py is for DEVELOPMENT ONLY. "
        "Production services should use the gateway and twin microservices. "
        "Set AIVILLAGE_DEV_MODE=true to suppress this warning.",
        DeprecationWarning,
        stacklevel=2,
    )
    logger.warning(
        "server.py started without AIVILLAGE_DEV_MODE=true. " "This service is deprecated for production use."
    )

# Load configuration and security components
CONFIG = load_config()
API_KEY = CONFIG["API_KEY"]
if not API_KEY:
    logger.warning("API_KEY not set - running without authentication")

MAX_FILE_SIZE = CONFIG["MAX_FILE_SIZE"]
ALLOWED_EXTENSIONS = CONFIG["ALLOWED_EXTENSIONS"]
CHUNK_SIZE = CONFIG["CHUNK_SIZE"]

rate_limiter = RateLimiter(
    max_requests=CONFIG["RATE_LIMIT_REQUESTS"],
    window_seconds=CONFIG["RATE_LIMIT_WINDOW"],
)


class DeprecationMiddleware(BaseHTTPMiddleware):
    """Add deprecation headers to responses."""

    DEPRECATED_ROUTES = {
        "/query": "Use POST /v1/query via Twin service",
        "/upload": "Use POST /v1/upload via Twin service",
        "/status": "Use GET /healthz",
        "/bayes": "Use GET /v1/debug/bayes via Twin service",
        "/logs": "Use GET /v1/debug/logs via Twin service",
        "/v1/explanation": "Use POST /v1/evidence via Twin service",
        "/explain": "Use POST /explain via Twin service",
    }

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Add deprecation header if this is a deprecated route
        path = request.url.path
        if path in self.DEPRECATED_ROUTES:
            response.headers["X-Deprecated"] = self.DEPRECATED_ROUTES[path]
            response.headers["X-Deprecation-Date"] = "2025-02-01"

            if not IS_DEV_MODE:
                logger.warning(
                    f"Deprecated route {path} accessed from {request.client.host}. "
                    f"Migration: {self.DEPRECATED_ROUTES[path]}"
                )

        return response


app = FastAPI(
    middleware=[
        Middleware(SecurityMiddleware, api_key=API_KEY, rate_limiter=rate_limiter),
        Middleware(DeprecationMiddleware),
    ]
)
if Path("ui").exists():
    app.mount("/ui", StaticFiles(directory="ui"), name="ui")
else:  # pragma: no cover - development convenience
    logger.warning("UI directory not found; static interface disabled")

rag_pipeline = EnhancedRAGPipeline()
vector_store = rag_pipeline.hybrid_retriever.vector_store
knowledge_tracker = UnifiedKnowledgeTracker(
    rag_pipeline.hybrid_retriever.vector_store,
    rag_pipeline.hybrid_retriever.graph_store,
)
rag_pipeline.knowledge_tracker = knowledge_tracker


class SecureQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000)

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        if not isinstance(v, str):
            msg = "Query must be a string"
            raise ValueError(msg)
        v = html.escape(v.strip())
        v = "".join(char for char in v if ord(char) >= 32 or char in "\n\t")
        if len(v.strip()) == 0:
            msg = "Query cannot be empty after sanitization"
            raise ValueError(msg)
        dangerous_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"data:.*base64",
            r"vbscript:",
        ]
        v_lower = v.lower()
        for pattern in dangerous_patterns:
            if re.search(pattern, v_lower, re.IGNORECASE | re.DOTALL):
                msg = "Query contains potentially dangerous content"
                raise ValueError(msg)
        return v


class SecureUploadFile(BaseModel):
    """Secure file upload validation."""

    filename: str
    content_type: str
    size: int

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, v: str) -> str:
        if not v or ".." in v or "/" in v or "\\" in v:
            msg = "Invalid filename"
            raise ValueError(msg)
        ext = Path(v).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            msg = f"File type {ext} not allowed"
            raise ValueError(msg)
        return v

    @field_validator("size")
    @classmethod
    def validate_size(cls, v: int) -> int:
        if v > MAX_FILE_SIZE:
            msg = f"File too large: {v} bytes (max {MAX_FILE_SIZE})"
            raise ValueError(msg)
        return v


async def stream_file_safely(file: UploadFile) -> str:
    """Stream file upload safely with size checking."""
    if not file.filename:
        msg = "No filename provided"
        raise ValueError(msg)

    content_type = file.content_type or mimetypes.guess_type(file.filename)[0]
    if not content_type or not content_type.startswith(("text/", "application/")):
        msg = f"Content type {content_type} not allowed"
        raise ValueError(msg)

    total_size = 0
    spooled = tempfile.SpooledTemporaryFile(max_size=MAX_FILE_SIZE)

    try:
        while True:
            chunk = await file.read(CHUNK_SIZE)
            if not chunk:
                break

            total_size += len(chunk)
            if total_size > MAX_FILE_SIZE:
                msg = f"File too large: exceeds {MAX_FILE_SIZE} bytes"
                raise ValueError(msg)

            spooled.write(chunk)

        spooled.seek(0)
        content = spooled.read()
        text = content.decode("utf-8", errors="replace")
        if len(text.strip()) == 0:
            msg = "File appears to be empty"
            raise ValueError(msg)
        return text

    except Exception as e:
        logger.exception(f"File upload failed: {e}")
        msg = f"Failed to process file: {e}"
        raise ValueError(msg) from e


@app.on_event("startup")
async def startup_event() -> None:
    await rag_pipeline.initialize()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await rag_pipeline.shutdown()


@app.post("/query")
async def query_endpoint(request: SecureQueryRequest):
    try:
        result = await rag_pipeline.process(request.query)
        return result
    except Exception as e:
        logger.exception(f"Query processing failed: {e}")
        return JSONResponse(status_code=500, content={"detail": "Query processing failed"})


@app.post("/upload")
async def upload_endpoint(file: UploadFile = File(...)):  # noqa: B008
    """Secure file upload with streaming and validation."""
    try:
        if not file.filename:
            return JSONResponse(status_code=400, content={"detail": "No file selected"})

        text = await stream_file_safely(file)

        await vector_store.add_texts([text])

        return {
            "status": "uploaded",
            "filename": file.filename,
            "size": len(text),
            "message": "File processed successfully",
        }

    except ValueError as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})
    except Exception as e:
        logger.exception(f"Upload endpoint error: {e}")
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/")
async def root():
    return FileResponse("ui/index.html")


@app.get("/status")
async def status_endpoint():
    """Deprecated: Redirect to /healthz."""
    return RedirectResponse(url="/healthz", status_code=307)


@app.get("/healthz")
async def healthz_endpoint():
    """Health check endpoint (replacement for /status)."""
    try:
        status = await rag_pipeline.get_status()
        return {
            "status": "ok" if status.get("healthy", False) else "unhealthy",
            "service": "server",
            "mode": "development" if IS_DEV_MODE else "production-deprecated",
            "details": status,
        }
    except Exception as e:
        logger.exception(f"Health check failed: {e}")
        return JSONResponse(status_code=503, content={"status": "unhealthy", "error": str(e)})


@app.get("/bayes")
async def bayes_endpoint():
    return rag_pipeline.get_bayes_net_snapshot()


@app.get("/logs")
async def logs_endpoint():
    return knowledge_tracker.retrieval_log


@app.get("/v1/explanation")
async def v1_explanation_endpoint(chat_id: str):
    """Return evidence packs associated with a chat id."""
    try:
        # Query actual evidence from knowledge tracker if available
        if hasattr(app.state, "knowledge_tracker") and app.state.knowledge_tracker:
            evidence_packs = app.state.knowledge_tracker.get_evidence_for_chat(chat_id)
            if evidence_packs:
                return evidence_packs

        # Fallback to cached evidence if available
        if hasattr(app.state, "evidence_cache") and chat_id in app.state.evidence_cache:
            return app.state.evidence_cache[chat_id]

        # Return empty evidence list when no evidence is found
        logger.info(f"No evidence found for chat_id: {chat_id}, returning empty evidence list")
        return []
    except Exception as e:
        logger.exception(f"Error retrieving evidence for chat {chat_id}: {e}")
        return []


@app.get("/explain")
async def explain_endpoint(start: str, end: str, hops: int = MAX_HOPS):
    """Return graph explanation path between two nodes."""
    return explain_path(start, end, hops)
