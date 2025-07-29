#!/usr/bin/env python3
"""⚠️ DEVELOPMENT SERVER ONLY ⚠️

This server is for development and testing purposes only.
- Set AIVILLAGE_DEV_MODE=true to suppress warnings
- For production, use Gateway and Twin microservices in services/
- See docs/architecture.md for production deployment guide

WARNING: This server is NOT production-ready despite being shown in quick start guides.
"""

from collections import defaultdict
import html
import mimetypes
import os
from pathlib import Path
import re
import tempfile
import time
import warnings

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware import Middleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError, validator
from starlette.middleware.base import BaseHTTPMiddleware

from core.evidence import Chunk, ConfidenceTier, EvidencePack
from rag_system.core.pipeline import EnhancedRAGPipeline
from rag_system.graph_explain import MAX_HOPS, explain_path
from rag_system.tracking.unified_knowledge_tracker import UnifiedKnowledgeTracker
from rag_system.utils.logging import setup_logger as get_logger

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
        "server.py started without AIVILLAGE_DEV_MODE=true. "
        "This service is deprecated for production use."
    )

# Validate required environment variables
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    logger.warning("API_KEY not set - running without authentication")


# Rate limiting configuration - should use Redis in production
class RateLimiter:
    """Simple in-memory rate limiter - use Redis for production."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
        logger.warning("Using in-memory rate limiter - consider Redis for production")

    def is_allowed(self, client_id: str) -> bool:
        """Check if client is within rate limits."""
        now = time.time()
        self.requests[client_id] = [
            req_time
            for req_time in self.requests[client_id]
            if now - req_time < self.window_seconds
        ]
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        self.requests[client_id].append(now)
        return True


rate_limiter = RateLimiter(
    max_requests=CONFIG["RATE_LIMIT_REQUESTS"],
    window_seconds=CONFIG["RATE_LIMIT_WINDOW"],
)

# Configuration - should be moved to config file
CONFIG = {
    "MAX_FILE_SIZE": int(os.getenv("MAX_FILE_SIZE", 50 * 1024 * 1024)),  # 50MB default
    "ALLOWED_EXTENSIONS": {".txt", ".md", ".pdf", ".docx", ".html"},
    "CHUNK_SIZE": int(os.getenv("CHUNK_SIZE", 8192)),  # 8KB default
    "RATE_LIMIT_REQUESTS": int(os.getenv("RATE_LIMIT_REQUESTS", 100)),
    "RATE_LIMIT_WINDOW": int(os.getenv("RATE_LIMIT_WINDOW", 60)),
}

MAX_FILE_SIZE = CONFIG["MAX_FILE_SIZE"]
ALLOWED_EXTENSIONS = CONFIG["ALLOWED_EXTENSIONS"]
CHUNK_SIZE = CONFIG["CHUNK_SIZE"]


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


class SecurityMiddleware(BaseHTTPMiddleware):
    """Enhanced security middleware with rate limiting."""

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        client_id = request.headers.get("x-forwarded-for", client_ip)

        if request.url.path.startswith(("/query", "/upload")):
            if not rate_limiter.is_allowed(client_id):
                return JSONResponse(
                    status_code=429, content={"detail": "Rate limit exceeded"}
                )

        if API_KEY and request.url.path not in (
            "/",
            "/ui",
            "/ui/index.html",
            "/status",
        ):
            key = request.headers.get("x-api-key")
            if key != API_KEY:
                return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

        try:
            response = await call_next(request)
        except ValidationError:
            logger.warning(f"Validation error from {client_id}")
            return JSONResponse(
                status_code=400, content={"detail": "Invalid request format"}
            )
        except Exception as e:
            logger.error(f"Server error from {client_id}: {e}")
            return JSONResponse(
                status_code=500, content={"detail": "Internal server error"}
            )

        return response


app = FastAPI(
    middleware=[Middleware(SecurityMiddleware), Middleware(DeprecationMiddleware)]
)
app.mount("/ui", StaticFiles(directory="ui"), name="ui")

rag_pipeline = EnhancedRAGPipeline()
vector_store = rag_pipeline.hybrid_retriever.vector_store
knowledge_tracker = UnifiedKnowledgeTracker(
    rag_pipeline.hybrid_retriever.vector_store,
    rag_pipeline.hybrid_retriever.graph_store,
)
rag_pipeline.knowledge_tracker = knowledge_tracker


class SecureQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000)

    @validator("query")
    def validate_query(cls, v):
        if not isinstance(v, str):
            raise ValueError("Query must be a string")
        v = html.escape(v.strip())
        v = "".join(char for char in v if ord(char) >= 32 or char in "\n\t")
        if len(v.strip()) == 0:
            raise ValueError("Query cannot be empty after sanitization")
        dangerous_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"data:.*base64",
            r"vbscript:",
        ]
        v_lower = v.lower()
        for pattern in dangerous_patterns:
            if re.search(pattern, v_lower, re.IGNORECASE | re.DOTALL):
                raise ValueError("Query contains potentially dangerous content")
        return v


class SecureUploadFile(BaseModel):
    """Secure file upload validation."""

    filename: str
    content_type: str
    size: int

    @validator("filename")
    def validate_filename(cls, v):
        if not v or ".." in v or "/" in v or "\\" in v:
            raise ValueError("Invalid filename")
        ext = Path(v).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(f"File type {ext} not allowed")
        return v

    @validator("size")
    def validate_size(cls, v):
        if v > MAX_FILE_SIZE:
            raise ValueError(f"File too large: {v} bytes (max {MAX_FILE_SIZE})")
        return v


async def stream_file_safely(file: UploadFile) -> str:
    """Stream file upload safely with size checking."""
    if not file.filename:
        raise ValueError("No filename provided")

    content_type = file.content_type or mimetypes.guess_type(file.filename)[0]
    if not content_type or not content_type.startswith(("text/", "application/")):
        raise ValueError(f"Content type {content_type} not allowed")

    total_size = 0
    spooled = tempfile.SpooledTemporaryFile(max_size=MAX_FILE_SIZE)

    try:
        while True:
            chunk = await file.read(CHUNK_SIZE)
            if not chunk:
                break

            total_size += len(chunk)
            if total_size > MAX_FILE_SIZE:
                raise ValueError(f"File too large: exceeds {MAX_FILE_SIZE} bytes")

            spooled.write(chunk)

        spooled.seek(0)
        content = spooled.read()
        text = content.decode("utf-8", errors="replace")
        if len(text.strip()) == 0:
            raise ValueError("File appears to be empty")
        return text

    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise ValueError(f"Failed to process file: {e}")


@app.on_event("startup")
async def startup_event():
    await rag_pipeline.initialize()


@app.on_event("shutdown")
async def shutdown_event():
    await rag_pipeline.shutdown()


@app.post("/query")
async def query_endpoint(request: SecureQueryRequest):
    try:
        result = await rag_pipeline.process(request.query)
        return result
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        return JSONResponse(
            status_code=500, content={"detail": "Query processing failed"}
        )


@app.post("/upload")
async def upload_endpoint(file: UploadFile = File(...)):
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
        logger.error(f"Upload endpoint error: {e}")
        return JSONResponse(
            status_code=500, content={"detail": "Internal server error"}
        )


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
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503, content={"status": "unhealthy", "error": str(e)}
        )


@app.get("/bayes")
async def bayes_endpoint():
    return rag_pipeline.get_bayes_net_snapshot()


@app.get("/logs")
async def logs_endpoint():
    return knowledge_tracker.retrieval_log


@app.get("/v1/explanation")
async def v1_explanation_endpoint(chat_id: str):
    """Return evidence packs associated with a chat id."""
    # TODO: Replace with actual evidence pack retrieval logic
    logger.warning(f"Using placeholder data for chat_id: {chat_id}")

    try:
        # In production, this should query actual evidence from knowledge tracker
        # evidence_packs = knowledge_tracker.get_evidence_for_chat(chat_id)

        # Placeholder implementation
        placeholder_packs = [
            EvidencePack(
                query=f"query_for_{chat_id}",
                chunks=[
                    Chunk(
                        id=f"c1_{chat_id}",
                        text="No evidence available - implement evidence retrieval",
                        score=0.1,
                        source_uri="placeholder://none",
                    )
                ],
                proto_confidence=0.1,
                confidence_tier=ConfidenceTier.LOW,
            )
        ]
        return [pack.dict() for pack in placeholder_packs]
    except Exception as e:
        logger.error(f"Error retrieving evidence for chat {chat_id}: {e}")
        return []


@app.get("/explain")
async def explain_endpoint(start: str, end: str, hops: int = MAX_HOPS):
    """Return graph explanation path between two nodes."""
    return explain_path(start, end, hops)
