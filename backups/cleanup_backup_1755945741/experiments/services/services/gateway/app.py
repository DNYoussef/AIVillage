"""Atlantis API-Gateway – v0.2.0
• Adds simple in-memory IP rate-limit (100 req / 60 s)
• Health cascade probes Twin.
"""

from __future__ import annotations

import logging
import os

# Import unified error handling and configuration
import sys
import time

from cachetools import TTLCache
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import httpx
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from services.core.config import get_config
from services.core.service_error_handler import gateway_error_handler, network_error, rate_limit_error

# Load unified configuration
config = get_config()
gateway_config = config.gateway

TWIN_URL = config.external_services.get("twin_url", "http://twin:8001")
RATE_LIMIT_REQ = config.security.rate_limit_requests
RATE_LIMIT_WINDOW = config.security.rate_limit_window
ALLOWED_ORIGINS = gateway_config.cors_origins

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gateway")

app = FastAPI(title="Atlantis Gateway", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["Authorization", "Content-Type"],
    allow_credentials=True,
)

# Add global exception handlers
from fastapi.exceptions import RequestValidationError

app.add_exception_handler(Exception, gateway_error_handler.http_exception_handler)
app.add_exception_handler(RequestValidationError, gateway_error_handler.http_exception_handler)


class SecurityHeaders(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        resp = await call_next(request)
        resp.headers.update(
            {
                "Strict-Transport-Security": "max-age=63072000; includeSubDomains",
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY",
            }
        )
        return resp


app.add_middleware(SecurityHeaders)

rl_cache: TTLCache = TTLCache(maxsize=10000, ttl=RATE_LIMIT_WINDOW)

# ─── Prometheus -------------------------------------------------------------
G_REQS = Counter("gw_requests_total", "Gateway HTTP requests", ["path"], registry=REGISTRY)
G_RL = Counter("gw_rate_limited_total", "Requests dropped by rate-limit", registry=REGISTRY)
G_LAT = Histogram(
    "gw_latency_seconds",
    "Gateway latency",
    buckets=(0.05, 0.1, 0.2, 0.5, 1, 2, 5),
    registry=REGISTRY,
)


def rate_limit(req: Request) -> None:
    ip = "unknown"
    if req.client and req.client.host:
        ip = req.client.host
    hits = rl_cache.get(ip, 0) + 1
    rl_cache[ip] = hits
    if hits > RATE_LIMIT_REQ:
        G_RL.inc()
        msg = "Rate limit exceeded"
        raise rate_limit_error(
            msg,
            details={
                "ip": ip,
                "limit": RATE_LIMIT_REQ,
                "window": RATE_LIMIT_WINDOW,
                "current_hits": hits,
            },
        )


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = str(time.time() - start)
    G_LAT.observe(time.time() - start)
    return response


@app.middleware("http")
async def prom_count(request: Request, call_next):
    G_REQS.labels(path=request.url.path).inc()
    return await call_next(request)


@app.get("/healthz")
async def health():
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{TWIN_URL}/healthz")
        status = "ok" if r.status_code == 200 else "degraded"
        details = {"status_code": r.status_code, "response": r.text}
    except Exception as exc:
        # Create structured network error
        network_exc = network_error(
            f"Twin service health check failed: {exc}",
            details={"twin_url": TWIN_URL, "error_type": type(exc).__name__},
        )

        # Log the error
        logger.exception(
            "Health check failed",
            extra={
                "error": network_exc.to_dict(),
                "twin_url": TWIN_URL,
            },
        )

        status = "degraded"
        details = {"error": str(exc), "type": type(exc).__name__}

    resp = {"gateway": "ok", "twin": status, "details": details}
    from fastapi.responses import JSONResponse

    response = JSONResponse(resp)
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/metrics")
async def metrics():
    from fastapi.responses import Response

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/chat")
async def proxy_chat(req: Request, _=Depends(rate_limit)):
    body = await req.body()
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(
                f"{TWIN_URL}/v1/chat",
                content=body,
                headers={"content-type": "application/json"},
            )
        r.raise_for_status()
        json_body = r.json()
        headers = {
            "Cache-Control": "no-store",
            "Pragma": "no-cache",
        }
        from fastapi.responses import JSONResponse

        return JSONResponse(content=json_body, headers=headers)
    except httpx.HTTPStatusError as exc:
        msg = f"Twin service returned HTTP {exc.response.status_code}"
        raise network_error(
            msg,
            details={
                "status_code": exc.response.status_code,
                "response_text": exc.response.text[:200],
                "twin_url": f"{TWIN_URL}/v1/chat",
            },
        )
    except httpx.RequestError as exc:
        msg = f"Twin service unreachable: {exc}"
        raise network_error(
            msg,
            details={
                "error_type": type(exc).__name__,
                "twin_url": f"{TWIN_URL}/v1/chat",
            },
        )
