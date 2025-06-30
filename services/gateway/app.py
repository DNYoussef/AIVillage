"""Atlantis API-Gateway – v0.2.0
• Adds simple in-memory IP rate-limit (100 req / 60 s)
• Health cascade probes Twin
"""

from __future__ import annotations

import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import os

from cachetools import TTLCache

import httpx
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

TWIN_URL = os.getenv("TWIN_URL", "http://twin:8001")
RATE_LIMIT_REQ = int(os.getenv("RATE_LIMIT_REQUESTS", 100))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", 60))
ALLOWED_ORIGINS = os.getenv("GATEWAY_ALLOW_ORIGINS", "http://localhost").split(",")

app = FastAPI(title="Atlantis Gateway", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["Authorization", "Content-Type"],
    allow_credentials=True,
)


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
G_REQS = Counter("gw_requests_total", "Gateway HTTP requests", ["path"])
G_RL = Counter("gw_rate_limited_total", "Requests dropped by rate-limit")
G_LAT = Histogram(
    "gw_latency_seconds",
    "Gateway latency",
    buckets=(0.05, 0.1, 0.2, 0.5, 1, 2, 5),
)


def rate_limit(req: Request):
    ip = req.client.host
    hits = rl_cache.get(ip, 0) + 1
    rl_cache[ip] = hits
    if hits > RATE_LIMIT_REQ:
        G_RL.inc()
        raise HTTPException(429, "Rate limit exceeded")


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
    except Exception as exc:  # pylint: disable=broad-except
        status, r = "degraded", {"error": str(exc)}
    resp = {"gateway": "ok", "twin": status, "details": r}
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
        raise HTTPException(
            exc.response.status_code, f"Twin error {exc.response.text[:100]}"
        )
    except httpx.RequestError as exc:
        raise HTTPException(502, f"Upstream Twin unreachable: {exc}")
