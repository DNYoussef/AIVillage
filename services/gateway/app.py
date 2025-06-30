"""Atlantis API-Gateway – v0.2.0
• Adds simple in-memory IP rate-limit (100 req / 60 s)
• Health cascade probes Twin
"""

from __future__ import annotations

import time
import os

from cachetools import TTLCache

import httpx
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware

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

rl_cache: TTLCache = TTLCache(maxsize=10000, ttl=RATE_LIMIT_WINDOW)


def rate_limit(req: Request):
    ip = req.client.host
    hits = rl_cache.get(ip, 0) + 1
    rl_cache[ip] = hits
    if hits > RATE_LIMIT_REQ:
        raise HTTPException(429, "Rate limit exceeded")


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = str(time.time() - start)
    return response


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
        return r.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            exc.response.status_code, f"Twin error {exc.response.text[:100]}"
        )
    except httpx.RequestError as exc:
        raise HTTPException(502, f"Upstream Twin unreachable: {exc}")
