"""Credits API - REST endpoints for credits ledger operations."""

from datetime import datetime, timezone
import logging

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from .credits_ledger import (
    CreditsConfig,
    CreditsLedger,
)

logger = logging.getLogger(__name__)

# Prometheus metrics
CREDITS_REQUESTS = Counter(
    "credits_requests_total", "Credits API requests", ["endpoint", "method"]
)
CREDITS_ERRORS = Counter(
    "credits_errors_total", "Credits API errors", ["endpoint", "error_type"]
)
CREDITS_LATENCY = Histogram(
    "credits_latency_seconds", "Credits API latency", ["endpoint"]
)

# Pydantic models for API


class CreateUserRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64, regex="^[a-zA-Z0-9_-]+$")
    node_id: str | None = Field(None, max_length=128)


class TransferRequest(BaseModel):
    from_username: str = Field(..., min_length=3, max_length=64)
    to_username: str = Field(..., min_length=3, max_length=64)
    amount: int = Field(..., gt=0, description="Amount in credits (positive integer)")


class EarnRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    scrape_timestamp: datetime = Field(..., description="Prometheus scrape timestamp")
    uptime_seconds: int = Field(..., ge=0, description="Node uptime in seconds")
    flops: int = Field(..., ge=0, description="Floating point operations")
    bandwidth_bytes: int = Field(..., ge=0, description="Bandwidth used in bytes")


class BalanceResponseAPI(BaseModel):
    user_id: int
    username: str
    balance: int
    last_updated: datetime


class TransactionResponseAPI(BaseModel):
    id: int
    from_user: str
    to_user: str
    amount: int
    burn_amount: int
    net_amount: int
    transaction_type: str
    status: str
    created_at: datetime
    completed_at: datetime | None = None


class EarningResponseAPI(BaseModel):
    id: int
    user_id: int
    scrape_timestamp: datetime
    uptime_seconds: int
    flops: int
    bandwidth_bytes: int
    credits_earned: int
    created_at: datetime


class ErrorResponse(BaseModel):
    error: str
    message: str
    timestamp: datetime


# FastAPI app
app = FastAPI(
    title="Credits Ledger API",
    description="Fixed-supply shell currency with Prometheus-based earning",
    version="1.0.0",
)

# Global ledger instance
ledger: CreditsLedger | None = None


def get_ledger() -> CreditsLedger:
    """Dependency to get ledger instance."""
    global ledger
    if ledger is None:
        config = CreditsConfig()
        ledger = CreditsLedger(config)
    return ledger


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    CREDITS_ERRORS.labels(endpoint=request.url.path, error_type="validation").inc()
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error="validation_error",
            message=str(exc),
            timestamp=datetime.now(timezone.utc),
        ).dict(),
    )


@app.exception_handler(Exception)
async def general_error_handler(request, exc):
    CREDITS_ERRORS.labels(endpoint=request.url.path, error_type="internal").inc()
    logger.error(f"Unexpected error in {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_error",
            message="An unexpected error occurred",
            timestamp=datetime.now(timezone.utc),
        ).dict(),
    )


# API Endpoints


@app.post("/users", response_model=dict, status_code=201)
async def create_user(
    request: CreateUserRequest, ledger: CreditsLedger = Depends(get_ledger)
):
    """Create a new user with wallet."""
    CREDITS_REQUESTS.labels(endpoint="/users", method="POST").inc()

    with CREDITS_LATENCY.labels(endpoint="/users").time():
        try:
            user = ledger.create_user(request.username, request.node_id)
            return {
                "user_id": user.id,
                "username": user.username,
                "node_id": user.node_id,
                "created_at": user.created_at,
                "message": "User created successfully",
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))


@app.get("/balance/{username}", response_model=BalanceResponseAPI)
async def get_balance(username: str, ledger: CreditsLedger = Depends(get_ledger)):
    """Get user balance."""
    CREDITS_REQUESTS.labels(endpoint="/balance", method="GET").inc()

    with CREDITS_LATENCY.labels(endpoint="/balance").time():
        try:
            balance = ledger.get_balance(username)
            return BalanceResponseAPI(
                user_id=balance.user_id,
                username=balance.username,
                balance=balance.balance,
                last_updated=balance.last_updated,
            )
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))


@app.post("/transfer", response_model=TransactionResponseAPI)
async def transfer_credits(
    request: TransferRequest, ledger: CreditsLedger = Depends(get_ledger)
):
    """Transfer credits between users with 1% burn."""
    CREDITS_REQUESTS.labels(endpoint="/transfer", method="POST").inc()

    with CREDITS_LATENCY.labels(endpoint="/transfer").time():
        try:
            transaction = ledger.transfer(
                request.from_username, request.to_username, request.amount
            )
            return TransactionResponseAPI(
                id=transaction.id,
                from_user=transaction.from_user,
                to_user=transaction.to_user,
                amount=transaction.amount,
                burn_amount=transaction.burn_amount,
                net_amount=transaction.net_amount,
                transaction_type=transaction.transaction_type,
                status=transaction.status,
                created_at=transaction.created_at,
                completed_at=transaction.completed_at,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))


@app.post("/earn", response_model=EarningResponseAPI)
async def earn_credits(
    request: EarnRequest, ledger: CreditsLedger = Depends(get_ledger)
):
    """Mint credits based on Prometheus metrics. Idempotent by scrape timestamp."""
    CREDITS_REQUESTS.labels(endpoint="/earn", method="POST").inc()

    with CREDITS_LATENCY.labels(endpoint="/earn").time():
        try:
            earning = ledger.earn_credits(
                request.username,
                request.scrape_timestamp,
                request.uptime_seconds,
                request.flops,
                request.bandwidth_bytes,
            )
            return EarningResponseAPI(
                id=earning.id,
                user_id=earning.user_id,
                scrape_timestamp=earning.scrape_timestamp,
                uptime_seconds=earning.uptime_seconds,
                flops=earning.flops,
                bandwidth_bytes=earning.bandwidth_bytes,
                credits_earned=earning.credits_earned,
                created_at=earning.created_at,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))


@app.get("/transactions/{username}", response_model=list[TransactionResponseAPI])
async def get_transactions(
    username: str,
    limit: int = Query(default=100, ge=1, le=1000),
    ledger: CreditsLedger = Depends(get_ledger),
):
    """Get transaction history for user."""
    CREDITS_REQUESTS.labels(endpoint="/transactions", method="GET").inc()

    with CREDITS_LATENCY.labels(endpoint="/transactions").time():
        try:
            transactions = ledger.get_transactions(username, limit)
            return [
                TransactionResponseAPI(
                    id=tx.id,
                    from_user=tx.from_user,
                    to_user=tx.to_user,
                    amount=tx.amount,
                    burn_amount=tx.burn_amount,
                    net_amount=tx.net_amount,
                    transaction_type=tx.transaction_type,
                    status=tx.status,
                    created_at=tx.created_at,
                    completed_at=tx.completed_at,
                )
                for tx in transactions
            ]
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))


@app.get("/supply")
async def get_total_supply(ledger: CreditsLedger = Depends(get_ledger)):
    """Get total credits in circulation."""
    CREDITS_REQUESTS.labels(endpoint="/supply", method="GET").inc()

    with CREDITS_LATENCY.labels(endpoint="/supply").time():
        total_supply = ledger.get_total_supply()
        return {
            "total_supply": total_supply,
            "max_supply": ledger.config.fixed_supply,
            "burn_rate": ledger.config.burn_rate,
            "timestamp": datetime.now(timezone.utc),
        }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "credits-api",
        "timestamp": datetime.now(timezone.utc),
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest()


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize ledger on startup."""
    global ledger
    config = CreditsConfig()
    ledger = CreditsLedger(config)

    # Create tables if they don't exist
    try:
        ledger.create_tables()
        logger.info("Credits ledger initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize credits ledger: {e}")
        raise


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
