"""
Protocol Handlers for Fog API Operations

Specialized handlers for different API domains (jobs, sandboxes, marketplace, etc.).
Each handler focuses on one API domain following single responsibility principle.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Protocol

from .client_types import FogClientError, JobResult, MarketplacePrices, NamespaceUsage, PriceQuote, SandboxInfo

logger = logging.getLogger(__name__)


class HTTPClientProtocol(Protocol):
    """Protocol for HTTP client dependency - weak connascence."""

    async def request(
        self, method: str, path: str, json_data: dict | None = None, params: dict | None = None
    ) -> dict[str, Any]:
        """Make HTTP request."""
        ...


class JobHandler:
    """
    Handler for job-related API operations.

    Focused on job lifecycle management (submit, monitor, cancel).
    """

    def __init__(self, http_client: HTTPClientProtocol):
        self._http_client = http_client

    async def submit_job(
        self,
        image: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        resources: dict[str, Any] | None = None,
        priority: str = "B",
        namespace: str | None = None,
        labels: dict[str, str] | None = None,
    ) -> JobResult:
        """Submit a new fog job for execution."""
        job_spec = {
            "namespace": namespace,
            "image": image,
            "priority": priority,
            "args": args or [],
            "env": env or {},
            "resources": resources or {},
            "labels": labels or {},
        }

        if not job_spec["namespace"]:
            raise FogClientError("Namespace must be provided")

        response = await self._http_client.request("POST", "/v1/fog/jobs", json_data=job_spec)
        return JobResult(job_id=response["job_id"], status=response["status"])

    async def get_job(self, job_id: str) -> JobResult:
        """Get job status and results."""
        response = await self._http_client.request("GET", f"/v1/fog/jobs/{job_id}")

        return JobResult(
            job_id=response["job_id"],
            status=response["status"],
            exit_code=response.get("exit_code"),
            cpu_seconds_used=response.get("cpu_seconds_used", 0.0),
            memory_mb_peak=response.get("memory_mb_peak", 0),
            duration_ms=response.get("execution_latency_ms"),
            error_message=response.get("error_message"),
        )

    async def cancel_job(self, job_id: str) -> dict[str, str]:
        """Cancel a running or queued job."""
        return await self._http_client.request("DELETE", f"/v1/fog/jobs/{job_id}")

    async def get_job_logs(self, job_id: str) -> str:
        """Get job execution logs."""
        response = await self._http_client.request("GET", f"/v1/fog/jobs/{job_id}/logs")
        return response.get("logs", "")

    async def list_jobs(
        self, namespace: str | None = None, status: str | None = None, limit: int = 100
    ) -> list[JobResult]:
        """List jobs with optional filtering."""
        params = {"limit": limit}
        if namespace:
            params["namespace"] = namespace
        if status:
            params["status"] = status

        response = await self._http_client.request("GET", "/v1/fog/jobs", params=params)

        return [
            JobResult(
                job_id=job["job_id"],
                status=job["status"],
                exit_code=job.get("exit_code"),
                cpu_seconds_used=job.get("cpu_seconds_used", 0.0),
                memory_mb_peak=job.get("memory_mb_peak", 0),
                duration_ms=job.get("execution_latency_ms"),
                error_message=job.get("error_message"),
            )
            for job in response
        ]

    async def wait_for_job(self, job_id: str, timeout: float = 300.0, poll_interval: float = 2.0) -> JobResult:
        """Wait for job completion with polling."""
        start_time = asyncio.get_event_loop().time()

        while True:
            job = await self.get_job(job_id)

            if job.status in ["completed", "failed", "cancelled"]:
                return job

            # Check timeout
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= timeout:
                raise FogClientError(f"Timeout waiting for job {job_id}")

            await asyncio.sleep(poll_interval)


class SandboxHandler:
    """
    Handler for sandbox-related API operations.

    Focused on interactive environment management.
    """

    def __init__(self, http_client: HTTPClientProtocol):
        self._http_client = http_client

    async def create_sandbox(
        self,
        image: str,
        sandbox_type: str = "interactive",
        resources: dict[str, Any] | None = None,
        env: dict[str, str] | None = None,
        network_access: bool = False,
        namespace: str | None = None,
    ) -> SandboxInfo:
        """Create interactive sandbox environment."""
        sandbox_spec = {
            "namespace": namespace,
            "type": sandbox_type,
            "image": image,
            "resources": resources or {},
            "env": env or {},
            "network_access": network_access,
        }

        if not sandbox_spec["namespace"]:
            raise FogClientError("Namespace must be provided")

        response = await self._http_client.request("POST", "/v1/fog/sandboxes", json_data=sandbox_spec)

        return SandboxInfo(
            sandbox_id=response["sandbox_id"],
            status=response["status"],
            connection_url=response.get("connection_url"),
            ssh_command=response.get("ssh_command"),
            created_at=response["created_at"],
        )

    async def get_sandbox(self, sandbox_id: str) -> SandboxInfo:
        """Get sandbox status and connection info."""
        response = await self._http_client.request("GET", f"/v1/fog/sandboxes/{sandbox_id}")

        return SandboxInfo(
            sandbox_id=response["sandbox_id"],
            status=response["status"],
            connection_url=response.get("connection_url"),
            ssh_command=response.get("ssh_command"),
            created_at=response["created_at"],
            idle_minutes=response.get("idle_minutes", 0),
        )

    async def exec_in_sandbox(
        self,
        sandbox_id: str,
        command: str,
        args: list[str] | None = None,
        working_dir: str | None = None,
        timeout: int = 30,
    ) -> dict[str, Any]:
        """Execute command in sandbox."""
        exec_spec = {"command": command, "args": args or [], "timeout_seconds": timeout}

        if working_dir:
            exec_spec["working_dir"] = working_dir

        return await self._http_client.request("POST", f"/v1/fog/sandboxes/{sandbox_id}/exec", json_data=exec_spec)

    async def delete_sandbox(self, sandbox_id: str) -> dict[str, str]:
        """Terminate and delete sandbox."""
        return await self._http_client.request("DELETE", f"/v1/fog/sandboxes/{sandbox_id}")

    async def list_sandboxes(
        self, namespace: str | None = None, status: str | None = None, limit: int = 100
    ) -> list[SandboxInfo]:
        """List sandboxes with optional filtering."""
        params = {"limit": limit}
        if namespace:
            params["namespace"] = namespace
        if status:
            params["status"] = status

        response = await self._http_client.request("GET", "/v1/fog/sandboxes", params=params)

        return [
            SandboxInfo(
                sandbox_id=sandbox["sandbox_id"],
                status=sandbox["status"],
                connection_url=sandbox.get("connection_url"),
                ssh_command=sandbox.get("ssh_command"),
                created_at=sandbox["created_at"],
                idle_minutes=sandbox.get("idle_minutes", 0),
            )
            for sandbox in response
        ]


class UsageHandler:
    """
    Handler for usage and billing API operations.

    Focused on cost tracking and quota management.
    """

    def __init__(self, http_client: HTTPClientProtocol):
        self._http_client = http_client

    async def get_usage(self, namespace: str | None = None, period: str = "day") -> list[NamespaceUsage]:
        """Get usage metrics and costs."""
        params = {"period": period}
        if namespace:
            params["namespace"] = namespace

        response = await self._http_client.request("GET", "/v1/fog/usage", params=params)

        return [
            NamespaceUsage(
                namespace=usage["namespace"],
                total_cost=usage["total_cost"],
                cpu_seconds=usage["cpu_seconds"],
                memory_mb_hours=usage["memory_mb_hours"],
                job_executions=usage["job_executions"],
                period_start=usage["period_start"],
                period_end=usage["period_end"],
            )
            for usage in response
        ]

    async def get_pricing(self) -> dict[str, Any]:
        """Get current pricing information."""
        return await self._http_client.request("GET", "/v1/fog/usage/pricing")

    async def get_quotas(self, namespace: str | None = None) -> list[dict[str, Any]]:
        """Get namespace quotas and current usage."""
        params = {}
        if namespace:
            params["namespace"] = namespace

        return await self._http_client.request("GET", "/v1/fog/usage/quotas", params=params)


class MarketplaceHandler:
    """
    Handler for marketplace and pricing API operations.

    Focused on dynamic pricing and resource bidding.
    """

    def __init__(self, http_client: HTTPClientProtocol):
        self._http_client = http_client

    async def get_price_quote(
        self,
        cpu_cores: float,
        memory_gb: float,
        estimated_duration_hours: float = 1.0,
        disk_gb: float = 2.0,
        bid_type: str = "spot",
        pricing_tier: str = "basic",
        min_trust_score: float = 0.3,
        max_latency_ms: float = 500.0,
    ) -> PriceQuote:
        """Get price quote for resource requirements."""
        quote_request = {
            "cpu_cores": cpu_cores,
            "memory_gb": memory_gb,
            "disk_gb": disk_gb,
            "estimated_duration_hours": estimated_duration_hours,
            "bid_type": bid_type,
            "pricing_tier": pricing_tier,
            "min_trust_score": min_trust_score,
            "max_latency_ms": max_latency_ms,
        }

        response = await self._http_client.request("POST", "/v1/fog/quotes", json=quote_request)

        return PriceQuote(
            available=response.get("available", False),
            min_price=response.get("quote", {}).get("min_price"),
            max_price=response.get("quote", {}).get("max_price"),
            avg_price=response.get("quote", {}).get("avg_price"),
            market_estimate=response.get("quote", {}).get("market_estimate"),
            current_spot_rate=response.get("market_conditions", {}).get("current_spot_rate"),
            current_on_demand_rate=response.get("market_conditions", {}).get("current_on_demand_rate"),
            price_volatility=response.get("market_conditions", {}).get("price_volatility"),
            available_providers=response.get("market_conditions", {}).get("available_providers"),
            suggested_max_price=response.get("recommendations", {}).get("suggested_max_price"),
            estimated_wait_time_minutes=response.get("recommendations", {}).get("estimated_wait_time_minutes"),
            reason=response.get("reason"),
        )

    async def get_marketplace_prices(self) -> MarketplacePrices:
        """Get current marketplace pricing information."""
        response = await self._http_client.request("GET", "/v1/fog/prices")

        return MarketplacePrices(
            currency=response.get("currency", "USD"),
            last_updated=datetime.fromisoformat(response["last_updated"]),
            spot_price_per_cpu_hour=response["spot_price_per_cpu_hour"],
            on_demand_price_per_cpu_hour=response["on_demand_price_per_cpu_hour"],
            price_volatility=response["price_volatility"],
            utilization_rate=response["market_conditions"]["utilization_rate"],
            demand_supply_ratio=response["market_conditions"]["demand_supply_ratio"],
            available_providers=response["market_conditions"]["available_providers"],
            pricing_tiers=response["pricing_tiers"],
        )

    async def submit_bid(
        self,
        image: str,
        cpu_cores: float,
        memory_gb: float,
        max_price: float,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        disk_gb: float = 2.0,
        estimated_duration_hours: float = 1.0,
        bid_type: str = "spot",
        pricing_tier: str = "basic",
        min_trust_score: float = 0.3,
        namespace: str | None = None,
    ) -> dict[str, Any]:
        """Submit marketplace bid for resource execution."""
        bid_request = {
            "namespace": namespace,
            "image": image,
            "args": args or [],
            "env": env or {},
            "resources": {"cpu_cores": cpu_cores, "memory_gb": memory_gb, "disk_gb": disk_gb},
            "estimated_duration_hours": estimated_duration_hours,
            "max_price": max_price,
            "bid_type": bid_type,
            "pricing_tier": pricing_tier,
            "min_trust_score": min_trust_score,
        }

        if not bid_request["namespace"]:
            raise ValueError("Namespace must be provided")

        return await self._http_client.request("POST", "/v1/fog/marketplace/bids", json=bid_request)

    async def get_bid_status(self, bid_id: str) -> dict[str, Any]:
        """Get marketplace bid status."""
        return await self._http_client.request("GET", f"/v1/fog/marketplace/bids/{bid_id}")

    async def cancel_bid(self, bid_id: str) -> dict[str, Any]:
        """Cancel marketplace bid."""
        return await self._http_client.request("DELETE", f"/v1/fog/marketplace/bids/{bid_id}")

    async def get_marketplace_status(self) -> dict[str, Any]:
        """Get marketplace status and statistics."""
        return await self._http_client.request("GET", "/v1/fog/marketplace/status")
