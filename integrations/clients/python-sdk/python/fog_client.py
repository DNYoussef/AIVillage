"""
Refactored AIVillage Fog Computing Client

High-level Python client refactored to follow connascence principles.
Uses composition and dependency injection to reduce coupling between concerns.

Example usage:
    ```python
    from fog_sdk import FogClient

    # Initialize client
    client = FogClient(
        base_url="https://gateway.aivillage.ai",
        api_key="your-api-key",
        namespace="myorg/team"
    )

    # Submit a job
    job = await client.submit_job(
        image="sha256:abc123...",
        args=["python", "script.py"],
        resources={"cpu_cores": 2.0, "memory_mb": 1024}
    )

    # Wait for completion
    result = await client.wait_for_job(job.job_id)
    print(f"Exit code: {result.exit_code}")
    print(f"Output: {result.stdout}")
    ```
"""

import asyncio
from collections.abc import AsyncGenerator
import logging
from typing import Any

from .client_types import (
    CostEstimate,
    FogClientError,
    JobResult,
    MarketplacePrices,
    NamespaceUsage,
    PriceQuote,
    SandboxInfo,
)
from .connection_manager import AuthenticationConfig, ConnectionConfig, HTTPConnectionManager
from .protocol_handlers import JobHandler, MarketplaceHandler, SandboxHandler, UsageHandler

logger = logging.getLogger(__name__)


class FogClient:
    """
    Refactored high-level client for AIVillage fog computing.

    Uses composition to coordinate specialized handlers for different API domains.
    Follows dependency injection and single responsibility principles.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str | None = None,
        namespace: str | None = None,
        timeout: float = 30.0,
    ):
        """
        Initialize fog client with dependency injection.

        Args:
            base_url: Base URL of AIVillage gateway
            api_key: Authentication API key
            namespace: Default namespace for operations
            timeout: Request timeout in seconds
        """
        # Store default namespace
        self._default_namespace = namespace

        # Create configuration objects
        self._connection_config = ConnectionConfig(base_url, timeout)
        self._auth_config = AuthenticationConfig(api_key)

        # Initialize connection manager
        self._connection_manager = HTTPConnectionManager(self._connection_config, self._auth_config)

        # Initialize protocol handlers (will be set up in context manager)
        self._job_handler: JobHandler | None = None
        self._sandbox_handler: SandboxHandler | None = None
        self._usage_handler: UsageHandler | None = None
        self._marketplace_handler: MarketplaceHandler | None = None

    async def __aenter__(self):
        """Async context manager entry"""
        await self._connection_manager.__aenter__()

        # Initialize protocol handlers with the connection manager
        self._job_handler = JobHandler(self._connection_manager)
        self._sandbox_handler = SandboxHandler(self._connection_manager)
        self._usage_handler = UsageHandler(self._connection_manager)
        self._marketplace_handler = MarketplaceHandler(self._connection_manager)

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self._connection_manager.__aexit__(exc_type, exc_val, exc_tb)

        # Clear handler references
        self._job_handler = None
        self._sandbox_handler = None
        self._usage_handler = None
        self._marketplace_handler = None

    # Job Management API

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
        self._ensure_initialized()
        return await self._job_handler.submit_job(
            image=image,
            args=args,
            env=env,
            resources=resources,
            priority=priority,
            namespace=namespace or self._default_namespace,
            labels=labels,
        )

    async def get_job(self, job_id: str) -> JobResult:
        """Get job status and results."""
        self._ensure_initialized()
        return await self._job_handler.get_job(job_id)

    async def cancel_job(self, job_id: str) -> dict[str, str]:
        """Cancel a running or queued job."""
        self._ensure_initialized()
        return await self._job_handler.cancel_job(job_id)

    async def get_job_logs(self, job_id: str) -> str:
        """Get job execution logs."""
        self._ensure_initialized()
        return await self._job_handler.get_job_logs(job_id)

    async def list_jobs(
        self, namespace: str | None = None, status: str | None = None, limit: int = 100
    ) -> list[JobResult]:
        """List jobs with optional filtering."""
        self._ensure_initialized()
        return await self._job_handler.list_jobs(
            namespace=namespace or self._default_namespace, status=status, limit=limit
        )

    async def wait_for_job(self, job_id: str, timeout: float = 300.0, poll_interval: float = 2.0) -> JobResult:
        """Wait for job completion with polling."""
        self._ensure_initialized()
        return await self._job_handler.wait_for_job(job_id, timeout, poll_interval)

    # Sandbox Management API

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
        self._ensure_initialized()
        return await self._sandbox_handler.create_sandbox(
            image=image,
            sandbox_type=sandbox_type,
            resources=resources,
            env=env,
            network_access=network_access,
            namespace=namespace or self._default_namespace,
        )

    async def get_sandbox(self, sandbox_id: str) -> SandboxInfo:
        """Get sandbox status and connection info."""
        self._ensure_initialized()
        return await self._sandbox_handler.get_sandbox(sandbox_id)

    async def exec_in_sandbox(
        self,
        sandbox_id: str,
        command: str,
        args: list[str] | None = None,
        working_dir: str | None = None,
        timeout: int = 30,
    ) -> dict[str, Any]:
        """Execute command in sandbox."""
        self._ensure_initialized()
        return await self._sandbox_handler.exec_in_sandbox(sandbox_id, command, args, working_dir, timeout)

    async def delete_sandbox(self, sandbox_id: str) -> dict[str, str]:
        """Terminate and delete sandbox."""
        self._ensure_initialized()
        return await self._sandbox_handler.delete_sandbox(sandbox_id)

    async def list_sandboxes(
        self, namespace: str | None = None, status: str | None = None, limit: int = 100
    ) -> list[SandboxInfo]:
        """List sandboxes with optional filtering."""
        self._ensure_initialized()
        return await self._sandbox_handler.list_sandboxes(
            namespace=namespace or self._default_namespace, status=status, limit=limit
        )

    # Usage and Billing API

    async def get_usage(self, namespace: str | None = None, period: str = "day") -> list[NamespaceUsage]:
        """Get usage metrics and costs."""
        self._ensure_initialized()
        return await self._usage_handler.get_usage(namespace=namespace or self._default_namespace, period=period)

    async def get_pricing(self) -> dict[str, Any]:
        """Get current pricing information."""
        self._ensure_initialized()
        return await self._usage_handler.get_pricing()

    async def get_quotas(self, namespace: str | None = None) -> list[dict[str, Any]]:
        """Get namespace quotas and current usage."""
        self._ensure_initialized()
        return await self._usage_handler.get_quotas(namespace=namespace or self._default_namespace)

    # Marketplace API

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
        self._ensure_initialized()
        return await self._marketplace_handler.get_price_quote(
            cpu_cores,
            memory_gb,
            estimated_duration_hours,
            disk_gb,
            bid_type,
            pricing_tier,
            min_trust_score,
            max_latency_ms,
        )

    async def get_marketplace_prices(self) -> MarketplacePrices:
        """Get current marketplace pricing information."""
        self._ensure_initialized()
        return await self._marketplace_handler.get_marketplace_prices()

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
        self._ensure_initialized()
        return await self._marketplace_handler.submit_bid(
            image,
            cpu_cores,
            memory_gb,
            max_price,
            args,
            env,
            disk_gb,
            estimated_duration_hours,
            bid_type,
            pricing_tier,
            min_trust_score,
            namespace or self._default_namespace,
        )

    async def get_bid_status(self, bid_id: str) -> dict[str, Any]:
        """Get marketplace bid status."""
        self._ensure_initialized()
        return await self._marketplace_handler.get_bid_status(bid_id)

    async def cancel_bid(self, bid_id: str) -> dict[str, Any]:
        """Cancel marketplace bid."""
        self._ensure_initialized()
        return await self._marketplace_handler.cancel_bid(bid_id)

    async def get_marketplace_status(self) -> dict[str, Any]:
        """Get marketplace status and statistics."""
        self._ensure_initialized()
        return await self._marketplace_handler.get_marketplace_status()

    # High-level convenience methods

    async def run_job(
        self,
        image: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        resources: dict[str, Any] | None = None,
        timeout: float = 300.0,
        namespace: str | None = None,
    ) -> JobResult:
        """Submit job and wait for completion (convenience method)."""
        # Submit job
        job = await self.submit_job(image=image, args=args, env=env, resources=resources, namespace=namespace)

        # Wait for completion
        return await self.wait_for_job(job.job_id, timeout=timeout)

    async def stream_logs(self, job_id: str, follow: bool = True) -> AsyncGenerator[str, None]:
        """Stream job logs (placeholder - WebSocket implementation needed)."""
        # Implementation required: Implement WebSocket log streaming
        # For now, just return current logs
        logs = await self.get_job_logs(job_id)

        for line in logs.split("\n"):
            if line.strip():
                yield line

    async def estimate_job_cost(
        self,
        image: str,
        cpu_cores: float = 1.0,
        memory_gb: float = 1.0,
        estimated_duration_hours: float = 1.0,
        bid_type: str = "spot",
        pricing_tier: str = "basic",
        include_recommendations: bool = True,
    ) -> CostEstimate:
        """Estimate cost for job execution."""
        # Get price quote first
        quote = await self.get_price_quote(
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            estimated_duration_hours=estimated_duration_hours,
            bid_type=bid_type,
            pricing_tier=pricing_tier,
        )

        if not quote.available:
            raise FogClientError(f"No resources available: {quote.reason}")

        # Calculate confidence based on price volatility
        confidence = 1.0 - min(0.8, quote.price_volatility or 0.0)

        # Create cost breakdown
        cost_breakdown = {
            "cpu_cost": quote.avg_price * 0.7 if quote.avg_price else 0.0,
            "memory_cost": quote.avg_price * 0.2 if quote.avg_price else 0.0,
            "disk_cost": quote.avg_price * 0.05 if quote.avg_price else 0.0,
            "network_cost": quote.avg_price * 0.05 if quote.avg_price else 0.0,
        }

        # Price range estimate
        price_range = {
            "min": quote.min_price or 0.0,
            "max": quote.max_price or 0.0,
            "market": quote.market_estimate or 0.0,
        }

        # Generate recommendations
        recommendations = []
        if include_recommendations:
            recommendations = self._generate_cost_recommendations(quote, bid_type)

        return CostEstimate(
            estimated_cost=quote.avg_price or 0.0,
            cost_breakdown=cost_breakdown,
            pricing_tier=pricing_tier,
            bid_type=bid_type,
            duration_hours=estimated_duration_hours,
            confidence_level=confidence,
            price_range=price_range,
            recommendations=recommendations,
        )

    async def run_job_with_budget(
        self,
        image: str,
        max_price: float,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        resources: dict[str, Any] | None = None,
        bid_type: str = "spot",
        timeout: float = 300.0,
        namespace: str | None = None,
    ) -> JobResult:
        """Submit job with budget constraint and wait for completion."""
        # Parse resources
        cpu_cores = resources.get("cpu_cores", 1.0) if resources else 1.0
        memory_gb = resources.get("memory_gb", 1.0) if resources else 1.0

        # Submit marketplace bid
        bid_result = await self.submit_bid(
            image=image,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            max_price=max_price,
            args=args,
            env=env,
            bid_type=bid_type,
            namespace=namespace,
        )

        # If bid was immediately matched to a job, wait for completion
        if "job_id" in bid_result:
            return await self.wait_for_job(bid_result["job_id"], timeout=timeout)

        # Otherwise, monitor bid status until job is created
        bid_id = bid_result["bid_id"]

        for _ in range(int(timeout / 5)):  # Check every 5 seconds
            bid_status = await self.get_bid_status(bid_id)

            if bid_status["status"] == "matched" and "job_id" in bid_status:
                return await self.wait_for_job(bid_status["job_id"], timeout=timeout)
            elif bid_status["status"] in ["failed", "cancelled"]:
                raise FogClientError(f"Bid failed: {bid_status.get('message', 'Unknown error')}")

            await asyncio.sleep(5)

        # Timeout waiting for bid to be matched
        await self.cancel_bid(bid_id)
        raise FogClientError(f"Bid not matched within {timeout} seconds")

    # Private helper methods

    def _ensure_initialized(self) -> None:
        """Ensure the client is properly initialized."""
        if not self._connection_manager.is_connected():
            raise FogClientError("Client must be used as async context manager")

    def _generate_cost_recommendations(self, quote: PriceQuote, bid_type: str) -> list[str]:
        """Generate cost optimization recommendations."""
        recommendations = []

        if quote.price_volatility and quote.price_volatility > 0.3:
            recommendations.append("High price volatility detected - consider on-demand pricing for predictable costs")

        if bid_type == "on_demand" and quote.current_spot_rate and quote.current_on_demand_rate:
            potential_savings = (
                (quote.current_on_demand_rate - quote.current_spot_rate) / quote.current_on_demand_rate * 100
            )
            if potential_savings > 20:
                recommendations.append(f"Spot pricing could save ~{potential_savings:.0f}% vs on-demand")

        if quote.available_providers and quote.available_providers < 3:
            recommendations.append("Limited providers available - consider flexible timing or lower requirements")

        if quote.suggested_max_price and quote.avg_price:
            if quote.suggested_max_price > quote.avg_price * 1.2:
                recommendations.append(
                    f"Consider max_price of ${quote.suggested_max_price:.4f} for better availability"
                )

        return recommendations


# Backward compatibility exports
__all__ = [
    "FogClient",
    "JobResult",
    "SandboxInfo",
    "NamespaceUsage",
    "PriceQuote",
    "MarketplacePrices",
    "CostEstimate",
    "FogClientError",
    "JobNotFoundError",
    "SandboxNotFoundError",
    "QuotaExceededError",
    "AuthenticationError",
]
