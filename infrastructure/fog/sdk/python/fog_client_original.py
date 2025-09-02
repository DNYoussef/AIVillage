"""
AIVillage Fog Computing Client

High-level Python client for interacting with the AIVillage fog network.
Provides simple interfaces for job submission, sandbox management, and usage tracking.

Example usage:
    ```python
    from fog_sdk import FogClient

    # Initialize client
    client = FogClient(
        base_url="https://gateway.aivillage.ai",
        api_key="your-api-key",  # pragma: allowlist secret
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
from datetime import datetime
import logging
from typing import Any

import aiohttp
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class FogClientError(Exception):
    """Base exception for fog client errors"""

    pass


class JobNotFoundError(FogClientError):
    """Job not found error"""

    pass


class SandboxNotFoundError(FogClientError):
    """Sandbox not found error"""

    pass


class QuotaExceededError(FogClientError):
    """Namespace quota exceeded error"""

    pass


class AuthenticationError(FogClientError):
    """Authentication/authorization error"""

    pass


class JobResult(BaseModel):
    """Job execution result"""

    job_id: str
    status: str
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    cpu_seconds_used: float = 0.0
    memory_mb_peak: int = 0
    duration_ms: float | None = None
    error_message: str | None = None


class SandboxInfo(BaseModel):
    """Sandbox information"""

    sandbox_id: str
    status: str
    connection_url: str | None = None
    ssh_command: str | None = None
    created_at: datetime
    idle_minutes: int = 0


class NamespaceUsage(BaseModel):
    """Namespace usage summary"""

    namespace: str
    total_cost: float
    cpu_seconds: float
    memory_mb_hours: float
    job_executions: int
    period_start: datetime
    period_end: datetime


class PriceQuote(BaseModel):
    """Price quote for resource requirements"""

    available: bool
    min_price: float | None = None
    max_price: float | None = None
    avg_price: float | None = None
    market_estimate: float | None = None
    current_spot_rate: float | None = None
    current_on_demand_rate: float | None = None
    price_volatility: float | None = None
    available_providers: int | None = None
    suggested_max_price: float | None = None
    estimated_wait_time_minutes: int | None = None
    reason: str | None = None


class MarketplacePrices(BaseModel):
    """Current marketplace pricing information"""

    currency: str = "USD"
    last_updated: datetime
    spot_price_per_cpu_hour: float
    on_demand_price_per_cpu_hour: float
    price_volatility: float
    utilization_rate: float
    demand_supply_ratio: float
    available_providers: int
    pricing_tiers: dict[str, dict[str, Any]]


class CostEstimate(BaseModel):
    """Cost estimate for job execution"""

    estimated_cost: float
    cost_breakdown: dict[str, float]
    pricing_tier: str
    bid_type: str
    duration_hours: float
    confidence_level: float
    price_range: dict[str, float]
    recommendations: list[str]


class FogClient:
    """High-level client for AIVillage fog computing"""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str | None = None,
        namespace: str | None = None,
        timeout: float = 30.0,
    ):
        """
        Initialize fog client

        Args:
            base_url: Base URL of AIVillage gateway
            api_key: Authentication API key
            namespace: Default namespace for operations
            timeout: Request timeout in seconds
        """

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.namespace = namespace
        self.timeout = aiohttp.ClientTimeout(total=timeout)

        # Headers for authentication
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    async def __aenter__(self):
        """Async context manager entry"""
        self._session = aiohttp.ClientSession(timeout=self.timeout, headers=self.headers)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if hasattr(self, "_session"):
            await self._session.close()

    async def _request(
        self, method: str, path: str, json_data: dict | None = None, params: dict | None = None
    ) -> dict[str, Any]:
        """Make authenticated API request"""

        if not hasattr(self, "_session"):
            raise FogClientError("Client must be used as async context manager")

        url = f"{self.base_url}{path}"

        try:
            async with self._session.request(method, url, json=json_data, params=params) as response:
                # Handle common HTTP errors
                if response.status == 401:
                    raise AuthenticationError("Invalid API key or unauthorized")
                elif response.status == 404:
                    if "/jobs/" in path:
                        raise JobNotFoundError("Job not found")
                    elif "/sandboxes/" in path:
                        raise SandboxNotFoundError("Sandbox not found")
                    else:
                        raise FogClientError("Resource not found")
                elif response.status == 429:
                    raise QuotaExceededError("Rate limit or quota exceeded")
                elif response.status >= 400:
                    error_text = await response.text()
                    raise FogClientError(f"API error {response.status}: {error_text}")

                return await response.json()

        except aiohttp.ClientError as e:
            raise FogClientError(f"Request failed: {e}")

    # Job Management

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
        """
        Submit a new fog job for execution

        Args:
            image: Container image or WASI module (CAS hash or registry ref)
            args: Command arguments
            env: Environment variables
            resources: Resource requirements (cpu_cores, memory_mb, disk_mb, max_duration_s)
            priority: Priority class ("B", "A", "S")
            namespace: Target namespace (uses default if not specified)
            labels: User labels for job

        Returns:
            JobResult with job ID and initial status
        """

        job_spec = {
            "namespace": namespace or self.namespace,
            "image": image,
            "priority": priority,
            "args": args or [],
            "env": env or {},
            "resources": resources or {},
            "labels": labels or {},
        }

        if not job_spec["namespace"]:
            raise FogClientError("Namespace must be provided")

        response = await self._request("POST", "/v1/fog/jobs", json_data=job_spec)

        return JobResult(job_id=response["job_id"], status=response["status"])

    async def get_job(self, job_id: str) -> JobResult:
        """Get job status and results"""

        response = await self._request("GET", f"/v1/fog/jobs/{job_id}")

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
        """Cancel a running or queued job"""
        return await self._request("DELETE", f"/v1/fog/jobs/{job_id}")

    async def get_job_logs(self, job_id: str) -> str:
        """Get job execution logs"""
        response = await self._request("GET", f"/v1/fog/jobs/{job_id}/logs")
        return response.get("logs", "")

    async def list_jobs(
        self, namespace: str | None = None, status: str | None = None, limit: int = 100
    ) -> list[JobResult]:
        """List jobs with optional filtering"""

        params = {"limit": limit}
        if namespace:
            params["namespace"] = namespace
        if status:
            params["status"] = status

        response = await self._request("GET", "/v1/fog/jobs", params=params)

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
        """
        Wait for job completion with polling

        Args:
            job_id: Job to wait for
            timeout: Maximum wait time in seconds
            poll_interval: Polling interval in seconds

        Returns:
            JobResult when job completes
        """

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

    # Sandbox Management

    async def create_sandbox(
        self,
        image: str,
        sandbox_type: str = "interactive",
        resources: dict[str, Any] | None = None,
        env: dict[str, str] | None = None,
        network_access: bool = False,
        namespace: str | None = None,
    ) -> SandboxInfo:
        """
        Create interactive sandbox environment

        Args:
            image: Base image for sandbox
            sandbox_type: Type of sandbox ("interactive", "batch", "persistent")
            resources: Resource allocation
            env: Environment variables
            network_access: Allow network access
            namespace: Target namespace

        Returns:
            SandboxInfo with connection details
        """

        sandbox_spec = {
            "namespace": namespace or self.namespace,
            "type": sandbox_type,
            "image": image,
            "resources": resources or {},
            "env": env or {},
            "network_access": network_access,
        }

        if not sandbox_spec["namespace"]:
            raise FogClientError("Namespace must be provided")

        response = await self._request("POST", "/v1/fog/sandboxes", json_data=sandbox_spec)

        return SandboxInfo(
            sandbox_id=response["sandbox_id"],
            status=response["status"],
            connection_url=response.get("connection_url"),
            ssh_command=response.get("ssh_command"),
            created_at=response["created_at"],
        )

    async def get_sandbox(self, sandbox_id: str) -> SandboxInfo:
        """Get sandbox status and connection info"""

        response = await self._request("GET", f"/v1/fog/sandboxes/{sandbox_id}")

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
        """Execute command in sandbox"""

        exec_spec = {"command": command, "args": args or [], "timeout_seconds": timeout}

        if working_dir:
            exec_spec["working_dir"] = working_dir

        return await self._request("POST", f"/v1/fog/sandboxes/{sandbox_id}/exec", json_data=exec_spec)

    async def delete_sandbox(self, sandbox_id: str) -> dict[str, str]:
        """Terminate and delete sandbox"""
        return await self._request("DELETE", f"/v1/fog/sandboxes/{sandbox_id}")

    async def list_sandboxes(
        self, namespace: str | None = None, status: str | None = None, limit: int = 100
    ) -> list[SandboxInfo]:
        """List sandboxes with optional filtering"""

        params = {"limit": limit}
        if namespace:
            params["namespace"] = namespace
        if status:
            params["status"] = status

        response = await self._request("GET", "/v1/fog/sandboxes", params=params)

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

    # Usage and Billing

    async def get_usage(self, namespace: str | None = None, period: str = "day") -> list[NamespaceUsage]:
        """Get usage metrics and costs"""

        params = {"period": period}
        if namespace:
            params["namespace"] = namespace

        response = await self._request("GET", "/v1/fog/usage", params=params)

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
        """Get current pricing information"""
        return await self._request("GET", "/v1/fog/usage/pricing")

    async def get_quotas(self, namespace: str | None = None) -> list[dict[str, Any]]:
        """Get namespace quotas and current usage"""

        params = {}
        if namespace:
            params["namespace"] = namespace

        return await self._request("GET", "/v1/fog/usage/quotas", params=params)

    # Convenience methods

    async def run_job(
        self,
        image: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        resources: dict[str, Any] | None = None,
        timeout: float = 300.0,
        namespace: str | None = None,
    ) -> JobResult:
        """
        Submit job and wait for completion (convenience method)

        Args:
            image: Container image or WASI module
            args: Command arguments
            env: Environment variables
            resources: Resource requirements
            timeout: Maximum wait time
            namespace: Target namespace

        Returns:
            JobResult with execution results
        """

        # Submit job
        job = await self.submit_job(image=image, args=args, env=env, resources=resources, namespace=namespace)

        # Wait for completion
        return await self.wait_for_job(job.job_id, timeout=timeout)

    async def stream_logs(self, job_id: str, follow: bool = True) -> AsyncGenerator[str, None]:
        """
        Stream job logs (placeholder - WebSocket implementation needed)

        Args:
            job_id: Job to stream logs from
            follow: Follow logs in real-time

        Yields:
            Log lines as they become available
        """

        # Reference implementation: WebSocket log streaming capabilities
        # For now, just return current logs
        logs = await self.get_job_logs(job_id)

        for line in logs.split("\n"):
            if line.strip():
                yield line

    # Marketplace methods

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
        """
        Get price quote for resource requirements

        Args:
            cpu_cores: Required CPU cores
            memory_gb: Required memory in GB
            estimated_duration_hours: Expected job duration
            disk_gb: Required disk space in GB
            bid_type: "spot" or "on_demand"
            pricing_tier: "basic", "standard", or "premium"
            min_trust_score: Minimum node trust score (0.0-1.0)
            max_latency_ms: Maximum acceptable latency

        Returns:
            PriceQuote with pricing information
        """

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

        response = await self._request("POST", "/v1/fog/quotes", json=quote_request)

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
        """
        Get current marketplace pricing information

        Returns:
            MarketplacePrices with current market rates
        """

        response = await self._request("GET", "/v1/fog/prices")

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
        """
        Estimate cost for job execution

        Args:
            image: Container image or WASI module
            cpu_cores: Required CPU cores
            memory_gb: Required memory in GB
            estimated_duration_hours: Expected job duration
            bid_type: "spot" or "on_demand"
            pricing_tier: "basic", "standard", or "premium"
            include_recommendations: Include cost optimization recommendations

        Returns:
            CostEstimate with detailed cost breakdown
        """

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
            "cpu_cost": quote.avg_price * 0.7 if quote.avg_price else 0.0,  # 70% CPU
            "memory_cost": quote.avg_price * 0.2 if quote.avg_price else 0.0,  # 20% memory
            "disk_cost": quote.avg_price * 0.05 if quote.avg_price else 0.0,  # 5% disk
            "network_cost": quote.avg_price * 0.05 if quote.avg_price else 0.0,  # 5% network
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
            if quote.price_volatility and quote.price_volatility > 0.3:
                recommendations.append(
                    "High price volatility detected - consider on-demand pricing for predictable costs"
                )

            if bid_type == "on_demand" and quote.current_spot_rate:
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
        """
        Submit marketplace bid for resource execution

        Args:
            image: Container image or WASI module
            cpu_cores: Required CPU cores
            memory_gb: Required memory in GB
            max_price: Maximum willing to pay in USD
            args: Command arguments
            env: Environment variables
            disk_gb: Required disk space in GB
            estimated_duration_hours: Expected job duration
            bid_type: "spot" or "on_demand"
            pricing_tier: "basic", "standard", or "premium"
            min_trust_score: Minimum node trust score (0.0-1.0)
            namespace: Target namespace

        Returns:
            Dict with bid submission result
        """

        bid_request = {
            "namespace": namespace or self.namespace,
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
            raise ValueError("Namespace must be provided either in constructor or method call")

        return await self._request("POST", "/v1/fog/marketplace/bids", json=bid_request)

    async def get_bid_status(self, bid_id: str) -> dict[str, Any]:
        """
        Get marketplace bid status

        Args:
            bid_id: Bid ID to check

        Returns:
            Dict with bid status information
        """

        return await self._request("GET", f"/v1/fog/marketplace/bids/{bid_id}")

    async def cancel_bid(self, bid_id: str) -> dict[str, Any]:
        """
        Cancel marketplace bid

        Args:
            bid_id: Bid ID to cancel

        Returns:
            Dict with cancellation result
        """

        return await self._request("DELETE", f"/v1/fog/marketplace/bids/{bid_id}")

    async def get_marketplace_status(self) -> dict[str, Any]:
        """
        Get marketplace status and statistics

        Returns:
            Dict with marketplace metrics and health
        """

        return await self._request("GET", "/v1/fog/marketplace/status")

    # Enhanced convenience methods with marketplace support

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
        """
        Submit job with budget constraint and wait for completion

        Args:
            image: Container image or WASI module
            max_price: Maximum willing to pay in USD
            args: Command arguments
            env: Environment variables
            resources: Resource requirements
            bid_type: "spot" or "on_demand"
            timeout: Maximum wait time
            namespace: Target namespace

        Returns:
            JobResult with execution results
        """

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
