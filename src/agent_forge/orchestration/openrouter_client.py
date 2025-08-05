"""OpenRouter API client with error handling, rate limiting, and cost tracking."""

import asyncio
from dataclasses import dataclass
import json
import logging
import os
import time
from typing import Any

import aiohttp
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .model_config import MODEL_ROUTING_CONFIG, RATE_LIMITS, TaskType

logger = logging.getLogger(__name__)


@dataclass
class APIResponse:
    """Response from OpenRouter API."""

    content: str
    model_used: str
    usage: dict[str, int]
    cost: float
    latency: float
    task_type: TaskType


@dataclass
class ModelMetrics:
    """Metrics for model performance tracking."""

    total_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    total_latency: float = 0.0
    error_count: int = 0
    last_request_time: float | None = None


class OpenRouterClient:
    """Client for OpenRouter API with intelligent routing and fallback."""

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key. If not provided, reads from environment.
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            msg = "OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable."
            raise ValueError(msg)

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/AIVillage/agent-forge",
            "X-Title": "Agent Forge Multi-Model Training",
        }

        # Rate limiting state
        self.rate_limiters: dict[str, float] = {}
        self.model_metrics: dict[str, ModelMetrics] = {}

        # Cost tracking
        self.total_cost = 0.0
        self.cost_by_task: dict[TaskType, float] = dict.fromkeys(TaskType, 0.0)

        # Initialize metrics for all models
        for model_list in MODEL_ROUTING_CONFIG.values():
            primary = model_list.get("primary")
            if primary and primary not in self.model_metrics:
                self.model_metrics[primary] = ModelMetrics()

            for fallback in model_list.get("fallback", []):
                if fallback not in self.model_metrics:
                    self.model_metrics[fallback] = ModelMetrics()

    async def _check_rate_limit(self, model: str) -> None:
        """Check and enforce rate limits for a model."""
        current_time = time.time()
        rate_limit = RATE_LIMITS.get(model, 30)  # Default 30 requests/minute

        if model in self.rate_limiters:
            time_since_last = current_time - self.rate_limiters[model]
            min_interval = 60.0 / rate_limit

            if time_since_last < min_interval:
                sleep_time = min_interval - time_since_last
                logger.debug(f"Rate limiting {model}: sleeping {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)

        self.rate_limiters[model] = time.time()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
    )
    async def _make_request(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        """Make a request to OpenRouter API with retries."""
        await self._check_rate_limit(model)

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        start_time = time.time()

        async with (
            aiohttp.ClientSession() as session,
            session.post(
                f"{self.BASE_URL}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as response,
        ):
            latency = time.time() - start_time

            if response.status != 200:
                error_text = await response.text()
                logger.error(f"OpenRouter API error: {response.status} - {error_text}")
                raise aiohttp.ClientResponseError(
                    request_info=response.request_info,
                    history=response.history,
                    status=response.status,
                )

            data = await response.json()
            data["latency"] = latency
            return data

    async def complete(
        self,
        task_type: TaskType,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        model_override: str | None = None,
    ) -> APIResponse:
        """Complete a chat request with intelligent model routing.

        Args:
            task_type: Type of task to route
            messages: Chat messages
            max_tokens: Max tokens to generate (uses config default if not specified)
            temperature: Generation temperature (uses config default if not specified)
            model_override: Override model selection

        Returns:
            APIResponse with generated content and metrics
        """
        config = MODEL_ROUTING_CONFIG[task_type]
        max_tokens = max_tokens or config["max_tokens"]
        temperature = temperature or config["temperature"]

        # Select models to try
        if model_override:
            models_to_try = [model_override]
        else:
            models_to_try = [config["primary"], *config.get("fallback", [])]

        last_error = None

        for model in models_to_try:
            try:
                logger.info(f"Attempting {task_type.value} with model: {model}")

                response_data = await self._make_request(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                # Extract response details
                content = response_data["choices"][0]["message"]["content"]
                usage = response_data.get("usage", {})

                # Calculate cost (simplified - OpenRouter provides this)
                cost = self._calculate_cost(model, usage)

                # Update metrics
                self._update_metrics(model, usage, cost, response_data["latency"])
                self.total_cost += cost
                self.cost_by_task[task_type] += cost

                return APIResponse(
                    content=content,
                    model_used=model,
                    usage=usage,
                    cost=cost,
                    latency=response_data["latency"],
                    task_type=task_type,
                )

            except Exception as e:
                logger.warning(f"Failed with model {model}: {e!s}")
                last_error = e

                # Update error metrics
                if model in self.model_metrics:
                    self.model_metrics[model].error_count += 1

                continue

        # All models failed
        msg = f"All models failed for {task_type.value}. Last error: {last_error}"
        raise Exception(msg)

    def _calculate_cost(self, model: str, usage: dict[str, int]) -> float:
        """Calculate cost based on token usage.

        Note: In production, OpenRouter provides actual costs in the response.
        This is a simplified estimation.
        """
        # Simplified cost calculation (cents per 1K tokens)
        cost_per_1k = {
            "anthropic/claude-3-opus-20240229": 0.015,
            "anthropic/claude-3-sonnet-20240229": 0.003,
            "anthropic/claude-3-haiku-20240307": 0.00025,
            "openai/gpt-4-turbo": 0.01,
            "openai/gpt-4o-mini": 0.00015,
            "google/gemini-pro-1.5": 0.00125,
            "google/gemini-flash-1.5": 0.00035,
            "meta-llama/llama-3.1-70b-instruct": 0.0008,
            "deepseek/deepseek-coder-v2-instruct": 0.0006,
        }

        rate = cost_per_1k.get(model, 0.001)  # Default rate
        total_tokens = usage.get("total_tokens", 0)

        return (total_tokens / 1000) * rate

    def _update_metrics(self, model: str, usage: dict[str, int], cost: float, latency: float) -> None:
        """Update model performance metrics."""
        if model not in self.model_metrics:
            self.model_metrics[model] = ModelMetrics()

        metrics = self.model_metrics[model]
        metrics.total_requests += 1
        metrics.total_tokens += usage.get("total_tokens", 0)
        metrics.total_cost += cost
        metrics.total_latency += latency
        metrics.last_request_time = time.time()

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get summary of all metrics."""
        summary = {
            "total_cost": self.total_cost,
            "cost_by_task": {task.value: cost for task, cost in self.cost_by_task.items()},
            "model_performance": {},
        }

        for model, metrics in self.model_metrics.items():
            if metrics.total_requests > 0:
                summary["model_performance"][model] = {
                    "requests": metrics.total_requests,
                    "avg_latency": metrics.total_latency / metrics.total_requests,
                    "total_tokens": metrics.total_tokens,
                    "total_cost": metrics.total_cost,
                    "error_rate": metrics.error_count / metrics.total_requests,
                }

        return summary

    async def close(self) -> None:
        """Clean up resources."""
        # Save metrics to file for analysis
        metrics_file = "openrouter_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(self.get_metrics_summary(), f, indent=2)
        logger.info(f"Metrics saved to {metrics_file}")
