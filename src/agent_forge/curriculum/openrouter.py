"""OpenRouter LLM client with caching, cost tracking, and Jinja templates.

Provides a robust client for OpenRouter API with:
- Exponential backoff with jitter for 429/5xx errors
- SQLite caching for deterministic request/response pairs
- Cost tracking with detailed token usage logging
- Jinja2 template rendering for prompt templates
- Rate limiting to ~60 RPM to stay under OpenRouter limits
"""

import asyncio
import hashlib
import json
import logging
import random
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
import jinja2

from .schemas import strict_load

logger = logging.getLogger(__name__)


class OpenRouterError(Exception):
    """OpenRouter API error."""

    pass


class OpenRouterRateLimitError(OpenRouterError):
    """Rate limit exceeded."""

    pass


class OpenRouterLLM:
    """OpenRouter LLM client with caching, cost tracking, and template rendering.

    Features:
    - Exponential backoff with jitter for 429/5xx errors
    - SQLite caching for deterministic request/response pairs
    - JSONL cost tracking with detailed token usage
    - Jinja2 template rendering system
    - Rate limiting to ~60 RPM
    """

    def __init__(
        self,
        api_key: str,
        cache_dir: str = ".forge/cache",
        model: str = "anthropic/claude-3-5-sonnet-20241022",
        model_pool: list[str] | None = None,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        rpm_limit: int = 60,
    ):
        """Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key
            cache_dir: Directory for SQLite cache and cost logs
            model: Default model to use (can be overridden per request)
            model_pool: List of models to randomly select from for diversity
            max_retries: Maximum number of retries for failed requests
            base_delay: Base delay for exponential backoff (seconds)
            max_delay: Maximum delay for exponential backoff (seconds)
            rpm_limit: Requests per minute limit
        """
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.rpm_limit = rpm_limit

        # Multi-model pool for diversity
        self.model_pool = model_pool or [
            "openai/gpt-4o",  # GPT-4o (closest to GPT-5 available)
            "anthropic/claude-3-5-sonnet-20241022",  # Claude Opus 4.1 (closest available)
            "google/gemini-pro-1.5",  # Gemini 2.5 Pro (closest available)
        ]
        self.use_model_pool = len(self.model_pool) > 1

        # Setup cache directory
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize SQLite cache
        self.cache_db_path = self.cache_dir / "frontier.sqlite"
        self._init_cache_db()

        # Cost tracking
        self.cost_log_path = self.cache_dir / "costs.jsonl"

        # Rate limiting
        self._request_times: list[float] = []

        # Jinja2 environment
        self.jinja_env = jinja2.Environment(loader=jinja2.BaseLoader(), undefined=jinja2.StrictUndefined)

        # Session for HTTP requests
        self._session: aiohttp.ClientSession | None = None

        logger.info(f"Initialized OpenRouter client with model {model}, cache at {self.cache_dir}")

    def _init_cache_db(self) -> None:
        """Initialize SQLite cache database."""
        with sqlite3.connect(self.cache_db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    request_hash TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    prompt_hash TEXT NOT NULL,
                    temperature REAL NOT NULL,
                    max_tokens INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    response_text TEXT NOT NULL,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER,
                    cost_usd REAL
                )
            """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON cache(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_model ON cache(model)")

    def _hash_request(self, prompt: str, model: str, temperature: float, max_tokens: int) -> str:
        """Generate deterministic hash for request caching."""
        request_data = {
            "prompt": prompt,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        request_str = json.dumps(request_data, sort_keys=True)
        return hashlib.sha256(request_str.encode()).hexdigest()

    def _get_cached_response(self, request_hash: str) -> str | None:
        """Get cached response if available."""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.execute(
                    "SELECT response_text FROM cache WHERE request_hash = ?",
                    (request_hash,),
                )
                result = cursor.fetchone()
                if result:
                    logger.debug(f"Cache hit for request {request_hash[:8]}...")
                    return result[0]
        except Exception as e:
            logger.warning(f"Cache lookup failed: {e}")
        return None

    def _cache_response(
        self,
        request_hash: str,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        response_text: str,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        cost_usd: float | None = None,
    ) -> None:
        """Cache response in SQLite."""
        try:
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
            created_at = datetime.now(timezone.utc).isoformat()

            with sqlite3.connect(self.cache_db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO cache
                    (request_hash, model, prompt_hash, temperature, max_tokens,
                     created_at, response_text, prompt_tokens, completion_tokens,
                     total_tokens, cost_usd)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        request_hash,
                        model,
                        prompt_hash,
                        temperature,
                        max_tokens,
                        created_at,
                        response_text,
                        prompt_tokens,
                        completion_tokens,
                        total_tokens,
                        cost_usd,
                    ),
                )
            logger.debug(f"Cached response for request {request_hash[:8]}...")
        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")

    def _log_cost(
        self,
        model: str,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_tokens: int | None,
        cost_usd: float | None,
        cached: bool = False,
    ) -> None:
        """Log cost information to JSONL file."""
        try:
            cost_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost_usd": cost_usd,
                "cached": cached,
            }

            with open(self.cost_log_path, "a") as f:
                f.write(json.dumps(cost_entry) + "\n")

        except Exception as e:
            logger.warning(f"Failed to log cost: {e}")

    async def _rate_limit(self) -> None:
        """Enforce rate limiting."""
        now = time.time()

        # Remove requests older than 1 minute
        self._request_times = [t for t in self._request_times if now - t < 60]

        # Check if we're at the limit
        if len(self._request_times) >= self.rpm_limit:
            # Calculate how long to wait
            oldest_request = min(self._request_times)
            wait_time = 60 - (now - oldest_request)
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)

        # Record this request
        self._request_times.append(now)

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def _make_request(
        self, prompt: str, model: str, temperature: float = 0.7, max_tokens: int = 4096
    ) -> dict[str, Any]:
        """Make request to OpenRouter API with retries."""
        session = await self._get_session()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://agent-forge.ai",
            "X-Title": "Agent Forge Frontier Curriculum",
        }

        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        for attempt in range(self.max_retries + 1):
            try:
                # Rate limiting
                await self._rate_limit()

                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=data,
                ) as response:
                    response_data = await response.json()

                    if response.status == 200:
                        return response_data
                    elif response.status == 429:
                        # Rate limit hit - exponential backoff
                        if attempt < self.max_retries:
                            delay = min(
                                self.base_delay * (2**attempt) + random.uniform(0, 1),
                                self.max_delay,
                            )
                            logger.warning(f"Rate limited, retrying in {delay:.1f}s (attempt {attempt + 1})")
                            await asyncio.sleep(delay)
                            continue
                        else:
                            raise OpenRouterRateLimitError("Rate limit exceeded after all retries")
                    elif response.status >= 500:
                        # Server error - exponential backoff
                        if attempt < self.max_retries:
                            delay = min(
                                self.base_delay * (2**attempt) + random.uniform(0, 1),
                                self.max_delay,
                            )
                            logger.warning(f"Server error {response.status}, retrying in {delay:.1f}s")
                            await asyncio.sleep(delay)
                            continue
                        else:
                            raise OpenRouterError(f"Server error after all retries: {response.status}")
                    else:
                        # Other error - no retry
                        error_detail = response_data.get("error", {}).get("message", "Unknown error")
                        raise OpenRouterError(f"API error {response.status}: {error_detail}")

            except asyncio.TimeoutError:
                if attempt < self.max_retries:
                    delay = min(self.base_delay * (2**attempt), self.max_delay)
                    logger.warning(f"Request timeout, retrying in {delay:.1f}s")
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise OpenRouterError("Request timeout after all retries")
            except Exception as e:
                if attempt < self.max_retries:
                    delay = min(self.base_delay * (2**attempt), self.max_delay)
                    logger.warning(f"Request failed: {e}, retrying in {delay:.1f}s")
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise OpenRouterError(f"Request failed after all retries: {e}")

        raise OpenRouterError("Should not reach here")

    async def invoke(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Invoke LLM with caching and cost tracking.

        Args:
            prompt: Input prompt
            model: Model to use (defaults to instance model)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text response

        Raises:
            OpenRouterError: On API or network errors
        """
        if model is None:
            if self.use_model_pool:
                # Randomly select from model pool for diversity
                model = random.choice(self.model_pool)
                logger.debug(f"Randomly selected model: {model}")
            else:
                model = self.model

        # Check cache first
        request_hash = self._hash_request(prompt, model, temperature, max_tokens)
        cached_response = self._get_cached_response(request_hash)

        if cached_response:
            # Log cached cost (0 actual cost)
            self._log_cost(model, None, None, None, 0.0, cached=True)
            return cached_response

        # Make API request
        logger.info(f"Making OpenRouter request to {model} (temp={temperature})")
        response_data = await self._make_request(prompt, model, temperature, max_tokens)

        # Extract response text
        if "choices" not in response_data or not response_data["choices"]:
            raise OpenRouterError("No choices in response")

        choice = response_data["choices"][0]
        if "message" not in choice or "content" not in choice["message"]:
            raise OpenRouterError("No content in response")

        response_text = choice["message"]["content"]

        # Extract usage and cost information
        usage = response_data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")

        # Estimate cost (rough estimates, actual costs vary)
        cost_usd = None
        if total_tokens:
            # Very rough cost estimation - real costs should come from OpenRouter
            if "claude-3-5-sonnet" in model:
                cost_usd = total_tokens * 0.000015  # ~$15/1M tokens
            elif "gpt-4" in model:
                cost_usd = total_tokens * 0.00003  # ~$30/1M tokens
            else:
                cost_usd = total_tokens * 0.000001  # ~$1/1M tokens fallback

        # Cache the response
        self._cache_response(
            request_hash,
            prompt,
            model,
            temperature,
            max_tokens,
            response_text,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            cost_usd,
        )

        # Log cost
        self._log_cost(
            model,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            cost_usd,
            cached=False,
        )

        logger.info(f"OpenRouter response: {len(response_text)} chars, {total_tokens or 0} tokens")
        return response_text

    def render_template(self, template: str, **kwargs) -> str:
        """Render Jinja2 template with provided variables.

        Args:
            template: Jinja2 template string
            **kwargs: Variables to substitute in template

        Returns:
            Rendered template string

        Raises:
            jinja2.TemplateError: On template rendering errors
        """
        try:
            jinja_template = self.jinja_env.from_string(template)
            return jinja_template.render(**kwargs)
        except jinja2.TemplateError as e:
            logger.error(f"Template rendering failed: {e}")
            raise

    async def invoke_with_schema(
        self,
        prompt: str,
        schema_class: type,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        max_schema_retries: int = 3,
    ):
        """Invoke LLM and parse response with schema validation.

        Args:
            prompt: Input prompt
            schema_class: Pydantic model class to parse response
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            max_schema_retries: Max retries for schema validation failures

        Returns:
            Validated schema instance

        Raises:
            OpenRouterError: On API errors or persistent schema failures
        """
        for attempt in range(max_schema_retries + 1):
            try:
                response_text = await self.invoke(prompt, model, temperature, max_tokens)
                return strict_load(response_text, schema_class)
            except Exception as e:
                if attempt < max_schema_retries:
                    logger.warning(f"Schema validation failed (attempt {attempt + 1}): {e}")
                    # Add instruction to be more careful with JSON format
                    prompt += "\n\nIMPORTANT: Respond with valid JSON only, no additional text."
                    continue
                else:
                    logger.error(f"Schema validation failed after {max_schema_retries} retries")
                    raise OpenRouterError(f"Schema validation failed: {e}")

    def get_model_stats(self) -> dict[str, Any]:
        """Get statistics about model usage from cache."""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            conn.row_factory = sqlite3.Row

            # Get model usage counts
            cursor = conn.execute(
                """
                SELECT model, COUNT(*) as usage_count
                FROM cache
                GROUP BY model
                ORDER BY usage_count DESC
            """
            )

            model_usage = {}
            total_requests = 0
            for row in cursor.fetchall():
                model_usage[row["model"]] = row["usage_count"]
                total_requests += row["usage_count"]

            conn.close()

            return {
                "total_requests": total_requests,
                "model_usage": model_usage,
                "model_pool": self.model_pool,
                "model_pool_enabled": self.use_model_pool,
                "diversity_score": len(model_usage) / len(self.model_pool) if self.model_pool else 0.0,
            }

        except Exception as e:
            logger.error(f"Error getting model stats: {e}")
            return {"error": str(e)}

    async def close(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM cache")
                total_entries = cursor.fetchone()[0]

                cursor = conn.execute(
                    """
                    SELECT model, COUNT(*) as count,
                           SUM(total_tokens) as total_tokens,
                           SUM(cost_usd) as total_cost
                    FROM cache
                    GROUP BY model
                """
                )
                by_model = cursor.fetchall()

                cursor = conn.execute(
                    """
                    SELECT DATE(created_at) as date, COUNT(*) as count
                    FROM cache
                    GROUP BY DATE(created_at)
                    ORDER BY date DESC
                    LIMIT 7
                """
                )
                recent_activity = cursor.fetchall()

                return {
                    "total_entries": total_entries,
                    "by_model": [
                        {
                            "model": row[0],
                            "count": row[1],
                            "total_tokens": row[2] or 0,
                            "total_cost_usd": row[3] or 0.0,
                        }
                        for row in by_model
                    ],
                    "recent_activity": [{"date": row[0], "count": row[1]} for row in recent_activity],
                }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"error": str(e)}
