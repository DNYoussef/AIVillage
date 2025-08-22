# AIVillage Python SDK

A comprehensive, production-ready Python client for the AIVillage API with full async support, type safety, and built-in reliability patterns.

## Features

- **Type Safety**: Full type hints with mypy support and comprehensive models
- **Async/Await**: Native asyncio support with proper resource management
- **Reliability**: Automatic retries with exponential backoff and circuit breaker patterns
- **Idempotency**: Safe retry of mutating operations with idempotency keys
- **Rate Limiting**: Built-in rate limit awareness with automatic backoff
- **Error Handling**: Rich exception types with detailed error context
- **Authentication**: Bearer token and API key authentication methods

## Installation

```bash
# pip
pip install aivillage-client

# pipenv
pipenv install aivillage-client

# poetry
poetry add aivillage-client

# conda
conda install -c aivillage aivillage-client
```

## Quick Start

```python
import asyncio
import aivillage_client
from aivillage_client.models import ChatRequest, QueryRequest
from aivillage_client.exceptions import ApiException, RateLimitException

# Configure API client
configuration = aivillage_client.Configuration(
    host="https://api.aivillage.io/v1",
    access_token="your-api-key"
)

async def main():
    async with aivillage_client.ApiClient(configuration) as api_client:
        chat_api = aivillage_client.ChatApi(api_client)

        # Chat with AI agents
        response = await chat_api.chat(
            ChatRequest(
                message="How can I optimize machine learning model inference on mobile devices?",
                agent_preference="magi",  # Research specialist
                mode="comprehensive",
                user_context={
                    "device_type": "mobile",
                    "battery_level": 75,
                    "network_type": "wifi"
                }
            )
        )

        print(f"{response.agent_used}: {response.response}")
        print(f"Processing time: {response.processing_time_ms}ms")

# Run the example
asyncio.run(main())
```

## Configuration

### Basic Configuration

```python
import aivillage_client

# Method 1: Direct configuration
configuration = aivillage_client.Configuration(
    host="https://api.aivillage.io/v1",
    access_token="your-api-key",
    timeout=30.0,
    retries=3
)

# Method 2: Environment variables
import os
configuration = aivillage_client.Configuration(
    host=os.getenv("AIVILLAGE_API_URL", "https://api.aivillage.io/v1"),
    access_token=os.getenv("AIVILLAGE_API_KEY")
)
```

### Advanced Configuration

```python
import aiohttp
from aivillage_client import Configuration, ApiClient

# Custom HTTP session with connection pooling
connector = aiohttp.TCPConnector(
    limit=100,  # Total connection pool size
    limit_per_host=10,  # Per-host connection limit
    ttl_dns_cache=300,  # DNS cache TTL
    use_dns_cache=True
)

configuration = Configuration(
    host="https://api.aivillage.io/v1",
    access_token="your-api-key",
    timeout=60.0,
    retries=5,
    backoff_factor=1.5
)

# Use custom connector
async with ApiClient(configuration, connector=connector) as client:
    # Use client APIs here
    pass
```

## API Reference

### Chat API

Interact with AIVillage's specialized AI agents.

```python
from aivillage_client.models import ChatRequest, ChatRequestUserContext

async def chat_example(api_client):
    chat_api = aivillage_client.ChatApi(api_client)

    # Basic chat
    response = await chat_api.chat(
        ChatRequest(
            message="Explain federated learning advantages and challenges",
            agent_preference="sage",  # Knowledge specialist
            mode="analytical"
        )
    )

    # Chat with context and conversation continuation
    contextual_response = await chat_api.chat(
        ChatRequest(
            message="How can I implement this on mobile devices?",
            conversation_id=response.conversation_id,  # Continue conversation
            agent_preference="navigator",  # Routing specialist
            mode="balanced",
            user_context=ChatRequestUserContext(
                device_type="mobile",
                battery_level=60,
                network_type="cellular",
                data_budget_mb=100
            )
        )
    )

    print(f"Agent: {contextual_response.agent_used}")
    print(f"Response: {contextual_response.response}")
    print(f"Confidence: {contextual_response.metadata.confidence}")

    return contextual_response
```

**Available Agents:**
- `king`: Coordination and oversight with public thought bubbles
- `magi`: Research and comprehensive analysis
- `sage`: Deep knowledge and wisdom
- `oracle`: Predictions and forecasting
- `navigator`: Routing and mobile optimization
- `any`: Auto-select best agent (default)

**Response Modes:**
- `fast`: Quick responses with minimal processing
- `balanced`: Good balance of speed and thoroughness (default)
- `comprehensive`: Detailed analysis with full context
- `creative`: Innovative and creative insights
- `analytical`: Systematic analysis and reasoning

### RAG API

Advanced knowledge retrieval with Bayesian trust networks.

```python
from aivillage_client.models import QueryRequest

async def rag_example(api_client):
    rag_api = aivillage_client.RAGApi(api_client)

    # Basic knowledge query
    result = await rag_api.process_query(
        QueryRequest(
            query="What are the latest advances in differential privacy for machine learning?",
            mode="comprehensive",
            include_sources=True,
            max_results=10
        )
    )

    print(f"Query ID: {result.query_id}")
    print(f"Response: {result.response}")
    print(f"Bayesian confidence: {result.metadata.bayesian_confidence}")

    # Process sources with confidence scores
    for source in result.sources:
        print(f"Source: {source.title}")
        print(f"  Confidence: {source.confidence:.3f}")
        print(f"  Type: {source.source_type}")
        if source.url:
            print(f"  URL: {source.url}")

    return result

# Multi-query processing with asyncio.gather
async def batch_queries(api_client):
    rag_api = aivillage_client.RAGApi(api_client)

    queries = [
        "Federated learning privacy techniques",
        "Mobile AI model compression methods",
        "Edge computing optimization strategies"
    ]

    # Process multiple queries concurrently
    tasks = [
        rag_api.process_query(QueryRequest(query=q, mode="fast"))
        for q in queries
    ]

    results = await asyncio.gather(*tasks)

    for i, result in enumerate(results):
        print(f"Query {i+1}: {result.response[:100]}...")
```

### Agents API

Manage and monitor AI agents.

```python
from aivillage_client.models import AgentTaskRequest

async def agents_example(api_client):
    agents_api = aivillage_client.AgentsApi(api_client)

    # List available agents
    agents_list = await agents_api.list_agents(
        category="knowledge",
        available_only=True
    )

    for agent in agents_list.agents:
        print(f"Agent: {agent.name}")
        print(f"  Status: {agent.status}")
        print(f"  Load: {agent.current_load}%")
        print(f"  Capabilities: {', '.join(agent.capabilities)}")

    # Assign task to specific agent
    if agents_list.agents:
        first_agent = agents_list.agents[0]

        task_result = await agents_api.assign_agent_task(
            agent_id=first_agent.id,
            agent_task_request=AgentTaskRequest(
                task_description="Research quantum computing applications in federated learning",
                priority="high",
                timeout_seconds=600,
                context={
                    "domain": "quantum_ai",
                    "depth": "comprehensive",
                    "include_implementation": True
                }
            )
        )

        print(f"Task assigned: {task_result.task_id}")
        print(f"Estimated completion: {task_result.estimated_completion_time}")
```

### P2P API

Monitor peer-to-peer mesh network status.

```python
async def p2p_example(api_client):
    p2p_api = aivillage_client.P2PApi(api_client)

    # Get overall network status
    status = await p2p_api.get_p2p_status()
    print(f"Network status: {status.status}")
    print(f"Connected peers: {status.peer_count}")
    print(f"Network health: {status.health_score}")

    # List connected peers by transport type
    transport_types = ["bitchat", "betanet"]

    for transport in transport_types:
        peers = await p2p_api.list_peers(transport_type=transport)

        print(f"\n{transport.upper()} peers ({len(peers.peers)}):")
        for peer in peers.peers:
            print(f"  {peer.id[:8]}... - {peer.status} ({peer.latency_ms}ms)")
```

### Digital Twin API

Privacy-preserving personal AI assistant.

```python
from aivillage_client.models import DigitalTwinDataUpdate

async def digital_twin_example(api_client):
    twin_api = aivillage_client.DigitalTwinApi(api_client)

    # Get current profile
    profile = await twin_api.get_digital_twin_profile()
    print(f"Model size: {profile.model_size_mb}MB")
    print(f"Accuracy: {profile.learning_stats.accuracy_score:.3f}")
    print(f"Privacy level: {profile.privacy_settings.level}")

    # Update with new interaction data
    await twin_api.update_digital_twin_data(
        DigitalTwinDataUpdate(
            data_type="interaction",
            data_points=[
                {
                    "timestamp": "2025-08-19T10:30:00Z",
                    "content": {
                        "interaction_type": "chat",
                        "user_satisfaction": 0.9,
                        "context": "mobile_optimization_help"
                    },
                    "prediction_accuracy": 0.87
                }
            ]
        )
    )

    print("Digital twin updated with new interaction data")
```

## Error Handling

### Exception Types

```python
from aivillage_client.exceptions import (
    ApiException,
    RateLimitException,
    AuthenticationException,
    ValidationException,
    ServerException
)

async def robust_api_call(api_client):
    chat_api = aivillage_client.ChatApi(api_client)

    try:
        response = await chat_api.chat(
            ChatRequest(message="Test message", mode="fast")
        )
        return response

    except RateLimitException as e:
        print(f"Rate limited. Retry after {e.retry_after}s")
        # Automatic retry after rate limit period
        await asyncio.sleep(e.retry_after)
        return await chat_api.chat(ChatRequest(message="Test message", mode="fast"))

    except AuthenticationException as e:
        print(f"Authentication failed: {e.detail}")
        print("Please check your API key")
        raise

    except ValidationException as e:
        print(f"Validation error: {e.detail}")
        print(f"Field errors: {e.field_errors}")
        raise

    except ServerException as e:
        print(f"Server error: {e.status} - {e.detail}")
        print(f"Request ID: {e.request_id}")
        # Could implement retry logic here
        raise

    except ApiException as e:
        print(f"General API error: {e}")
        raise
```

### Retry Pattern with Circuit Breaker

```python
import asyncio
from typing import TypeVar, Callable, Any
from functools import wraps

T = TypeVar('T')

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open

    async def call(self, func: Callable[[], T]) -> T:
        if self.state == 'open':
            if time.time() - self.last_failure_time < self.timeout:
                raise Exception("Circuit breaker is open")
            else:
                self.state = 'half-open'

        try:
            result = await func()
            # Success - reset failure count
            if self.state == 'half-open':
                self.state = 'closed'
            self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = 'open'

            raise

async def resilient_chat_request(
    chat_api: aivillage_client.ChatApi,
    request: ChatRequest,
    max_retries: int = 3,
    base_delay: float = 1.0
) -> ChatResponse:
    """Make a chat request with exponential backoff retry."""

    circuit_breaker = CircuitBreaker()

    for attempt in range(max_retries + 1):
        try:
            return await circuit_breaker.call(
                lambda: chat_api.chat(request)
            )

        except RateLimitException as e:
            if attempt == max_retries:
                raise
            await asyncio.sleep(e.retry_after)

        except (ServerException, ApiException) as e:
            if attempt == max_retries:
                raise

            # Exponential backoff with jitter
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            await asyncio.sleep(min(delay, 32.0))

        except Exception as e:
            # Don't retry client errors or unexpected errors
            raise

    raise Exception("Max retries exceeded")
```

## Advanced Usage

### Context Manager Pattern

```python
from contextlib import asynccontextmanager
from typing import Dict, Any

@asynccontextmanager
async def aivillage_client_context(api_key: str):
    """Context manager for AIVillage API client with proper resource cleanup."""
    configuration = aivillage_client.Configuration(
        host="https://api.aivillage.io/v1",
        access_token=api_key,
        timeout=30.0
    )

    async with aivillage_client.ApiClient(configuration) as api_client:
        yield {
            'chat': aivillage_client.ChatApi(api_client),
            'rag': aivillage_client.RAGApi(api_client),
            'agents': aivillage_client.AgentsApi(api_client),
            'p2p': aivillage_client.P2PApi(api_client),
            'digital_twin': aivillage_client.DigitalTwinApi(api_client),
        }

async def multi_api_workflow():
    """Example using multiple APIs in a coordinated workflow."""
    async with aivillage_client_context("your-api-key") as apis:
        # Step 1: Get available agents
        agents = await apis['agents'].list_agents(category="knowledge")
        print(f"Available agents: {len(agents.agents)}")

        # Step 2: Process a complex query
        rag_result = await apis['rag'].process_query(
            QueryRequest(
                query="Compare different neural architecture search methods",
                mode="comprehensive",
                include_sources=True
            )
        )

        # Step 3: Follow up with specialized agent
        chat_response = await apis['chat'].chat(
            ChatRequest(
                message="Can you elaborate on the mobile deployment challenges mentioned?",
                conversation_id=rag_result.query_id,
                agent_preference="navigator",  # Mobile optimization specialist
                mode="balanced"
            )
        )

        # Step 4: Check network status for deployment planning
        p2p_status = await apis['p2p'].get_p2p_status()

        return {
            'knowledge': rag_result,
            'deployment_advice': chat_response,
            'network_status': p2p_status
        }
```

### Idempotency for Safe Retries

```python
import uuid
from datetime import datetime

def generate_idempotency_key(operation: str, context: str = None) -> str:
    """Generate a unique idempotency key for safe retries."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    unique_id = str(uuid.uuid4())[:8]

    if context:
        return f"{operation}-{context}-{timestamp}-{unique_id}"
    return f"{operation}-{timestamp}-{unique_id}"

async def idempotent_chat_request(
    chat_api: aivillage_client.ChatApi,
    message: str,
    agent_preference: str = "any"
) -> ChatResponse:
    """Send a chat request with idempotency key for safe retries."""

    idempotency_key = generate_idempotency_key("chat", agent_preference)

    request = ChatRequest(
        message=message,
        agent_preference=agent_preference,
        mode="balanced"
    )

    # Add idempotency key to request headers
    response = await chat_api.chat(
        request,
        _headers={"Idempotency-Key": idempotency_key}
    )

    return response

# Usage example
async def safe_operation_example():
    configuration = aivillage_client.Configuration(
        host="https://api.aivillage.io/v1",
        access_token="your-api-key"
    )

    async with aivillage_client.ApiClient(configuration) as api_client:
        chat_api = aivillage_client.ChatApi(api_client)

        # This request is safe to retry - same idempotency key will return cached result
        response1 = await idempotent_chat_request(
            chat_api,
            "Explain the benefits of model quantization",
            "magi"
        )

        # Retrying with same parameters will return cached response
        response2 = await idempotent_chat_request(
            chat_api,
            "Explain the benefits of model quantization",
            "magi"
        )

        print(f"Same response: {response1.response == response2.response}")
```

### Pydantic Models and Validation

```python
from pydantic import BaseModel, validator
from typing import Optional, List
from aivillage_client.models import ChatRequest, ChatRequestUserContext

class MobileChatRequest(BaseModel):
    """Extended chat request with mobile-specific validation."""

    message: str
    agent_preference: str = "navigator"
    battery_level: Optional[int] = None
    network_type: Optional[str] = None
    data_budget_mb: Optional[int] = None

    @validator('message')
    def message_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Message cannot be empty')
        return v.strip()

    @validator('battery_level')
    def valid_battery_level(cls, v):
        if v is not None and (v < 0 or v > 100):
            raise ValueError('Battery level must be between 0 and 100')
        return v

    @validator('network_type')
    def valid_network_type(cls, v):
        if v is not None and v not in ['wifi', 'cellular', 'ethernet']:
            raise ValueError('Network type must be wifi, cellular, or ethernet')
        return v

    def to_aivillage_request(self) -> ChatRequest:
        """Convert to AIVillage ChatRequest format."""
        user_context = None
        if any([self.battery_level, self.network_type, self.data_budget_mb]):
            user_context = ChatRequestUserContext(
                device_type="mobile",
                battery_level=self.battery_level,
                network_type=self.network_type,
                data_budget_mb=self.data_budget_mb
            )

        return ChatRequest(
            message=self.message,
            agent_preference=self.agent_preference,
            mode="balanced",
            user_context=user_context
        )

# Usage with validation
async def validated_mobile_chat():
    try:
        # This will validate input automatically
        mobile_request = MobileChatRequest(
            message="How can I optimize my app's battery usage?",
            battery_level=45,
            network_type="cellular",
            data_budget_mb=50
        )

        # Convert to API format
        api_request = mobile_request.to_aivillage_request()

        # Send request
        configuration = aivillage_client.Configuration(
            host="https://api.aivillage.io/v1",
            access_token="your-api-key"
        )

        async with aivillage_client.ApiClient(configuration) as api_client:
            chat_api = aivillage_client.ChatApi(api_client)
            response = await chat_api.chat(api_request)

            print(f"Mobile-optimized response: {response.response}")

    except ValueError as e:
        print(f"Validation error: {e}")
```

## Testing

### Unit Testing with pytest

```python
import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from aivillage_client import Configuration, ApiClient, ChatApi
from aivillage_client.models import ChatRequest, ChatResponse

@pytest.fixture
def api_config():
    return Configuration(
        host="https://api.aivillage.io/v1",
        access_token="test-api-key"
    )

@pytest.fixture
def mock_chat_response():
    return ChatResponse(
        response="Mocked response for testing",
        agent_used="test-agent",
        processing_time_ms=100,
        conversation_id="test-conv-123",
        metadata={
            "confidence": 0.95,
            "features_used": ["test-feature"]
        }
    )

@pytest.mark.asyncio
async def test_chat_request(api_config, mock_chat_response):
    """Test basic chat functionality."""

    with patch('aivillage_client.ChatApi.chat', new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = mock_chat_response

        async with ApiClient(api_config) as api_client:
            chat_api = ChatApi(api_client)

            response = await chat_api.chat(
                ChatRequest(
                    message="Test message",
                    mode="fast"
                )
            )

            assert response.response == "Mocked response for testing"
            assert response.agent_used == "test-agent"
            mock_chat.assert_called_once()

@pytest.mark.asyncio
async def test_retry_logic():
    """Test retry logic with rate limiting."""
    from aivillage_client.exceptions import RateLimitException

    configuration = Configuration(
        host="https://api.aivillage.io/v1",
        access_token="test-key"
    )

    with patch('aivillage_client.ChatApi.chat') as mock_chat:
        # First call raises rate limit, second succeeds
        mock_chat.side_effect = [
            RateLimitException("Rate limited", retry_after=1),
            ChatResponse(response="Success", agent_used="test", processing_time_ms=50)
        ]

        async with ApiClient(configuration) as api_client:
            chat_api = ChatApi(api_client)

            # This should retry and succeed
            response = await resilient_chat_request(
                chat_api,
                ChatRequest(message="Test"),
                max_retries=1
            )

            assert response.response == "Success"
            assert mock_chat.call_count == 2

def test_mobile_request_validation():
    """Test mobile request validation."""

    # Valid request
    valid_request = MobileChatRequest(
        message="Test message",
        battery_level=75,
        network_type="wifi"
    )

    assert valid_request.message == "Test message"
    assert valid_request.battery_level == 75

    # Invalid battery level
    with pytest.raises(ValueError, match="Battery level must be between 0 and 100"):
        MobileChatRequest(
            message="Test",
            battery_level=150
        )

    # Invalid network type
    with pytest.raises(ValueError, match="Network type must be"):
        MobileChatRequest(
            message="Test",
            network_type="invalid"
        )

# Integration test example
@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_api_integration():
    """Integration test with real API (requires valid API key)."""
    import os

    api_key = os.getenv("AIVILLAGE_TEST_API_KEY")
    if not api_key:
        pytest.skip("AIVILLAGE_TEST_API_KEY not set")

    configuration = Configuration(
        host="https://staging-api.aivillage.io/v1",
        access_token=api_key
    )

    async with ApiClient(configuration) as api_client:
        chat_api = ChatApi(api_client)

        response = await chat_api.chat(
            ChatRequest(
                message="Hello, this is a test message",
                mode="fast"
            )
        )

        assert response.response is not None
        assert response.agent_used is not None
        assert response.processing_time_ms > 0
```

### Performance Testing

```python
import time
import asyncio
from aivillage_client import Configuration, ApiClient, ChatApi
from aivillage_client.models import ChatRequest

async def benchmark_chat_performance(num_requests: int = 10):
    """Benchmark chat API performance."""

    configuration = Configuration(
        host="https://api.aivillage.io/v1",
        access_token="your-api-key"
    )

    async with ApiClient(configuration) as api_client:
        chat_api = ChatApi(api_client)

        # Sequential requests
        start_time = time.time()
        for i in range(num_requests):
            await chat_api.chat(
                ChatRequest(message=f"Test message {i}", mode="fast")
            )
        sequential_time = time.time() - start_time

        # Concurrent requests
        start_time = time.time()
        tasks = [
            chat_api.chat(ChatRequest(message=f"Concurrent test {i}", mode="fast"))
            for i in range(num_requests)
        ]
        await asyncio.gather(*tasks)
        concurrent_time = time.time() - start_time

        print(f"Sequential: {sequential_time:.2f}s ({num_requests/sequential_time:.1f} req/s)")
        print(f"Concurrent: {concurrent_time:.2f}s ({num_requests/concurrent_time:.1f} req/s)")
        print(f"Speedup: {sequential_time/concurrent_time:.1f}x")

# Run benchmark
if __name__ == "__main__":
    asyncio.run(benchmark_chat_performance())
```

## Deployment

### Production Environment

```python
import os
import logging
from aivillage_client import Configuration

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def create_production_config():
    """Create production-ready configuration."""
    return Configuration(
        host=os.getenv("AIVILLAGE_API_URL", "https://api.aivillage.io/v1"),
        access_token=os.getenv("AIVILLAGE_API_KEY"),
        timeout=float(os.getenv("AIVILLAGE_TIMEOUT", "30.0")),
        retries=int(os.getenv("AIVILLAGE_RETRIES", "3")),
        verify_ssl=os.getenv("AIVILLAGE_VERIFY_SSL", "true").lower() == "true"
    )

# Health check function
async def health_check() -> bool:
    """Check if AIVillage API is healthy."""
    try:
        config = create_production_config()
        async with aivillage_client.ApiClient(config) as api_client:
            health_api = aivillage_client.HealthApi(api_client)
            health = await health_api.get_health()
            return health.status == "healthy"
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        return False
```

### Docker Integration

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . /app
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV AIVILLAGE_API_URL=https://api.aivillage.io/v1

# Run health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import asyncio; from app import health_check; asyncio.run(health_check())" || exit 1

CMD ["python", "app.py"]
```

## Troubleshooting

### Common Issues

**SSL/TLS Errors:**
```python
# Disable SSL verification for development (not recommended for production)
configuration = Configuration(
    host="https://api.aivillage.io/v1",
    access_token="your-api-key",
    verify_ssl=False
)
```

**Timeout Issues:**
```python
# Increase timeout for slow networks
configuration = Configuration(
    host="https://api.aivillage.io/v1",
    access_token="your-api-key",
    timeout=60.0  # 60 seconds
)
```

**Memory Issues with Large Responses:**
```python
# Use streaming for large responses (if supported)
async def handle_large_response():
    async with aivillage_client.ApiClient(configuration) as api_client:
        # Configure smaller chunk sizes for memory-constrained environments
        api_client.set_chunk_size(1024)  # 1KB chunks
```

### Debug Logging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('aivillage_client')
logger.setLevel(logging.DEBUG)

# This will log all HTTP requests and responses
configuration = Configuration(
    host="https://api.aivillage.io/v1",
    access_token="your-api-key",
    debug=True
)
```

## Support

- **Documentation**: [docs.aivillage.io](https://docs.aivillage.io)
- **API Reference**: [docs.aivillage.io/api](https://docs.aivillage.io/api)
- **GitHub Issues**: [github.com/DNYoussef/AIVillage/issues](https://github.com/DNYoussef/AIVillage/issues)
- **Python Package**: [pypi.org/project/aivillage-client](https://pypi.org/project/aivillage-client)

## License

MIT License - see [LICENSE](../../LICENSE) for details.
