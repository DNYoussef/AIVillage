"""
Pytest configuration and shared fixtures for market testing.

Provides common test fixtures, mocks, and utilities shared across
all market-based pricing system tests.
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    redis_mock = MagicMock()

    # Mock common Redis operations
    redis_mock.get = AsyncMock(return_value=None)
    redis_mock.set = AsyncMock(return_value=True)
    redis_mock.hget = AsyncMock(return_value=None)
    redis_mock.hset = AsyncMock(return_value=True)
    redis_mock.lpush = AsyncMock(return_value=1)
    redis_mock.lrange = AsyncMock(return_value=[])
    redis_mock.expire = AsyncMock(return_value=True)

    return redis_mock


@pytest.fixture
def mock_database():
    """Mock database connection for testing."""
    db_mock = MagicMock()

    # Mock database operations
    db_mock.execute = AsyncMock(return_value=MagicMock(rowcount=1))
    db_mock.fetch_one = AsyncMock(return_value=None)
    db_mock.fetch_all = AsyncMock(return_value=[])
    db_mock.begin = AsyncMock()
    db_mock.commit = AsyncMock()
    db_mock.rollback = AsyncMock()

    return db_mock


@pytest.fixture
def mock_tokenomics_service():
    """Mock tokenomics service for testing."""
    service_mock = MagicMock()

    # Mock tokenomics operations
    service_mock.get_balance = AsyncMock(return_value=Decimal("1000.0"))
    service_mock.validate_transaction = AsyncMock(return_value=True)
    service_mock.process_payment = AsyncMock(return_value="tx_123")
    service_mock.hold_escrow = AsyncMock(return_value="escrow_456")
    service_mock.release_escrow = AsyncMock(return_value=True)
    service_mock.refund_escrow = AsyncMock(return_value=True)

    return service_mock


@pytest.fixture
def mock_fog_scheduler():
    """Mock fog scheduler for testing."""
    scheduler_mock = MagicMock()

    # Mock scheduler operations
    scheduler_mock.submit_task = AsyncMock(return_value="task_789")
    scheduler_mock.get_task_status = AsyncMock(return_value={
        "status": "running",
        "progress": 0.5,
        "resources_allocated": True
    })
    scheduler_mock.cancel_task = AsyncMock(return_value=True)
    scheduler_mock.get_resource_utilization = AsyncMock(return_value={
        "cpu": Decimal("0.65"),
        "memory": Decimal("0.72"),
        "storage": Decimal("0.48")
    })

    return scheduler_mock


@pytest.fixture
def sample_market_conditions():
    """Sample market condition data for testing."""
    return {
        "timestamp": datetime.utcnow(),
        "cpu_utilization": Decimal("0.68"),
        "memory_utilization": Decimal("0.71"),
        "storage_utilization": Decimal("0.52"),
        "bandwidth_utilization": Decimal("0.77"),
        "gpu_utilization": Decimal("0.34"),
        "demand_multiplier": Decimal("1.15"),
        "supply_multiplier": Decimal("0.92"),
        "volatility_index": Decimal("0.18"),
        "manipulation_risk": Decimal("0.05")
    }


@pytest.fixture
def sample_resource_requirements():
    """Sample resource requirements for testing."""
    return {
        "cpu_cores": Decimal("4"),
        "memory_gb": Decimal("8"),
        "storage_gb": Decimal("100"),
        "bandwidth_mbps": Decimal("1000"),
        "gpu_units": Decimal("0"),
        "specialized_hardware": [],
        "duration_hours": Decimal("24"),
        "quality_requirements": {
            "min_trust_score": Decimal("0.8"),
            "min_reputation": Decimal("0.7"),
            "max_latency_ms": 50,
            "availability_requirement": Decimal("0.99")
        }
    }


@pytest.fixture
def sample_provider_profiles():
    """Sample provider profiles for testing."""
    return [
        {
            "provider_id": "test_provider_1",
            "node_id": "test_node_1",
            "trust_score": Decimal("0.95"),
            "reputation_score": Decimal("0.92"),
            "available_resources": {
                "cpu_cores": Decimal("16"),
                "memory_gb": Decimal("32"),
                "storage_gb": Decimal("1000"),
                "bandwidth_mbps": Decimal("5000")
            },
            "pricing_history": [
                {"timestamp": datetime.utcnow(), "price": Decimal("0.10")},
                {"timestamp": datetime.utcnow() - timedelta(hours=1), "price": Decimal("0.09")},
            ],
            "location": "us-east-1",
            "uptime_percentage": Decimal("99.8")
        },
        {
            "provider_id": "test_provider_2",
            "node_id": "test_node_2",
            "trust_score": Decimal("0.87"),
            "reputation_score": Decimal("0.84"),
            "available_resources": {
                "cpu_cores": Decimal("8"),
                "memory_gb": Decimal("16"),
                "storage_gb": Decimal("500"),
                "bandwidth_mbps": Decimal("2000"),
                "gpu_units": Decimal("2")
            },
            "pricing_history": [
                {"timestamp": datetime.utcnow(), "price": Decimal("0.12")},
                {"timestamp": datetime.utcnow() - timedelta(hours=1), "price": Decimal("0.11")},
            ],
            "location": "us-west-2",
            "uptime_percentage": Decimal("99.5"),
            "specialized_hardware": ["cuda", "tensor_rt"]
        }
    ]


@pytest.fixture
def sample_auction_config():
    """Sample auction configuration for testing."""
    return {
        "default_duration_minutes": 30,
        "minimum_deposit_percentage": Decimal("0.10"),  # 10% of reserve price
        "quality_weight": Decimal("0.30"),  # 30% weight for quality scoring
        "price_weight": Decimal("0.70"),   # 70% weight for price scoring
        "settlement_type": "second_price", # Vickrey auction
        "anti_griefing_threshold": Decimal("0.75"),
        "circuit_breaker_multiplier": Decimal("3.0")
    }


@pytest.fixture
def sample_pricing_config():
    """Sample pricing configuration for testing."""
    return {
        "base_prices": {
            "cpu": Decimal("0.08"),
            "memory": Decimal("0.02"),
            "storage": Decimal("0.001"),
            "bandwidth": Decimal("0.0001"),
            "gpu": Decimal("1.20"),
            "specialized": Decimal("2.00")
        },
        "multipliers": {
            "demand_low": Decimal("0.8"),
            "demand_normal": Decimal("1.0"),
            "demand_high": Decimal("1.5"),
            "demand_premium": Decimal("2.2")
        },
        "volatility_threshold": Decimal("0.25"),
        "circuit_breaker_threshold": Decimal("5.0"),
        "bulk_discount_tiers": [
            {"min_quantity": Decimal("100"), "discount": Decimal("0.05")},
            {"min_quantity": Decimal("1000"), "discount": Decimal("0.10")},
            {"min_quantity": Decimal("10000"), "discount": Decimal("0.15")}
        ]
    }


@pytest.fixture
def mock_performance_metrics():
    """Mock performance metrics for testing."""
    return {
        "auction_metrics": {
            "average_settlement_time_seconds": Decimal("45.3"),
            "successful_auctions_percentage": Decimal("0.94"),
            "average_bid_count": Decimal("5.7"),
            "price_discovery_efficiency": Decimal("0.89")
        },
        "pricing_metrics": {
            "price_accuracy_score": Decimal("0.91"),
            "volatility_prediction_accuracy": Decimal("0.84"),
            "circuit_breaker_activations": 3,
            "market_manipulation_detections": 1
        },
        "system_metrics": {
            "average_response_time_ms": Decimal("120.5"),
            "throughput_requests_per_second": Decimal("85.7"),
            "error_rate_percentage": Decimal("0.02"),
            "availability_percentage": Decimal("99.95")
        }
    }


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Automatically setup test environment for each test."""
    # Setup code that runs before each test
    test_start_time = datetime.utcnow()

    yield  # This is where the test runs

    # Cleanup code that runs after each test
    test_duration = datetime.utcnow() - test_start_time
    if test_duration.total_seconds() > 30:
        print(f"Warning: Test took {test_duration.total_seconds():.2f} seconds")


# Custom pytest markers for organizing tests
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "auction: mark test as auction engine test"
    )
    config.addinivalue_line(
        "markers", "pricing: mark test as pricing manager test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )


# Async test utilities
class AsyncContextManager:
    """Utility class for async context management in tests."""

    def __init__(self, mock_obj):
        self.mock_obj = mock_obj

    async def __aenter__(self):
        return self.mock_obj

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def async_context_manager():
    """Factory for creating async context managers in tests."""
    return AsyncContextManager
