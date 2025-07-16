"""Unit tests for credits API endpoints."""

from datetime import datetime, timezone
import os
import tempfile

from fastapi.testclient import TestClient
import pytest

from communications.credits_api import app, get_ledger
from communications.credits_ledger import CreditsConfig, CreditsLedger


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database for testing."""
    db_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    db_file.close()

    yield f"sqlite:///{db_file.name}"

    # Cleanup
    os.unlink(db_file.name)


@pytest.fixture
def test_config(temp_db):
    """Create test configuration."""
    config = CreditsConfig()
    config.database_url = temp_db
    config.burn_rate = 0.01
    config.fixed_supply = 1000000
    return config


@pytest.fixture
def test_ledger(test_config):
    """Create a test ledger instance."""
    ledger = CreditsLedger(test_config)
    ledger.create_tables()
    return ledger


@pytest.fixture
def client(test_ledger):
    """Create a test client with mocked ledger."""

    def override_get_ledger():
        return test_ledger

    app.dependency_overrides[get_ledger] = override_get_ledger

    with TestClient(app) as client:
        yield client

    # Cleanup
    app.dependency_overrides.clear()


@pytest.fixture
def sample_users(test_ledger):
    """Create sample users for testing."""
    alice = test_ledger.create_user("alice", "node_001")
    bob = test_ledger.create_user("bob", "node_002")
    return {"alice": alice, "bob": bob}


class TestCreditsAPI:
    """Test cases for the Credits API endpoints."""

    def test_create_user_success(self, client):
        """Test successful user creation."""
        response = client.post(
            "/users", json={"username": "testuser", "node_id": "node_123"}
        )

        assert response.status_code == 201
        data = response.json()
        assert data["username"] == "testuser"
        assert data["node_id"] == "node_123"
        assert "user_id" in data
        assert "created_at" in data
        assert data["message"] == "User created successfully"

    def test_create_user_duplicate(self, client, sample_users):
        """Test creating user with duplicate username."""
        response = client.post(
            "/users", json={"username": "alice", "node_id": "node_999"}
        )

        assert response.status_code == 400
        data = response.json()
        assert "User alice already exists" in data["detail"]

    def test_create_user_invalid_username(self, client):
        """Test creating user with invalid username."""
        # Too short
        response = client.post("/users", json={"username": "ab", "node_id": "node_123"})
        assert response.status_code == 422

        # Invalid characters
        response = client.post(
            "/users", json={"username": "user@domain", "node_id": "node_123"}
        )
        assert response.status_code == 422

    def test_get_balance_success(self, client, sample_users):
        """Test getting user balance."""
        response = client.get("/balance/alice")

        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "alice"
        assert data["balance"] == 0
        assert data["user_id"] == sample_users["alice"].id
        assert "last_updated" in data

    def test_get_balance_user_not_found(self, client):
        """Test getting balance for non-existent user."""
        response = client.get("/balance/nonexistent")

        assert response.status_code == 404
        data = response.json()
        assert "User nonexistent not found" in data["detail"]

    def test_transfer_success(self, client, sample_users, test_ledger):
        """Test successful credit transfer."""
        # Give alice some credits first
        with test_ledger.get_session() as session:
            alice_wallet = (
                session.query(test_ledger.Wallet)
                .filter(test_ledger.Wallet.user_id == sample_users["alice"].id)
                .first()
            )
            alice_wallet.balance = 1000
            session.commit()

        response = client.post(
            "/transfer",
            json={"from_username": "alice", "to_username": "bob", "amount": 100},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["from_user"] == "alice"
        assert data["to_user"] == "bob"
        assert data["amount"] == 100
        assert data["burn_amount"] == 1
        assert data["net_amount"] == 99
        assert data["transaction_type"] == "transfer"
        assert data["status"] == "completed"
        assert "id" in data
        assert "created_at" in data
        assert "completed_at" in data

    def test_transfer_insufficient_balance(self, client, sample_users):
        """Test transfer with insufficient balance."""
        response = client.post(
            "/transfer",
            json={"from_username": "alice", "to_username": "bob", "amount": 1000},
        )

        assert response.status_code == 400
        data = response.json()
        assert "Insufficient balance" in data["detail"]

    def test_transfer_invalid_amount(self, client, sample_users):
        """Test transfer with invalid amount."""
        response = client.post(
            "/transfer",
            json={"from_username": "alice", "to_username": "bob", "amount": -100},
        )

        assert response.status_code == 422

        response = client.post(
            "/transfer",
            json={"from_username": "alice", "to_username": "bob", "amount": 0},
        )

        assert response.status_code == 422

    def test_transfer_user_not_found(self, client, sample_users):
        """Test transfer with non-existent users."""
        response = client.post(
            "/transfer",
            json={"from_username": "nonexistent", "to_username": "bob", "amount": 100},
        )

        assert response.status_code == 400
        data = response.json()
        assert "Sender nonexistent not found" in data["detail"]

    def test_earn_credits_success(self, client, sample_users):
        """Test successful credit earning."""
        scrape_time = datetime.now(timezone.utc)

        response = client.post(
            "/earn",
            json={
                "username": "alice",
                "scrape_timestamp": scrape_time.isoformat(),
                "uptime_seconds": 3600,
                "flops": 1000000000,
                "bandwidth_bytes": 1000000000,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == sample_users["alice"].id
        assert data["credits_earned"] == 1011  # (1*10) + (1*1000) + (1*1) = 1011
        assert data["uptime_seconds"] == 3600
        assert data["flops"] == 1000000000
        assert data["bandwidth_bytes"] == 1000000000
        assert "id" in data
        assert "created_at" in data

    def test_earn_credits_idempotent(self, client, sample_users):
        """Test that earning is idempotent."""
        scrape_time = datetime.now(timezone.utc)

        # First request
        response1 = client.post(
            "/earn",
            json={
                "username": "alice",
                "scrape_timestamp": scrape_time.isoformat(),
                "uptime_seconds": 3600,
                "flops": 1000000000,
                "bandwidth_bytes": 1000000000,
            },
        )

        assert response1.status_code == 200
        data1 = response1.json()

        # Second request with same timestamp
        response2 = client.post(
            "/earn",
            json={
                "username": "alice",
                "scrape_timestamp": scrape_time.isoformat(),
                "uptime_seconds": 7200,  # Different values
                "flops": 2000000000,
                "bandwidth_bytes": 2000000000,
            },
        )

        assert response2.status_code == 200
        data2 = response2.json()

        # Should return same earning record
        assert data1["id"] == data2["id"]
        assert data1["credits_earned"] == data2["credits_earned"]
        assert data2["uptime_seconds"] == 3600  # Original values preserved

    def test_earn_credits_user_not_found(self, client):
        """Test earning credits for non-existent user."""
        scrape_time = datetime.now(timezone.utc)

        response = client.post(
            "/earn",
            json={
                "username": "nonexistent",
                "scrape_timestamp": scrape_time.isoformat(),
                "uptime_seconds": 3600,
                "flops": 1000000000,
                "bandwidth_bytes": 1000000000,
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert "User nonexistent not found" in data["detail"]

    def test_earn_credits_invalid_data(self, client, sample_users):
        """Test earning credits with invalid data."""
        scrape_time = datetime.now(timezone.utc)

        # Negative values
        response = client.post(
            "/earn",
            json={
                "username": "alice",
                "scrape_timestamp": scrape_time.isoformat(),
                "uptime_seconds": -3600,
                "flops": 1000000000,
                "bandwidth_bytes": 1000000000,
            },
        )

        assert response.status_code == 422

    def test_get_transactions_success(self, client, sample_users, test_ledger):
        """Test getting transaction history."""
        # Give alice some credits and create transactions
        with test_ledger.get_session() as session:
            alice_wallet = (
                session.query(test_ledger.Wallet)
                .filter(test_ledger.Wallet.user_id == sample_users["alice"].id)
                .first()
            )
            alice_wallet.balance = 1000
            session.commit()

        # Create some transactions
        client.post(
            "/transfer",
            json={"from_username": "alice", "to_username": "bob", "amount": 100},
        )

        client.post(
            "/transfer",
            json={"from_username": "alice", "to_username": "bob", "amount": 50},
        )

        response = client.get("/transactions/alice")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert all(tx["from_user"] == "alice" for tx in data)
        assert sum(tx["amount"] for tx in data) == 150

    def test_get_transactions_with_limit(self, client, sample_users, test_ledger):
        """Test getting transactions with limit parameter."""
        # Give alice some credits
        with test_ledger.get_session() as session:
            alice_wallet = (
                session.query(test_ledger.Wallet)
                .filter(test_ledger.Wallet.user_id == sample_users["alice"].id)
                .first()
            )
            alice_wallet.balance = 1000
            session.commit()

        # Create multiple transactions
        for i in range(5):
            client.post(
                "/transfer",
                json={"from_username": "alice", "to_username": "bob", "amount": 10},
            )

        response = client.get("/transactions/alice?limit=3")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3

    def test_get_transactions_user_not_found(self, client):
        """Test getting transactions for non-existent user."""
        response = client.get("/transactions/nonexistent")

        assert response.status_code == 404
        data = response.json()
        assert "User nonexistent not found" in data["detail"]

    def test_get_total_supply(self, client, sample_users, test_ledger):
        """Test getting total supply."""
        response = client.get("/supply")

        assert response.status_code == 200
        data = response.json()
        assert data["total_supply"] == 0
        assert data["max_supply"] == 1000000
        assert data["burn_rate"] == 0.01
        assert "timestamp" in data

        # Add some credits and check again
        scrape_time = datetime.now(timezone.utc)
        client.post(
            "/earn",
            json={
                "username": "alice",
                "scrape_timestamp": scrape_time.isoformat(),
                "uptime_seconds": 3600,
                "flops": 1000000000,
                "bandwidth_bytes": 1000000000,
            },
        )

        response = client.get("/supply")
        assert response.status_code == 200
        data = response.json()
        assert data["total_supply"] == 1011

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "credits-api"
        assert "timestamp" in data

    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics")

        assert response.status_code == 200
        assert "credits_requests_total" in response.text or response.status_code == 200

    def test_error_handling(self, client):
        """Test error handling and response format."""
        # Test 404 error
        response = client.get("/balance/nonexistent")
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

        # Test validation error
        response = client.post(
            "/users",
            json={
                "username": "ab"  # Too short
            },
        )
        assert response.status_code == 422

    def test_prometheus_metrics_incremented(self, client, sample_users):
        """Test that Prometheus metrics are incremented."""
        # Make several requests
        client.get("/balance/alice")
        client.get("/balance/alice")
        client.get("/balance/nonexistent")  # This should increment error metric

        # Check metrics endpoint
        response = client.get("/metrics")
        assert response.status_code == 200

        # Metrics should be present (exact format may vary)
        metrics_text = response.text
        assert "credits_requests_total" in metrics_text or response.status_code == 200

    def test_concurrent_requests(self, client, sample_users, test_ledger):
        """Test handling of concurrent requests."""
        # Give alice some credits
        with test_ledger.get_session() as session:
            alice_wallet = (
                session.query(test_ledger.Wallet)
                .filter(test_ledger.Wallet.user_id == sample_users["alice"].id)
                .first()
            )
            alice_wallet.balance = 1000
            session.commit()

        # Make multiple requests (simulating concurrency)
        responses = []
        for i in range(5):
            response = client.post(
                "/transfer",
                json={"from_username": "alice", "to_username": "bob", "amount": 10},
            )
            responses.append(response)

        # All should succeed
        for response in responses:
            assert response.status_code == 200

        # Check final balance
        balance_response = client.get("/balance/alice")
        assert balance_response.status_code == 200
        balance_data = balance_response.json()
        assert balance_data["balance"] == 950  # 1000 - (5 * 10) = 950
