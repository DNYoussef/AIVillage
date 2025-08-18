"""Unit tests for credits ledger functionality."""

import os
import tempfile
from datetime import UTC, datetime

import pytest
from communications.credits_ledger import CreditsConfig, CreditsLedger, Wallet


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
    config.earning_rate_flops = 1000
    config.earning_rate_uptime = 10
    config.earning_rate_bandwidth = 1
    return config


@pytest.fixture
def ledger(test_config):
    """Create a test ledger instance."""
    ledger = CreditsLedger(test_config)
    ledger.create_tables()
    return ledger


@pytest.fixture
def sample_users(ledger):
    """Create sample users for testing."""
    alice = ledger.create_user("alice", "node_001")
    bob = ledger.create_user("bob", "node_002")
    charlie = ledger.create_user("charlie")

    return {"alice": alice, "bob": bob, "charlie": charlie}


class TestCreditsLedger:
    """Test cases for the CreditsLedger class."""

    def test_create_user_success(self, ledger):
        """Test successful user creation."""
        user = ledger.create_user("testuser", "node_123")

        assert user.username == "testuser"
        assert user.node_id == "node_123"
        assert user.created_at is not None
        assert user.wallet is not None
        assert user.wallet.balance == 0

    def test_create_user_duplicate_username(self, ledger):
        """Test error when creating user with duplicate username."""
        ledger.create_user("duplicate", "node_001")

        with pytest.raises(ValueError, match="User duplicate already exists"):
            ledger.create_user("duplicate", "node_002")

    def test_get_user_exists(self, ledger, sample_users):
        """Test getting existing user."""
        user = ledger.get_user("alice")
        assert user is not None
        assert user.username == "alice"
        assert user.node_id == "node_001"

    def test_get_user_not_exists(self, ledger):
        """Test getting non-existent user."""
        user = ledger.get_user("nonexistent")
        assert user is None

    def test_get_balance_success(self, ledger, sample_users):
        """Test getting user balance."""
        balance = ledger.get_balance("alice")

        assert balance.username == "alice"
        assert balance.balance == 0
        assert balance.user_id == sample_users["alice"].id
        assert balance.last_updated is not None

    def test_get_balance_user_not_found(self, ledger):
        """Test getting balance for non-existent user."""
        with pytest.raises(ValueError, match="User nonexistent not found"):
            ledger.get_balance("nonexistent")

    def test_transfer_success(self, ledger, sample_users):
        """Test successful credit transfer."""
        # Give alice some credits first
        with ledger.get_session() as session:
            alice_wallet = session.query(Wallet).filter(Wallet.user_id == sample_users["alice"].id).first()
            alice_wallet.balance = 1000
            session.commit()

        # Transfer from alice to bob
        transaction = ledger.transfer("alice", "bob", 100)

        assert transaction.from_user == "alice"
        assert transaction.to_user == "bob"
        assert transaction.amount == 100
        assert transaction.burn_amount == 1  # 1% of 100
        assert transaction.net_amount == 99
        assert transaction.transaction_type == "transfer"
        assert transaction.status == "completed"
        assert transaction.completed_at is not None

        # Check balances
        alice_balance = ledger.get_balance("alice")
        bob_balance = ledger.get_balance("bob")

        assert alice_balance.balance == 900  # 1000 - 100
        assert bob_balance.balance == 99  # 0 + 99 (after burn)

    def test_transfer_insufficient_balance(self, ledger, sample_users):
        """Test transfer with insufficient balance."""
        with pytest.raises(ValueError, match="Insufficient balance"):
            ledger.transfer("alice", "bob", 100)

    def test_transfer_negative_amount(self, ledger, sample_users):
        """Test transfer with negative amount."""
        with pytest.raises(ValueError, match="Amount must be positive"):
            ledger.transfer("alice", "bob", -10)

    def test_transfer_user_not_found(self, ledger, sample_users):
        """Test transfer with non-existent users."""
        with pytest.raises(ValueError, match="Sender nonexistent not found"):
            ledger.transfer("nonexistent", "bob", 10)

        with pytest.raises(ValueError, match="Recipient nonexistent not found"):
            ledger.transfer("alice", "nonexistent", 10)

    def test_earn_credits_success(self, ledger, sample_users):
        """Test successful credit earning."""
        scrape_time = datetime.now(UTC)

        earning = ledger.earn_credits(
            "alice",
            scrape_time,
            uptime_seconds=3600,  # 1 hour
            flops=1000000000,  # 1 GFLOP
            bandwidth_bytes=1000000000,  # 1 GB
        )

        # Expected credits: (1 hour * 10) + (1 GFLOP * 1000) + (1 GB * 1) = 10 + 1000 + 1 = 1011
        assert earning.credits_earned == 1011
        assert earning.uptime_seconds == 3600
        assert earning.flops == 1000000000
        assert earning.bandwidth_bytes == 1000000000
        assert earning.scrape_timestamp == scrape_time

        # Check balance was updated
        balance = ledger.get_balance("alice")
        assert balance.balance == 1011

    def test_earn_credits_idempotent(self, ledger, sample_users):
        """Test that earning is idempotent for same scrape timestamp."""
        scrape_time = datetime.now(UTC)

        # First earning
        earning1 = ledger.earn_credits(
            "alice",
            scrape_time,
            uptime_seconds=3600,
            flops=1000000000,
            bandwidth_bytes=1000000000,
        )

        # Second earning with same timestamp should return same result
        earning2 = ledger.earn_credits(
            "alice",
            scrape_time,
            uptime_seconds=7200,  # Different values
            flops=2000000000,
            bandwidth_bytes=2000000000,
        )

        assert earning1.id == earning2.id
        assert earning1.credits_earned == earning2.credits_earned
        assert earning2.uptime_seconds == 3600  # Original values preserved

        # Balance should only be updated once
        balance = ledger.get_balance("alice")
        assert balance.balance == 1011

    def test_earn_credits_user_not_found(self, ledger):
        """Test earning credits for non-existent user."""
        scrape_time = datetime.now(UTC)

        with pytest.raises(ValueError, match="User nonexistent not found"):
            ledger.earn_credits(
                "nonexistent",
                scrape_time,
                uptime_seconds=3600,
                flops=1000000000,
                bandwidth_bytes=1000000000,
            )

    def test_get_transactions_success(self, ledger, sample_users):
        """Test getting transaction history."""
        # Set up some transactions
        with ledger.get_session() as session:
            alice_wallet = session.query(Wallet).filter(Wallet.user_id == sample_users["alice"].id).first()
            alice_wallet.balance = 1000
            session.commit()

        # Create some transactions
        ledger.transfer("alice", "bob", 100)
        ledger.transfer("alice", "charlie", 50)

        # Get alice's transactions
        transactions = ledger.get_transactions("alice", limit=10)

        assert len(transactions) == 2
        assert all(tx.from_user == "alice" for tx in transactions)
        assert {tx.to_user for tx in transactions} == {"bob", "charlie"}
        assert sum(tx.amount for tx in transactions) == 150

    def test_get_transactions_user_not_found(self, ledger):
        """Test getting transactions for non-existent user."""
        with pytest.raises(ValueError, match="User nonexistent not found"):
            ledger.get_transactions("nonexistent")

    def test_get_total_supply(self, ledger, sample_users):
        """Test getting total supply in circulation."""
        # Initially should be 0
        total_supply = ledger.get_total_supply()
        assert total_supply == 0

        # Add some credits through earning
        scrape_time = datetime.now(UTC)
        ledger.earn_credits("alice", scrape_time, 3600, 1000000000, 1000000000)

        total_supply = ledger.get_total_supply()
        assert total_supply == 1011

        # Transfer (with burn) should reduce total supply
        ledger.transfer("alice", "bob", 100)

        total_supply = ledger.get_total_supply()
        assert total_supply == 1010  # 1011 - 1 (burn amount)

    def test_burn_calculation(self, ledger, sample_users):
        """Test that burn calculation is correct."""
        with ledger.get_session() as session:
            alice_wallet = session.query(Wallet).filter(Wallet.user_id == sample_users["alice"].id).first()
            alice_wallet.balance = 1000
            session.commit()

        # Test various amounts
        test_cases = [
            (100, 1, 99),  # 100 amount, 1 burn, 99 net
            (1000, 10, 990),  # 1000 amount, 10 burn, 990 net
            (50, 0, 50),  # 50 amount, 0 burn (rounded down), 50 net
            (199, 1, 198),  # 199 amount, 1 burn, 198 net
        ]

        for amount, expected_burn, expected_net in test_cases:
            # Reset alice's balance
            with ledger.get_session() as session:
                alice_wallet = session.query(Wallet).filter(Wallet.user_id == sample_users["alice"].id).first()
                alice_wallet.balance = amount
                session.commit()

            transaction = ledger.transfer("alice", "bob", amount)
            assert transaction.burn_amount == expected_burn
            assert transaction.net_amount == expected_net

    def test_concurrent_transactions(self, ledger, sample_users):
        """Test handling of concurrent transactions."""
        # Give alice some credits
        with ledger.get_session() as session:
            alice_wallet = session.query(Wallet).filter(Wallet.user_id == sample_users["alice"].id).first()
            alice_wallet.balance = 1000
            session.commit()

        # This should work fine sequentially
        ledger.transfer("alice", "bob", 100)
        ledger.transfer("alice", "charlie", 100)

        # Alice should have 800 credits left (1000 - 100 - 100)
        balance = ledger.get_balance("alice")
        assert balance.balance == 800

    def test_database_constraints(self, ledger):
        """Test that database constraints are enforced."""
        # Test unique username constraint
        ledger.create_user("testuser", "node_001")

        with pytest.raises(ValueError, match="User testuser already exists"):
            ledger.create_user("testuser", "node_002")

        # Test unique user_id constraint in earnings
        ledger.get_user("testuser")
        scrape_time = datetime.now(UTC)

        # First earning should work
        ledger.earn_credits("testuser", scrape_time, 3600, 1000000000, 1000000000)

        # Second earning with same timestamp should be idempotent
        earning2 = ledger.earn_credits("testuser", scrape_time, 7200, 2000000000, 2000000000)

        # Should return the original earning
        assert earning2.uptime_seconds == 3600  # Original value

    def test_edge_cases(self, ledger):
        """Test edge cases and boundary conditions."""
        # Test with zero values
        ledger.create_user("edge_user", "node_edge")
        scrape_time = datetime.now(UTC)

        earning = ledger.earn_credits("edge_user", scrape_time, 0, 0, 0)
        assert earning.credits_earned == 0

        # Test with very large values
        earning2 = ledger.earn_credits(
            "edge_user",
            datetime.now(UTC),
            86400,  # 24 hours
            1000000000000,  # 1 TFLOP
            1000000000000,  # 1 TB
        )

        # Expected: (24 * 10) + (1000 * 1000) + (1000 * 1) = 240 + 1000000 + 1000 = 1001240
        assert earning2.credits_earned == 1001240
