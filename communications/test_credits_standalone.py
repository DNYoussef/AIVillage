#!/usr/bin/env python3
"""Standalone test script for credits ledger functionality."""

from datetime import datetime, timezone
import os
import sys
import tempfile

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from credits_ledger import CreditsConfig, CreditsLedger


def test_credits_ledger():
    """Test the credits ledger functionality."""
    print("Testing Credits Ledger...")

    # Create temporary database
    db_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    db_file.close()

    try:
        # Create test configuration
        config = CreditsConfig()
        config.database_url = f"sqlite:///{db_file.name}"
        config.burn_rate = 0.01
        config.fixed_supply = 1000000

        # Create ledger
        ledger = CreditsLedger(config)
        ledger.create_tables()

        # Test 1: Create users
        print("+ Creating users...")
        alice = ledger.create_user("alice", "node_001")
        bob = ledger.create_user("bob", "node_002")

        # Get fresh objects from database
        alice = ledger.get_user("alice")
        bob = ledger.get_user("bob")
        assert alice.username == "alice"
        assert bob.username == "bob"

        # Test 2: Get balance
        print("+ Testing balance retrieval...")
        balance = ledger.get_balance("alice")
        assert balance.balance == 0
        assert balance.username == "alice"

        # Test 3: Earn credits
        print("+ Testing credit earning...")
        scrape_time = datetime.now(timezone.utc)
        earning = ledger.earn_credits(
            "alice",
            scrape_time,
            uptime_seconds=3600,  # 1 hour
            flops=1000000000,  # 1 GFLOP
            bandwidth_bytes=1000000000,  # 1 GB
        )

        # Expected: (1 hour * 10) + (1 GFLOP * 1000) + (1 GB * 1) = 1011
        assert earning.credits_earned == 1011

        # Check balance updated
        balance = ledger.get_balance("alice")
        assert balance.balance == 1011

        # Test 4: Transfer credits
        print("+ Testing credit transfer...")
        transaction = ledger.transfer("alice", "bob", 100)
        assert transaction.amount == 100
        assert transaction.burn_amount == 1  # 1% of 100
        assert transaction.net_amount == 99

        # Check balances
        alice_balance = ledger.get_balance("alice")
        bob_balance = ledger.get_balance("bob")
        assert alice_balance.balance == 911  # 1011 - 100
        assert bob_balance.balance == 99  # 0 + 99

        # Test 5: Idempotent earning
        print("+ Testing idempotent earning...")
        earning2 = ledger.earn_credits(
            "alice",
            scrape_time,  # Same timestamp
            uptime_seconds=7200,  # Different values
            flops=2000000000,
            bandwidth_bytes=2000000000,
        )

        # Should return same earning
        assert earning.id == earning2.id
        assert earning2.uptime_seconds == 3600  # Original values preserved

        # Test 6: Transaction history
        print("+ Testing transaction history...")
        transactions = ledger.get_transactions("alice", limit=10)
        assert len(transactions) == 1
        assert transactions[0].from_user == "alice"
        assert transactions[0].to_user == "bob"

        # Test 7: Total supply
        print("+ Testing total supply...")
        total_supply = ledger.get_total_supply()
        print(f"  Total supply: {total_supply}")
        # Alice had 1011, sent 100 to Bob (99 net), so total should be 1010
        assert total_supply == 1010  # 1011 - 1 (burn)

        # Test 8: Error cases
        print("+ Testing error cases...")
        try:
            ledger.create_user("alice", "node_003")  # Duplicate username
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "already exists" in str(e)

        try:
            ledger.transfer("alice", "bob", 2000)  # Insufficient balance
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Insufficient balance" in str(e)

        print("SUCCESS: All tests passed!")
        return True

    finally:
        # Cleanup
        try:
            os.unlink(db_file.name)
        except (OSError, PermissionError):
            pass  # Ignore cleanup errors


def test_api_endpoints():
    """Test the API endpoints using a simple HTTP client."""
    print("Testing API endpoints...")

    try:
        from credits_api import app, get_ledger
        from fastapi.testclient import TestClient

        # Create test database
        db_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        db_file.close()

        # Override the ledger dependency
        config = CreditsConfig()
        config.database_url = f"sqlite:///{db_file.name}"
        test_ledger = CreditsLedger(config)
        test_ledger.create_tables()

        def override_get_ledger():
            return test_ledger

        app.dependency_overrides[get_ledger] = override_get_ledger

        with TestClient(app) as client:
            # Test 1: Health check
            print("+ Testing health check...")
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"

            # Test 2: Create user
            print("+ Testing user creation...")
            response = client.post(
                "/users", json={"username": "testuser", "node_id": "node_123"}
            )
            assert response.status_code == 201
            data = response.json()
            assert data["username"] == "testuser"

            # Test 3: Get balance
            print("+ Testing balance retrieval...")
            response = client.get("/balance/testuser")
            assert response.status_code == 200
            data = response.json()
            assert data["balance"] == 0

            # Test 4: Earn credits
            print("+ Testing credit earning...")
            scrape_time = datetime.now(timezone.utc)
            response = client.post(
                "/earn",
                json={
                    "username": "testuser",
                    "scrape_timestamp": scrape_time.isoformat(),
                    "uptime_seconds": 3600,
                    "flops": 1000000000,
                    "bandwidth_bytes": 1000000000,
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert data["credits_earned"] == 1011

            # Test 5: Create second user and transfer
            print("+ Testing transfer...")
            client.post("/users", json={"username": "recipient", "node_id": "node_456"})

            response = client.post(
                "/transfer",
                json={
                    "from_username": "testuser",
                    "to_username": "recipient",
                    "amount": 100,
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert data["amount"] == 100
            assert data["burn_amount"] == 1
            assert data["net_amount"] == 99

            # Test 6: Get transactions
            print("+ Testing transaction history...")
            response = client.get("/transactions/testuser")
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["from_user"] == "testuser"

            # Test 7: Get total supply
            print("+ Testing total supply...")
            response = client.get("/supply")
            assert response.status_code == 200
            data = response.json()
            assert data["total_supply"] == 1010  # 1011 - 1 (burn)

        print("SUCCESS: All API tests passed!")

        # Cleanup
        try:
            os.unlink(db_file.name)
        except (OSError, PermissionError):
            pass  # Ignore cleanup errors
        return True

    except ImportError as e:
        print(f"WARNING: Skipping API tests due to missing dependencies: {e}")
        return True


def main():
    """Run all tests."""
    print("Running Credits Ledger MVP Tests")
    print("=" * 40)

    success = True

    try:
        success &= test_credits_ledger()
        success &= test_api_endpoints()

        if success:
            print("\nSUCCESS: All tests passed! Credits ledger is working correctly.")
            return 0
        print("\nFAILED: Some tests failed.")
        return 1

    except Exception as e:
        print(f"\nERROR: Test execution failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
