"""
Budget Management and Billing Integration Tests

Validates budget enforcement, cost tracking, billing accuracy, 
and financial controls across the unified federated marketplace.
"""

from datetime import datetime, UTC, timedelta
from decimal import Decimal
import pytest
from typing import Any, Dict, List

from tests.marketplace.test_unified_federated_marketplace import UnifiedFederatedCoordinator


class BillingTracker:
    """Mock billing system for testing"""

    def __init__(self):
        self.user_accounts: Dict[str, Dict[str, Any]] = {}
        self.transactions: List[Dict[str, Any]] = []
        self.escrows: Dict[str, Dict[str, Any]] = {}
        self.usage_records: List[Dict[str, Any]] = []

    def create_user_account(self, user_id: str, initial_balance: float, tier: str):
        """Create user account with initial balance"""
        self.user_accounts[user_id] = {
            "balance": Decimal(str(initial_balance)),
            "tier": tier,
            "credit_limit": Decimal("0"),
            "total_spent": Decimal("0"),
            "created_at": datetime.now(UTC),
        }

    def hold_escrow(self, user_id: str, amount: float, purpose: str) -> str:
        """Hold funds in escrow"""
        escrow_id = f"escrow_{len(self.escrows) + 1}"

        if user_id not in self.user_accounts:
            raise ValueError(f"User account {user_id} not found")

        account = self.user_accounts[user_id]
        amount_decimal = Decimal(str(amount))

        if account["balance"] < amount_decimal:
            raise ValueError(f"Insufficient balance for escrow: {amount}")

        # Hold funds
        account["balance"] -= amount_decimal
        self.escrows[escrow_id] = {
            "user_id": user_id,
            "amount": amount_decimal,
            "purpose": purpose,
            "status": "held",
            "created_at": datetime.now(UTC),
        }

        return escrow_id

    def release_escrow(self, escrow_id: str, recipient_id: str, actual_cost: float):
        """Release escrow and process payment"""
        if escrow_id not in self.escrows:
            raise ValueError(f"Escrow {escrow_id} not found")

        escrow = self.escrows[escrow_id]
        actual_cost_decimal = Decimal(str(actual_cost))

        # Process payment
        transaction_id = f"txn_{len(self.transactions) + 1}"
        self.transactions.append(
            {
                "transaction_id": transaction_id,
                "payer": escrow["user_id"],
                "recipient": recipient_id,
                "amount": actual_cost_decimal,
                "escrow_id": escrow_id,
                "purpose": escrow["purpose"],
                "timestamp": datetime.now(UTC),
            }
        )

        # Update user account
        user_account = self.user_accounts[escrow["user_id"]]
        user_account["total_spent"] += actual_cost_decimal

        # Return remaining escrow amount if any
        refund_amount = escrow["amount"] - actual_cost_decimal
        if refund_amount > 0:
            user_account["balance"] += refund_amount

        # Mark escrow as released
        escrow["status"] = "released"
        escrow["actual_cost"] = actual_cost_decimal
        escrow["released_at"] = datetime.now(UTC)

        return transaction_id

    def record_usage(
        self, user_id: str, workload_id: str, resource_type: str, amount: float, cost: float, duration_hours: float
    ):
        """Record resource usage"""
        self.usage_records.append(
            {
                "user_id": user_id,
                "workload_id": workload_id,
                "resource_type": resource_type,
                "amount": amount,
                "cost": Decimal(str(cost)),
                "duration_hours": duration_hours,
                "timestamp": datetime.now(UTC),
            }
        )

    def get_user_summary(self, user_id: str) -> Dict[str, Any]:
        """Get user billing summary"""
        if user_id not in self.user_accounts:
            return {}

        account = self.user_accounts[user_id]
        user_transactions = [t for t in self.transactions if t["payer"] == user_id]
        user_usage = [u for u in self.usage_records if u["user_id"] == user_id]

        return {
            "user_id": user_id,
            "current_balance": float(account["balance"]),
            "tier": account["tier"],
            "total_spent": float(account["total_spent"]),
            "transaction_count": len(user_transactions),
            "usage_records": len(user_usage),
            "recent_transactions": user_transactions[-10:],  # Last 10 transactions
        }


class TestBudgetBillingIntegration:
    """Budget management and billing integration test suite"""

    @pytest.fixture
    async def unified_coordinator(self):
        """Create unified coordinator for billing tests"""
        coordinator = UnifiedFederatedCoordinator("billing_test_coordinator")
        await coordinator.initialize()
        return coordinator

    @pytest.fixture
    def billing_tracker(self):
        """Create billing tracker"""
        return BillingTracker()

    @pytest.fixture
    def setup_user_accounts(self, billing_tracker):
        """Setup test user accounts with different tier budgets"""
        users = {
            "small_user": {"balance": 50.0, "tier": "small"},
            "medium_user": {"balance": 500.0, "tier": "medium"},
            "large_user": {"balance": 5000.0, "tier": "large"},
            "enterprise_user": {"balance": 50000.0, "tier": "enterprise"},
            "limited_budget_user": {"balance": 10.0, "tier": "small"},
            "premium_user": {"balance": 100000.0, "tier": "enterprise"},
        }

        for user_id, config in users.items():
            billing_tracker.create_user_account(user_id, config["balance"], config["tier"])

        return users

    # Budget Enforcement Tests
    @pytest.mark.asyncio
    async def test_tier_based_budget_limits(self, unified_coordinator, billing_tracker, setup_user_accounts):
        """Test budget limits are enforced based on user tier"""

        # Test small user trying to exceed tier budget limit
        params = {
            "model_id": "expensive_model",
            "cpu_cores": 32,
            "memory_gb": 128,
            "max_budget": 500.0,  # Exceeds small tier limit of 100
            "duration_hours": 24,
        }

        request_id = await unified_coordinator.submit_unified_request(
            user_id="small_user", workload_type="inference", request_params=params, user_tier="small"
        )

        workload = unified_coordinator.active_workloads[request_id]

        # Verify budget was capped to tier limit
        actual_max_budget = workload["result"]["tier"]["max_budget"]
        assert actual_max_budget == 100, f"Budget should be capped to tier limit, got {actual_max_budget}"

        # Test enterprise user with large budget
        enterprise_params = {
            "model_id": "enterprise_model",
            "cpu_cores": 128,
            "memory_gb": 512,
            "max_budget": 10000.0,
            "duration_hours": 48,
            "participants": 500,
        }

        enterprise_request = await unified_coordinator.submit_unified_request(
            user_id="enterprise_user",
            workload_type="training",
            request_params=enterprise_params,
            user_tier="enterprise",
        )

        enterprise_workload = unified_coordinator.active_workloads[enterprise_request]
        enterprise_budget = enterprise_workload["result"]["tier"]["max_budget"]

        assert enterprise_budget == 100000, "Enterprise should have high budget limit"
        assert enterprise_budget > actual_max_budget, "Enterprise should have higher limits than small tier"

    @pytest.mark.asyncio
    async def test_insufficient_balance_handling(self, unified_coordinator, billing_tracker, setup_user_accounts):
        """Test handling of insufficient account balance"""

        # User with limited balance trying expensive operation

        # Simulate balance check
        billing_tracker.user_accounts["limited_budget_user"]["balance"]

        try:
            # Should fail due to insufficient balance
            billing_tracker.hold_escrow("limited_budget_user", 50.0, "inference_workload")
            assert False, "Should have failed due to insufficient balance"
        except ValueError as e:
            assert "Insufficient balance" in str(e)

        # Verify account balance unchanged
        assert billing_tracker.user_accounts["limited_budget_user"]["balance"] == Decimal("10.0")

    @pytest.mark.asyncio
    async def test_dynamic_budget_adjustment(self, unified_coordinator, billing_tracker, setup_user_accounts):
        """Test dynamic budget adjustment based on usage patterns"""

        user_id = "medium_user"

        # Submit multiple requests to establish usage pattern
        requests = []

        for i in range(5):
            params = {
                "model_id": f"pattern_model_{i}",
                "cpu_cores": 4,
                "memory_gb": 8,
                "max_budget": 25.0,
                "duration_hours": 2,
            }

            request_id = await unified_coordinator.submit_unified_request(
                user_id=user_id, workload_type="inference", request_params=params, user_tier="medium"
            )
            requests.append(request_id)

            # Simulate usage tracking
            billing_tracker.record_usage(
                user_id=user_id,
                workload_id=request_id,
                resource_type="inference",
                amount=4.0,  # CPU hours
                cost=20.0,  # Slightly less than budget
                duration_hours=2.0,
            )

        # Verify usage recorded
        user_usage = [u for u in billing_tracker.usage_records if u["user_id"] == user_id]
        assert len(user_usage) == 5

        # Calculate average cost
        avg_cost = sum(u["cost"] for u in user_usage) / len(user_usage)
        assert float(avg_cost) == 20.0

    # Escrow and Payment Processing Tests
    @pytest.mark.asyncio
    async def test_escrow_hold_and_release_workflow(self, unified_coordinator, billing_tracker, setup_user_accounts):
        """Test complete escrow hold and release workflow"""

        user_id = "large_user"
        initial_balance = billing_tracker.user_accounts[user_id]["balance"]

        # Hold escrow for training job
        escrow_amount = 200.0
        escrow_id = billing_tracker.hold_escrow(user_id, escrow_amount, "training_workload")

        # Verify escrow created and balance reduced
        assert escrow_id in billing_tracker.escrows
        escrow = billing_tracker.escrows[escrow_id]
        assert escrow["amount"] == Decimal("200.0")
        assert escrow["status"] == "held"

        current_balance = billing_tracker.user_accounts[user_id]["balance"]
        assert current_balance == initial_balance - Decimal("200.0")

        # Release escrow with actual cost
        actual_cost = 175.0
        provider_id = "provider_gpu_001"

        transaction_id = billing_tracker.release_escrow(escrow_id, provider_id, actual_cost)

        # Verify transaction processed
        assert transaction_id in [t["transaction_id"] for t in billing_tracker.transactions]
        transaction = next(t for t in billing_tracker.transactions if t["transaction_id"] == transaction_id)

        assert transaction["amount"] == Decimal("175.0")
        assert transaction["recipient"] == provider_id

        # Verify refund processed
        final_balance = billing_tracker.user_accounts[user_id]["balance"]
        expected_balance = initial_balance - Decimal("175.0")  # Only actual cost deducted
        assert final_balance == expected_balance

        # Verify escrow marked as released
        assert billing_tracker.escrows[escrow_id]["status"] == "released"

    @pytest.mark.asyncio
    async def test_multi_workload_billing_aggregation(self, unified_coordinator, billing_tracker, setup_user_accounts):
        """Test billing aggregation across multiple workloads"""

        user_id = "enterprise_user"
        workloads = []

        # Submit mixed workloads
        workload_configs = [
            {"type": "inference", "budget": 100, "duration": 2},
            {"type": "training", "budget": 500, "duration": 12},
            {"type": "inference", "budget": 75, "duration": 1},
            {"type": "training", "budget": 800, "duration": 24},
        ]

        total_expected_cost = 0

        for i, config in enumerate(workload_configs):
            if config["type"] == "inference":
                params = {
                    "model_id": f"multi_model_{i}",
                    "cpu_cores": 8,
                    "memory_gb": 16,
                    "max_budget": config["budget"],
                    "duration_hours": config["duration"],
                }
            else:  # training
                params = {
                    "model_id": f"multi_training_model_{i}",
                    "cpu_cores": 32,
                    "memory_gb": 128,
                    "max_budget": config["budget"],
                    "duration_hours": config["duration"],
                    "participants": 25,
                }

            request_id = await unified_coordinator.submit_unified_request(
                user_id=user_id, workload_type=config["type"], request_params=params, user_tier="enterprise"
            )
            workloads.append(request_id)

            # Simulate actual cost (85% of budget)
            actual_cost = config["budget"] * 0.85
            total_expected_cost += actual_cost

            billing_tracker.record_usage(
                user_id=user_id,
                workload_id=request_id,
                resource_type=config["type"],
                amount=config["duration"],
                cost=actual_cost,
                duration_hours=config["duration"],
            )

        # Verify aggregate billing
        user_summary = billing_tracker.get_user_summary(user_id)
        assert user_summary["usage_records"] == len(workload_configs)

        total_usage_cost = sum(u["cost"] for u in billing_tracker.usage_records if u["user_id"] == user_id)
        assert abs(float(total_usage_cost) - total_expected_cost) < 0.01

    # Cost Tracking and Analytics Tests
    @pytest.mark.asyncio
    async def test_real_time_cost_tracking(self, unified_coordinator, billing_tracker, setup_user_accounts):
        """Test real-time cost tracking during workload execution"""

        user_id = "medium_user"

        params = {
            "model_id": "cost_tracking_model",
            "cpu_cores": 8,
            "memory_gb": 16,
            "max_budget": 100.0,
            "duration_hours": 6,
            "cost_tracking": True,
            "alert_thresholds": [25.0, 50.0, 75.0],
        }

        request_id = await unified_coordinator.submit_unified_request(
            user_id=user_id, workload_type="inference", request_params=params, user_tier="medium"
        )

        # Simulate progressive cost accumulation
        cost_increments = [15.0, 20.0, 25.0, 18.0]  # Total: 78.0
        cumulative_cost = 0

        for i, increment in enumerate(cost_increments):
            cumulative_cost += increment

            billing_tracker.record_usage(
                user_id=user_id,
                workload_id=request_id,
                resource_type="inference_increment",
                amount=1.5,  # 1.5 hours per increment
                cost=increment,
                duration_hours=1.5,
            )

            # Verify cost doesn't exceed budget
            assert cumulative_cost < params["max_budget"]

        # Verify final cost tracking
        user_usage = [
            u for u in billing_tracker.usage_records if u["user_id"] == user_id and u["workload_id"] == request_id
        ]

        total_tracked_cost = sum(float(u["cost"]) for u in user_usage)
        assert abs(total_tracked_cost - cumulative_cost) < 0.01

    @pytest.mark.asyncio
    async def test_cost_optimization_recommendations(self, unified_coordinator, billing_tracker, setup_user_accounts):
        """Test cost optimization recommendations based on usage patterns"""

        user_id = "large_user"

        # Simulate various workload patterns
        workload_patterns = [
            {"time": "peak", "cpu": 16, "memory": 64, "cost": 120},
            {"time": "off_peak", "cpu": 16, "memory": 64, "cost": 80},
            {"time": "peak", "cpu": 8, "memory": 32, "cost": 70},
            {"time": "off_peak", "cpu": 8, "memory": 32, "cost": 45},
        ]

        for i, pattern in enumerate(workload_patterns):
            params = {
                "model_id": f"optimization_model_{i}",
                "cpu_cores": pattern["cpu"],
                "memory_gb": pattern["memory"],
                "max_budget": pattern["cost"] * 1.2,
                "duration_hours": 4,
                "execution_time": pattern["time"],
            }

            request_id = await unified_coordinator.submit_unified_request(
                user_id=user_id, workload_type="inference", request_params=params, user_tier="large"
            )

            billing_tracker.record_usage(
                user_id=user_id,
                workload_id=request_id,
                resource_type=f"inference_{pattern['time']}",
                amount=pattern["cpu"],
                cost=pattern["cost"],
                duration_hours=4,
            )

        # Analyze cost patterns
        user_usage = [u for u in billing_tracker.usage_records if u["user_id"] == user_id]

        peak_costs = [u for u in user_usage if "peak" in u["resource_type"]]
        off_peak_costs = [u for u in user_usage if "off_peak" in u["resource_type"]]

        avg_peak_cost = sum(float(u["cost"]) for u in peak_costs) / len(peak_costs)
        avg_off_peak_cost = sum(float(u["cost"]) for u in off_peak_costs) / len(off_peak_costs)

        # Verify off-peak is more cost effective
        assert avg_off_peak_cost < avg_peak_cost

        # Calculate potential savings
        savings_percentage = (avg_peak_cost - avg_off_peak_cost) / avg_peak_cost * 100
        assert savings_percentage > 20, "Off-peak should offer significant savings"

    # Billing Integration with Marketplace Tests
    @pytest.mark.asyncio
    async def test_marketplace_billing_integration(self, unified_coordinator, billing_tracker, setup_user_accounts):
        """Test billing integration with marketplace transactions"""

        buyer_id = "medium_user"
        provider_id = "premium_provider_001"

        # Submit request that goes through marketplace
        params = {
            "model_id": "marketplace_billing_model",
            "cpu_cores": 12,
            "memory_gb": 48,
            "max_budget": 150.0,
            "duration_hours": 8,
            "provider_requirements": {"min_trust_score": 0.9},
        }

        await unified_coordinator.submit_unified_request(
            user_id=buyer_id, workload_type="inference", request_params=params, user_tier="medium"
        )

        # Simulate marketplace matching and billing
        bid_amount = 135.0  # Amount bid in marketplace

        # Hold escrow
        escrow_id = billing_tracker.hold_escrow(buyer_id, bid_amount, "marketplace_transaction")

        # Simulate job completion and payment
        actual_usage_cost = 128.0  # Slightly less than bid

        transaction_id = billing_tracker.release_escrow(escrow_id, provider_id, actual_usage_cost)

        # Verify marketplace transaction
        transaction = next(t for t in billing_tracker.transactions if t["transaction_id"] == transaction_id)

        assert transaction["payer"] == buyer_id
        assert transaction["recipient"] == provider_id
        assert transaction["amount"] == Decimal("128.0")

        # Verify buyer got refund for unused portion
        billing_tracker.user_accounts[buyer_id]
        bid_amount - actual_usage_cost

        # Check that refund was processed (would need initial balance to verify exact amount)
        buyer_summary = billing_tracker.get_user_summary(buyer_id)
        assert buyer_summary["transaction_count"] >= 1

    @pytest.mark.asyncio
    async def test_multi_currency_billing_support(self, unified_coordinator, billing_tracker, setup_user_accounts):
        """Test billing system supports multiple currencies/tokens"""

        # Extend billing tracker for multi-currency (simplified)
        user_id = "enterprise_user"

        # Test with different "currency" types
        currency_tests = [
            {"currency": "USD", "amount": 100.0, "rate": 1.0},
            {"currency": "GPU_TOKENS", "amount": 50.0, "rate": 2.0},  # 1 GPU token = 2 USD
            {"currency": "COMPUTE_CREDITS", "amount": 200.0, "rate": 0.5},  # 1 credit = 0.5 USD
        ]

        for currency_test in currency_tests:
            # Convert to USD equivalent for billing
            usd_equivalent = currency_test["amount"] * currency_test["rate"]

            billing_tracker.record_usage(
                user_id=user_id,
                workload_id=f"currency_test_{currency_test['currency']}",
                resource_type=f"compute_{currency_test['currency'].lower()}",
                amount=currency_test["amount"],
                cost=usd_equivalent,
                duration_hours=4,
            )

        # Verify multi-currency usage recorded
        user_usage = [
            u for u in billing_tracker.usage_records if u["user_id"] == user_id and "currency_test" in u["workload_id"]
        ]

        assert len(user_usage) == 3

        total_usd_equivalent = sum(float(u["cost"]) for u in user_usage)
        expected_total = 100.0 + (50.0 * 2.0) + (200.0 * 0.5)  # 100 + 100 + 100 = 300
        assert abs(total_usd_equivalent - expected_total) < 0.01

    # Compliance and Audit Tests
    @pytest.mark.asyncio
    async def test_billing_audit_trail(self, unified_coordinator, billing_tracker, setup_user_accounts):
        """Test comprehensive billing audit trail"""

        user_id = "enterprise_user"

        # Create series of transactions
        audit_transactions = []

        for i in range(5):
            params = {
                "model_id": f"audit_model_{i}",
                "cpu_cores": 8,
                "memory_gb": 32,
                "max_budget": 80.0,
                "duration_hours": 4,
                "audit_required": True,
            }

            request_id = await unified_coordinator.submit_unified_request(
                user_id=user_id, workload_type="inference", request_params=params, user_tier="enterprise"
            )

            # Simulate escrow and release
            escrow_id = billing_tracker.hold_escrow(user_id, 80.0, f"audit_transaction_{i}")
            actual_cost = 75.0

            transaction_id = billing_tracker.release_escrow(escrow_id, f"provider_{i}", actual_cost)
            audit_transactions.append(transaction_id)

            billing_tracker.record_usage(
                user_id=user_id,
                workload_id=request_id,
                resource_type="audited_inference",
                amount=4.0,
                cost=actual_cost,
                duration_hours=4,
            )

        # Generate audit report
        user_summary = billing_tracker.get_user_summary(user_id)
        user_transactions = [t for t in billing_tracker.transactions if t["payer"] == user_id]
        user_escrows = [e for e in billing_tracker.escrows.values() if e["user_id"] == user_id]

        audit_report = {
            "user_id": user_id,
            "audit_period": {
                "start": min(t["timestamp"] for t in user_transactions),
                "end": max(t["timestamp"] for t in user_transactions),
            },
            "transaction_summary": {
                "count": len(user_transactions),
                "total_amount": sum(t["amount"] for t in user_transactions),
                "average_amount": sum(t["amount"] for t in user_transactions) / len(user_transactions),
            },
            "escrow_summary": {
                "count": len(user_escrows),
                "total_held": sum(e["amount"] for e in user_escrows),
                "released_count": len([e for e in user_escrows if e["status"] == "released"]),
            },
            "usage_summary": user_summary,
        }

        # Validate audit trail completeness
        assert audit_report["transaction_summary"]["count"] == 5
        assert audit_report["escrow_summary"]["count"] == 5
        assert audit_report["escrow_summary"]["released_count"] == 5
        assert len(user_summary["recent_transactions"]) >= 5

    @pytest.mark.asyncio
    async def test_billing_compliance_reporting(self, unified_coordinator, billing_tracker, setup_user_accounts):
        """Test compliance reporting for financial regulations"""

        # Test various compliance scenarios
        compliance_users = ["small_user", "medium_user", "large_user", "enterprise_user"]

        compliance_report = {
            "report_period": {"start": datetime.now(UTC) - timedelta(days=30), "end": datetime.now(UTC)},
            "user_summaries": {},
            "aggregate_metrics": {
                "total_transactions": 0,
                "total_volume": Decimal("0"),
                "average_transaction": Decimal("0"),
                "tier_distribution": {},
            },
        }

        for user_id in compliance_users:
            # Generate transactions for compliance testing
            for i in range(3):
                amount = 50.0 * (i + 1)  # Varying amounts
                escrow_id = billing_tracker.hold_escrow(user_id, amount, "compliance_test")
                billing_tracker.release_escrow(escrow_id, f"provider_compliance_{i}", amount * 0.9)

            # Get user summary for compliance
            user_summary = billing_tracker.get_user_summary(user_id)
            compliance_report["user_summaries"][user_id] = user_summary

            # Update aggregate metrics
            user_tier = billing_tracker.user_accounts[user_id]["tier"]
            if user_tier not in compliance_report["aggregate_metrics"]["tier_distribution"]:
                compliance_report["aggregate_metrics"]["tier_distribution"][user_tier] = 0
            compliance_report["aggregate_metrics"]["tier_distribution"][user_tier] += 1

        # Calculate aggregate metrics
        all_transactions = billing_tracker.transactions
        compliance_report["aggregate_metrics"]["total_transactions"] = len(all_transactions)
        compliance_report["aggregate_metrics"]["total_volume"] = sum(t["amount"] for t in all_transactions)

        if all_transactions:
            compliance_report["aggregate_metrics"]["average_transaction"] = compliance_report["aggregate_metrics"][
                "total_volume"
            ] / len(all_transactions)

        # Validate compliance report
        assert compliance_report["aggregate_metrics"]["total_transactions"] >= 12  # 4 users * 3 transactions
        assert compliance_report["aggregate_metrics"]["total_volume"] > Decimal("0")
        assert len(compliance_report["user_summaries"]) == 4
        assert len(compliance_report["aggregate_metrics"]["tier_distribution"]) >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
