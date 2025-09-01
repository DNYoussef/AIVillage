"""
Comprehensive test suite for the Auction Engine.

Tests reverse auction mechanics, second-price settlement, deposit system,
and anti-griefing mechanisms.
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))

from infrastructure.fog.market.auction_engine import AuctionEngine, AuctionStatus, BidStatus, ResourceRequirement


class TestAuctionEngine:
    """Test suite for AuctionEngine functionality."""

    @pytest.fixture
    async def auction_engine(self):
        """Create a test auction engine instance."""
        engine = AuctionEngine()
        await engine.initialize()
        return engine

    @pytest.fixture
    def sample_resource_requirement(self):
        """Create sample resource requirements for testing."""
        return ResourceRequirement(
            cpu_cores=Decimal("4"),
            memory_gb=Decimal("8"),
            storage_gb=Decimal("100"),
            bandwidth_mbps=Decimal("1000"),
            gpu_units=Decimal("0"),
            specialized_hardware=[],
            duration_hours=Decimal("24"),
            quality_requirements={
                "min_trust_score": Decimal("0.8"),
                "min_reputation": Decimal("0.7"),
                "max_latency_ms": 50,
            },
        )

    @pytest.fixture
    def sample_bids(self):
        """Create sample bids for auction testing."""
        return [
            {
                "bidder_id": "provider_1",
                "node_id": "node_1",
                "bid_price": Decimal("0.10"),
                "trust_score": Decimal("0.95"),
                "reputation_score": Decimal("0.90"),
                "available_resources": {
                    "cpu_cores": Decimal("8"),
                    "memory_gb": Decimal("16"),
                    "storage_gb": Decimal("500"),
                    "bandwidth_mbps": Decimal("2000"),
                },
            },
            {
                "bidder_id": "provider_2",
                "node_id": "node_2",
                "bid_price": Decimal("0.08"),
                "trust_score": Decimal("0.85"),
                "reputation_score": Decimal("0.80"),
                "available_resources": {
                    "cpu_cores": Decimal("4"),
                    "memory_gb": Decimal("8"),
                    "storage_gb": Decimal("200"),
                    "bandwidth_mbps": Decimal("1500"),
                },
            },
            {
                "bidder_id": "provider_3",
                "node_id": "node_3",
                "bid_price": Decimal("0.12"),
                "trust_score": Decimal("0.90"),
                "reputation_score": Decimal("0.85"),
                "available_resources": {
                    "cpu_cores": Decimal("6"),
                    "memory_gb": Decimal("12"),
                    "storage_gb": Decimal("300"),
                    "bandwidth_mbps": Decimal("1800"),
                },
            },
        ]

    @pytest.mark.asyncio
    async def test_create_auction(self, auction_engine, sample_resource_requirement):
        """Test auction creation with proper initialization."""
        requester_id = "test_requester"
        reserve_price = Decimal("0.15")
        duration_minutes = 30
        deposit_amount = Decimal("10.0")

        auction_id = await auction_engine.create_auction(
            requester_id=requester_id,
            resource_requirement=sample_resource_requirement,
            reserve_price=reserve_price,
            auction_duration_minutes=duration_minutes,
            deposit_amount=deposit_amount,
        )

        assert auction_id is not None
        assert auction_id in auction_engine.auctions

        auction = auction_engine.auctions[auction_id]
        assert auction.requester_id == requester_id
        assert auction.resource_requirement == sample_resource_requirement
        assert auction.reserve_price == reserve_price
        assert auction.status == AuctionStatus.ACTIVE
        assert auction.deposit_amount == deposit_amount

    @pytest.mark.asyncio
    async def test_submit_valid_bid(self, auction_engine, sample_resource_requirement, sample_bids):
        """Test submitting a valid bid to an active auction."""
        # Create auction
        auction_id = await auction_engine.create_auction(
            requester_id="test_requester",
            resource_requirement=sample_resource_requirement,
            reserve_price=Decimal("0.15"),
            auction_duration_minutes=30,
            deposit_amount=Decimal("10.0"),
        )

        # Submit bid
        bid_data = sample_bids[0]
        bid_id = await auction_engine.submit_bid(auction_id=auction_id, **bid_data)

        assert bid_id is not None
        auction = auction_engine.auctions[auction_id]
        assert len(auction.bids) == 1

        bid = auction.bids[0]
        assert bid.bidder_id == bid_data["bidder_id"]
        assert bid.bid_price == bid_data["bid_price"]
        assert bid.status == BidStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_bid_validation_insufficient_resources(self, auction_engine, sample_resource_requirement):
        """Test bid rejection when resources are insufficient."""
        # Create auction
        auction_id = await auction_engine.create_auction(
            requester_id="test_requester",
            resource_requirement=sample_resource_requirement,
            reserve_price=Decimal("0.15"),
            auction_duration_minutes=30,
            deposit_amount=Decimal("10.0"),
        )

        # Submit bid with insufficient resources
        bid_id = await auction_engine.submit_bid(
            auction_id=auction_id,
            bidder_id="insufficient_provider",
            node_id="insufficient_node",
            bid_price=Decimal("0.10"),
            trust_score=Decimal("0.95"),
            reputation_score=Decimal("0.90"),
            available_resources={
                "cpu_cores": Decimal("2"),  # Insufficient (requires 4)
                "memory_gb": Decimal("4"),  # Insufficient (requires 8)
                "storage_gb": Decimal("50"),  # Insufficient (requires 100)
                "bandwidth_mbps": Decimal("500"),  # Insufficient (requires 1000)
            },
        )

        assert bid_id is None
        auction = auction_engine.auctions[auction_id]
        assert len(auction.bids) == 0

    @pytest.mark.asyncio
    async def test_bid_validation_low_quality_scores(self, auction_engine, sample_resource_requirement):
        """Test bid rejection when quality scores are too low."""
        # Create auction
        auction_id = await auction_engine.create_auction(
            requester_id="test_requester",
            resource_requirement=sample_resource_requirement,
            reserve_price=Decimal("0.15"),
            auction_duration_minutes=30,
            deposit_amount=Decimal("10.0"),
        )

        # Submit bid with low quality scores
        bid_id = await auction_engine.submit_bid(
            auction_id=auction_id,
            bidder_id="low_quality_provider",
            node_id="low_quality_node",
            bid_price=Decimal("0.10"),
            trust_score=Decimal("0.60"),  # Below minimum (0.8)
            reputation_score=Decimal("0.50"),  # Below minimum (0.7)
            available_resources={
                "cpu_cores": Decimal("8"),
                "memory_gb": Decimal("16"),
                "storage_gb": Decimal("500"),
                "bandwidth_mbps": Decimal("2000"),
            },
        )

        assert bid_id is None
        auction = auction_engine.auctions[auction_id]
        assert len(auction.bids) == 0

    @pytest.mark.asyncio
    async def test_second_price_settlement(self, auction_engine, sample_resource_requirement, sample_bids):
        """Test Vickrey (second-price) auction settlement mechanism."""
        # Create auction
        auction_id = await auction_engine.create_auction(
            requester_id="test_requester",
            resource_requirement=sample_resource_requirement,
            reserve_price=Decimal("0.15"),
            auction_duration_minutes=30,
            deposit_amount=Decimal("10.0"),
        )

        # Submit multiple bids
        bid_ids = []
        for bid_data in sample_bids:
            bid_id = await auction_engine.submit_bid(auction_id=auction_id, **bid_data)
            if bid_id:
                bid_ids.append(bid_id)

        assert len(bid_ids) == 3

        # Force auction completion for testing
        auction = auction_engine.auctions[auction_id]
        auction.status = AuctionStatus.ENDED

        # Process settlement
        result = await auction_engine.settle_auction(auction_id)

        assert result is not None
        assert "winning_bid" in result
        assert "settlement_price" in result

        # In second-price auction, winner pays second-lowest price
        # Bids: provider_2 (0.08), provider_1 (0.10), provider_3 (0.12)
        # Winner should be provider_2, paying provider_1's price (0.10)
        winning_bid = result["winning_bid"]
        settlement_price = result["settlement_price"]

        assert winning_bid.bidder_id == "provider_2"  # Lowest bidder wins
        assert settlement_price == Decimal("0.10")  # Pays second-lowest price

    @pytest.mark.asyncio
    async def test_deposit_handling(self, auction_engine, sample_resource_requirement):
        """Test deposit system for anti-griefing protection."""
        deposit_amount = Decimal("10.0")

        # Create auction with deposit requirement
        auction_id = await auction_engine.create_auction(
            requester_id="test_requester",
            resource_requirement=sample_resource_requirement,
            reserve_price=Decimal("0.15"),
            auction_duration_minutes=30,
            deposit_amount=deposit_amount,
        )

        auction = auction_engine.auctions[auction_id]
        assert auction.deposit_amount == deposit_amount

        # Mock tokenomics integration for deposit validation
        with patch.object(auction_engine, "tokenomics_integration") as mock_tokenomics:
            mock_tokenomics.validate_deposit = AsyncMock(return_value=True)
            mock_tokenomics.hold_deposit = AsyncMock(return_value=True)

            # Submit bid requiring deposit validation
            bid_id = await auction_engine.submit_bid(
                auction_id=auction_id,
                bidder_id="provider_with_deposit",
                node_id="node_with_deposit",
                bid_price=Decimal("0.10"),
                trust_score=Decimal("0.95"),
                reputation_score=Decimal("0.90"),
                available_resources={
                    "cpu_cores": Decimal("8"),
                    "memory_gb": Decimal("16"),
                    "storage_gb": Decimal("500"),
                    "bandwidth_mbps": Decimal("2000"),
                },
                deposit_tx_hash="mock_deposit_hash",
            )

            assert bid_id is not None
            mock_tokenomics.validate_deposit.assert_called_once()
            mock_tokenomics.hold_deposit.assert_called_once()

    @pytest.mark.asyncio
    async def test_auction_expiration(self, auction_engine, sample_resource_requirement):
        """Test automatic auction expiration handling."""
        # Create auction with very short duration
        auction_id = await auction_engine.create_auction(
            requester_id="test_requester",
            resource_requirement=sample_resource_requirement,
            reserve_price=Decimal("0.15"),
            auction_duration_minutes=1,  # 1 minute
            deposit_amount=Decimal("10.0"),
        )

        auction = auction_engine.auctions[auction_id]
        assert auction.status == AuctionStatus.ACTIVE

        # Manually set expiration time to past for testing
        auction.end_time = datetime.utcnow() - timedelta(minutes=1)

        # Check if auction is marked as expired
        is_expired = await auction_engine.check_auction_expiry(auction_id)
        assert is_expired is True

        # Verify auction status is updated
        assert auction.status == AuctionStatus.EXPIRED

    @pytest.mark.asyncio
    async def test_quality_weighted_scoring(self, auction_engine, sample_resource_requirement):
        """Test quality-weighted bid scoring system."""
        # Create auction
        auction_id = await auction_engine.create_auction(
            requester_id="test_requester",
            resource_requirement=sample_resource_requirement,
            reserve_price=Decimal("0.15"),
            auction_duration_minutes=30,
            deposit_amount=Decimal("10.0"),
        )

        # Submit bids with different quality scores
        high_price_high_quality = await auction_engine.submit_bid(
            auction_id=auction_id,
            bidder_id="high_quality_provider",
            node_id="high_quality_node",
            bid_price=Decimal("0.12"),
            trust_score=Decimal("0.98"),
            reputation_score=Decimal("0.95"),
            available_resources={
                "cpu_cores": Decimal("8"),
                "memory_gb": Decimal("16"),
                "storage_gb": Decimal("500"),
                "bandwidth_mbps": Decimal("2000"),
            },
        )

        low_price_low_quality = await auction_engine.submit_bid(
            auction_id=auction_id,
            bidder_id="low_quality_provider",
            node_id="low_quality_node",
            bid_price=Decimal("0.09"),
            trust_score=Decimal("0.82"),
            reputation_score=Decimal("0.75"),
            available_resources={
                "cpu_cores": Decimal("4"),
                "memory_gb": Decimal("8"),
                "storage_gb": Decimal("200"),
                "bandwidth_mbps": Decimal("1200"),
            },
        )

        assert high_price_high_quality is not None
        assert low_price_low_quality is not None

        auction = auction_engine.auctions[auction_id]

        # Calculate quality scores for comparison
        high_quality_bid = next(b for b in auction.bids if b.bidder_id == "high_quality_provider")
        low_quality_bid = next(b for b in auction.bids if b.bidder_id == "low_quality_provider")

        high_quality_score = await auction_engine.calculate_quality_score(high_quality_bid, sample_resource_requirement)
        low_quality_score = await auction_engine.calculate_quality_score(low_quality_bid, sample_resource_requirement)

        # Higher quality should result in better overall score
        assert high_quality_score > low_quality_score

    @pytest.mark.asyncio
    async def test_concurrent_bid_submission(self, auction_engine, sample_resource_requirement, sample_bids):
        """Test handling of concurrent bid submissions."""
        # Create auction
        auction_id = await auction_engine.create_auction(
            requester_id="test_requester",
            resource_requirement=sample_resource_requirement,
            reserve_price=Decimal("0.15"),
            auction_duration_minutes=30,
            deposit_amount=Decimal("10.0"),
        )

        # Submit multiple bids concurrently
        tasks = []
        for i, bid_data in enumerate(sample_bids):
            bid_data_copy = bid_data.copy()
            bid_data_copy["bidder_id"] = f"concurrent_provider_{i}"
            bid_data_copy["node_id"] = f"concurrent_node_{i}"

            task = asyncio.create_task(auction_engine.submit_bid(auction_id=auction_id, **bid_data_copy))
            tasks.append(task)

        # Wait for all bids to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify all bids were processed successfully
        successful_bids = [r for r in results if isinstance(r, str)]
        assert len(successful_bids) == len(sample_bids)

        auction = auction_engine.auctions[auction_id]
        assert len(auction.bids) == len(sample_bids)

    @pytest.mark.asyncio
    async def test_anti_griefing_protection(self, auction_engine, sample_resource_requirement):
        """Test anti-griefing mechanisms."""
        # Create auction
        auction_id = await auction_engine.create_auction(
            requester_id="test_requester",
            resource_requirement=sample_resource_requirement,
            reserve_price=Decimal("0.15"),
            auction_duration_minutes=30,
            deposit_amount=Decimal("10.0"),
        )

        # Mock anti-griefing system
        with patch.object(auction_engine, "anti_griefing") as mock_anti_griefing:
            # Test case: Valid bidder passes validation
            mock_anti_griefing.validate_bidder = AsyncMock(
                return_value={"is_valid": True, "confidence_score": Decimal("0.95")}
            )

            valid_bid_id = await auction_engine.submit_bid(
                auction_id=auction_id,
                bidder_id="valid_provider",
                node_id="valid_node",
                bid_price=Decimal("0.10"),
                trust_score=Decimal("0.95"),
                reputation_score=Decimal("0.90"),
                available_resources={
                    "cpu_cores": Decimal("8"),
                    "memory_gb": Decimal("16"),
                    "storage_gb": Decimal("500"),
                    "bandwidth_mbps": Decimal("2000"),
                },
            )

            assert valid_bid_id is not None
            mock_anti_griefing.validate_bidder.assert_called()

            # Test case: Invalid bidder fails validation
            mock_anti_griefing.validate_bidder.return_value = {
                "is_valid": False,
                "confidence_score": Decimal("0.30"),
                "flags": ["suspicious_pattern", "low_reputation_history"],
            }

            invalid_bid_id = await auction_engine.submit_bid(
                auction_id=auction_id,
                bidder_id="suspicious_provider",
                node_id="suspicious_node",
                bid_price=Decimal("0.05"),  # Suspiciously low
                trust_score=Decimal("0.95"),
                reputation_score=Decimal("0.90"),
                available_resources={
                    "cpu_cores": Decimal("8"),
                    "memory_gb": Decimal("16"),
                    "storage_gb": Decimal("500"),
                    "bandwidth_mbps": Decimal("2000"),
                },
            )

            assert invalid_bid_id is None

    @pytest.mark.asyncio
    async def test_auction_statistics(self, auction_engine, sample_resource_requirement, sample_bids):
        """Test auction statistics collection and reporting."""
        # Create auction
        auction_id = await auction_engine.create_auction(
            requester_id="test_requester",
            resource_requirement=sample_resource_requirement,
            reserve_price=Decimal("0.15"),
            auction_duration_minutes=30,
            deposit_amount=Decimal("10.0"),
        )

        # Submit multiple bids
        for bid_data in sample_bids:
            await auction_engine.submit_bid(auction_id=auction_id, **bid_data)

        # Get auction statistics
        stats = await auction_engine.get_auction_statistics(auction_id)

        assert stats is not None
        assert "total_bids" in stats
        assert "unique_bidders" in stats
        assert "average_bid_price" in stats
        assert "lowest_bid_price" in stats
        assert "highest_bid_price" in stats

        assert stats["total_bids"] == len(sample_bids)
        assert stats["unique_bidders"] == len(sample_bids)
        assert stats["lowest_bid_price"] == Decimal("0.08")
        assert stats["highest_bid_price"] == Decimal("0.12")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
