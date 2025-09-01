"""
Market-Based Pricing Test Suite

Comprehensive test coverage for fog computing market-based pricing system including:
- Auction engine with reverse auction mechanics
- Dynamic pricing manager with lane-specific pricing
- Market orchestration and integration testing
- Tokenomics and anti-griefing integration
- End-to-end workflow validation

Test Structure:
- test_auction_engine.py: Auction mechanics, bidding, settlement
- test_pricing_manager.py: Dynamic pricing, circuit breakers, market analysis
- test_market_integration.py: End-to-end integration and system workflows

Usage:
    pytest tests/fog/market/ -v                    # Run all market tests
    pytest tests/fog/market/test_auction_engine.py # Run auction tests only
    pytest tests/fog/market/ -k "integration"      # Run integration tests only
    pytest tests/fog/market/ --cov=infrastructure.fog.market # With coverage
"""

from .test_auction_engine import TestAuctionEngine
from .test_market_integration import TestMarketIntegration
from .test_pricing_manager import TestDynamicPricingManager

__all__ = ["TestAuctionEngine", "TestDynamicPricingManager", "TestMarketIntegration"]
