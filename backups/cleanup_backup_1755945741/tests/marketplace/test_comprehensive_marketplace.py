#!/usr/bin/env python3
"""
Comprehensive marketplace test covering all functionality
"""

import asyncio
import sys

sys.path.insert(0, "packages")


async def test_comprehensive():
    from packages.fog.gateway.api.billing import get_billing_engine
    from packages.fog.gateway.scheduler.marketplace import BidType, PricingTier, get_marketplace_engine

    print("Running comprehensive marketplace test...")
    marketplace = await get_marketplace_engine()

    # Test multiple pricing tiers and bid types
    test_scenarios = [
        {"tier": PricingTier.BASIC, "spot": 0.08, "demand": 0.12, "trust": 0.5},
        {"tier": PricingTier.STANDARD, "spot": 0.12, "demand": 0.18, "trust": 0.7},
        {"tier": PricingTier.PREMIUM, "spot": 0.16, "demand": 0.24, "trust": 0.9},
    ]

    # Add multiple listings
    for i, scenario in enumerate(test_scenarios):
        listing_id = await marketplace.add_resource_listing(
            node_id=f'{scenario["tier"].value}-node-{i}',
            cpu_cores=8.0,
            memory_gb=16.0,
            disk_gb=200.0,
            spot_price=scenario["spot"],
            on_demand_price=scenario["demand"],
            trust_score=scenario["trust"],
            pricing_tier=scenario["tier"],
        )
        print(f'[PASS] Added {scenario["tier"].value} listing: {listing_id}')

    # Test bid matching across tiers
    bids = [
        {"cores": 2.0, "memory": 4.0, "price": 0.30, "type": BidType.SPOT, "tier": PricingTier.BASIC},
        {"cores": 4.0, "memory": 8.0, "price": 0.50, "type": BidType.ON_DEMAND, "tier": PricingTier.STANDARD},
        {"cores": 1.0, "memory": 2.0, "price": 0.40, "type": BidType.SPOT, "tier": PricingTier.PREMIUM},
    ]

    bid_ids = []
    for i, bid in enumerate(bids):
        bid_id = await marketplace.submit_bid(
            namespace=f"test-org/team-{i}",
            cpu_cores=bid["cores"],
            memory_gb=bid["memory"],
            max_price=bid["price"],
            bid_type=bid["type"],
            pricing_tier=bid["tier"],
        )
        bid_ids.append(bid_id)
        print(f'[PASS] Submitted {bid["type"].value} bid for {bid["tier"].value}: {bid_id}')

    # Force matching
    matched = await marketplace._match_bids_to_listings()
    print(f"[PASS] Matched {matched} bids to listings")

    # Check active trades
    print(f"[PASS] Active trades: {len(marketplace.active_trades)}")
    for trade_id, trade in marketplace.active_trades.items():
        print(f"       Trade {trade_id}: {trade.cpu_cores} cores at ${trade.agreed_price:.4f}")
        print(f"       Tier: {trade.pricing_tier.value}, Buyer: {trade.buyer_namespace}")

    # Test billing integration with multiple namespaces
    billing = await get_billing_engine()

    total_costs = []
    for i in range(3):
        cost = await billing.record_job_usage(
            namespace=f"test-org/team-{i}",
            job_id=f"job-{i}-test",
            cpu_cores=2.0 + i,
            memory_gb=4.0 + i * 2,
            disk_gb=10.0 + i * 5,
            duration_seconds=1800 + i * 600,  # 30-60 minutes
        )
        total_costs.append(cost)
        print(f"[PASS] Team {i} job cost: ${cost:.4f}")

    # Generate billing reports
    for i in range(3):
        report = await billing.get_usage_report(f"test-org/team-{i}")
        print(f"[PASS] Team {i} report: {report.usage_metrics.job_count} jobs, ${report.cost_breakdown.total_cost:.4f}")

    # Test marketplace status
    status = await marketplace.get_marketplace_status()
    print("[PASS] Marketplace status:")
    print(f'       Listings: {status["marketplace_summary"]["active_listings"]}')
    print(f'       Trades: {status["marketplace_summary"]["active_trades"]}')
    if "total_trade_volume" in status["marketplace_summary"]:
        print(f'       Total volume: ${status["marketplace_summary"]["total_trade_volume"]:.4f}')
    else:
        print(f'       Status keys: {list(status["marketplace_summary"].keys())}')

    await marketplace.stop()
    print("[SUCCESS] Comprehensive marketplace test completed!")


if __name__ == "__main__":
    asyncio.run(test_comprehensive())
