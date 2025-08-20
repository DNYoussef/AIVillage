#!/usr/bin/env python3
"""
Test marketplace pricing functionality directly
"""

import asyncio
import sys

sys.path.insert(0, 'packages')

async def test_pricing():
    from packages.fog.gateway.scheduler.marketplace import BidType, PricingTier, get_marketplace_engine

    print('Testing marketplace pricing...')
    marketplace = await get_marketplace_engine()

    # Add listings with different pricing tiers
    basic_listing = await marketplace.add_resource_listing(
        node_id='basic-server',
        cpu_cores=4.0,
        memory_gb=8.0,
        disk_gb=100.0,
        spot_price=0.08,
        on_demand_price=0.12,
        trust_score=0.5,
        pricing_tier=PricingTier.BASIC
    )
    print(f'[PASS] Added basic listing: {basic_listing}')

    standard_listing = await marketplace.add_resource_listing(
        node_id='standard-server',
        cpu_cores=4.0,
        memory_gb=8.0,
        disk_gb=100.0,
        spot_price=0.12,
        on_demand_price=0.18,
        trust_score=0.7,
        pricing_tier=PricingTier.STANDARD
    )
    print(f'[PASS] Added standard listing: {standard_listing}')

    premium_listing = await marketplace.add_resource_listing(
        node_id='premium-server',
        cpu_cores=4.0,
        memory_gb=8.0,
        disk_gb=100.0,
        spot_price=0.16,
        on_demand_price=0.24,
        trust_score=0.9,
        pricing_tier=PricingTier.PREMIUM
    )
    print(f'[PASS] Added premium listing: {premium_listing}')

    # Test different bid types and pricing tiers
    test_cases = [
        {'bid_type': BidType.SPOT, 'tier': PricingTier.BASIC, 'name': 'Spot Basic'},
        {'bid_type': BidType.ON_DEMAND, 'tier': PricingTier.STANDARD, 'name': 'On-Demand Standard'},
        {'bid_type': BidType.SPOT, 'tier': PricingTier.PREMIUM, 'name': 'Spot Premium'},
    ]

    for case in test_cases:
        quote = await marketplace.get_price_quote(
            cpu_cores=2.0,
            memory_gb=4.0,
            duration_hours=1.0,
            bid_type=case['bid_type'],
            pricing_tier=case['tier']
        )

        if quote['available']:
            print(f'[PASS] {case["name"]} quote: ${quote["quote"]["avg_price"]:.4f}')
            print(f'       Providers: {quote["market_conditions"]["available_providers"]}')
        else:
            print(f'[FAIL] {case["name"]} not available: {quote.get("reason", "Unknown")}')

    # Test bid submission and matching
    bid_id = await marketplace.submit_bid(
        namespace='test-org/pricing-test',
        cpu_cores=1.5,
        memory_gb=3.0,
        max_price=0.30,
        bid_type=BidType.SPOT,
        pricing_tier=PricingTier.BASIC
    )
    print(f'[PASS] Submitted test bid: {bid_id}')

    # Force bid matching
    matched_count = await marketplace._match_bids_to_listings()
    print(f'[PASS] Matched {matched_count} bids')

    if marketplace.active_trades:
        trade = list(marketplace.active_trades.values())[0]
        print(f'[PASS] Trade executed: ${trade.agreed_price:.4f} for {trade.cpu_cores} cores')
        print(f'       Seller: {trade.seller_node_id}')
        print(f'       Pricing tier: {trade.pricing_tier.value}')

    # Test market status
    status = await marketplace.get_marketplace_status()
    print(f'[PASS] Market status:')
    print(f'       Active listings: {status["marketplace_summary"]["active_listings"]}')
    print(f'       Pending bids: {status["marketplace_summary"]["pending_bids"]}')
    print(f'       Spot price: ${status["pricing"]["current_spot_price_per_cpu_hour"]:.4f}/cpu-hour')
    print(f'       On-demand price: ${status["pricing"]["current_on_demand_price_per_cpu_hour"]:.4f}/cpu-hour')

    await marketplace.stop()
    print('[SUCCESS] Pricing functionality test passed!')

if __name__ == '__main__':
    asyncio.run(test_pricing())
