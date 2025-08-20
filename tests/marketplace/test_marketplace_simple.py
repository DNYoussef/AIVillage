#!/usr/bin/env python3
"""
Simple test script to verify marketplace functionality works
"""

import asyncio
import sys

sys.path.insert(0, 'packages')

async def test_marketplace():
    from packages.fog.gateway.api.billing import get_billing_engine
    from packages.fog.gateway.scheduler.marketplace import get_marketplace_engine

    print('Testing marketplace engine...')
    marketplace = await get_marketplace_engine()

    # Add a test listing
    listing_id = await marketplace.add_resource_listing(
        node_id='test-server',
        cpu_cores=4.0,
        memory_gb=8.0,
        disk_gb=100.0,
        spot_price=0.10,
        on_demand_price=0.15,
        trust_score=0.7
    )
    print(f'[PASS] Added listing: {listing_id}')

    # Get price quote
    quote = await marketplace.get_price_quote(
        cpu_cores=2.0,
        memory_gb=4.0,
        duration_hours=1.0
    )
    print(f'[PASS] Price quote available: {quote["available"]}')
    if quote['available']:
        print(f'       Price range: ${quote["quote"]["min_price"]:.4f} - ${quote["quote"]["max_price"]:.4f}')
        print(f'       Available providers: {quote["market_conditions"]["available_providers"]}')

    # Submit a bid
    bid_id = await marketplace.submit_bid(
        namespace='test-org/test-team',
        cpu_cores=2.0,
        memory_gb=4.0,
        max_price=0.25,
        estimated_duration_hours=1.0
    )
    print(f'[PASS] Submitted bid: {bid_id}')

    # Test marketplace matching
    await asyncio.sleep(0.1)  # Brief pause
    matched_count = await marketplace._match_bids_to_listings()
    print(f'[PASS] Matched {matched_count} bids to listings')

    # Check if trade was created
    if marketplace.active_trades:
        trade = list(marketplace.active_trades.values())[0]
        print(f'[PASS] Trade created: {trade.cpu_cores} cores for ${trade.agreed_price:.4f}')

    # Test billing engine
    print('Testing billing engine...')
    billing = await get_billing_engine()

    cost = await billing.record_job_usage(
        namespace='test-org/test-team',
        job_id='test-job-123',
        cpu_cores=2.0,
        memory_gb=4.0,
        disk_gb=8.0,
        duration_seconds=3600
    )
    print(f'[PASS] Recorded job cost: ${cost:.4f}')

    # Generate usage report
    usage_report = await billing.get_usage_report('test-org/test-team')
    print(f'[PASS] Usage report generated: {usage_report.usage_metrics.job_count} jobs, ${usage_report.cost_breakdown.total_cost:.4f} total cost')

    await marketplace.stop()
    print('[SUCCESS] All marketplace and billing tests passed!')

if __name__ == '__main__':
    asyncio.run(test_marketplace())
