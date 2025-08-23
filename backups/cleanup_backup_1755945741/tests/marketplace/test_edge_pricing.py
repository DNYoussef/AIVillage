#!/usr/bin/env python3
"""
Test edge device pricing integration with marketplace
"""

import asyncio
import sys

sys.path.insert(0, "packages")


async def test_edge_pricing():
    from packages.fog.edge.beacon import CapabilityBeacon, DeviceType, PowerProfile
    from packages.fog.gateway.scheduler.marketplace import get_marketplace_engine

    print("Testing edge device pricing...")

    # Create a mobile device beacon
    beacon = CapabilityBeacon(
        device_name="test-mobile-device",
        operator_namespace="test-org/mobile",
        device_type=DeviceType.MOBILE_PHONE,
        betanet_endpoint="htx://mobile:8080",
    )

    # Set up device state
    beacon.capability.cpu_cores = 4.0
    beacon.capability.memory_mb = 6144  # 6GB
    beacon.capability.battery_percent = 85.0
    beacon.capability.power_profile = PowerProfile.BALANCED
    beacon.capability.thermal_state = "normal"
    beacon.capability.cpu_usage_percent = 30.0
    beacon.capability.memory_usage_percent = 45.0
    beacon.capability.trust_score = 0.75

    # Update pricing based on device conditions
    await beacon._update_marketplace_pricing()
    print("[PASS] Mobile device pricing calculated")
    print(f"       Spot price: ${beacon.capability.spot_price_per_cpu_hour:.4f}/cpu-hour")
    print(f"       On-demand price: ${beacon.capability.on_demand_price_per_cpu_hour:.4f}/cpu-hour")
    print(f"       Pricing tier: {beacon.capability.pricing_tier}")
    print(f"       Accepts bids: {beacon.capability.accepts_marketplace_bids}")

    # Get marketplace listing data
    listing_data = beacon.get_marketplace_listing()
    print("[PASS] Generated marketplace listing")
    print(f'       Available CPU: {listing_data["cpu_cores"]:.1f} cores')
    print(f'       Available memory: {listing_data["memory_gb"]:.1f} GB')
    print(f'       Trust score: {listing_data["trust_score"]:.2f}')

    # Test cost estimation
    estimated_cost = beacon.estimate_job_cost(
        cpu_cores=1.0, memory_gb=1.0, disk_gb=2.0, duration_hours=0.5, bid_type="spot"
    )
    print(f"[PASS] Estimated job cost: ${estimated_cost:.4f} for 0.5 hours")

    # Test marketplace integration
    marketplace = await get_marketplace_engine()

    # Add the mobile device listing to marketplace
    marketplace_listing_id = await marketplace.add_resource_listing(
        node_id=listing_data["node_id"],
        cpu_cores=listing_data["cpu_cores"],
        memory_gb=listing_data["memory_gb"],
        disk_gb=listing_data["disk_gb"],
        spot_price=listing_data["spot_price_per_cpu_hour"],
        on_demand_price=listing_data["on_demand_price_per_cpu_hour"],
        trust_score=listing_data["trust_score"],
    )
    print(f"[PASS] Added mobile device to marketplace: {marketplace_listing_id}")

    # Get price quote that should include mobile device
    quote = await marketplace.get_price_quote(
        cpu_cores=0.5, memory_gb=1.0, duration_hours=0.5, min_trust_score=0.7  # Small job suitable for mobile
    )

    if quote["available"]:
        print("[PASS] Price quote includes mobile device")
        print(f'       Price range: ${quote["quote"]["min_price"]:.4f} - ${quote["quote"]["max_price"]:.4f}')
        print(f'       Available providers: {quote["market_conditions"]["available_providers"]}')
    else:
        print(f'[FAIL] No resources available: {quote.get("reason", "Unknown")}')

    await marketplace.stop()
    print("[SUCCESS] Edge device pricing integration test passed!")


if __name__ == "__main__":
    asyncio.run(test_edge_pricing())
