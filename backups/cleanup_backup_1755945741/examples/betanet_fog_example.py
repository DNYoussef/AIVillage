"""
Example: Using BetaNet Transport in Fog Computing

This example demonstrates how fog compute nodes can use BetaNet transport
protocols for secure, privacy-preserving job distribution while keeping
the bounty code completely separate.

Run this example to see BetaNet integration in action:
    python packages/fog/examples/betanet_fog_example.py
"""

import asyncio
import json
import logging
from pathlib import Path

# Import the fog compute BetaNet bridge
try:
    from packages.fog.bridges.betanet_integration import (
        BetaNetFogTransport,
        FogComputeBetaNetService,
        get_betanet_capabilities,
        is_betanet_available,
    )
except ImportError:
    print("Error: Could not import BetaNet fog integration")
    print("Make sure you're running from the AIVillage root directory")
    exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_basic_transport():
    """Demonstrate basic BetaNet transport usage in fog computing"""
    print("\n🌫️ === Basic BetaNet Fog Transport Demo ===")

    # Check BetaNet availability
    capabilities = get_betanet_capabilities()
    print(f"BetaNet Capabilities: {capabilities}")

    # Create transport
    transport = BetaNetFogTransport(privacy_mode="balanced", enable_covert=True, mobile_optimization=True)

    # Example job data
    job_data = b"{'job_type': 'ml_inference', 'model': 'test_model', 'data_size': 1024}"
    destination = "fog-node-002.example.com"

    print(f"\n📤 Sending job data ({len(job_data)} bytes) to {destination}")

    # Send job data
    result = await transport.send_job_data(job_data=job_data, destination=destination, priority="high")

    print(f"✅ Send result: {result}")

    # Get transport statistics
    stats = transport.get_transport_stats()
    print(f"📊 Transport stats: {stats}")


async def demo_device_optimization():
    """Demonstrate device optimization features"""
    print("\n📱 === Device Optimization Demo ===")

    transport = BetaNetFogTransport(mobile_optimization=True)

    # Example device information
    device_info = {
        "battery_percent": 25,  # Low battery
        "cpu_temp": 45,  # Normal temperature
        "network_type": "cellular",
        "available_memory": 512,  # MB
        "device_type": "mobile",
    }

    print(f"Device info: {device_info}")

    # Get optimization recommendations
    optimization = await transport.optimize_for_device(device_info)
    print(f"🔧 Optimization recommendations: {optimization}")


async def demo_fog_service():
    """Demonstrate the high-level fog compute service"""
    print("\n🚀 === Fog Compute Service Demo ===")

    # Create fog compute service
    service = FogComputeBetaNetService()

    # Initialize service
    node_id = "fog-node-alpha"
    success = await service.initialize(
        node_id=node_id, privacy_mode="strict", enable_covert=True  # High privacy for sensitive jobs
    )

    if not success:
        print("❌ Failed to initialize BetaNet service")
        return

    print(f"✅ Initialized BetaNet service for node {node_id}")

    # Submit a job to another node
    job_data = {
        "job_type": "distributed_training",
        "model_params": {"layers": 12, "dim": 768},
        "data_shard": "shard_001.parquet",
        "epochs": 5,
        "batch_size": 32,
    }

    peer_node = "fog-node-beta"
    print(f"\n📋 Submitting job to peer node {peer_node}")

    job_id = await service.submit_job_to_peer(peer_node=peer_node, job_data=job_data, priority="normal")

    print(f"✅ Job submitted with ID: {job_id}")

    # Check job status
    status = await service.get_job_status(job_id)
    print(f"📊 Job status: {status}")

    # Get service statistics
    service_stats = service.get_service_stats()
    print(f"📈 Service statistics: {service_stats}")


async def demo_privacy_modes():
    """Demonstrate different privacy modes"""
    print("\n🔒 === Privacy Modes Demo ===")

    privacy_modes = ["performance", "balanced", "strict"]
    job_data = b"sensitive_model_weights.bin"

    for mode in privacy_modes:
        print(f"\n🔐 Testing {mode} privacy mode")

        transport = BetaNetFogTransport(privacy_mode=mode, enable_covert=True)

        result = await transport.send_job_data(
            job_data=job_data, destination="secure-node.example.com", priority="high"
        )

        print(f"  Transport: {result.get('transport')}")
        print(f"  Duration: {result.get('duration', 0):.3f}s")
        print(f"  Privacy hops: {result.get('privacy_hops', 0)}")


async def demo_covert_channels():
    """Demonstrate covert channel capabilities"""
    print("\n🕵️ === Covert Channels Demo ===")

    if not is_betanet_available():
        print("❌ BetaNet bounty not available - using fallback transport")
        print("📝 To enable covert channels, ensure BetaNet bounty is properly installed")
        return

    transport = BetaNetFogTransport(enable_covert=True)

    # Test different priority levels (each uses different covert channel)
    test_cases = [
        ("low", "routine_maintenance_job"),
        ("normal", "standard_inference_job"),
        ("high", "urgent_security_update"),
    ]

    for priority, job_description in test_cases:
        print(f"\n📡 Testing {priority} priority ({job_description})")

        job_payload = json.dumps(
            {"description": job_description, "priority": priority, "timestamp": asyncio.get_event_loop().time()}
        ).encode("utf-8")

        result = await transport.send_job_data(
            job_data=job_payload, destination="covert-endpoint.example.com", priority=priority
        )

        print(f"  ✅ Sent via {result.get('transport')} transport")
        print(f"  📊 {result.get('bytes_sent')} bytes in {result.get('chunks', 1)} chunks")


async def main():
    """Run all BetaNet fog compute demos"""
    print("🌐 BetaNet Fog Computing Integration Demo")
    print("=" * 50)

    # Check if we're in the right directory
    if not Path("packages/fog/bridges/betanet_integration.py").exists():
        print("❌ Error: Run this script from the AIVillage root directory")
        return

    try:
        await demo_basic_transport()
        await demo_device_optimization()
        await demo_fog_service()
        await demo_privacy_modes()
        await demo_covert_channels()

        print("\n✅ All demos completed successfully!")
        print("\n💡 Key Integration Points:")
        print("  • BetaNet bounty code remains completely separate")
        print("  • Fog compute gains secure transport capabilities")
        print("  • Privacy-preserving job distribution")
        print("  • Mobile-optimized resource usage")
        print("  • Covert channels for sensitive operations")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logger.exception("Demo error")


if __name__ == "__main__":
    asyncio.run(main())
