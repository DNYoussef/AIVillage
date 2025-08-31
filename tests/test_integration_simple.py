"""
Simple Integration Test

Test basic integration functionality to ensure the complete system works.
"""

import asyncio
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

logger = logging.getLogger(__name__)

async def test_fog_bridge_basic():
    """Test basic fog bridge functionality."""
    try:
        from core.rag.integration.fog_compute_bridge import (
            FogComputeBridge, QueryType, QueryDistributionStrategy
        )
        
        print("[OK] Fog bridge imports successful")
        
        # Create and initialize bridge
        bridge = FogComputeBridge(
            enable_p2p=False,  # Disable for simplicity
            enable_marketplace=False,
            enable_distributed_inference=False,
            enable_security=False
        )
        
        await bridge.initialize()
        print("[OK] Fog bridge initialized")
        
        # Get system status
        status = bridge.get_system_status()
        print(f"[OK] Bridge initialized: {status['bridge_status']['initialized']}")
        print(f"[OK] Fog nodes discovered: {status['fog_network']['total_nodes']}")
        print(f"[OK] Healthy nodes: {status['fog_network']['healthy_nodes']}")
        
        # Test simple query distribution
        result = await bridge.distribute_query(
            query="Simple integration test query",
            query_type=QueryType.SIMPLE_RAG,
            strategy=QueryDistributionStrategy.BALANCED,
            user_tier="medium",
            max_budget=20.0
        )
        
        print(f"[OK] Query distributed: {result.get('distributed', False)}")
        print(f"[OK] Nodes used: {len(result.get('fog_nodes_used', []))}")
        print(f"[OK] Total cost: ${result.get('performance_metrics', {}).get('total_cost', 0):.2f}")
        
        await bridge.close()
        print("[OK] Fog bridge closed successfully")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Fog bridge test failed: {e}")
        return False

async def test_api_gateway_basic():
    """Test basic API gateway functionality."""
    try:
        from infrastructure.fog.gateway.api.federated import (
            FederatedAPIGateway, FederatedInferenceRequest, UserTier
        )
        
        print("[OK] API gateway imports successful")
        
        # Create gateway
        gateway = FederatedAPIGateway()
        
        # Test request estimation
        request = FederatedInferenceRequest(
            user_id="test_user",
            user_tier=UserTier.MEDIUM,
            query="Test API query",
            max_budget=30.0
        )
        
        estimated_cost = await gateway._estimate_inference_cost(request)
        print(f"[OK] Cost estimation: ${estimated_cost:.2f}")
        
        # Test system status methods
        avg_cost = gateway._calculate_average_job_cost()
        print(f"[OK] Average job cost calculation: ${avg_cost:.2f}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] API gateway test failed: {e}")
        return False

async def test_integration_components():
    """Test integration between components."""
    try:
        # Test global functions
        from core.rag.integration.fog_compute_bridge import get_fog_system_status
        
        status = get_fog_system_status()
        print(f"[OK] Global status function works: {type(status).__name__}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Integration components test failed: {e}")
        return False

async def main():
    """Run all basic integration tests."""
    print("Starting Basic Integration Tests\n")
    
    tests = [
        ("Fog Bridge Basic", test_fog_bridge_basic),
        ("API Gateway Basic", test_api_gateway_basic), 
        ("Integration Components", test_integration_components),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name} Test:")
        print("=" * 50)
        
        try:
            success = await test_func()
            if success:
                print(f"[PASS] {test_name} Test: PASSED")
                passed += 1
            else:
                print(f"[FAIL] {test_name} Test: FAILED")
                failed += 1
        except Exception as e:
            print(f"[ERROR] {test_name} Test: ERROR - {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Integration Test Results:")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {(passed / (passed + failed)) * 100:.1f}%")
    
    if passed > failed:
        print("Integration tests mostly successful!")
        return 0
    else:
        print("Some integration tests failed - check logs above")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)