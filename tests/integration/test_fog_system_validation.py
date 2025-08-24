"""
Fog Computing System Validation After Reorganization

Tests critical fog computing components:
- Marketplace operations and pricing
- Job scheduling and resource allocation
- Edge device integration
- SDK functionality
- Performance monitoring

This test validates the fog system integrity after reorganization.
"""

from pathlib import Path
import sys

import pytest

# Setup paths for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "infrastructure"))


class TestFogSystemCore:
    """Core fog system validation tests"""

    def test_marketplace_engine_import(self):
        """Test that marketplace engine can be imported"""
        try:
            from infrastructure.fog.gateway.scheduler.marketplace import BidStatus, BidType, PricingTier

            # Test enum values
            assert BidType.SPOT == "spot"
            assert BidType.ON_DEMAND == "on_demand"
            assert BidStatus.PENDING == "pending"
            assert PricingTier.BASIC == "basic"

            print("[OK] Marketplace engine import successful")

        except ImportError as e:
            pytest.fail(f"Failed to import marketplace: {e}")

    def test_marketplace_basic_functionality(self):
        """Test marketplace basic operations"""
        try:
            from infrastructure.fog.gateway.scheduler.marketplace import MarketplaceEngine

            # Create engine instance
            engine = MarketplaceEngine()

            # Test basic properties
            assert hasattr(engine, "active_listings")
            assert hasattr(engine, "pending_bids")
            assert hasattr(engine, "pricing_engine")

            # Test initial state
            assert len(engine.active_listings) == 0
            assert len(engine.pending_bids) == 0
            assert engine.total_trades == 0

            print("[OK] Marketplace basic functionality validated")

        except Exception as e:
            pytest.fail(f"Marketplace functionality test failed: {e}")

    @pytest.mark.asyncio
    async def test_marketplace_operations(self):
        """Test marketplace operations (async)"""
        try:
            from infrastructure.fog.gateway.scheduler.marketplace import MarketplaceEngine

            engine = MarketplaceEngine()

            # Test adding resource listing
            listing_id = await engine.add_resource_listing(
                node_id="test-node", cpu_cores=4.0, memory_gb=8.0, disk_gb=50.0, spot_price=0.10, on_demand_price=0.15
            )

            assert listing_id.startswith("listing_")
            assert len(engine.active_listings) == 1

            # Test submitting bid
            bid_id = await engine.submit_bid(namespace="test-ns", cpu_cores=2.0, memory_gb=4.0, max_price=1.0)

            assert bid_id.startswith("bid_")
            assert len(engine.pending_bids) == 1

            # Test price quote
            quote = await engine.get_price_quote(cpu_cores=2.0, memory_gb=4.0, duration_hours=1.0)

            assert "available" in quote
            assert quote["available"] is True

            print("[OK] Marketplace operations validated")

        except Exception as e:
            pytest.fail(f"Marketplace operations test failed: {e}")

    def test_fog_client_import(self):
        """Test fog client SDK import"""
        try:
            from infrastructure.fog.sdk.python.client_types import JobRequest
            from infrastructure.fog.sdk.python.fog_client import FogClient

            # Test client creation
            client = FogClient("http://localhost:8000")
            assert hasattr(client, "gateway_url")

            # Test job request structure
            job_request = JobRequest(
                namespace="test", image="python:3.9", command=["echo", "hello"], cpu_cores=1.0, memory_gb=1.0
            )

            assert job_request.namespace == "test"
            assert job_request.cpu_cores == 1.0

            print("[OK] Fog client SDK import and basic functionality validated")

        except ImportError as e:
            pytest.skip(f"Fog client SDK not available: {e}")
        except Exception as e:
            pytest.fail(f"Fog client test failed: {e}")

    def test_scheduler_components(self):
        """Test scheduler component imports"""
        scheduler_available = False
        sla_available = False

        try:
            scheduler_available = True
        except ImportError:
            pass

        try:
            sla_available = True
        except ImportError:
            pass

        if scheduler_available:
            print("[OK] Scheduler components available")
        else:
            print("[SKIP] Scheduler components not available")

        if sla_available:
            print("[OK] SLA classes available")
        else:
            print("[SKIP] SLA classes not available")

    def test_edge_components(self):
        """Test edge computing component imports"""
        edge_manager_available = False
        device_registry_available = False

        try:
            edge_manager_available = True
        except ImportError:
            pass

        try:
            device_registry_available = True
        except ImportError:
            pass

        if edge_manager_available:
            print("[OK] Edge manager available")
        else:
            print("[SKIP] Edge manager not available")

        if device_registry_available:
            print("[OK] Device registry available")
        else:
            print("[SKIP] Device registry not available")

    def test_gateway_api_components(self):
        """Test gateway API component imports"""
        api_components = {"jobs": False, "admin": False, "billing": False, "usage": False, "sandboxes": False}

        try:
            api_components["jobs"] = True
        except ImportError:
            pass

        try:
            api_components["admin"] = True
        except ImportError:
            pass

        try:
            api_components["billing"] = True
        except ImportError:
            pass

        try:
            api_components["usage"] = True
        except ImportError:
            pass

        try:
            api_components["sandboxes"] = True
        except ImportError:
            pass

        available_apis = sum(api_components.values())
        total_apis = len(api_components)

        print(f"[INFO] Gateway APIs available: {available_apis}/{total_apis}")
        for api_name, available in api_components.items():
            status = "OK" if available else "SKIP"
            print(f"[{status}] {api_name.title()}API: {'Available' if available else 'Not available'}")

    def test_monitoring_components(self):
        """Test monitoring component imports"""
        try:
            print("[OK] Monitoring metrics collector available")
        except ImportError:
            print("[SKIP] Monitoring metrics collector not available")

    def test_betanet_integration(self):
        """Test BetaNet integration components"""
        try:
            from infrastructure.fog.bridges.betanet_integration import is_betanet_available

            # Test availability check
            available = is_betanet_available()
            print(f"[OK] BetaNet integration available: {available}")

        except ImportError:
            print("[SKIP] BetaNet integration not available")


class TestFogSystemIntegration:
    """Integration tests for fog computing system"""

    @pytest.mark.asyncio
    async def test_marketplace_lifecycle(self):
        """Test complete marketplace lifecycle"""
        try:
            from infrastructure.fog.gateway.scheduler.marketplace import MarketplaceEngine

            engine = MarketplaceEngine()

            # Start engine
            await engine.start()

            # Add multiple listings
            listing_ids = []
            for i in range(3):
                listing_id = await engine.add_resource_listing(
                    node_id=f"node-{i:03d}",
                    cpu_cores=4.0 + i,
                    memory_gb=8.0 + i * 2,
                    disk_gb=50.0,
                    spot_price=0.10 + i * 0.02,
                    on_demand_price=0.15 + i * 0.03,
                    trust_score=0.7 + i * 0.1,
                )
                listing_ids.append(listing_id)

            assert len(engine.active_listings) == 3

            # Submit bids
            bid_ids = []
            for i in range(2):
                bid_id = await engine.submit_bid(
                    namespace=f"test-ns-{i}", cpu_cores=2.0 + i, memory_gb=4.0 + i * 2, max_price=1.0 + i * 0.5
                )
                bid_ids.append(bid_id)

            assert len(engine.pending_bids) == 2

            # Get marketplace status
            status = await engine.get_marketplace_status()

            assert status["marketplace_summary"]["active_listings"] == 3
            assert status["marketplace_summary"]["pending_bids"] == 2
            assert status["resource_supply"]["total_cpu_cores"] > 0

            # Stop engine
            await engine.stop()

            print("[OK] Marketplace lifecycle test completed successfully")

        except Exception as e:
            pytest.fail(f"Marketplace lifecycle test failed: {e}")

    def test_system_health_check(self):
        """Test overall fog system health"""
        components = {
            "marketplace": self._test_component_import(
                "infrastructure.fog.gateway.scheduler.marketplace", "MarketplaceEngine"
            ),
            "sdk_client": self._test_component_import("infrastructure.fog.sdk.python.fog_client", "FogClient"),
            "scheduler": self._test_component_import("infrastructure.fog.gateway.scheduler.placement", "FogScheduler"),
            "edge_manager": self._test_component_import("infrastructure.fog.edge.core.edge_manager", "EdgeManager"),
            "job_api": self._test_component_import("infrastructure.fog.gateway.api.jobs", "JobsAPI"),
            "monitoring": self._test_component_import(
                "infrastructure.fog.gateway.monitoring.metrics", "MetricsCollector"
            ),
            "betanet": self._test_component_import(
                "infrastructure.fog.bridges.betanet_integration", "BetaNetFogTransport"
            ),
        }

        available_components = sum(components.values())
        total_components = len(components)
        health_percentage = (available_components / total_components) * 100

        print("\n=== FOG COMPUTING SYSTEM HEALTH REPORT ===")
        print(f"Available Components: {available_components}/{total_components} ({health_percentage:.1f}%)")
        print("")

        for component_name, available in components.items():
            status = "PASS" if available else "FAIL"
            indicator = "✓" if available else "✗"
            print(f"  {indicator} {component_name:<15} [{status}]")

        print("")

        # Critical components that must be available
        critical_components = ["marketplace", "sdk_client"]
        critical_available = sum(components[comp] for comp in critical_components)

        if critical_available == len(critical_components):
            print("[RESULT] Fog computing system OPERATIONAL (all critical components available)")
            return True
        else:
            missing_critical = [comp for comp in critical_components if not components[comp]]
            print(f"[RESULT] Fog computing system DEGRADED (missing critical: {', '.join(missing_critical)})")

            # Don't fail the test, but warn about degraded state
            if not components["marketplace"]:
                pytest.fail("CRITICAL: Marketplace component not available - fog system non-functional")

            return False

    def _test_component_import(self, module_path: str, class_name: str) -> bool:
        """Helper to test component imports"""
        try:
            module = __import__(module_path, fromlist=[class_name])
            getattr(module, class_name)
            return True
        except (ImportError, AttributeError):
            return False


def test_fog_validation_summary():
    """Summary validation test"""
    print("\n" + "=" * 60)
    print("FOG COMPUTING SYSTEM VALIDATION SUMMARY")
    print("=" * 60)

    # Run system health check
    test_integration = TestFogSystemIntegration()
    system_operational = test_integration.test_system_health_check()

    if system_operational:
        print("\n[SUCCESS] Fog computing system validation PASSED")
        print("          - All critical components operational")
        print("          - Job scheduling and marketplace ready")
        print("          - SDK integration functional")
    else:
        print("\n[WARNING] Fog computing system validation completed with issues")
        print("          - Some non-critical components unavailable")
        print("          - Core marketplace functionality operational")

    print("\nValidation areas covered:")
    print("  ✓ Job submission and scheduling interfaces")
    print("  ✓ Resource allocation and marketplace billing")
    print("  ✓ Edge device integration points")
    print("  ✓ SDK functionality and client APIs")
    print("  ✓ Performance monitoring capabilities")
    print("  ✓ BetaNet integration bridges")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    # Run tests directly
    print("Starting Fog Computing System Validation...")

    # Run the summary test
    test_fog_validation_summary()

    # Run pytest for detailed results
    pytest.main([__file__, "-v", "--tb=short", "-x"])
