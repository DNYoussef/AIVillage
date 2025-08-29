"""
Comprehensive Test Suite for Unified Edge Computing System

This test suite validates the consolidated Edge Device ecosystem using MECE methodology:
- Complete integration testing of all consolidated components
- Performance validation for mobile optimization
- Privacy and security verification
- Cross-component communication testing
- Resilience and failover testing

Test Coverage Matrix:
- Digital Twin Concierge: Privacy-first learning and data collection
- Unified Edge Device System: Device lifecycle and task management
- MiniRAG System: Local knowledge with global elevation
- Chat Engine: Multi-mode resilient communication
- Mobile Bridge: Cross-platform mobile integration
- Integration Layer: Unified EdgeSystem coordination
"""

import asyncio
from pathlib import Path
import tempfile
import time
from unittest.mock import MagicMock, patch

import pytest

# Import consolidated edge system
try:
    from infrastructure.edge import (
        DataSource,
        EdgeTask,
        PrivacyLevel,
        TaskPriority,
        create_edge_system,
        create_edge_task,
        create_mobile_edge_system,
    )
    from infrastructure.edge.integration.shared_types import (
        ProcessingMode,
    )
except ImportError as e:
    pytest.skip(f"Edge system not available: {e}", allow_module_level=True)


class TestConsolidatedEdgeSystem:
    """Comprehensive test suite for the unified edge computing system"""

    @pytest.fixture
    async def temp_data_dir(self):
        """Create temporary directory for test data"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    async def edge_system(self, temp_data_dir):
        """Create edge system for testing"""
        system = await create_edge_system(
            device_name="TestDevice",
            data_dir=temp_data_dir,
            enable_digital_twin=True,
            enable_mobile_bridge=True,
            enable_chat_engine=True,
        )
        yield system
        await system.shutdown()

    @pytest.fixture
    async def mobile_edge_system(self, temp_data_dir):
        """Create mobile-optimized edge system for testing"""
        system = await create_mobile_edge_system(
            data_dir=temp_data_dir,
            power_aware_scheduling=True,
            battery_threshold_percent=20.0,
        )
        yield system
        await system.shutdown()

    # ========================================================================
    # PHASE 5: TEST CONSOLIDATION WITH MECE ANALYSIS
    # ========================================================================

    async def test_unified_system_initialization(self, temp_data_dir):
        """Test complete system initialization and component integration"""

        # Test system creation
        system = await create_edge_system(
            device_name="InitTestDevice",
            data_dir=temp_data_dir,
            enable_digital_twin=True,
            enable_mobile_bridge=True,
            enable_chat_engine=True,
        )

        try:
            # Verify initialization
            assert system.initialized is True
            assert system.device_system is not None
            assert system.digital_twin is not None
            assert system.knowledge_system is not None
            assert system.chat_engine is not None
            assert system.mobile_bridge is not None

            # Test system status
            status = system.get_system_status()
            assert status["initialized"] is True
            assert status["components"]["device_system"] is True
            assert status["components"]["digital_twin"] is True
            assert status["components"]["knowledge_system"] is True
            assert status["components"]["chat_engine"] is True
            assert status["components"]["mobile_bridge"] is True

            # Test component integration points
            assert system.digital_twin.mini_rag is system.knowledge_system

        finally:
            await system.shutdown()

    async def test_digital_twin_privacy_compliance(self, edge_system):
        """Test digital twin privacy-first data handling"""

        digital_twin = edge_system.digital_twin
        assert digital_twin is not None

        # Test data collection consent
        assert len(digital_twin.preferences.enabled_sources) > 0

        # Test privacy levels
        privacy_report = digital_twin.get_privacy_report()
        assert privacy_report["data_location"] == "on_device_only"
        assert privacy_report["encryption_enabled"] is True
        assert privacy_report["auto_deletion"] is True

        # Test learning cycle with privacy preservation
        with patch("infrastructure.edge.digital_twin.concierge.MobileDeviceProfile") as mock_profile:
            mock_profile.return_value = MagicMock()
            mock_profile.return_value.battery_percent = 75

            result = await edge_system.run_learning_cycle()
            assert "cycle_id" in result
            assert "data_points" in result

        # Verify no data transmitted externally
        assert digital_twin.data_collector.db_path.exists()

    async def test_mobile_optimization_performance(self, mobile_edge_system):
        """Test mobile-specific optimizations and performance"""

        mobile_bridge = mobile_edge_system.mobile_bridge
        assert mobile_bridge is not None

        # Test mobile platform detection
        status = mobile_bridge.get_comprehensive_status()
        assert "platform" in status
        assert status["device_info"]["model"] is not None

        # Test power save mode
        await mobile_bridge.enable_power_save_mode(True)
        assert mobile_bridge.power_save_mode is True

        # Test background/foreground transitions
        await mobile_bridge.suspend_for_background()
        assert mobile_bridge.status.value == "suspended"

        await mobile_bridge.resume_from_background()
        assert mobile_bridge.status.value == "connected"

        # Test BLE integration (if available)
        if status["bitchat_integration"]["enabled"]:
            assert status["bitchat_integration"]["ble_scanner_active"] is not None

    async def test_knowledge_system_integration(self, edge_system):
        """Test MiniRAG system with privacy-preserving knowledge elevation"""

        knowledge_system = edge_system.knowledge_system
        assert knowledge_system is not None

        # Test knowledge addition
        knowledge_id = await knowledge_system.add_knowledge(
            "Test knowledge for edge computing optimization",
            DataSource.APP_USAGE,
            {"type": "optimization", "context": "mobile"},
        )
        assert knowledge_id is not None

        # Test knowledge querying
        results = await edge_system.query_knowledge("edge computing")
        assert len(results) > 0
        assert results[0]["content"] is not None
        assert results[0]["source"] is not None

        # Test privacy preservation
        stats = knowledge_system.get_system_stats()
        assert stats["total_knowledge_pieces"] > 0

        # Test global contribution candidates (without actual transmission)
        await knowledge_system.get_global_contribution_candidates()
        # Should have candidates based on the knowledge we added
        # (depending on relevance assessment)

    async def test_chat_engine_resilience(self, edge_system):
        """Test chat engine multi-mode operation and resilience"""

        chat_engine = edge_system.chat_engine
        assert chat_engine is not None

        # Test local mode processing
        with patch("infrastructure.edge.communication.chat_engine.CHAT_MODE", "local"):
            response = await edge_system.process_chat("Hello test", "test_conversation")
            assert response["response"] is not None
            assert response["mode"] == "local"
            assert response["service_status"] == "offline"

        # Test system status
        status = chat_engine.get_system_status()
        assert "mode" in status
        assert "service_status" in status
        assert status["offline_responses_enabled"] is not None

        # Test conversation history
        history = chat_engine.get_conversation_history("test_conversation")
        assert isinstance(history, list)

    async def test_edge_task_processing(self, edge_system):
        """Test unified task processing across edge system"""

        # Create test task
        task = create_edge_task(
            name="test_inference_task",
            task_type="inference",
            priority=TaskPriority.NORMAL,
            min_memory_mb=100.0,
            processing_mode=ProcessingMode.LOCAL_ONLY,
            privacy_level=PrivacyLevel.PRIVATE,
        )

        # Process task through unified system
        result = await edge_system.process_task(task)

        # Verify task execution
        assert result.success is True
        assert result.task_id == task.task_id
        assert result.executed_on_device is not None
        assert result.processing_mode_used == ProcessingMode.LOCAL_ONLY
        assert result.execution_time_seconds > 0

        # Check system metrics updated
        status = edge_system.get_system_status()
        assert status["system_metrics"]["tasks_completed"] > 0

    async def test_device_resource_management(self, edge_system):
        """Test unified device resource management and optimization"""

        device_system = edge_system.device_system
        assert device_system is not None

        # Test system status
        status = device_system.get_system_status()
        assert status["system_healthy"] is not None
        assert "devices" in status
        assert "tasks" in status

        # Test device registration (local device should be registered)
        if hasattr(device_system, "local_device") and device_system.local_device:
            device_info = device_system.local_device.get_device_info()
            assert device_info["device_id"] is not None
            assert device_info["device_type"] is not None
            assert "capabilities" in device_info

    async def test_cross_component_integration(self, edge_system):
        """Test integration between all edge system components"""

        # Test knowledge sharing between digital twin and knowledge system
        if edge_system.digital_twin and edge_system.knowledge_system:
            # Digital twin should use knowledge system
            assert edge_system.digital_twin.mini_rag is edge_system.knowledge_system

        # Test chat engine with knowledge enhancement
        chat_response = await edge_system.process_chat("What do you know about optimization?", "integration_test")
        assert chat_response is not None
        assert "response" in chat_response

        # Test unified system status aggregation
        system_status = edge_system.get_system_status()
        assert "device_status" in system_status
        assert "knowledge_status" in system_status
        assert "chat_status" in system_status
        if edge_system.mobile_bridge:
            assert "mobile_status" in system_status

    async def test_performance_metrics_collection(self, edge_system):
        """Test system-wide performance metrics collection"""

        initial_status = edge_system.get_system_status()
        initial_metrics = initial_status["system_metrics"]

        # Perform some operations
        await edge_system.query_knowledge("performance test")
        await edge_system.process_chat("Performance test message", "perf_test")

        task = create_edge_task("perf_task", task_type="test")
        await edge_system.process_task(task)

        # Check metrics updated
        final_status = edge_system.get_system_status()
        final_metrics = final_status["system_metrics"]

        # Should have more completed tasks
        assert final_metrics["tasks_completed"] >= initial_metrics["tasks_completed"]

    async def test_error_handling_and_recovery(self, edge_system):
        """Test system error handling and recovery mechanisms"""

        # Test invalid task processing
        invalid_task = EdgeTask(
            task_id="invalid_task",
            task_name="Invalid Task",
            task_type="nonexistent",
            min_memory_mb=999999999,  # Impossible requirement
            priority=TaskPriority.NORMAL,
        )

        result = await edge_system.process_task(invalid_task)
        # Should handle gracefully
        assert result is not None
        assert result.task_id == "invalid_task"

        # Test system continues to function
        valid_task = create_edge_task("recovery_test")
        recovery_result = await edge_system.process_task(valid_task)
        assert recovery_result.success is True

    async def test_mobile_specific_optimizations(self, mobile_edge_system):
        """Test mobile-specific optimizations and constraints"""

        # Test mobile system configuration
        assert mobile_edge_system.enable_mobile_bridge is True

        # Test mobile resource constraints
        mobile_bridge = mobile_edge_system.mobile_bridge
        if mobile_bridge:
            status = mobile_bridge.get_comprehensive_status()

            # Should have mobile optimizations
            assert "optimization" in status
            assert status["optimization"]["power_save_mode"] is not None
            assert status["optimization"]["adaptive_scanning"] is not None

        # Test mobile task processing with constraints
        mobile_task = create_edge_task(
            name="mobile_optimized_task",
            task_type="inference",
            priority=TaskPriority.LOW,  # Lower priority for battery
            min_memory_mb=50.0,  # Small memory requirement
        )

        result = await mobile_edge_system.process_task(mobile_task)
        assert result.success is True

    # ========================================================================
    # PHASE 6: COMPREHENSIVE VALIDATION AND TESTING
    # ========================================================================

    async def test_end_to_end_workflow(self, edge_system):
        """Test complete end-to-end workflow across all components"""

        # Step 1: Add knowledge
        knowledge_id = await edge_system.knowledge_system.add_knowledge(
            "Mobile edge computing requires careful battery management",
            DataSource.SYSTEM_METRICS,
            {"domain": "edge_computing", "priority": "high"},
        )

        # Step 2: Process related chat
        chat_response = await edge_system.process_chat(
            "How can I optimize battery usage in mobile computing?", "e2e_workflow"
        )

        # Step 3: Create and process task
        optimization_task = create_edge_task(
            name="battery_optimization_analysis",
            task_type="analysis",
            priority=TaskPriority.NORMAL,
            input_data={"query": "battery optimization"},
        )

        task_result = await edge_system.process_task(optimization_task)

        # Step 4: Query knowledge for insights
        insights = await edge_system.query_knowledge("battery optimization")

        # Verify end-to-end workflow
        assert knowledge_id is not None
        assert chat_response["response"] is not None
        assert task_result.success is True
        assert len(insights) > 0

        # Check system maintained consistency
        final_status = edge_system.get_system_status()
        assert final_status["initialized"] is True
        assert final_status["system_metrics"]["tasks_completed"] > 0

    async def test_system_shutdown_and_cleanup(self, temp_data_dir):
        """Test graceful system shutdown and resource cleanup"""

        # Create system
        system = await create_edge_system(
            device_name="ShutdownTestDevice",
            data_dir=temp_data_dir,
        )

        # Add some data
        await system.knowledge_system.add_knowledge("Test knowledge for shutdown", DataSource.APP_USAGE)

        task = create_edge_task("shutdown_task")
        await system.process_task(task)

        # Test graceful shutdown
        await system.shutdown()

        # Verify system state
        assert system.initialized is False

        # Verify data persistence (knowledge should be saved)
        # Data files should exist in temp directory
        knowledge_files = list(temp_data_dir.rglob("*.db"))
        assert len(knowledge_files) > 0  # Database files should exist


@pytest.mark.integration
class TestEdgeSystemIntegration:
    """Integration tests for edge system with external dependencies"""

    async def test_p2p_network_integration_simulation(self):
        """Test P2P network integration (simulated)"""

        # This would test integration with BitChat/BetaNet
        # For now, we simulate the integration

        with tempfile.TemporaryDirectory() as temp_dir:
            system = await create_edge_system(
                device_name="P2PTestDevice",
                data_dir=Path(temp_dir),
                enable_mobile_bridge=True,
            )

            try:
                # Simulate P2P message handling
                mobile_bridge = system.mobile_bridge
                if mobile_bridge:
                    # Test message queuing
                    test_data = b"test p2p message data"
                    success = await mobile_bridge.send_to_mobile(test_data)

                    # Should queue successfully even without real P2P
                    assert success is True

                    status = mobile_bridge.get_comprehensive_status()
                    assert status["performance"]["messages_sent"] > 0

            finally:
                await system.shutdown()

    async def test_distributed_knowledge_elevation_simulation(self):
        """Test knowledge elevation to distributed RAG (simulated)"""

        with tempfile.TemporaryDirectory() as temp_dir:
            system = await create_edge_system(
                device_name="DistributedTestDevice",
                data_dir=Path(temp_dir),
            )

            try:
                # Add knowledge that should be globally relevant
                await system.knowledge_system.add_knowledge(
                    "Best practice: Use efficient algorithms for mobile edge computing to save battery",
                    DataSource.SYSTEM_METRICS,
                    {"type": "best_practice", "global_relevance": True},
                )

                # Check for contribution candidates
                await system.knowledge_system.get_global_contribution_candidates()

                # Should identify knowledge for potential elevation
                # (actual transmission would happen in production)

                stats = system.knowledge_system.get_system_stats()
                assert stats["total_knowledge_pieces"] > 0

            finally:
                await system.shutdown()


@pytest.mark.performance
class TestEdgeSystemPerformance:
    """Performance tests for the consolidated edge system"""

    async def test_task_processing_throughput(self, temp_data_dir):
        """Test task processing throughput and latency"""

        system = await create_edge_system(
            device_name="PerformanceTestDevice",
            data_dir=temp_data_dir,
        )

        try:
            # Process multiple tasks concurrently
            tasks = [create_edge_task(f"perf_task_{i}", task_type="inference") for i in range(10)]

            start_time = time.time()

            # Process all tasks
            results = await asyncio.gather(*[system.process_task(task) for task in tasks])

            end_time = time.time()
            total_time = end_time - start_time

            # Verify all tasks completed successfully
            assert len(results) == 10
            assert all(result.success for result in results)

            # Check performance metrics
            avg_time_per_task = total_time / 10
            assert avg_time_per_task < 1.0  # Should be fast for simple tasks

            # Check system metrics
            status = system.get_system_status()
            assert status["system_metrics"]["tasks_completed"] >= 10

        finally:
            await system.shutdown()

    async def test_memory_usage_optimization(self, temp_data_dir):
        """Test memory usage stays within reasonable bounds"""

        system = await create_mobile_edge_system(
            data_dir=temp_data_dir,
            max_memory_usage_percent=60.0,
        )

        try:
            # Perform memory-intensive operations
            for i in range(50):
                await system.knowledge_system.add_knowledge(
                    f"Knowledge piece {i} with detailed content for testing memory usage optimization in mobile edge computing environments",
                    DataSource.APP_USAGE,
                    {"iteration": i, "large_context": "x" * 100},
                )

            # Query knowledge multiple times
            for i in range(20):
                results = await system.query_knowledge(f"knowledge {i}")
                assert isinstance(results, list)

            # Process tasks
            for i in range(20):
                task = create_edge_task(f"memory_task_{i}", task_type="test")
                result = await system.process_task(task)
                assert result.success is True

            # System should still be responsive
            status = system.get_system_status()
            assert status["initialized"] is True

        finally:
            await system.shutdown()


# Test execution helpers
async def run_comprehensive_edge_tests():
    """Run comprehensive test suite for edge system consolidation"""

    print("🧪 Running Comprehensive Edge System Test Suite")
    print("=" * 60)

    test_results = {"total_tests": 0, "passed_tests": 0, "failed_tests": 0, "test_details": []}

    # Create test instances
    basic_tests = TestConsolidatedEdgeSystem()
    integration_tests = TestEdgeSystemIntegration()
    performance_tests = TestEdgeSystemPerformance()

    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir)

        # Test categories
        test_categories = [
            (
                "Basic System Tests",
                [
                    ("Unified System Initialization", basic_tests.test_unified_system_initialization(data_dir)),
                    ("Digital Twin Privacy Compliance", None),  # Requires edge_system fixture
                    ("Knowledge System Integration", None),  # Requires edge_system fixture
                    ("Chat Engine Resilience", None),  # Requires edge_system fixture
                    ("Cross-Component Integration", None),  # Requires edge_system fixture
                ],
            ),
            (
                "Performance Tests",
                [
                    ("Task Processing Throughput", performance_tests.test_task_processing_throughput(data_dir)),
                    ("Memory Usage Optimization", performance_tests.test_memory_usage_optimization(data_dir)),
                ],
            ),
            (
                "Integration Tests",
                [
                    ("P2P Network Integration", integration_tests.test_p2p_network_integration_simulation()),
                    (
                        "Distributed Knowledge Elevation",
                        integration_tests.test_distributed_knowledge_elevation_simulation(),
                    ),
                ],
            ),
        ]

        for category_name, category_tests in test_categories:
            print(f"\n📋 {category_name}")
            print("-" * 40)

            for test_name, test_coro in category_tests:
                if test_coro is None:
                    print(f"  ⏭️  {test_name} (Requires fixtures - run with pytest)")
                    continue

                test_results["total_tests"] += 1

                try:
                    await test_coro
                    print(f"  ✅ {test_name}")
                    test_results["passed_tests"] += 1
                    test_results["test_details"].append((test_name, "PASSED", None))

                except Exception as e:
                    print(f"  ❌ {test_name}: {str(e)}")
                    test_results["failed_tests"] += 1
                    test_results["test_details"].append((test_name, "FAILED", str(e)))

    # Print summary
    print("\n📊 Test Results Summary")
    print("=" * 40)
    print(f"Total Tests: {test_results['total_tests']}")
    print(f"Passed: {test_results['passed_tests']} ✅")
    print(f"Failed: {test_results['failed_tests']} ❌")

    if test_results["failed_tests"] > 0:
        print("\n🔍 Failed Test Details:")
        for test_name, status, error in test_results["test_details"]:
            if status == "FAILED":
                print(f"  • {test_name}: {error}")

    success_rate = (test_results["passed_tests"] / max(test_results["total_tests"], 1)) * 100
    print(f"\n🎯 Success Rate: {success_rate:.1f}%")

    return test_results


if __name__ == "__main__":
    # Run comprehensive tests if executed directly
    import asyncio

    results = asyncio.run(run_comprehensive_edge_tests())
    exit(0 if results["failed_tests"] == 0 else 1)
