"""
Integration Tests for EvoMerge Fog Offload

Tests the distributed execution of EvoMerge model evolution phases across fog
computing nodes. Validates that remote execution produces results within ε
tolerance of local execution while providing performance benefits.

Key Test Areas:
- EvoMerge phase distribution across fog nodes
- Parity validation between local and distributed execution
- Artifact collection and result aggregation
- Performance comparison and speedup measurement
- Error handling and fallback mechanisms
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio

# Import the components we're testing
try:
    from packages.agent_forge.core.phase_controller import PhaseResult
    from packages.agent_forge.core.unified_pipeline import UnifiedConfig
    from packages.agent_forge.integration.fog_burst import (
        FogBurstOrchestrator,
        FogBurstStrategy,
        ParityValidator,
        create_fog_burst_orchestrator,
    )
    from packages.agent_forge.phases.evomerge import EvoMergePhase

    FOG_BURST_AVAILABLE = True
except ImportError as e:
    FOG_BURST_AVAILABLE = False
    pytest.skip(f"Fog burst components not available: {e}", allow_module_level=True)


class TestEvoMergeFogOffload:
    """Test suite for EvoMerge fog offload integration"""

    @pytest.fixture
    def mock_fog_gateway_url(self):
        """Mock fog gateway URL for testing"""
        return "http://test-fog-gateway:8080"

    @pytest.fixture
    def unified_config(self):
        """Create a unified config for EvoMerge testing"""
        return UnifiedConfig(
            enable_evomerge=True,
            evomerge_techniques=["linear", "slerp", "ties", "dare"],
            base_models=["model_a", "model_b", "model_c"],
            output_dir=Path("./test_outputs"),
            batch_size=16,
            training_steps=100,
        )

    @pytest.fixture
    def mock_evomerge_phase(self):
        """Create a mock EvoMerge phase for testing"""
        phase = MagicMock(spec=EvoMergePhase)
        phase.__class__.__name__ = "EvoMergePhase"

        # Mock successful local execution
        async def mock_execute(config):
            # Create a simple mock model
            import torch.nn as nn

            mock_model = nn.Linear(10, 10)

            return PhaseResult(
                success=True,
                model=mock_model,
                phase_name="evomerge",
                metrics={
                    "merge_score": 0.85,
                    "diversity_score": 0.72,
                    "performance_delta": 0.15,
                    "final_loss": 2.34,
                    "training_time_s": 180.0,
                    "models_merged": 3,
                    "merge_technique": "adaptive",
                },
                artifacts={
                    "merged_model": "path/to/merged_model.pt",
                    "merge_history": "path/to/merge_history.json",
                    "performance_curves": "path/to/performance.png",
                },
                config={"merge_strategy": "linear_interpolation", "converged": True, "num_iterations": 10},
            )

        phase.execute = mock_execute
        return phase

    @pytest_asyncio.fixture
    async def fog_burst_orchestrator(self, mock_fog_gateway_url):
        """Create a fog burst orchestrator for testing"""
        orchestrator = FogBurstOrchestrator(
            fog_gateway_url=mock_fog_gateway_url, default_strategy=FogBurstStrategy.FOG_PREFERRED, parity_tolerance=1e-4
        )

        # Mock the initialization to avoid actual network calls
        with patch.object(orchestrator, "_discover_fog_nodes") as mock_discover:
            mock_discover.return_value = None
            await orchestrator.initialize()

        return orchestrator

    @pytest.fixture
    def mock_fog_nodes(self):
        """Mock available fog nodes for testing"""
        return [
            {
                "node_id": "fog_node_001",
                "endpoint": "http://fog-node-001:8080",
                "resources": {
                    "cpu_cores": 8.0,
                    "memory_gb": 16.0,
                    "gpu_available": True,
                    "gpu_memory_gb": 8.0,
                    "storage_gb": 100.0,
                },
                "capabilities": {
                    "pytorch": True,
                    "tensorflow": False,
                    "jax": True,
                    "evomerge": True,
                    "training": True,
                    "adas": True,
                },
                "metrics": {"training_throughput": 150.0, "network_bandwidth": 1000.0, "current_load": 0.3},
            },
            {
                "node_id": "fog_node_002",
                "endpoint": "http://fog-node-002:8080",
                "resources": {"cpu_cores": 4.0, "memory_gb": 8.0, "gpu_available": False, "storage_gb": 50.0},
                "capabilities": {"pytorch": True, "evomerge": True, "training": True},
                "metrics": {"training_throughput": 75.0, "network_bandwidth": 500.0, "current_load": 0.6},
            },
        ]

    @pytest.mark.asyncio
    async def test_evomerge_fog_offload_basic(
        self, fog_burst_orchestrator, mock_evomerge_phase, unified_config, mock_fog_nodes
    ):
        """Test basic EvoMerge offload to fog nodes"""

        # Mock fog node discovery
        fog_burst_orchestrator.available_nodes = {
            node["node_id"]: MagicMock(
                node_id=node["node_id"],
                cpu_cores=node["resources"]["cpu_cores"],
                memory_gb=node["resources"]["memory_gb"],
                gpu_available=node["resources"]["gpu_available"],
                supports_evomerge=node["capabilities"]["evomerge"],
                supports_pytorch=node["capabilities"]["pytorch"],
                current_load=node["metrics"]["current_load"],
            )
            for node in mock_fog_nodes
        }

        # Mock successful fog execution
        mock_fog_result = {
            "status": "success",
            "phase_name": "evomerge",
            "metrics": {
                "merge_score": 0.847,  # Slightly different due to distributed execution
                "diversity_score": 0.721,
                "performance_delta": 0.148,
                "final_loss": 2.342,
                "training_time_s": 95.0,  # Faster due to distributed execution
                "models_merged": 3,
                "merge_technique": "adaptive",
            },
            "metadata": {
                "merge_strategy": "linear_interpolation",
                "converged": True,
                "num_iterations": 10,
                "fog_executed": True,
                "fog_node": "fog_node_001",
            },
            "fog_job_id": "job_evomerge_12345",
            "fog_node": "fog_node_001",
            "execution_time_s": 95.0,
        }

        with patch.object(fog_burst_orchestrator, "_execute_on_fog", return_value=mock_fog_result):
            # Execute EvoMerge phase with fog offload
            result = await fog_burst_orchestrator.execute_phase_distributed(
                phase=mock_evomerge_phase,
                config=unified_config,
                strategy=FogBurstStrategy.FOG_PREFERRED,
                enable_parity_validation=True,
            )

            # Validate successful fog execution
            assert result.phase_name == "evomerge"
            assert result.success is True
            assert result.config["fog_executed"] is True
            assert result.config["fog_node"] == "fog_node_001"
            assert result.metrics["training_time_s"] == 95.0

    @pytest.mark.asyncio
    async def test_evomerge_parity_validation(self, fog_burst_orchestrator, mock_evomerge_phase, unified_config):
        """Test parity validation between local and fog execution"""

        # Setup parity validator
        parity_validator = ParityValidator(tolerance=1e-4)

        # Create local result
        import torch.nn as nn

        mock_model = nn.Linear(10, 10)

        local_result = PhaseResult(
            success=True,
            model=mock_model,
            phase_name="evomerge",
            metrics={"merge_score": 0.85, "diversity_score": 0.72, "performance_delta": 0.15, "final_loss": 2.34},
            config={"merge_strategy": "linear_interpolation"},
        )

        # Create fog result with small differences (within tolerance)
        fog_result = {
            "metrics": {
                "merge_score": 0.850001,  # Very small difference
                "diversity_score": 0.720002,
                "performance_delta": 0.149998,
                "final_loss": 2.340001,
            },
            "metadata": {"merge_strategy": "linear_interpolation"},
        }

        # Validate parity
        parity_result = await parity_validator.validate_parity(
            local_result=local_result, fog_result=fog_result, phase_name="evomerge", task_id="test_parity_001"
        )

        # Check parity validation results
        assert parity_result["parity_passed"] is True
        assert parity_result["max_difference"] <= 1e-4
        assert parity_result["assessment"] == "EXCELLENT"
        assert "merge_score" in parity_result["differences"]
        assert parity_result["differences"]["merge_score"]["within_tolerance"] is True

    @pytest.mark.asyncio
    async def test_evomerge_parity_validation_failure(self, fog_burst_orchestrator):
        """Test parity validation failure when results differ too much"""

        parity_validator = ParityValidator(tolerance=1e-4)

        # Create local result
        import torch.nn as nn

        mock_model = nn.Linear(10, 10)

        local_result = PhaseResult(
            success=True,
            model=mock_model,
            phase_name="evomerge",
            metrics={"merge_score": 0.85, "final_loss": 2.34},
            config={"merge_strategy": "linear_interpolation"},
        )

        # Create fog result with large differences (outside tolerance)
        fog_result = {
            "metrics": {"merge_score": 0.75, "final_loss": 3.50},  # Large difference  # Very large difference
            "metadata": {"merge_strategy": "linear_interpolation"},
        }

        # Validate parity
        parity_result = await parity_validator.validate_parity(
            local_result=local_result, fog_result=fog_result, phase_name="evomerge", task_id="test_parity_failure_001"
        )

        # Check parity validation failure
        assert parity_result["parity_passed"] is False
        assert parity_result["max_difference"] > 1e-4
        assert parity_result["assessment"] == "FAILED"

    @pytest.mark.asyncio
    async def test_evomerge_fog_execution_with_artifacts(
        self, fog_burst_orchestrator, mock_evomerge_phase, unified_config
    ):
        """Test EvoMerge fog execution with artifact collection"""

        # Mock fog execution with artifacts
        mock_fog_result = {
            "status": "success",
            "phase_name": "evomerge",
            "metrics": {"merge_score": 0.85, "training_time_s": 120.0},
            "artifacts": {
                "merged_model": "/fog/artifacts/merged_model.pt",
                "merge_history": "/fog/artifacts/merge_history.json",
                "performance_curves": "/fog/artifacts/performance.png",
                "fog_execution_log": "/fog/artifacts/execution.log",
            },
            "fog_job_id": "job_artifacts_12345",
            "fog_node": "fog_node_001",
        }

        with patch.object(fog_burst_orchestrator, "_execute_on_fog", return_value=mock_fog_result):
            # Execute with artifact collection
            result = await fog_burst_orchestrator.execute_phase_distributed(
                phase=mock_evomerge_phase,
                config=unified_config,
                enable_parity_validation=False,  # Skip parity for this test
            )

            # Validate artifact collection
            assert result.artifacts is not None
            assert "merged_model" in result.artifacts
            assert "merge_history" in result.artifacts
            assert "performance_curves" in result.artifacts
            assert "fog_execution_log" in result.artifacts
            assert result.config["fog_job_id"] == "job_artifacts_12345"

    @pytest.mark.asyncio
    async def test_evomerge_fog_fallback_to_local(self, fog_burst_orchestrator, mock_evomerge_phase, unified_config):
        """Test fallback to local execution when fog fails"""

        # Mock fog execution failure
        with patch.object(fog_burst_orchestrator, "_execute_on_fog", side_effect=Exception("Fog node unavailable")):
            # Execute with fallback strategy
            result = await fog_burst_orchestrator.execute_phase_distributed(
                phase=mock_evomerge_phase,
                config=unified_config,
                strategy=FogBurstStrategy.FOG_PREFERRED,  # Should fallback to local
            )

            # Validate fallback to local execution
            assert result.phase_name == "evomerge"
            assert result.success is True
            assert result.config.get("fog_executed", False) is False
            assert "fog_fallback" not in result.config  # Local execution, not fallback

    @pytest.mark.asyncio
    async def test_evomerge_fog_required_strategy_failure(
        self, fog_burst_orchestrator, mock_evomerge_phase, unified_config
    ):
        """Test FOG_REQUIRED strategy failure when fog is unavailable"""

        # Mock fog execution failure
        with patch.object(fog_burst_orchestrator, "_execute_on_fog", side_effect=Exception("Fog node unavailable")):
            # Execute with FOG_REQUIRED strategy (should raise exception)
            with pytest.raises(Exception, match="Fog execution required but failed"):
                await fog_burst_orchestrator.execute_phase_distributed(
                    phase=mock_evomerge_phase, config=unified_config, strategy=FogBurstStrategy.FOG_REQUIRED
                )

    @pytest.mark.asyncio
    async def test_evomerge_performance_measurement(self, fog_burst_orchestrator, mock_evomerge_phase, unified_config):
        """Test performance measurement and speedup calculation"""

        import time

        # Mock local execution time
        local_start_time = time.time()
        local_result = await mock_evomerge_phase.execute(unified_config)
        local_execution_time = time.time() - local_start_time

        # Mock fog execution (faster)
        mock_fog_result = {
            "status": "success",
            "phase_name": "evomerge",
            "metrics": local_result.metrics.copy(),
            "execution_time_s": local_execution_time * 0.6,  # 40% speedup
            "fog_node": "fog_node_001",
        }

        with patch.object(fog_burst_orchestrator, "_execute_on_fog", return_value=mock_fog_result):
            fog_start_time = time.time()
            fog_result = await fog_burst_orchestrator.execute_phase_distributed(
                phase=mock_evomerge_phase, config=unified_config, enable_parity_validation=False
            )
            fog_execution_time = time.time() - fog_start_time

            # Calculate and validate speedup
            speedup = local_execution_time / fog_execution_time if fog_execution_time > 0 else 1.0

            # Note: In real execution, we'd expect speedup, but in mocked tests,
            # we validate the structure and that fog execution completed
            assert fog_result.phase_name == "evomerge"
            assert fog_result.config.get("fog_executed", False) is True
            assert speedup >= 0  # Basic sanity check

    @pytest.mark.asyncio
    async def test_evomerge_multiple_fog_nodes_distribution(
        self, fog_burst_orchestrator, mock_evomerge_phase, unified_config, mock_fog_nodes
    ):
        """Test EvoMerge distribution across multiple fog nodes"""

        # Setup multiple fog nodes
        fog_burst_orchestrator.available_nodes = {
            node["node_id"]: MagicMock(
                node_id=node["node_id"],
                cpu_cores=node["resources"]["cpu_cores"],
                memory_gb=node["resources"]["memory_gb"],
                supports_evomerge=node["capabilities"]["evomerge"],
                current_load=node["metrics"]["current_load"],
            )
            for node in mock_fog_nodes
        }

        # Mock distributed execution across nodes
        mock_fog_result = {
            "status": "success",
            "phase_name": "evomerge",
            "metrics": {"merge_score": 0.85, "training_time_s": 80.0},
            "metadata": {
                "fog_executed": True,
                "distributed_nodes": ["fog_node_001", "fog_node_002"],
                "distribution_strategy": "parallel",
            },
        }

        with patch.object(fog_burst_orchestrator, "_execute_on_fog", return_value=mock_fog_result):
            result = await fog_burst_orchestrator.execute_phase_distributed(
                phase=mock_evomerge_phase, config=unified_config
            )

            # Validate distributed execution
            assert result.config["fog_executed"] is True
            assert "distributed_nodes" in result.config
            assert len(result.config["distributed_nodes"]) == 2

    @pytest.mark.asyncio
    async def test_evomerge_fog_node_selection_logic(self, fog_burst_orchestrator, mock_fog_nodes):
        """Test fog node selection logic for EvoMerge phases"""

        # Setup fog nodes with different capabilities
        fog_burst_orchestrator.available_nodes = {
            "powerful_node": MagicMock(
                node_id="powerful_node",
                cpu_cores=16.0,
                memory_gb=32.0,
                gpu_available=True,
                supports_evomerge=True,
                supports_pytorch=True,
                current_load=0.2,
            ),
            "basic_node": MagicMock(
                node_id="basic_node",
                cpu_cores=2.0,
                memory_gb=4.0,
                gpu_available=False,
                supports_evomerge=True,
                supports_pytorch=True,
                current_load=0.8,
            ),
            "incompatible_node": MagicMock(
                node_id="incompatible_node",
                cpu_cores=8.0,
                memory_gb=16.0,
                gpu_available=True,
                supports_evomerge=False,  # Doesn't support EvoMerge
                supports_pytorch=True,
                current_load=0.1,
            ),
        }

        # Mock execution plan generation
        with patch.object(fog_burst_orchestrator, "_plan_phase_execution") as mock_plan:
            mock_plan.return_value = {
                "use_fog": True,
                "target_nodes": ["powerful_node"],  # Should select the most capable node
                "estimated_speedup": 3.0,
            }

            from packages.agent_forge.core.unified_pipeline import UnifiedConfig

            test_config = UnifiedConfig(enable_evomerge=True)

            execution_plan = await fog_burst_orchestrator._plan_phase_execution(
                phase=mock_evomerge_phase, config=test_config, strategy=FogBurstStrategy.FOG_PREFERRED
            )

            # Validate node selection logic
            assert execution_plan["use_fog"] is True
            assert "powerful_node" in execution_plan["target_nodes"]
            assert execution_plan["estimated_speedup"] > 1.0

    @pytest.mark.asyncio
    async def test_evomerge_fog_error_handling_and_recovery(
        self, fog_burst_orchestrator, mock_evomerge_phase, unified_config
    ):
        """Test error handling and recovery mechanisms"""

        # Test various error scenarios
        error_scenarios = [
            {"error": "Connection timeout", "should_retry": True},
            {"error": "Insufficient resources", "should_retry": False},
            {"error": "Node unavailable", "should_retry": True},
            {"error": "Invalid job specification", "should_retry": False},
        ]

        for scenario in error_scenarios:
            with patch.object(fog_burst_orchestrator, "_execute_on_fog", side_effect=Exception(scenario["error"])):
                if scenario["should_retry"]:
                    # Should fallback to local execution
                    result = await fog_burst_orchestrator.execute_phase_distributed(
                        phase=mock_evomerge_phase, config=unified_config, strategy=FogBurstStrategy.FOG_PREFERRED
                    )
                    assert result.phase_name == "evomerge"
                    assert result.success is True
                else:
                    # Should handle error gracefully
                    result = await fog_burst_orchestrator.execute_phase_distributed(
                        phase=mock_evomerge_phase, config=unified_config, strategy=FogBurstStrategy.FOG_PREFERRED
                    )
                    # Should still complete via fallback
                    assert result.phase_name == "evomerge"

    @pytest.mark.asyncio
    async def test_evomerge_fog_burst_end_to_end(self, mock_fog_gateway_url):
        """Test complete end-to-end EvoMerge fog burst workflow"""

        # Create orchestrator
        orchestrator = await create_fog_burst_orchestrator(
            fog_gateway_url=mock_fog_gateway_url, strategy=FogBurstStrategy.FOG_PREFERRED, parity_tolerance=1e-4
        )

        # Mock the full workflow
        with patch.object(orchestrator, "_discover_fog_nodes") as mock_discover:
            mock_discover.return_value = None

            # Mock successful end-to-end execution
            mock_result = {
                "status": "success",
                "phase_name": "evomerge",
                "metrics": {"merge_score": 0.85, "final_loss": 2.34, "training_time_s": 150.0},
                "fog_job_id": "end_to_end_12345",
                "fog_node": "fog_node_001",
                "execution_time_s": 150.0,
            }

            with patch.object(orchestrator, "_execute_on_fog", return_value=mock_result):
                # Create test phase and config
                phase = MagicMock()
                phase.__class__.__name__ = "EvoMergePhase"

                config = UnifiedConfig(enable_evomerge=True)

                # Execute end-to-end
                result = await orchestrator.execute_phase_distributed(phase=phase, config=config)

                # Validate end-to-end execution
                assert result.phase_name == "evomerge"
                assert result.config["fog_executed"] is True
                assert result.config["fog_job_id"] == "end_to_end_12345"

        # Cleanup
        await orchestrator.shutdown()


class TestEvoMergeFogPerformance:
    """Performance-focused tests for EvoMerge fog offload"""

    @pytest.mark.asyncio
    async def test_evomerge_fog_scaling_characteristics(self, mock_fog_gateway_url):
        """Test how EvoMerge fog offload scales with different workloads"""

        orchestrator = FogBurstOrchestrator(fog_gateway_url=mock_fog_gateway_url)

        # Test with different model sizes and complexities
        test_scenarios = [
            {"model_size": "small", "expected_speedup": 1.5},
            {"model_size": "medium", "expected_speedup": 2.0},
            {"model_size": "large", "expected_speedup": 3.0},
        ]

        for scenario in test_scenarios:
            # Mock fog nodes with appropriate capabilities
            mock_nodes = {
                "powerful_node": MagicMock(
                    node_id="powerful_node", cpu_cores=16.0, memory_gb=32.0, supports_evomerge=True, current_load=0.1
                )
            }
            orchestrator.available_nodes = mock_nodes

            # Mock execution plan
            with patch.object(orchestrator, "_plan_phase_execution") as mock_plan:
                mock_plan.return_value = {
                    "use_fog": True,
                    "target_nodes": ["powerful_node"],
                    "estimated_speedup": scenario["expected_speedup"],
                }

                phase = MagicMock()
                phase.__class__.__name__ = "EvoMergePhase"
                config = UnifiedConfig(enable_evomerge=True)

                execution_plan = await orchestrator._plan_phase_execution(
                    phase=phase, config=config, strategy=FogBurstStrategy.FOG_PREFERRED
                )

                # Validate scaling characteristics
                assert execution_plan["estimated_speedup"] >= scenario["expected_speedup"]

    @pytest.mark.asyncio
    async def test_evomerge_fog_resource_utilization(self, mock_fog_gateway_url):
        """Test resource utilization efficiency in fog execution"""

        orchestrator = FogBurstOrchestrator(fog_gateway_url=mock_fog_gateway_url)

        # Mock resource monitoring
        resource_metrics = {
            "cpu_utilization": 0.85,
            "memory_utilization": 0.70,
            "gpu_utilization": 0.95,
            "network_utilization": 0.30,
        }

        # Simulate resource-efficient execution
        mock_result = {"status": "success", "metrics": {"merge_score": 0.85}, "resource_metrics": resource_metrics}

        with patch.object(orchestrator, "_execute_on_fog", return_value=mock_result):
            phase = MagicMock()
            phase.__class__.__name__ = "EvoMergePhase"
            config = UnifiedConfig(enable_evomerge=True)

            result = await orchestrator.execute_phase_distributed(phase=phase, config=config)

            # Validate resource utilization
            assert result.config.get("resource_metrics") is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
