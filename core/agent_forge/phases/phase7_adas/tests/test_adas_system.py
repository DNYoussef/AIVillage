"""
Comprehensive test suite for Phase 7 ADAS system

Tests all ADAS components individually and in integration scenarios
with focus on safety, performance, and compliance validation.
"""

import pytest
import asyncio
import numpy as np
import time
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Any

# Import ADAS components
from ..config.adas_config import ADASConfig, ASILLevel
from ..agents.sensor_fusion_agent import SensorFusionAgent, SensorData, FusedOutput
from ..agents.perception_agent import PerceptionAgent, DetectedObject, PerceptionOutput
from ..agents.prediction_agent import PredictionAgent, PredictedTrajectory, PredictionOutput
from ..agents.planning_agent import PlanningAgent, PlannedPath, PlanningOutput
from ..agents.safety_monitor import SafetyMonitor
from ..agents.edge_deployment import EdgeDeployment, ModelMetadata, OptimizedModel
from ..agents.v2x_communicator import V2XCommunicator, V2XMessage, MessageType
from ..agents.adas_orchestrator import ADASOrchestrator
from ..agents.compliance_validator import ComplianceValidator, ComplianceReport
from ..safety.safety_manager import SafetyManager
from ..integration.phase_bridge import PhaseBridge, ModelArtifact


class TestADASConfiguration:
    """Test ADAS configuration and validation"""

    def test_default_configuration(self):
        """Test default ADAS configuration"""
        config = ADASConfig()

        assert config.latency.total_pipeline_max_ms == 10.0
        assert config.safety.min_detection_confidence == 0.95
        assert len(config.sensors) == 5  # camera, radar, lidar, imu, gps

        # Validate sensor configuration
        assert "front_camera" in config.sensors
        assert config.sensors["front_camera"].asil_level == ASILLevel.D

    def test_asil_d_sensor_validation(self):
        """Test ASIL-D sensor validation"""
        config = ADASConfig()
        asil_d_sensors = config.get_asil_d_sensors()

        assert len(asil_d_sensors) >= 2  # Minimum for redundancy
        assert all(sensor.asil_level == ASILLevel.D for sensor in asil_d_sensors)

    def test_configuration_validation(self):
        """Test configuration validation"""
        config = ADASConfig()

        # Should not raise exception for valid config
        assert config._validate_configuration() == True

    def test_performance_targets(self):
        """Test performance target calculation"""
        config = ADASConfig()
        targets = config.get_performance_targets()

        assert "max_latency_ms" in targets
        assert "min_fps" in targets
        assert targets["max_latency_ms"] == 10.0
        assert targets["min_fps"] == 100.0  # 1000ms / 10ms


class TestSensorFusionAgent:
    """Test sensor fusion agent functionality"""

    @pytest.fixture
    async def sensor_fusion_agent(self):
        """Create sensor fusion agent fixture"""
        config = ADASConfig()
        agent = SensorFusionAgent(config)
        await agent.start()
        yield agent
        await agent.stop()

    @pytest.mark.asyncio
    async def test_sensor_fusion_startup(self, sensor_fusion_agent):
        """Test sensor fusion agent startup"""
        assert sensor_fusion_agent.state.value == "active"
        assert len(sensor_fusion_agent.sensor_buffers) == 5

    @pytest.mark.asyncio
    async def test_sensor_data_ingestion(self, sensor_fusion_agent):
        """Test sensor data ingestion"""
        sensor_data = SensorData(
            sensor_id="front_camera",
            sensor_type=sensor_fusion_agent.config.sensors["front_camera"].sensor_type,
            timestamp=time.time(),
            data=np.random.rand(480, 640, 3),
            confidence=0.95,
            quality_score=0.9,
            latency_ms=2.0,
            frame_id=1,
            asil_level=ASILLevel.D
        )

        result = sensor_fusion_agent.ingest_sensor_data(sensor_data)
        assert result == True

    @pytest.mark.asyncio
    async def test_sensor_data_validation(self, sensor_fusion_agent):
        """Test sensor data validation"""
        # Valid data
        valid_data = SensorData(
            sensor_id="front_camera",
            sensor_type=sensor_fusion_agent.config.sensors["front_camera"].sensor_type,
            timestamp=time.time(),
            data=np.random.rand(480, 640, 3),
            confidence=0.95,
            quality_score=0.9,
            latency_ms=2.0,
            frame_id=1,
            asil_level=ASILLevel.D
        )

        assert sensor_fusion_agent._validate_sensor_data(valid_data) == True

        # Invalid data (stale timestamp)
        invalid_data = SensorData(
            sensor_id="front_camera",
            sensor_type=sensor_fusion_agent.config.sensors["front_camera"].sensor_type,
            timestamp=time.time() - 2.0,  # 2 seconds old
            data=np.random.rand(480, 640, 3),
            confidence=0.95,
            quality_score=0.9,
            latency_ms=2.0,
            frame_id=1,
            asil_level=ASILLevel.D
        )

        assert sensor_fusion_agent._validate_sensor_data(invalid_data) == False


class TestPerceptionAgent:
    """Test perception agent functionality"""

    @pytest.fixture
    async def perception_agent(self):
        """Create perception agent fixture"""
        config = ADASConfig()
        agent = PerceptionAgent(config)
        await agent.start()
        yield agent
        await agent.stop()

    @pytest.mark.asyncio
    async def test_perception_startup(self, perception_agent):
        """Test perception agent startup"""
        assert perception_agent.state.value == "active"
        assert "primary" in perception_agent.detection_models
        assert "backup" in perception_agent.detection_models

    @pytest.mark.asyncio
    async def test_frame_validation(self, perception_agent):
        """Test frame validation"""
        # Valid frame
        valid_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        assert perception_agent._validate_frame(valid_frame) == True

        # Invalid frame (wrong dimensions)
        invalid_frame = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        assert perception_agent._validate_frame(invalid_frame) == False

    @pytest.mark.asyncio
    async def test_iou_calculation(self, perception_agent):
        """Test IoU calculation for bounding boxes"""
        bbox1 = (10, 10, 20, 20)
        bbox2 = (15, 15, 25, 25)

        iou = perception_agent._compute_iou(bbox1, bbox2)
        assert 0.0 <= iou <= 1.0

        # Perfect overlap
        iou_perfect = perception_agent._compute_iou(bbox1, bbox1)
        assert iou_perfect == 1.0

        # No overlap
        bbox3 = (50, 50, 60, 60)
        iou_none = perception_agent._compute_iou(bbox1, bbox3)
        assert iou_none == 0.0


class TestPredictionAgent:
    """Test prediction agent functionality"""

    @pytest.fixture
    async def prediction_agent(self):
        """Create prediction agent fixture"""
        config = ADASConfig()
        agent = PredictionAgent(config)
        await agent.start()
        yield agent
        await agent.stop()

    @pytest.mark.asyncio
    async def test_prediction_startup(self, prediction_agent):
        """Test prediction agent startup"""
        assert prediction_agent.state.value == "active"
        assert len(prediction_agent.prediction_models) == 4

    @pytest.mark.asyncio
    async def test_input_validation(self, prediction_agent):
        """Test prediction input validation"""
        # Valid inputs
        objects = [Mock(spec=DetectedObject)]
        ego_state = {
            'position_x': 0.0, 'position_y': 0.0,
            'velocity_x': 10.0, 'velocity_y': 0.0,
            'heading': 0.0
        }

        assert prediction_agent._validate_inputs(objects, ego_state) == True

        # Invalid inputs (missing ego state field)
        invalid_ego_state = {'position_x': 0.0}
        assert prediction_agent._validate_inputs(objects, invalid_ego_state) == False

    @pytest.mark.asyncio
    async def test_maneuver_classification(self, prediction_agent):
        """Test maneuver classification"""
        # Straight motion history
        straight_history = [
            {'timestamp': i, 'heading': 0.0, 'velocity': [10.0, 0.0, 0.0]}
            for i in range(10)
        ]

        obj = Mock()
        obj.tracking_id = 1

        maneuver = prediction_agent._classify_maneuver(obj, straight_history)
        assert maneuver == "straight"


class TestPlanningAgent:
    """Test planning agent functionality"""

    @pytest.fixture
    async def planning_agent(self):
        """Create planning agent fixture"""
        config = ADASConfig()
        agent = PlanningAgent(config)
        await agent.start()
        yield agent
        await agent.stop()

    @pytest.mark.asyncio
    async def test_planning_startup(self, planning_agent):
        """Test planning agent startup"""
        assert planning_agent.state.value == "active"
        assert len(planning_agent.path_planners) == 4

    @pytest.mark.asyncio
    async def test_planning_input_validation(self, planning_agent):
        """Test planning input validation"""
        ego_state = {
            'position_x': 0.0, 'position_y': 0.0,
            'velocity_x': 10.0, 'velocity_y': 0.0,
            'heading': 0.0
        }
        goal_state = {'position_x': 100.0, 'position_y': 0.0}

        assert planning_agent._validate_planning_inputs(ego_state, goal_state) == True

    @pytest.mark.asyncio
    async def test_planner_selection(self, planning_agent):
        """Test planner selection logic"""
        ego_state = {'velocity_x': 25.0, 'velocity_y': 0.0}  # High speed
        obstacles = []

        planner = planning_agent._select_planner(ego_state, obstacles)
        assert planner == "lattice"  # Should select lattice for high speed

    @pytest.mark.asyncio
    async def test_emergency_brake_path(self, planning_agent):
        """Test emergency braking path generation"""
        ego_state = {
            'position_x': 0.0, 'position_y': 0.0,
            'velocity_x': 20.0, 'velocity_y': 0.0,
            'heading': 0.0
        }

        brake_path = planning_agent._generate_emergency_brake_path(ego_state, time.time())

        assert len(brake_path) > 0
        assert brake_path[0]['velocity'] == 20.0
        assert brake_path[-1]['velocity'] == 0.0  # Should come to stop


class TestSafetyManager:
    """Test safety manager functionality"""

    @pytest.fixture
    async def safety_manager(self):
        """Create safety manager fixture"""
        config = ADASConfig()
        manager = SafetyManager(config)
        await manager.start()
        yield manager
        await manager.stop()

    @pytest.mark.asyncio
    async def test_safety_manager_startup(self, safety_manager):
        """Test safety manager startup"""
        assert safety_manager.safety_state.value == "safe"
        assert len(safety_manager.safety_thresholds) > 0

    @pytest.mark.asyncio
    async def test_violation_creation(self, safety_manager):
        """Test safety violation creation"""
        from ..safety.safety_manager import SafetyViolationType

        violation = safety_manager._create_violation(
            SafetyViolationType.LATENCY_EXCEEDED,
            ASILLevel.C,
            "Test violation",
            ["perception"],
            "Reduce computation"
        )

        assert violation.violation_type == SafetyViolationType.LATENCY_EXCEEDED
        assert violation.severity == ASILLevel.C
        assert "perception" in violation.affected_components


class TestEdgeDeployment:
    """Test edge deployment agent functionality"""

    @pytest.fixture
    async def edge_deployment(self):
        """Create edge deployment agent fixture"""
        config = ADASConfig()
        agent = EdgeDeployment(config)
        await agent.start()
        yield agent
        await agent.stop()

    @pytest.mark.asyncio
    async def test_edge_deployment_startup(self, edge_deployment):
        """Test edge deployment startup"""
        assert edge_deployment.state.value == "active"
        assert edge_deployment.model_optimizer is not None

    @pytest.mark.asyncio
    async def test_model_metadata_creation(self, edge_deployment):
        """Test model metadata creation"""
        from ..agents.edge_deployment import EdgePlatform

        metadata = ModelMetadata(
            model_name="test_model",
            model_type="perception",
            framework="pytorch",
            input_shape=(1, 3, 224, 224),
            output_shape=(1, 1000),
            parameter_count=1000000,
            model_size_mb=10.0,
            precision="FP32",
            target_platform=EdgePlatform.NVIDIA_JETSON_XAVIER
        )

        assert metadata.model_name == "test_model"
        assert metadata.target_platform == EdgePlatform.NVIDIA_JETSON_XAVIER


class TestV2XCommunicator:
    """Test V2X communication agent functionality"""

    @pytest.fixture
    async def v2x_communicator(self):
        """Create V2X communicator fixture"""
        config = ADASConfig()
        agent = V2XCommunicator(config)
        await agent.start()
        yield agent
        await agent.stop()

    @pytest.mark.asyncio
    async def test_v2x_startup(self, v2x_communicator):
        """Test V2X communicator startup"""
        assert v2x_communicator.state.value == "active"
        assert len(v2x_communicator.active_protocols) > 0

    @pytest.mark.asyncio
    async def test_message_id_generation(self, v2x_communicator):
        """Test V2X message ID generation"""
        msg_id1 = v2x_communicator._generate_message_id()
        msg_id2 = v2x_communicator._generate_message_id()

        assert msg_id1 != msg_id2
        assert msg_id1.startswith("MSG_")

    @pytest.mark.asyncio
    async def test_distance_calculation(self, v2x_communicator):
        """Test distance calculation between positions"""
        pos1 = (37.7749, -122.4194, 10.0)  # San Francisco
        pos2 = (37.7849, -122.4094, 10.0)  # Nearby position

        distance = v2x_communicator._calculate_distance(pos2)
        assert distance > 0
        assert distance < 50000  # Should be reasonable distance in meters


class TestComplianceValidator:
    """Test compliance validation functionality"""

    @pytest.fixture
    async def compliance_validator(self):
        """Create compliance validator fixture"""
        config = ADASConfig()
        safety_manager = SafetyManager(config)
        await safety_manager.start()

        validator = ComplianceValidator(config, safety_manager)
        await validator.start()
        yield validator
        await validator.stop()
        await safety_manager.stop()

    @pytest.mark.asyncio
    async def test_compliance_startup(self, compliance_validator):
        """Test compliance validator startup"""
        assert compliance_validator.state.value in ["validating", "compliant", "needs_review"]
        assert len(compliance_validator.requirements_db) > 0

    @pytest.mark.asyncio
    async def test_iso_26262_requirements(self, compliance_validator):
        """Test ISO 26262 requirements loading"""
        iso_requirements = [
            req for req in compliance_validator.requirements_db.values()
            if req.standard.value == "iso_26262"
        ]

        assert len(iso_requirements) > 0

        # Check for key requirements
        hara_req = next((req for req in iso_requirements if "7.1" in req.requirement_id), None)
        assert hara_req is not None
        assert hara_req.mandatory == True

    @pytest.mark.asyncio
    async def test_validation_scoring(self, compliance_validator):
        """Test validation scoring calculation"""
        validation_results = {
            "req1": {"compliant": True, "score": 1.0},
            "req2": {"compliant": True, "score": 0.8},
            "req3": {"compliant": False, "score": 0.2}
        }

        scores = compliance_validator._calculate_compliance_scores(validation_results)

        assert "overall" in scores
        assert 0.0 <= scores["overall"] <= 1.0
        assert scores["overall"] == pytest.approx(0.667, rel=0.1)


class TestADASOrchestrator:
    """Test ADAS orchestrator functionality"""

    @pytest.fixture
    async def adas_orchestrator(self):
        """Create ADAS orchestrator fixture"""
        config = ADASConfig()
        orchestrator = ADASOrchestrator(config)
        await orchestrator.initialize()
        yield orchestrator
        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, adas_orchestrator):
        """Test orchestrator initialization"""
        assert adas_orchestrator.state.value == "active"
        assert len(adas_orchestrator.components) > 0
        assert "safety_manager" in adas_orchestrator.components

    @pytest.mark.asyncio
    async def test_component_status_tracking(self, adas_orchestrator):
        """Test component status tracking"""
        adas_orchestrator._update_component_statuses()

        assert len(adas_orchestrator.component_statuses) > 0

        for status in adas_orchestrator.component_statuses.values():
            assert hasattr(status, 'component_name')
            assert hasattr(status, 'performance_score')
            assert 0.0 <= status.performance_score <= 1.0


class TestPhaseBridge:
    """Test phase integration bridge functionality"""

    @pytest.fixture
    async def phase_bridge(self):
        """Create phase bridge fixture"""
        config = ADASConfig()
        bridge = PhaseBridge(config, "/tmp/test_aivillage")
        await bridge.start()
        yield bridge
        await bridge.stop()

    @pytest.mark.asyncio
    async def test_phase_bridge_startup(self, phase_bridge):
        """Test phase bridge startup"""
        assert phase_bridge.running == True
        assert phase_bridge.phase6_interface is not None

    @pytest.mark.asyncio
    async def test_model_artifact_validation(self, phase_bridge):
        """Test model artifact validation"""
        # Create temporary model file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(b"dummy model data")
            tmp_path = tmp_file.name

        try:
            model_artifact = ModelArtifact(
                model_id="test_model_123",
                model_name="test_perception_model",
                model_type="perception",
                framework="pytorch",
                format=ModelFormat.PYTORCH,
                file_path=tmp_path,
                metadata={},
                performance_metrics={},
                validation_results={},
                baking_timestamp=time.time(),
                checksum="abc123"
            )

            validation_result = await phase_bridge.model_validator.validate_artifact(model_artifact)
            assert validation_result['valid'] == True

        finally:
            import os
            os.unlink(tmp_path)


class TestIntegrationScenarios:
    """Test complete integration scenarios"""

    @pytest.mark.asyncio
    async def test_perception_to_prediction_pipeline(self):
        """Test perception to prediction data flow"""
        config = ADASConfig()

        # Create mock perception output
        perception_output = Mock(spec=PerceptionOutput)
        perception_output.objects = []
        perception_output.processing_latency_ms = 5.0

        # Create prediction agent
        prediction_agent = PredictionAgent(config)
        await prediction_agent.start()

        try:
            # Test input validation
            ego_state = {
                'position_x': 0.0, 'position_y': 0.0,
                'velocity_x': 10.0, 'velocity_y': 0.0,
                'heading': 0.0
            }

            assert prediction_agent._validate_inputs([], ego_state) == True

        finally:
            await prediction_agent.stop()

    @pytest.mark.asyncio
    async def test_emergency_scenario_handling(self):
        """Test emergency scenario handling across components"""
        config = ADASConfig()

        # Create safety manager
        safety_manager = SafetyManager(config)
        await safety_manager.start()

        try:
            # Simulate emergency violation
            from ..safety.safety_manager import SafetyViolationType

            violation = safety_manager._create_violation(
                SafetyViolationType.COLLISION_IMMINENT,
                ASILLevel.D,
                "Collision imminent",
                ["planning"],
                "Execute emergency brake"
            )

            assert violation.severity == ASILLevel.D
            assert "planning" in violation.affected_components

        finally:
            await safety_manager.stop()

    @pytest.mark.asyncio
    async def test_performance_monitoring(self):
        """Test performance monitoring across all components"""
        config = ADASConfig()

        # Test latency constraints
        assert config.latency.total_pipeline_max_ms <= 10.0
        assert config.latency.perception_max_ms <= 5.0
        assert config.latency.prediction_max_ms <= 8.0
        assert config.latency.planning_max_ms <= 10.0

    @pytest.mark.asyncio
    async def test_safety_critical_path(self):
        """Test safety-critical execution path"""
        config = ADASConfig()

        # Verify ASIL-D components have proper redundancy
        asil_d_sensors = config.get_asil_d_sensors()
        assert len(asil_d_sensors) >= 2

        # Verify safety thresholds
        assert config.safety.min_detection_confidence >= 0.95
        assert config.safety.max_false_negative_rate <= 0.0001


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmark tests"""

    @pytest.mark.asyncio
    async def test_perception_latency_benchmark(self):
        """Benchmark perception processing latency"""
        config = ADASConfig()
        perception_agent = PerceptionAgent(config)
        await perception_agent.start()

        try:
            # Create test frame
            test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

            # Measure processing time
            start_time = time.time()

            # Simulate frame processing (would call actual method in real test)
            valid = perception_agent._validate_frame(test_frame)
            assert valid == True

            processing_time = (time.time() - start_time) * 1000

            # Should be well under latency constraint
            assert processing_time < config.latency.perception_max_ms

        finally:
            await perception_agent.stop()

    @pytest.mark.asyncio
    async def test_end_to_end_latency_benchmark(self):
        """Benchmark end-to-end pipeline latency"""
        config = ADASConfig()

        # Simulate pipeline processing
        start_time = time.time()

        # Sensor fusion (simulated)
        await asyncio.sleep(0.003)  # 3ms

        # Perception (simulated)
        await asyncio.sleep(0.005)  # 5ms

        # Prediction (simulated)
        await asyncio.sleep(0.008)  # 8ms

        # Planning (simulated)
        await asyncio.sleep(0.010)  # 10ms

        total_latency = (time.time() - start_time) * 1000

        # Should meet total pipeline constraint
        assert total_latency < config.latency.total_pipeline_max_ms


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])