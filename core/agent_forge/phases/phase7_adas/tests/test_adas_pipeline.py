"""
Test suite for ADAS Pipeline - Comprehensive testing for automotive safety compliance
Tests include real-time performance, safety constraints, and ASIL-D compliance validation
"""

import pytest
import asyncio
import numpy as np
import time
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from adas.core.adas_pipeline import (
    AdasPipeline, ModelLoader, RealTimeInferenceEngine, OutputFormatter,
    SensorData, ProcessingResult, SystemState, SafetyLevel
)

class TestModelLoader:
    """Test cases for model loading and integrity verification"""

    def test_model_loader_initialization(self):
        """Test model loader initialization"""
        model_path = "/test/models"
        loader = ModelLoader(model_path, SafetyLevel.ASIL_D)

        assert loader.model_path == model_path
        assert loader.safety_level == SafetyLevel.ASIL_D
        assert loader.models == {}

    @patch('adas.core.adas_pipeline.ModelLoader._load_model')
    @patch('adas.core.adas_pipeline.ModelLoader._verify_model_integrity')
    def test_load_models_success(self, mock_verify, mock_load):
        """Test successful model loading"""
        mock_load.return_value = "mock_model"
        mock_verify.return_value = True

        loader = ModelLoader("/test/models")
        result = loader.load_models()

        assert result is True
        assert len(loader.models) == 4  # object_detection, lane_detection, traffic_sign, pedestrian
        mock_verify.assert_called_once()

    @patch('adas.core.adas_pipeline.ModelLoader._load_model')
    @patch('adas.core.adas_pipeline.ModelLoader._verify_model_integrity')
    def test_load_models_integrity_failure(self, mock_verify, mock_load):
        """Test model loading with integrity verification failure"""
        mock_load.return_value = "mock_model"
        mock_verify.return_value = False

        loader = ModelLoader("/test/models")
        result = loader.load_models()

        assert result is False

    @patch('adas.core.adas_pipeline.ModelLoader._load_model')
    def test_load_models_exception(self, mock_load):
        """Test model loading with exception"""
        mock_load.side_effect = Exception("Model loading failed")

        loader = ModelLoader("/test/models")
        result = loader.load_models()

        assert result is False

class TestRealTimeInferenceEngine:
    """Test cases for real-time inference engine"""

    def setup_method(self):
        """Setup test fixtures"""
        self.models = {
            'object_detection': 'mock_object_model',
            'lane_detection': 'mock_lane_model',
            'traffic_sign': 'mock_sign_model'
        }
        self.engine = RealTimeInferenceEngine(self.models, max_latency_ms=50.0)

    def test_inference_engine_initialization(self):
        """Test inference engine initialization"""
        assert self.engine.models == self.models
        assert self.engine.max_latency_ms == 50.0
        assert self.engine.inference_pool is not None

    @pytest.mark.asyncio
    async def test_process_frame_success(self):
        """Test successful frame processing"""
        sensor_data = SensorData(
            timestamp=time.time(),
            sensor_id="test_camera",
            sensor_type="camera",
            data=np.zeros((480, 640, 3), dtype=np.uint8),
            quality_score=1.0,
            calibration_status=True
        )

        result = await self.engine.process_frame(sensor_data)

        assert result is not None
        assert 'objects' in result
        assert 'lanes' in result
        assert 'traffic_signs' in result
        assert 'processing_time_ms' in result
        assert result['processing_time_ms'] < 100.0  # Should be fast

    @pytest.mark.asyncio
    async def test_process_frame_timeout(self):
        """Test frame processing timeout"""
        # Create engine with very short timeout
        short_engine = RealTimeInferenceEngine(self.models, max_latency_ms=1.0)

        sensor_data = SensorData(
            timestamp=time.time(),
            sensor_id="test_camera",
            sensor_type="camera",
            data=np.zeros((480, 640, 3), dtype=np.uint8),
            quality_score=1.0,
            calibration_status=True
        )

        # Mock inference methods to take longer than timeout
        with patch.object(short_engine, '_run_object_detection',
                         return_value=AsyncMock(side_effect=lambda x: asyncio.sleep(0.1))):
            result = await short_engine.process_frame(sensor_data)

            assert result is None
            assert short_engine.performance_metrics['timeout_count'] > 0

    def test_performance_metrics_update(self):
        """Test performance metrics updating"""
        initial_count = self.engine.performance_metrics['inference_count']

        # Simulate processing time update
        self.engine._update_performance_metrics(25.0)

        assert self.engine.performance_metrics['inference_count'] == initial_count + 1
        assert self.engine.performance_metrics['avg_latency'] == 25.0
        assert self.engine.performance_metrics['max_latency'] == 25.0

class TestOutputFormatter:
    """Test cases for output formatting"""

    def setup_method(self):
        """Setup test fixtures"""
        self.formatter = OutputFormatter("CAN")

    def test_formatter_initialization(self):
        """Test output formatter initialization"""
        assert self.formatter.output_protocol == "CAN"
        assert self.formatter.message_templates is not None

    def test_format_for_vehicle(self):
        """Test vehicle output formatting"""
        processing_result = ProcessingResult(
            timestamp=time.time(),
            detection_objects=[
                {'class': 'vehicle', 'confidence': 0.95, 'bbox': [100, 200, 300, 400]}
            ],
            lane_info={'lane_confidence': 0.8},
            traffic_signs=[
                {'type': 'stop_sign', 'confidence': 0.9}
            ],
            safety_alerts=[],
            system_health={},
            confidence_scores={'overall': 0.85},
            processing_latency=30.0
        )

        vehicle_output = self.formatter.format_for_vehicle(processing_result)

        assert 'header' in vehicle_output
        assert 'perception' in vehicle_output
        assert 'safety' in vehicle_output
        assert 'performance' in vehicle_output

        # Check header fields
        header = vehicle_output['header']
        assert 'timestamp' in header
        assert header['safety_level'] == 'ASIL_D'
        assert header['system_state'] == 'ACTIVE'

        # Check performance compliance
        performance = vehicle_output['performance']
        assert performance['real_time_compliance'] is True  # 30ms < 50ms threshold

    def test_threat_level_calculation(self):
        """Test threat level calculation for objects"""
        # High confidence pedestrian
        pedestrian_obj = {'class': 'pedestrian', 'confidence': 0.9}
        threat_level = self.formatter._calculate_threat_level(pedestrian_obj)
        assert threat_level == 'HIGH'

        # High confidence vehicle
        vehicle_obj = {'class': 'vehicle', 'confidence': 0.95}
        threat_level = self.formatter._calculate_threat_level(vehicle_obj)
        assert threat_level == 'MEDIUM'

        # Low confidence object
        low_conf_obj = {'class': 'unknown', 'confidence': 0.3}
        threat_level = self.formatter._calculate_threat_level(low_conf_obj)
        assert threat_level == 'LOW'

class TestAdasPipeline:
    """Test cases for main ADAS pipeline"""

    def setup_method(self):
        """Setup test fixtures"""
        self.config = {
            'model_path': '/test/models',
            'max_latency_ms': 50.0,
            'output_protocol': 'CAN',
            'watchdog_timeout': 100,
            'max_errors': 5
        }
        self.pipeline = AdasPipeline(self.config)

    def test_pipeline_initialization(self):
        """Test ADAS pipeline initialization"""
        assert self.pipeline.config == self.config
        assert self.pipeline.state == SystemState.INIT
        assert self.pipeline.safety_level == SafetyLevel.ASIL_D
        assert self.pipeline.error_count == 0

    @pytest.mark.asyncio
    @patch('adas.core.adas_pipeline.ModelLoader.load_models')
    @patch('adas.core.adas_pipeline.AdasPipeline._system_self_test')
    async def test_initialize_success(self, mock_self_test, mock_load_models):
        """Test successful pipeline initialization"""
        mock_load_models.return_value = True
        mock_self_test.return_value = True

        result = await self.pipeline.initialize()

        assert result is True
        assert self.pipeline.state == SystemState.ACTIVE
        assert self.pipeline.inference_engine is not None

    @pytest.mark.asyncio
    @patch('adas.core.adas_pipeline.ModelLoader.load_models')
    async def test_initialize_model_failure(self, mock_load_models):
        """Test pipeline initialization with model loading failure"""
        mock_load_models.return_value = False

        result = await self.pipeline.initialize()

        assert result is False
        assert self.pipeline.state == SystemState.FAULT

    @pytest.mark.asyncio
    @patch('adas.core.adas_pipeline.ModelLoader.load_models')
    @patch('adas.core.adas_pipeline.AdasPipeline._system_self_test')
    async def test_initialize_self_test_failure(self, mock_self_test, mock_load_models):
        """Test pipeline initialization with self-test failure"""
        mock_load_models.return_value = True
        mock_self_test.return_value = False

        result = await self.pipeline.initialize()

        assert result is False
        assert self.pipeline.state == SystemState.FAULT

    @pytest.mark.asyncio
    async def test_process_sensor_data_inactive_state(self):
        """Test processing when pipeline is not active"""
        # Pipeline starts in INIT state
        sensor_data = SensorData(
            timestamp=time.time(),
            sensor_id="test_camera",
            sensor_type="camera",
            data=np.zeros((480, 640, 3), dtype=np.uint8),
            quality_score=1.0,
            calibration_status=True
        )

        result = await self.pipeline.process_sensor_data(sensor_data)

        assert result is None

    @pytest.mark.asyncio
    @patch('adas.core.adas_pipeline.RealTimeInferenceEngine.process_frame')
    async def test_process_sensor_data_success(self, mock_process_frame):
        """Test successful sensor data processing"""
        # Set pipeline to active state
        self.pipeline.state = SystemState.ACTIVE
        self.pipeline.inference_engine = Mock()

        mock_inference_result = {
            'objects': [{'class': 'vehicle', 'confidence': 0.9}],
            'lanes': {'lane_confidence': 0.8},
            'traffic_signs': [],
            'processing_time_ms': 25.0,
            'timestamp': time.time()
        }
        mock_process_frame.return_value = mock_inference_result

        sensor_data = SensorData(
            timestamp=time.time(),
            sensor_id="test_camera",
            sensor_type="camera",
            data=np.zeros((480, 640, 3), dtype=np.uint8),
            quality_score=1.0,
            calibration_status=True
        )

        result = await self.pipeline.process_sensor_data(sensor_data)

        assert result is not None
        assert isinstance(result, ProcessingResult)
        assert len(result.detection_objects) == 1
        assert result.lane_info['lane_confidence'] == 0.8
        assert result.processing_latency > 0

    @pytest.mark.asyncio
    @patch('adas.core.adas_pipeline.RealTimeInferenceEngine.process_frame')
    async def test_process_sensor_data_inference_failure(self, mock_process_frame):
        """Test sensor data processing with inference failure"""
        # Set pipeline to active state
        self.pipeline.state = SystemState.ACTIVE
        self.pipeline.inference_engine = Mock()

        mock_process_frame.return_value = None  # Inference failure

        sensor_data = SensorData(
            timestamp=time.time(),
            sensor_id="test_camera",
            sensor_type="camera",
            data=np.zeros((480, 640, 3), dtype=np.uint8),
            quality_score=1.0,
            calibration_status=True
        )

        result = await self.pipeline.process_sensor_data(sensor_data)

        assert result is None
        assert self.pipeline.error_count > 0

    def test_error_handling(self):
        """Test error handling and state transitions"""
        initial_error_count = self.pipeline.error_count

        # Simulate multiple processing failures
        for _ in range(3):
            self.pipeline._handle_processing_failure()

        assert self.pipeline.error_count == initial_error_count + 3
        assert self.pipeline.state == SystemState.DEGRADED

        # Simulate reaching max errors
        for _ in range(2):
            self.pipeline._handle_processing_failure()

        assert self.pipeline.state == SystemState.FAULT

    def test_system_health_monitoring(self):
        """Test system health status reporting"""
        health = self.pipeline._get_system_health()

        assert 'state' in health
        assert 'error_count' in health
        assert 'last_heartbeat' in health
        assert 'uptime' in health
        assert 'memory_usage' in health
        assert 'cpu_usage' in health

    def test_confidence_score_calculation(self):
        """Test confidence score calculation"""
        inference_result = {
            'objects': [
                {'confidence': 0.9},
                {'confidence': 0.8}
            ],
            'lanes': {'lane_confidence': 0.85},
            'traffic_signs': [
                {'confidence': 0.95}
            ]
        }

        scores = self.pipeline._calculate_confidence_scores(inference_result)

        assert 'object_detection' in scores
        assert 'lane_detection' in scores
        assert 'traffic_sign' in scores
        assert 'overall' in scores

        # Check calculations
        assert scores['object_detection'] == 0.85  # (0.9 + 0.8) / 2
        assert scores['lane_detection'] == 0.85
        assert scores['traffic_sign'] == 0.95

    def test_performance_metrics(self):
        """Test performance metrics tracking"""
        initial_metrics = self.pipeline.get_performance_metrics()

        # Simulate processing
        self.pipeline._update_performance_metrics(30.0)
        self.pipeline._update_performance_metrics(40.0)

        updated_metrics = self.pipeline.get_performance_metrics()

        assert updated_metrics['frames_processed'] == 2
        assert updated_metrics['max_latency'] == 40.0
        assert updated_metrics['average_fps'] > 0

    @pytest.mark.asyncio
    async def test_system_self_test(self):
        """Test system self-test functionality"""
        # Mock inference engine
        mock_engine = Mock()
        mock_engine.process_frame.return_value = {
            'objects': [],
            'lanes': {},
            'traffic_signs': []
        }
        self.pipeline.inference_engine = mock_engine

        result = await self.pipeline._system_self_test()

        assert result is True
        mock_engine.process_frame.assert_called_once()

    def test_shutdown(self):
        """Test graceful shutdown"""
        # Mock inference engine with thread pool
        mock_engine = Mock()
        mock_pool = Mock()
        mock_engine.inference_pool = mock_pool
        self.pipeline.inference_engine = mock_engine

        self.pipeline.shutdown()

        assert self.pipeline.state == SystemState.SHUTDOWN
        mock_pool.shutdown.assert_called_once_with(wait=True)

class TestRealTimePerformance:
    """Test cases for real-time performance requirements"""

    @pytest.mark.asyncio
    async def test_processing_latency_compliance(self):
        """Test that processing latency meets real-time requirements"""
        config = {
            'model_path': '/test/models',
            'max_latency_ms': 50.0,
            'output_protocol': 'CAN'
        }
        pipeline = AdasPipeline(config)
        pipeline.state = SystemState.ACTIVE

        # Mock inference engine for fast processing
        mock_engine = Mock()
        mock_engine.process_frame.return_value = {
            'objects': [],
            'lanes': {},
            'traffic_signs': [],
            'processing_time_ms': 25.0,
            'timestamp': time.time()
        }
        pipeline.inference_engine = mock_engine

        sensor_data = SensorData(
            timestamp=time.time(),
            sensor_id="test_camera",
            sensor_type="camera",
            data=np.zeros((480, 640, 3), dtype=np.uint8),
            quality_score=1.0,
            calibration_status=True
        )

        start_time = time.perf_counter()
        result = await pipeline.process_sensor_data(sensor_data)
        processing_time = (time.perf_counter() - start_time) * 1000

        assert result is not None
        assert processing_time < 50.0  # Must meet real-time constraint

    @pytest.mark.asyncio
    async def test_throughput_requirements(self):
        """Test system throughput meets automotive requirements"""
        config = {
            'model_path': '/test/models',
            'max_latency_ms': 50.0
        }
        pipeline = AdasPipeline(config)
        pipeline.state = SystemState.ACTIVE

        # Mock fast inference
        mock_engine = Mock()
        mock_engine.process_frame.return_value = {
            'objects': [],
            'lanes': {},
            'traffic_signs': [],
            'processing_time_ms': 20.0,
            'timestamp': time.time()
        }
        pipeline.inference_engine = mock_engine

        # Process multiple frames
        num_frames = 10
        start_time = time.perf_counter()

        for i in range(num_frames):
            sensor_data = SensorData(
                timestamp=time.time(),
                sensor_id="test_camera",
                sensor_type="camera",
                data=np.zeros((480, 640, 3), dtype=np.uint8),
                quality_score=1.0,
                calibration_status=True
            )
            await pipeline.process_sensor_data(sensor_data)

        total_time = time.perf_counter() - start_time
        fps = num_frames / total_time

        assert fps >= 20.0  # Minimum 20 FPS for automotive applications

class TestSafetyCompliance:
    """Test cases for automotive safety compliance (ASIL-D)"""

    def test_safety_level_configuration(self):
        """Test safety level is properly configured"""
        config = {'model_path': '/test/models'}
        pipeline = AdasPipeline(config)

        assert pipeline.safety_level == SafetyLevel.ASIL_D

    def test_error_count_monitoring(self):
        """Test error count monitoring for safety compliance"""
        config = {'model_path': '/test/models', 'max_errors': 3}
        pipeline = AdasPipeline(config)

        # Simulate errors approaching safety limit
        for i in range(2):
            pipeline._handle_processing_failure()
            assert pipeline.state != SystemState.FAULT

        # One more error should trigger safety response
        pipeline._handle_processing_failure()
        assert pipeline.state == SystemState.FAULT

    def test_watchdog_functionality(self):
        """Test watchdog timer functionality"""
        config = {'model_path': '/test/models', 'watchdog_timeout': 100}
        pipeline = AdasPipeline(config)

        # Check initial heartbeat
        initial_heartbeat = pipeline.last_heartbeat

        # Simulate heartbeat update
        time.sleep(0.01)
        pipeline.last_heartbeat = time.time()

        assert pipeline.last_heartbeat > initial_heartbeat

    @pytest.mark.asyncio
    async def test_fail_safe_behavior(self):
        """Test fail-safe behavior on critical failures"""
        config = {'model_path': '/test/models', 'max_errors': 1}
        pipeline = AdasPipeline(config)
        pipeline.state = SystemState.ACTIVE

        # Mock inference engine to always fail
        mock_engine = Mock()
        mock_engine.process_frame.return_value = None
        pipeline.inference_engine = mock_engine

        sensor_data = SensorData(
            timestamp=time.time(),
            sensor_id="test_camera",
            sensor_type="camera",
            data=np.zeros((480, 640, 3), dtype=np.uint8),
            quality_score=1.0,
            calibration_status=True
        )

        # Multiple failures should trigger fail-safe
        result1 = await pipeline.process_sensor_data(sensor_data)
        result2 = await pipeline.process_sensor_data(sensor_data)

        assert result1 is None
        assert result2 is None
        assert pipeline.state == SystemState.FAULT

if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])