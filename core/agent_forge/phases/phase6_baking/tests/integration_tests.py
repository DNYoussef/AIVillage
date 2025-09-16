"""
Comprehensive Integration Tests for Phase 6 Baking System

This test suite provides comprehensive validation of all Phase 6 integration
components, ensuring system-wide functionality and reliability.
"""

import unittest
import json
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.append('src')

from phase6.integration.phase5_connector import Phase5Connector, create_phase5_connector
from phase6.integration.phase7_preparer import Phase7Preparer, create_phase7_preparer
from phase6.integration.pipeline_validator import PipelineValidator, create_pipeline_validator
from phase6.integration.state_manager import StateManager, create_state_manager, Phase, StateStatus
from phase6.integration.quality_coordinator import QualityCoordinator, create_quality_coordinator
from phase6.integration.working_validator import WorkingValidator, create_working_validator

class TestPhase5Connector(unittest.TestCase):
    """Test cases for Phase 5 integration connector"""

    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'phase5_model_dir': self.temp_dir + '/phase5',
            'supported_architectures': ['ResNet', 'VGG'],
            'min_accuracy': 0.8,
            'min_dataset_size': 1000
        }
        self.connector = Phase5Connector(self.config)

        # Create test model directory structure
        self.test_model_dir = Path(self.temp_dir) / 'phase5' / 'test_model'
        self.test_model_dir.mkdir(parents=True, exist_ok=True)

        # Create test metadata
        test_metadata = {
            'architecture': 'ResNet',
            'epochs': 100,
            'final_accuracy': 0.95,
            'validation_loss': 0.05,
            'optimizer_config': {'lr': 0.001},
            'training_time': 3600.0,
            'dataset_size': 5000,
            'hyperparameters': {'batch_size': 32},
            'checkpoint_path': 'checkpoint.pth'
        }

        with open(self.test_model_dir / 'training_metadata.json', 'w') as f:
            json.dump(test_metadata, f)

        # Create dummy model file
        (self.test_model_dir / 'model.pth').touch()

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_discover_trained_models(self):
        """Test model discovery functionality"""
        models = self.connector.discover_trained_models()
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]['model_id'], 'test_model')

    def test_load_training_metadata(self):
        """Test metadata loading"""
        metadata = self.connector.load_training_metadata(str(self.test_model_dir))
        self.assertEqual(metadata.model_architecture, 'ResNet')
        self.assertEqual(metadata.final_accuracy, 0.95)
        self.assertEqual(metadata.training_epochs, 100)

    def test_validate_model_compatibility(self):
        """Test model compatibility validation"""
        compatible, score, validation_results = self.connector.validate_model_compatibility(
            str(self.test_model_dir)
        )
        self.assertTrue(compatible)
        self.assertGreaterEqual(score, 0.8)
        self.assertTrue(validation_results['compatible'])

    def test_transfer_model(self):
        """Test model transfer functionality"""
        target_path = Path(self.temp_dir) / 'target'
        result = self.connector.transfer_model(str(self.test_model_dir), str(target_path))

        self.assertTrue(result.success)
        self.assertEqual(result.metadata.model_architecture, 'ResNet')
        self.assertTrue((target_path / 'model.pth').exists())
        self.assertTrue((target_path / 'phase6_integration.json').exists())

    def test_get_best_model(self):
        """Test best model selection"""
        best_model = self.connector.get_best_model('accuracy')
        self.assertIsNotNone(best_model)
        self.assertEqual(best_model['model_id'], 'test_model')

    def test_validate_integration_pipeline(self):
        """Test integration pipeline validation"""
        validation_results = self.connector.validate_integration_pipeline()
        self.assertEqual(validation_results['phase5_models_found'], 1)
        self.assertEqual(validation_results['compatible_models'], 1)

class TestPhase7Preparer(unittest.TestCase):
    """Test cases for Phase 7 preparation module"""

    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'phase6_output_dir': self.temp_dir + '/phase6',
            'adas_export_dir': self.temp_dir + '/adas',
            'max_inference_time_ms': 50.0,
            'min_accuracy': 0.95,
            'target_hardware': 'NVIDIA_Xavier'
        }
        self.preparer = Phase7Preparer(self.config)

        # Create test baked model
        self.test_model_dir = Path(self.temp_dir) / 'phase6' / 'baked_model'
        self.test_model_dir.mkdir(parents=True, exist_ok=True)

        test_baking_metadata = {
            'final_accuracy': 0.96,
            'inference_time_ms': 45.0,
            'model_size_mb': 85.0,
            'validation_passed': True,
            'quality_gates_passed': True,
            'performance_verified': True
        }

        with open(self.test_model_dir / 'baking_metadata.json', 'w') as f:
            json.dump(test_baking_metadata, f)

        (self.test_model_dir / 'optimized_model.pth').touch()

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_discover_baked_models(self):
        """Test baked model discovery"""
        models = self.preparer.discover_baked_models()
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]['model_id'], 'baked_model')

    def test_assess_adas_readiness(self):
        """Test ADAS readiness assessment"""
        readiness_report = self.preparer.assess_adas_readiness(str(self.test_model_dir))
        self.assertTrue(readiness_report.ready_for_deployment)
        self.assertGreaterEqual(readiness_report.performance_metrics['accuracy'], 0.95)

    def test_prepare_for_adas_deployment(self):
        """Test ADAS deployment preparation"""
        result = self.preparer.prepare_for_adas_deployment(str(self.test_model_dir))
        self.assertTrue(result['success'])
        self.assertIn('export_directory', result)

    def test_validate_phase7_pipeline(self):
        """Test Phase 7 pipeline validation"""
        validation_results = self.preparer.validate_phase7_pipeline()
        self.assertEqual(validation_results['baked_models_found'], 1)

class TestPipelineValidator(unittest.TestCase):
    """Test cases for pipeline validator"""

    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'phase5_config': {'phase5_model_dir': self.temp_dir + '/phase5'},
            'phase7_config': {'phase6_output_dir': self.temp_dir + '/phase6'},
            'min_health_score': 70.0
        }

        # Mock the component validators to avoid file system dependencies
        with patch('phase6.integration.pipeline_validator.create_phase5_connector'), \
             patch('phase6.integration.pipeline_validator.create_phase7_preparer'):
            self.validator = PipelineValidator(self.config)

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('psutil.cpu_count', return_value=8)
    @patch('psutil.cpu_percent', return_value=30.0)
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_validate_system_resources(self, mock_disk, mock_memory, mock_cpu_percent, mock_cpu_count):
        """Test system resource validation"""
        # Mock memory info
        mock_memory_obj = Mock()
        mock_memory_obj.total = 16 * 1024**3  # 16GB
        mock_memory_obj.available = 8 * 1024**3  # 8GB available
        mock_memory_obj.percent = 50.0
        mock_memory.return_value = mock_memory_obj

        # Mock disk info
        mock_disk_obj = Mock()
        mock_disk_obj.total = 1000 * 1024**3  # 1TB
        mock_disk_obj.free = 500 * 1024**3   # 500GB free
        mock_disk_obj.used = 500 * 1024**3   # 500GB used
        mock_disk.return_value = mock_disk_obj

        result = self.validator._validate_system_resources()
        self.assertTrue(result.passed)
        self.assertGreaterEqual(result.score, 70.0)

    def test_validate_complete_pipeline(self):
        """Test complete pipeline validation"""
        # Mock all validation methods to return passing results
        with patch.object(self.validator, '_validate_phase5_integration') as mock_phase5, \
             patch.object(self.validator, '_validate_baking_core') as mock_baking, \
             patch.object(self.validator, '_validate_optimization_engine') as mock_opt, \
             patch.object(self.validator, '_validate_quality_gates') as mock_quality, \
             patch.object(self.validator, '_validate_phase7_preparation') as mock_phase7, \
             patch.object(self.validator, '_validate_system_resources') as mock_resources, \
             patch.object(self.validator, '_validate_data_flow') as mock_data, \
             patch.object(self.validator, '_validate_error_handling') as mock_error:

            # Mock all validations to pass
            mock_result = Mock()
            mock_result.passed = True
            mock_result.score = 85.0
            mock_result.issues = []
            mock_result.component = 'test'
            mock_result.execution_time_ms = 100.0

            for mock_method in [mock_phase5, mock_baking, mock_opt, mock_quality,
                              mock_phase7, mock_resources, mock_data, mock_error]:
                mock_method.return_value = mock_result

            pipeline_health = self.validator.validate_complete_pipeline()
            self.assertIn(pipeline_health.overall_health, ['EXCELLENT', 'GOOD'])
            self.assertGreaterEqual(pipeline_health.health_score, 70.0)

class TestStateManager(unittest.TestCase):
    """Test cases for state manager"""

    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'storage_dir': self.temp_dir + '/state',
            'max_state_age_days': 7
        }
        self.state_manager = StateManager(self.config)

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_state(self):
        """Test state creation"""
        test_data = {'test': True, 'value': 42}
        success = self.state_manager.create_state(
            'test_state',
            Phase.PHASE6_BAKING,
            test_data,
            model_id='test_model'
        )
        self.assertTrue(success)

    def test_get_state(self):
        """Test state retrieval"""
        test_data = {'test': True, 'value': 42}
        self.state_manager.create_state('test_state', Phase.PHASE6_BAKING, test_data)

        retrieved = self.state_manager.get_state('test_state')
        self.assertIsNotNone(retrieved)
        data, metadata = retrieved
        self.assertEqual(data['test'], True)
        self.assertEqual(data['value'], 42)

    def test_update_state(self):
        """Test state updating"""
        test_data = {'test': True}
        self.state_manager.create_state('test_state', Phase.PHASE6_BAKING, test_data)

        updated_data = {'test': False, 'updated': True}
        success = self.state_manager.update_state(
            'test_state',
            updated_data,
            status=StateStatus.COMPLETED
        )
        self.assertTrue(success)

        retrieved = self.state_manager.get_state('test_state')
        data, metadata = retrieved
        self.assertEqual(data['test'], False)
        self.assertEqual(metadata.status, StateStatus.COMPLETED)

    def test_delete_state(self):
        """Test state deletion"""
        test_data = {'test': True}
        self.state_manager.create_state('test_state', Phase.PHASE6_BAKING, test_data)

        success = self.state_manager.delete_state('test_state')
        self.assertTrue(success)

        retrieved = self.state_manager.get_state('test_state')
        self.assertIsNone(retrieved)

    def test_list_states(self):
        """Test state listing"""
        # Create multiple states
        for i in range(3):
            self.state_manager.create_state(
                f'test_state_{i}',
                Phase.PHASE6_BAKING,
                {'index': i}
            )

        states = self.state_manager.list_states(phase=Phase.PHASE6_BAKING)
        self.assertEqual(len(states), 3)

    def test_validate_state_consistency(self):
        """Test state consistency validation"""
        test_data = {'test': True}
        self.state_manager.create_state('test_state', Phase.PHASE6_BAKING, test_data)

        validation = self.state_manager.validate_state_consistency()
        self.assertTrue(validation['consistent'])
        self.assertEqual(validation['total_states'], 1)

    def test_create_checkpoint(self):
        """Test checkpoint creation"""
        test_data = {'test': True}
        self.state_manager.create_state('test_state', Phase.PHASE6_BAKING, test_data)

        success = self.state_manager.create_checkpoint('test_checkpoint')
        self.assertTrue(success)

    def test_restore_checkpoint(self):
        """Test checkpoint restoration"""
        # Create and checkpoint state
        test_data = {'test': True}
        self.state_manager.create_state('test_state', Phase.PHASE6_BAKING, test_data)
        self.state_manager.create_checkpoint('test_checkpoint')

        # Delete state
        self.state_manager.delete_state('test_state')

        # Restore checkpoint
        success = self.state_manager.restore_checkpoint('test_checkpoint', overwrite_existing=True)
        self.assertTrue(success)

        # Verify state is restored
        retrieved = self.state_manager.get_state('test_state')
        self.assertIsNotNone(retrieved)

class TestQualityCoordinator(unittest.TestCase):
    """Test cases for quality coordinator"""

    def setUp(self):
        """Set up test environment"""
        self.config = {'parallel_execution': True}
        self.coordinator = QualityCoordinator(self.config)

    def test_register_quality_gate(self):
        """Test quality gate registration"""
        gate_config = {
            'name': 'Test Gate',
            'description': 'Test quality gate',
            'metrics': ['model_accuracy', 'inference_latency'],
            'weight': 1.0,
            'threshold': 80.0
        }

        success = self.coordinator.register_quality_gate('test_gate', gate_config)
        self.assertTrue(success)

    def test_execute_quality_gate(self):
        """Test quality gate execution"""
        # Register test gate
        gate_config = {
            'name': 'Performance Gate',
            'description': 'Performance validation',
            'metrics': ['model_accuracy', 'inference_latency'],
            'weight': 1.0,
            'threshold': 80.0
        }
        self.coordinator.register_quality_gate('perf_gate', gate_config)

        # Test model data
        model_data = {
            'accuracy': 0.96,
            'inference_time_ms': 45.0,
            'model_size_mb': 85.0
        }

        result = self.coordinator.execute_quality_gate('perf_gate', model_data)
        self.assertTrue(result.passed)
        self.assertGreaterEqual(result.score, 80.0)

    def test_execute_all_quality_gates(self):
        """Test execution of all quality gates"""
        # Register multiple gates
        for i in range(3):
            gate_config = {
                'name': f'Test Gate {i}',
                'description': f'Test gate {i}',
                'metrics': ['model_accuracy'],
                'weight': 1.0,
                'threshold': 75.0
            }
            self.coordinator.register_quality_gate(f'gate_{i}', gate_config)

        model_data = {'accuracy': 0.96}
        assessment = self.coordinator.execute_all_quality_gates(model_data)

        self.assertEqual(assessment.total_gates, 3)
        self.assertEqual(assessment.passed_gates, 3)

    def test_get_quality_dashboard_data(self):
        """Test quality dashboard data generation"""
        dashboard_data = self.coordinator.get_quality_dashboard_data()
        self.assertIn('total_gates', dashboard_data)
        self.assertIn('total_metrics', dashboard_data)

class TestWorkingValidator(unittest.TestCase):
    """Test cases for working validator"""

    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'test_results_dir': self.temp_dir + '/test_results',
            'phase5_config': {},
            'phase7_config': {},
            'pipeline_config': {},
            'state_config': {},
            'quality_config': {}
        }

        # Mock all component validators
        with patch('phase6.integration.working_validator.create_phase5_connector'), \
             patch('phase6.integration.working_validator.create_phase7_preparer'), \
             patch('phase6.integration.working_validator.create_pipeline_validator'), \
             patch('phase6.integration.working_validator.create_state_manager'), \
             patch('phase6.integration.working_validator.create_quality_coordinator'):
            self.validator = WorkingValidator(self.config)

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_system_tests_defined(self):
        """Test that system tests are properly defined"""
        tests = self.validator.system_tests
        self.assertGreater(len(tests), 0)

        # Check for critical tests
        critical_tests = [t for t in tests if t.critical]
        self.assertGreater(len(critical_tests), 0)

    @patch('psutil.cpu_percent', return_value=30.0)
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_resource_utilization_test(self, mock_disk, mock_memory, mock_cpu):
        """Test resource utilization testing"""
        # Mock memory
        mock_memory_obj = Mock()
        mock_memory_obj.percent = 50.0
        mock_memory.return_value = mock_memory_obj

        # Mock disk
        mock_disk_obj = Mock()
        mock_disk_obj.used = 500 * 1024**3
        mock_disk_obj.total = 1000 * 1024**3
        mock_disk.return_value = mock_disk_obj

        passed, score, output = self.validator._test_resource_utilization()
        self.assertTrue(passed)
        self.assertGreaterEqual(score, 70.0)

    def test_deterministic_behavior_test(self):
        """Test deterministic behavior testing"""
        passed, score, output = self.validator._test_deterministic_behavior()
        self.assertTrue(passed)
        self.assertEqual(score, 100.0)

    def test_error_recovery_test(self):
        """Test error recovery testing"""
        passed, score, output = self.validator._test_error_recovery()
        self.assertTrue(passed)
        self.assertEqual(score, 100.0)

    def test_concurrent_processing_test(self):
        """Test concurrent processing testing"""
        passed, score, output = self.validator._test_concurrent_processing()
        self.assertTrue(passed)
        self.assertGreaterEqual(score, 90.0)

    @patch('psutil.Process')
    def test_memory_leak_detection_test(self, mock_process):
        """Test memory leak detection"""
        # Mock process memory info
        mock_memory_info = Mock()
        mock_memory_info.rss = 100 * 1024 * 1024  # 100MB constant
        mock_process.return_value.memory_info.return_value = mock_memory_info

        passed, score, output = self.validator._test_memory_leak_detection()
        self.assertTrue(passed)
        self.assertGreaterEqual(score, 80.0)

    def test_validate_system_working(self):
        """Test complete system working validation"""
        # Mock all test methods to return passing results
        test_methods = [
            '_test_phase5_integration',
            '_test_phase7_preparation',
            '_test_pipeline_validation',
            '_test_state_management',
            '_test_quality_coordination',
            '_test_e2e_model_baking',
            '_test_e2e_quality_validation',
            '_test_performance_baseline',
            '_test_concurrent_processing',
            '_test_deterministic_behavior',
            '_test_error_recovery',
            '_test_resource_utilization',
            '_test_memory_leak_detection'
        ]

        for method_name in test_methods:
            method = getattr(self.validator, method_name, None)
            if method:
                with patch.object(self.validator, method_name, return_value=(True, 90.0, 'Test passed')):
                    pass

        # Mock all test methods
        def mock_test_method(*args, **kwargs):
            return True, 90.0, 'Test passed'

        for method_name in test_methods:
            setattr(self.validator, method_name, mock_test_method)

        system_health = self.validator.validate_system_working(parallel=False)
        self.assertTrue(system_health.healthy)
        self.assertGreaterEqual(system_health.health_score, 70.0)

class TestIntegrationE2E(unittest.TestCase):
    """End-to-end integration tests"""

    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_integration_workflow(self):
        """Test complete integration workflow"""
        # This would test the full workflow from Phase 5 to Phase 7
        # For now, we'll test component interaction

        # Create state manager
        state_config = {'storage_dir': self.temp_dir + '/state'}
        state_manager = create_state_manager(state_config)

        # Create test state
        test_data = {'workflow_test': True, 'timestamp': time.time()}
        success = state_manager.create_state(
            'integration_test',
            Phase.PHASE6_BAKING,
            test_data
        )
        self.assertTrue(success)

        # Verify state exists
        retrieved = state_manager.get_state('integration_test')
        self.assertIsNotNone(retrieved)

        # Create quality coordinator
        quality_config = {}
        quality_coordinator = create_quality_coordinator(quality_config)

        # Test quality assessment
        model_data = {
            'accuracy': 0.96,
            'inference_time_ms': 45.0,
            'model_size_mb': 85.0
        }

        assessment = quality_coordinator.execute_all_quality_gates(model_data)
        self.assertIsNotNone(assessment)

        # Update state with quality results
        success = state_manager.update_state(
            'integration_test',
            {'quality_score': assessment.overall_score},
            status=StateStatus.COMPLETED
        )
        self.assertTrue(success)

    def test_component_interaction(self):
        """Test interaction between components"""
        # Test that components can work together
        config = {
            'storage_dir': self.temp_dir + '/state',
            'parallel_execution': True
        }

        # Create components
        state_manager = create_state_manager(config)
        quality_coordinator = create_quality_coordinator(config)

        # Test workflow
        workflow_id = f'test_workflow_{int(time.time())}'

        # Step 1: Create initial state
        initial_data = {'phase': 'start', 'timestamp': time.time()}
        success = state_manager.create_state(
            workflow_id,
            Phase.PHASE6_BAKING,
            initial_data
        )
        self.assertTrue(success)

        # Step 2: Process through quality gates
        model_data = {'accuracy': 0.95, 'inference_time_ms': 50.0}
        assessment = quality_coordinator.execute_all_quality_gates(model_data)

        # Step 3: Update state with results
        result_data = {
            'phase': 'quality_checked',
            'quality_score': assessment.overall_score,
            'quality_status': assessment.overall_status.value
        }
        success = state_manager.update_state(
            workflow_id,
            result_data,
            status=StateStatus.COMPLETED
        )
        self.assertTrue(success)

        # Step 4: Verify final state
        final_state = state_manager.get_state(workflow_id)
        self.assertIsNotNone(final_state)
        data, metadata = final_state
        self.assertEqual(data['phase'], 'quality_checked')
        self.assertEqual(metadata.status, StateStatus.COMPLETED)

def run_integration_tests():
    """Run all integration tests"""
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestPhase5Connector,
        TestPhase7Preparer,
        TestPipelineValidator,
        TestStateManager,
        TestQualityCoordinator,
        TestWorkingValidator,
        TestIntegrationE2E
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Generate report
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    success_rate = (total_tests - failures - errors) / total_tests if total_tests > 0 else 0

    print(f"\n{'='*50}")
    print(f"INTEGRATION TEST RESULTS")
    print(f"{'='*50}")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_tests - failures - errors}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")
    print(f"Success Rate: {success_rate:.1%}")

    if result.failures:
        print(f"\nFAILURES:")
        for test, failure in result.failures:
            print(f"- {test}: {failure}")

    if result.errors:
        print(f"\nERRORS:")
        for test, error in result.errors:
            print(f"- {test}: {error}")

    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_integration_tests()
    exit(0 if success else 1)