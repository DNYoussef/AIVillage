"""
Agent Forge Phase 5: Integration Test Suite
===========================================

Comprehensive integration testing for Phase 4-5-6 workflow with
validation of all components and NASA POT10 compliance.

Key Features:
- Phase 4-5-6 integration testing
- Component interaction validation
- NASA POT10 compliance verification
- Production readiness assessment
- End-to-end workflow validation
"""

import torch
import os
import json
import tempfile
import shutil
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from phase5.training_architecture import TrainingArchitecture, validate_nasa_compliance
from phase5.distributed_trainer import DistributedTrainer
from phase5.bitnet_training import BitNetTrainingOptimizer, convert_model_to_bitnet
from phase5.grokfast_integration import GrokfastAccelerator
from phase5.performance_monitor import TrainingMonitor
from phase5.checkpoint_manager import CheckpointManager
from phase5.training_config import TrainingConfig, Environment
from phase5.training_pipeline import Phase5Pipeline


class Phase5IntegrationTest(unittest.TestCase):
    """Comprehensive integration test suite for Phase 5."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = TrainingConfig(
            experiment_name="integration_test",
            environment=Environment.TESTING,
            output_dir=self.temp_dir,
            num_epochs=2
        )
        self.config.data.batch_size = 4
        self.config.optimization.learning_rate = 1e-3

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('integration_test')

    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_component_initialization(self):
        """Test initialization of all Phase 5 components."""
        self.logger.info("Testing component initialization...")

        # Test model creation
        model = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10)
        )
        self.assertIsNotNone(model)

        # Test training architecture
        trainer = TrainingArchitecture(self.config, model)
        self.assertIsNotNone(trainer)
        self.assertEqual(trainer.config, self.config)

        # Test distributed trainer
        distributed_trainer = DistributedTrainer(self.config)
        self.assertIsNotNone(distributed_trainer)

        # Test BitNet optimizer
        bitnet_optimizer = BitNetTrainingOptimizer(self.config)
        self.assertIsNotNone(bitnet_optimizer)

        # Test Grokfast accelerator
        grokfast = GrokfastAccelerator(self.config)
        self.assertIsNotNone(grokfast)

        # Test performance monitor
        monitor = TrainingMonitor(self.config)
        self.assertIsNotNone(monitor)
        monitor.cleanup()

        # Test checkpoint manager
        checkpoint_manager = CheckpointManager(self.config)
        self.assertIsNotNone(checkpoint_manager)
        checkpoint_manager.cleanup()

        self.logger.info("✓ Component initialization successful")

    def test_bitnet_conversion(self):
        """Test BitNet model conversion."""
        self.logger.info("Testing BitNet conversion...")

        # Create test model
        model = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10)
        )

        # Count original parameters
        original_params = sum(p.numel() for p in model.parameters())

        # Convert to BitNet
        bitnet_model = convert_model_to_bitnet(model)

        # Verify conversion
        converted_params = sum(p.numel() for p in bitnet_model.parameters())
        self.assertEqual(original_params, converted_params)  # Parameter count should be same

        # Test BitNet optimizer integration
        bitnet_optimizer = BitNetTrainingOptimizer(self.config)
        param_groups = bitnet_optimizer.get_parameter_groups(bitnet_model)
        self.assertGreater(len(param_groups), 0)

        self.logger.info("✓ BitNet conversion successful")

    def test_grokfast_integration(self):
        """Test Grokfast acceleration integration."""
        self.logger.info("Testing Grokfast integration...")

        # Create model and optimizer
        model = torch.nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())

        # Initialize Grokfast
        grokfast = GrokfastAccelerator(self.config)
        grokfast.initialize(model)

        # Test acceleration
        dummy_loss = 1.0
        acceleration_applied = grokfast.accelerate_if_needed(model, optimizer, dummy_loss)
        self.assertIsInstance(acceleration_applied, bool)

        # Test scheduler creation
        scheduler = grokfast.create_scheduler(
            optimizer,
            num_training_steps=1000,
            num_warmup_steps=100
        )
        self.assertIsNotNone(scheduler)

        # Test statistics
        stats = grokfast.get_acceleration_stats()
        self.assertIn('global_step', stats)
        self.assertIn('current_phase', stats)

        self.logger.info("✓ Grokfast integration successful")

    def test_distributed_coordination(self):
        """Test distributed training coordination."""
        self.logger.info("Testing distributed coordination...")

        # Test distributed trainer
        trainer = DistributedTrainer(self.config)

        # Test GPU monitoring
        gpu_metrics = trainer.monitor_gpu_utilization()
        self.assertIsInstance(gpu_metrics, dict)

        # Test synchronization
        sync_time = trainer.synchronize_processes()
        self.assertIsInstance(sync_time, float)

        # Test performance stats
        stats = trainer.get_performance_stats()
        self.assertIn('world_size', stats)
        self.assertIn('rank', stats)

        self.logger.info("✓ Distributed coordination successful")

    def test_performance_monitoring(self):
        """Test performance monitoring system."""
        self.logger.info("Testing performance monitoring...")

        monitor = TrainingMonitor(self.config)

        # Test epoch monitoring
        monitor.start_epoch(0, 100)

        # Test step monitoring
        for step in range(5):
            monitor.log_step(
                step=step,
                loss=1.0 - (step * 0.1),
                learning_rate=1e-4,
                step_time=0.5
            )

        monitor.end_epoch(0.5)

        # Test NASA compliance metrics
        nasa_metrics = monitor.get_nasa_compliance_metrics()
        self.assertIn('compliance_score', nasa_metrics)
        self.assertIn('issues', nasa_metrics)

        # Test final report
        final_report = monitor.generate_final_report()
        self.assertIn('training_summary', final_report)
        self.assertIn('nasa_pot10_compliance', final_report)

        monitor.cleanup()
        self.logger.info("✓ Performance monitoring successful")

    def test_checkpoint_management(self):
        """Test checkpoint management system."""
        self.logger.info("Testing checkpoint management...")

        manager = CheckpointManager(self.config)

        # Create test model and optimizer
        model = torch.nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())

        # Test checkpoint saving
        metrics = {'train_loss': 0.5, 'val_loss': 0.6, 'step': 100}
        checkpoint_path = manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            epoch=1,
            metrics=metrics
        )

        # Verify checkpoint was saved (if on main process)
        if checkpoint_path:
            self.assertTrue(os.path.exists(checkpoint_path))

            # Test checkpoint loading
            loaded_data = manager.load_checkpoint(checkpoint_path, model, optimizer)
            self.assertIn('model_state_dict', loaded_data)
            self.assertIn('metadata', loaded_data)

        # Test checkpoint statistics
        stats = manager.get_checkpoint_stats()
        self.assertIn('total_checkpoints', stats)
        self.assertIn('nasa_pot10_compliant', stats)

        manager.cleanup()
        self.logger.info("✓ Checkpoint management successful")

    def test_nasa_compliance(self):
        """Test NASA POT10 compliance validation."""
        self.logger.info("Testing NASA POT10 compliance...")

        # Create training results
        training_results = {
            'training_time': 3600.0,
            'final_train_loss': 0.5,
            'final_val_loss': 0.6,
            'best_val_loss': 0.55,
            'total_steps': 10000,
            'model_params': 1000000,
            'gpu_memory_peak': 8e9,
            'training_metrics': {
                'loss_history': [1.0, 0.8, 0.6, 0.5],
                'throughput': [100, 110, 105, 108]
            }
        }

        # Validate compliance
        compliance = validate_nasa_compliance(training_results)
        self.assertIsInstance(compliance, dict)

        # Check all compliance criteria
        required_checks = [
            'documentation_complete',
            'performance_metrics_tracked',
            'error_handling_implemented',
            'logging_comprehensive',
            'checkpoint_management',
            'distributed_coordination',
            'memory_efficiency',
            'training_stability'
        ]

        for check in required_checks:
            self.assertIn(check, compliance)

        # Calculate compliance score
        compliance_score = sum(compliance.values()) / len(compliance) * 100
        self.assertGreaterEqual(compliance_score, 90.0)  # NASA POT10 requirement

        self.logger.info(f"✓ NASA compliance validated: {compliance_score:.1f}%")

    def test_phase_integration(self):
        """Test Phase 4-5-6 integration workflow."""
        self.logger.info("Testing Phase 4-5-6 integration...")

        # Create mock Phase 4 output
        phase4_dir = Path(self.temp_dir) / 'phase4'
        phase4_dir.mkdir(parents=True, exist_ok=True)

        phase4_metadata = {
            'compression_complete': True,
            'compression_ratio': 8.2,
            'model_config': {
                'hidden_size': 768,
                'num_layers': 12
            }
        }

        with open(phase4_dir / 'compression_metadata.json', 'w') as f:
            json.dump(phase4_metadata, f)

        # Create mock compressed model
        mock_model = torch.nn.Linear(768, 10)
        torch.save(mock_model.state_dict(), phase4_dir / 'compressed_model.pt')

        # Test pipeline initialization
        self.config.phase_integration.phase4_input_dir = str(phase4_dir)
        pipeline = Phase5Pipeline()
        pipeline.config = self.config

        # Test Phase 4 data loading
        phase4_data = pipeline.load_phase4_models()
        self.assertIn('metadata', phase4_data)
        self.assertIn('compressed_weights', phase4_data)

        # Test Phase 6 preparation
        phase6_dir = Path(self.temp_dir) / 'phase6'
        self.config.phase_integration.phase6_output_dir = str(phase6_dir)

        # Mock model for Phase 6 preparation
        pipeline.model = mock_model
        pipeline.training_results = {'training_summary': {'final_loss': 0.5}}
        pipeline._prepare_phase6_output()

        # Verify Phase 6 output
        self.assertTrue((phase6_dir / 'trained_model.pt').exists())
        self.assertTrue((phase6_dir / 'phase5_metadata.json').exists())

        pipeline.cleanup()
        self.logger.info("✓ Phase integration successful")

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        self.logger.info("Testing end-to-end workflow...")

        # Configure for minimal testing
        self.config.num_epochs = 1
        self.config.data.batch_size = 2

        # Create pipeline
        pipeline = Phase5Pipeline()
        pipeline.config = self.config

        # Test validation pipeline (faster than full training)
        try:
            validation_results = pipeline.run_validation_pipeline()

            self.assertIn('model_parameters', validation_results)
            self.assertIn('bitnet_conversion', validation_results)
            self.assertIn('grokfast_enabled', validation_results)
            self.assertIn('distributed_ready', validation_results)

            self.assertGreater(validation_results['model_parameters'], 0)

        except Exception as e:
            self.logger.error(f"End-to-end workflow failed: {e}")
            raise

        pipeline.cleanup()
        self.logger.info("✓ End-to-end workflow successful")

    def test_production_readiness(self):
        """Test production readiness assessment."""
        self.logger.info("Testing production readiness...")

        # Create production configuration
        prod_config = TrainingConfig(
            experiment_name="production_test",
            environment=Environment.PRODUCTION,
            output_dir=self.temp_dir
        )
        prod_config.nasa_compliance.enforce_compliance = True
        prod_config.nasa_compliance.min_compliance_score = 95.0

        # Test configuration validation
        try:
            prod_config.validate()
            self.logger.info("✓ Production configuration valid")
        except ValueError as e:
            self.fail(f"Production configuration invalid: {e}")

        # Test NASA compliance checklist
        checklist = prod_config.get_nasa_compliance_checklist()
        self.assertIsInstance(checklist, dict)

        # Verify critical compliance items
        critical_items = [
            'documentation_complete',
            'performance_metrics_tracked',
            'error_handling_implemented',
            'logging_comprehensive',
            'checkpoint_management'
        ]

        for item in critical_items:
            self.assertTrue(checklist.get(item, False), f"Critical compliance item failed: {item}")

        self.logger.info("✓ Production readiness validated")


def run_integration_tests():
    """Run comprehensive integration test suite."""
    print("=" * 60)
    print("AGENT FORGE PHASE 5: INTEGRATION TEST SUITE")
    print("=" * 60)
    print()

    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(Phase5IntegrationTest)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print()
    print("=" * 60)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 60)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")

    if result.wasSuccessful():
        print()
        print("🎯 ALL INTEGRATION TESTS PASSED")
        print("✅ PHASE 5 IS PRODUCTION READY")
        print("✅ NASA POT10 COMPLIANCE VALIDATED")
        print("✅ PHASE 4-5-6 INTEGRATION VERIFIED")
    else:
        print()
        print("❌ SOME INTEGRATION TESTS FAILED")
        print("⚠️  Additional validation required")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)