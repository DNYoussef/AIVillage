"""
Comprehensive Test Suite for the Self-Evolving Agent System

Tests all components of the evolution system including:
- Agent Evolution Engine
- Safe Code Modification
- Meta-Learning Engine
- Evolution Orchestrator
- Dashboard Integration
"""

import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

import numpy as np
import pytest

# Import evolution system components
from agent_forge.evolution import (
    AgentEvolutionEngine,
    AgentKPIs,
    AgentGenome,
    SafeCodeModifier,
    CodeTransformations,
    MetaLearningEngine,
    EvolutionOrchestrator,
    OrchestrationConfig,
    initialize_evolution_system,
    quick_evolution_cycle
)

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestAgentEvolutionEngine(unittest.TestCase):
    """Test the core evolution engine functionality"""

    def setUp(self):
        """Setup test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.engine = AgentEvolutionEngine(
            evolution_data_path=self.test_dir,
            population_size=6  # Small population for testing
        )

    def tearDown(self):
        """Cleanup test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    async def test_population_initialization(self):
        """Test agent population initialization"""
        await self.engine.initialize_population()

        self.assertEqual(len(self.engine.agent_population), 6)

        # Check that agents have different specializations
        specializations = set()
        for agent in self.engine.agent_population:
            # Extract specialization from agent_id
            spec = agent.agent_id.split('_')[-1]
            specializations.add(spec)

        self.assertGreater(len(specializations), 1, "Agents should have diverse specializations")

    async def test_fitness_evaluation(self):
        """Test fitness evaluation of agent population"""
        await self.engine.initialize_population()

        # Create mock evaluation tasks
        async def mock_task(genome):
            return np.random.uniform(0.3, 0.9)

        evaluation_tasks = [mock_task, mock_task]

        fitness_scores = await self.engine._evaluate_population(evaluation_tasks)

        self.assertEqual(len(fitness_scores), 6)
        for score in fitness_scores.values():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    async def test_genetic_operations(self):
        """Test genetic crossover and mutation operations"""
        await self.engine.initialize_population()

        parent1 = self.engine.agent_population[0]
        parent2 = self.engine.agent_population[1]

        # Test crossover
        child1, child2 = self.engine.genetic_optimizer.crossover(parent1, parent2)

        self.assertNotEqual(child1.agent_id, parent1.agent_id)
        self.assertNotEqual(child2.agent_id, parent2.agent_id)
        self.assertEqual(child1.generation, max(parent1.generation, parent2.generation) + 1)

        # Test mutation
        mutated = self.engine.genetic_optimizer.mutate(child1)
        # Mutation might not always change values, so we just check it doesn't crash
        self.assertIsInstance(mutated, AgentGenome)

    async def test_kpi_tracking(self):
        """Test KPI tracking and fitness scoring"""
        kpis = AgentKPIs(
            agent_id="test_agent",
            task_success_rate=0.85,
            user_satisfaction=0.9,
            resource_efficiency=0.75,
            code_quality_score=0.8
        )

        # Test fitness calculation
        fitness = kpis.fitness_score()
        self.assertGreaterEqual(fitness, 0.0)
        self.assertLessEqual(fitness, 1.0)

        # Test KPI recording
        self.engine.kpi_tracker.record_kpis(kpis)

        fitness_scores = self.engine.kpi_tracker.get_fitness_scores(["test_agent"])
        self.assertIn("test_agent", fitness_scores)

    async def test_evolution_cycle(self):
        """Test complete evolution cycle"""
        await self.engine.initialize_population()

        # Mock evaluation tasks
        async def mock_evaluation_task(genome):
            # Simulate varying performance based on genome characteristics
            base_fitness = 0.5
            if genome.hyperparameters.get('learning_rate', 0.001) < 0.005:
                base_fitness += 0.2
            return min(1.0, base_fitness + np.random.normal(0, 0.1))

        evaluation_tasks = [mock_evaluation_task]

        results = await self.engine.run_evolution_cycle(
            generations=2,
            evaluation_tasks=evaluation_tasks
        )

        self.assertTrue(results['generations_run'] == 2)
        self.assertEqual(len(results['best_fitness_history']), 2)
        self.assertGreater(len(self.engine.agent_population), 0)


class TestSafeCodeModifier(unittest.TestCase):
    """Test safe code modification system"""

    def setUp(self):
        """Setup test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.modifier = SafeCodeModifier(backup_path=f"{self.test_dir}/backups")

        # Create test Python file
        self.test_file = Path(self.test_dir) / "test_agent.py"
        with open(self.test_file, 'w') as f:
            f.write('''
def train_model(learning_rate=0.01, batch_size=16):
    """Train a model with given parameters"""
    print(f"Training with lr={learning_rate}, batch={batch_size}")
    return {"loss": 0.5, "accuracy": 0.8}

class TestAgent:
    def __init__(self):
        self.name = "test"

    def predict(self, data):
        return {"prediction": "positive"}
''')

    def tearDown(self):
        """Cleanup test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    async def test_code_modification_proposal(self):
        """Test proposing code modifications"""

        def hyperparameter_transformer(code):
            return CodeTransformations.optimize_hyperparameters(
                code, {'learning_rate': 0.001, 'batch_size': 32}
            )

        modification = await self.modifier.propose_modification(
            agent_id="test_agent",
            file_path=str(self.test_file),
            modification_type="hyperparameter_optimization",
            description="Optimize learning rate and batch size",
            code_transformer=hyperparameter_transformer
        )

        self.assertIsNotNone(modification.modification_id)
        self.assertEqual(modification.agent_id, "test_agent")
        self.assertGreater(modification.safety_score, 0.0)
        self.assertIn("learning_rate = 0.001", modification.modified_code)

    async def test_code_validation(self):
        """Test code validation system"""

        # Test valid code
        valid_code = """
def safe_function():
    return "safe"
"""

        modification = Mock()
        modification.modified_code = valid_code
        modification.modification_id = "test_mod"

        validation_results = await self.modifier.validator.validate_modification(modification)

        self.assertTrue(validation_results['syntax_valid'])
        self.assertTrue(validation_results['security_safe'])
        self.assertGreater(validation_results['safety_score'], 0.5)

        # Test unsafe code
        unsafe_code = """
import os
os.system("rm -rf /")
"""

        modification.modified_code = unsafe_code
        validation_results = await self.modifier.validator.validate_modification(modification)

        self.assertFalse(validation_results['security_safe'])
        self.assertLess(validation_results['safety_score'], 0.5)

    async def test_sandbox_testing(self):
        """Test sandbox environment for code testing"""

        # Create a simple modification
        def simple_transformer(code):
            return code + "\n# Added comment\n"

        modification = await self.modifier.propose_modification(
            agent_id="test_agent",
            file_path=str(self.test_file),
            modification_type="documentation",
            description="Add comment",
            code_transformer=simple_transformer
        )

        # Test in sandbox
        if modification.safety_score >= 0.8:
            test_results = await self.modifier.test_modification(modification.modification_id)

            # Should succeed for simple comment addition
            self.assertIn('success', test_results)

    async def test_code_transformations(self):
        """Test various code transformation utilities"""

        test_code = '''
def example_function(x, y):
    return x + y

class ExampleClass:
    def method(self):
        return "result"
'''

        # Test documentation improvement
        improved_code = CodeTransformations.improve_documentation(test_code)
        self.assertIn('"""', improved_code)

        # Test error handling addition
        error_handled_code = CodeTransformations.add_error_handling(test_code, ['example_function'])
        # Should contain try-except block (implementation may vary)
        self.assertIn('try', error_handled_code.lower())

        # Test hyperparameter optimization
        optimized_code = CodeTransformations.optimize_hyperparameters(
            "learning_rate = 0.01", {'learning_rate': 0.001}
        )
        self.assertIn('0.001', optimized_code)


class TestMetaLearningEngine(unittest.TestCase):
    """Test meta-learning system"""

    def setUp(self):
        """Setup test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.meta_engine = MetaLearningEngine(
            storage_path=f"{self.test_dir}/meta_learning"
        )

    def tearDown(self):
        """Cleanup test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    async def test_learning_rate_optimization(self):
        """Test learning rate optimization"""

        # Record some learning experiences
        for i in range(5):
            self.meta_engine.lr_optimizer.record_learning_experience(
                agent_id="test_agent",
                learning_rate=0.001 * (i + 1),
                performance_improvement=0.1 + 0.1 * i,
                convergence_steps=50 - i * 5,
                task_difficulty=0.5
            )

        # Optimize learning rate
        optimal_lr = self.meta_engine.lr_optimizer.optimize_learning_rate(
            agent_id="test_agent",
            task_difficulty=0.5
        )

        self.assertGreater(optimal_lr, 0.0)
        self.assertLess(optimal_lr, 0.1)

    async def test_strategy_optimization(self):
        """Test learning strategy optimization"""

        task_characteristics = {
            'difficulty': 0.7,
            'data_size': 1000,
            'task_complexity': 'high'
        }

        learning_config = await self.meta_engine.optimize_agent_learning(
            agent_id="test_agent",
            task_type="classification",
            task_characteristics=task_characteristics,
            current_performance=0.6
        )

        self.assertIn('learning_rate', learning_config)
        self.assertIn('strategy', learning_config)
        self.assertGreater(learning_config['learning_rate'], 0.0)

    async def test_few_shot_learning(self):
        """Test few-shot learning capabilities"""

        # Mock support examples
        support_examples = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        support_labels = [0, 1, 0]

        # Create prototypes
        prototypes = self.meta_engine.few_shot_learner.create_prototype(
            task_id="test_task",
            support_examples=support_examples,
            support_labels=support_labels
        )

        self.assertIn(0, prototypes)
        self.assertIn(1, prototypes)

        # Test prediction
        query_example = [2, 3, 4]
        prediction, confidence = self.meta_engine.few_shot_learner.few_shot_predict(
            task_id="test_task",
            query_example=query_example
        )

        self.assertIn(prediction, [0, 1])
        self.assertGreater(confidence, 0.0)

    async def test_learning_outcome_recording(self):
        """Test recording and analyzing learning outcomes"""

        learning_config = {
            'learning_rate': 0.001,
            'strategy': 'conservative_lr',
            'meta_features': {'difficulty': 0.5}
        }

        await self.meta_engine.record_learning_outcome(
            agent_id="test_agent",
            task_type="classification",
            initial_performance=0.5,
            final_performance=0.8,
            learning_config=learning_config,
            learning_time=120.0,
            convergence_steps=50
        )

        # Check experience was recorded
        self.assertGreater(len(self.meta_engine.experiences), 0)

        # Get agent profile
        profile = self.meta_engine.get_agent_learning_profile("test_agent")

        self.assertEqual(profile['agent_id'], "test_agent")
        self.assertGreater(profile['total_experiences'], 0)
        self.assertGreater(profile['avg_improvement'], 0.0)


class TestEvolutionOrchestrator(unittest.TestCase):
    """Test evolution orchestration system"""

    def setUp(self):
        """Setup test environment"""
        self.test_dir = tempfile.mkdtemp()

        config = OrchestrationConfig(
            evolution_interval_hours=1,  # Short interval for testing
            monitoring_interval_minutes=1,
            auto_evolution_enabled=False,  # Manual control for testing
            safety_mode=True,
            max_population_size=6
        )

        self.orchestrator = EvolutionOrchestrator(
            config=config,
            storage_path=self.test_dir
        )

    def tearDown(self):
        """Cleanup test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    async def test_orchestrator_initialization(self):
        """Test orchestrator initialization"""

        # Should initialize without errors
        self.assertIsNotNone(self.orchestrator.evolution_engine)
        self.assertIsNotNone(self.orchestrator.code_modifier)
        self.assertIsNotNone(self.orchestrator.meta_learning_engine)
        self.assertFalse(self.orchestrator.state.is_running)

    async def test_orchestrator_lifecycle(self):
        """Test orchestrator start/stop lifecycle"""

        # Start orchestrator
        await self.orchestrator.start()
        self.assertTrue(self.orchestrator.state.is_running)

        # Get status
        status = await self.orchestrator.get_orchestration_status()
        self.assertIn('orchestrator', status)
        self.assertIn('evolution', status)

        # Stop orchestrator
        await self.orchestrator.stop()
        self.assertFalse(self.orchestrator.state.is_running)

    async def test_health_monitoring(self):
        """Test health monitoring system"""

        await self.orchestrator.start()

        try:
            # Run health check
            health_status = await self.orchestrator.health_monitor.check_system_health()

            self.assertIn('overall_healthy', health_status)
            self.assertIn('metrics', health_status)
            self.assertIn('alerts', health_status)

        finally:
            await self.orchestrator.stop()

    async def test_evolution_triggering(self):
        """Test manual evolution triggering"""

        await self.orchestrator.start()

        try:
            # Trigger evolution
            results = await self.orchestrator.trigger_evolution(generations=1, force=True)

            self.assertTrue(results['success'])
            self.assertIn('results', results)
            self.assertGreater(len(results['results']['best_fitness_history']), 0)

        finally:
            await self.orchestrator.stop()


class TestIntegrationScenarios(unittest.TestCase):
    """Test complete integration scenarios"""

    def setUp(self):
        """Setup test environment"""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Cleanup test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    async def test_full_system_initialization(self):
        """Test complete system initialization"""

        orchestrator = await initialize_evolution_system(
            evolution_data_path=self.test_dir,
            population_size=6,
            auto_evolution=False,
            safety_mode=True
        )

        self.assertIsNotNone(orchestrator)
        self.assertEqual(len(orchestrator.evolution_engine.agent_population), 6)

    async def test_evolution_cycle_integration(self):
        """Test complete evolution cycle with all components"""

        orchestrator = await initialize_evolution_system(
            evolution_data_path=self.test_dir,
            population_size=4,
            auto_evolution=False
        )

        async with orchestrator.orchestration_context():
            # Run evolution cycle
            results = await quick_evolution_cycle(orchestrator, generations=1)

            self.assertTrue(results['success'])

            # Check that meta-learning was engaged
            self.assertGreater(len(orchestrator.meta_learning_engine.experiences), 0)

            # Check that KPIs were recorded
            kpi_history = orchestrator.evolution_engine.kpi_tracker.kpi_history
            self.assertGreater(len(kpi_history), 0)

    @patch('agent_forge.evolution.evolution_dashboard.Flask')
    async def test_dashboard_integration(self, mock_flask):
        """Test dashboard integration (mocked)"""

        orchestrator = await initialize_evolution_system(
            evolution_data_path=self.test_dir,
            population_size=4
        )

        # Test dashboard data generation
        dashboard_data = await orchestrator.evolution_engine.get_evolution_dashboard_data()

        self.assertIn('population_stats', dashboard_data)
        self.assertIn('fitness_scores', dashboard_data)
        self.assertIn('performance_trends', dashboard_data)


class TestPerformanceAndScalability(unittest.TestCase):
    """Test performance and scalability aspects"""

    def setUp(self):
        """Setup test environment"""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Cleanup test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    async def test_large_population_evolution(self):
        """Test evolution with larger population"""

        engine = AgentEvolutionEngine(
            evolution_data_path=self.test_dir,
            population_size=18  # Full population
        )

        start_time = time.time()

        await engine.initialize_population()

        # Simple evaluation task
        async def fast_eval_task(genome):
            return np.random.uniform(0.4, 0.9)

        results = await engine.run_evolution_cycle(
            generations=1,
            evaluation_tasks=[fast_eval_task]
        )

        end_time = time.time()
        execution_time = end_time - start_time

        # Should complete within reasonable time (adjust threshold as needed)
        self.assertLess(execution_time, 60.0, "Evolution should complete within 60 seconds")
        self.assertTrue(results['generations_run'] == 1)

    async def test_concurrent_modifications(self):
        """Test handling multiple concurrent modifications"""

        modifier = SafeCodeModifier()

        # Create test files
        test_files = []
        for i in range(3):
            test_file = Path(self.test_dir) / f"agent_{i}.py"
            with open(test_file, 'w') as f:
                f.write(f"def agent_{i}_function(): return {i}")
            test_files.append(test_file)

        # Submit multiple modifications concurrently
        tasks = []
        for i, test_file in enumerate(test_files):
            task = modifier.propose_modification(
                agent_id=f"agent_{i}",
                file_path=str(test_file),
                modification_type="documentation",
                description=f"Add docs to agent {i}",
                code_transformer=lambda code, i=i: code + f"\n# Agent {i} documentation\n"
            )
            tasks.append(task)

        # Run concurrently
        modifications = await asyncio.gather(*tasks)

        self.assertEqual(len(modifications), 3)
        for mod in modifications:
            self.assertIsNotNone(mod.modification_id)


# Utility functions for running tests
def run_async_test(coro):
    """Helper to run async tests"""
    return asyncio.run(coro)


async def run_comprehensive_test_suite():
    """Run the complete test suite"""

    print("Running Comprehensive Evolution System Tests...")
    print("=" * 60)

    # Test categories
    test_classes = [
        TestAgentEvolutionEngine,
        TestSafeCodeModifier,
        TestMetaLearningEngine,
        TestEvolutionOrchestrator,
        TestIntegrationScenarios,
        TestPerformanceAndScalability
    ]

    total_tests = 0
    passed_tests = 0
    failed_tests = 0

    for test_class in test_classes:
        print(f"\n--- Testing {test_class.__name__} ---")

        test_instance = test_class()
        test_methods = [method for method in dir(test_instance) if method.startswith('test_')]

        for test_method in test_methods:
            total_tests += 1

            try:
                print(f"  Running {test_method}... ", end="")

                # Setup
                if hasattr(test_instance, 'setUp'):
                    test_instance.setUp()

                # Run test
                method = getattr(test_instance, test_method)
                if asyncio.iscoroutinefunction(method):
                    await method()
                else:
                    method()

                print("PASSED")
                passed_tests += 1

            except Exception as e:
                print(f"FAILED: {str(e)}")
                failed_tests += 1
                logger.error(f"Test {test_method} failed: {e}")

            finally:
                # Cleanup
                if hasattr(test_instance, 'tearDown'):
                    test_instance.tearDown()

    print("\n" + "=" * 60)
    print(f"Test Results:")
    print(f"  Total Tests: {total_tests}")
    print(f"  Passed: {passed_tests}")
    print(f"  Failed: {failed_tests}")
    print(f"  Success Rate: {(passed_tests/total_tests)*100:.1f}%")

    if failed_tests == 0:
        print("\n🎉 All tests passed! Evolution system is ready for deployment.")
    else:
        print(f"\n⚠️  {failed_tests} tests failed. Please review and fix issues.")

    return failed_tests == 0


# Interactive test functions
async def test_evolution_demo():
    """Run an interactive evolution demonstration"""

    print("Evolution System Demo")
    print("=" * 30)

    # Initialize system
    print("Initializing evolution system...")
    orchestrator = await initialize_evolution_system(
        evolution_data_path="demo_evolution_data",
        population_size=6,
        auto_evolution=False,
        safety_mode=True
    )

    async with orchestrator.orchestration_context():
        print(f"✅ System initialized with {len(orchestrator.evolution_engine.agent_population)} agents")

        # Show initial status
        status = await orchestrator.get_orchestration_status()
        print(f"📊 Initial avg fitness: {status['evolution']['avg_fitness']:.3f}")

        # Run evolution cycles
        for cycle in range(3):
            print(f"\n🧬 Running evolution cycle {cycle + 1}/3...")

            results = await quick_evolution_cycle(orchestrator, generations=1)

            if results['success']:
                best_fitness = max(results['results']['best_fitness_history'])
                print(f"   Best fitness: {best_fitness:.3f}")

                # Show population diversity
                diversity = results['results']['diversity_history'][-1]
                print(f"   Population diversity: {diversity:.3f}")
            else:
                print(f"   ❌ Evolution failed: {results.get('error', 'Unknown error')}")

        # Final status
        final_status = await orchestrator.get_orchestration_status()
        print(f"\n🏁 Final Results:")
        print(f"   Generation: {final_status['orchestrator']['current_generation']}")
        print(f"   Avg Fitness: {final_status['evolution']['avg_fitness']:.3f}")
        print(f"   Max Fitness: {final_status['evolution']['max_fitness']:.3f}")
        print(f"   Trend: {final_status['orchestrator']['performance_trend']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evolution System Testing")
    parser.add_argument("--demo", action="store_true", help="Run interactive demo")
    parser.add_argument("--comprehensive", action="store_true", help="Run comprehensive test suite")
    args = parser.parse_args()

    if args.demo:
        asyncio.run(test_evolution_demo())
    elif args.comprehensive:
        success = asyncio.run(run_comprehensive_test_suite())
        exit(0 if success else 1)
    else:
        # Run basic functionality test
        print("Running basic evolution system test...")

        async def basic_test():
            orchestrator = await initialize_evolution_system(
                evolution_data_path="test_evolution_data",
                population_size=4,
                auto_evolution=False
            )

            async with orchestrator.orchestration_context():
                results = await quick_evolution_cycle(orchestrator, generations=1)

                if results['success']:
                    print("✅ Basic evolution test passed!")
                    return True
                else:
                    print("❌ Basic evolution test failed!")
                    return False

        success = asyncio.run(basic_test())
        exit(0 if success else 1)
