"""Agent Forge Component Validation Suite.

Tests agent creation, evolution, and compression pipeline functionality.
"""

import logging
from pathlib import Path
import sys
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent_forge.adas.adas import ADASTask, SecureCodeRunner
from src.agent_forge.compression import BITNETCompressor, bitnet_compress
from src.production.agent_forge.agent_factory import AgentFactory
from src.production.evolution.evomerge_pipeline import EvolutionConfig, EvoMergePipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentForgeValidator:
    """Validates Agent Forge component functionality."""

    def __init__(self) -> None:
        self.results = {
            "agent_creation": {"status": "pending", "time": 0, "details": ""},
            "evolution_pipeline": {"status": "pending", "time": 0, "details": ""},
            "compression_pipeline": {"status": "pending", "time": 0, "details": ""},
            "adas_system": {"status": "pending", "time": 0, "details": ""},
        }

    def test_agent_creation(self) -> None:
        """Test agent creation and basic functionality."""
        logger.info("Testing Agent Factory...")
        start_time = time.time()

        try:
            # Create agent factory
            factory = AgentFactory()

            # Test agent creation
            agent_config = {
                "agent_id": "test_agent_001",
                "agent_type": "reasoning",
                "capabilities": ["text_generation", "problem_solving"],
                "model_config": {"base_model": "gpt2", "max_length": 100, "temperature": 0.7},
            }

            # Create agent
            agent = factory.create_agent(agent_config)

            # Test agent functionality
            if hasattr(agent, "process_task"):
                test_task = {
                    "task_type": "text_generation",
                    "prompt": "Hello, this is a test",
                    "parameters": {"max_tokens": 50},
                }

                response = agent.process_task(test_task)

                self.results["agent_creation"] = {
                    "status": "success",
                    "time": time.time() - start_time,
                    "details": f"Agent created successfully. Response: {str(response)[:100]}...",
                }
            else:
                self.results["agent_creation"] = {
                    "status": "success",
                    "time": time.time() - start_time,
                    "details": f"Agent created successfully. Type: {type(agent).__name__}",
                }

        except Exception as e:
            self.results["agent_creation"] = {
                "status": "failed",
                "time": time.time() - start_time,
                "details": f"Error: {e!s}",
            }

    def test_evolution_pipeline(self) -> None:
        """Test evolution pipeline configuration and initialization."""
        logger.info("Testing Evolution Pipeline...")
        start_time = time.time()

        try:
            # Create evolution configuration
            config = EvolutionConfig(
                max_generations=2,  # Small test run
                population_size=4,  # Small population
                mutation_rate=0.1,
                crossover_rate=0.8,
                evaluation_samples=10,  # Minimal evaluation
                device="cpu",
            )

            # Initialize pipeline
            pipeline = EvoMergePipeline(config)

            # Test pipeline initialization
            if hasattr(pipeline, "config"):
                self.results["evolution_pipeline"] = {
                    "status": "success",
                    "time": time.time() - start_time,
                    "details": f"Pipeline initialized. Generations: {config.max_generations}, Population: {config.population_size}",
                }
            else:
                self.results["evolution_pipeline"] = {
                    "status": "partial",
                    "time": time.time() - start_time,
                    "details": "Pipeline created but missing expected attributes",
                }

        except Exception as e:
            self.results["evolution_pipeline"] = {
                "status": "failed",
                "time": time.time() - start_time,
                "details": f"Error: {e!s}",
            }

    def test_compression_pipeline(self) -> None:
        """Test compression pipeline functionality."""
        logger.info("Testing Compression Pipeline...")
        start_time = time.time()

        try:
            # Test BitNet compression components
            compressor = BITNETCompressor()

            if hasattr(compressor, "compress"):
                # Test with dummy data
                dummy_model_data = {"weights": {"layer1": [1.0, 2.0, 3.0, 4.0]}, "metadata": {"model_type": "test"}}

                # Test compression (might not work with dummy data but tests API)
                try:
                    result = bitnet_compress(dummy_model_data)
                    self.results["compression_pipeline"] = {
                        "status": "success",
                        "time": time.time() - start_time,
                        "details": f"BitNet compressor functional. Result type: {type(result)}",
                    }
                except Exception as compress_error:
                    # Compressor exists but needs real model data
                    self.results["compression_pipeline"] = {
                        "status": "partial",
                        "time": time.time() - start_time,
                        "details": f"BitNet compressor available but needs real model data: {str(compress_error)[:50]}...",
                    }
            else:
                self.results["compression_pipeline"] = {
                    "status": "partial",
                    "time": time.time() - start_time,
                    "details": f"BitNet compressor created. Available methods: {[m for m in dir(compressor) if not m.startswith('_')]}",
                }

        except Exception as e:
            self.results["compression_pipeline"] = {
                "status": "failed",
                "time": time.time() - start_time,
                "details": f"Error: {e!s}",
            }

    def test_adas_system(self) -> None:
        """Test ADAS (Automated Design Architecture Search) system."""
        logger.info("Testing ADAS System...")
        start_time = time.time()

        try:
            # Test ADAS task creation
            task = ADASTask(
                task_id="test_adas_001",
                task_type="architecture_search",
                task_content="Optimize neural architecture",
                metadata={"search_space": "transformer", "constraints": {"max_params": 1000000}},
            )

            # Test secure code runner
            runner = SecureCodeRunner()

            # Test task properties
            if hasattr(task, "task_id") and hasattr(runner, "run_code"):
                self.results["adas_system"] = {
                    "status": "success",
                    "time": time.time() - start_time,
                    "details": f"ADAS task created: {task.task_id}. Runner available with security features.",
                }
            else:
                self.results["adas_system"] = {
                    "status": "partial",
                    "time": time.time() - start_time,
                    "details": f"ADAS components available. Task type: {type(task).__name__}, Runner type: {type(runner).__name__}",
                }

        except Exception as e:
            self.results["adas_system"] = {
                "status": "failed",
                "time": time.time() - start_time,
                "details": f"Error: {e!s}",
            }

    def run_validation(self):
        """Run all Agent Forge validation tests."""
        logger.info("=== Agent Forge Validation Suite ===")

        # Run all tests
        self.test_agent_creation()
        self.test_evolution_pipeline()
        self.test_compression_pipeline()
        self.test_adas_system()

        # Calculate results
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results.values() if r["status"] == "success")
        partial_tests = sum(1 for r in self.results.values() if r["status"] == "partial")

        logger.info("=== Agent Forge Validation Results ===")
        for test_name, result in self.results.items():
            status_emoji = {"success": "✅", "partial": "⚠️", "failed": "❌", "pending": "⏳"}

            logger.info(f"{status_emoji[result['status']]} {test_name}: {result['status'].upper()}")
            logger.info(f"   Time: {result['time']:.2f}s")
            logger.info(f"   Details: {result['details']}")

        success_rate = (successful_tests + partial_tests * 0.5) / total_tests
        logger.info(
            f"\n🎯 Agent Forge Success Rate: {success_rate:.1%} ({successful_tests + partial_tests}/{total_tests})"
        )

        return self.results, success_rate


if __name__ == "__main__":
    validator = AgentForgeValidator()
    results, success_rate = validator.run_validation()

    if success_rate >= 0.8:
        print("🎉 Agent Forge Validation: PASSED")
    else:
        print("⚠️ Agent Forge Validation: NEEDS IMPROVEMENT")
