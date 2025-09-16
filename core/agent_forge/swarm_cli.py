#!/usr/bin/env python3
"""
Agent Forge Swarm CLI

Command-line interface for initializing and managing the Agent Forge swarm system.
Provides comprehensive control over the 8-phase pipeline execution with specialized
agent coordination, monitoring, and quality gate enforcement.

Usage:
    python swarm_cli.py init --topology hierarchical --max-agents 50
    python swarm_cli.py execute --phases 3,4,5 --monitor
    python swarm_cli.py status --detailed
    python swarm_cli.py remediate --phase 3 --theater-detection
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Import swarm components
from .swarm_coordinator import SwarmCoordinator, SwarmConfig, SwarmTopology
from .swarm_execution import SwarmExecutionManager, create_and_execute_swarm
from .swarm_monitor import SwarmMonitor, create_swarm_monitor
from agent_forge.unified_pipeline import UnifiedPipeline, UnifiedConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('agent_forge_swarm.log')
    ]
)

logger = logging.getLogger("SwarmCLI")


class SwarmCLI:
    """Main CLI controller for Agent Forge swarm system."""

    def __init__(self):
        self.coordinator = None
        self.execution_manager = None
        self.monitor = None
        self.logger = logging.getLogger("SwarmCLI")

    async def initialize_swarm(self, topology: str = "hierarchical",
                             max_agents: int = 50,
                             config_file: Optional[str] = None) -> bool:
        """Initialize the Agent Forge swarm system."""
        try:
            self.logger.info(f"Initializing Agent Forge swarm (topology: {topology}, agents: {max_agents})")

            # Load configuration
            if config_file and Path(config_file).exists():
                with open(config_file) as f:
                    config_data = json.load(f)
            else:
                config_data = {}

            # Create swarm configuration
            swarm_config = SwarmConfig(
                topology=SwarmTopology(topology),
                max_agents=max_agents,
                **config_data
            )

            # Initialize coordinator
            self.coordinator = SwarmCoordinator(swarm_config)
            success = await self.coordinator.initialize_swarm()

            if not success:
                self.logger.error("Failed to initialize swarm coordinator")
                return False

            # Initialize execution manager
            self.execution_manager = SwarmExecutionManager(self.coordinator)

            # Initialize monitoring system
            self.monitor = create_swarm_monitor(self.coordinator)

            self.logger.info("Agent Forge swarm initialization completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Swarm initialization failed: {str(e)}")
            return False

    async def execute_phases(self, phases: List[int],
                           enable_monitoring: bool = True,
                           initial_model_path: Optional[str] = None) -> Dict[str, any]:
        """Execute specified phases with swarm coordination."""
        if not self.coordinator:
            raise RuntimeError("Swarm not initialized. Run 'init' command first.")

        try:
            self.logger.info(f"Executing phases: {phases}")

            # Start monitoring if enabled
            if enable_monitoring and self.monitor:
                await self.monitor.start_monitoring()

            # Prepare initial data
            initial_data = {"model": None}
            if initial_model_path:
                # Load model from path (implementation would load actual model)
                self.logger.info(f"Loading initial model from: {initial_model_path}")

            # Execute phases
            results = []
            current_data = initial_data

            for phase in phases:
                self.logger.info(f"Starting Phase {phase} execution")

                result = await self.execution_manager.execute_pipeline_phase(phase, current_data)
                results.append(result)

                # Print phase summary
                self._print_phase_summary(phase, result)

                if result.success:
                    # Prepare data for next phase
                    current_data = {
                        "model": result.model,
                        "previous_phase_result": result,
                        "pipeline_state": self.coordinator.memory
                    }
                else:
                    self.logger.error(f"Phase {phase} failed, stopping execution")
                    break

            # Stop monitoring
            if enable_monitoring and self.monitor:
                await self.monitor.stop_monitoring()

            # Generate execution report
            execution_summary = self._generate_execution_summary(phases, results)

            return {
                "success": all(r.success for r in results),
                "phases_executed": len(results),
                "phases_successful": sum(1 for r in results if r.success),
                "results": results,
                "execution_summary": execution_summary
            }

        except Exception as e:
            self.logger.error(f"Phase execution failed: {str(e)}")
            return {"success": False, "error": str(e)}

    async def remediate_phase(self, phase: int,
                            theater_detection: bool = True,
                            deep_analysis: bool = False) -> Dict[str, any]:
        """Remediate issues in a specific phase."""
        if not self.coordinator:
            raise RuntimeError("Swarm not initialized. Run 'init' command first.")

        try:
            self.logger.info(f"Starting Phase {phase} remediation")

            # Get phase data from memory
            phase_data = self.coordinator.memory.get("phase_states", {}).get(phase, {})

            if not phase_data:
                self.logger.warning(f"No existing data for Phase {phase}, using defaults")
                phase_data = {"phase": phase, "remediation_mode": True}

            # Run theater detection if enabled
            theater_result = None
            if theater_detection and self.monitor:
                self.logger.info("Running theater detection analysis")
                theater_result = await self.monitor.run_theater_detection(phase_data)
                self._print_theater_analysis(theater_result)

            # Execute remediation with specialized handling
            if phase == 3:
                # Phase 3: Quiet-STaR remediation
                remediation_data = {
                    **phase_data,
                    "remediation_mode": True,
                    "theater_analysis": theater_result,
                    "deep_analysis": deep_analysis
                }

                result = await self.execution_manager.execute_pipeline_phase(phase, remediation_data)

            else:
                # Generic remediation for other phases
                remediation_data = {
                    **phase_data,
                    "remediation_mode": True,
                    "theater_analysis": theater_result
                }

                result = await self.execution_manager.execute_pipeline_phase(phase, remediation_data)

            # Print remediation summary
            self._print_remediation_summary(phase, result, theater_result)

            return {
                "success": result.success,
                "phase": phase,
                "theater_detected": theater_result.get("theater_detected", False) if theater_result else False,
                "remediation_result": result,
                "theater_analysis": theater_result
            }

        except Exception as e:
            self.logger.error(f"Phase {phase} remediation failed: {str(e)}")
            return {"success": False, "error": str(e)}

    async def get_status(self, detailed: bool = False) -> Dict[str, any]:
        """Get comprehensive swarm system status."""
        if not self.coordinator:
            return {"error": "Swarm not initialized"}

        try:
            # Get coordinator status
            coordinator_status = await self.coordinator.get_swarm_status()

            # Get execution manager status
            execution_status = None
            if self.execution_manager:
                execution_status = await self.execution_manager.get_execution_status()

            # Get monitoring status
            monitoring_status = None
            if self.monitor:
                monitoring_status = await self.monitor.get_monitoring_status()

            status = {
                "swarm_initialized": True,
                "coordinator": coordinator_status,
                "execution_manager": execution_status is not None,
                "monitoring": monitoring_status,
                "pipeline_state": {
                    "completed_phases": list(self.coordinator.memory.get("phase_states", {}).keys()),
                    "total_memory_entries": len(self.coordinator.memory)
                }
            }

            if detailed:
                status.update({
                    "detailed_execution": execution_status,
                    "memory_dump": self.coordinator.memory,
                    "agent_configurations": {
                        agent_id: {
                            "role": agent.config.role.value,
                            "phase": agent.config.phase,
                            "state": agent.state,
                            "memory_mb": agent.config.max_memory,
                            "timeout": agent.config.timeout_seconds
                        }
                        for agent_id, agent in self.coordinator.agents.items()
                    }
                })

            return status

        except Exception as e:
            self.logger.error(f"Status retrieval failed: {str(e)}")
            return {"error": str(e)}

    async def run_quality_gates(self, phase: int,
                              phase_data: Optional[Dict[str, any]] = None) -> Dict[str, any]:
        """Run quality gates for a specific phase."""
        if not self.monitor:
            raise RuntimeError("Monitoring system not initialized")

        try:
            if phase_data is None:
                # Get phase data from memory
                phase_data = self.coordinator.memory.get("phase_states", {}).get(phase, {})

            self.logger.info(f"Running quality gates for Phase {phase}")
            result = await self.monitor.validate_quality_gates(phase, phase_data)

            self._print_quality_gate_results(phase, result)
            return result

        except Exception as e:
            self.logger.error(f"Quality gate validation failed: {str(e)}")
            return {"error": str(e)}

    def _print_phase_summary(self, phase: int, result):
        """Print phase execution summary."""
        status = "✓ PASSED" if result.success else "✗ FAILED"
        duration = result.duration_seconds

        print(f"\n{'='*60}")
        print(f"Phase {phase} Execution Summary")
        print(f"{'='*60}")
        print(f"Status: {status}")
        print(f"Duration: {duration:.2f} seconds")

        if result.metrics:
            print("\nMetrics:")
            for key, value in result.metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")

        if not result.success and result.error:
            print(f"\nError: {result.error}")

        print(f"{'='*60}\n")

    def _print_theater_analysis(self, theater_result: Dict[str, any]):
        """Print theater detection analysis."""
        print(f"\n{'='*60}")
        print("Theater Detection Analysis")
        print(f"{'='*60}")

        if theater_result.get("theater_detected", False):
            print("🚨 PERFORMANCE THEATER DETECTED")
            print(f"Theater Score: {theater_result.get('theater_score', 0):.3f}")
            print(f"Confidence: {theater_result.get('confidence', 0):.3f}")
        else:
            print("✓ No theater detected")

        if "indicators" in theater_result:
            print("\nIndicators:")
            for category, analysis in theater_result["indicators"].items():
                score = analysis.get("score", 0)
                status = "⚠️ DETECTED" if score > 0.5 else "✓ Clear"
                print(f"  {category}: {status} (score: {score:.3f})")

        if "recommendations" in theater_result and theater_result["recommendations"]:
            print("\nRecommendations:")
            for rec in theater_result["recommendations"]:
                print(f"  • {rec}")

        print(f"{'='*60}\n")

    def _print_remediation_summary(self, phase: int, result, theater_result):
        """Print remediation summary."""
        print(f"\n{'='*60}")
        print(f"Phase {phase} Remediation Summary")
        print(f"{'='*60}")

        if theater_result:
            theater_status = "DETECTED" if theater_result.get("theater_detected") else "NOT DETECTED"
            print(f"Theater Status: {theater_status}")

        remediation_status = "✓ SUCCESSFUL" if result.success else "✗ FAILED"
        print(f"Remediation: {remediation_status}")
        print(f"Duration: {result.duration_seconds:.2f} seconds")

        print(f"{'='*60}\n")

    def _print_quality_gate_results(self, phase: int, result: Dict[str, any]):
        """Print quality gate validation results."""
        print(f"\n{'='*60}")
        print(f"Phase {phase} Quality Gates")
        print(f"{'='*60}")

        overall_status = "✓ PASSED" if result.get("all_gates_passed") else "✗ FAILED"
        print(f"Overall Status: {overall_status}")
        print(f"Gates Passed: {result.get('gates_passed', 0)}/{result.get('gates_validated', 0)}")

        if "gate_results" in result:
            print("\nIndividual Gates:")
            for gate_name, gate_result in result["gate_results"].items():
                status = "✓ PASSED" if gate_result.get("passed") else "✗ FAILED"
                print(f"  {gate_name}: {status}")

                if not gate_result.get("passed") and "failure_reasons" in gate_result:
                    for reason in gate_result["failure_reasons"]:
                        print(f"    - {reason}")

        if "blocking_failures" in result and result["blocking_failures"]:
            print(f"\nBlocking Failures: {', '.join(result['blocking_failures'])}")

        print(f"{'='*60}\n")

    def _generate_execution_summary(self, phases: List[int], results) -> Dict[str, any]:
        """Generate comprehensive execution summary."""
        total_duration = sum(r.duration_seconds for r in results)
        successful_phases = [i for i, r in enumerate(results) if r.success]

        summary = {
            "total_phases": len(phases),
            "successful_phases": len(successful_phases),
            "failed_phases": len(phases) - len(successful_phases),
            "total_duration": total_duration,
            "average_phase_duration": total_duration / len(results) if results else 0,
            "phase_breakdown": {}
        }

        for i, (phase, result) in enumerate(zip(phases, results)):
            summary["phase_breakdown"][phase] = {
                "success": result.success,
                "duration": result.duration_seconds,
                "error": result.error if not result.success else None
            }

        return summary


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Agent Forge Swarm Management CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Initialize command
    init_parser = subparsers.add_parser("init", help="Initialize the swarm system")
    init_parser.add_argument("--topology", choices=["hierarchical", "mesh", "star", "ring"],
                           default="hierarchical", help="Swarm topology")
    init_parser.add_argument("--max-agents", type=int, default=50, help="Maximum number of agents")
    init_parser.add_argument("--config", help="Configuration file path")

    # Execute command
    execute_parser = subparsers.add_parser("execute", help="Execute pipeline phases")
    execute_parser.add_argument("--phases", required=True, help="Comma-separated phase numbers (e.g., 3,4,5)")
    execute_parser.add_argument("--monitor", action="store_true", help="Enable monitoring")
    execute_parser.add_argument("--initial-model", help="Path to initial model")

    # Remediate command
    remediate_parser = subparsers.add_parser("remediate", help="Remediate phase issues")
    remediate_parser.add_argument("--phase", type=int, required=True, help="Phase to remediate")
    remediate_parser.add_argument("--theater-detection", action="store_true",
                                help="Enable theater detection")
    remediate_parser.add_argument("--deep-analysis", action="store_true",
                                help="Enable deep analysis")

    # Status command
    status_parser = subparsers.add_parser("status", help="Get swarm status")
    status_parser.add_argument("--detailed", action="store_true", help="Show detailed status")

    # Quality gates command
    gates_parser = subparsers.add_parser("gates", help="Run quality gates")
    gates_parser.add_argument("--phase", type=int, required=True, help="Phase to validate")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize CLI
    cli = SwarmCLI()

    try:
        if args.command == "init":
            success = await cli.initialize_swarm(
                topology=args.topology,
                max_agents=args.max_agents,
                config_file=args.config
            )
            if success:
                print("✓ Swarm initialization completed successfully")
            else:
                print("✗ Swarm initialization failed")
                return 1

        elif args.command == "execute":
            # Initialize swarm first
            await cli.initialize_swarm()

            phases = [int(p.strip()) for p in args.phases.split(",")]
            result = await cli.execute_phases(
                phases=phases,
                enable_monitoring=args.monitor,
                initial_model_path=args.initial_model
            )

            if result["success"]:
                print(f"✓ Pipeline execution completed: {result['phases_successful']}/{result['phases_executed']} phases successful")
            else:
                print(f"✗ Pipeline execution failed: {result.get('error', 'Unknown error')}")
                return 1

        elif args.command == "remediate":
            # Initialize swarm first
            await cli.initialize_swarm()

            result = await cli.remediate_phase(
                phase=args.phase,
                theater_detection=args.theater_detection,
                deep_analysis=args.deep_analysis
            )

            if result["success"]:
                print(f"✓ Phase {args.phase} remediation completed successfully")
            else:
                print(f"✗ Phase {args.phase} remediation failed: {result.get('error')}")
                return 1

        elif args.command == "status":
            # Initialize swarm first if not done
            await cli.initialize_swarm()

            status = await cli.get_status(detailed=args.detailed)

            if "error" in status:
                print(f"✗ Status retrieval failed: {status['error']}")
                return 1

            print(json.dumps(status, indent=2, default=str))

        elif args.command == "gates":
            # Initialize swarm first
            await cli.initialize_swarm()

            result = await cli.run_quality_gates(phase=args.phase)

            if "error" in result:
                print(f"✗ Quality gate validation failed: {result['error']}")
                return 1

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))