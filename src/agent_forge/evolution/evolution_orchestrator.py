"""Evolution Orchestrator - Coordinates the Self-Evolving Agent Ecosystem

Main orchestration system that manages the 18-agent ecosystem evolution,
coordinates between components, and ensures stable autonomous improvement.
"""

import asyncio
import json
import logging
import signal
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from .agent_evolution_engine import AgentEvolutionEngine, AgentGenome, AgentKPIs
from .evolution_dashboard import EvolutionDashboard, PerformanceAnalyzer
from .meta_learning_engine import MetaLearningEngine
from .safe_code_modifier import CodeModification, SafeCodeModifier

logger = logging.getLogger(__name__)


@dataclass
class OrchestrationConfig:
    """Configuration for evolution orchestration"""

    evolution_interval_hours: int = 24
    monitoring_interval_minutes: int = 15
    emergency_rollback_threshold: float = 0.3  # Fitness drop threshold
    max_concurrent_modifications: int = 3
    backup_retention_days: int = 30
    auto_evolution_enabled: bool = True
    safety_mode: bool = True
    max_population_size: int = 18
    min_population_size: int = 12


@dataclass
class OrchestrationState:
    """Current state of the orchestration system"""

    is_running: bool = False
    last_evolution_time: datetime | None = None
    current_generation: int = 0
    active_modifications: int = 0
    emergency_mode: bool = False
    last_health_check: datetime | None = None
    performance_trend: str = "stable"
    total_uptime: float = 0.0


class HealthMonitor:
    """Monitors system health and triggers emergency responses"""

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.health_history = []
        self.alert_thresholds = {
            "fitness_drop": 0.3,
            "error_rate": 0.1,
            "response_time": 10.0,
            "memory_usage": 0.9,
        }

    async def check_system_health(self) -> dict[str, Any]:
        """Comprehensive system health check"""
        health_status = {
            "timestamp": datetime.now(),
            "overall_healthy": True,
            "alerts": [],
            "metrics": {},
        }

        try:
            # Check evolution engine health
            dashboard_data = (
                await self.orchestrator.evolution_engine.get_evolution_dashboard_data()
            )

            # Fitness health
            current_fitness = dashboard_data["population_stats"]["avg_fitness"]
            if len(self.health_history) > 0:
                prev_fitness = self.health_history[-1]["metrics"].get("avg_fitness", 0)
                fitness_drop = prev_fitness - current_fitness

                if fitness_drop > self.alert_thresholds["fitness_drop"]:
                    health_status["alerts"].append(
                        {
                            "type": "fitness_drop",
                            "severity": "high",
                            "message": f"Fitness dropped by {fitness_drop:.3f}",
                            "value": fitness_drop,
                        }
                    )
                    health_status["overall_healthy"] = False

            # Population diversity
            diversity = dashboard_data["population_stats"]["diversity"]
            if diversity < 0.3:  # Low diversity threshold
                health_status["alerts"].append(
                    {
                        "type": "low_diversity",
                        "severity": "medium",
                        "message": f"Population diversity is low: {diversity:.3f}",
                        "value": diversity,
                    }
                )

            # Active modifications
            active_mods = len(
                [
                    mod
                    for mod in self.orchestrator.code_modifier.modifications.values()
                    if not mod.applied
                ]
            )
            if active_mods > self.orchestrator.config.max_concurrent_modifications:
                health_status["alerts"].append(
                    {
                        "type": "too_many_modifications",
                        "severity": "medium",
                        "message": f"Too many active modifications: {active_mods}",
                        "value": active_mods,
                    }
                )

            # Update metrics
            health_status["metrics"] = {
                "avg_fitness": current_fitness,
                "max_fitness": dashboard_data["population_stats"]["max_fitness"],
                "diversity": diversity,
                "total_agents": dashboard_data["population_stats"]["total_agents"],
                "active_modifications": active_mods,
                "current_generation": dashboard_data["population_stats"][
                    "current_generation"
                ],
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_status["overall_healthy"] = False
            health_status["alerts"].append(
                {
                    "type": "health_check_error",
                    "severity": "high",
                    "message": f"Health check failed: {e!s}",
                    "value": None,
                }
            )

        # Store health history
        self.health_history.append(health_status)
        if len(self.health_history) > 100:  # Keep last 100 records
            self.health_history = self.health_history[-100:]

        return health_status

    async def handle_emergency(self, alert: dict[str, Any]):
        """Handle emergency situations"""
        logger.warning(f"Handling emergency: {alert}")

        if alert["type"] == "fitness_drop" and alert["severity"] == "high":
            # Emergency rollback
            logger.info("Triggering emergency rollback due to fitness drop")
            success = await self.orchestrator.evolution_engine.emergency_rollback(
                generations_back=1
            )

            if success:
                logger.info("Emergency rollback successful")
                self.orchestrator.state.emergency_mode = False
            else:
                logger.error("Emergency rollback failed - entering emergency mode")
                self.orchestrator.state.emergency_mode = True

        elif alert["type"] == "too_many_modifications":
            # Cancel pending modifications
            logger.info("Cancelling excessive modifications")
            await self.orchestrator._cancel_pending_modifications()

        # Pause auto-evolution during emergencies
        if alert["severity"] == "high":
            self.orchestrator.config.auto_evolution_enabled = False
            logger.info("Auto-evolution disabled due to emergency")


class TaskScheduler:
    """Manages scheduled tasks for evolution orchestration"""

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.scheduler = AsyncIOScheduler()
        self.scheduled_jobs = {}

    def setup_schedules(self):
        """Setup scheduled tasks"""
        # Evolution cycle
        if self.orchestrator.config.auto_evolution_enabled:
            evolution_job = self.scheduler.add_job(
                self.orchestrator._run_scheduled_evolution,
                IntervalTrigger(
                    hours=self.orchestrator.config.evolution_interval_hours
                ),
                id="evolution_cycle",
                name="Evolution Cycle",
            )
            self.scheduled_jobs["evolution_cycle"] = evolution_job

        # Health monitoring
        health_job = self.scheduler.add_job(
            self.orchestrator._run_health_check,
            IntervalTrigger(
                minutes=self.orchestrator.config.monitoring_interval_minutes
            ),
            id="health_check",
            name="Health Check",
        )
        self.scheduled_jobs["health_check"] = health_job

        # Daily maintenance
        maintenance_job = self.scheduler.add_job(
            self.orchestrator._run_daily_maintenance,
            CronTrigger(hour=2, minute=0),  # 2 AM daily
            id="daily_maintenance",
            name="Daily Maintenance",
        )
        self.scheduled_jobs["daily_maintenance"] = maintenance_job

        # Backup cleanup
        backup_job = self.scheduler.add_job(
            self.orchestrator._cleanup_old_backups,
            CronTrigger(hour=3, minute=30, day_of_week=0),  # Weekly on Sunday
            id="backup_cleanup",
            name="Backup Cleanup",
        )
        self.scheduled_jobs["backup_cleanup"] = backup_job

    def start(self):
        """Start the scheduler"""
        self.scheduler.start()
        logger.info("Task scheduler started")

    def stop(self):
        """Stop the scheduler"""
        self.scheduler.shutdown()
        logger.info("Task scheduler stopped")

    def pause_job(self, job_id: str):
        """Pause a specific job"""
        if job_id in self.scheduled_jobs:
            self.scheduler.pause_job(job_id)
            logger.info(f"Paused job: {job_id}")

    def resume_job(self, job_id: str):
        """Resume a specific job"""
        if job_id in self.scheduled_jobs:
            self.scheduler.resume_job(job_id)
            logger.info(f"Resumed job: {job_id}")


class EvolutionOrchestrator:
    """Main orchestration system for the self-evolving agent ecosystem"""

    def __init__(
        self,
        config: OrchestrationConfig | None = None,
        storage_path: str = "evolution_data",
    ):
        self.config = config or OrchestrationConfig()
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize state
        self.state = OrchestrationState()
        self.start_time = datetime.now()

        # Initialize components
        self.evolution_engine = AgentEvolutionEngine(
            evolution_data_path=str(self.storage_path),
            population_size=self.config.max_population_size,
        )

        self.code_modifier = SafeCodeModifier(
            backup_path=str(self.storage_path / "backups")
        )

        self.meta_learning_engine = MetaLearningEngine(
            storage_path=str(self.storage_path / "meta_learning")
        )

        self.dashboard = EvolutionDashboard(self.evolution_engine, port=5000)

        self.performance_analyzer = PerformanceAnalyzer(self.evolution_engine)

        # Initialize monitoring and scheduling
        self.health_monitor = HealthMonitor(self)
        self.task_scheduler = TaskScheduler(self)

        # Thread pool for concurrent operations
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Active tasks tracking
        self.active_tasks: set[asyncio.Task] = set()

        logger.info("Evolution Orchestrator initialized")

    async def start(self):
        """Start the orchestration system"""
        if self.state.is_running:
            logger.warning("Orchestrator is already running")
            return

        logger.info("Starting Evolution Orchestrator...")

        try:
            # Initialize agent population
            await self.evolution_engine.initialize_population()

            # Setup signal handlers
            self._setup_signal_handlers()

            # Start task scheduler
            self.task_scheduler.setup_schedules()
            self.task_scheduler.start()

            # Initial health check
            await self._run_health_check()

            # Update state
            self.state.is_running = True
            self.state.last_health_check = datetime.now()

            logger.info("Evolution Orchestrator started successfully")

            # Start background tasks
            await self._start_background_tasks()

        except Exception as e:
            logger.error(f"Failed to start orchestrator: {e}")
            await self.stop()
            raise

    async def stop(self):
        """Stop the orchestration system"""
        logger.info("Stopping Evolution Orchestrator...")

        # Update state
        self.state.is_running = False

        # Cancel active tasks
        for task in self.active_tasks:
            if not task.done():
                task.cancel()

        if self.active_tasks:
            await asyncio.gather(*self.active_tasks, return_exceptions=True)

        # Stop scheduler
        self.task_scheduler.stop()

        # Save state
        await self._save_orchestration_state()

        # Shutdown thread pool
        self.executor.shutdown(wait=True)

        logger.info("Evolution Orchestrator stopped")

    async def trigger_evolution(
        self, generations: int = 1, force: bool = False
    ) -> dict[str, Any]:
        """Manually trigger evolution cycle"""
        if not self.state.is_running and not force:
            raise RuntimeError("Orchestrator is not running")

        if self.state.emergency_mode and not force:
            raise RuntimeError("System is in emergency mode")

        logger.info(f"Triggering evolution cycle: {generations} generations")

        try:
            # Create evaluation tasks
            evaluation_tasks = await self._create_evaluation_tasks()

            # Run evolution
            results = await self.evolution_engine.run_evolution_cycle(
                generations=generations, evaluation_tasks=evaluation_tasks
            )

            # Update state
            self.state.last_evolution_time = datetime.now()
            self.state.current_generation = self.evolution_engine.current_generation

            # Analyze results
            analysis = await self._analyze_evolution_results(results)

            # Update performance trend
            self._update_performance_trend(results)

            logger.info(
                f"Evolution cycle completed. Best fitness: {max(results['best_fitness_history']):.4f}"
            )

            return {
                "success": True,
                "results": results,
                "analysis": analysis,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Evolution cycle failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    async def apply_safe_modification(
        self,
        agent_id: str,
        modification_type: str,
        description: str,
        code_transformer: Callable[[str], str],
        file_path: str,
    ) -> dict[str, Any]:
        """Apply safe code modification to an agent"""
        if self.state.active_modifications >= self.config.max_concurrent_modifications:
            raise RuntimeError("Too many concurrent modifications")

        if self.state.emergency_mode:
            raise RuntimeError("Modifications disabled in emergency mode")

        try:
            self.state.active_modifications += 1

            # Propose modification
            modification = await self.code_modifier.propose_modification(
                agent_id=agent_id,
                file_path=file_path,
                modification_type=modification_type,
                description=description,
                code_transformer=code_transformer,
            )

            # Test modification
            test_results = await self.code_modifier.test_modification(
                modification.modification_id
            )

            # Apply if safe
            if (
                modification.safety_score >= 0.8
                and test_results.get("success", False)
                and not self.config.safety_mode
            ):
                success = await self.code_modifier.apply_modification(
                    modification.modification_id
                )

                if success:
                    logger.info(f"Applied modification {modification.modification_id}")

                    # Record for meta-learning
                    await self._record_modification_outcome(modification, success)

                    return {
                        "success": True,
                        "modification_id": modification.modification_id,
                        "safety_score": modification.safety_score,
                    }

            return {
                "success": False,
                "modification_id": modification.modification_id,
                "safety_score": modification.safety_score,
                "reason": "Safety threshold not met or testing failed",
            }

        finally:
            self.state.active_modifications = max(
                0, self.state.active_modifications - 1
            )

    async def get_orchestration_status(self) -> dict[str, Any]:
        """Get comprehensive orchestration status"""
        # Calculate uptime
        uptime = (datetime.now() - self.start_time).total_seconds()

        # Get latest health status
        latest_health = (
            self.health_monitor.health_history[-1]
            if self.health_monitor.health_history
            else None
        )

        # Get evolution status
        dashboard_data = await self.evolution_engine.get_evolution_dashboard_data()

        # Get meta-learning stats
        meta_report = await self.meta_learning_engine.generate_meta_learning_report()

        return {
            "orchestrator": {
                "is_running": self.state.is_running,
                "uptime_seconds": uptime,
                "emergency_mode": self.state.emergency_mode,
                "last_evolution": self.state.last_evolution_time.isoformat()
                if self.state.last_evolution_time
                else None,
                "current_generation": self.state.current_generation,
                "active_modifications": self.state.active_modifications,
                "performance_trend": self.state.performance_trend,
            },
            "evolution": dashboard_data["population_stats"],
            "health": latest_health,
            "meta_learning": {
                "total_experiences": meta_report["total_experiences"],
                "active_agents": meta_report["active_agents"],
                "learning_trends": meta_report.get("learning_trends", {}),
            },
            "config": asdict(self.config),
            "timestamp": datetime.now().isoformat(),
        }

    async def _start_background_tasks(self):
        """Start background monitoring tasks"""
        # Performance monitoring task
        monitor_task = asyncio.create_task(self._performance_monitor_loop())
        self.active_tasks.add(monitor_task)

        # Meta-learning optimization task
        meta_task = asyncio.create_task(self._meta_learning_loop())
        self.active_tasks.add(meta_task)

        # Cleanup task
        cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.active_tasks.add(cleanup_task)

    async def _performance_monitor_loop(self):
        """Background performance monitoring"""
        while self.state.is_running:
            try:
                # Monitor agent performance
                dashboard_data = (
                    await self.evolution_engine.get_evolution_dashboard_data()
                )

                # Record KPIs for agents with low performance
                for agent_id, fitness in dashboard_data["fitness_scores"].items():
                    if fitness < 0.5:  # Low performance threshold
                        # Create KPI record
                        kpis = AgentKPIs(
                            agent_id=agent_id,
                            task_success_rate=fitness,
                            user_satisfaction=fitness * 0.9,  # Approximation
                            resource_efficiency=np.random.uniform(0.3, 0.8),
                            adaptation_speed=np.random.uniform(0.2, 0.7),
                        )

                        self.evolution_engine.kpi_tracker.record_kpis(kpis)

                await asyncio.sleep(300)  # 5 minutes

            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(60)

    async def _meta_learning_loop(self):
        """Background meta-learning optimization"""
        while self.state.is_running:
            try:
                # Optimize learning for active agents
                dashboard_data = (
                    await self.evolution_engine.get_evolution_dashboard_data()
                )

                for agent_id, fitness in dashboard_data["fitness_scores"].items():
                    # Optimize learning strategy
                    task_characteristics = {
                        "difficulty": 1.0
                        - fitness,  # Higher difficulty for lower fitness
                        "data_size": 1000,
                        "task_complexity": "medium",
                    }

                    await self.meta_learning_engine.optimize_agent_learning(
                        agent_id=agent_id,
                        task_type="general",
                        task_characteristics=task_characteristics,
                        current_performance=fitness,
                    )

                # Save meta-learning data
                self.meta_learning_engine.save_meta_learning_data()

                await asyncio.sleep(1800)  # 30 minutes

            except Exception as e:
                logger.error(f"Meta-learning optimization error: {e}")
                await asyncio.sleep(300)

    async def _cleanup_loop(self):
        """Background cleanup tasks"""
        while self.state.is_running:
            try:
                # Cleanup old sandbox environments
                await self.code_modifier.sandbox.cleanup_old_sandboxes(max_age_hours=24)

                # Cleanup old backups
                await self.code_modifier.cleanup_old_backups(
                    max_age_days=self.config.backup_retention_days
                )

                await asyncio.sleep(3600)  # 1 hour

            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(300)

    async def _run_scheduled_evolution(self):
        """Scheduled evolution cycle"""
        if not self.config.auto_evolution_enabled or self.state.emergency_mode:
            return

        logger.info("Running scheduled evolution cycle")
        await self.trigger_evolution(generations=1)

    async def _run_health_check(self):
        """Scheduled health check"""
        health_status = await self.health_monitor.check_system_health()

        # Handle alerts
        for alert in health_status["alerts"]:
            if alert["severity"] == "high":
                await self.health_monitor.handle_emergency(alert)

        self.state.last_health_check = datetime.now()

    async def _run_daily_maintenance(self):
        """Daily maintenance tasks"""
        logger.info("Running daily maintenance")

        try:
            # Generate evolution report
            report = await self.performance_analyzer.generate_evolution_report()

            # Save report
            report_file = (
                self.storage_path
                / f"daily_report_{datetime.now().strftime('%Y%m%d')}.json"
            )
            with open(report_file, "w") as f:
                json.dump(report, f, indent=2, default=str)

            # Optimize agent population
            if report["performance_analysis"]["performance_distribution"]["mean"] < 0.4:
                logger.info(
                    "Low performance detected - triggering optimization evolution"
                )
                await self.trigger_evolution(generations=2, force=True)

        except Exception as e:
            logger.error(f"Daily maintenance failed: {e}")

    async def _cleanup_old_backups(self):
        """Weekly backup cleanup"""
        logger.info("Running weekly backup cleanup")
        await self.code_modifier.cleanup_old_backups(
            max_age_days=self.config.backup_retention_days
        )

    async def _create_evaluation_tasks(self) -> list[Callable]:
        """Create evaluation tasks for evolution"""

        async def fitness_evaluation_task(genome: AgentGenome) -> float:
            """Evaluate agent fitness"""
            try:
                # Get recent performance data
                fitness_scores = self.evolution_engine.kpi_tracker.get_fitness_scores(
                    [genome.agent_id]
                )
                return fitness_scores.get(
                    genome.agent_id, 0.5
                )  # Default to medium fitness
            except Exception:
                return 0.5

        async def specialization_task(genome: AgentGenome) -> float:
            """Evaluate specialization effectiveness"""
            try:
                spec_focus = genome.specialization_config.get("focus_areas", {})
                if spec_focus:
                    return sum(spec_focus.values()) / len(spec_focus)
                return 0.5
            except Exception:
                return 0.5

        async def collaboration_task(genome: AgentGenome) -> float:
            """Evaluate collaboration capability"""
            try:
                collab_weight = genome.behavior_weights.get("collaboration", 0.5)
                return min(1.0, collab_weight + np.random.normal(0, 0.1))
            except Exception:
                return 0.5

        return [fitness_evaluation_task, specialization_task, collaboration_task]

    async def _analyze_evolution_results(
        self, results: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze evolution results"""
        analysis = {
            "improvement": 0.0,
            "diversity_change": 0.0,
            "convergence_rate": 0.0,
            "recommendations": [],
        }

        try:
            fitness_history = results["best_fitness_history"]
            diversity_history = results["diversity_history"]

            if len(fitness_history) > 1:
                analysis["improvement"] = fitness_history[-1] - fitness_history[0]
                analysis["convergence_rate"] = analysis["improvement"] / len(
                    fitness_history
                )

            if len(diversity_history) > 1:
                analysis["diversity_change"] = (
                    diversity_history[-1] - diversity_history[0]
                )

            # Generate recommendations
            if analysis["improvement"] < 0.05:
                analysis["recommendations"].append(
                    "Low improvement - consider increasing mutation rate"
                )

            if analysis["diversity_change"] < -0.1:
                analysis["recommendations"].append(
                    "Diversity decreasing - add diversity preservation"
                )

        except Exception as e:
            logger.error(f"Evolution analysis failed: {e}")

        return analysis

    def _update_performance_trend(self, results: dict[str, Any]):
        """Update performance trend based on results"""
        try:
            fitness_history = results["best_fitness_history"]

            if len(fitness_history) >= 3:
                recent_trend = fitness_history[-1] - fitness_history[-3]

                if recent_trend > 0.05:
                    self.state.performance_trend = "improving"
                elif recent_trend < -0.05:
                    self.state.performance_trend = "declining"
                else:
                    self.state.performance_trend = "stable"
        except Exception:
            self.state.performance_trend = "unknown"

    async def _record_modification_outcome(
        self, modification: CodeModification, success: bool
    ):
        """Record modification outcome for meta-learning"""
        try:
            # Record learning outcome
            await self.meta_learning_engine.record_learning_outcome(
                agent_id=modification.agent_id,
                task_type=modification.modification_type,
                initial_performance=0.5,  # Would be actual performance
                final_performance=0.8 if success else 0.3,
                learning_config={"modification_type": modification.modification_type},
                learning_time=60.0,  # Would be actual time
                convergence_steps=1,
            )
        except Exception as e:
            logger.error(f"Failed to record modification outcome: {e}")

    async def _cancel_pending_modifications(self):
        """Cancel excessive pending modifications"""
        pending_mods = [
            mod for mod in self.code_modifier.modifications.values() if not mod.applied
        ]

        # Keep only the highest priority modifications
        sorted_mods = sorted(pending_mods, key=lambda x: x.safety_score, reverse=True)

        for mod in sorted_mods[self.config.max_concurrent_modifications:]:
            # Remove from tracking
            if mod.modification_id in self.code_modifier.modifications:
                del self.code_modifier.modifications[mod.modification_id]

        logger.info(
            f"Cancelled {len(sorted_mods) - self.config.max_concurrent_modifications} pending modifications"
        )

    def _setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers"""

        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum} - initiating graceful shutdown")
            asyncio.create_task(self.stop())

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def _save_orchestration_state(self):
        """Save orchestration state to disk"""
        try:
            state_file = self.storage_path / "orchestration_state.json"
            state_data = {
                "state": asdict(self.state),
                "config": asdict(self.config),
                "timestamp": datetime.now().isoformat(),
            }

            with open(state_file, "w") as f:
                json.dump(state_data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Failed to save orchestration state: {e}")

    async def _load_orchestration_state(self):
        """Load orchestration state from disk"""
        try:
            state_file = self.storage_path / "orchestration_state.json"

            if state_file.exists():
                with open(state_file) as f:
                    state_data = json.load(f)

                # Restore state (selective restore to avoid conflicts)
                if "state" in state_data:
                    self.state.current_generation = state_data["state"].get(
                        "current_generation", 0
                    )
                    self.state.performance_trend = state_data["state"].get(
                        "performance_trend", "stable"
                    )

        except Exception as e:
            logger.error(f"Failed to load orchestration state: {e}")

    @asynccontextmanager
    async def orchestration_context(self):
        """Context manager for orchestration lifecycle"""
        try:
            await self.start()
            yield self
        finally:
            await self.stop()


if __name__ == "__main__":

    async def main():
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        # Create orchestrator
        config = OrchestrationConfig(
            evolution_interval_hours=6,  # More frequent for demo
            monitoring_interval_minutes=5,
            auto_evolution_enabled=True,
            safety_mode=False,  # Allow modifications for demo
        )

        orchestrator = EvolutionOrchestrator(config)

        # Run orchestrator
        async with orchestrator.orchestration_context():
            logger.info("Evolution Orchestrator is running...")
            logger.info("Access dashboard at http://localhost:5000")

            # Keep running until interrupted
            try:
                while True:
                    status = await orchestrator.get_orchestration_status()
                    logger.info(
                        f"Status: Generation {status['orchestrator']['current_generation']}, "
                        f"Fitness {status['evolution']['avg_fitness']:.3f}"
                    )
                    await asyncio.sleep(60)

            except KeyboardInterrupt:
                logger.info("Shutdown requested")

    asyncio.run(main())
