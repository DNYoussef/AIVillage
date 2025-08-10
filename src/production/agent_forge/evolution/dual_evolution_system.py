"""Dual Evolution System - Master coordinator for nightly + breakthrough evolution."""

import asyncio
import contextlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from .base import EvolvableAgent
from .evolution_metrics import EvolutionMetricsCollector
from .evolution_scheduler import EvolutionScheduler, SchedulerConfig
from .magi_architectural_evolution import MagiArchitecturalEvolution
from .nightly_evolution_orchestrator import NightlyEvolutionOrchestrator

logger = logging.getLogger(__name__)


@dataclass
class EvolutionSchedule:
    """Evolution scheduling configuration."""

    nightly_time_utc: str = "02:00"  # 2 AM UTC for nightly evolution
    breakthrough_interval_days: int = 7  # Weekly breakthrough attempts
    emergency_threshold: float = (
        0.3  # Performance drop threshold for emergency evolution
    )
    cooling_period_hours: int = 4  # Minimum time between evolutions
    max_concurrent_evolutions: int = 3  # Max agents evolving simultaneously


@dataclass
class EvolutionEvent:
    """Record of an evolution event."""

    timestamp: float
    agent_id: str
    evolution_type: str  # "nightly", "breakthrough", "emergency"
    trigger_reason: str
    pre_evolution_kpis: dict[str, float]
    post_evolution_kpis: dict[str, float] | None = None
    success: bool = False
    duration_seconds: float = 0
    generation_change: int = 0
    insights: list[str] = field(default_factory=list)


class DualEvolutionSystem:
    """Master coordinator for dual evolution: nightly incremental + breakthrough discoveries."""

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}
        self.schedule = EvolutionSchedule(**self.config.get("schedule", {}))

        # Evolution orchestrators
        self.nightly_orchestrator = NightlyEvolutionOrchestrator(
            self.config.get("nightly_config", {})
        )
        self.magi_evolution = MagiArchitecturalEvolution(
            self.config.get("magi_config", {})
        )
        self.metrics_collector = EvolutionMetricsCollector(
            self.config.get("metrics_config", {})
        )
        self.scheduler = EvolutionScheduler(
            SchedulerConfig(**self.config.get("scheduler", {}))
        )

        # Agent registry
        self.registered_agents: dict[str, EvolvableAgent] = {}
        self.agent_metadata: dict[str, dict[str, Any]] = {}

        # Evolution state
        self.evolution_history: list[EvolutionEvent] = []
        self.active_evolutions: dict[str, EvolutionEvent] = {}
        self.last_nightly_run: float | None = None
        self.last_breakthrough_run: float | None = None

        # Control flags
        self.system_active = False
        self.emergency_mode = False
        self.maintenance_task: asyncio.Task | None = None

        logger.info("Dual Evolution System initialized")

    def register_agent(
        self, agent: EvolvableAgent, metadata: dict | None = None
    ) -> None:
        """Register agent for evolution management."""
        self.registered_agents[agent.agent_id] = agent
        self.agent_metadata[agent.agent_id] = metadata or {}

        # Initialize agent monitoring
        self.agent_metadata[agent.agent_id].update(
            {
                "registration_time": time.time(),
                "last_evolution": None,
                "evolution_count": 0,
                "performance_trend": "stable",
                "priority_level": metadata.get("priority", "medium"),
            }
        )

        logger.info(f"Registered agent {agent.agent_id} for evolution")

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister agent from evolution management."""
        if agent_id in self.registered_agents:
            del self.registered_agents[agent_id]
            del self.agent_metadata[agent_id]
            logger.info(f"Unregistered agent {agent_id}")

    async def start_system(self) -> None:
        """Start the dual evolution system."""
        if self.system_active:
            logger.warning("Evolution system already active")
            return

        self.system_active = True

        # Start background maintenance
        self.maintenance_task = asyncio.create_task(self._maintenance_loop())

        # Initialize metrics collection
        await self.metrics_collector.start()

        logger.info("Dual Evolution System started")

    async def stop_system(self) -> None:
        """Stop the dual evolution system."""
        self.system_active = False

        if self.maintenance_task:
            self.maintenance_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.maintenance_task

        await self.metrics_collector.stop()

        logger.info("Dual Evolution System stopped")

    async def _maintenance_loop(self) -> None:
        """Main system maintenance loop."""
        while self.system_active:
            try:
                # Check for scheduled evolutions
                await self._check_scheduled_evolutions()

                # Monitor agent performance for emergency evolution
                await self._monitor_agent_performance()

                # Update metrics
                await self._update_system_metrics()

                # Cleanup completed evolutions
                self._cleanup_evolution_history()

                # Sleep until next check
                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logger.exception(f"Error in evolution maintenance loop: {e}")
                await asyncio.sleep(60)  # Sleep on error

    async def _check_scheduled_evolutions(self) -> None:
        """Check if scheduled evolutions should run."""
        current_time = time.time()
        current_utc = datetime.utcfromtimestamp(current_time)

        # Check nightly evolution
        if self._should_run_nightly(current_utc, current_time):
            await self._run_nightly_evolution()

        # Check breakthrough evolution
        if self._should_run_breakthrough(current_time):
            await self._run_breakthrough_evolution()

    def _should_run_nightly(self, current_utc: datetime, current_time: float) -> bool:
        """Check if nightly evolution should run."""
        # Parse scheduled time
        hour, minute = map(int, self.schedule.nightly_time_utc.split(":"))
        scheduled_time = current_utc.replace(
            hour=hour, minute=minute, second=0, microsecond=0
        )

        # Check if we've crossed the scheduled time since last run
        if self.last_nightly_run:
            last_run_date = datetime.utcfromtimestamp(self.last_nightly_run).date()
            if current_utc.date() <= last_run_date:
                return False  # Already ran today

        # Check if current time is past scheduled time
        return current_utc >= scheduled_time

    def _should_run_breakthrough(self, current_time: float) -> bool:
        """Check if breakthrough evolution should run."""
        if not self.last_breakthrough_run:
            return True  # First run

        days_since_last = (current_time - self.last_breakthrough_run) / 86400
        return days_since_last >= self.schedule.breakthrough_interval_days

    async def _run_nightly_evolution(self) -> None:
        """Run nightly incremental evolution."""
        logger.info("Starting nightly evolution cycle")
        self.last_nightly_run = time.time()

        # Select agents for nightly evolution
        candidates = await self._select_nightly_candidates()

        if not candidates:
            logger.info("No agents selected for nightly evolution")
            return

        # Run evolution in batches to respect concurrency limits
        for batch in self._create_evolution_batches(candidates):
            evolution_tasks = []

            for agent_id in batch:
                agent = self.registered_agents[agent_id]
                task = self._evolve_agent_nightly(agent)
                evolution_tasks.append(task)

            # Wait for batch to complete
            await asyncio.gather(*evolution_tasks, return_exceptions=True)

        await self.metrics_collector.record_system_event(
            "nightly_evolution_completed",
            {"agents_evolved": len(candidates), "timestamp": time.time()},
        )

        logger.info(f"Nightly evolution completed for {len(candidates)} agents")

    async def _run_breakthrough_evolution(self) -> None:
        """Run breakthrough architectural evolution."""
        logger.info("Starting breakthrough evolution cycle")
        self.last_breakthrough_run = time.time()

        # Select top candidates for breakthrough evolution
        candidates = await self._select_breakthrough_candidates()

        if not candidates:
            logger.info("No agents selected for breakthrough evolution")
            return

        # Run breakthrough evolution (typically fewer agents, more intensive)
        results = []
        for agent_id in candidates[:3]:  # Limit breakthrough evolutions
            agent = self.registered_agents[agent_id]
            result = await self._evolve_agent_breakthrough(agent)
            results.append(result)

        successful_breakthroughs = sum(1 for r in results if r)

        await self.metrics_collector.record_system_event(
            "breakthrough_evolution_completed",
            {
                "candidates": len(candidates),
                "successful": successful_breakthroughs,
                "timestamp": time.time(),
            },
        )

        logger.info(
            f"Breakthrough evolution completed: {successful_breakthroughs}/{len(candidates)} successful"
        )

    async def _monitor_agent_performance(self) -> None:
        """Monitor agent performance for emergency evolution triggers."""
        current_time = time.time()

        for agent_id, agent in list(self.registered_agents.items()):
            try:
                kpis = agent.evaluate_kpi()
                performance = kpis.get("performance", 0.5)

                # Scheduler-driven actions
                action = self.scheduler.get_action(agent)
                if action == "retire":
                    logger.info(f"Retiring agent {agent_id} based on KPIs")
                    self.unregister_agent(agent_id)
                    continue
                if action == "evolve" and not self._agent_in_cooling_period(
                    agent_id, current_time
                ):
                    await self._evolve_agent_nightly(agent)

                # Check for emergency evolution trigger
                if (
                    performance < self.schedule.emergency_threshold
                    and not self._agent_in_cooling_period(agent_id, current_time)
                ):
                    logger.warning(
                        f"Emergency evolution triggered for agent {agent_id} "
                        f"(performance: {performance:.2f})"
                    )

                    await self._evolve_agent_emergency(agent)

                # Update performance trends
                self._update_performance_trend(agent_id, performance)

            except Exception as e:
                logger.exception(f"Error monitoring agent {agent_id}: {e}")

    def _agent_in_cooling_period(self, agent_id: str, current_time: float) -> bool:
        """Check if agent is in post-evolution cooling period."""
        metadata = self.agent_metadata.get(agent_id, {})
        last_evolution = metadata.get("last_evolution")

        if not last_evolution:
            return False

        hours_since_evolution = (current_time - last_evolution) / 3600
        return hours_since_evolution < self.schedule.cooling_period_hours

    def _update_performance_trend(
        self, agent_id: str, current_performance: float
    ) -> None:
        """Update agent performance trend."""
        metadata = self.agent_metadata.get(agent_id, {})

        # Simple trend tracking (could be enhanced with more sophisticated analysis)
        recent_performances = metadata.get("recent_performances", [])
        recent_performances.append(current_performance)

        # Keep only last 10 measurements
        recent_performances = recent_performances[-10:]
        metadata["recent_performances"] = recent_performances

        # Calculate trend
        if len(recent_performances) >= 3:
            early_avg = sum(recent_performances[:3]) / 3
            late_avg = sum(recent_performances[-3:]) / 3

            if late_avg > early_avg + 0.05:
                trend = "improving"
            elif late_avg < early_avg - 0.05:
                trend = "declining"
            else:
                trend = "stable"

            metadata["performance_trend"] = trend

    async def _select_nightly_candidates(self, max_agents: int = 10) -> list[str]:
        """Select agents for nightly evolution."""
        candidates = []

        for agent_id, agent in self.registered_agents.items():
            # Skip agents in cooling period or actively evolving
            if (
                self._agent_in_cooling_period(agent_id, time.time())
                or agent_id in self.active_evolutions
            ):
                continue

            # Check if agent needs evolution
            if agent.needs_evolution():
                readiness_score = agent.get_evolution_readiness_score()
                candidates.append((agent_id, readiness_score))

        # Sort by readiness score and return top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [agent_id for agent_id, _ in candidates[:max_agents]]

    async def _select_breakthrough_candidates(self, max_agents: int = 5) -> list[str]:
        """Select agents for breakthrough evolution."""
        candidates = []

        for agent_id, agent in self.registered_agents.items():
            metadata = self.agent_metadata[agent_id]

            # Skip recent evolutions
            if self._agent_in_cooling_period(agent_id, time.time()):
                continue

            # Prioritize high-performing agents with stable trends
            kpis = agent.evaluate_kpi()
            performance = kpis.get("performance", 0.5)
            trend = metadata.get("performance_trend", "stable")

            # Good candidates for breakthrough: high performance, stable or improving
            if performance > 0.7 and trend in ["stable", "improving"]:
                # Calculate breakthrough potential
                potential = performance * 0.7 + (0.3 if trend == "improving" else 0.2)
                candidates.append((agent_id, potential))

        # Sort by potential and return top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [agent_id for agent_id, _ in candidates[:max_agents]]

    def _create_evolution_batches(self, agent_ids: list[str]) -> list[list[str]]:
        """Create batches of agents respecting concurrency limits."""
        batches = []
        batch_size = self.schedule.max_concurrent_evolutions

        for i in range(0, len(agent_ids), batch_size):
            batch = agent_ids[i : i + batch_size]
            batches.append(batch)

        return batches

    async def _evolve_agent_nightly(self, agent: EvolvableAgent) -> bool:
        """Perform nightly evolution on agent."""
        evolution_event = EvolutionEvent(
            timestamp=time.time(),
            agent_id=agent.agent_id,
            evolution_type="nightly",
            trigger_reason="scheduled_nightly",
            pre_evolution_kpis=agent.evaluate_kpi(),
        )

        self.active_evolutions[agent.agent_id] = evolution_event

        try:
            # Record pre-evolution state
            await self.metrics_collector.record_evolution_start(evolution_event)

            # Perform nightly evolution
            success = await self.nightly_orchestrator.evolve_agent(agent)

            # Record results
            evolution_event.success = success
            evolution_event.duration_seconds = time.time() - evolution_event.timestamp
            evolution_event.post_evolution_kpis = agent.evaluate_kpi()

            if success:
                # Update agent metadata
                self.agent_metadata[agent.agent_id]["last_evolution"] = time.time()
                self.agent_metadata[agent.agent_id]["evolution_count"] += 1

                logger.info(f"Nightly evolution successful for agent {agent.agent_id}")
            else:
                logger.warning(f"Nightly evolution failed for agent {agent.agent_id}")

            await self.metrics_collector.record_evolution_completion(evolution_event)
            return success

        except Exception as e:
            logger.exception(
                f"Error in nightly evolution for agent {agent.agent_id}: {e}"
            )
            evolution_event.success = False
            evolution_event.insights.append(f"Evolution failed: {e!s}")
            return False

        finally:
            # Clean up active evolution
            if agent.agent_id in self.active_evolutions:
                del self.active_evolutions[agent.agent_id]

            # Store in history
            self.evolution_history.append(evolution_event)

    async def _evolve_agent_breakthrough(self, agent: EvolvableAgent) -> bool:
        """Perform breakthrough evolution on agent."""
        evolution_event = EvolutionEvent(
            timestamp=time.time(),
            agent_id=agent.agent_id,
            evolution_type="breakthrough",
            trigger_reason="scheduled_breakthrough",
            pre_evolution_kpis=agent.evaluate_kpi(),
        )

        self.active_evolutions[agent.agent_id] = evolution_event

        try:
            # Record pre-evolution state
            await self.metrics_collector.record_evolution_start(evolution_event)

            # Perform breakthrough evolution
            breakthrough_result = await self.magi_evolution.evolve_agent(agent)

            success = breakthrough_result.get("success", False)
            generation_jump = breakthrough_result.get("generation_increase", 0)

            # Record results
            evolution_event.success = success
            evolution_event.duration_seconds = time.time() - evolution_event.timestamp
            evolution_event.post_evolution_kpis = agent.evaluate_kpi()
            evolution_event.generation_change = generation_jump
            evolution_event.insights = breakthrough_result.get("insights", [])

            if success:
                # Update agent metadata
                self.agent_metadata[agent.agent_id]["last_evolution"] = time.time()
                self.agent_metadata[agent.agent_id]["evolution_count"] += 1

                logger.info(
                    f"Breakthrough evolution successful for agent {agent.agent_id} "
                    f"(+{generation_jump} generations)"
                )
            else:
                logger.warning(
                    f"Breakthrough evolution failed for agent {agent.agent_id}"
                )

            await self.metrics_collector.record_evolution_completion(evolution_event)
            return success

        except Exception as e:
            logger.exception(
                f"Error in breakthrough evolution for agent {agent.agent_id}: {e}"
            )
            evolution_event.success = False
            evolution_event.insights.append(f"Breakthrough evolution failed: {e!s}")
            return False

        finally:
            # Clean up active evolution
            if agent.agent_id in self.active_evolutions:
                del self.active_evolutions[agent.agent_id]

            # Store in history
            self.evolution_history.append(evolution_event)

    async def _evolve_agent_emergency(self, agent: EvolvableAgent) -> bool:
        """Perform emergency evolution on agent."""
        evolution_event = EvolutionEvent(
            timestamp=time.time(),
            agent_id=agent.agent_id,
            evolution_type="emergency",
            trigger_reason="performance_degradation",
            pre_evolution_kpis=agent.evaluate_kpi(),
        )

        self.active_evolutions[agent.agent_id] = evolution_event

        try:
            # For emergency evolution, try nightly first, then breakthrough if needed
            logger.info(f"Starting emergency evolution for agent {agent.agent_id}")

            # Try nightly evolution first (faster)
            nightly_success = await self.nightly_orchestrator.evolve_agent(agent)

            if nightly_success:
                # Check if performance improved sufficiently
                new_kpis = agent.evaluate_kpi()
                new_performance = new_kpis.get("performance", 0.5)

                if new_performance > self.schedule.emergency_threshold + 0.1:
                    # Success with nightly evolution
                    evolution_event.success = True
                    evolution_event.insights.append(
                        "Emergency resolved with nightly evolution"
                    )
                else:
                    # Try breakthrough evolution
                    logger.info(
                        f"Nightly evolution insufficient, trying breakthrough for {agent.agent_id}"
                    )
                    breakthrough_result = await self.magi_evolution.evolve_agent(agent)
                    evolution_event.success = breakthrough_result.get("success", False)
                    evolution_event.insights.extend(
                        breakthrough_result.get("insights", [])
                    )
            else:
                # Nightly failed, try breakthrough
                logger.info(
                    f"Nightly evolution failed, trying breakthrough for {agent.agent_id}"
                )
                breakthrough_result = await self.magi_evolution.evolve_agent(agent)
                evolution_event.success = breakthrough_result.get("success", False)
                evolution_event.insights.extend(breakthrough_result.get("insights", []))

            # Record results
            evolution_event.duration_seconds = time.time() - evolution_event.timestamp
            evolution_event.post_evolution_kpis = agent.evaluate_kpi()

            if evolution_event.success:
                self.agent_metadata[agent.agent_id]["last_evolution"] = time.time()
                self.agent_metadata[agent.agent_id]["evolution_count"] += 1
                logger.info(
                    f"Emergency evolution successful for agent {agent.agent_id}"
                )
            else:
                logger.error(f"Emergency evolution failed for agent {agent.agent_id}")

            await self.metrics_collector.record_evolution_completion(evolution_event)
            return evolution_event.success

        except Exception as e:
            logger.exception(
                f"Error in emergency evolution for agent {agent.agent_id}: {e}"
            )
            evolution_event.success = False
            evolution_event.insights.append(f"Emergency evolution failed: {e!s}")
            return False

        finally:
            # Clean up active evolution
            if agent.agent_id in self.active_evolutions:
                del self.active_evolutions[agent.agent_id]

            # Store in history
            self.evolution_history.append(evolution_event)

    async def _update_system_metrics(self) -> None:
        """Update system-wide evolution metrics."""
        current_time = time.time()

        # System status metrics
        system_metrics = {
            "registered_agents": len(self.registered_agents),
            "active_evolutions": len(self.active_evolutions),
            "total_evolutions_today": len(
                [
                    e
                    for e in self.evolution_history
                    if current_time - e.timestamp < 86400
                ]
            ),
            "success_rate_24h": self._calculate_success_rate(86400),
            "avg_evolution_duration": self._calculate_avg_duration(),
            "system_uptime": current_time - (self.last_nightly_run or current_time),
        }

        await self.metrics_collector.record_system_metrics(system_metrics)

    def _calculate_success_rate(self, window_seconds: float) -> float:
        """Calculate evolution success rate in time window."""
        cutoff = time.time() - window_seconds
        recent_events = [e for e in self.evolution_history if e.timestamp > cutoff]

        if not recent_events:
            return 0.0

        successful = sum(1 for e in recent_events if e.success)
        return successful / len(recent_events)

    def _calculate_avg_duration(self) -> float:
        """Calculate average evolution duration."""
        if not self.evolution_history:
            return 0.0

        durations = [e.duration_seconds for e in self.evolution_history[-100:]]
        return sum(durations) / len(durations)

    def _cleanup_evolution_history(self) -> None:
        """Clean up old evolution history."""
        # Keep only last 1000 events
        if len(self.evolution_history) > 1000:
            self.evolution_history = self.evolution_history[-1000:]

    def get_system_status(self) -> dict[str, Any]:
        """Get current system status."""
        current_time = time.time()

        return {
            "system_active": self.system_active,
            "emergency_mode": self.emergency_mode,
            "registered_agents": len(self.registered_agents),
            "active_evolutions": list(self.active_evolutions.keys()),
            "last_nightly_run": self.last_nightly_run,
            "last_breakthrough_run": self.last_breakthrough_run,
            "evolution_history_size": len(self.evolution_history),
            "success_rate_24h": self._calculate_success_rate(86400),
            "next_nightly_time": self._get_next_nightly_time(),
            "next_breakthrough_time": self._get_next_breakthrough_time(),
            "system_metrics": {
                "avg_evolution_duration": self._calculate_avg_duration(),
                "total_evolutions": len(self.evolution_history),
                "emergency_evolutions_today": len(
                    [
                        e
                        for e in self.evolution_history
                        if (
                            current_time - e.timestamp < 86400
                            and e.evolution_type == "emergency"
                        )
                    ]
                ),
            },
        }

    def _get_next_nightly_time(self) -> float | None:
        """Get next scheduled nightly evolution time."""
        if not self.system_active:
            return None

        now = datetime.utcnow()
        hour, minute = map(int, self.schedule.nightly_time_utc.split(":"))

        # Calculate next occurrence
        next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if next_run <= now:
            next_run += timedelta(days=1)

        return next_run.timestamp()

    def _get_next_breakthrough_time(self) -> float | None:
        """Get next scheduled breakthrough evolution time."""
        if not self.system_active or not self.last_breakthrough_run:
            return None

        return self.last_breakthrough_run + (
            self.schedule.breakthrough_interval_days * 86400
        )

    async def force_evolution(
        self, agent_id: str, evolution_type: str = "nightly"
    ) -> bool:
        """Force evolution of specific agent."""
        if agent_id not in self.registered_agents:
            logger.error(f"Agent {agent_id} not registered")
            return False

        if agent_id in self.active_evolutions:
            logger.warning(f"Agent {agent_id} already evolving")
            return False

        agent = self.registered_agents[agent_id]

        if evolution_type == "nightly":
            return await self._evolve_agent_nightly(agent)
        if evolution_type == "breakthrough":
            return await self._evolve_agent_breakthrough(agent)
        if evolution_type == "emergency":
            return await self._evolve_agent_emergency(agent)
        logger.error(f"Unknown evolution type: {evolution_type}")
        return False

    def export_evolution_history(self, filepath: str | None = None) -> str:
        """Export evolution history to JSON file."""
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"evolution_history_{timestamp}.json"

        history_data = []
        for event in self.evolution_history:
            history_data.append(
                {
                    "timestamp": event.timestamp,
                    "agent_id": event.agent_id,
                    "evolution_type": event.evolution_type,
                    "trigger_reason": event.trigger_reason,
                    "success": event.success,
                    "duration_seconds": event.duration_seconds,
                    "generation_change": event.generation_change,
                    "pre_evolution_kpis": event.pre_evolution_kpis,
                    "post_evolution_kpis": event.post_evolution_kpis,
                    "insights": event.insights,
                }
            )

        with open(filepath, "w") as f:
            json.dump(
                {
                    "export_timestamp": time.time(),
                    "system_config": self.config,
                    "evolution_history": history_data,
                },
                f,
                indent=2,
            )

        logger.info(f"Evolution history exported to {filepath}")
        return filepath
