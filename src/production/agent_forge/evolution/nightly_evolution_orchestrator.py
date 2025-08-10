"""Nightly Evolution Orchestrator - Incremental agent improvements."""

from dataclasses import dataclass
import logging
import random
import time
from typing import Any

from .base import EvolvableAgent

logger = logging.getLogger(__name__)


@dataclass
class EvolutionStrategy:
    """Configuration for an evolution strategy."""

    name: str
    description: str
    target_performance_gain: float  # Expected improvement (0.0-1.0)
    risk_level: str  # "low", "medium", "high"
    success_probability: float  # Estimated success rate
    rollback_capable: bool  # Can rollback if evolution fails
    min_data_points: int  # Minimum performance records needed


class NightlyEvolutionOrchestrator:
    """Orchestrates incremental nightly evolution of agents."""

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}

        # Evolution strategies
        self.strategies = self._initialize_strategies()

        # Configuration
        self.max_evolution_time_minutes = self.config.get("max_evolution_time", 30)
        self.success_threshold = self.config.get(
            "success_threshold", 0.05
        )  # 5% improvement
        self.rollback_on_failure = self.config.get("rollback_on_failure", True)
        self.learning_rate_adjustment = self.config.get(
            "learning_rate_adjustment", True
        )

        # Evolution state
        self.evolution_results: dict[str, list[dict]] = {}
        self.strategy_effectiveness: dict[str, float] = {}

        logger.info("Nightly Evolution Orchestrator initialized")

    def _initialize_strategies(self) -> dict[str, EvolutionStrategy]:
        """Initialize available evolution strategies."""
        return {
            "parameter_tuning": EvolutionStrategy(
                name="parameter_tuning",
                description="Fine-tune existing parameters based on performance data",
                target_performance_gain=0.1,
                risk_level="low",
                success_probability=0.8,
                rollback_capable=True,
                min_data_points=50,
            ),
            "prompt_optimization": EvolutionStrategy(
                name="prompt_optimization",
                description="Optimize prompts based on successful patterns",
                target_performance_gain=0.15,
                risk_level="low",
                success_probability=0.7,
                rollback_capable=True,
                min_data_points=100,
            ),
            "learning_rate_adaptation": EvolutionStrategy(
                name="learning_rate_adaptation",
                description="Adapt learning rates based on recent performance trends",
                target_performance_gain=0.12,
                risk_level="medium",
                success_probability=0.65,
                rollback_capable=True,
                min_data_points=75,
            ),
            "confidence_threshold_tuning": EvolutionStrategy(
                name="confidence_threshold_tuning",
                description="Adjust confidence thresholds for better accuracy/coverage balance",
                target_performance_gain=0.08,
                risk_level="low",
                success_probability=0.75,
                rollback_capable=True,
                min_data_points=30,
            ),
            "specialization_enhancement": EvolutionStrategy(
                name="specialization_enhancement",
                description="Enhance specialization based on task performance patterns",
                target_performance_gain=0.2,
                risk_level="medium",
                success_probability=0.6,
                rollback_capable=False,
                min_data_points=200,
            ),
            "context_pattern_learning": EvolutionStrategy(
                name="context_pattern_learning",
                description="Learn and apply successful context patterns",
                target_performance_gain=0.18,
                risk_level="medium",
                success_probability=0.55,
                rollback_capable=True,
                min_data_points=150,
            ),
        }

    async def evolve_agent(self, agent: EvolvableAgent) -> bool:
        """Perform nightly evolution on an agent."""
        start_time = time.time()
        agent_id = agent.agent_id

        logger.info(f"Starting nightly evolution for agent {agent_id}")

        try:
            # Pre-evolution assessment
            pre_evolution_kpis = agent.evaluate_kpi()
            pre_evolution_state = agent.export_state()

            # Select best evolution strategy
            strategy = await self._select_evolution_strategy(agent)
            if not strategy:
                logger.warning(
                    f"No suitable evolution strategy found for agent {agent_id}"
                )
                return False

            logger.info(f"Selected strategy '{strategy.name}' for agent {agent_id}")

            # Apply evolution strategy
            evolution_result = await self._apply_evolution_strategy(agent, strategy)

            if not evolution_result["success"]:
                logger.warning(f"Evolution strategy failed for agent {agent_id}")
                if strategy.rollback_capable and self.rollback_on_failure:
                    await self._rollback_evolution(agent, pre_evolution_state)
                return False

            # Post-evolution assessment
            post_evolution_kpis = agent.evaluate_kpi()

            # Validate improvement
            improvement = self._calculate_improvement(
                pre_evolution_kpis, post_evolution_kpis
            )

            if improvement >= self.success_threshold:
                # Success - record results
                await self._record_evolution_success(
                    agent_id,
                    strategy,
                    pre_evolution_kpis,
                    post_evolution_kpis,
                    improvement,
                    evolution_result,
                )

                # Update strategy effectiveness
                self._update_strategy_effectiveness(strategy.name, True, improvement)

                logger.info(
                    f"Nightly evolution successful for agent {agent_id} "
                    f"(improvement: {improvement:.1%})"
                )
                return True
            # Insufficient improvement
            logger.warning(
                f"Insufficient improvement for agent {agent_id} "
                f"(improvement: {improvement:.1%}, required: {self.success_threshold:.1%})"
            )

            if strategy.rollback_capable and self.rollback_on_failure:
                await self._rollback_evolution(agent, pre_evolution_state)

            self._update_strategy_effectiveness(strategy.name, False, improvement)
            return False

        except Exception as e:
            logger.exception(
                f"Error during nightly evolution for agent {agent_id}: {e}"
            )
            return False

        finally:
            duration = time.time() - start_time
            logger.info(
                f"Nightly evolution completed for agent {agent_id} in {duration:.1f}s"
            )

    async def _select_evolution_strategy(
        self, agent: EvolvableAgent
    ) -> EvolutionStrategy | None:
        """Select the best evolution strategy for an agent."""
        # Get agent data
        performance_history = agent.performance_history
        agent.evaluate_kpi()

        # Filter strategies by data requirements
        viable_strategies = []
        for strategy in self.strategies.values():
            if len(performance_history) >= strategy.min_data_points:
                # Calculate strategy score
                score = await self._calculate_strategy_score(agent, strategy)
                viable_strategies.append((strategy, score))

        if not viable_strategies:
            return None

        # Sort by score and select best
        viable_strategies.sort(key=lambda x: x[1], reverse=True)
        selected_strategy, score = viable_strategies[0]

        logger.debug(
            f"Selected strategy {selected_strategy.name} with score {score:.2f}"
        )
        return selected_strategy

    async def _calculate_strategy_score(
        self, agent: EvolvableAgent, strategy: EvolutionStrategy
    ) -> float:
        """Calculate score for a strategy based on agent state and strategy effectiveness."""
        # Base score from strategy characteristics
        base_score = strategy.success_probability * strategy.target_performance_gain

        # Adjust based on historical effectiveness
        historical_effectiveness = self.strategy_effectiveness.get(strategy.name, 0.5)
        effectiveness_weight = 0.3
        score = (
            base_score * (1 - effectiveness_weight)
            + historical_effectiveness * effectiveness_weight
        )

        # Adjust based on agent characteristics
        current_kpis = agent.evaluate_kpi()

        # Prefer strategies that target agent's weak areas
        if (
            strategy.name == "parameter_tuning"
            and current_kpis.get("efficiency", 0.5) < 0.6
        ):
            score *= 1.2
        elif (
            strategy.name == "prompt_optimization"
            and current_kpis.get("accuracy", 0.7) < 0.7
        ):
            score *= 1.3
        elif (
            strategy.name == "confidence_threshold_tuning"
            and current_kpis.get("confidence", 0.5) < 0.5
        ):
            score *= 1.1
        elif (
            strategy.name == "specialization_enhancement"
            and agent.specialization_domain == "general"
        ):
            score *= 1.15

        # Risk adjustment based on agent stability
        reliability = current_kpis.get("reliability", 0.5)
        if strategy.risk_level == "high" and reliability < 0.7:
            score *= 0.8  # Reduce score for high-risk strategies on unstable agents
        elif strategy.risk_level == "low" and reliability > 0.8:
            score *= 1.1  # Boost low-risk strategies for stable agents

        return max(0.0, score)

    async def _apply_evolution_strategy(
        self, agent: EvolvableAgent, strategy: EvolutionStrategy
    ) -> dict[str, Any]:
        """Apply specific evolution strategy to agent."""
        strategy_name = strategy.name

        if strategy_name == "parameter_tuning":
            return await self._apply_parameter_tuning(agent, strategy)
        if strategy_name == "prompt_optimization":
            return await self._apply_prompt_optimization(agent, strategy)
        if strategy_name == "learning_rate_adaptation":
            return await self._apply_learning_rate_adaptation(agent, strategy)
        if strategy_name == "confidence_threshold_tuning":
            return await self._apply_confidence_threshold_tuning(agent, strategy)
        if strategy_name == "specialization_enhancement":
            return await self._apply_specialization_enhancement(agent, strategy)
        if strategy_name == "context_pattern_learning":
            return await self._apply_context_pattern_learning(agent, strategy)
        logger.error(f"Unknown evolution strategy: {strategy_name}")
        return {"success": False, "error": f"Unknown strategy: {strategy_name}"}

    async def _apply_parameter_tuning(
        self, agent: EvolvableAgent, strategy: EvolutionStrategy
    ) -> dict[str, Any]:
        """Apply parameter tuning evolution strategy."""
        try:
            # Analyze performance vs parameters
            parameter_analysis = await self._analyze_parameter_performance(agent)

            # Suggest parameter adjustments
            adjustments = await agent._suggest_parameter_adjustments()

            # Apply adjustments with small increments
            applied_changes = {}
            for param, adjustment in adjustments.items():
                if param in agent.parameters:
                    old_value = agent.parameters[param]

                    if adjustment == "increase":
                        # Increase by 10-20%
                        factor = 1.1 + random.uniform(0, 0.1)
                        new_value = old_value * factor
                    elif adjustment == "decrease":
                        # Decrease by 10-20%
                        factor = 0.8 + random.uniform(0, 0.1)
                        new_value = old_value * factor
                    else:
                        continue

                    agent.parameters[param] = new_value
                    applied_changes[param] = {
                        "old": old_value,
                        "new": new_value,
                        "change": adjustment,
                    }

            return {
                "success": True,
                "strategy": "parameter_tuning",
                "changes": applied_changes,
                "analysis": parameter_analysis,
            }

        except Exception as e:
            logger.exception(f"Parameter tuning failed: {e}")
            return {"success": False, "error": str(e)}

    async def _apply_prompt_optimization(
        self, agent: EvolvableAgent, strategy: EvolutionStrategy
    ) -> dict[str, Any]:
        """Apply prompt optimization evolution strategy."""
        try:
            # Analyze prompt effectiveness
            prompt_effectiveness = agent._analyze_prompt_effectiveness()

            # Identify prompts that need improvement
            improved_prompts = agent.design_successor_prompts()

            # Apply improvements
            changes = {}
            for prompt_name, new_prompt in improved_prompts.items():
                if prompt_name in agent.prompts:
                    old_prompt = agent.prompts[prompt_name]
                    if new_prompt != old_prompt:
                        agent.prompts[prompt_name] = new_prompt
                        changes[prompt_name] = {
                            "old_length": len(old_prompt),
                            "new_length": len(new_prompt),
                            "effectiveness": prompt_effectiveness.get(prompt_name, 0.5),
                        }

            return {
                "success": True,
                "strategy": "prompt_optimization",
                "changes": changes,
                "effectiveness_analysis": prompt_effectiveness,
            }

        except Exception as e:
            logger.exception(f"Prompt optimization failed: {e}")
            return {"success": False, "error": str(e)}

    async def _apply_learning_rate_adaptation(
        self, agent: EvolvableAgent, strategy: EvolutionStrategy
    ) -> dict[str, Any]:
        """Apply learning rate adaptation evolution strategy."""
        try:
            current_kpis = agent.evaluate_kpi()
            adaptability = current_kpis.get("adaptability", 0.5)
            reliability = current_kpis.get("reliability", 0.5)

            # Calculate optimal learning rate
            if adaptability < 0.4:  # Too slow to adapt
                learning_rate_factor = 1.2  # Increase learning rate
            elif reliability < 0.6:  # Too unstable
                learning_rate_factor = 0.8  # Decrease learning rate
            else:
                learning_rate_factor = 1.0 + random.uniform(
                    -0.1, 0.1
                )  # Small random adjustment

            # Apply learning rate changes
            changes = {}
            if "learning_rate" in agent.parameters:
                old_lr = agent.parameters["learning_rate"]
                new_lr = old_lr * learning_rate_factor
                agent.parameters["learning_rate"] = new_lr
                changes["learning_rate"] = {
                    "old": old_lr,
                    "new": new_lr,
                    "factor": learning_rate_factor,
                }

            return {
                "success": True,
                "strategy": "learning_rate_adaptation",
                "changes": changes,
                "reasoning": {
                    "adaptability": adaptability,
                    "reliability": reliability,
                    "factor": learning_rate_factor,
                },
            }

        except Exception as e:
            logger.exception(f"Learning rate adaptation failed: {e}")
            return {"success": False, "error": str(e)}

    async def _apply_confidence_threshold_tuning(
        self, agent: EvolvableAgent, strategy: EvolutionStrategy
    ) -> dict[str, Any]:
        """Apply confidence threshold tuning evolution strategy."""
        try:
            current_kpis = agent.evaluate_kpi()
            confidence = current_kpis.get("confidence", 0.5)
            accuracy = current_kpis.get("accuracy", 0.7)

            # Analyze confidence vs accuracy trade-off
            changes = {}

            if accuracy < 0.6 and confidence > 0.4:
                # Low accuracy, raise confidence threshold
                threshold_adjustment = 0.1
                new_threshold = min(0.9, confidence + threshold_adjustment)
                reasoning = "Raising threshold to improve accuracy"
            elif accuracy > 0.8 and confidence < 0.6:
                # High accuracy but low confidence, lower threshold
                threshold_adjustment = -0.05
                new_threshold = max(0.1, confidence + threshold_adjustment)
                reasoning = "Lowering threshold to increase confidence"
            else:
                # Fine-tune with small adjustment
                threshold_adjustment = random.uniform(-0.02, 0.02)
                new_threshold = max(0.1, min(0.9, confidence + threshold_adjustment))
                reasoning = "Fine-tuning threshold"

            # Apply threshold changes
            if "confidence_threshold" in agent.parameters:
                old_threshold = agent.parameters["confidence_threshold"]
                agent.parameters["confidence_threshold"] = new_threshold
                changes["confidence_threshold"] = {
                    "old": old_threshold,
                    "new": new_threshold,
                    "adjustment": threshold_adjustment,
                }

            return {
                "success": True,
                "strategy": "confidence_threshold_tuning",
                "changes": changes,
                "reasoning": reasoning,
                "current_metrics": {"confidence": confidence, "accuracy": accuracy},
            }

        except Exception as e:
            logger.exception(f"Confidence threshold tuning failed: {e}")
            return {"success": False, "error": str(e)}

    async def _apply_specialization_enhancement(
        self, agent: EvolvableAgent, strategy: EvolutionStrategy
    ) -> dict[str, Any]:
        """Apply specialization enhancement evolution strategy."""
        try:
            # Analyze task performance to identify specialization opportunities
            domain_knowledge = agent._extract_domain_knowledge()
            task_preferences = domain_knowledge.get("task_type_preferences", {})

            # Find best performing task types
            best_tasks = sorted(
                [
                    (task, info["success_rate"])
                    for task, info in task_preferences.items()
                    if info["confidence"] > 0.7
                ],
                key=lambda x: x[1],
                reverse=True,
            )

            changes = {}
            if best_tasks:
                # Update specialization based on best performance
                top_task, success_rate = best_tasks[0]

                if success_rate > 0.8 and agent.specialization_domain == "general":
                    # Specialize in the top-performing task type
                    agent.specialization_domain = top_task
                    if top_task not in agent.expertise_areas:
                        agent.expertise_areas.append(top_task)

                    changes["specialization"] = {
                        "old_domain": "general",
                        "new_domain": top_task,
                        "success_rate": success_rate,
                    }

                elif len(best_tasks) > 1:
                    # Add expertise areas
                    for task, rate in best_tasks[:3]:  # Top 3 tasks
                        if rate > 0.75 and task not in agent.expertise_areas:
                            agent.expertise_areas.append(task)

                    changes["expertise_areas"] = agent.expertise_areas

            return {
                "success": True,
                "strategy": "specialization_enhancement",
                "changes": changes,
                "task_analysis": task_preferences,
            }

        except Exception as e:
            logger.exception(f"Specialization enhancement failed: {e}")
            return {"success": False, "error": str(e)}

    async def _apply_context_pattern_learning(
        self, agent: EvolvableAgent, strategy: EvolutionStrategy
    ) -> dict[str, Any]:
        """Apply context pattern learning evolution strategy."""
        try:
            # Extract successful patterns from performance history
            strong_patterns = await agent._identify_strong_patterns()
            weak_spots = await agent._identify_weak_spots()

            # Update learned patterns
            changes = {}
            for pattern in strong_patterns:
                pattern_key = f"{pattern['pattern_type']}_{pattern['feature']}"
                if pattern_key not in agent.learned_patterns:
                    agent.learned_patterns[pattern_key] = {
                        "type": pattern["pattern_type"],
                        "feature": pattern["feature"],
                        "success_correlation": pattern["success_correlation"],
                        "learned_at": time.time(),
                    }
                    changes[pattern_key] = "added"

            # Remove patterns associated with weak spots
            for weak_spot in weak_spots:
                if weak_spot["area"] in agent.learned_patterns:
                    del agent.learned_patterns[weak_spot["area"]]
                    changes[weak_spot["area"]] = "removed"

            return {
                "success": True,
                "strategy": "context_pattern_learning",
                "changes": changes,
                "strong_patterns": len(strong_patterns),
                "weak_spots_addressed": len(
                    [w for w in weak_spots if w["area"] in changes]
                ),
            }

        except Exception as e:
            logger.exception(f"Context pattern learning failed: {e}")
            return {"success": False, "error": str(e)}

    async def _analyze_parameter_performance(
        self, agent: EvolvableAgent
    ) -> dict[str, Any]:
        """Analyze how parameters correlate with performance."""
        # This is a simplified analysis - in production would be more sophisticated
        analysis = {
            "data_points": len(agent.performance_history),
            "parameters_tracked": list(agent.parameters.keys()),
            "performance_variance": 0.0,
        }

        if len(agent.performance_history) > 10:
            recent_performance = [r.success for r in agent.performance_history[-50:]]
            analysis["recent_success_rate"] = sum(recent_performance) / len(
                recent_performance
            )

        return analysis

    def _calculate_improvement(
        self, pre_kpis: dict[str, float], post_kpis: dict[str, float]
    ) -> float:
        """Calculate overall improvement from KPI changes."""
        # Weight different KPIs
        weights = {
            "performance": 0.4,
            "accuracy": 0.3,
            "efficiency": 0.15,
            "reliability": 0.1,
            "adaptability": 0.05,
        }

        total_improvement = 0.0
        total_weight = 0.0

        for kpi, weight in weights.items():
            if kpi in pre_kpis and kpi in post_kpis:
                improvement = post_kpis[kpi] - pre_kpis[kpi]
                total_improvement += improvement * weight
                total_weight += weight

        return total_improvement / total_weight if total_weight > 0 else 0.0

    async def _rollback_evolution(
        self, agent: EvolvableAgent, pre_evolution_state: dict[str, Any]
    ) -> None:
        """Rollback agent to pre-evolution state."""
        try:
            # Restore agent state
            agent.parameters = pre_evolution_state["parameters"].copy()
            agent.prompts = pre_evolution_state["prompts"].copy()
            agent.specialization_domain = pre_evolution_state["specialization_domain"]
            agent.expertise_areas = pre_evolution_state["expertise_areas"].copy()

            logger.info(f"Rolled back evolution for agent {agent.agent_id}")

        except Exception as e:
            logger.exception(
                f"Failed to rollback evolution for agent {agent.agent_id}: {e}"
            )

    async def _record_evolution_success(
        self,
        agent_id: str,
        strategy: EvolutionStrategy,
        pre_kpis: dict[str, float],
        post_kpis: dict[str, float],
        improvement: float,
        evolution_result: dict[str, Any],
    ) -> None:
        """Record successful evolution."""
        if agent_id not in self.evolution_results:
            self.evolution_results[agent_id] = []

        result_record = {
            "timestamp": time.time(),
            "strategy": strategy.name,
            "pre_kpis": pre_kpis,
            "post_kpis": post_kpis,
            "improvement": improvement,
            "changes": evolution_result.get("changes", {}),
            "success": True,
        }

        self.evolution_results[agent_id].append(result_record)

        # Keep only recent results
        if len(self.evolution_results[agent_id]) > 100:
            self.evolution_results[agent_id] = self.evolution_results[agent_id][-100:]

    def _update_strategy_effectiveness(
        self, strategy_name: str, success: bool, improvement: float
    ) -> None:
        """Update effectiveness tracking for evolution strategy."""
        if strategy_name not in self.strategy_effectiveness:
            self.strategy_effectiveness[strategy_name] = 0.5  # Start neutral

        current_effectiveness = self.strategy_effectiveness[strategy_name]

        if success:
            # Increase effectiveness based on improvement magnitude
            adjustment = min(0.1, improvement * 0.5)
            new_effectiveness = min(1.0, current_effectiveness + adjustment)
        else:
            # Decrease effectiveness on failure
            adjustment = 0.05
            new_effectiveness = max(0.0, current_effectiveness - adjustment)

        self.strategy_effectiveness[strategy_name] = new_effectiveness

        logger.debug(
            f"Updated {strategy_name} effectiveness: {current_effectiveness:.2f} -> {new_effectiveness:.2f}"
        )

    def get_evolution_statistics(self) -> dict[str, Any]:
        """Get evolution statistics."""
        total_evolutions = sum(
            len(results) for results in self.evolution_results.values()
        )
        successful_evolutions = sum(
            sum(1 for r in results if r["success"])
            for results in self.evolution_results.values()
        )

        return {
            "total_agents": len(self.evolution_results),
            "total_evolutions": total_evolutions,
            "successful_evolutions": successful_evolutions,
            "success_rate": (
                successful_evolutions / total_evolutions if total_evolutions > 0 else 0
            ),
            "strategy_effectiveness": self.strategy_effectiveness.copy(),
            "available_strategies": list(self.strategies.keys()),
        }
