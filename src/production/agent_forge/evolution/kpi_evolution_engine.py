"""KPI-based evolution engine with retirement and evolution logic."""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from pathlib import Path
import pickle
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class EvolutionStrategy(Enum):
    """Evolution strategies for agents."""
    HOT_SWAP = "hot_swap"                    # In-place code optimization
    GENERATIONAL = "generational"           # Create new generation
    HYBRID = "hybrid"                       # Combine hot swap and generational
    KNOWLEDGE_DISTILLATION = "knowledge_distillation"  # Transfer knowledge to new model
    ARCHITECTURAL = "architectural"         # Modify agent architecture


class RetirementReason(Enum):
    """Reasons for agent retirement."""
    LOW_PERFORMANCE = "low_performance"
    OUTDATED_KNOWLEDGE = "outdated_knowledge"
    RESOURCE_INEFFICIENT = "resource_inefficient"
    SECURITY_VULNERABILITY = "security_vulnerability"
    SUPERSEDED = "superseded"
    USER_REQUEST = "user_request"


@dataclass
class AgentKPI:
    """Key Performance Indicators for an agent."""
    agent_id: str
    
    # Performance metrics
    accuracy: float = 0.0                   # Task accuracy (0-1)
    response_time_ms: float = 0.0          # Average response time
    throughput_tps: float = 0.0            # Tasks per second
    success_rate: float = 0.0              # Task success rate (0-1)
    
    # Resource efficiency
    memory_usage_mb: float = 0.0           # Average memory usage
    cpu_utilization: float = 0.0           # Average CPU usage (0-1)
    energy_efficiency: float = 0.0         # Tasks per watt-hour
    
    # Learning metrics
    adaptation_rate: float = 0.0           # How quickly agent adapts (0-1)
    knowledge_retention: float = 0.0       # Knowledge retention over time (0-1)
    generalization_ability: float = 0.0    # Performance on new tasks (0-1)
    
    # Quality metrics
    output_quality: float = 0.0            # Quality of outputs (0-1)
    consistency: float = 0.0               # Consistency across tasks (0-1)
    reliability: float = 0.0               # Reliability over time (0-1)
    
    # Specialized metrics
    security_score: float = 0.0            # Security compliance (0-1)
    bias_score: float = 0.0                # Bias in outputs (0=biased, 1=unbiased)
    interpretability: float = 0.0          # Explainability of decisions (0-1)
    
    # Meta metrics
    last_updated: float = field(default_factory=time.time)
    evaluation_count: int = 0
    confidence_interval: float = 0.95      # Confidence in metrics
    
    def overall_performance(self) -> float:
        """Calculate overall performance score."""
        # Weighted combination of key metrics
        weights = {
            'accuracy': 0.25,
            'success_rate': 0.20,
            'response_time': 0.15,  # Inverted - lower is better
            'throughput': 0.10,
            'resource_efficiency': 0.10,
            'quality': 0.10,
            'reliability': 0.10,
        }
        
        # Normalize response time (invert and normalize)
        normalized_response_time = max(0, 1 - (self.response_time_ms / 10000))  # 10s = 0 score
        
        # Calculate resource efficiency
        resource_efficiency = 0.0
        if self.memory_usage_mb > 0 and self.cpu_utilization > 0:
            # Higher throughput, lower resource usage = higher efficiency
            resource_efficiency = min(1.0, self.throughput_tps / (self.memory_usage_mb / 1000 + self.cpu_utilization))
            
        score = (
            self.accuracy * weights['accuracy'] +
            self.success_rate * weights['success_rate'] +
            normalized_response_time * weights['response_time'] +
            min(1.0, self.throughput_tps / 10) * weights['throughput'] +  # Normalize to reasonable range
            resource_efficiency * weights['resource_efficiency'] +
            self.output_quality * weights['quality'] +
            self.reliability * weights['reliability']
        )
        
        return min(1.0, max(0.0, score))
        
    def should_retire(self, threshold: float = 0.5) -> bool:
        """Determine if agent should be retired based on KPIs."""
        overall = self.overall_performance()
        
        # Multiple criteria for retirement
        retire_conditions = [
            overall < threshold,
            self.success_rate < 0.3,
            self.reliability < 0.4,
            self.accuracy < 0.4,
        ]
        
        return any(retire_conditions)
        
    def should_evolve(self, threshold: float = 0.6) -> bool:
        """Determine if agent should evolve (but not retire)."""
        overall = self.overall_performance()
        
        # Evolution candidates: decent performance but room for improvement
        evolve_conditions = [
            threshold <= overall < 0.8,
            self.adaptation_rate < 0.6,
            self.energy_efficiency < 0.7,
            self.response_time_ms > 5000,  # Slow responses
        ]
        
        return any(evolve_conditions) and not self.should_retire()
        
    def get_improvement_areas(self) -> List[str]:
        """Identify areas needing improvement."""
        areas = []
        
        if self.accuracy < 0.7:
            areas.append("accuracy")
        if self.response_time_ms > 3000:
            areas.append("response_time")
        if self.throughput_tps < 1.0:
            areas.append("throughput")
        if self.memory_usage_mb > 2000:
            areas.append("memory_usage")
        if self.cpu_utilization > 0.8:
            areas.append("cpu_utilization")
        if self.adaptation_rate < 0.6:
            areas.append("adaptation")
        if self.output_quality < 0.7:
            areas.append("output_quality")
        if self.reliability < 0.8:
            areas.append("reliability")
            
        return areas
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert KPI to dictionary."""
        return {
            "agent_id": self.agent_id,
            "accuracy": self.accuracy,
            "response_time_ms": self.response_time_ms,
            "throughput_tps": self.throughput_tps,
            "success_rate": self.success_rate,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_utilization": self.cpu_utilization,
            "energy_efficiency": self.energy_efficiency,
            "adaptation_rate": self.adaptation_rate,
            "knowledge_retention": self.knowledge_retention,
            "generalization_ability": self.generalization_ability,
            "output_quality": self.output_quality,
            "consistency": self.consistency,
            "reliability": self.reliability,
            "security_score": self.security_score,
            "bias_score": self.bias_score,
            "interpretability": self.interpretability,
            "overall_performance": self.overall_performance(),
            "last_updated": self.last_updated,
            "evaluation_count": self.evaluation_count,
            "confidence_interval": self.confidence_interval,
        }


@dataclass
class RetirementCriteria:
    """Criteria for agent retirement."""
    performance_threshold: float = 0.5
    reliability_threshold: float = 0.4
    max_age_days: Optional[int] = None
    min_evaluation_count: int = 10
    consecutive_poor_performance: int = 5
    resource_efficiency_threshold: float = 0.3
    
    
@dataclass
class EvolutionResult:
    """Result of an evolution operation."""
    agent_id: str
    strategy_used: EvolutionStrategy
    success: bool
    improvements: Dict[str, float] = field(default_factory=dict)
    new_agent_id: Optional[str] = None
    evolution_time_seconds: float = 0.0
    error_message: Optional[str] = None
    
    
class KnowledgeDistiller:
    """Distills knowledge from retiring agents."""
    
    @staticmethod
    def distill_knowledge(agent_kpis: List[AgentKPI], agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract valuable knowledge from agent before retirement."""
        knowledge = {
            "performance_patterns": {},
            "successful_strategies": [],
            "common_failures": [],
            "optimization_insights": {},
            "specialized_knowledge": {},
        }
        
        # Analyze performance patterns
        if agent_kpis:
            latest_kpi = agent_kpis[-1]
            knowledge["performance_patterns"] = {
                "peak_performance": max(kpi.overall_performance() for kpi in agent_kpis),
                "average_performance": sum(kpi.overall_performance() for kpi in agent_kpis) / len(agent_kpis),
                "improvement_areas": latest_kpi.get_improvement_areas(),
                "strengths": KnowledgeDistiller._identify_strengths(latest_kpi),
            }
            
        # Extract successful strategies
        high_performing_periods = [kpi for kpi in agent_kpis if kpi.overall_performance() > 0.7]
        if high_performing_periods:
            knowledge["successful_strategies"] = [
                "high_accuracy_period" if kpi.accuracy > 0.8 else None
                for kpi in high_performing_periods
            ]
            knowledge["successful_strategies"] = [s for s in knowledge["successful_strategies"] if s]
            
        # Common failure patterns
        poor_performing_periods = [kpi for kpi in agent_kpis if kpi.overall_performance() < 0.5]
        if poor_performing_periods:
            knowledge["common_failures"] = [
                "memory_pressure" if kpi.memory_usage_mb > 1500 else None
                for kpi in poor_performing_periods
            ]
            knowledge["common_failures"] = [f for f in knowledge["common_failures"] if f]
            
        return knowledge
        
    @staticmethod
    def _identify_strengths(kpi: AgentKPI) -> List[str]:
        """Identify agent's key strengths."""
        strengths = []
        
        if kpi.accuracy > 0.8:
            strengths.append("high_accuracy")
        if kpi.response_time_ms < 1000:
            strengths.append("fast_response")
        if kpi.throughput_tps > 2.0:
            strengths.append("high_throughput")
        if kpi.reliability > 0.9:
            strengths.append("high_reliability")
        if kpi.energy_efficiency > 0.8:
            strengths.append("energy_efficient")
        if kpi.generalization_ability > 0.7:
            strengths.append("generalizes_well")
            
        return strengths


class KPIEvolutionEngine:
    """KPI-based evolution engine with retirement and evolution logic."""
    
    def __init__(
        self,
        retirement_criteria: Optional[RetirementCriteria] = None,
        evolution_strategies: Optional[List[EvolutionStrategy]] = None,
        population_size_range: Tuple[int, int] = (4, 8),
        evolution_interval_hours: float = 24.0,
        storage_path: str = "evolution_data",
    ):
        self.retirement_criteria = retirement_criteria or RetirementCriteria()
        self.evolution_strategies = evolution_strategies or [
            EvolutionStrategy.HOT_SWAP,
            EvolutionStrategy.GENERATIONAL,
            EvolutionStrategy.KNOWLEDGE_DISTILLATION,
        ]
        self.population_size_range = population_size_range
        self.evolution_interval_hours = evolution_interval_hours
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Agent management
        self.active_agents: Dict[str, Dict[str, Any]] = {}  # agent_id -> agent_info
        self.agent_kpis: Dict[str, List[AgentKPI]] = defaultdict(list)  # agent_id -> KPI history
        self.retired_agents: Dict[str, Dict[str, Any]] = {}  # Retired agent archive
        self.expert_knowledge: Dict[str, Dict[str, Any]] = {}  # Distilled knowledge from experts
        
        # Evolution tracking
        self.evolution_history: List[EvolutionResult] = []
        self.population_stats: Dict[str, Any] = {
            "generations": 0,
            "total_agents_created": 0,
            "total_agents_retired": 0,
            "avg_lifespan_hours": 0.0,
            "performance_trend": "stable",
        }
        
        # Evolution scheduling
        self.evolution_active = False
        self.evolution_thread: Optional[threading.Thread] = None
        self.last_evolution_time = time.time()
        
        # Callbacks
        self.retirement_callbacks: List[Callable[[str, RetirementReason], None]] = []
        self.evolution_callbacks: List[Callable[[EvolutionResult], None]] = []
        
        logger.info(f"KPI Evolution Engine initialized with population range {population_size_range}")
        
    def register_agent(
        self,
        agent_id: str,
        agent_type: str,
        initial_config: Dict[str, Any],
        template_path: Optional[str] = None,
    ) -> None:
        """Register a new agent in the evolution system."""
        agent_info = {
            "agent_id": agent_id,
            "agent_type": agent_type,
            "config": initial_config,
            "template_path": template_path,
            "created_at": time.time(),
            "generation": 0,
            "parent_id": None,
            "evolution_count": 0,
            "last_evolution": None,
            "status": "active",
        }
        
        self.active_agents[agent_id] = agent_info
        self.population_stats["total_agents_created"] += 1
        
        logger.info(f"Registered agent {agent_id} of type {agent_type}")
        
    def update_agent_kpi(self, agent_id: str, kpi: AgentKPI) -> None:
        """Update KPI for an agent."""
        if agent_id not in self.active_agents:
            logger.warning(f"Agent {agent_id} not registered")
            return
            
        kpi.evaluation_count = len(self.agent_kpis[agent_id]) + 1
        self.agent_kpis[agent_id].append(kpi)
        
        # Keep only recent KPI history
        max_history = 100
        if len(self.agent_kpis[agent_id]) > max_history:
            self.agent_kpis[agent_id] = self.agent_kpis[agent_id][-max_history:]
            
        logger.debug(f"Updated KPI for agent {agent_id}: performance={kpi.overall_performance():.3f}")
        
    def evaluate_population(self) -> Dict[str, Any]:
        """Evaluate all agents and trigger evolution/retirement."""
        evaluation_results = {
            "agents_evaluated": 0,
            "retirement_candidates": [],
            "evolution_candidates": [],
            "actions_taken": [],
        }
        
        for agent_id in list(self.active_agents.keys()):
            if agent_id not in self.agent_kpis or not self.agent_kpis[agent_id]:
                continue
                
            latest_kpi = self.agent_kpis[agent_id][-1]
            agent_info = self.active_agents[agent_id]
            
            evaluation_results["agents_evaluated"] += 1
            
            # Check for retirement
            if self._should_retire_agent(agent_id, latest_kpi, agent_info):
                evaluation_results["retirement_candidates"].append(agent_id)
                retirement_reason = self._determine_retirement_reason(latest_kpi, agent_info)
                
                # Retire agent
                asyncio.create_task(self.retire_agent(agent_id, retirement_reason))
                evaluation_results["actions_taken"].append(f"retired_{agent_id}")
                
            # Check for evolution
            elif self._should_evolve_agent(agent_id, latest_kpi, agent_info):
                evaluation_results["evolution_candidates"].append(agent_id)
                
                # Evolve agent
                strategy = self._choose_evolution_strategy(latest_kpi)
                asyncio.create_task(self.evolve_agent(agent_id, strategy))
                evaluation_results["actions_taken"].append(f"evolved_{agent_id}")
                
        return evaluation_results
        
    async def retire_agent(self, agent_id: str, reason: RetirementReason) -> bool:
        """Gracefully retire an agent."""
        if agent_id not in self.active_agents:
            logger.warning(f"Agent {agent_id} not found for retirement")
            return False
            
        logger.info(f"Retiring agent {agent_id} due to {reason.value}")
        
        agent_info = self.active_agents[agent_id]
        agent_kpis = self.agent_kpis.get(agent_id, [])
        
        try:
            # Extract valuable knowledge
            knowledge = KnowledgeDistiller.distill_knowledge(agent_kpis, agent_info)
            
            # Create successor if needed
            successor_id = None
            if len(self.active_agents) <= self.population_size_range[0]:
                successor_id = await self._create_successor(agent_info, knowledge)
                
            # Archive retired agent
            retirement_info = {
                **agent_info,
                "retired_at": time.time(),
                "retirement_reason": reason.value,
                "final_kpi": agent_kpis[-1].to_dict() if agent_kpis else None,
                "distilled_knowledge": knowledge,
                "successor_id": successor_id,
            }
            
            self.retired_agents[agent_id] = retirement_info
            
            # Store as expert if performance was good
            if agent_kpis and agent_kpis[-1].overall_performance() > 0.7:
                self.expert_knowledge[agent_id] = knowledge
                
            # Remove from active agents
            del self.active_agents[agent_id]
            self.population_stats["total_agents_retired"] += 1
            
            # Update lifespan statistics
            lifespan_hours = (time.time() - agent_info["created_at"]) / 3600
            self._update_lifespan_stats(lifespan_hours)
            
            # Notify callbacks
            for callback in self.retirement_callbacks:
                try:
                    callback(agent_id, reason)
                except Exception as e:
                    logger.error(f"Error in retirement callback: {e}")
                    
            # Save state
            await self._save_evolution_state()
            
            logger.info(f"Successfully retired agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error retiring agent {agent_id}: {e}")
            return False
            
    async def evolve_agent(self, agent_id: str, strategy: EvolutionStrategy) -> EvolutionResult:
        """Evolve an agent using specified strategy."""
        start_time = time.time()
        
        result = EvolutionResult(
            agent_id=agent_id,
            strategy_used=strategy,
            success=False,
        )
        
        if agent_id not in self.active_agents:
            result.error_message = f"Agent {agent_id} not found"
            return result
            
        logger.info(f"Evolving agent {agent_id} using {strategy.value} strategy")
        
        try:
            agent_info = self.active_agents[agent_id]
            latest_kpi = self.agent_kpis[agent_id][-1] if self.agent_kpis[agent_id] else None
            
            if strategy == EvolutionStrategy.HOT_SWAP:
                result = await self._hot_swap_evolution(agent_id, agent_info, latest_kpi)
            elif strategy == EvolutionStrategy.GENERATIONAL:
                result = await self._generational_evolution(agent_id, agent_info, latest_kpi)
            elif strategy == EvolutionStrategy.KNOWLEDGE_DISTILLATION:
                result = await self._knowledge_distillation_evolution(agent_id, agent_info, latest_kpi)
            elif strategy == EvolutionStrategy.ARCHITECTURAL:
                result = await self._architectural_evolution(agent_id, agent_info, latest_kpi)
            else:
                result.error_message = f"Unsupported strategy: {strategy}"
                
            result.evolution_time_seconds = time.time() - start_time
            
            if result.success:
                # Update agent info
                self.active_agents[agent_id]["evolution_count"] += 1
                self.active_agents[agent_id]["last_evolution"] = time.time()
                
                logger.info(f"Successfully evolved agent {agent_id}")
            else:
                logger.warning(f"Evolution failed for agent {agent_id}: {result.error_message}")
                
            # Record evolution
            self.evolution_history.append(result)
            
            # Notify callbacks
            for callback in self.evolution_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"Error in evolution callback: {e}")
                    
            return result
            
        except Exception as e:
            result.error_message = str(e)
            result.evolution_time_seconds = time.time() - start_time
            logger.error(f"Error evolving agent {agent_id}: {e}")
            return result
            
    def start_evolution_scheduler(self) -> None:
        """Start automatic evolution scheduling."""
        if self.evolution_active:
            return
            
        self.evolution_active = True
        self.evolution_thread = threading.Thread(
            target=self._evolution_loop,
            daemon=True
        )
        self.evolution_thread.start()
        
        logger.info("Evolution scheduler started")
        
    def stop_evolution_scheduler(self) -> None:
        """Stop automatic evolution scheduling."""
        self.evolution_active = False
        
        if self.evolution_thread and self.evolution_thread.is_alive():
            self.evolution_thread.join(timeout=10.0)
            
        logger.info("Evolution scheduler stopped")
        
    def _evolution_loop(self) -> None:
        """Background evolution scheduling loop."""
        while self.evolution_active:
            try:
                current_time = time.time()
                
                # Check if it's time for evolution
                time_since_last = (current_time - self.last_evolution_time) / 3600  # hours
                
                if time_since_last >= self.evolution_interval_hours:
                    logger.info("Starting scheduled evolution cycle")
                    
                    # Evaluate population
                    results = self.evaluate_population()
                    
                    self.last_evolution_time = current_time
                    self.population_stats["generations"] += 1
                    
                    logger.info(f"Evolution cycle complete: {results}")
                    
                # Sleep for 1 hour before checking again
                time.sleep(3600)
                
            except Exception as e:
                logger.error(f"Error in evolution loop: {e}")
                time.sleep(3600)
                
    def _should_retire_agent(self, agent_id: str, kpi: AgentKPI, agent_info: Dict[str, Any]) -> bool:
        """Determine if agent should be retired."""
        # Performance-based retirement
        if kpi.should_retire(self.retirement_criteria.performance_threshold):
            return True
            
        # Reliability-based retirement
        if kpi.reliability < self.retirement_criteria.reliability_threshold:
            return True
            
        # Age-based retirement
        if self.retirement_criteria.max_age_days:
            age_days = (time.time() - agent_info["created_at"]) / (24 * 3600)
            if age_days > self.retirement_criteria.max_age_days:
                return True
                
        # Consecutive poor performance
        recent_kpis = self.agent_kpis[agent_id][-self.retirement_criteria.consecutive_poor_performance:]
        if len(recent_kpis) >= self.retirement_criteria.consecutive_poor_performance:
            poor_performance_count = sum(
                1 for k in recent_kpis
                if k.overall_performance() < self.retirement_criteria.performance_threshold
            )
            if poor_performance_count >= self.retirement_criteria.consecutive_poor_performance:
                return True
                
        # Resource efficiency
        if kpi.energy_efficiency < self.retirement_criteria.resource_efficiency_threshold:
            return True
            
        return False
        
    def _should_evolve_agent(self, agent_id: str, kpi: AgentKPI, agent_info: Dict[str, Any]) -> bool:
        """Determine if agent should evolve."""
        return kpi.should_evolve() and not self._should_retire_agent(agent_id, kpi, agent_info)
        
    def _determine_retirement_reason(self, kpi: AgentKPI, agent_info: Dict[str, Any]) -> RetirementReason:
        """Determine primary reason for retirement."""
        if kpi.overall_performance() < 0.3:
            return RetirementReason.LOW_PERFORMANCE
        elif kpi.energy_efficiency < 0.3:
            return RetirementReason.RESOURCE_INEFFICIENT
        elif kpi.reliability < 0.4:
            return RetirementReason.LOW_PERFORMANCE
        else:
            return RetirementReason.LOW_PERFORMANCE  # Default
            
    def _choose_evolution_strategy(self, kpi: AgentKPI) -> EvolutionStrategy:
        """Choose evolution strategy based on KPI analysis."""
        improvement_areas = kpi.get_improvement_areas()
        
        # Strategy selection logic
        if "response_time" in improvement_areas or "cpu_utilization" in improvement_areas:
            return EvolutionStrategy.HOT_SWAP  # Quick optimization
        elif "accuracy" in improvement_areas or "output_quality" in improvement_areas:
            return EvolutionStrategy.KNOWLEDGE_DISTILLATION  # Learn from experts
        elif len(improvement_areas) > 3:
            return EvolutionStrategy.GENERATIONAL  # Major overhaul needed
        else:
            return EvolutionStrategy.HOT_SWAP  # Default
            
    async def _hot_swap_evolution(
        self,
        agent_id: str,
        agent_info: Dict[str, Any],
        kpi: Optional[AgentKPI]
    ) -> EvolutionResult:
        """Perform hot-swap evolution (in-place optimization)."""
        result = EvolutionResult(agent_id=agent_id, strategy_used=EvolutionStrategy.HOT_SWAP, success=True)
        
        # Simulate code optimization
        improvements = {}
        
        if kpi:
            # Optimize based on improvement areas
            areas = kpi.get_improvement_areas()
            
            if "response_time" in areas:
                improvements["response_time_improvement"] = 0.2  # 20% improvement
            if "memory_usage" in areas:
                improvements["memory_optimization"] = 0.15
            if "cpu_utilization" in areas:
                improvements["cpu_optimization"] = 0.1
                
        result.improvements = improvements
        return result
        
    async def _generational_evolution(
        self,
        agent_id: str,
        agent_info: Dict[str, Any],
        kpi: Optional[AgentKPI]
    ) -> EvolutionResult:
        """Perform generational evolution (create new agent)."""
        result = EvolutionResult(agent_id=agent_id, strategy_used=EvolutionStrategy.GENERATIONAL, success=True)
        
        # Create new agent ID
        new_agent_id = f"{agent_id}_gen{agent_info['generation'] + 1}_{int(time.time())}"
        
        # Simulate creating new generation
        new_config = agent_info["config"].copy()
        new_config["generation"] = agent_info["generation"] + 1
        new_config["parent_id"] = agent_id
        
        result.new_agent_id = new_agent_id
        result.improvements = {"generational_upgrade": 0.3}
        
        # Register new agent
        self.register_agent(
            new_agent_id,
            agent_info["agent_type"],
            new_config,
            agent_info.get("template_path")
        )
        
        return result
        
    async def _knowledge_distillation_evolution(
        self,
        agent_id: str,
        agent_info: Dict[str, Any],
        kpi: Optional[AgentKPI]
    ) -> EvolutionResult:
        """Perform knowledge distillation evolution."""
        result = EvolutionResult(agent_id=agent_id, strategy_used=EvolutionStrategy.KNOWLEDGE_DISTILLATION, success=True)
        
        # Find expert knowledge to transfer
        relevant_experts = []
        for expert_id, knowledge in self.expert_knowledge.items():
            if agent_info["agent_type"] in expert_id or "general" in knowledge.get("applicable_types", []):
                relevant_experts.append((expert_id, knowledge))
                
        if relevant_experts:
            result.improvements = {"knowledge_transfer": 0.25}
        else:
            result.improvements = {"self_optimization": 0.1}
            
        return result
        
    async def _architectural_evolution(
        self,
        agent_id: str,
        agent_info: Dict[str, Any],
        kpi: Optional[AgentKPI]
    ) -> EvolutionResult:
        """Perform architectural evolution."""
        result = EvolutionResult(agent_id=agent_id, strategy_used=EvolutionStrategy.ARCHITECTURAL, success=True)
        
        # Simulate architectural improvements
        result.improvements = {
            "architecture_optimization": 0.4,
            "model_size_optimization": 0.2,
            "inference_optimization": 0.3,
        }
        
        return result
        
    async def _create_successor(self, agent_info: Dict[str, Any], knowledge: Dict[str, Any]) -> str:
        """Create successor agent with inherited knowledge."""
        successor_id = f"successor_{agent_info['agent_id']}_{int(time.time())}"
        
        # Create successor configuration
        successor_config = agent_info["config"].copy()
        successor_config["inherited_knowledge"] = knowledge
        successor_config["parent_id"] = agent_info["agent_id"]
        successor_config["generation"] = agent_info.get("generation", 0) + 1
        
        # Register successor
        self.register_agent(
            successor_id,
            agent_info["agent_type"],
            successor_config,
            agent_info.get("template_path")
        )
        
        logger.info(f"Created successor {successor_id} for retired agent {agent_info['agent_id']}")
        return successor_id
        
    def _update_lifespan_stats(self, lifespan_hours: float) -> None:
        """Update average lifespan statistics."""
        total_retired = self.population_stats["total_agents_retired"]
        current_avg = self.population_stats["avg_lifespan_hours"]
        
        # Calculate new average
        self.population_stats["avg_lifespan_hours"] = (
            (current_avg * (total_retired - 1) + lifespan_hours) / total_retired
        )
        
    async def _save_evolution_state(self) -> None:
        """Save evolution state to disk."""
        state = {
            "active_agents": self.active_agents,
            "retired_agents": dict(list(self.retired_agents.items())[-100:]),  # Keep last 100
            "expert_knowledge": self.expert_knowledge,
            "evolution_history": self.evolution_history[-100:],  # Keep last 100
            "population_stats": self.population_stats,
            "last_evolution_time": self.last_evolution_time,
        }
        
        state_file = self.storage_path / "evolution_state.json"
        
        try:
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save evolution state: {e}")
            
    def register_retirement_callback(self, callback: Callable[[str, RetirementReason], None]) -> None:
        """Register callback for agent retirement events."""
        self.retirement_callbacks.append(callback)
        
    def register_evolution_callback(self, callback: Callable[[EvolutionResult], None]) -> None:
        """Register callback for agent evolution events."""
        self.evolution_callbacks.append(callback)
        
    def get_population_status(self) -> Dict[str, Any]:
        """Get current population status."""
        active_count = len(self.active_agents)
        
        # Calculate performance statistics
        current_performances = []
        for agent_id in self.active_agents:
            if self.agent_kpis[agent_id]:
                current_performances.append(self.agent_kpis[agent_id][-1].overall_performance())
                
        avg_performance = sum(current_performances) / len(current_performances) if current_performances else 0.0
        
        return {
            "active_agents": active_count,
            "retired_agents": len(self.retired_agents),
            "expert_knowledge_bases": len(self.expert_knowledge),
            "avg_performance": avg_performance,
            "population_range": self.population_size_range,
            "within_range": self.population_size_range[0] <= active_count <= self.population_size_range[1],
            "evolution_active": self.evolution_active,
            "last_evolution": self.last_evolution_time,
            "next_evolution": self.last_evolution_time + (self.evolution_interval_hours * 3600),
            "statistics": self.population_stats,
        }
        
    def get_agent_performance_history(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get performance history for an agent."""
        if agent_id not in self.agent_kpis:
            return []
            
        return [kpi.to_dict() for kpi in self.agent_kpis[agent_id]]

