"""Resource-Constrained Evolution System for Mobile-First Operation"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

# Import resource management and dual evolution
from ....core.resources import DeviceProfiler, ResourceMonitor, ConstraintManager, AdaptiveLoader
from ....core.resources.constraint_manager import ResourceConstraints, ConstraintViolation, ConstraintSeverity
from .dual_evolution_system import DualEvolutionSystem, EvolutionEvent, EvolutionSchedule
from .base import EvolvableAgent

logger = logging.getLogger(__name__)

class ResourceAdaptationStrategy(Enum):
    """Strategies for adapting to resource constraints"""
    PAUSE_AND_RETRY = "pause_and_retry"
    DEGRADE_QUALITY = "degrade_quality"
    REDUCE_SCOPE = "reduce_scope"
    OFFLOAD_TO_PEER = "offload_to_peer"
    EMERGENCY_STOP = "emergency_stop"

@dataclass
class ResourceConstrainedConfig:
    """Configuration for resource-constrained evolution"""
    # Memory management
    memory_limit_multiplier: float = 0.8  # Use 80% of available memory
    memory_cleanup_threshold: float = 0.9  # Cleanup when 90% full
    enable_memory_monitoring: bool = True
    
    # CPU management
    cpu_limit_multiplier: float = 0.75  # Use 75% of available CPU
    cpu_throttle_threshold: float = 85.0  # Throttle when CPU > 85%
    enable_cpu_throttling: bool = True
    
    # Battery management
    battery_minimum_percent: float = 15.0
    battery_pause_threshold: float = 20.0
    battery_optimization_mode: bool = True
    
    # Thermal management
    thermal_throttle_temperature: float = 80.0
    thermal_pause_temperature: float = 85.0
    enable_thermal_protection: bool = True
    
    # Adaptive strategies
    enable_quality_degradation: bool = True
    enable_scope_reduction: bool = True
    enable_pause_resume: bool = True
    max_pause_duration_minutes: float = 30.0
    
    # Performance targets
    target_evolution_time_minutes: float = 45.0
    acceptable_quality_loss: float = 0.1  # 10% quality loss acceptable
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'memory_limit_multiplier': self.memory_limit_multiplier,
            'memory_cleanup_threshold': self.memory_cleanup_threshold,
            'enable_memory_monitoring': self.enable_memory_monitoring,
            'cpu_limit_multiplier': self.cpu_limit_multiplier,
            'cpu_throttle_threshold': self.cpu_throttle_threshold,
            'enable_cpu_throttling': self.enable_cpu_throttling,
            'battery_minimum_percent': self.battery_minimum_percent,
            'battery_pause_threshold': self.battery_pause_threshold,
            'battery_optimization_mode': self.battery_optimization_mode,
            'thermal_throttle_temperature': self.thermal_throttle_temperature,
            'thermal_pause_temperature': self.thermal_pause_temperature,
            'enable_thermal_protection': self.enable_thermal_protection,
            'enable_quality_degradation': self.enable_quality_degradation,
            'enable_scope_reduction': self.enable_scope_reduction,
            'enable_pause_resume': self.enable_pause_resume,
            'max_pause_duration_minutes': self.max_pause_duration_minutes,
            'target_evolution_time_minutes': self.target_evolution_time_minutes,
            'acceptable_quality_loss': self.acceptable_quality_loss
        }

@dataclass
class EvolutionResourceState:
    """Current resource state for evolution"""
    memory_allocated_mb: int
    memory_used_mb: int
    cpu_allocated_percent: float
    cpu_used_percent: float
    battery_percent: Optional[float]
    temperature_celsius: Optional[float]
    is_constrained: bool
    constraint_types: List[str] = field(default_factory=list)
    adaptation_strategy: Optional[ResourceAdaptationStrategy] = None
    
class ResourceConstrainedEvolution(DualEvolutionSystem):
    """Enhanced dual evolution system with comprehensive resource constraints"""
    
    def __init__(self, 
                 device_profiler: DeviceProfiler,
                 resource_monitor: ResourceMonitor,
                 constraint_manager: ConstraintManager,
                 adaptive_loader: Optional[AdaptiveLoader] = None,
                 config: Optional[ResourceConstrainedConfig] = None):
        
        # Initialize parent class
        super().__init__()
        
        # Resource management components
        self.device_profiler = device_profiler
        self.resource_monitor = resource_monitor
        self.constraint_manager = constraint_manager
        self.adaptive_loader = adaptive_loader
        self.config = config or ResourceConstrainedConfig()
        
        # Resource state tracking
        self.current_resource_state: Optional[EvolutionResourceState] = None
        self.paused_evolutions: Dict[str, Dict[str, Any]] = {}
        
        # Resource event callbacks
        self.resource_violation_callbacks: List[Callable] = []
        self.adaptation_callbacks: List[Callable] = []
        
        # Statistics
        self.resource_stats = {
            'resource_violations': 0,
            'adaptations_triggered': 0,
            'evolutions_paused': 0,
            'evolutions_resumed': 0,
            'quality_degradations': 0,
            'scope_reductions': 0,
            'emergency_stops': 0,
            'memory_cleanups': 0,
            'thermal_throttles': 0
        }
        
        # Register constraint callbacks
        self._register_constraint_callbacks()
        
    def _register_constraint_callbacks(self):
        """Register callbacks for constraint violations"""
        self.constraint_manager.register_violation_callback(self._handle_constraint_violation)
        self.constraint_manager.register_enforcement_callback(self._handle_constraint_enforcement)
        
    async def start_system(self):
        """Start resource-constrained evolution system"""
        # Start parent system
        await super().start_system()
        
        # Start resource monitoring
        await self.resource_monitor.start_monitoring()
        
        # Register resource event callbacks
        self.resource_monitor.register_event_callback(self._handle_resource_event)
        
        logger.info("Resource-constrained evolution system started")
        
    async def stop_system(self):
        """Stop resource-constrained evolution system"""
        # Resume any paused evolutions before stopping
        for evolution_id in list(self.paused_evolutions.keys()):
            await self._resume_evolution(evolution_id)
            
        # Stop resource monitoring
        await self.resource_monitor.stop_monitoring()
        
        # Stop parent system
        await super().stop_system()
        
        logger.info("Resource-constrained evolution system stopped")
        
    async def _evolve_agent_nightly(self, agent: EvolvableAgent) -> bool:
        """Enhanced nightly evolution with resource constraints"""
        evolution_id = f"nightly_{agent.agent_id}_{int(time.time())}"
        
        # Check initial resource availability
        if not await self._check_evolution_feasibility("nightly", agent):
            logger.warning(f"Nightly evolution not feasible for agent {agent.agent_id}")
            return False
            
        # Register evolution task with constraint manager
        if not self.constraint_manager.register_task(evolution_id, "nightly"):
            logger.warning(f"Failed to register nightly evolution task for agent {agent.agent_id}")
            return False
            
        try:
            # Update resource state
            await self._update_resource_state()
            
            # Load models adaptively if available
            if self.adaptive_loader:
                await self._load_evolution_models(agent, "nightly")
                
            # Execute evolution with resource monitoring
            result = await self._execute_monitored_evolution(
                agent, 
                "nightly", 
                super()._evolve_agent_nightly
            )
            
            return result
            
        finally:
            # Cleanup
            self.constraint_manager.unregister_task(evolution_id)
            await self._cleanup_evolution_resources(evolution_id)
            
    async def _evolve_agent_breakthrough(self, agent: EvolvableAgent) -> bool:
        """Enhanced breakthrough evolution with resource constraints"""
        evolution_id = f"breakthrough_{agent.agent_id}_{int(time.time())}"
        
        # Breakthrough evolution has higher resource requirements
        if not await self._check_evolution_feasibility("breakthrough", agent):
            logger.warning(f"Breakthrough evolution not feasible for agent {agent.agent_id}")
            return False
            
        if not self.constraint_manager.register_task(evolution_id, "breakthrough"):
            logger.warning(f"Failed to register breakthrough evolution task for agent {agent.agent_id}")
            return False
            
        try:
            await self._update_resource_state()
            
            # Load high-quality models for breakthrough
            if self.adaptive_loader:
                await self._load_evolution_models(agent, "breakthrough")
                
            # Execute with enhanced monitoring
            result = await self._execute_monitored_evolution(
                agent,
                "breakthrough", 
                super()._evolve_agent_breakthrough
            )
            
            return result
            
        finally:
            self.constraint_manager.unregister_task(evolution_id)
            await self._cleanup_evolution_resources(evolution_id)
            
    async def _evolve_agent_emergency(self, agent: EvolvableAgent) -> bool:
        """Enhanced emergency evolution with resource constraints"""
        evolution_id = f"emergency_{agent.agent_id}_{int(time.time())}"
        
        # Emergency evolution gets priority but still needs basic resources
        if not await self._check_evolution_feasibility("emergency", agent):
            logger.warning(f"Emergency evolution not feasible for agent {agent.agent_id}")
            return False
            
        if not self.constraint_manager.register_task(evolution_id, "emergency"):
            logger.warning(f"Failed to register emergency evolution task for agent {agent.agent_id}")
            return False
            
        try:
            await self._update_resource_state()
            
            # Load lightweight models for emergency
            if self.adaptive_loader:
                await self._load_evolution_models(agent, "emergency")
                
            # Execute with priority handling
            result = await self._execute_monitored_evolution(
                agent,
                "emergency",
                super()._evolve_agent_emergency
            )
            
            return result
            
        finally:
            self.constraint_manager.unregister_task(evolution_id)
            await self._cleanup_evolution_resources(evolution_id)
            
    async def _check_evolution_feasibility(self, evolution_type: str, agent: EvolvableAgent) -> bool:
        """Check if evolution is feasible given current resources"""
        # Check device capability
        if not self.device_profiler.profile.evolution_capable:
            return False
            
        # Check current resource state
        current_snapshot = self.device_profiler.current_snapshot
        if not current_snapshot:
            return False
            
        # Check evolution suitability
        suitability = current_snapshot.evolution_suitability_score
        
        # Different thresholds for different evolution types
        thresholds = {
            'emergency': 0.3,    # Lower threshold for emergency
            'nightly': 0.5,      # Medium threshold for nightly  
            'breakthrough': 0.7  # Higher threshold for breakthrough
        }
        
        required_suitability = thresholds.get(evolution_type, 0.5)
        
        if suitability < required_suitability:
            logger.debug(f"Evolution suitability {suitability:.2f} below threshold {required_suitability:.2f}")
            return False
            
        # Check specific constraints
        constraints = self.constraint_manager.get_constraint_template(evolution_type)
        if constraints:
            # Memory check
            available_memory_mb = current_snapshot.memory_available / (1024 * 1024)
            if available_memory_mb < constraints.max_memory_mb * 0.5:
                return False
                
            # CPU check
            if current_snapshot.cpu_percent > 90:
                return False
                
            # Battery check
            if (constraints.min_battery_percent and 
                current_snapshot.battery_percent and
                current_snapshot.battery_percent < constraints.min_battery_percent):
                return False
                
            # Thermal check
            if (current_snapshot.thermal_state.value in 
                ['hot', 'critical', 'throttling']):
                return False
                
        return True
        
    async def _execute_monitored_evolution(self, 
                                         agent: EvolvableAgent,
                                         evolution_type: str,
                                         evolution_func: Callable) -> bool:
        """Execute evolution with comprehensive resource monitoring"""
        start_time = time.time()
        
        try:
            # Set up monitoring
            await self._update_resource_state()
            
            # Execute evolution with periodic monitoring
            evolution_task = asyncio.create_task(evolution_func(agent))
            monitoring_task = asyncio.create_task(
                self._monitor_evolution_resources(agent.agent_id, evolution_type)
            )
            
            # Wait for evolution to complete or resource constraints
            done, pending = await asyncio.wait(
                [evolution_task, monitoring_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel remaining tasks
            for task in pending:
                task.cancel()
                
            # Check results
            if evolution_task in done:
                result = evolution_task.result()
                logger.info(f"Evolution completed successfully for agent {agent.agent_id}")
                return result
            else:
                # Evolution was interrupted by resource constraints
                logger.warning(f"Evolution interrupted by resource constraints for agent {agent.agent_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error in monitored evolution: {e}")
            return False
            
    async def _monitor_evolution_resources(self, agent_id: str, evolution_type: str):
        """Monitor resources during evolution"""
        while True:
            try:
                await asyncio.sleep(self.config.target_evolution_time_minutes / 10)  # Check 10 times during evolution
                
                # Update resource state
                await self._update_resource_state()
                
                # Check if we need to adapt
                if self.current_resource_state and self.current_resource_state.is_constrained:
                    strategy = await self._determine_adaptation_strategy(agent_id, evolution_type)
                    if strategy:
                        await self._apply_adaptation_strategy(agent_id, strategy)
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                await asyncio.sleep(5)
                
    async def _update_resource_state(self):
        """Update current resource state"""
        snapshot = self.device_profiler.current_snapshot
        if not snapshot:
            return
            
        # Calculate resource usage
        memory_used_mb = snapshot.memory_used / (1024 * 1024)
        memory_allocated_mb = int(snapshot.memory_total / (1024 * 1024) * self.config.memory_limit_multiplier)
        
        cpu_used = snapshot.cpu_percent
        cpu_allocated = 100 * self.config.cpu_limit_multiplier
        
        # Determine constraints
        constraint_types = []
        is_constrained = False
        
        if memory_used_mb > memory_allocated_mb * self.config.memory_cleanup_threshold:
            constraint_types.append("memory")
            is_constrained = True
            
        if cpu_used > self.config.cpu_throttle_threshold:
            constraint_types.append("cpu")
            is_constrained = True
            
        if (snapshot.battery_percent and 
            snapshot.battery_percent < self.config.battery_pause_threshold):
            constraint_types.append("battery")
            is_constrained = True
            
        if (snapshot.cpu_temp and 
            snapshot.cpu_temp > self.config.thermal_throttle_temperature):
            constraint_types.append("thermal")
            is_constrained = True
            
        self.current_resource_state = EvolutionResourceState(
            memory_allocated_mb=memory_allocated_mb,
            memory_used_mb=int(memory_used_mb),
            cpu_allocated_percent=cpu_allocated,
            cpu_used_percent=cpu_used,
            battery_percent=snapshot.battery_percent,
            temperature_celsius=snapshot.cpu_temp,
            is_constrained=is_constrained,
            constraint_types=constraint_types
        )
        
    async def _determine_adaptation_strategy(self, agent_id: str, evolution_type: str) -> Optional[ResourceAdaptationStrategy]:
        """Determine appropriate adaptation strategy"""
        if not self.current_resource_state:
            return None
            
        constraint_types = self.current_resource_state.constraint_types
        
        # Emergency stop for critical conditions
        if ("thermal" in constraint_types and 
            self.current_resource_state.temperature_celsius and
            self.current_resource_state.temperature_celsius > self.config.thermal_pause_temperature):
            return ResourceAdaptationStrategy.EMERGENCY_STOP
            
        if ("battery" in constraint_types and
            self.current_resource_state.battery_percent and
            self.current_resource_state.battery_percent < self.config.battery_minimum_percent):
            return ResourceAdaptationStrategy.EMERGENCY_STOP
            
        # Pause and retry for recoverable conditions
        if "battery" in constraint_types and self.config.enable_pause_resume:
            return ResourceAdaptationStrategy.PAUSE_AND_RETRY
            
        if "thermal" in constraint_types and self.config.enable_pause_resume:
            return ResourceAdaptationStrategy.PAUSE_AND_RETRY
            
        # Memory cleanup
        if "memory" in constraint_types:
            # Try memory cleanup first
            await self._cleanup_memory()
            return None  # No further adaptation needed if cleanup successful
            
        # Quality degradation for CPU constraints
        if ("cpu" in constraint_types and 
            self.config.enable_quality_degradation and
            evolution_type in ["nightly", "breakthrough"]):
            return ResourceAdaptationStrategy.DEGRADE_QUALITY
            
        # Scope reduction as last resort
        if self.config.enable_scope_reduction:
            return ResourceAdaptationStrategy.REDUCE_SCOPE
            
        return None
        
    async def _apply_adaptation_strategy(self, agent_id: str, strategy: ResourceAdaptationStrategy):
        """Apply resource adaptation strategy"""
        self.resource_stats['adaptations_triggered'] += 1
        
        logger.info(f"Applying adaptation strategy {strategy.value} for agent {agent_id}")
        
        if strategy == ResourceAdaptationStrategy.PAUSE_AND_RETRY:
            await self._pause_evolution(agent_id)
            self.resource_stats['evolutions_paused'] += 1
            
        elif strategy == ResourceAdaptationStrategy.DEGRADE_QUALITY:
            await self._degrade_evolution_quality(agent_id)
            self.resource_stats['quality_degradations'] += 1
            
        elif strategy == ResourceAdaptationStrategy.REDUCE_SCOPE:
            await self._reduce_evolution_scope(agent_id)
            self.resource_stats['scope_reductions'] += 1
            
        elif strategy == ResourceAdaptationStrategy.EMERGENCY_STOP:
            await self._emergency_stop_evolution(agent_id)
            self.resource_stats['emergency_stops'] += 1
            
        # Notify callbacks
        for callback in self.adaptation_callbacks:
            try:
                await callback(agent_id, strategy)
            except Exception as e:
                logger.error(f"Error in adaptation callback: {e}")
                
    async def _pause_evolution(self, agent_id: str):
        """Pause evolution for an agent"""
        if agent_id not in self.paused_evolutions:
            self.paused_evolutions[agent_id] = {
                'paused_at': time.time(),
                'reason': 'resource_constraint',
                'resume_attempts': 0
            }
            
        logger.info(f"Paused evolution for agent {agent_id}")
        
    async def _resume_evolution(self, agent_id: str):
        """Resume paused evolution"""
        if agent_id in self.paused_evolutions:
            paused_info = self.paused_evolutions[agent_id]
            paused_info['resume_attempts'] += 1
            
            # Check if we can resume
            if await self._check_evolution_feasibility("nightly", None):  # Simplified check
                del self.paused_evolutions[agent_id]
                self.resource_stats['evolutions_resumed'] += 1
                logger.info(f"Resumed evolution for agent {agent_id}")
            else:
                logger.debug(f"Cannot resume evolution for agent {agent_id} yet")
                
    async def _degrade_evolution_quality(self, agent_id: str):
        """Degrade evolution quality to reduce resource usage"""
        # This would involve loading lower-quality models or reducing complexity
        if self.adaptive_loader:
            # Switch to lighter model variants
            logger.info(f"Degrading evolution quality for agent {agent_id}")
            
    async def _reduce_evolution_scope(self, agent_id: str):
        """Reduce evolution scope to save resources"""
        # This would involve reducing the number of evolution steps or parameters
        logger.info(f"Reducing evolution scope for agent {agent_id}")
        
    async def _emergency_stop_evolution(self, agent_id: str):
        """Emergency stop evolution due to critical resource constraints"""
        logger.warning(f"Emergency stopping evolution for agent {agent_id}")
        
        # Force cleanup
        await self._cleanup_evolution_resources(agent_id)
        
    async def _cleanup_memory(self):
        """Cleanup memory during evolution"""
        self.resource_stats['memory_cleanups'] += 1
        
        # Unload cached models if adaptive loader is available
        if self.adaptive_loader:
            self.adaptive_loader.clear_cache()
            
        # Force garbage collection
        import gc
        gc.collect()
        
        logger.info("Performed memory cleanup")
        
    async def _cleanup_evolution_resources(self, evolution_id: str):
        """Cleanup resources after evolution"""
        # Unload models specific to this evolution
        if self.adaptive_loader:
            # In a real implementation, we'd track models per evolution
            pass
            
        # Remove from paused evolutions if present
        agent_id = evolution_id.split('_')[1] if '_' in evolution_id else evolution_id
        if agent_id in self.paused_evolutions:
            del self.paused_evolutions[agent_id]
            
    async def _load_evolution_models(self, agent: EvolvableAgent, evolution_type: str):
        """Load models for evolution with resource awareness"""
        if not self.adaptive_loader:
            return
            
        from ....core.resources.adaptive_loader import LoadingContext
        
        # Create context based on evolution type and current resources
        quality_preferences = {
            'emergency': 0.5,      # Lower quality for speed
            'nightly': 0.7,        # Balanced quality
            'breakthrough': 0.9    # High quality
        }
        
        max_loading_times = {
            'emergency': 30.0,     # 30 seconds max
            'nightly': 120.0,      # 2 minutes max
            'breakthrough': 300.0  # 5 minutes max
        }
        
        context = LoadingContext(
            task_type=evolution_type,
            priority_level=3 if evolution_type == "breakthrough" else 2,
            max_loading_time_seconds=max_loading_times.get(evolution_type, 120.0),
            quality_preference=quality_preferences.get(evolution_type, 0.7),
            resource_constraints=self.constraint_manager.get_constraint_template(evolution_type),
            allow_degraded_quality=True,
            cache_enabled=True
        )
        
        # Load evolution model
        model, loading_info = await self.adaptive_loader.load_model_adaptive(
            "base_evolution_model",
            context
        )
        
        if model:
            logger.info(f"Loaded evolution model for {evolution_type}: {loading_info}")
        else:
            logger.warning(f"Failed to load evolution model for {evolution_type}")
            
    async def _handle_constraint_violation(self, task_id: str, violation: ConstraintViolation):
        """Handle constraint violation during evolution"""
        self.resource_stats['resource_violations'] += 1
        
        logger.warning(f"Constraint violation in task {task_id}: {violation.message}")
        
        # Determine if we need immediate action
        if violation.severity in [ConstraintSeverity.CRITICAL, ConstraintSeverity.HIGH]:
            # Extract agent_id from task_id
            agent_id = task_id.split('_')[1] if '_' in task_id else task_id
            strategy = await self._determine_adaptation_strategy(agent_id, "unknown")
            if strategy:
                await self._apply_adaptation_strategy(agent_id, strategy)
                
    async def _handle_constraint_enforcement(self, task_id: str, action: str):
        """Handle constraint enforcement action"""
        logger.info(f"Constraint enforcement action '{action}' for task {task_id}")
        
        # Extract agent_id and handle action
        agent_id = task_id.split('_')[1] if '_' in task_id else task_id
        
        if action == "pause":
            await self._pause_evolution(agent_id)
        elif action == "interrupt":
            await self._emergency_stop_evolution(agent_id)
            
    async def _handle_resource_event(self, event_type: str, event_data: Dict[str, Any]):
        """Handle resource events from resource monitor"""
        logger.info(f"Resource event: {event_type} - {event_data}")
        
        # Handle specific events
        if event_type == "memory_spike":
            await self._cleanup_memory()
        elif event_type == "thermal_rise":
            self.resource_stats['thermal_throttles'] += 1
            # Pause all active evolutions temporarily
            for agent_id in list(self.active_evolutions.keys()):
                await self._pause_evolution(agent_id)
                
    def get_resource_constrained_status(self) -> Dict[str, Any]:
        """Get resource-constrained evolution status"""
        status = self.get_system_status()
        
        status.update({
            'resource_config': self.config.to_dict(),
            'current_resource_state': (
                self.current_resource_state.__dict__ 
                if self.current_resource_state 
                else None
            ),
            'paused_evolutions': len(self.paused_evolutions),
            'resource_stats': self.resource_stats.copy(),
            'device_evolution_capable': self.device_profiler.profile.evolution_capable,
            'current_evolution_suitability': (
                self.device_profiler.current_snapshot.evolution_suitability_score
                if self.device_profiler.current_snapshot
                else None
            )
        })
        
        return status