"""
Refactored Phase Controller

Breaks down the monolithic phase controller into smaller,
more manageable components with clear responsibilities.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
from datetime import datetime
import logging
import asyncio
from dataclasses import dataclass, field

from ..exceptions import PhaseExecutionError, handle_exception
from ..interfaces.base_interfaces import BasePhase, PhaseResult, BaseModel
from ..config.base_config import BaseConfig

logger = logging.getLogger(__name__)


@dataclass
class PhaseMetrics:
    """Metrics collected during phase execution."""
    
    phase_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def finalize(self, success: bool = True, error: Optional[str] = None):
        """Finalize metrics collection."""
        self.end_time = datetime.now()
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        self.success = success
        self.error_message = error


class PhaseValidator:
    """Validates phase configuration and prerequisites."""
    
    def __init__(self):
        self.validation_rules: Dict[str, List[callable]] = {}
    
    def add_rule(self, phase_name: str, rule: callable):
        """Add a validation rule for a phase."""
        if phase_name not in self.validation_rules:
            self.validation_rules[phase_name] = []
        self.validation_rules[phase_name].append(rule)
    
    def validate_phase(self, phase: BasePhase, model: BaseModel) -> List[str]:
        """Validate a phase before execution."""
        errors = []
        
        # Basic validation
        if not hasattr(phase, 'phase_name') or not phase.phase_name:
            errors.append("Phase must have a valid phase_name")
        
        if not hasattr(phase, 'config') or not phase.config:
            errors.append("Phase must have configuration")
        
        # Model validation
        if model and not model.is_loaded:
            errors.append("Model must be loaded before phase execution")
        
        # Custom validation rules
        phase_name = getattr(phase, 'phase_name', '')
        if phase_name in self.validation_rules:
            for rule in self.validation_rules[phase_name]:
                try:
                    result = rule(phase, model)
                    if isinstance(result, str):
                        errors.append(result)
                    elif isinstance(result, list):
                        errors.extend(result)
                except Exception as e:
                    errors.append(f"Validation rule failed: {str(e)}")
        
        return errors


class PhaseMonitor:
    """Monitors phase execution and collects metrics."""
    
    def __init__(self):
        self.active_phases: Dict[str, PhaseMetrics] = {}
        self.completed_phases: List[PhaseMetrics] = []
    
    def start_phase(self, phase_name: str) -> PhaseMetrics:
        """Start monitoring a phase."""
        metrics = PhaseMetrics(
            phase_name=phase_name,
            start_time=datetime.now()
        )
        self.active_phases[phase_name] = metrics
        logger.info(f"Started monitoring phase: {phase_name}")
        return metrics
    
    def end_phase(self, phase_name: str, success: bool = True, error: Optional[str] = None):
        """End monitoring a phase."""
        if phase_name in self.active_phases:
            metrics = self.active_phases.pop(phase_name)
            metrics.finalize(success, error)
            self.completed_phases.append(metrics)
            
            logger.info(
                f"Completed phase {phase_name}: "
                f"duration={metrics.duration_seconds:.2f}s, "
                f"success={success}"
            )
        
    def get_phase_metrics(self, phase_name: str) -> Optional[PhaseMetrics]:
        """Get metrics for a specific phase."""
        for metrics in self.completed_phases:
            if metrics.phase_name == phase_name:
                return metrics
        return self.active_phases.get(phase_name)
    
    def get_all_metrics(self) -> List[PhaseMetrics]:
        """Get all collected metrics."""
        return self.completed_phases.copy()


class PhaseOrchestrator:
    """Orchestrates phase execution with proper error handling and monitoring."""
    
    def __init__(self, validator: Optional[PhaseValidator] = None):
        self.validator = validator or PhaseValidator()
        self.monitor = PhaseMonitor()
        self.phase_registry: Dict[str, Type[BasePhase]] = {}
        self.middleware: List[callable] = []
    
    def register_phase(self, phase_name: str, phase_class: Type[BasePhase]):
        """Register a phase class."""
        self.phase_registry[phase_name] = phase_class
        logger.info(f"Registered phase: {phase_name}")
    
    def add_middleware(self, middleware: callable):
        """Add middleware for phase execution."""
        self.middleware.append(middleware)
    
    async def execute_phase(
        self, 
        phase: BasePhase, 
        model: BaseModel,
        validate_first: bool = True
    ) -> PhaseResult:
        """
        Execute a single phase with full monitoring and error handling.
        """
        phase_name = getattr(phase, 'phase_name', 'unknown')
        
        # Validation
        if validate_first:
            validation_errors = self.validator.validate_phase(phase, model)
            if validation_errors:
                error_msg = f"Phase validation failed: {'; '.join(validation_errors)}"
                logger.error(error_msg)
                return PhaseResult(
                    success=False,
                    phase_name=phase_name,
                    error=error_msg
                )
        
        # Start monitoring
        metrics = self.monitor.start_phase(phase_name)
        
        try:
            # Apply middleware
            for middleware_func in self.middleware:
                try:
                    await middleware_func(phase, model, "before")
                except Exception as e:
                    logger.warning(f"Middleware failed (before): {e}")
            
            # Execute phase
            logger.info(f"Executing phase: {phase_name}")
            result = await phase.run(model)
            
            # Apply middleware
            for middleware_func in self.middleware:
                try:
                    await middleware_func(phase, model, "after")
                except Exception as e:
                    logger.warning(f"Middleware failed (after): {e}")
            
            # Update monitoring
            self.monitor.end_phase(phase_name, result.success, result.error)
            
            # Enhance result with metrics
            if hasattr(result, 'metrics'):
                result.metrics.update(metrics.custom_metrics)
            
            return result
            
        except Exception as e:
            error_msg = f"Phase {phase_name} failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            self.monitor.end_phase(phase_name, False, error_msg)
            
            return PhaseResult(
                success=False,
                phase_name=phase_name,
                error=error_msg
            )
    
    async def execute_pipeline(
        self, 
        phases: List[BasePhase], 
        model: BaseModel,
        stop_on_failure: bool = True
    ) -> List[PhaseResult]:
        """
        Execute a sequence of phases.
        """
        results = []
        current_model = model
        
        logger.info(f"Starting pipeline execution with {len(phases)} phases")
        
        for i, phase in enumerate(phases):
            phase_name = getattr(phase, 'phase_name', f'phase_{i}')
            
            try:
                result = await self.execute_phase(phase, current_model)
                results.append(result)
                
                if not result.success:
                    logger.error(f"Phase {phase_name} failed: {result.error}")
                    if stop_on_failure:
                        logger.info("Stopping pipeline due to phase failure")
                        break
                else:
                    # Update model if phase returned a new one
                    if result.model:
                        current_model = result.model
                
            except Exception as e:
                error_msg = f"Unexpected error in phase {phase_name}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                
                results.append(PhaseResult(
                    success=False,
                    phase_name=phase_name,
                    error=error_msg
                ))
                
                if stop_on_failure:
                    break
        
        logger.info(f"Pipeline execution completed. {len(results)} phases executed")
        return results
    
    def create_phase(self, phase_name: str, config: Dict[str, Any]) -> BasePhase:
        """Create a phase instance from registry."""
        if phase_name not in self.phase_registry:
            raise PhaseExecutionError(
                f"Phase '{phase_name}' not found in registry",
                phase_name=phase_name
            )
        
        phase_class = self.phase_registry[phase_name]
        return phase_class(phase_name, config)
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get summary of pipeline execution."""
        all_metrics = self.monitor.get_all_metrics()
        
        if not all_metrics:
            return {"message": "No phases executed yet"}
        
        total_duration = sum(m.duration_seconds for m in all_metrics)
        success_count = sum(1 for m in all_metrics if m.success)
        
        return {
            "total_phases": len(all_metrics),
            "successful_phases": success_count,
            "failed_phases": len(all_metrics) - success_count,
            "total_duration_seconds": total_duration,
            "average_phase_duration": total_duration / len(all_metrics),
            "phases": [
                {
                    "name": m.phase_name,
                    "duration": m.duration_seconds,
                    "success": m.success,
                    "error": m.error_message
                }
                for m in all_metrics
            ]
        }


class PhaseFactory:
    """Factory for creating configured phase instances."""
    
    def __init__(self):
        self.templates: Dict[str, Dict[str, Any]] = {}
        self.orchestrator = PhaseOrchestrator()
    
    def register_template(self, name: str, template: Dict[str, Any]):
        """Register a phase template."""
        self.templates[name] = template
    
    def create_from_template(self, template_name: str, overrides: Optional[Dict[str, Any]] = None) -> BasePhase:
        """Create a phase from a template."""
        if template_name not in self.templates:
            raise PhaseExecutionError(f"Template '{template_name}' not found")
        
        template = self.templates[template_name].copy()
        if overrides:
            template.update(overrides)
        
        phase_type = template.pop('type')
        return self.orchestrator.create_phase(phase_type, template)