# rag_system/error_handling/error_control.py

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
from .error_handler import error_handler, AIVillageException

logger = logging.getLogger(__name__)

@dataclass
class ErrorControl:
    """
    Error control system that manages error thresholds and recovery strategies.
    """
    max_retries: int = 3
    retry_delay: float = 1.0
    error_thresholds: Dict[str, int] = field(default_factory=lambda: {
        'CRITICAL': 1,
        'ERROR': 5,
        'WARNING': 10
    })
    component_thresholds: Dict[str, Dict[str, int]] = field(default_factory=dict)
    recovery_strategies: Dict[str, callable] = field(default_factory=dict)
    error_counts: Dict[str, int] = field(default_factory=lambda: {
        'CRITICAL': 0,
        'ERROR': 0,
        'WARNING': 0
    })
    component_error_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def __post_init__(self):
        error_handler.register_error_callback(self.handle_error_callback)

    def handle_error_callback(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Handle error callback from error_handler."""
        if isinstance(error, AIVillageException):
            severity = error.severity
            component = error.component
        else:
            severity = 'ERROR'
            component = context.get('component', 'unknown') if context else 'unknown'

        self.error_counts[severity] = self.error_counts.get(severity, 0) + 1
        
        if component not in self.component_error_counts:
            self.component_error_counts[component] = {}
        self.component_error_counts[component][severity] = self.component_error_counts[component].get(severity, 0) + 1

        self.check_thresholds(severity, component)

    def check_thresholds(self, severity: str, component: str):
        """Check if error thresholds have been exceeded."""
        # Check global thresholds
        if self.error_counts[severity] >= self.error_thresholds[severity]:
            self.trigger_recovery(severity, component)

        # Check component-specific thresholds
        if component in self.component_thresholds:
            if self.component_error_counts[component][severity] >= self.component_thresholds[component][severity]:
                self.trigger_recovery(severity, component)

    def trigger_recovery(self, severity: str, component: str):
        """Trigger appropriate recovery strategy."""
        if component in self.recovery_strategies:
            try:
                self.recovery_strategies[component](severity)
            except Exception as e:
                logger.error(f"Error in recovery strategy for {component}: {str(e)}")
        else:
            self.default_recovery(severity, component)

    def default_recovery(self, severity: str, component: str):
        """Default recovery strategy."""
        logger.warning(f"No specific recovery strategy for {component}. Using default recovery.")
        if severity == 'CRITICAL':
            self.reset_component(component)
        elif severity == 'ERROR':
            self.pause_component(component)
        else:
            self.log_warning(component)

    def reset_component(self, component: str):
        """Reset a component to its initial state."""
        logger.info(f"Resetting component: {component}")
        # Implementation would depend on component-specific reset logic
        self.component_error_counts[component] = {
            'CRITICAL': 0,
            'ERROR': 0,
            'WARNING': 0
        }

    def pause_component(self, component: str):
        """Temporarily pause a component."""
        logger.info(f"Pausing component: {component}")
        # Implementation would depend on component-specific pause logic

    def log_warning(self, component: str):
        """Log a warning about component errors."""
        logger.warning(f"Multiple warnings in component: {component}")

    def register_recovery_strategy(self, component: str, strategy: callable):
        """Register a custom recovery strategy for a component."""
        self.recovery_strategies[component] = strategy

    def set_component_threshold(self, component: str, severity: str, threshold: int):
        """Set error threshold for a specific component."""
        if component not in self.component_thresholds:
            self.component_thresholds[component] = {}
        self.component_thresholds[component][severity] = threshold

    def get_error_status(self) -> Dict[str, Any]:
        """Get current error status."""
        return {
            'error_counts': dict(self.error_counts),
            'component_error_counts': {
                component: dict(counts)
                for component, counts in self.component_error_counts.items()
            },
            'thresholds': {
                'global': dict(self.error_thresholds),
                'component': {
                    component: dict(thresholds)
                    for component, thresholds in self.component_thresholds.items()
                }
            }
        }

    async def monitor_error_rates(self, interval: int = 60):
        """Monitor error rates and trigger alerts if necessary."""
        while True:
            try:
                metrics = error_handler.get_error_metrics()
                for component, errors in metrics['component_errors'].items():
                    if component in self.component_thresholds:
                        if errors > self.component_thresholds[component].get('ERROR', float('inf')):
                            logger.error(f"Error rate threshold exceeded for component: {component}")
                            self.trigger_recovery('ERROR', component)
                
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Error in error rate monitoring: {str(e)}")
                await asyncio.sleep(interval)

# Create singleton instance
error_control = ErrorControl()

# Start error rate monitoring
asyncio.create_task(error_control.monitor_error_rates())
