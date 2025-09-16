"""
Circuit Breaker Implementation

Circuit breaker pattern for transport reliability.
Consolidates circuit breaker functionality from edge chat engine.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, Optional
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failures detected, circuit open
    HALF_OPEN = "half_open" # Testing if service recovered


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""
    pass


class CircuitBreaker:
    """Circuit breaker for transport reliability"""
    
    def __init__(self, config: Dict[str, Any]):
        # Configuration
        self.failure_threshold = config.get("failure_threshold", 5)
        self.success_threshold = config.get("success_threshold", 3)
        self.timeout_seconds = config.get("timeout_seconds", 60)
        self.half_open_max_calls = config.get("half_open_max_calls", 5)
        
        # State tracking
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_calls = 0
        
        # Metrics
        self.total_calls = 0
        self.total_failures = 0
        self.total_successes = 0
        self.state_changes = 0
        
        logger.info(f"Circuit breaker initialized: threshold={self.failure_threshold}, timeout={self.timeout_seconds}s")
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        self.total_calls += 1
        
        # Check if circuit should be opened or closed
        await self._update_state()
        
        # Handle based on current state
        if self.state == CircuitState.OPEN:
            self.total_failures += 1
            raise CircuitBreakerOpenError("Circuit breaker is open")
        
        elif self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls >= self.half_open_max_calls:
                self.total_failures += 1
                raise CircuitBreakerOpenError("Circuit breaker half-open call limit reached")
            
            self.half_open_calls += 1
        
        # Execute the function
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
            
        except Exception as e:
            await self._on_failure()
            raise e
    
    async def _update_state(self) -> None:
        """Update circuit breaker state based on current conditions"""
        if self.state == CircuitState.OPEN:
            # Check if timeout has elapsed
            if self._should_attempt_reset():
                await self._transition_to_half_open()
        
        elif self.state == CircuitState.CLOSED:
            # Check if failure threshold exceeded
            if self.failure_count >= self.failure_threshold:
                await self._transition_to_open()
        
        elif self.state == CircuitState.HALF_OPEN:
            # State transitions handled in success/failure callbacks
            pass
    
    async def _on_success(self) -> None:
        """Handle successful operation"""
        self.total_successes += 1
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            
            # Check if enough successes to close circuit
            if self.success_count >= self.success_threshold:
                await self._transition_to_closed()
        
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0
    
    async def _on_failure(self) -> None:
        """Handle failed operation"""
        self.total_failures += 1
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitState.HALF_OPEN:
            # Failure in half-open state immediately opens circuit
            await self._transition_to_open()
        
        elif self.state == CircuitState.CLOSED:
            # Check if failure threshold exceeded
            if self.failure_count >= self.failure_threshold:
                await self._transition_to_open()
    
    async def _transition_to_open(self) -> None:
        """Transition to OPEN state"""
        if self.state != CircuitState.OPEN:
            logger.warning(f"Circuit breaker opening: {self.failure_count} failures")
            self.state = CircuitState.OPEN
            self.state_changes += 1
            self.last_failure_time = datetime.now()
            await self._on_state_change(CircuitState.OPEN)
    
    async def _transition_to_half_open(self) -> None:
        """Transition to HALF_OPEN state"""
        if self.state != CircuitState.HALF_OPEN:
            logger.info("Circuit breaker transitioning to half-open")
            self.state = CircuitState.HALF_OPEN
            self.state_changes += 1
            self.success_count = 0
            self.half_open_calls = 0
            await self._on_state_change(CircuitState.HALF_OPEN)
    
    async def _transition_to_closed(self) -> None:
        """Transition to CLOSED state"""
        if self.state != CircuitState.CLOSED:
            logger.info(f"Circuit breaker closing: {self.success_count} successes")
            self.state = CircuitState.CLOSED
            self.state_changes += 1
            self.failure_count = 0
            self.success_count = 0
            self.half_open_calls = 0
            await self._on_state_change(CircuitState.CLOSED)
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        
        elapsed = datetime.now() - self.last_failure_time
        return elapsed >= timedelta(seconds=self.timeout_seconds)
    
    async def _on_state_change(self, new_state: CircuitState) -> None:
        """Handle state change events (can be overridden for custom behavior)"""
        # This method can be extended for custom state change handling
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        uptime_seconds = 0
        if self.last_failure_time:
            uptime_seconds = (datetime.now() - self.last_failure_time).total_seconds()
        
        success_rate = 0.0
        if self.total_calls > 0:
            success_rate = self.total_successes / self.total_calls
        
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "half_open_calls": self.half_open_calls,
            "total_calls": self.total_calls,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "success_rate": success_rate,
            "state_changes": self.state_changes,
            "uptime_seconds": uptime_seconds,
            "configuration": {
                "failure_threshold": self.failure_threshold,
                "success_threshold": self.success_threshold,
                "timeout_seconds": self.timeout_seconds,
                "half_open_max_calls": self.half_open_max_calls
            }
        }
    
    def reset(self) -> None:
        """Manually reset circuit breaker to closed state"""
        logger.info("Circuit breaker manually reset")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0
        self.state_changes += 1
    
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)"""
        return self.state == CircuitState.CLOSED
    
    def is_open(self) -> bool:
        """Check if circuit is open (failing fast)"""
        return self.state == CircuitState.OPEN
    
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)"""
        return self.state == CircuitState.HALF_OPEN
    
    def __repr__(self) -> str:
        return (f"CircuitBreaker(state={self.state.value}, "
                f"failures={self.failure_count}/{self.failure_threshold}, "
                f"successes={self.success_count}/{self.success_threshold})")


class CircuitBreakerManager:
    """Manager for multiple circuit breakers"""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
    def get_circuit_breaker(self, name: str, config: Dict[str, Any]) -> CircuitBreaker:
        """Get or create circuit breaker by name"""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(config)
            logger.info(f"Created circuit breaker: {name}")
        
        return self.circuit_breakers[name]
    
    def remove_circuit_breaker(self, name: str) -> bool:
        """Remove circuit breaker by name"""
        if name in self.circuit_breakers:
            del self.circuit_breakers[name]
            logger.info(f"Removed circuit breaker: {name}")
            return True
        return False
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers"""
        return {
            name: cb.get_stats() 
            for name, cb in self.circuit_breakers.items()
        }
    
    def reset_all(self) -> None:
        """Reset all circuit breakers"""
        for cb in self.circuit_breakers.values():
            cb.reset()
        logger.info("All circuit breakers reset")
    
    def __len__(self) -> int:
        return len(self.circuit_breakers)
    
    def __contains__(self, name: str) -> bool:
        return name in self.circuit_breakers
