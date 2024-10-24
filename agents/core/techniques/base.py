"""Base classes for MAGI reasoning techniques."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic
from dataclasses import dataclass
from datetime import datetime
import asyncio
import logging
from ..core.exceptions import MAGIException, ExecutionError
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Type variables for input and output
I = TypeVar('I')  # Input type
O = TypeVar('O')  # Output type

@dataclass
class TechniqueMetrics:
    """Metrics for technique execution."""
    execution_time: float
    success: bool
    confidence: float
    uncertainty: float
    timestamp: datetime
    additional_metrics: Dict[str, Any] = None

@dataclass
class TechniqueResult(Generic[O]):
    """Result of technique execution."""
    output: O
    metrics: TechniqueMetrics
    intermediate_steps: List[Dict[str, Any]]
    reasoning_trace: List[str]

class BaseTechnique(Generic[I, O], ABC):
    """Base class for all reasoning techniques."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.history: List[TechniqueResult[O]] = []
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize technique-specific resources."""
        pass
    
    @abstractmethod
    async def execute(self, input_data: I) -> TechniqueResult[O]:
        """Execute the technique on input data."""
        pass
    
    @abstractmethod
    async def validate_input(self, input_data: I) -> bool:
        """Validate input data."""
        pass
    
    @abstractmethod
    async def validate_output(self, output_data: O) -> bool:
        """Validate output data."""
        pass
    
    async def __call__(self, input_data: I) -> TechniqueResult[O]:
        """Make the technique callable."""
        if not self._initialized:
            await self.initialize()
            self._initialized = True
        
        if not await self.validate_input(input_data):
            raise ExecutionError(f"Invalid input for technique {self.name}")
        
        result = await self.execute(input_data)
        
        if not await self.validate_output(result.output):
            raise ExecutionError(f"Invalid output from technique {self.name}")
        
        self.history.append(result)
        return result
    
    async def cleanup(self) -> None:
        """Clean up technique resources."""
        pass
    
    def get_history(self) -> List[TechniqueResult[O]]:
        """Get execution history."""
        return self.history.copy()
    
    def clear_history(self) -> None:
        """Clear execution history."""
        self.history.clear()

class CompositeTechnique(BaseTechnique[I, O]):
    """Base class for techniques composed of multiple sub-techniques."""
    
    def __init__(
        self,
        name: str,
        description: str,
        techniques: List[BaseTechnique]
    ):
        super().__init__(name, description)
        self.techniques = techniques
    
    async def initialize(self) -> None:
        """Initialize all sub-techniques."""
        for technique in self.techniques:
            await technique.initialize()
    
    async def cleanup(self) -> None:
        """Clean up all sub-techniques."""
        for technique in self.techniques:
            await technique.cleanup()

class AdaptiveTechnique(BaseTechnique[I, O]):
    """Base class for techniques that adapt based on performance."""
    
    def __init__(
        self,
        name: str,
        description: str,
        learning_rate: float = 0.01
    ):
        super().__init__(name, description)
        self.learning_rate = learning_rate
        self.performance_history: List[float] = []
    
    @abstractmethod
    async def adapt(self, performance: float) -> None:
        """Adapt technique based on performance."""
        pass
    
    async def update_performance(self, performance: float) -> None:
        """Update performance history and adapt."""
        self.performance_history.append(performance)
        await self.adapt(performance)

class ProbabilisticTechnique(BaseTechnique[I, O]):
    """Base class for techniques that use probabilistic reasoning."""
    
    def __init__(
        self,
        name: str,
        description: str,
        confidence_threshold: float = 0.7
    ):
        super().__init__(name, description)
        self.confidence_threshold = confidence_threshold
    
    @abstractmethod
    async def estimate_confidence(self, result: O) -> float:
        """Estimate confidence in the result."""
        pass
    
    @abstractmethod
    async def estimate_uncertainty(self, result: O) -> float:
        """Estimate uncertainty in the result."""
        pass

class IterativeTechnique(BaseTechnique[I, O]):
    """Base class for techniques that use iterative refinement."""
    
    def __init__(
        self,
        name: str,
        description: str,
        max_iterations: int = 10,
        convergence_threshold: float = 0.01
    ):
        super().__init__(name, description)
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
    
    @abstractmethod
    async def should_continue(self, current: O, previous: Optional[O]) -> bool:
        """Determine if iteration should continue."""
        pass
    
    @abstractmethod
    async def refine(self, current: O) -> O:
        """Refine the current result."""
        pass

class EnsembleTechnique(BaseTechnique[I, O]):
    """Base class for techniques that combine multiple approaches."""
    
    def __init__(
        self,
        name: str,
        description: str,
        techniques: List[BaseTechnique[I, O]],
        weights: Optional[List[float]] = None
    ):
        super().__init__(name, description)
        self.techniques = techniques
        self.weights = weights or [1.0] * len(techniques)
        
        if len(self.weights) != len(techniques):
            raise ValueError("Number of weights must match number of techniques")
    
    @abstractmethod
    async def combine_results(
        self,
        results: List[TechniqueResult[O]]
    ) -> TechniqueResult[O]:
        """Combine results from multiple techniques."""
        pass
    
    async def execute(self, input_data: I) -> TechniqueResult[O]:
        """Execute all techniques and combine results."""
        results = []
        for technique in self.techniques:
            try:
                result = await technique(input_data)
                results.append(result)
            except Exception as e:
                logger.warning(f"Technique {technique.name} failed: {str(e)}")
        
        if not results:
            raise ExecutionError("All techniques failed")
        
        return await self.combine_results(results)

class MetaTechnique(BaseTechnique[I, O]):
    """Base class for techniques that learn to select other techniques."""
    
    def __init__(
        self,
        name: str,
        description: str,
        techniques: Dict[str, BaseTechnique[I, O]]
    ):
        super().__init__(name, description)
        self.techniques = techniques
        self.technique_performance: Dict[str, List[float]] = {
            name: [] for name in techniques
        }
    
    @abstractmethod
    async def select_technique(self, input_data: I) -> str:
        """Select the most appropriate technique for the input."""
        pass
    
    async def execute(self, input_data: I) -> TechniqueResult[O]:
        """Execute the selected technique."""
        technique_name = await self.select_technique(input_data)
        technique = self.techniques[technique_name]
        return await technique(input_data)
    
    async def update_performance(
        self,
        technique_name: str,
        performance: float
    ) -> None:
        """Update performance history for a technique."""
        self.technique_performance[technique_name].append(performance)
