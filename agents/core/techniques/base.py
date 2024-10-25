"""Base classes for reasoning techniques."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Type variables for input and output
I = TypeVar('I')  # Input type
O = TypeVar('O')  # Output type

class TechniqueError(Exception):
    """Base exception for technique errors."""
    pass

class ValidationError(TechniqueError):
    """Error during input/output validation."""
    pass

class ExecutionError(TechniqueError):
    """Error during technique execution."""
    pass

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
        
        # Input validation
        try:
            is_valid = await self.validate_input(input_data)
            if not is_valid:
                raise ValidationError(f"Invalid input for technique {self.name}")
        except Exception as e:
            raise ValidationError(f"Input validation failed: {str(e)}")
        
        # Execute technique
        result = await self.execute(input_data)
        
        # Output validation
        try:
            is_valid = await self.validate_output(result.output)
            if not is_valid:
                raise ValidationError(f"Invalid output from technique {self.name}")
        except Exception as e:
            raise ValidationError(f"Output validation failed: {str(e)}")
        
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
