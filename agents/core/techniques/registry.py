"""Registry for reasoning techniques."""

from typing import Dict, Type, Any, List, Optional, TypeVar, Generic
from dataclasses import dataclass
import asyncio
import logging
from .base import BaseTechnique, TechniqueResult

logger = logging.getLogger(__name__)

# Type variables for input and output
I = TypeVar('I')  # Input type
O = TypeVar('O')  # Output type

class TechniqueRegistryError(Exception):
    """Base exception for technique registry."""
    pass

@dataclass
class TechniqueInfo:
    """Information about a registered technique."""
    name: str
    description: str
    technique_class: Type[BaseTechnique]
    input_type: Type
    output_type: Type
    parameters: Dict[str, Any]
    tags: List[str]
    version: str
    created_at: str
    updated_at: str

class TechniqueRegistry(Generic[I, O]):
    """Registry for managing and selecting techniques."""
    
    def __init__(self):
        self._techniques: Dict[str, TechniqueInfo] = {}
        self._instances: Dict[str, BaseTechnique] = {}
        self._performance_history: Dict[str, List[float]] = {}
    
    def register(
        self,
        technique_class: Type[BaseTechnique],
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        version: str = "1.0.0",
        **kwargs
    ) -> None:
        """Register a new technique."""
        name = name or technique_class.__name__
        
        if name in self._techniques:
            raise TechniqueRegistryError(f"Technique {name} already registered")
        
        info = TechniqueInfo(
            name=name,
            description=description or technique_class.__doc__ or "",
            technique_class=technique_class,
            input_type=kwargs.get('input_type', Any),
            output_type=kwargs.get('output_type', Any),
            parameters=parameters or {},
            tags=tags or [],
            version=version,
            created_at=kwargs.get('created_at', ''),
            updated_at=kwargs.get('updated_at', '')
        )
        
        self._techniques[name] = info
        self._performance_history[name] = []
        logger.info(f"Registered technique: {name}")
    
    def unregister(self, name: str) -> None:
        """Unregister a technique."""
        if name not in self._techniques:
            raise TechniqueRegistryError(f"Technique {name} not registered")
        
        if name in self._instances:
            del self._instances[name]
        
        del self._techniques[name]
        del self._performance_history[name]
        logger.info(f"Unregistered technique: {name}")
    
    async def get_instance(self, name: str) -> BaseTechnique[I, O]:
        """Get or create an instance of a technique."""
        if name not in self._techniques:
            raise TechniqueRegistryError(f"Technique {name} not registered")
        
        if name not in self._instances:
            info = self._techniques[name]
            instance = info.technique_class(
                name=info.name,
                description=info.description,
                **info.parameters
            )
            await instance.initialize()
            self._instances[name] = instance
        
        return self._instances[name]
    
    def get_info(self, name: str) -> TechniqueInfo:
        """Get information about a technique."""
        if name not in self._techniques:
            raise TechniqueRegistryError(f"Technique {name} not registered")
        
        return self._techniques[name]
    
    def list_techniques(
        self,
        tags: Optional[List[str]] = None,
        input_type: Optional[Type] = None,
        output_type: Optional[Type] = None
    ) -> List[TechniqueInfo]:
        """List registered techniques with optional filtering."""
        techniques = list(self._techniques.values())
        
        if tags:
            techniques = [
                t for t in techniques
                if any(tag in t.tags for tag in tags)
            ]
        
        if input_type:
            techniques = [
                t for t in techniques
                if issubclass(input_type, t.input_type)
            ]
        
        if output_type:
            techniques = [
                t for t in techniques
                if issubclass(output_type, t.output_type)
            ]
        
        return techniques
    
    async def select_technique(
        self,
        input_data: I,
        tags: Optional[List[str]] = None
    ) -> BaseTechnique[I, O]:
        """Select the most appropriate technique for input data."""
        candidates = self.list_techniques(tags=tags)
        
        if not candidates:
            raise TechniqueRegistryError("No suitable techniques found")
        
        # Select based on performance history
        best_technique = max(
            candidates,
            key=lambda t: sum(self._performance_history[t.name]) / len(self._performance_history[t.name])
            if self._performance_history[t.name] else 0
        )
        
        return await self.get_instance(best_technique.name)
    
    async def execute(
        self,
        input_data: I,
        technique_name: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> TechniqueResult[O]:
        """Execute a technique on input data."""
        if technique_name:
            technique = await self.get_instance(technique_name)
        else:
            technique = await self.select_technique(input_data, tags)
        
        result = await technique(input_data)
        
        # Update performance history
        self._performance_history[technique.name].append(
            result.metrics.confidence
        )
        
        return result
    
    async def cleanup(self) -> None:
        """Clean up all technique instances."""
        for instance in self._instances.values():
            await instance.cleanup()
        self._instances.clear()
    
    def get_performance_history(self, name: str) -> List[float]:
        """Get performance history for a technique."""
        if name not in self._techniques:
            raise TechniqueRegistryError(f"Technique {name} not registered")
        
        return self._performance_history[name].copy()
    
    def clear_performance_history(self, name: Optional[str] = None) -> None:
        """Clear performance history for one or all techniques."""
        if name:
            if name not in self._techniques:
                raise TechniqueRegistryError(f"Technique {name} not registered")
            self._performance_history[name].clear()
        else:
            for history in self._performance_history.values():
                history.clear()
