"""
Orchestrator Registry

Centralized registry for managing orchestrator discovery, instantiation, and lifecycle.
This eliminates the scattered instantiation patterns identified in Agent 1's analysis.
"""

import logging
from typing import Any, Dict, List, Optional, Type, Callable
from dataclasses import dataclass, field

from .interfaces import OrchestrationInterface, ConfigurationSpec

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorRegistration:
    """Registration information for an orchestrator type."""
    orchestrator_type: str
    orchestrator_class: Type[OrchestrationInterface]
    factory_function: Optional[Callable] = None
    default_config: Optional[ConfigurationSpec] = None
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    priority: int = 100
    enabled: bool = True
    
    
class OrchestratorRegistry:
    """
    Central registry for orchestrator types and instances.
    
    This registry provides:
    1. Type registration and discovery
    2. Factory-based instantiation with proper configuration
    3. Dependency management
    4. Lifecycle tracking
    """
    
    def __init__(self):
        """Initialize the orchestrator registry."""
        self._registrations: Dict[str, OrchestratorRegistration] = {}
        self._instances: Dict[str, OrchestrationInterface] = {}
        
        logger.info("Orchestrator registry initialized")
    
    def register_type(
        self,
        orchestrator_type: str,
        orchestrator_class: Type[OrchestrationInterface],
        factory_function: Optional[Callable] = None,
        default_config: Optional[ConfigurationSpec] = None,
        description: str = "",
        dependencies: Optional[List[str]] = None,
        priority: int = 100,
        enabled: bool = True
    ) -> bool:
        """
        Register an orchestrator type.
        
        Args:
            orchestrator_type: Unique type identifier
            orchestrator_class: Orchestrator class
            factory_function: Optional custom factory function
            default_config: Default configuration
            description: Human-readable description
            dependencies: List of orchestrator types this depends on
            priority: Initialization priority (lower = earlier)
            enabled: Whether this orchestrator type is enabled
            
        Returns:
            bool: True if registration successful
        """
        try:
            if orchestrator_type in self._registrations:
                logger.warning(f"Orchestrator type {orchestrator_type} already registered")
                return False
            
            registration = OrchestratorRegistration(
                orchestrator_type=orchestrator_type,
                orchestrator_class=orchestrator_class,
                factory_function=factory_function,
                default_config=default_config,
                description=description,
                dependencies=dependencies or [],
                priority=priority,
                enabled=enabled
            )
            
            self._registrations[orchestrator_type] = registration
            
            logger.info(f"Registered orchestrator type: {orchestrator_type}")
            return True
            
        except Exception as e:
            logger.exception(f"Failed to register orchestrator type {orchestrator_type}: {e}")
            return False
    
    def unregister_type(self, orchestrator_type: str) -> bool:
        """
        Unregister an orchestrator type.
        
        Args:
            orchestrator_type: Type to unregister
            
        Returns:
            bool: True if successful
        """
        try:
            if orchestrator_type not in self._registrations:
                logger.warning(f"Orchestrator type {orchestrator_type} not registered")
                return False
            
            # Check if any instances exist
            active_instances = [
                instance_id for instance_id, instance in self._instances.items()
                if orchestrator_type in instance.orchestrator_id
            ]
            
            if active_instances:
                logger.error(f"Cannot unregister {orchestrator_type}: active instances {active_instances}")
                return False
            
            del self._registrations[orchestrator_type]
            logger.info(f"Unregistered orchestrator type: {orchestrator_type}")
            return True
            
        except Exception as e:
            logger.exception(f"Failed to unregister orchestrator type {orchestrator_type}: {e}")
            return False
    
    def create_instance(
        self,
        orchestrator_type: str,
        instance_id: Optional[str] = None,
        config: Optional[ConfigurationSpec] = None,
        **kwargs
    ) -> Optional[OrchestrationInterface]:
        """
        Create an orchestrator instance.
        
        Args:
            orchestrator_type: Type of orchestrator to create
            instance_id: Optional custom instance ID
            config: Optional custom configuration
            **kwargs: Additional arguments for the orchestrator constructor
            
        Returns:
            OrchestrationInterface: Created instance or None if failed
        """
        try:
            if orchestrator_type not in self._registrations:
                logger.error(f"Orchestrator type {orchestrator_type} not registered")
                return None
            
            registration = self._registrations[orchestrator_type]
            
            if not registration.enabled:
                logger.warning(f"Orchestrator type {orchestrator_type} is disabled")
                return None
            
            # Use factory function if provided, otherwise use class constructor
            if registration.factory_function:
                instance = registration.factory_function(
                    orchestrator_type=orchestrator_type,
                    orchestrator_id=instance_id,
                    config=config or registration.default_config,
                    **kwargs
                )
            else:
                instance = registration.orchestrator_class(
                    orchestrator_type=orchestrator_type,
                    orchestrator_id=instance_id,
                    **kwargs
                )
            
            if instance:
                self._instances[instance.orchestrator_id] = instance
                logger.info(f"Created orchestrator instance: {instance.orchestrator_id}")
            
            return instance
            
        except Exception as e:
            logger.exception(f"Failed to create orchestrator instance {orchestrator_type}: {e}")
            return None
    
    def get_instance(self, instance_id: str) -> Optional[OrchestrationInterface]:
        """
        Get an orchestrator instance by ID.
        
        Args:
            instance_id: Instance identifier
            
        Returns:
            OrchestrationInterface: Instance or None if not found
        """
        return self._instances.get(instance_id)
    
    def remove_instance(self, instance_id: str) -> bool:
        """
        Remove an orchestrator instance from the registry.
        
        Args:
            instance_id: Instance to remove
            
        Returns:
            bool: True if removed successfully
        """
        try:
            if instance_id not in self._instances:
                logger.warning(f"Instance {instance_id} not found in registry")
                return False
            
            del self._instances[instance_id]
            logger.info(f"Removed orchestrator instance: {instance_id}")
            return True
            
        except Exception as e:
            logger.exception(f"Failed to remove instance {instance_id}: {e}")
            return False
    
    def list_types(self, enabled_only: bool = True) -> List[str]:
        """
        List registered orchestrator types.
        
        Args:
            enabled_only: Only return enabled types
            
        Returns:
            List[str]: List of orchestrator type names
        """
        if enabled_only:
            return [
                orch_type for orch_type, reg in self._registrations.items()
                if reg.enabled
            ]
        else:
            return list(self._registrations.keys())
    
    def list_instances(self) -> List[str]:
        """
        List active orchestrator instances.
        
        Returns:
            List[str]: List of instance IDs
        """
        return list(self._instances.keys())
    
    def get_type_info(self, orchestrator_type: str) -> Optional[OrchestratorRegistration]:
        """
        Get registration information for an orchestrator type.
        
        Args:
            orchestrator_type: Type to get info for
            
        Returns:
            OrchestratorRegistration: Registration info or None
        """
        return self._registrations.get(orchestrator_type)
    
    def get_dependency_order(self) -> List[str]:
        """
        Get orchestrator types in dependency order for initialization.
        
        Returns:
            List[str]: Orchestrator types in initialization order
        """
        try:
            # Simple topological sort
            visited = set()
            temp_visited = set()
            order = []
            
            def visit(orch_type: str):
                if orch_type in temp_visited:
                    raise ValueError(f"Circular dependency detected involving {orch_type}")
                if orch_type in visited:
                    return
                
                temp_visited.add(orch_type)
                
                registration = self._registrations.get(orch_type)
                if registration:
                    for dep_type in registration.dependencies:
                        if dep_type in self._registrations:
                            visit(dep_type)
                
                temp_visited.remove(orch_type)
                visited.add(orch_type)
                order.append(orch_type)
            
            # Visit all enabled orchestrator types
            for orch_type in self.list_types(enabled_only=True):
                if orch_type not in visited:
                    visit(orch_type)
            
            # Sort by priority within dependency constraints
            priority_map = {
                orch_type: self._registrations[orch_type].priority
                for orch_type in order
            }
            
            return sorted(order, key=lambda x: priority_map.get(x, 100))
            
        except Exception as e:
            logger.exception(f"Failed to determine dependency order: {e}")
            return self.list_types(enabled_only=True)
    
    def create_all_instances(self, configs: Optional[Dict[str, ConfigurationSpec]] = None) -> Dict[str, OrchestrationInterface]:
        """
        Create instances of all enabled orchestrator types.
        
        Args:
            configs: Optional configurations for each orchestrator type
            
        Returns:
            Dict[str, OrchestrationInterface]: Created instances by type
        """
        instances = {}
        configs = configs or {}
        
        for orch_type in self.get_dependency_order():
            try:
                config = configs.get(orch_type)
                instance = self.create_instance(orch_type, config=config)
                
                if instance:
                    instances[orch_type] = instance
                else:
                    logger.error(f"Failed to create instance of {orch_type}")
                    
            except Exception as e:
                logger.exception(f"Error creating instance of {orch_type}: {e}")
        
        logger.info(f"Created {len(instances)} orchestrator instances")
        return instances
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        enabled_types = self.list_types(enabled_only=True)
        all_types = self.list_types(enabled_only=False)
        
        return {
            'total_registered_types': len(all_types),
            'enabled_types': len(enabled_types),
            'disabled_types': len(all_types) - len(enabled_types),
            'active_instances': len(self._instances),
            'registered_types': all_types,
            'enabled_type_names': enabled_types,
            'active_instance_ids': list(self._instances.keys()),
        }