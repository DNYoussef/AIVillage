"""Base component for RAG system modules."""

from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import logging
from datetime import datetime
from ..utils.error_handling import log_and_handle_errors, ErrorContext

logger = logging.getLogger(__name__)

class BaseComponent(ABC):
    """Abstract base class for RAG system components."""
    
    def __init__(self):
        """Initialize base component."""
        self.initialized = False
        self.last_error: Optional[str] = None
        self.stats = {
            "initialization_time": None,
            "last_update": None,
            "error_count": 0,
            "operation_count": 0,
            "average_operation_time": 0.0
        }
        logger.info(f"Created {self.__class__.__name__}")
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the component."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the component."""
        pass
    
    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """Get component status."""
        pass
    
    @abstractmethod
    async def update_config(self, config: Dict[str, Any]) -> None:
        """Update component configuration."""
        pass
    
    async def _pre_initialize(self) -> None:
        """Pre-initialization checks and setup."""
        try:
            logger.info(f"Pre-initializing {self.__class__.__name__}...")
            
            if self.initialized:
                logger.warning(f"{self.__class__.__name__} already initialized")
                return
            
            start_time = datetime.now()
            self.stats["initialization_time"] = start_time.isoformat()
            
            logger.info(f"Successfully pre-initialized {self.__class__.__name__}")
            
        except Exception as e:
            self.last_error = str(e)
            self.stats["error_count"] += 1
            logger.error(f"Error in pre-initialization of {self.__class__.__name__}: {str(e)}")
            raise
    
    async def _post_initialize(self) -> None:
        """Post-initialization cleanup and verification."""
        try:
            logger.info(f"Post-initializing {self.__class__.__name__}...")
            
            self.initialized = True
            self.stats["last_update"] = datetime.now().isoformat()
            
            logger.info(f"Successfully post-initialized {self.__class__.__name__}")
            
        except Exception as e:
            self.last_error = str(e)
            self.stats["error_count"] += 1
            logger.error(f"Error in post-initialization of {self.__class__.__name__}: {str(e)}")
            raise
    
    async def _pre_shutdown(self) -> None:
        """Pre-shutdown checks and cleanup."""
        try:
            logger.info(f"Pre-shutting down {self.__class__.__name__}...")
            
            if not self.initialized:
                logger.warning(f"{self.__class__.__name__} not initialized")
                return
            
            # Log final stats
            logger.info(f"Final stats for {self.__class__.__name__}: {self.stats}")
            
            logger.info(f"Successfully pre-shut down {self.__class__.__name__}")
            
        except Exception as e:
            self.last_error = str(e)
            self.stats["error_count"] += 1
            logger.error(f"Error in pre-shutdown of {self.__class__.__name__}: {str(e)}")
            raise
    
    async def _post_shutdown(self) -> None:
        """Post-shutdown cleanup and verification."""
        try:
            logger.info(f"Post-shutting down {self.__class__.__name__}...")
            
            self.initialized = False
            self.stats["last_update"] = datetime.now().isoformat()
            
            logger.info(f"Successfully post-shut down {self.__class__.__name__}")
            
        except Exception as e:
            self.last_error = str(e)
            self.stats["error_count"] += 1
            logger.error(f"Error in post-shutdown of {self.__class__.__name__}: {str(e)}")
            raise
    
    def _update_stats(self, operation_time: float) -> None:
        """Update component statistics."""
        try:
            self.stats["operation_count"] += 1
            current_avg = self.stats["average_operation_time"]
            self.stats["average_operation_time"] = (
                (current_avg * (self.stats["operation_count"] - 1) + operation_time) /
                self.stats["operation_count"]
            )
            self.stats["last_update"] = datetime.now().isoformat()
        except Exception as e:
            logger.error(f"Error updating stats for {self.__class__.__name__}: {str(e)}")
    
    async def _safe_operation(self, operation_name: str, operation: Any) -> Any:
        """Safely execute an operation with timing and error handling."""
        if not self.initialized:
            raise RuntimeError(f"{self.__class__.__name__} not initialized")
        
        async with ErrorContext(f"{self.__class__.__name__}.{operation_name}"):
            start_time = datetime.now()
            
            try:
                result = await operation
                
                # Update stats
                operation_time = (datetime.now() - start_time).total_seconds()
                self._update_stats(operation_time)
                
                return result
                
            except Exception as e:
                self.last_error = str(e)
                self.stats["error_count"] += 1
                logger.error(f"Error in {operation_name} of {self.__class__.__name__}: {str(e)}")
                raise
    
    async def get_base_status(self) -> Dict[str, Any]:
        """Get base component status."""
        return {
            "initialized": self.initialized,
            "last_error": self.last_error,
            "stats": self.stats,
            "component_type": self.__class__.__name__
        }
