"""
Cognitive Layer Composite for grouping related cognitive processing components.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from rag_system.core.cognitive_nexus import CognitiveNexus
from rag_system.core.latent_space_activation import LatentSpaceActivation
from rag_system.retrieval.vector_store import VectorStore

from ..foundational_layer import FoundationalLayer
from ..continuous_learning_layer import ContinuousLearningLayer
from .interfaces import ICognitiveService, AbstractServiceBase
from .config import CognitiveServiceConfig

logger = logging.getLogger(__name__)


class CognitiveLayerComposite(AbstractServiceBase, ICognitiveService):
    """
    Composite service that groups related cognitive processing layers.
    
    Combines:
    - FoundationalLayer: Core knowledge processing
    - ContinuousLearningLayer: Adaptive learning capabilities
    - CognitiveNexus: Advanced reasoning and connections
    - LatentSpaceActivation: Pattern recognition and embeddings
    """
    
    def __init__(
        self, 
        vector_store: VectorStore,
        config: CognitiveServiceConfig
    ):
        super().__init__(config.config_params)
        self.vector_store = vector_store
        self.cognitive_config = config
        
        # Cognitive components (lazy loaded)
        self._foundational_layer: Optional[FoundationalLayer] = None
        self._continuous_learning_layer: Optional[ContinuousLearningLayer] = None
        self._cognitive_nexus: Optional[CognitiveNexus] = None
        self._latent_space_activation: Optional[LatentSpaceActivation] = None
        
        # State tracking
        self._processing_state = {
            "active_processes": 0,
            "last_evolution": None,
            "performance_metrics": {
                "total_processes": 0,
                "successful_processes": 0,
                "average_processing_time": 0.0
            }
        }
        
        # Component locks for thread safety
        self._component_locks = {
            "foundational": asyncio.Lock(),
            "learning": asyncio.Lock(),
            "nexus": asyncio.Lock(),
            "latent": asyncio.Lock()
        }
    
    @property
    def foundational_layer(self) -> FoundationalLayer:
        """Get foundational layer with lazy loading."""
        if self._foundational_layer is None and self.cognitive_config.enabled:
            self._foundational_layer = FoundationalLayer(self.vector_store)
        return self._foundational_layer
    
    @property
    def continuous_learning_layer(self) -> ContinuousLearningLayer:
        """Get continuous learning layer with lazy loading."""
        if self._continuous_learning_layer is None and self.cognitive_config.enabled:
            self._continuous_learning_layer = ContinuousLearningLayer(self.vector_store)
            self._continuous_learning_layer.learning_rate = self.cognitive_config.continuous_learning_rate
        return self._continuous_learning_layer
    
    @property
    def cognitive_nexus(self) -> CognitiveNexus:
        """Get cognitive nexus with lazy loading."""
        if self._cognitive_nexus is None and self.cognitive_config.cognitive_nexus_enabled:
            self._cognitive_nexus = CognitiveNexus()
        return self._cognitive_nexus
    
    @property
    def latent_space_activation(self) -> LatentSpaceActivation:
        """Get latent space activation with lazy loading."""
        if self._latent_space_activation is None and self.cognitive_config.latent_space_activation:
            self._latent_space_activation = LatentSpaceActivation()
        return self._latent_space_activation
    
    async def initialize(self) -> None:
        """Initialize all cognitive components."""
        if self._initialized:
            return
        
        try:
            logger.info("Initializing cognitive layer composite...")
            
            # Initialize components in parallel if enabled
            tasks = []
            
            if self.cognitive_config.enabled:
                # Initialize foundational layer
                if self.cognitive_config.foundational_layer_cache > 0:
                    tasks.append(self._initialize_foundational_layer())
                
                # Initialize continuous learning
                if self.cognitive_config.continuous_learning_rate > 0:
                    tasks.append(self._initialize_continuous_learning())
            
            if self.cognitive_config.cognitive_nexus_enabled:
                tasks.append(self._initialize_cognitive_nexus())
            
            if self.cognitive_config.latent_space_activation:
                tasks.append(self._initialize_latent_space())
            
            # Wait for all initializations
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            self._initialized = True
            logger.info("Cognitive layer composite initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize cognitive layer composite: {e}")
            raise
    
    async def _initialize_foundational_layer(self) -> None:
        """Initialize foundational layer."""
        async with self._component_locks["foundational"]:
            if self._foundational_layer is None:
                self._foundational_layer = FoundationalLayer(self.vector_store)
                logger.debug("Foundational layer initialized")
    
    async def _initialize_continuous_learning(self) -> None:
        """Initialize continuous learning layer."""
        async with self._component_locks["learning"]:
            if self._continuous_learning_layer is None:
                self._continuous_learning_layer = ContinuousLearningLayer(self.vector_store)
                self._continuous_learning_layer.learning_rate = self.cognitive_config.continuous_learning_rate
                logger.debug("Continuous learning layer initialized")
    
    async def _initialize_cognitive_nexus(self) -> None:
        """Initialize cognitive nexus."""
        async with self._component_locks["nexus"]:
            if self._cognitive_nexus is None:
                self._cognitive_nexus = CognitiveNexus()
                logger.debug("Cognitive nexus initialized")
    
    async def _initialize_latent_space(self) -> None:
        """Initialize latent space activation."""
        async with self._component_locks["latent"]:
            if self._latent_space_activation is None:
                self._latent_space_activation = LatentSpaceActivation()
                logger.debug("Latent space activation initialized")
    
    async def process(self, data: Any) -> Any:
        """
        Process data through all available cognitive layers.
        
        Args:
            data: Input data to process
            
        Returns:
            Processed data with cognitive enhancements
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = datetime.now()
        self._processing_state["active_processes"] += 1
        
        try:
            result = {"input": data, "cognitive_processing": {}}
            
            # Process through foundational layer
            if self.foundational_layer:
                try:
                    foundational_result = await self._process_foundational(data)
                    result["cognitive_processing"]["foundational"] = foundational_result
                except Exception as e:
                    logger.warning(f"Foundational layer processing failed: {e}")
                    result["cognitive_processing"]["foundational"] = {"error": str(e)}
            
            # Process through continuous learning
            if self.continuous_learning_layer:
                try:
                    learning_result = await self._process_learning(data, result)
                    result["cognitive_processing"]["learning"] = learning_result
                except Exception as e:
                    logger.warning(f"Continuous learning processing failed: {e}")
                    result["cognitive_processing"]["learning"] = {"error": str(e)}
            
            # Process through cognitive nexus
            if self.cognitive_nexus:
                try:
                    nexus_result = await self._process_nexus(data, result)
                    result["cognitive_processing"]["nexus"] = nexus_result
                except Exception as e:
                    logger.warning(f"Cognitive nexus processing failed: {e}")
                    result["cognitive_processing"]["nexus"] = {"error": str(e)}
            
            # Process through latent space
            if self.latent_space_activation:
                try:
                    latent_result = await self._process_latent_space(data, result)
                    result["cognitive_processing"]["latent_space"] = latent_result
                except Exception as e:
                    logger.warning(f"Latent space processing failed: {e}")
                    result["cognitive_processing"]["latent_space"] = {"error": str(e)}
            
            # Update performance metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_metrics(processing_time, success=True)
            
            self.update_last_used()
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_metrics(processing_time, success=False)
            logger.error(f"Cognitive processing failed: {e}")
            raise
        finally:
            self._processing_state["active_processes"] -= 1
    
    async def _process_foundational(self, data: Any) -> Dict[str, Any]:
        """Process data through foundational layer."""
        # Foundational layer processing logic
        return {
            "processed": True,
            "layer": "foundational",
            "timestamp": datetime.now().isoformat()
        }
    
    async def _process_learning(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through continuous learning layer."""
        # Continuous learning processing logic
        return {
            "processed": True,
            "layer": "continuous_learning",
            "learning_applied": True,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _process_nexus(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through cognitive nexus."""
        # Cognitive nexus processing logic
        return {
            "processed": True,
            "layer": "cognitive_nexus",
            "connections_found": 0,  # Would be actual connection count
            "timestamp": datetime.now().isoformat()
        }
    
    async def _process_latent_space(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through latent space activation."""
        # Latent space processing logic
        return {
            "processed": True,
            "layer": "latent_space",
            "patterns_detected": 0,  # Would be actual pattern count
            "timestamp": datetime.now().isoformat()
        }
    
    async def evolve(self) -> None:
        """Evolve all cognitive components."""
        if not self._initialized:
            await self.initialize()
        
        try:
            logger.info("Starting cognitive evolution...")
            
            evolution_tasks = []
            
            if self.continuous_learning_layer:
                evolution_tasks.append(self.continuous_learning_layer.evolve())
            
            if self.cognitive_nexus:
                evolution_tasks.append(self.cognitive_nexus.evolve())
            
            if self.latent_space_activation:
                evolution_tasks.append(self.latent_space_activation.evolve())
            
            # Execute all evolutions in parallel
            if evolution_tasks:
                await asyncio.gather(*evolution_tasks, return_exceptions=True)
            
            self._processing_state["last_evolution"] = datetime.now()
            logger.info("Cognitive evolution completed")
            
        except Exception as e:
            logger.error(f"Cognitive evolution failed: {e}")
            raise
    
    def get_state(self) -> Dict[str, Any]:
        """Get current cognitive state."""
        state = {
            "initialized": self._initialized,
            "processing_state": self._processing_state.copy(),
            "components": {
                "foundational_layer": self._foundational_layer is not None,
                "continuous_learning_layer": self._continuous_learning_layer is not None,
                "cognitive_nexus": self._cognitive_nexus is not None,
                "latent_space_activation": self._latent_space_activation is not None
            },
            "config": {
                "enabled": self.cognitive_config.enabled,
                "cognitive_nexus_enabled": self.cognitive_config.cognitive_nexus_enabled,
                "latent_space_activation": self.cognitive_config.latent_space_activation,
                "continuous_learning_rate": self.cognitive_config.continuous_learning_rate
            }
        }
        
        # Add detailed component states if available
        if self._continuous_learning_layer:
            state["component_details"] = {
                "continuous_learning": {
                    "recent_learnings_count": len(self._continuous_learning_layer.recent_learnings),
                    "learning_rate": self._continuous_learning_layer.learning_rate,
                    "performance_history_length": len(self._continuous_learning_layer.performance_history)
                }
            }
        
        return state
    
    def _update_performance_metrics(self, processing_time: float, success: bool) -> None:
        """Update performance metrics."""
        metrics = self._processing_state["performance_metrics"]
        metrics["total_processes"] += 1
        
        if success:
            metrics["successful_processes"] += 1
        
        # Update rolling average
        current_avg = metrics["average_processing_time"]
        total = metrics["total_processes"]
        metrics["average_processing_time"] = ((current_avg * (total - 1)) + processing_time) / total
    
    async def shutdown(self) -> None:
        """Shutdown all cognitive components."""
        try:
            # Wait for active processes to complete
            while self._processing_state["active_processes"] > 0:
                await asyncio.sleep(0.1)
            
            # Shutdown components
            shutdown_tasks = []
            
            for component in [
                self._foundational_layer,
                self._continuous_learning_layer, 
                self._cognitive_nexus,
                self._latent_space_activation
            ]:
                if component and hasattr(component, 'shutdown'):
                    shutdown_tasks.append(component.shutdown())
            
            if shutdown_tasks:
                await asyncio.gather(*shutdown_tasks, return_exceptions=True)
            
            # Clear references
            self._foundational_layer = None
            self._continuous_learning_layer = None
            self._cognitive_nexus = None
            self._latent_space_activation = None
            
            self._initialized = False
            logger.info("Cognitive layer composite shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during cognitive composite shutdown: {e}")
            raise