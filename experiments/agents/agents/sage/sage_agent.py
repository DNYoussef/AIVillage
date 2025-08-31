from datetime import datetime
import logging
import time
from typing import Any, Optional
import uuid

from bs4 import BeautifulSoup
from rag_system.core.config import UnifiedConfig
from rag_system.core.exploration_mode import ExplorationMode
from rag_system.retrieval.vector_store import VectorStore
from rag_system.tracking.unified_knowledge_tracker import UnifiedKnowledgeTracker
import requests

from agents.unified_base_agent import SelfEvolvingSystem, UnifiedBaseAgent
from agents.utils.evidence_helpers import wrap_in_pack
from agents.utils.task import Task as LangroidTask
from core.error_handling import Message, MessageType, StandardCommunicationProtocol
from core.evidence import EvidencePack

from .services import (
    SageAgentServiceLocator, 
    SageAgentConfig,
    CognitiveLayerComposite, 
    ProcessingChainFactory,
    CognitiveServiceConfig,
    ProcessingServiceConfig,
    LearningServiceConfig,
    CollaborationServiceConfig,
    ResearchServiceConfig
)
from .services.service_factories import create_service_factory_registry

logger = logging.getLogger(__name__)


class SageAgent(UnifiedBaseAgent):
    def __init__(
        self,
        config: UnifiedConfig,
        communication_protocol: StandardCommunicationProtocol,
        vector_store: VectorStore,
        knowledge_tracker: Optional[UnifiedKnowledgeTracker] = None,
        sage_config: Optional[SageAgentConfig] = None,
    ) -> None:
        super().__init__(config, communication_protocol, knowledge_tracker)
        
        # Create SageAgent configuration
        self.sage_config = sage_config or SageAgentConfig.from_unified_config(config)
        
        # Initialize service locator
        self.services = SageAgentServiceLocator(self.sage_config)
        
        # Core dependencies (minimal direct instantiation)
        self.vector_store = vector_store
        self.research_capabilities = self.sage_config.research_capabilities
        
        # Initialize services through service locator
        self._setup_services(config, knowledge_tracker)
        
        # Legacy systems (temporarily kept for compatibility)
        self.self_evolving_system = SelfEvolvingSystem([self])
        
        # Performance metrics
        self.performance_metrics = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "average_execution_time": 0,
        }
        
        logger.info(f"SageAgent initialized with {len(self.services.get_registered_services())} services")

    def _setup_services(
        self, 
        config: UnifiedConfig, 
        knowledge_tracker: Optional[UnifiedKnowledgeTracker]
    ) -> None:
        """Setup service locator with all required services."""
        from .services.service_factories import (
            create_cognitive_composite,
            create_processing_chain,
            create_rag_system,
            create_cognitive_nexus,
            create_latent_space_activation,
            create_error_controller,
            create_confidence_estimator,
            create_collaboration_manager,
            create_research_capabilities,
            create_user_intent_interpreter
        )
        
        # Register service factories with configurations
        self.services.register_service_factory(
            "rag_system",
            lambda: create_rag_system(config, knowledge_tracker)
        )
        
        self.services.register_service_factory(
            "cognitive_nexus",
            create_cognitive_nexus
        )
        
        self.services.register_service_factory(
            "latent_space_activation", 
            create_latent_space_activation
        )
        
        self.services.register_service_factory(
            "error_controller",
            create_error_controller
        )
        
        self.services.register_service_factory(
            "confidence_estimator",
            create_confidence_estimator
        )
        
        self.services.register_service_factory(
            "user_intent_interpreter",
            create_user_intent_interpreter
        )
        
        # Register composite services
        self.services.register_service_factory(
            "cognitive_composite",
            lambda: create_cognitive_composite(
                self.vector_store,
                self.sage_config.cognitive
            )
        )
        
        # Processing chain needs to be created after basic services
        async def create_processing_chain_with_deps():
            rag_system = await self.services.get_service("rag_system")
            latent_space = await self.services.get_service("latent_space_activation")
            cognitive_nexus = await self.services.get_service("cognitive_nexus")
            confidence_estimator = await self.services.get_service("confidence_estimator")
            
            return await create_processing_chain(
                rag_system=rag_system,
                latent_space_activation=latent_space,
                cognitive_nexus=cognitive_nexus,
                confidence_estimator=confidence_estimator,
                sage_agent=self,
                config=self.sage_config.processing
            )
        
        self.services.register_service_factory(
            "processing_chain",
            create_processing_chain_with_deps
        )
        
        # Collaboration manager
        self.services.register_service_factory(
            "collaboration_manager",
            lambda: create_collaboration_manager(self, self.sage_config.collaboration)
        )
        
        # Research capabilities
        self.services.register_service_factory(
            "research_capabilities",
            lambda: create_research_capabilities(self, self.sage_config.research)
        )
        
        # Setup exploration mode after RAG system is available
        async def create_exploration_mode():
            rag_system = await self.services.get_service("rag_system")
            return ExplorationMode(rag_system)
        
        self.services.register_service_factory(
            "exploration_mode",
            create_exploration_mode
        )
    
    # Lazy loading properties for backward compatibility
    @property
    async def rag_system(self):
        """Get RAG system with lazy loading."""
        return await self.services.get_service("rag_system")
    
    @property
    async def cognitive_composite(self):
        """Get cognitive layer composite."""
        return await self.services.get_service("cognitive_composite")
    
    @property
    async def processing_chain(self):
        """Get processing chain factory."""
        return await self.services.get_service("processing_chain")
    
    @property
    async def error_controller(self):
        """Get error controller with lazy loading."""
        return await self.services.get_service("error_controller")
    
    @property
    async def collaboration_manager(self):
        """Get collaboration manager with lazy loading."""
        return await self.services.get_service("collaboration_manager")
    
    @property
    async def research_capabilities_manager(self):
        """Get research capabilities manager with lazy loading."""
        return await self.services.get_service("research_capabilities")
    
    @property
    async def exploration_mode(self):
        """Get exploration mode with lazy loading."""
        return await self.services.get_service("exploration_mode")

    async def execute_task(self, task: LangroidTask):
        self.performance_metrics["total_tasks"] += 1
        start_time = time.time()
        try:
            if getattr(task, "is_user_query", False):
                result = await self.process_user_query(task.content)
            else:
                # Use processing chain for task execution
                processing_chain = await self.processing_chain
                result = await processing_chain.execute_task(
                    {
                        "type": getattr(task, "type", "general"),
                        "content": task.content,
                        "priority": getattr(task, "priority", 1),
                        "id": getattr(task, "task_id", ""),
                    }
                )
            self.performance_metrics["successful_tasks"] += 1
            return result
        except Exception as e:
            self.performance_metrics["failed_tasks"] += 1
            logger.exception(f"Error executing task: {e!s}")
            error_controller = await self.error_controller
            return await error_controller.handle_error(e, task)
        finally:
            execution_time = time.time() - start_time
            self.performance_metrics["average_execution_time"] = (
                self.performance_metrics["average_execution_time"] * (self.performance_metrics["total_tasks"] - 1)
                + execution_time
            ) / self.performance_metrics["total_tasks"]

    async def process_user_query(self, query: str) -> dict[str, Any]:
        # Use processing chain for unified query processing
        processing_chain = await self.processing_chain
        result = await processing_chain.process_query(query)
        
        # Wrap in evidence pack for compatibility
        if "rag_result" in result:
            wrap_in_pack(result["rag_result"], query)
        
        return result

    async def pre_process_query(self, query: str) -> str:
        # Implement query pre-processing logic here
        return query

    async def post_process_result(
        self,
        rag_result: dict[str, Any],
        original_query: str,
        response: str,
        intent: dict[str, Any],
    ) -> dict[str, Any]:
        confidence_estimator = await self.services.get_service("confidence_estimator")
        processed_result = {
            "original_query": original_query,
            "interpreted_intent": intent,
            "rag_result": rag_result,
            "response": response,
            "confidence": await confidence_estimator.estimate(original_query, rag_result),
        }
        return processed_result

    async def perform_web_scrape(self, url: str) -> dict[str, Any]:
        """Scrape a web page and store the content in the RAG system."""
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator=" ", strip=True)

        doc_id = str(uuid.uuid4())
        embedding = await self.get_embedding(text)
        document = {
            "id": doc_id,
            "content": text,
            "embedding": embedding,
            "timestamp": datetime.now(),
        }
        
        # Get RAG system through service locator
        rag_system = await self.rag_system
        rag_system.hybrid_retriever.vector_store.add_documents([document])
        await rag_system.update_bayes_net(doc_id, text)

        return {"url": url, "doc_id": doc_id, "snippet": text[:200]}

    async def perform_web_search(self, query: str) -> dict[str, Any]:
        """Perform a simple web search and return snippets."""
        resp = requests.get("https://duckduckgo.com/html/", params={"q": query}, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for a in soup.select("a.result__a")[:3]:
            results.append(a.get_text())
        return {"query": query, "results": results}

    async def analyze_data(self, data: Any) -> dict[str, Any]:
        """Placeholder for data analysis."""
        return {"summary": str(data)[:100]}

    async def synthesize_information(self, info: str) -> str:
        """Placeholder for information synthesis."""
        return info

    async def handle_message(self, message: Message) -> None:
        if message.type == MessageType.TASK:
            task_content = message.content.get("content")
            task_type = message.content.get("task_type", "general")
            is_user_query = message.content.get("is_user_query", False)
            task = LangroidTask(self, task_content, "", 1)
            task.type = task_type
            task.is_user_query = is_user_query
            result = await self.execute_task(task)
            response = Message(
                type=MessageType.RESPONSE,
                sender=self.name,
                receiver=message.sender,
                content=result,
                parent_id=message.id,
            )
            await self.communication_protocol.send_message(response)
            if isinstance(result, dict):
                rag = result.get("rag_result", {})
                pack = rag.get("evidence_pack")
                if isinstance(pack, EvidencePack):
                    await self.communication_protocol.send_message(
                        Message(
                            type=MessageType.EVIDENCE,
                            sender=self.name,
                            receiver=message.sender,
                            content=pack.dict(),
                            parent_id=message.id,
                        )
                    )
        elif message.type == MessageType.COLLABORATION_REQUEST:
            collaboration_manager = await self.collaboration_manager
            await collaboration_manager.handle_collaboration_request(message)
        else:
            await super().handle_message(message)

    async def evolve(self) -> None:
        # Evolve self-evolving system (legacy)
        await self.self_evolving_system.evolve()
        
        # Evolve cognitive composite
        if self.services.is_service_instantiated("cognitive_composite"):
            cognitive_composite = await self.cognitive_composite
            await cognitive_composite.evolve()
        
        # Evolve error controller
        if self.services.is_service_instantiated("error_controller"):
            error_controller = await self.error_controller
            await error_controller.evolve(self.performance_metrics)
        
        # Evolve research capabilities
        if self.services.is_service_instantiated("research_capabilities"):
            research_mgr = await self.research_capabilities_manager
            await research_mgr.evolve_research_capabilities()
        
        logger.info("SageAgent evolved")

    async def introspect(self):
        base_info = await super().introspect()
        
        # Get service statistics
        service_stats = self.services.get_service_stats()
        performance_metrics = self.services.get_performance_metrics()
        
        # Get cognitive state if available
        cognitive_state = {}
        if self.services.is_service_instantiated("cognitive_composite"):
            cognitive_composite = await self.cognitive_composite
            cognitive_state = cognitive_composite.get_state()
        
        # Get collaboration info if available
        collaborating_agents = []
        if self.services.is_service_instantiated("collaboration_manager"):
            collaboration_mgr = await self.collaboration_manager
            collaborating_agents = list(collaboration_mgr.collaborating_agents.keys())
        
        return {
            **base_info,
            "research_capabilities": self.research_capabilities,
            "service_architecture": {
                "total_services": len(self.services.get_registered_services()),
                "instantiated_services": len([s for s in self.services.get_registered_services() 
                                             if self.services.is_service_instantiated(s)]),
                "registered_services": self.services.get_registered_services(),
                "performance_metrics": performance_metrics
            },
            "advanced_techniques": {
                "reasoning": [
                    "Chain-of-Thought",
                    "Self-Consistency",
                    "Tree-of-Thoughts",
                ],
                "NLP_models": [
                    "BERTEmbeddingModel",
                    "NamedEntityRecognizer",
                    "RelationExtractor",
                ],
            },
            "layers": {
                "SelfEvolvingSystem": "Active",
                "CognitiveComposite": "Active" if cognitive_state.get("initialized") else "Lazy",
                "ProcessingChain": "Active" if self.services.is_service_instantiated("processing_chain") else "Lazy",
                "ServiceLocator": "Active",
            },
            "query_processing": "Service-based pipeline with lazy loading",
            "exploration_capabilities": "Enhanced with multi-strategy approach and result synthesis",
            "collaboration_capabilities": {
                "knowledge_sharing": "Active",
                "task_delegation": "Active", 
                "joint_reasoning": "Active",
            },
            "collaborating_agents": collaborating_agents,
            "error_handling": "Adaptive error control with confidence estimation",
            "performance_metrics": self.performance_metrics,
            "cognitive_state": cognitive_state,
            "self_evolving_system": {
                "current_architecture": self.self_evolving_system.current_architecture,
                "evolution_rate": self.self_evolving_system.evolution_rate,
                "mutation_rate": self.self_evolving_system.mutation_rate,
                "learning_rate": self.self_evolving_system.learning_rate,
            },
            "service_statistics": service_stats,
        }
