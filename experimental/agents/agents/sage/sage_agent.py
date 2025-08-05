from datetime import datetime
import logging
import time
from typing import Any
import uuid

from bs4 import BeautifulSoup
import requests

from agents.unified_base_agent import SelfEvolvingSystem, UnifiedBaseAgent
from agents.utils.evidence_helpers import wrap_in_pack
from agents.utils.task import Task as LangroidTask
from core.error_handling import Message, MessageType, StandardCommunicationProtocol
from core.evidence import EvidencePack
from rag_system.core.cognitive_nexus import CognitiveNexus
from rag_system.core.config import UnifiedConfig
from rag_system.core.exploration_mode import ExplorationMode
from rag_system.core.latent_space_activation import LatentSpaceActivation
from rag_system.core.pipeline import EnhancedRAGPipeline
from rag_system.error_handling.adaptive_controller import AdaptiveErrorController
from rag_system.processing.confidence_estimator import ConfidenceEstimator
from rag_system.retrieval.vector_store import VectorStore
from rag_system.tracking.unified_knowledge_tracker import UnifiedKnowledgeTracker

from .collaboration import CollaborationManager
from .continuous_learning_layer import ContinuousLearningLayer
from .foundational_layer import FoundationalLayer
from .query_processing import QueryProcessor
from .research_capabilities import ResearchCapabilities
from .response_generator import ResponseGenerator
from .task_execution import TaskExecutor
from .user_intent_interpreter import UserIntentInterpreter

logger = logging.getLogger(__name__)


class SageAgent(UnifiedBaseAgent):
    def __init__(
        self,
        config: UnifiedConfig,
        communication_protocol: StandardCommunicationProtocol,
        vector_store: VectorStore,
        knowledge_tracker: UnifiedKnowledgeTracker | None = None,
    ):
        super().__init__(config, communication_protocol, knowledge_tracker)
        self.research_capabilities = config.get("research_capabilities", [])
        self.rag_system = EnhancedRAGPipeline(config, knowledge_tracker)
        self.vector_store = vector_store
        self.exploration_mode = ExplorationMode(self.rag_system)
        self.self_evolving_system = SelfEvolvingSystem([self])
        self.foundational_layer = FoundationalLayer(vector_store)
        self.continuous_learning_layer = ContinuousLearningLayer(vector_store)
        self.cognitive_nexus = CognitiveNexus()
        self.latent_space_activation = LatentSpaceActivation()
        self.error_controller = AdaptiveErrorController()
        self.confidence_estimator = ConfidenceEstimator()
        self.query_processor = QueryProcessor(self.rag_system, self.latent_space_activation, self.cognitive_nexus)
        self.task_executor = TaskExecutor(self)
        self.collaboration_manager = CollaborationManager(self)
        self.research_capabilities_manager = ResearchCapabilities(self)
        self.user_intent_interpreter = UserIntentInterpreter()
        self.response_generator = ResponseGenerator()
        self.performance_metrics = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "average_execution_time": 0,
        }

    async def execute_task(self, task: LangroidTask):
        self.performance_metrics["total_tasks"] += 1
        start_time = time.time()
        try:
            if getattr(task, "is_user_query", False):
                result = await self.process_user_query(task.content)
            else:
                result = await self.task_executor.execute_task(
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
            logger.error(f"Error executing task: {e!s}")
            return await self.error_controller.handle_error(e, task)
        finally:
            execution_time = time.time() - start_time
            self.performance_metrics["average_execution_time"] = (
                self.performance_metrics["average_execution_time"] * (self.performance_metrics["total_tasks"] - 1)
                + execution_time
            ) / self.performance_metrics["total_tasks"]

    async def process_user_query(self, query: str) -> dict[str, Any]:
        intent = await self.user_intent_interpreter.interpret_intent(query)
        processed_query = await self.pre_process_query(query)
        rag_result = await self.rag_system.process_query(processed_query)
        wrap_in_pack(rag_result, processed_query)
        response = await self.response_generator.generate_response(query, rag_result, intent)
        final_result = await self.post_process_result(rag_result, query, response, intent)
        return final_result

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
        processed_result = {
            "original_query": original_query,
            "interpreted_intent": intent,
            "rag_result": rag_result,
            "response": response,
            "confidence": await self.confidence_estimator.estimate(original_query, rag_result),
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
        self.rag_system.hybrid_retriever.vector_store.add_documents([document])
        await self.rag_system.update_bayes_net(doc_id, text)

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

    async def handle_message(self, message: Message):
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
            await self.collaboration_manager.handle_collaboration_request(message)
        else:
            await super().handle_message(message)

    async def evolve(self):
        await self.self_evolving_system.evolve()
        await self.continuous_learning_layer.evolve()
        await self.cognitive_nexus.evolve()
        await self.latent_space_activation.evolve()
        await self.error_controller.evolve(self.performance_metrics)
        await self.research_capabilities_manager.evolve_research_capabilities()
        logger.info("SageAgent evolved")

    async def introspect(self):
        base_info = await super().introspect()
        return {
            **base_info,
            "research_capabilities": self.research_capabilities,
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
                "FoundationalLayer": "Active",
                "ContinuousLearningLayer": "Active",
                "CognitiveNexus": "Active",
                "LatentSpaceActivation": "Active",
            },
            "query_processing": "Streamlined pipeline integrating all advanced components",
            "exploration_capabilities": "Enhanced with multi-strategy approach and result synthesis",
            "collaboration_capabilities": {
                "knowledge_sharing": "Active",
                "task_delegation": "Active",
                "joint_reasoning": "Active",
            },
            "collaborating_agents": list(self.collaboration_manager.collaborating_agents.keys()),
            "error_handling": "Adaptive error control with confidence estimation",
            "performance_metrics": self.performance_metrics,
            "continuous_learning": {
                "recent_learnings_count": len(self.continuous_learning_layer.recent_learnings),
                "learning_rate": self.continuous_learning_layer.learning_rate,
                "performance_history_length": len(self.continuous_learning_layer.performance_history),
            },
            "self_evolving_system": {
                "current_architecture": self.self_evolving_system.current_architecture,
                "evolution_rate": self.self_evolving_system.evolution_rate,
                "mutation_rate": self.self_evolving_system.mutation_rate,
                "learning_rate": self.self_evolving_system.learning_rate,
            },
        }
