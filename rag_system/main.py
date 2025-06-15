import asyncio
from datetime import datetime
from typing import Dict, Any

from rag_system.core.unified_config import unified_config
from rag_system.core.pipeline import EnhancedRAGPipeline
from rag_system.tracking.unified_knowledge_tracker import UnifiedKnowledgeTracker
from rag_system.utils.embedding import BERTEmbeddingModel
from rag_system.utils.advanced_analytics import AdvancedAnalytics
from rag_system.utils.standardized_formats import create_standardized_prompt, create_standardized_output, OutputFormat
from rag_system.retrieval.hybrid_retriever import HybridRetriever
from rag_system.processing.reasoning_engine import UncertaintyAwareReasoningEngine
from rag_system.processing.advanced_nlp import AdvancedNLP
from rag_system.core.cognitive_nexus import CognitiveNexus
from rag_system.core.exploration_mode import ExplorationMode
from rag_system.evaluation.comprehensive_evaluation import ComprehensiveEvaluationFramework
from agents.king.king_agent import KingAgent
from agents.unified_base_agent import UnifiedAgentConfig
from communications.protocol import StandardCommunicationProtocol
from rag_system.retrieval.vector_store import VectorStore
from rag_system.retrieval.graph_store import GraphStore
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from rag_system.utils.error_handling import log_and_handle_errors, setup_logging, RAGSystemError

logger = setup_logging()

@log_and_handle_errors
async def initialize_components() -> Dict[str, Any]:
    vector_store = VectorStore()
    graph_store = GraphStore()
    llm_config = OpenAIGPTConfig(chat_model="gpt-4")
    advanced_analytics = AdvancedAnalytics()
    advanced_nlp = AdvancedNLP()
    communication_protocol = StandardCommunicationProtocol()

    king_agent_config = UnifiedAgentConfig(
        name="KingAgent",
        description="The main coordinating agent for the RAG system",
        capabilities=["task_coordination", "decision_making", "problem_analysis"],
        rag_config=unified_config,
        vector_store=vector_store,
        model="gpt-4",
        instructions="You are the main coordinating agent for the RAG system. Your role is to manage tasks, make decisions, and analyze problems."
    )

    components = {
        "embedding_model": BERTEmbeddingModel(),
        "vector_store": vector_store,
        "graph_store": graph_store,
        "hybrid_retriever": HybridRetriever(unified_config),
        "reasoning_engine": UncertaintyAwareReasoningEngine(unified_config),
        "cognitive_nexus": CognitiveNexus(),
        "pipeline": EnhancedRAGPipeline(unified_config),
        "communication_protocol": communication_protocol,
        "king_agent": KingAgent(king_agent_config, communication_protocol, vector_store),
        "knowledge_tracker": UnifiedKnowledgeTracker(vector_store, graph_store),
        "exploration_mode": ExplorationMode(graph_store, llm_config, advanced_nlp),
        "advanced_analytics": advanced_analytics,
        "evaluation_framework": ComprehensiveEvaluationFramework(advanced_analytics),
        "advanced_nlp": advanced_nlp
    }

    # Initialize evaluation metrics
    components["evaluation_framework"].add_metric("query_processing_time", "Time taken to process a user query")
    components["evaluation_framework"].add_metric("task_processing_time", "Time taken to process an agent task")
    components["evaluation_framework"].add_metric("retrieval_count", "Number of items retrieved for a query")
    components["evaluation_framework"].add_metric("exploration_time", "Time taken for knowledge graph exploration")
    components["evaluation_framework"].add_metric("explored_nodes", "Number of nodes explored in the knowledge graph")
    components["evaluation_framework"].add_metric("new_relations", "Number of new relations discovered")
    components["evaluation_framework"].add_metric("response_relevance", "Relevance score of the response to the query")
    components["evaluation_framework"].add_metric("response_coherence", "Coherence score of the generated response")
    components["evaluation_framework"].add_metric("knowledge_graph_density", "Density of the knowledge graph")
    components["evaluation_framework"].add_metric("system_latency", "Overall system latency")

    return components

@log_and_handle_errors
async def process_user_query(components: Dict[str, Any], query: str) -> Dict[str, Any]:
    king_agent = components["king_agent"]
    task = {
        "type": "query",
        "content": query,
        "timestamp": datetime.now().isoformat()
    }
    return await king_agent.execute_task(task)

@log_and_handle_errors
async def run_creative_exploration(components: Dict[str, Any], start_node: str, end_node: str):
    king_agent = components["king_agent"]
    task = {
        "type": "creative_exploration",
        "content": {
            "start_node": start_node,
            "end_node": end_node
        },
        "timestamp": datetime.now().isoformat()
    }
    return await king_agent.execute_task(task)

@log_and_handle_errors
async def main():
    # Load configuration
    config_path = "config/rag_config.json"
    unified_config.load_config(config_path)

    components = await initialize_components()

    user_query = "What are the key features of the RAG system?"
    query_result = await process_user_query(components, user_query)
    print(f"Query result: {query_result}")

    # Run a creative exploration
    start_node = "artificial_intelligence"
    end_node = "human_creativity"
    creative_exploration_result = await run_creative_exploration(components, start_node, end_node)
    print(f"Creative Exploration result: {creative_exploration_result}")

    # Generate and log evaluation report
    evaluation_results = components["evaluation_framework"].evaluate_system_performance()
    components["evaluation_framework"].log_evaluation_results(evaluation_results)

    print(f"Evaluation complete. Check logs for detailed results.")

    # Evolve the KingAgent
    await components["king_agent"].evolve()

if __name__ == "__main__":
    asyncio.run(main())
