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
from agents.sage.sage_agent import SageAgent
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
    components = {
        "embedding_model": BERTEmbeddingModel(),
        "vector_store": vector_store,
        "graph_store": graph_store,
        "hybrid_retriever": HybridRetriever(unified_config),
        "reasoning_engine": UncertaintyAwareReasoningEngine(unified_config),
        "cognitive_nexus": CognitiveNexus(),
        "pipeline": EnhancedRAGPipeline(),
        "communication_protocol": StandardCommunicationProtocol(),
        "sage_agent": SageAgent(unified_config, StandardCommunicationProtocol(), vector_store),
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
    start_time = datetime.now()

    # Create standardized prompt
    prompt = create_standardized_prompt(
        task=query,
        context="User query to the RAG system",
        output_format=OutputFormat.JSON,
        constraints=["Provide a concise answer", "Use information from the knowledge base"],
        additional_instructions="Ensure the response is relevant and accurate",
        metadata={"query_type": "user_input", "timestamp": start_time.isoformat()}
    )

    # Use advanced NLP for query analysis
    advanced_nlp = components["advanced_nlp"]
    query_embedding = advanced_nlp.get_embeddings([query])[0]
    query_sentiment = advanced_nlp.analyze_sentiment(query)
    query_keywords = advanced_nlp.extract_keywords(query)

    retrieval_results = await components["hybrid_retriever"].retrieve(prompt.task, k=10)

    # Use semantic search to refine retrieval results
    semantic_results = advanced_nlp.semantic_search(query, [r['content'] for r in retrieval_results])
    refined_results = [r for r, s in zip(retrieval_results, semantic_results) if s['similarity'] > 0.5]

    # Process the retrieval results using the reasoning engine
    reasoning_result = await components["reasoning_engine"].reason(prompt.to_string(), refined_results)

    # Integrate the results using the cognitive nexus
    integrated_result = await components["cognitive_nexus"].integrate(prompt.task, reasoning_result, {})

    # Generate a summary of the integrated result
    result_summary = advanced_nlp.generate_summary(str(integrated_result))

    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()

    logger.info(f"User query result: {integrated_result}")

    components["knowledge_tracker"].record_change(
        entity="RAG system",
        relation="processed_query",
        old_value="",
        new_value=query,
        timestamp=datetime.now(),
        source="user_query"
    )

    components["knowledge_tracker"].track_changes(integrated_result, datetime.now())

    # Record metrics
    evaluation_framework = components["evaluation_framework"]
    evaluation_framework.record_metric("query_processing_time", processing_time)
    evaluation_framework.record_metric("retrieval_count", len(retrieval_results))
    evaluation_framework.record_metric("response_relevance", advanced_nlp.calculate_relevance(query, integrated_result))
    evaluation_framework.record_metric("response_coherence", advanced_nlp.calculate_coherence(integrated_result))
    evaluation_framework.record_metric("system_latency", processing_time)

    # Create standardized output
    output = create_standardized_output(
        task=query,
        response=integrated_result,
        confidence=reasoning_result.get('confidence', 0.0),
        sources=[r['source'] for r in retrieval_results if 'source' in r],
        metadata={
            "processing_time": processing_time,
            "query_embedding": query_embedding.tolist(),
            "query_sentiment": query_sentiment,
            "query_keywords": query_keywords,
            "result_summary": result_summary
        },
        reasoning=reasoning_result.get('reasoning', ''),
        uncertainty=1 - reasoning_result.get('confidence', 0.0),
        alternative_responses=reasoning_result.get('alternative_responses', [])
    )

    return output.to_dict()

@log_and_handle_errors
async def run_creative_exploration(components: Dict[str, Any], start_node: str, end_node: str):
    logger.info(f"Starting creative exploration between '{start_node}' and '{end_node}'")

    start_time = datetime.now()

    exploration_results = await components["exploration_mode"].creative_exploration(start_node, end_node)

    end_time = datetime.now()
    exploration_time = (end_time - start_time).total_seconds()

    logger.info(f"Creative exploration completed in {exploration_time:.2f} seconds")
    logger.info(f"Found {len(exploration_results['causal_paths'])} causal paths")
    logger.info(f"Generated {len(exploration_results['new_ideas'])} new ideas")

    # Record metrics
    evaluation_framework = components["evaluation_framework"]
    evaluation_framework.record_metric("exploration_time", exploration_time)
    evaluation_framework.record_metric("causal_paths_found", len(exploration_results['causal_paths']))
    evaluation_framework.record_metric("new_ideas_generated", len(exploration_results['new_ideas']))
    evaluation_framework.record_metric("knowledge_graph_density", components["graph_store"].calculate_graph_density())

    return exploration_results

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

if __name__ == "__main__":
    asyncio.run(main())
