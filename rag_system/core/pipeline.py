# rag_system/core/pipeline.py

from typing import List, Dict, Any, Tuple
from .config import RAGConfig
from ..retrieval.hybrid_retriever import HybridRetriever
from ..processing.knowledge_constructor import DefaultKnowledgeConstructor
from ..processing.reasoning_engine import DefaultReasoningEngine
from ..utils.embedding import DefaultEmbeddingModel
from ..active_rag.active_hybrid_retriever import ActiveHybridRetriever
from ..active_rag.latent_space_activator import LatentSpaceActivator
from ..plan_rag.planning_aware_retriever import PlanningAwareRetriever
from ..plan_rag.iterative_query_manager import IterativeQueryManager
from ..processing.cognitive_nexus import CognitiveNexus
from ..core.agent_interface import AgentInterface

class RAGPipeline:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.embedding_model = DefaultEmbeddingModel(config)
        self.hybrid_retriever = HybridRetriever(config)
        self.knowledge_constructor = DefaultKnowledgeConstructor(config)
        self.reasoning_engine = DefaultReasoningEngine(config)
        self.active_hybrid_retriever = ActiveHybridRetriever(config, self.hybrid_retriever)
        self.latent_space_activator = LatentSpaceActivator(config)
        self.planning_aware_retriever = PlanningAwareRetriever(config, self.hybrid_retriever)
        self.iterative_query_manager = IterativeQueryManager(config, self.planning_aware_retriever)
        self.cognitive_nexus = CognitiveNexus()

    async def process_query(self, query: str, agent: AgentInterface) -> str:
        query_embedding = await self.embedding_model.get_embedding(query)
        
        # Step 1: Latent Space Activation
        latent_activations = self.latent_space_activator.activate(query_embedding)

        # Step 2: Initial Plan Generation
        initial_plan = await self.planning_aware_retriever.generate_plan(query)

        # Step 3: Iterative Retrieval and Refinement
        final_results, final_plan, retrieval_history = await self._iterative_retrieval_refinement(query, initial_plan, agent)

        # Step 4: Active Hybrid Retrieval
        active_results = await self.active_hybrid_retriever.retrieve(query, final_results)

        # Step 5: Combine results
        combined_results = self._combine_results(active_results, latent_activations)

        # Step 6: Knowledge Construction
        constructed_knowledge = await self.knowledge_constructor.construct(query, combined_results)

        # Step 7: Cognitive Integration
        final_answer = await self.cognitive_nexus.integrate(
            query, 
            constructed_knowledge, 
            final_plan, 
            retrieval_history, 
            agent
        )

        return final_answer

    async def _iterative_retrieval_refinement(self, query: str, initial_plan: Dict[str, Any], agent: AgentInterface, max_iterations: int = 3) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
        current_plan = initial_plan
        current_results = []
        retrieval_history = []
        for _ in range(max_iterations):
            new_results = await self._retrieve(query, current_plan, agent)
            all_results = self._combine_results(current_results, new_results)
            reranked_results = await self._rerank_results(all_results, query, current_plan, agent)
            retrieval_history.append({
                "plan": current_plan,
                "top_results": reranked_results[:5]  # Store top 5 results for history
            })
            if self._is_satisfactory(reranked_results):
                return reranked_results, current_plan, retrieval_history
            current_plan = await self.planning_aware_retriever.refine_plan(query, current_plan, reranked_results)
            current_results = reranked_results
        return current_results, current_plan, retrieval_history

    async def _retrieve(self, query: str, plan: Dict[str, Any], agent: AgentInterface) -> List[Dict[str, Any]]:
        query_embedding = await agent.get_embedding(query)
        vector_results = await self.hybrid_retriever.vector_store.search(query_embedding, self.config.VECTOR_TOP_K)
        graph_results = await self.hybrid_retriever.graph_store.search(query, self.config.GRAPH_TOP_K)

        combined_results = self._combine_results(vector_results, graph_results)
        
        # Apply the plan to filter or modify results
        if "filter_keywords" in plan:
            combined_results = [r for r in combined_results if any(kw in r.get('content', '') for kw in plan["filter_keywords"])]
        
        if "max_date" in plan:
            combined_results = [r for r in combined_results if r.get('date', '') <= plan["max_date"]]
        
        return combined_results[:self.config.MAX_RESULTS]

    def _combine_results(self, results1: List[Dict[str, Any]], results2: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        combined = results1 + results2
        return sorted(combined, key=lambda x: x.get('score', 0), reverse=True)[:self.config.MAX_RESULTS]

    async def _rerank_results(self, results: List[Dict[str, Any]], query: str, plan: Dict[str, Any], agent: AgentInterface) -> List[Dict[str, Any]]:
        # Use the agent's rerank method
        reranked_results = await agent.rerank(query, results, self.config.MAX_RESULTS)
        
        # Apply any additional ranking criteria from the plan
        if "boost_sources" in plan:
            boost_sources = set(plan["boost_sources"])
            reranked_results.sort(key=lambda x: (x.get('source') in boost_sources, x.get('score', 0)), reverse=True)
        
        return reranked_results
    def _is_satisfactory(self, results: List[Dict[str, Any]]) -> bool:
        if len(results) < self.config.MIN_SATISFACTORY_RESULTS:
            return False

        # Check if we have results with high enough scores
        high_score_results = [r for r in results if r.get('score', 0) > self.config.HIGH_SCORE_THRESHOLD]
        if len(high_score_results) >= self.config.MIN_HIGH_SCORE_RESULTS:
            return True
        
        # Check for diversity in sources
        sources = set(r.get('source') for r in results if 'source' in r)
        if len(sources) >= self.config.MIN_DIVERSE_SOURCES:
            return True
        
        return False