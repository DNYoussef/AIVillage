#!/usr/bin/env python3
"""
Deploy Hyper RAG System for Sage Agent

Initializes and deploys the advanced RAG system with:
- Vector + Graph RAG integration
- Bayesian Belief Engine with probability ratings
- Cognitive Nexus multi-perspective analysis
- Dual context tags (book/chapter summaries)
- Hippo Cache for frequent items
- Read-only access for all agents

Usage:
    python deploy_hyper_rag.py [--sage-agent-id AGENT_ID] [--demo-data] [--dry-run]
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime
from typing import Any

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from software.hyper_rag import HyperRAGPipeline, RAGType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("hyper_rag_deployment.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class HyperRAGDeployer:
    """
    Deploys and configures the Hyper RAG system for Sage Agent.
    """

    def __init__(self, sage_agent_id: str = "sage", dry_run: bool = False):
        self.sage_agent_id = sage_agent_id
        self.dry_run = dry_run
        self.hyper_rag: HyperRAGPipeline = None

        logger.info(f"Hyper RAG Deployer initialized for Sage Agent: {sage_agent_id}")
        if dry_run:
            logger.info(
                "DRY RUN MODE: System will be initialized but not persistently deployed"
            )

    async def deploy_hyper_rag(self) -> bool:
        """
        Deploy the complete Hyper RAG system.

        Returns:
            bool: True if deployment successful, False otherwise
        """
        try:
            logger.info("🚀 Starting Hyper RAG System deployment...")

            # Step 1: Initialize Hyper RAG Pipeline
            if not await self._initialize_pipeline():
                return False

            # Step 2: Load demo knowledge data (if requested)
            if hasattr(self, "load_demo_data") and self.load_demo_data:
                if not await self._load_demo_knowledge():
                    return False

            # Step 3: Test all RAG retrieval methods
            if not await self._test_rag_methods():
                return False

            # Step 4: Validate Bayesian belief propagation
            if not await self._test_belief_propagation():
                return False

            # Step 5: Test Cognitive Nexus integration
            if not await self._test_cognitive_analysis():
                return False

            # Step 6: Performance and capacity testing
            if not await self._test_system_performance():
                return False

            logger.info("✅ Hyper RAG System deployment completed successfully!")

            # Display system statistics
            await self._display_system_stats()

            return True

        except Exception as e:
            logger.error(f"❌ Hyper RAG deployment failed: {e}")
            return False

    async def _initialize_pipeline(self) -> bool:
        """Initialize the Hyper RAG pipeline with all components."""
        try:
            logger.info("Initializing Hyper RAG Pipeline...")

            self.hyper_rag = HyperRAGPipeline(self.sage_agent_id)

            # Verify components initialized
            assert self.hyper_rag.belief_engine is not None
            assert self.hyper_rag.cognitive_nexus is not None
            assert isinstance(self.hyper_rag.knowledge_items, dict)
            assert isinstance(self.hyper_rag.context_hierarchy, dict)
            assert isinstance(self.hyper_rag.semantic_graph, dict)

            logger.info("✓ Hyper RAG Pipeline initialized successfully")
            logger.info(f"✓ Managed by Sage Agent: {self.sage_agent_id}")
            logger.info(
                "✓ All components ready: Bayesian Engine, Cognitive Nexus, Knowledge Store"
            )

            return True

        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            return False

    async def _load_demo_knowledge(self) -> bool:
        """Load demonstration knowledge data into the system."""
        try:
            logger.info("Loading demonstration knowledge data...")

            # Demo knowledge items with dual context tags
            demo_knowledge = [
                {
                    "content": "Machine learning models require large datasets for effective training. The quality of training data directly impacts model performance and generalization capabilities.",
                    "book_summary": "Artificial Intelligence Fundamentals",
                    "chapter_summary": "Machine Learning Basics",
                    "confidence": 0.9,
                },
                {
                    "content": "Neural networks use backpropagation to adjust weights during training. This process minimizes the loss function by computing gradients through the computational graph.",
                    "book_summary": "Artificial Intelligence Fundamentals",
                    "chapter_summary": "Deep Learning Concepts",
                    "confidence": 0.95,
                },
                {
                    "content": "Transformers have revolutionized natural language processing by using self-attention mechanisms. They can process sequences in parallel rather than sequentially.",
                    "book_summary": "Modern NLP Techniques",
                    "chapter_summary": "Transformer Architecture",
                    "confidence": 0.92,
                },
                {
                    "content": "Large language models exhibit emergent behaviors at scale, including few-shot learning and in-context learning capabilities that weren't explicitly trained.",
                    "book_summary": "Modern NLP Techniques",
                    "chapter_summary": "Emergent Capabilities",
                    "confidence": 0.87,
                },
                {
                    "content": "Reinforcement learning agents learn through interaction with environments, receiving rewards and penalties that shape their behavior over time.",
                    "book_summary": "Advanced AI Methods",
                    "chapter_summary": "Reinforcement Learning",
                    "confidence": 0.88,
                },
                {
                    "content": "Multi-agent systems involve multiple autonomous agents that can cooperate, compete, or negotiate to achieve individual or collective goals.",
                    "book_summary": "Advanced AI Methods",
                    "chapter_summary": "Multi-Agent Systems",
                    "confidence": 0.85,
                },
                {
                    "content": "Cybersecurity requires defense-in-depth strategies with multiple overlapping security controls to protect against sophisticated attacks.",
                    "book_summary": "Information Security Principles",
                    "chapter_summary": "Defense Strategies",
                    "confidence": 0.93,
                },
                {
                    "content": "Penetration testing simulates real-world attacks to identify vulnerabilities before malicious actors can exploit them.",
                    "book_summary": "Information Security Principles",
                    "chapter_summary": "Vulnerability Assessment",
                    "confidence": 0.91,
                },
                {
                    "content": "Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information in fundamentally different ways than classical computers.",
                    "book_summary": "Quantum Computing Introduction",
                    "chapter_summary": "Quantum Mechanics in Computing",
                    "confidence": 0.89,
                },
                {
                    "content": "Quantum algorithms like Shor's algorithm can factor large integers exponentially faster than classical algorithms, threatening current cryptographic systems.",
                    "book_summary": "Quantum Computing Introduction",
                    "chapter_summary": "Quantum Algorithms",
                    "confidence": 0.94,
                },
            ]

            # Ingest knowledge items
            ingested_ids = []
            for item in demo_knowledge:
                item_id = await self.hyper_rag.ingest_knowledge(
                    content=item["content"],
                    book_summary=item["book_summary"],
                    chapter_summary=item["chapter_summary"],
                    source_confidence=item["confidence"],
                )
                ingested_ids.append(item_id)

            logger.info(f"✓ Ingested {len(ingested_ids)} demo knowledge items")
            logger.info(
                f"✓ Context hierarchy: {len(self.hyper_rag.context_hierarchy)} books"
            )

            # Verify semantic connections were created
            total_connections = sum(
                len(item.semantic_connections or [])
                for item in self.hyper_rag.knowledge_items.values()
            )
            logger.info(f"✓ Generated {total_connections} semantic connections")

            return True

        except Exception as e:
            logger.error(f"Demo knowledge loading failed: {e}")
            return False

    async def _test_rag_methods(self) -> bool:
        """Test all RAG retrieval methods."""
        try:
            logger.info("Testing RAG retrieval methods...")

            test_queries = [
                "How do neural networks learn?",
                "What are transformers in AI?",
                "Explain cybersecurity defense strategies",
                "What is quantum computing?",
            ]

            # Test each RAG method
            for method in [
                RAGType.VECTOR,
                RAGType.GRAPH,
                RAGType.BAYESIAN,
                RAGType.HYBRID,
            ]:
                logger.info(f"Testing {method.value} retrieval...")

                method_results = []
                for query in test_queries:
                    result = await self.hyper_rag.retrieve_knowledge(
                        query=query, retrieval_type=method, max_results=3
                    )
                    method_results.append(result)

                # Verify results
                avg_confidence = sum(r.confidence_score for r in method_results) / len(
                    method_results
                )
                avg_items = sum(len(r.items) for r in method_results) / len(
                    method_results
                )

                logger.info(
                    f"✓ {method.value}: {avg_confidence:.3f} avg confidence, {avg_items:.1f} avg items"
                )

                # Ensure all methods return reasonable results
                if avg_confidence < 0.3:
                    logger.warning(f"{method.value} method has low confidence scores")

                if avg_items < 1:
                    logger.error(f"{method.value} method failed to return results")
                    return False

            logger.info("✓ All RAG retrieval methods tested successfully")
            return True

        except Exception as e:
            logger.error(f"RAG methods testing failed: {e}")
            return False

    async def _test_belief_propagation(self) -> bool:
        """Test Bayesian belief propagation functionality."""
        try:
            logger.info("Testing Bayesian belief propagation...")

            # Get initial belief state
            initial_beliefs = {
                belief_id: belief.probability
                for belief_id, belief in self.hyper_rag.belief_engine.beliefs.items()
            }

            # Perform Bayesian retrieval which should update beliefs
            test_query = "machine learning training data quality"
            bayesian_result = await self.hyper_rag.retrieve_knowledge(
                query=test_query, retrieval_type=RAGType.BAYESIAN, max_results=5
            )

            # Check belief updates
            updated_beliefs = {
                belief_id: belief.probability
                for belief_id, belief in self.hyper_rag.belief_engine.beliefs.items()
            }

            # Count belief changes
            beliefs_changed = 0
            significant_changes = 0

            for belief_id in initial_beliefs:
                if belief_id in updated_beliefs:
                    initial_prob = initial_beliefs[belief_id]
                    updated_prob = updated_beliefs[belief_id]
                    change = abs(updated_prob - initial_prob)

                    if change > 0.001:
                        beliefs_changed += 1
                    if change > 0.05:
                        significant_changes += 1

            logger.info(
                f"✓ Belief propagation: {beliefs_changed} beliefs changed, {significant_changes} significant changes"
            )

            # Verify propagation through semantic connections
            propagated_beliefs = 0
            for item in bayesian_result.items:
                for connected_id in item.semantic_connections or []:
                    if (
                        connected_id in updated_beliefs
                        and connected_id in initial_beliefs
                    ):
                        change = abs(
                            updated_beliefs[connected_id]
                            - initial_beliefs[connected_id]
                        )
                        if change > 0.01:
                            propagated_beliefs += 1

            logger.info(
                f"✓ Semantic propagation: {propagated_beliefs} connected beliefs updated"
            )

            # Verify Bayesian scores in results
            assert len(bayesian_result.bayesian_scores) == len(bayesian_result.items)
            avg_bayesian_score = sum(bayesian_result.bayesian_scores.values()) / len(
                bayesian_result.bayesian_scores
            )
            logger.info(f"✓ Average Bayesian score: {avg_bayesian_score:.3f}")

            return True

        except Exception as e:
            logger.error(f"Belief propagation testing failed: {e}")
            return False

    async def _test_cognitive_analysis(self) -> bool:
        """Test Cognitive Nexus integration for multi-perspective analysis."""
        try:
            logger.info("Testing Cognitive Nexus integration...")

            # Retrieve knowledge for analysis
            test_query = "AI and cybersecurity integration"
            retrieval_result = await self.hyper_rag.retrieve_knowledge(
                query=test_query, retrieval_type=RAGType.HYBRID, max_results=4
            )

            # Perform cognitive analysis
            cognitive_analysis = await self.hyper_rag.analyze_with_cognitive_nexus(
                retrieval_result=retrieval_result, query=test_query
            )

            # Verify cognitive analysis components
            required_components = [
                "multi_perspective_analysis",
                "synthesis_result",
                "uncertainty_quantification",
                "context_relevance_assessment",
            ]

            for component in required_components:
                if component not in cognitive_analysis:
                    logger.error(f"Missing cognitive analysis component: {component}")
                    return False

            logger.info("✓ Cognitive Nexus analysis completed successfully")

            # Log analysis insights
            if "synthesis_result" in cognitive_analysis:
                synthesis = cognitive_analysis["synthesis_result"]
                logger.info(
                    f"✓ Synthesis insights: {len(synthesis.get('key_insights', []))} insights generated"
                )

            if "uncertainty_quantification" in cognitive_analysis:
                uncertainty = cognitive_analysis["uncertainty_quantification"]
                logger.info(
                    f"✓ Uncertainty analysis: {uncertainty.get('overall_confidence', 0):.3f} confidence"
                )

            return True

        except Exception as e:
            logger.error(f"Cognitive analysis testing failed: {e}")
            return False

    async def _test_system_performance(self) -> bool:
        """Test system performance and capacity."""
        try:
            logger.info("Testing system performance and capacity...")

            # Test concurrent retrievals
            concurrent_queries = [
                "machine learning algorithms",
                "cybersecurity threats",
                "quantum computing applications",
                "AI ethics principles",
                "data privacy protection",
            ]

            # Perform concurrent retrievals
            start_time = datetime.now()

            concurrent_tasks = [
                self.hyper_rag.retrieve_knowledge(query, RAGType.HYBRID, 3)
                for query in concurrent_queries
            ]

            concurrent_results = await asyncio.gather(*concurrent_tasks)

            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()

            # Verify all queries succeeded
            for i, result in enumerate(concurrent_results):
                if len(result.items) == 0:
                    logger.error(f"Concurrent query {i} returned no results")
                    return False

            logger.info(
                f"✓ Concurrent performance: {len(concurrent_queries)} queries in {total_time:.3f}s"
            )
            logger.info(
                f"✓ Average query time: {total_time / len(concurrent_queries):.3f}s"
            )

            # Test cache performance
            cache_query = concurrent_queries[0]

            # First query (should populate cache)
            cache_start = datetime.now()
            first_result = await self.hyper_rag.retrieve_knowledge(
                cache_query, RAGType.HYBRID, 3
            )
            first_time = (datetime.now() - cache_start).total_seconds()

            # Second identical query (should hit cache)
            cache_start = datetime.now()
            cached_result = await self.hyper_rag.retrieve_knowledge(
                cache_query, RAGType.HYBRID, 3
            )
            cached_time = (datetime.now() - cache_start).total_seconds()

            # Cache should improve performance
            if self.hyper_rag.retrieval_stats["cache_hits"] > 0:
                logger.info(
                    f"✓ Cache performance: {first_time:.3f}s → {cached_time:.3f}s"
                )
                logger.info(
                    f"✓ Cache hits: {self.hyper_rag.retrieval_stats['cache_hits']}"
                )

            # Test memory usage (simplified)
            knowledge_items_size = len(self.hyper_rag.knowledge_items)
            beliefs_size = len(self.hyper_rag.belief_engine.beliefs)
            semantic_connections = sum(
                len(item.semantic_connections or [])
                for item in self.hyper_rag.knowledge_items.values()
            )

            logger.info(
                f"✓ Memory usage: {knowledge_items_size} items, {beliefs_size} beliefs, {semantic_connections} connections"
            )

            return True

        except Exception as e:
            logger.error(f"Performance testing failed: {e}")
            return False

    async def _display_system_stats(self):
        """Display comprehensive system statistics."""
        logger.info("📊 Hyper RAG System Statistics:")

        stats = self.hyper_rag.get_system_stats()

        # Knowledge statistics
        logger.info(f"  Knowledge Items: {stats['knowledge_items']}")
        logger.info(f"  Cached Items: {stats['cached_items']}")

        # Belief statistics
        belief_stats = stats["belief_statistics"]
        logger.info(f"  Total Beliefs: {belief_stats['total_beliefs']}")
        logger.info(
            f"  Average Belief Probability: {belief_stats['avg_probability']:.3f}"
        )
        logger.info(
            f"  High Confidence Beliefs (>0.8): {belief_stats['high_confidence_beliefs']}"
        )
        logger.info(
            f"  Low Confidence Beliefs (<0.3): {belief_stats['low_confidence_beliefs']}"
        )

        # Context statistics
        context_stats = stats["context_statistics"]
        logger.info(f"  Context Books: {context_stats['total_context_books']}")
        logger.info(f"  Context Chapters: {context_stats['total_context_chapters']}")
        logger.info(
            f"  Avg Chapters per Book: {context_stats['avg_chapters_per_book']:.1f}"
        )

        # Semantic graph statistics
        graph_stats = stats["semantic_graph_statistics"]
        logger.info(
            f"  Semantic Connections: {graph_stats['total_semantic_connections']}"
        )
        logger.info(
            f"  Highly Connected Items (>5 connections): {graph_stats['highly_connected_items']}"
        )
        logger.info(
            f"  Avg Connections per Item: {graph_stats['avg_connections_per_item']:.1f}"
        )

        # Retrieval statistics
        retrieval_stats = stats["retrieval_statistics"]
        logger.info(f"  Total Queries: {retrieval_stats['total_queries']}")
        logger.info(f"  Vector Retrievals: {retrieval_stats['vector_retrievals']}")
        logger.info(f"  Graph Retrievals: {retrieval_stats['graph_retrievals']}")
        logger.info(f"  Bayesian Retrievals: {retrieval_stats['bayesian_retrievals']}")
        logger.info(f"  Hybrid Retrievals: {retrieval_stats['hybrid_retrievals']}")
        logger.info(f"  Cache Hits: {retrieval_stats['cache_hits']}")
        logger.info(
            f"  Average Response Time: {retrieval_stats['avg_response_time']:.3f}s"
        )

    def get_deployment_status(self) -> dict[str, Any]:
        """Get deployment status summary."""
        status = {
            "deployment_time": datetime.now().isoformat(),
            "sage_agent_id": self.sage_agent_id,
            "dry_run_mode": self.dry_run,
            "system_deployed": self.hyper_rag is not None,
        }

        if self.hyper_rag:
            status["system_stats"] = self.hyper_rag.get_system_stats()

        return status


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(
        description="Deploy Hyper RAG System for Sage Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python deploy_hyper_rag.py                         # Deploy with default sage agent
  python deploy_hyper_rag.py --sage-agent-id custom  # Deploy for custom sage agent
  python deploy_hyper_rag.py --demo-data             # Load demonstration knowledge
  python deploy_hyper_rag.py --dry-run              # Test deployment without persistence
        """,
    )

    parser.add_argument(
        "--sage-agent-id",
        type=str,
        default="sage",
        help="Sage Agent ID to manage the RAG system (default: sage)",
    )

    parser.add_argument(
        "--demo-data", action="store_true", help="Load demonstration knowledge data"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test deployment without persistent storage",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create deployer
    deployer = HyperRAGDeployer(sage_agent_id=args.sage_agent_id, dry_run=args.dry_run)

    # Set demo data flag
    if args.demo_data:
        deployer.load_demo_data = True

    async def run_deployment():
        success = await deployer.deploy_hyper_rag()

        if success:
            logger.info("📈 Final deployment status:")
            status = deployer.get_deployment_status()
            for key, value in status.items():
                if key != "system_stats":  # Don't print detailed stats again
                    logger.info(f"  {key}: {value}")

            logger.info("🧠 Hyper RAG System is now operational!")
            logger.info(f"Managed by Sage Agent: {args.sage_agent_id}")
            logger.info(
                "Features: Vector+Graph+Bayesian RAG, Cognitive Nexus, Dual Context Tags"
            )

            return 0
        else:
            logger.error("❌ Deployment failed - see errors above")
            return 1

    try:
        exit_code = asyncio.run(run_deployment())
        return exit_code
    except KeyboardInterrupt:
        logger.info("Deployment interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
