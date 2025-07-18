"""Standardized Query Processing for SAGE Agent

This is a refactored version of query_processing.py that demonstrates
the standardized approach using BaseProcessHandler framework.
"""

import asyncio
import logging
from typing import Any

from agent_forge.adas.technique_archive import ChainOfThought, TreeOfThoughts
from agents.base import ProcessConfig, QueryProcessor
from rag_system.utils.embedding import BERTEmbeddingModel
from rag_system.utils.named_entity_recognition import NamedEntityRecognizer
from rag_system.utils.relation_extraction import RelationExtractor

logger = logging.getLogger(__name__)


class SageQueryProcessor(QueryProcessor):
    """Standardized SAGE query processor with enhanced error handling,
    performance monitoring, and retry logic.
    """

    def __init__(self, rag_system, latent_space_activation, cognitive_nexus):
        # Configure processing behavior
        config = ProcessConfig(
            timeout_seconds=30.0,  # 30 second timeout for complex queries
            retry_attempts=2,  # Retry failed queries twice
            retry_delay_seconds=1.0,  # 1 second delay between retries
            enable_logging=True,  # Enable detailed logging
            enable_metrics=True,  # Track performance metrics
            validation_enabled=True,  # Validate query inputs
        )

        super().__init__("SageQueryProcessor", config)

        # Initialize components
        self.rag_system = rag_system
        self.latent_space_activation = latent_space_activation
        self.cognitive_nexus = cognitive_nexus
        self.embedding_model = BERTEmbeddingModel()
        self.named_entity_recognizer = NamedEntityRecognizer()
        self.relation_extractor = RelationExtractor()
        self.chain_of_thought = ChainOfThought()
        self.tree_of_thoughts = TreeOfThoughts()

    async def _validate_input(self, input_data) -> None:
        """Enhanced input validation for SAGE queries."""
        await super()._validate_input(input_data)

        # Convert input to query string
        if isinstance(input_data, str):
            query = input_data
        elif isinstance(input_data, dict):
            query = input_data.get("query", str(input_data))
        else:
            query = str(input_data)

        # Validate query content
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if len(query) > 10000:  # Reasonable limit for query length
            raise ValueError("Query too long (max 10000 characters)")

    async def _process_query(self, query: str, **kwargs) -> str:
        """Core SAGE query processing with parallel component execution.

        This method orchestrates all SAGE components (latent space activation,
        cognitive nexus, advanced reasoning, and RAG) to provide enhanced
        query responses.
        """
        # Execute all components in parallel for efficiency
        results = await asyncio.gather(
            self.activate_latent_space(query),
            self.query_cognitive_nexus(query),
            self.apply_advanced_reasoning({"content": query}),
            self.query_rag(query),
            return_exceptions=True,  # Capture exceptions without failing entire process
        )

        # Process results with error handling
        activated_knowledge, cognitive_context, reasoning_result, rag_result = results

        # Handle any component failures gracefully
        activated_knowledge = self._handle_component_result(
            activated_knowledge,
            "latent space activation",
            "No latent knowledge available",
        )
        cognitive_context = self._handle_component_result(
            cognitive_context, "cognitive nexus", "No cognitive context available"
        )
        reasoning_result = self._handle_component_result(
            reasoning_result, "advanced reasoning", "No reasoning insights available"
        )
        rag_result = self._handle_component_result(
            rag_result, "RAG system", "No RAG results available"
        )

        # Construct enhanced response
        enhanced_query = f"""
Original Query: {query}

Latent Space Activation: {activated_knowledge}

Cognitive Context: {cognitive_context}

Advanced Reasoning: {reasoning_result}

RAG Results: {rag_result}

Synthesis: {self._synthesize_results(query, activated_knowledge, cognitive_context, reasoning_result, rag_result)}
"""

        return enhanced_query.strip()

    def _handle_component_result(
        self, result: Any, component_name: str, fallback: str
    ) -> str:
        """Handle individual component results with error checking."""
        if isinstance(result, Exception):
            self.logger.warning(f"{component_name} failed: {result}")
            return fallback
        if result is None or result == "":
            self.logger.info(f"{component_name} returned empty result")
            return fallback
        return str(result)

    def _synthesize_results(
        self,
        query: str,
        activated_knowledge: str,
        cognitive_context: str,
        reasoning_result: str,
        rag_result: str,
    ) -> str:
        """Synthesize all component results into a coherent response."""
        # Simple synthesis - can be enhanced with more sophisticated logic
        available_info = []

        if "No " not in activated_knowledge:
            available_info.append("latent knowledge")
        if "No " not in cognitive_context:
            available_info.append("cognitive insights")
        if "No " not in reasoning_result:
            available_info.append("reasoning analysis")
        if "No " not in rag_result:
            available_info.append("knowledge base information")

        if available_info:
            return f"Enhanced response incorporates {', '.join(available_info)} to provide comprehensive analysis."
        return "Response generated using available processing capabilities."

    async def activate_latent_space(self, content: str) -> str:
        """Activate latent space with enhanced error handling.

        This method is now a private utility called by the main processor
        but includes improved error handling and logging.
        """
        try:
            # Extract features in parallel
            features = await asyncio.gather(
                self._async_encode(content),
                self._async_recognize_entities(content),
                self._async_extract_relations(content),
            )

            embeddings, entities, relations = features

            # Activate latent space with all features
            result = await self.latent_space_activation.activate(
                content, embeddings, entities, relations
            )

            return result or "Latent space activated successfully"

        except Exception as e:
            self.logger.error(f"Latent space activation failed: {e}")
            raise  # Re-raise for standardized error handling

    async def query_cognitive_nexus(self, content: str) -> str:
        """Query cognitive nexus with enhanced error handling."""
        try:
            # Extract features for cognitive processing
            embeddings = await self._async_encode(content)
            entities = await self._async_recognize_entities(content)

            # Query cognitive nexus
            result = await self.cognitive_nexus.query(content, embeddings, entities)

            return result or "Cognitive processing completed"

        except Exception as e:
            self.logger.error(f"Cognitive nexus query failed: {e}")
            raise

    async def apply_advanced_reasoning(self, task: dict[str, Any]) -> str:
        """Apply advanced reasoning techniques with error handling."""
        try:
            content = task.get("content", "")

            # Apply reasoning techniques in parallel
            reasoning_results = await asyncio.gather(
                self._async_chain_of_thought(content),
                self._async_tree_of_thoughts(task),
                return_exceptions=True,
            )

            cot_result, tot_result = reasoning_results

            # Combine reasoning results
            results = []
            if not isinstance(cot_result, Exception):
                results.append(f"Chain of Thought: {cot_result}")
            if not isinstance(tot_result, Exception):
                results.append(f"Tree of Thoughts: {tot_result}")

            return " | ".join(results) if results else "Advanced reasoning applied"

        except Exception as e:
            self.logger.error(f"Advanced reasoning failed: {e}")
            raise

    async def query_rag(self, query: str) -> str:
        """Query RAG system with enhanced error handling."""
        try:
            if hasattr(self.rag_system, "process_query"):
                result = await self.rag_system.process_query(query)
            elif hasattr(self.rag_system, "query"):
                result = await self.rag_system.query(query)
            else:
                result = str(self.rag_system)

            return result or "RAG query processed"

        except Exception as e:
            self.logger.error(f"RAG query failed: {e}")
            raise

    # Async wrapper methods for synchronous operations
    async def _async_encode(self, content: str):
        """Async wrapper for embedding encoding."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.embedding_model.encode, content
        )

    async def _async_recognize_entities(self, content: str):
        """Async wrapper for entity recognition."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.named_entity_recognizer.recognize, content
        )

    async def _async_extract_relations(self, content: str):
        """Async wrapper for relation extraction."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.relation_extractor.extract, content
        )

    async def _async_chain_of_thought(self, content: str):
        """Async wrapper for chain of thought processing."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.chain_of_thought.process, content
        )

    async def _async_tree_of_thoughts(self, task: dict[str, Any]):
        """Async wrapper for tree of thoughts processing."""
        if asyncio.iscoroutinefunction(self.tree_of_thoughts.process):
            return await self.tree_of_thoughts.process(task)
        return await asyncio.get_event_loop().run_in_executor(
            None, self.tree_of_thoughts.process, task
        )


# Backward compatibility wrapper
class QueryProcessor:
    """Backward compatibility wrapper for existing code.

    This allows existing code to continue working while gradually
    migrating to the standardized approach.
    """

    def __init__(self, rag_system, latent_space_activation, cognitive_nexus):
        self._standardized_processor = SageQueryProcessor(
            rag_system, latent_space_activation, cognitive_nexus
        )

    async def process_query(self, query: str) -> str:
        """Backward compatible process_query method."""
        result = await self._standardized_processor.process(query)

        if result.is_error:
            # Return original query on error for backward compatibility
            logger.error(f"Standardized processing failed: {result.error}")
            return query

        return result.data

    # Expose other methods for compatibility
    async def activate_latent_space(self, content: str) -> str:
        return await self._standardized_processor.activate_latent_space(content)

    async def query_cognitive_nexus(self, content: str) -> str:
        return await self._standardized_processor.query_cognitive_nexus(content)

    async def apply_advanced_reasoning(self, task: dict[str, Any]) -> str:
        return await self._standardized_processor.apply_advanced_reasoning(task)

    async def query_rag(self, query: str) -> str:
        return await self._standardized_processor.query_rag(query)

    @property
    def metrics(self) -> dict[str, Any]:
        """Access performance metrics."""
        return self._standardized_processor.metrics

    @property
    def success_rate(self) -> float:
        """Access success rate statistics."""
        return self._standardized_processor.success_rate


# Example usage and testing
if __name__ == "__main__":
    import asyncio

    async def test_standardized_processor():
        """Test the standardized query processor."""

        # Mock components for testing
        class MockRAGSystem:
            async def process_query(self, query):
                return f"RAG response for: {query}"

        class MockLatentSpace:
            async def activate(self, content, embeddings, entities, relations):
                return f"Activated latent space for: {content[:50]}..."

        class MockCognitiveNexus:
            async def query(self, content, embeddings, entities):
                return f"Cognitive insights for: {content[:50]}..."

        # Create processor with mocks
        processor = SageQueryProcessor(
            MockRAGSystem(), MockLatentSpace(), MockCognitiveNexus()
        )

        # Test processing
        test_query = "What are the implications of artificial general intelligence?"
        result = await processor.process(test_query)

        print("Processing Result:")
        print(f"Status: {result.status}")
        print(f"Processing Time: {result.processing_time_ms:.2f}ms")
        print(f"Data: {result.data[:200]}...")

        # Test metrics
        print("\nMetrics:")
        print(f"Success Rate: {processor.success_rate:.1f}%")
        print(
            f"Average Processing Time: {processor.metrics.get('avg_processing_time', 0):.2f}ms"
        )

    # Run test
    asyncio.run(test_standardized_processor())
