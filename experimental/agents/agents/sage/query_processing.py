import asyncio
import logging
from typing import Any

from agent_forge.adas.technique_archive import ChainOfThought, TreeOfThoughts
from rag_system.utils.embedding import BERTEmbeddingModel
from rag_system.utils.named_entity_recognition import NamedEntityRecognizer
from rag_system.utils.relation_extraction import RelationExtractor

logger = logging.getLogger(__name__)


class QueryProcessor:
    def __init__(self, rag_system, latent_space_activation, cognitive_nexus) -> None:
        self.rag_system = rag_system
        self.latent_space_activation = latent_space_activation
        self.cognitive_nexus = cognitive_nexus
        self.embedding_model = BERTEmbeddingModel()
        self.named_entity_recognizer = NamedEntityRecognizer()
        self.relation_extractor = RelationExtractor()
        self.chain_of_thought = ChainOfThought()
        self.tree_of_thoughts = TreeOfThoughts()

    async def process_query(self, query: str) -> str:
        try:
            results = await asyncio.gather(
                self.activate_latent_space(query),
                self.query_cognitive_nexus(query),
                self.apply_advanced_reasoning({"content": query}),
                self.query_rag(query),
            )
            (
                activated_knowledge,
                cognitive_context,
                reasoning_result,
                rag_result,
            ) = results

            enhanced_query = f"""
            Original Query: {query}
            Activated Knowledge: {activated_knowledge}
            Cognitive Context: {cognitive_context}
            Advanced Reasoning: {reasoning_result}
            RAG Result: {rag_result}
            """

            return enhanced_query
        except Exception as e:
            logger.exception(f"Error processing query: {e!s}")
            return query

    async def activate_latent_space(self, content: str) -> str:
        try:
            embeddings = self.embedding_model.encode(content)
            entities = self.named_entity_recognizer.recognize(content)
            relations = self.relation_extractor.extract(content)
            return await self.latent_space_activation.activate(
                content, embeddings, entities, relations
            )
        except Exception as e:
            logger.exception(f"Error activating latent space: {e!s}")
            return ""

    async def query_cognitive_nexus(self, content: str) -> str:
        try:
            embeddings = self.embedding_model.encode(content)
            entities = self.named_entity_recognizer.recognize(content)
            return await self.cognitive_nexus.query(content, embeddings, entities)
        except Exception as e:
            logger.exception(f"Error querying cognitive nexus: {e!s}")
            return ""

    async def apply_advanced_reasoning(self, task: dict[str, Any]) -> str:
        try:
            chain_of_thought_result = self.chain_of_thought.process(task["content"])
            tree_of_thoughts_result = await self.tree_of_thoughts.process(
                task["content"]
            )

            combined_reasoning = f"Chain of Thought: {chain_of_thought_result}\n"
            combined_reasoning += f"Tree of Thoughts: {tree_of_thoughts_result}"

            return combined_reasoning
        except Exception as e:
            logger.exception(f"Error applying advanced reasoning: {e!s}")
            return ""

    async def query_rag(self, query: str) -> dict[str, Any]:
        try:
            embeddings = self.embedding_model.encode(query)
            concepts = self.named_entity_recognizer.recognize(query)
            activated_knowledge = await self.activate_latent_space(query)
            cognitive_context = await self.query_cognitive_nexus(query)

            return await self.rag_system.process_query(
                query,
                embeddings=embeddings,
                concepts=concepts,
                activated_knowledge=activated_knowledge,
                cognitive_context=cognitive_context,
            )
        except Exception as e:
            logger.exception(f"Error querying RAG system: {e!s}")
            return {"error": str(e)}
