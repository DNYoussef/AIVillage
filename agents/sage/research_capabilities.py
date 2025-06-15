from typing import Dict, Any
from agent_forge.adas.technique_archive import ChainOfThought
from rag_system.utils.named_entity_recognition import NamedEntityRecognizer
from rag_system.utils.relation_extraction import RelationExtractor
import logging

logger = logging.getLogger(__name__)

class ResearchCapabilities:
    def __init__(self, agent):
        self.agent = agent
        self.chain_of_thought = ChainOfThought()
        self.named_entity_recognizer = NamedEntityRecognizer()
        self.relation_extractor = RelationExtractor()

    async def handle_web_search(self, task):
        search_query = task['content']
        reasoning = self.chain_of_thought.process(search_query)
        search_result = await self.agent.perform_web_search(search_query)
        return {
            "search_query": search_query,
            "reasoning": reasoning,
            "search_result": search_result
        }

    async def handle_web_scrape(self, task):
        url = task['content']
        reasoning = self.chain_of_thought.process(f"Scrape information from {url}")
        scrape_result = await self.agent.perform_web_scrape(url)
        return {
            "url": url,
            "reasoning": reasoning,
            "scrape_result": scrape_result,
        }

    async def handle_data_analysis(self, task):
        data = task['content']
        reasoning = self.chain_of_thought.process(f"Analyze data: {data}")
        analysis_result = await self.agent.analyze_data(data)
        return {
            "data": data,
            "reasoning": reasoning,
            "analysis_result": analysis_result
        }

    async def handle_information_synthesis(self, task):
        info = task['content']
        reasoning = self.chain_of_thought.process(f"Synthesize information: {info}")
        entities = self.named_entity_recognizer.recognize(info)
        relations = self.relation_extractor.extract(info)
        synthesis_result = await self.agent.synthesize_information(info)
        return {
            "info": info,
            "reasoning": reasoning,
            "entities": entities,
            "relations": relations,
            "synthesis_result": synthesis_result
        }

    async def handle_exploration_mode(self, task):
        query = task['content']
        processed_query = await self.agent.query_processor.process_query(query)
        exploration_results = await self.agent.exploration_mode.discover_new_relations(processed_query)
        return {
            "query": query,
            "processed_query": processed_query,
            "exploration_results": exploration_results
        }

    async def evolve_research_capabilities(self):
        """Update internal components using recent learning data."""
        logger.info("Evolving research capabilities for %s", self.agent.name)
        if hasattr(self.agent, "continuous_learning_layer"):
            await self.agent.continuous_learning_layer.evolve()
