import logging

from agent_forge.adas.technique_archive import ChainOfThought
from rag_system.utils.named_entity_recognition import NamedEntityRecognizer
from rag_system.utils.relation_extraction import RelationExtractor

logger = logging.getLogger(__name__)


class ResearchCapabilities:
    def __init__(self, agent) -> None:
        self.agent = agent
        self.chain_of_thought = ChainOfThought()
        self.named_entity_recognizer = NamedEntityRecognizer()
        self.relation_extractor = RelationExtractor()
        self.capability_metrics: dict[str, dict[str, int]] = {
            cap: {"success": 0, "fail": 0}
            for cap in getattr(agent, "research_capabilities", [])
        }

    def record_result(self, capability: str, success: bool) -> None:
        """Record the outcome of a capability usage."""
        stats = self.capability_metrics.setdefault(
            capability, {"success": 0, "fail": 0}
        )
        if success:
            stats["success"] += 1
        else:
            stats["fail"] += 1

    async def handle_web_search(self, task):
        search_query = task["content"]
        reasoning = self.chain_of_thought.process(search_query)
        search_result = await self.agent.perform_web_search(search_query)
        return {
            "search_query": search_query,
            "reasoning": reasoning,
            "search_result": search_result,
        }

    async def handle_web_scrape(self, task):
        url = task["content"]
        reasoning = self.chain_of_thought.process(f"Scrape information from {url}")
        scrape_result = await self.agent.perform_web_scrape(url)
        return {
            "url": url,
            "reasoning": reasoning,
            "scrape_result": scrape_result,
        }

    async def handle_data_analysis(self, task):
        data = task["content"]
        reasoning = self.chain_of_thought.process(f"Analyze data: {data}")
        analysis_result = await self.agent.analyze_data(data)
        return {
            "data": data,
            "reasoning": reasoning,
            "analysis_result": analysis_result,
        }

    async def handle_information_synthesis(self, task):
        info = task["content"]
        reasoning = self.chain_of_thought.process(f"Synthesize information: {info}")
        entities = self.named_entity_recognizer.recognize(info)
        relations = self.relation_extractor.extract(info)
        synthesis_result = await self.agent.synthesize_information(info)
        return {
            "info": info,
            "reasoning": reasoning,
            "entities": entities,
            "relations": relations,
            "synthesis_result": synthesis_result,
        }

    async def handle_exploration_mode(self, task):
        query = task["content"]
        processed_query = await self.agent.query_processor.process_query(query)
        exploration_results = await self.agent.exploration_mode.discover_new_relations(
            processed_query
        )
        return {
            "query": query,
            "processed_query": processed_query,
            "exploration_results": exploration_results,
        }

    async def evolve_research_capabilities(self) -> None:
        """Adapt the agent's research capabilities based on past performance.

        Success and failure counts for each capability are stored in
        ``self.capability_metrics``.  When enough data points are available, the
        method will enable or disable the capability on ``self.agent`` depending
        on the observed success rate.
        """
        for capability, stats in self.capability_metrics.items():
            attempts = stats["success"] + stats["fail"]
            if attempts < 5:
                # Not enough data to make a decision
                continue

            success_rate = stats["success"] / attempts

            if success_rate < 0.3 and capability in self.agent.research_capabilities:
                self.agent.research_capabilities.remove(capability)
                logger.info(
                    "Disabled capability %s due to low success rate %.2f",
                    capability,
                    success_rate,
                )
            elif (
                success_rate > 0.8
                and capability not in self.agent.research_capabilities
            ):
                self.agent.research_capabilities.append(capability)
                logger.info(
                    "Enabled capability %s due to high success rate %.2f",
                    capability,
                    success_rate,
                )

        # Ensure metrics exist for any newly added capabilities
        for cap in getattr(self.agent, "research_capabilities", []):
            self.capability_metrics.setdefault(cap, {"success": 0, "fail": 0})
