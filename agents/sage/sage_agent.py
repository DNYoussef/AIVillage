from ..agent import Agent
from ...communications.protocol import StandardCommunicationProtocol
from scrapegraphai.graphs import SmartScraperGraph, SearchGraph
from gpt_researcher import GPTResearcher

class SageAgent(Agent):
    def __init__(self, communication_protocol: StandardCommunicationProtocol):
        super().__init__(
            name="Sage",
            model="gpt-4o-mini",
            instructions=(
                "You are Sage, an AI agent specializing in information gathering and research. "
                "Your role is to find, extract, and synthesize relevant data to support the AI Village."
            ),
            tools=[self.web_search, self.query_knowledge_base]
        )
        self.communication_protocol = communication_protocol
        self.knowledge_base = {}  # TODO: Implement proper knowledge base

    async def web_search(self, query: str) -> str:
        """Perform a web search using Scrapegraph-ai and gpt-researcher."""
        # Use SearchGraph for multi-page search
        search_graph = SearchGraph(query)
        search_results = search_graph.run()

        # Use GPTResearcher to summarize search results
        researcher = GPTResearcher(query=query, report_type="research_report")
        researcher.data = search_results
        summary = await researcher.write_report()

        # Store search results and summary in knowledge base
        self.knowledge_base[query] = {"search_results": search_results, "summary": summary}

        return summary

    async def query_knowledge_base(self, query: str) -> str:
        """Query the internal knowledge base for relevant information."""
        # TODO: Implement fuzzy matching and relevance scoring
        if query in self.knowledge_base:
            return self.knowledge_base[query]["summary"]
        else:
            return "No relevant information found in knowledge base."
