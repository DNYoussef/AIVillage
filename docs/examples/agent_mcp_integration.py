"""Example: Agent Integration with HypeRAG MCP Server.

Shows how different agents (King, Sage, Magi) would interact with the MCP server.
"""

import asyncio
import json
import logging
from typing import Any

import websockets

logger = logging.getLogger(__name__)


class HypeRAGMCPClient:
    """Client for connecting to HypeRAG MCP Server."""

    def __init__(self, uri: str, agent_id: str, api_key: str) -> None:
        self.uri = uri
        self.agent_id = agent_id
        self.api_key = api_key
        self.websocket = None
        self.request_id = 0

    async def connect(self) -> None:
        """Connect to the MCP server."""
        self.websocket = await websockets.connect(self.uri)
        logger.info(f"Agent {self.agent_id} connected to HypeRAG MCP Server")

    async def disconnect(self) -> None:
        """Disconnect from the server."""
        if self.websocket:
            await self.websocket.close()
            logger.info(f"Agent {self.agent_id} disconnected")

    async def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Call an MCP method."""
        if not self.websocket:
            msg = "Not connected to server"
            raise RuntimeError(msg)

        self.request_id += 1

        # Add authentication to params
        params["api_key"] = self.api_key

        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": f"{self.agent_id}_{self.request_id}",
        }

        # Send request
        await self.websocket.send(json.dumps(request))

        # Receive response
        response_str = await self.websocket.recv()
        response = json.loads(response_str)

        if "error" in response:
            msg = f"MCP Error: {response['error']['message']}"
            raise Exception(msg)

        return response.get("result", {})


class KingAgent:
    """King Agent - Full access, strategic decision making."""

    def __init__(self, mcp_client: HypeRAGMCPClient) -> None:
        self.mcp = mcp_client
        self.name = "King Agent"

    async def comprehensive_analysis(self, topic: str) -> dict[str, Any]:
        """Perform comprehensive analysis using King's full privileges."""
        logger.info(f"King performing comprehensive analysis on: {topic}")

        # 1. Initial query with high confidence threshold
        query_result = await self.mcp.call(
            "hyperag/query",
            {
                "query": f"Provide comprehensive analysis of {topic}",
                "mode": "NORMAL",
                "plan_hints": {
                    "max_depth": 5,
                    "time_budget_ms": 5000,
                    "confidence_threshold": 0.8,
                    "include_explanations": True,
                },
            },
        )

        # 2. Creative exploration for innovative insights
        creative_result = await self.mcp.call(
            "hyperag/creative",
            {
                "source_concept": topic,
                "creativity_parameters": {
                    "mode": "divergent",
                    "max_hops": 5,
                    "min_surprise": 0.7,
                },
            },
        )

        # 3. Check for potential repairs needed
        # (In a real implementation, this would check specific violations)

        # 4. Add strategic knowledge
        await self.mcp.call(
            "hyperag/knowledge/add",
            {
                "content": f"Strategic analysis of {topic} completed by King Agent",
                "content_type": "analysis",
                "metadata": {
                    "agent": "king",
                    "analysis_type": "comprehensive",
                    "topic": topic,
                },
            },
        )

        return {
            "analysis": query_result,
            "creative_insights": creative_result,
            "status": "comprehensive_analysis_complete",
        }


class SageAgent:
    """Sage Agent - Strategic analysis and knowledge management."""

    def __init__(self, mcp_client: HypeRAGMCPClient) -> None:
        self.mcp = mcp_client
        self.name = "Sage Agent"

    async def strategic_research(self, research_question: str) -> dict[str, Any]:
        """Perform strategic research."""
        logger.info(f"Sage performing strategic research: {research_question}")

        # 1. Search existing knowledge
        search_results = await self.mcp.call(
            "hyperag/knowledge/search",
            {
                "query": research_question,
                "limit": 20,
                "filters": {"content_type": ["research", "analysis", "document"]},
            },
        )

        # 2. Analytical query
        analysis = await self.mcp.call(
            "hyperag/query",
            {
                "query": f"Analyze the strategic implications of: {research_question}",
                "mode": "NORMAL",
                "plan_hints": {
                    "max_depth": 4,
                    "confidence_threshold": 0.75,
                    "prefer_analysis": True,
                },
            },
        )

        # 3. Web research simulation (in real implementation)
        # web_data = await self.perform_web_research(research_question)

        # 4. Add research findings
        await self.mcp.call(
            "hyperag/knowledge/add",
            {
                "content": f"Research findings: {research_question}",
                "content_type": "research",
                "metadata": {
                    "agent": "sage",
                    "research_type": "strategic",
                    "confidence": analysis.get("result", {}).get("confidence", 0.0),
                },
            },
        )

        return {
            "existing_knowledge": search_results,
            "analysis": analysis,
            "status": "strategic_research_complete",
        }


class MagiAgent:
    """Magi Agent - Technical development and code documentation."""

    def __init__(self, mcp_client: HypeRAGMCPClient) -> None:
        self.mcp = mcp_client
        self.name = "Magi Agent"

    async def technical_query(self, technical_question: str) -> dict[str, Any]:
        """Handle technical queries and documentation."""
        logger.info(f"Magi handling technical query: {technical_question}")

        # 1. Technical-focused query
        tech_result = await self.mcp.call(
            "hyperag/query",
            {
                "query": technical_question,
                "mode": "NORMAL",
                "plan_hints": {
                    "max_depth": 3,
                    "confidence_threshold": 0.7,
                    "domain_filter": "technical",
                },
            },
        )

        # 2. Search for existing technical documentation
        docs_search = await self.mcp.call(
            "hyperag/knowledge/search",
            {
                "query": technical_question,
                "limit": 10,
                "filters": {"content_type": ["code", "documentation", "technical"]},
            },
        )

        # 3. Add technical documentation (Magi has write_code_docs permission)
        await self.mcp.call(
            "hyperag/knowledge/add",
            {
                "content": f"Technical documentation: {technical_question}",
                "content_type": "documentation",
                "metadata": {
                    "agent": "magi",
                    "doc_type": "technical",
                    "programming_related": True,
                },
            },
        )

        return {
            "technical_analysis": tech_result,
            "existing_docs": docs_search,
            "status": "technical_query_complete",
        }


async def demonstrate_agent_integration():
    """Demonstrate how different agents integrate with MCP server."""
    # Agent configurations
    agents_config = [
        {
            "id": "king_001",
            "api_key": "hrag_prod_king456",
            "class": KingAgent,
            "task": "artificial intelligence governance",
        },
        {
            "id": "sage_001",
            "api_key": "hrag_prod_sage789",
            "class": SageAgent,
            "task": "future of machine learning research",
        },
        {
            "id": "magi_001",
            "api_key": "hrag_dev_test123",  # Note: limited permissions
            "class": MagiAgent,
            "task": "implementing neural network architectures",
        },
    ]

    mcp_uri = "ws://localhost:8765"
    results = []

    for config in agents_config:
        try:
            # Create MCP client
            client = HypeRAGMCPClient(mcp_uri, config["id"], config["api_key"])
            await client.connect()

            # Create agent
            agent = config["class"](client)

            # Execute agent-specific task
            if isinstance(agent, KingAgent):
                result = await agent.comprehensive_analysis(config["task"])
            elif isinstance(agent, SageAgent):
                result = await agent.strategic_research(config["task"])
            elif isinstance(agent, MagiAgent):
                result = await agent.technical_query(config["task"])
            else:
                result = {"error": "Unknown agent type"}

            results.append({"agent": config["id"], "task": config["task"], "result": result})

            await client.disconnect()

        except Exception as e:
            logger.exception(f"Error with agent {config['id']}: {e!s}")
            results.append({"agent": config["id"], "task": config["task"], "error": str(e)})

    return results


async def test_permission_system():
    """Test the permission system with different agent roles."""
    logger.info("Testing permission system...")

    # Test cases
    test_cases = [
        {
            "agent": "external",
            "api_key": "hrag_dev_test123",
            "method": "hyperag/repair",  # Should fail - no repair permissions
            "params": {
                "violation_type": "test",
                "details": {"description": "test repair"},
            },
            "should_fail": True,
        },
        {
            "agent": "king",
            "api_key": "hrag_prod_king456",
            "method": "hyperag/metrics",  # Should succeed - king has monitor permission
            "params": {},
            "should_fail": False,
        },
        {
            "agent": "sage",
            "api_key": "hrag_prod_sage789",
            "method": "hyperag/adapter/upload",  # Should fail - sage can't manage adapters
            "params": {"name": "test_adapter", "description": "test"},
            "should_fail": True,
        },
    ]

    results = []

    for test_case in test_cases:
        try:
            client = HypeRAGMCPClient("ws://localhost:8765", test_case["agent"], test_case["api_key"])
            await client.connect()

            try:
                await client.call(test_case["method"], test_case["params"])
                success = True
                error = None
            except Exception as e:
                success = False
                error = str(e)

            # Check if result matches expectation
            test_passed = (not success) == test_case["should_fail"]

            results.append(
                {
                    "agent": test_case["agent"],
                    "method": test_case["method"],
                    "expected_failure": test_case["should_fail"],
                    "actual_failure": not success,
                    "test_passed": test_passed,
                    "error": error,
                }
            )

            await client.disconnect()

        except Exception as e:
            results.append(
                {
                    "agent": test_case["agent"],
                    "method": test_case["method"],
                    "test_passed": False,
                    "connection_error": str(e),
                }
            )

    return results


async def main() -> None:
    """Main demonstration function."""
    logging.basicConfig(level=logging.INFO)

    print("=== HypeRAG MCP Server Agent Integration Demo ===\n")

    try:
        # Test 1: Permission system
        print("1. Testing permission system...")
        permission_results = await test_permission_system()

        for result in permission_results:
            status = "✓ PASS" if result["test_passed"] else "✗ FAIL"
            print(f"   {status} {result['agent']} -> {result['method']}")
            if not result["test_passed"] and result.get("error"):
                print(f"      Error: {result['error']}")

        print()

        # Test 2: Agent integration
        print("2. Testing agent integration...")
        integration_results = await demonstrate_agent_integration()

        for result in integration_results:
            if "error" in result:
                print(f"   ✗ {result['agent']}: {result['error']}")
            else:
                print(f"   ✓ {result['agent']}: {result['result']['status']}")

        print("\n=== Demo Complete ===")

    except Exception as e:
        print(f"Demo failed: {e!s}")
        print("Make sure the HypeRAG MCP Server is running on ws://localhost:8765")


if __name__ == "__main__":
    asyncio.run(main())
