from datetime import datetime, timedelta

import pytest
from mcp_servers.hyperag.auth import AuthContext
from mcp_servers.hyperag.mcp_server import HypeRAGMCPServer
from software.hyper_rag.hyper_rag_pipeline import HyperRAGPipeline


@pytest.mark.asyncio
async def test_pipeline_retrieval_hits_top_doc():
    pipe = HyperRAGPipeline()
    await pipe.ingest_knowledge("bananas are yellow", "food", "banana")
    await pipe.ingest_knowledge("apples are red", "food", "apple")
    await pipe.ingest_knowledge("car engines require fuel", "auto", "engine")

    result = await pipe.search("What color are bananas?")
    assert result.items
    assert "banana" in result.items[0].content.lower()
    assert result.metrics["n_candidates"] == 3


@pytest.mark.asyncio
async def test_mcp_round_trip():
    server = HypeRAGMCPServer()
    await server.initialize()
    ctx = AuthContext(
        user_id="u",
        agent_id="a",
        session_id="s",
        role="user",
        permissions=set(),
        expires_at=datetime.now() + timedelta(hours=1),
        ip_address="127.0.0.1",
    )
    await server._handle_memory({"action": "store", "content": "bananas are yellow"}, ctx)
    response = await server._handle_query({"query": "yellow banana"}, ctx)
    assert "answer" in response
    assert response["sources"]
    assert "retrieval_ms" in response["metrics"]
