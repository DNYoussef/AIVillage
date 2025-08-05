from datetime import datetime

import duckdb
import pytest

from src.mcp_servers.hyperag.memory.hippo_index import HippoIndex, HippoNode


@pytest.mark.asyncio
async def test_memory_stats_include_usage_and_consolidation(tmp_path):
    db_path = tmp_path / "hippo.duckdb"
    index = HippoIndex(db_path=str(db_path))
    index.duckdb_conn = duckdb.connect(str(db_path))
    await index._setup_duckdb_schema()

    node = HippoNode("test content")
    index.duckdb_conn.execute(
        "INSERT INTO hippo_nodes (id, content, node_type, memory_type) VALUES (?, ?, ?, ?)",
        [node.id, node.content, node.node_type, node.memory_type.value],
    )

    class DummyConsolidator:
        def __init__(self):
            self.last_consolidation = datetime.now()
            self.pending_consolidations = 3

    index.set_consolidator(DummyConsolidator())
    stats = await index.get_memory_stats()
    assert stats.total_nodes > 0
    assert stats.memory_usage_mb > 0
    assert stats.last_consolidation is not None
    assert stats.pending_consolidations == 3
