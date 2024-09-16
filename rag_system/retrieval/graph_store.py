# rag_system/retrieval/graph_store.py

from typing import List, Dict, Any
from neo4j import GraphDatabase
from ..core.interfaces import Retriever
from ..core.config import RAGConfig

class GraphStore(Retriever):
    def __init__(self, config: RAGConfig):
        self.config = config
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
        )

    async def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        with self.driver.session() as session:
            result = session.run(
                """
                CALL db.index.fulltext.queryNodes("nodeContent", $query) 
                YIELD node, score
                RETURN id(node) as id, node.content as content, score
                ORDER BY score DESC
                LIMIT $k
                """,
                query=query,
                k=k
            )
        return [{"id": record["id"], "content": record["content"], "score": record["score"]} for record in result]

    async def graph_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        with self.driver.session() as session:
            result = session.run(
                """
                CALL db.index.fulltext.queryNodes("nodeContent", $query) 
                YIELD node, score
                RETURN id(node) as id, node.content as content, score
                ORDER BY score DESC
                LIMIT $k
                """,
                query=query,
                k=k
            )
        return [{"id": record["id"], "content": record["content"], "score": record["score"]} for record in result]
    async def add_node(self, labels: List[str], properties: Dict[str, Any]) -> int:
        with self.driver.session() as session:
            result = session.write_transaction(self._create_node, labels, properties)
        return result

    @staticmethod
    def _create_node(tx, labels: List[str], properties: Dict[str, Any]) -> int:
        query = f"CREATE (n:{':'.join(labels)}) SET n = $properties RETURN id(n)"
        result = tx.run(query, properties=properties)
        return result.single()[0]

    def close(self):
        self.driver.close()