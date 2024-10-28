"""Graph store implementation for RAG system."""

from typing import List, Dict, Any, Tuple, Optional
import networkx as nx
from datetime import datetime
from neo4j import GraphDatabase
from ..core.base_component import BaseComponent
from ..core.config import UnifiedConfig
from ..core.structures import RetrievalResult
from ..utils.error_handling import log_and_handle_errors, ErrorContext

class GraphStore(BaseComponent):
    """Graph store for knowledge graph operations."""
    
    def __init__(self, config: UnifiedConfig):
        """
        Initialize graph store.
        
        Args:
            config: Configuration instance
        """
        self.config = config
        self.graph = nx.Graph()
        self.driver = None
        self.causal_edges = {}
        self.llm = None
        self.initialized = False
    
    @log_and_handle_errors()
    async def initialize(self) -> None:
        """Initialize graph store components."""
        if not self.initialized:
            # Initialize Neo4j driver
            if hasattr(self.config, 'neo4j_uri'):
                self.driver = GraphDatabase.driver(
                    self.config.neo4j_uri,
                    auth=(self.config.neo4j_user, self.config.neo4j_password)
                )
            
            # Load any saved graph state
            if hasattr(self.config, 'graph_store_path'):
                try:
                    self.graph = nx.read_gpickle(self.config.graph_store_path)
                except:
                    pass  # Use empty graph if file doesn't exist
            
            self.initialized = True
    
    @log_and_handle_errors()
    async def shutdown(self) -> None:
        """Shutdown graph store components."""
        if self.initialized:
            if self.driver:
                self.driver.close()
            
            # Save current graph state
            if hasattr(self.config, 'graph_store_path'):
                nx.write_gpickle(self.graph, self.config.graph_store_path)
            
            self.initialized = False
    
    @log_and_handle_errors()
    async def get_status(self) -> Dict[str, Any]:
        """Get component status."""
        return {
            "initialized": self.initialized,
            "graph_size": len(self.graph),
            "edge_count": self.graph.number_of_edges(),
            "driver_connected": bool(self.driver)
        }
    
    @log_and_handle_errors()
    async def update_config(self, config: Dict[str, Any]) -> None:
        """Update component configuration."""
        self.config = config

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the graph store."""
        for doc in documents:
            self.graph.add_node(doc['id'], **doc)
            # Add edges based on similarity or relationships
            for other_node in self.graph.nodes:
                if other_node != doc['id']:
                    similarity = self._calculate_similarity(doc, self.graph.nodes[other_node])
                    if similarity > self.config.similarity_threshold:
                        self.graph.add_edge(doc['id'], other_node, weight=similarity)

    @log_and_handle_errors()
    async def retrieve(self,
                      query: str,
                      k: int,
                      timestamp: Optional[datetime] = None) -> List[RetrievalResult]:
        """
        Retrieve relevant documents from the graph.
        
        Args:
            query: Query string
            k: Number of results to retrieve
            timestamp: Optional timestamp filter
            
        Returns:
            List of retrieval results
        """
        async with ErrorContext("GraphStore.retrieve"):
            if self.driver:
                with self.driver.session() as session:
                    if timestamp:
                        result = session.run(
                            """
                            CALL db.index.fulltext.queryNodes("nodeContent", $query) 
                            YIELD node, score
                            MATCH (node)-[:VERSION]->(v:NodeVersion)
                            WHERE v.timestamp <= $timestamp
                            WITH node, score, v
                            ORDER BY v.timestamp DESC, score DESC
                            LIMIT $k
                            RETURN id(node) as id, v.content as content, score, 
                                v.uncertainty as uncertainty, v.timestamp as timestamp, 
                                v.version as version
                            """,
                            query=query, timestamp=timestamp, k=k
                        )
                    else:
                        result = session.run(
                            """
                            CALL db.index.fulltext.queryNodes("nodeContent", $query) 
                            YIELD node, score
                            MATCH (node)-[:VERSION]->(v:NodeVersion)
                            WITH node, score, v
                            ORDER BY v.timestamp DESC, score DESC
                            LIMIT $k
                            RETURN id(node) as id, v.content as content, score, 
                                v.uncertainty as uncertainty, v.timestamp as timestamp, 
                                v.version as version
                            """,
                            query=query, k=k
                        )

                    return [
                        RetrievalResult(
                            id=record["id"],
                            content=record["content"],
                            score=record["score"],
                            uncertainty=record["uncertainty"],
                            timestamp=record["timestamp"],
                            version=record["version"]
                        )
                        for record in result
                    ]
            else:
                # Fallback to NetworkX graph if Neo4j is not available
                return await self._retrieve_from_graph(query, k, timestamp)

    async def _retrieve_from_graph(self,
                                 query: str,
                                 k: int,
                                 timestamp: Optional[datetime] = None) -> List[RetrievalResult]:
        """Retrieve from NetworkX graph when Neo4j is not available."""
        results = []
        for node_id, data in self.graph.nodes(data=True):
            if timestamp is None or data.get('timestamp', datetime.max) <= timestamp:
                score = self._calculate_relevance(query, data.get('content', ''))
                if score > 0:
                    results.append(
                        RetrievalResult(
                            id=node_id,
                            content=data.get('content', ''),
                            score=score,
                            timestamp=data.get('timestamp'),
                            version=data.get('version', '1.0')
                        )
                    )
        
        # Sort by score and limit to k results
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]

    def update_causal_strength(self,
                             source: str,
                             target: str,
                             observed_probability: float) -> None:
        """Update causal edge strength."""
        edge = self.causal_edges.get((source, target))
        if edge:
            learning_rate = 0.1
            edge.strength = (1 - learning_rate) * edge.strength + learning_rate * observed_probability

    async def get_snapshot(self, timestamp: datetime) -> Dict[str, Any]:
        """Get graph snapshot at timestamp."""
        snapshot = nx.Graph()
        for node, data in self.graph.nodes(data=True):
            if data.get('timestamp', datetime.max) <= timestamp:
                snapshot.add_node(node, **data)
        
        for u, v, data in self.graph.edges(data=True):
            if snapshot.has_node(u) and snapshot.has_node(v):
                snapshot.add_edge(u, v, **data)
        
        return {
            "nodes": list(snapshot.nodes(data=True)),
            "edges": list(snapshot.edges(data=True))
        }

    @log_and_handle_errors()
    async def beam_search(self,
                         query: str,
                         beam_width: int,
                         max_depth: int) -> List[Tuple[List[str], float]]:
        """
        Perform beam search in the graph.
        
        Args:
            query: Search query
            beam_width: Width of beam
            max_depth: Maximum search depth
            
        Returns:
            List of (path, score) tuples
        """
        initial_entities = await self.get_initial_entities(query)
        beams = [[entity] for entity in initial_entities]

        for _ in range(max_depth):
            candidates = []
            for beam in beams:
                neighbors = await self.get_neighbors(beam[-1])
                for neighbor in neighbors:
                    if neighbor not in beam:  # Avoid cycles
                        new_beam = beam + [neighbor]
                        score = await self.llm.score_path(query, new_beam)
                        candidates.append((new_beam, score))
            
            if not candidates:
                break
            
            beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]

        return beams

    async def get_initial_entities(self, query: str) -> List[str]:
        """Get initial entities for beam search."""
        # Simple keyword matching for now
        entities = []
        query_terms = set(query.lower().split())
        for node, data in self.graph.nodes(data=True):
            content = data.get('content', '').lower()
            if any(term in content for term in query_terms):
                entities.append(node)
        return entities[:10]  # Limit initial entities

    async def get_neighbors(self, entity: str) -> List[str]:
        """Get neighboring entities in graph."""
        if self.driver:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (n)-[r]-(m)
                    WHERE id(n) = $entity
                    RETURN id(m) as neighbor
                    """,
                    entity=entity
                )
                return [record["neighbor"] for record in result]
        else:
            return list(self.graph.neighbors(entity))

    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID."""
        if self.graph.has_node(doc_id):
            return dict(self.graph.nodes[doc_id])
        return None

    def _calculate_similarity(self, doc1: Dict[str, Any], doc2: Dict[str, Any]) -> float:
        """Calculate similarity between documents."""
        # Placeholder implementation
        # In practice, use proper similarity calculation
        common_keys = set(doc1.keys()) & set(doc2.keys())
        if not common_keys:
            return 0.0
        
        similarity = sum(1 for key in common_keys if doc1[key] == doc2[key])
        return similarity / len(common_keys)

    def _calculate_relevance(self, query: str, content: str) -> float:
        """Calculate relevance score."""
        # Placeholder implementation
        # In practice, use proper relevance calculation
        query_terms = set(query.lower().split())
        content_terms = set(content.lower().split())
        overlap = len(query_terms & content_terms)
        return overlap / len(query_terms) if query_terms else 0.0
