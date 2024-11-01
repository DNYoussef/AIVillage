"""Graph store implementation for RAG system."""

from typing import List, Dict, Any, Tuple, Optional
import networkx as nx
from datetime import datetime
import logging
import os
from neo4j import GraphDatabase
from ..core.base_component import BaseComponent
from ..core.config import UnifiedConfig
from ..core.structures import RetrievalResult
from ..utils.error_handling import log_and_handle_errors, ErrorContext

logger = logging.getLogger(__name__)

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
        self.stats = {
            "total_queries": 0,
            "total_nodes": 0,
            "total_edges": 0,
            "average_query_time": 0.0,
            "neo4j_connected": False
        }
        logger.info("Initialized GraphStore")
    
    @log_and_handle_errors()
    async def initialize(self) -> None:
        """Initialize graph store components."""
        try:
            logger.info("Initializing GraphStore...")
            
            if not self.initialized:
                # Initialize Neo4j driver
                if hasattr(self.config, 'neo4j_uri'):
                    try:
                        self.driver = GraphDatabase.driver(
                            self.config.neo4j_uri,
                            auth=(self.config.neo4j_user, self.config.neo4j_password)
                        )
                        # Test connection
                        with self.driver.session() as session:
                            session.run("RETURN 1")
                        self.stats["neo4j_connected"] = True
                        logger.info("Successfully connected to Neo4j")
                    except Exception as e:
                        logger.error(f"Error connecting to Neo4j: {str(e)}")
                        self.driver = None
                
                # Load any saved graph state
                if hasattr(self.config, 'graph_store_path'):
                    try:
                        graph_path = self.config.graph_store_path
                        if os.path.exists(graph_path):
                            self.graph = nx.read_gpickle(graph_path)
                            logger.info(f"Loaded graph from {graph_path}")
                        else:
                            logger.info("No existing graph found, starting with empty graph")
                    except Exception as e:
                        logger.error(f"Error loading graph: {str(e)}")
                        self.graph = nx.Graph()
                
                # Initialize stats
                self.stats.update({
                    "total_queries": 0,
                    "total_nodes": len(self.graph),
                    "total_edges": self.graph.number_of_edges(),
                    "average_query_time": 0.0,
                    "last_save": None,
                    "last_maintenance": None
                })
                
                self.initialized = True
                logger.info(f"Successfully initialized GraphStore with {len(self.graph)} nodes and {self.graph.number_of_edges()} edges")
            else:
                logger.warning("GraphStore already initialized")
            
        except Exception as e:
            logger.error(f"Error initializing GraphStore: {str(e)}")
            self.initialized = False
            raise
    
    @log_and_handle_errors()
    async def shutdown(self) -> None:
        """Shutdown graph store components."""
        try:
            logger.info("Shutting down GraphStore...")
            
            if self.initialized:
                # Close Neo4j driver
                if self.driver:
                    try:
                        self.driver.close()
                        logger.info("Closed Neo4j connection")
                    except Exception as e:
                        logger.error(f"Error closing Neo4j connection: {str(e)}")
                
                # Save current graph state
                if hasattr(self.config, 'graph_store_path'):
                    try:
                        graph_path = self.config.graph_store_path
                        os.makedirs(os.path.dirname(graph_path), exist_ok=True)
                        nx.write_gpickle(self.graph, graph_path)
                        logger.info(f"Saved graph to {graph_path}")
                    except Exception as e:
                        logger.error(f"Error saving graph: {str(e)}")
                
                # Log final stats
                logger.info(f"Final stats: {self.stats}")
                
                # Clear state
                self.graph.clear()
                self.causal_edges.clear()
                self.stats.clear()
                self.driver = None
                
                self.initialized = False
                logger.info("Successfully shut down GraphStore")
            else:
                logger.warning("GraphStore not initialized")
            
        except Exception as e:
            logger.error(f"Error shutting down GraphStore: {str(e)}")
            raise
    
    @log_and_handle_errors()
    async def get_status(self) -> Dict[str, Any]:
        """Get component status."""
        return {
            "initialized": self.initialized,
            "graph_size": len(self.graph),
            "edge_count": self.graph.number_of_edges(),
            "neo4j_connected": bool(self.driver),
            "stats": self.stats,
            "memory_usage": {
                "nodes": len(self.graph),
                "edges": self.graph.number_of_edges(),
                "causal_edges": len(self.causal_edges)
            }
        }
    
    @log_and_handle_errors()
    async def update_config(self, config: Dict[str, Any]) -> None:
        """Update component configuration."""
        self.config = config

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the graph store."""
        for doc in documents:
            self.graph.add_node(doc['id'], **doc)
            self.stats["total_nodes"] += 1
            
            # Add edges based on similarity or relationships
            for other_node in self.graph.nodes:
                if other_node != doc['id']:
                    similarity = self._calculate_similarity(doc, self.graph.nodes[other_node])
                    if similarity > self.config.similarity_threshold:
                        self.graph.add_edge(doc['id'], other_node, weight=similarity)
                        self.stats["total_edges"] += 1

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
        if not self.initialized:
            raise RuntimeError("GraphStore not initialized")
        
        async with ErrorContext("GraphStore.retrieve"):
            start_time = datetime.now()
            
            try:
                if self.driver:
                    results = await self._retrieve_from_neo4j(query, k, timestamp)
                else:
                    results = await self._retrieve_from_graph(query, k, timestamp)
                
                # Update stats
                query_time = (datetime.now() - start_time).total_seconds()
                self.stats["total_queries"] += 1
                self.stats["average_query_time"] = (
                    (self.stats["average_query_time"] * (self.stats["total_queries"] - 1) + query_time) /
                    self.stats["total_queries"]
                )
                
                return results
                
            except Exception as e:
                logger.error(f"Error during retrieval: {str(e)}")
                raise

    async def _retrieve_from_neo4j(self,
                                 query: str,
                                 k: int,
                                 timestamp: Optional[datetime]) -> List[RetrievalResult]:
        """Retrieve from Neo4j database."""
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
        if not self.initialized:
            raise RuntimeError("GraphStore not initialized")
        
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
