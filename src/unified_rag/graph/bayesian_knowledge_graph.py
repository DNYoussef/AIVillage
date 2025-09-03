"""
Bayesian Knowledge Graph RAG System
Implements probabilistic reasoning over knowledge graphs with uncertainty quantification
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import json
import uuid
import numpy as np
import networkx as nx
from pathlib import Path

logger = logging.getLogger(__name__)


class RelationType(Enum):
    """Types of relationships in the knowledge graph."""

    SEMANTIC = "semantic"
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    HIERARCHICAL = "hierarchical"
    ASSOCIATIVE = "associative"
    CONTRADICTION = "contradiction"
    EVIDENCE = "evidence"
    INFERENCE = "inference"


class TrustLevel(Enum):
    """Trust levels for information in the graph."""

    VERY_HIGH = 0.9
    HIGH = 0.8
    MEDIUM = 0.6
    LOW = 0.4
    VERY_LOW = 0.2
    UNKNOWN = 0.5


@dataclass
class GraphNode:
    """Node in the Bayesian trust graph."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    concept: str = ""
    trust_score: float = 0.5
    confidence: float = 0.5
    belief_strength: float = 0.5
    evidence_count: int = 0
    source_reliability: float = 0.5
    verification_status: str = "unverified"
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    temporal_decay: float = 0.95
    incoming_edges: set[str] = field(default_factory=set)
    outgoing_edges: set[str] = field(default_factory=set)
    embedding: Optional[np.ndarray] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """Edge representing a relationship between nodes."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""
    relation_type: RelationType = RelationType.ASSOCIATIVE
    relation_strength: float = 0.5
    trust_score: float = 0.5
    evidence_count: int = 0
    supporting_docs: List[str] = field(default_factory=list)
    conditional_probability: float = 0.5
    mutual_information: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def calculate_trust_propagation(self, source_trust: float) -> float:
        """Calculate trust propagation through this edge."""
        propagated_trust = source_trust * self.trust_score * self.relation_strength
        propagated_trust *= self.conditional_probability
        return min(1.0, max(0.0, propagated_trust))


@dataclass
class BayesianQueryResult:
    """Result from Bayesian graph query with probabilistic reasoning."""

    nodes: List[GraphNode] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)
    trust_scores: Dict[str, float] = field(default_factory=dict)
    belief_network: Dict[str, Dict[str, float]] = field(default_factory=dict)
    query_confidence: float = 0.0
    reasoning_path: List[str] = field(default_factory=list)
    query_time_ms: float = 0.0


class RelationshipType(Enum):
    """Types of relationships (alias for RelationType)."""

    SEMANTIC = "semantic"
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    HIERARCHICAL = "hierarchical"
    ASSOCIATIVE = "associative"
    CONTRADICTION = "contradiction"
    EVIDENCE = "evidence"
    INFERENCE = "inference"
    RELATES_TO = "relates_to"
    ENABLES = "enables"
    DEPENDS_ON = "depends_on"
    CONTAINS = "contains"


@dataclass
class Relationship:
    """Relationship between graph nodes."""

    subject_id: str
    predicate: RelationshipType
    object_id: str
    confidence: float = 0.5
    trust_score: float = 0.5
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def create_graph_node(
    content: str,
    node_id: Optional[str] = None,
    concept: Optional[str] = None,
    concepts: Optional[List[str]] = None,
    trust_score: float = 0.5,
    confidence: float = 0.5,
    metadata: Optional[Dict[str, Any]] = None,
) -> GraphNode:
    """Create a new graph node with the given parameters."""

    node = GraphNode(
        id=node_id or str(uuid.uuid4()),
        content=content,
        concept=concept or (concepts[0] if concepts else ""),
        trust_score=trust_score,
        confidence=confidence,
        metadata=metadata or {},
    )

    if concepts:
        node.tags.extend(concepts)

    return node
class EdgeType(Enum):
    """Types of edges in the knowledge graph"""
    CAUSAL = "causal"
    SIMILARITY = "similarity" 
    TEMPORAL = "temporal"
    HIERARCHICAL = "hierarchical"
    ASSOCIATIVE = "associative"

@dataclass
class KnowledgeNode:
    """Node in the Bayesian knowledge graph"""
    id: str
    content: str
    node_type: str
    confidence: float = 0.5  # Prior confidence
    evidence_count: int = 0
    last_updated: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_confidence(self, new_evidence: float, evidence_weight: float = 1.0):
        """Bayesian update of node confidence"""
        # Simple Bayesian update
        prior = self.confidence
        likelihood = new_evidence
        
        # Weighted combination
        posterior = (prior + likelihood * evidence_weight) / (1 + evidence_weight)
        self.confidence = max(0.0, min(1.0, posterior))
        self.evidence_count += 1

@dataclass
class KnowledgeEdge:
    """Edge in the Bayesian knowledge graph"""
    source: str
    target: str
    edge_type: EdgeType
    confidence: float = 0.5
    strength: float = 1.0
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BayesianQuery:
    """Query for Bayesian knowledge graph"""
    query_text: str
    query_type: str = "inference"
    evidence_nodes: List[str] = field(default_factory=list)
    target_nodes: List[str] = field(default_factory=list)
    confidence_threshold: float = 0.3
    max_hops: int = 3

@dataclass
class InferenceResult:
    """Result from Bayesian inference"""
    query: BayesianQuery
    inferred_nodes: List[KnowledgeNode]
    inference_path: List[str]
    confidence_score: float
    uncertainty_estimate: float
    evidence_strength: float

class BayesianKnowledgeGraphRAG:
    """
    RAG system with Bayesian knowledge graph reasoning
    Supports probabilistic inference and uncertainty quantification
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/bayesian_kg")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize graph structure
        self.graph = nx.MultiDiGraph()
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: Dict[str, KnowledgeEdge] = {}
        
        # Bayesian parameters
        self.confidence_decay = 0.95  # Decay factor for temporal updates
        self.evidence_threshold = 0.3  # Minimum evidence strength
        self.inference_depth = 3       # Maximum inference depth
        
        # Load existing graph
        self._load_graph()
    
    def _load_graph(self):
        """Load knowledge graph from storage"""
        try:
            nodes_file = self.storage_path / "nodes.json"
            edges_file = self.storage_path / "edges.json"
            
            if nodes_file.exists():
                with open(nodes_file, 'r', encoding='utf-8') as f:
                    nodes_data = json.load(f)
                    for node_data in nodes_data:
                        node = KnowledgeNode(**node_data)
                        self.nodes[node.id] = node
                        self.graph.add_node(node.id, **node_data)
            
            if edges_file.exists():
                with open(edges_file, 'r', encoding='utf-8') as f:
                    edges_data = json.load(f)
                    for edge_data in edges_data:
                        edge = KnowledgeEdge(
                            source=edge_data['source'],
                            target=edge_data['target'],
                            edge_type=EdgeType(edge_data['edge_type']),
                            confidence=edge_data['confidence'],
                            strength=edge_data['strength'],
                            evidence=edge_data.get('evidence', []),
                            metadata=edge_data.get('metadata', {})
                        )
                        edge_id = f"{edge.source}->{edge.target}"
                        self.edges[edge_id] = edge
                        self.graph.add_edge(
                            edge.source, edge.target,
                            edge_type=edge.edge_type.value,
                            confidence=edge.confidence,
                            strength=edge.strength
                        )
            
            logger.info(f"Loaded {len(self.nodes)} nodes and {len(self.edges)} edges")
            
        except Exception as e:
            logger.warning(f"Failed to load knowledge graph: {e}")
    
    def _save_graph(self):
        """Save knowledge graph to storage"""
        try:
            # Save nodes
            nodes_data = []
            for node in self.nodes.values():
                node_dict = {
                    'id': node.id,
                    'content': node.content,
                    'node_type': node.node_type,
                    'confidence': node.confidence,
                    'evidence_count': node.evidence_count,
                    'last_updated': node.last_updated,
                    'metadata': node.metadata
                }
                nodes_data.append(node_dict)
            
            nodes_file = self.storage_path / "nodes.json"
            with open(nodes_file, 'w', encoding='utf-8') as f:
                json.dump(nodes_data, f, indent=2, ensure_ascii=False)
            
            # Save edges
            edges_data = []
            for edge in self.edges.values():
                edge_dict = {
                    'source': edge.source,
                    'target': edge.target,
                    'edge_type': edge.edge_type.value,
                    'confidence': edge.confidence,
                    'strength': edge.strength,
                    'evidence': edge.evidence,
                    'metadata': edge.metadata
                }
                edges_data.append(edge_dict)
            
            edges_file = self.storage_path / "edges.json"
            with open(edges_file, 'w', encoding='utf-8') as f:
                json.dump(edges_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to save knowledge graph: {e}")
    
    async def add_knowledge_node(self,
                               content: str,
                               node_type: str,
                               confidence: float = 0.5,
                               metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add new knowledge node to the graph"""
        node_id = f"node_{len(self.nodes)}"
        
        node = KnowledgeNode(
            id=node_id,
            content=content,
            node_type=node_type,
            confidence=confidence,
            metadata=metadata or {}
        )
        
        self.nodes[node_id] = node
        self.graph.add_node(node_id, **{
            'content': content,
            'node_type': node_type,
            'confidence': confidence
        })
        
        self._save_graph()
        return node_id
    
    async def add_knowledge_edge(self,
                               source_id: str,
                               target_id: str,
                               edge_type: EdgeType,
                               confidence: float = 0.5,
                               strength: float = 1.0,
                               evidence: Optional[List[str]] = None) -> str:
        """Add edge between knowledge nodes"""
        if source_id not in self.nodes or target_id not in self.nodes:
            raise ValueError("Both source and target nodes must exist")
        
        edge = KnowledgeEdge(
            source=source_id,
            target=target_id,
            edge_type=edge_type,
            confidence=confidence,
            strength=strength,
            evidence=evidence or []
        )
        
        edge_id = f"{source_id}->{target_id}"
        self.edges[edge_id] = edge
        
        self.graph.add_edge(
            source_id, target_id,
            edge_type=edge_type.value,
            confidence=confidence,
            strength=strength
        )
        
        self._save_graph()
        return edge_id
    
    async def bayesian_inference(self, query: BayesianQuery) -> InferenceResult:
        """
        Perform Bayesian inference over the knowledge graph
        """
        try:
            # Find relevant nodes
            candidate_nodes = await self._find_relevant_nodes(query.query_text)
            
            # Perform inference
            inferred_nodes = []
            inference_paths = []
            total_confidence = 0.0
            
            for node_id in candidate_nodes:
                if node_id in self.nodes:
                    node = self.nodes[node_id]
                    
                    # Calculate inference confidence
                    path_confidence = await self._calculate_path_confidence(
                        node_id, query.evidence_nodes, query.max_hops
                    )
                    
                    if path_confidence >= query.confidence_threshold:
                        inferred_nodes.append(node)
                        inference_paths.append([node_id])  # Simplified path
                        total_confidence += path_confidence
            
            # Calculate overall confidence and uncertainty
            if inferred_nodes:
                confidence_score = total_confidence / len(inferred_nodes)
                uncertainty_estimate = 1.0 - confidence_score
            else:
                confidence_score = 0.0
                uncertainty_estimate = 1.0
            
            return InferenceResult(
                query=query,
                inferred_nodes=inferred_nodes,
                inference_path=inference_paths[0] if inference_paths else [],
                confidence_score=confidence_score,
                uncertainty_estimate=uncertainty_estimate,
                evidence_strength=confidence_score  # Simplified
            )
            
        except Exception as e:
            logger.error(f"Error in Bayesian inference: {e}")
            return InferenceResult(
                query=query,
                inferred_nodes=[],
                inference_path=[],
                confidence_score=0.0,
                uncertainty_estimate=1.0,
                evidence_strength=0.0
            )
    
    async def _find_relevant_nodes(self, query_text: str) -> List[str]:
        """Find nodes relevant to query text"""
        relevant_nodes = []
        query_lower = query_text.lower()
        
        for node_id, node in self.nodes.items():
            if query_lower in node.content.lower():
                relevant_nodes.append(node_id)
            elif any(query_lower in tag.lower() for tag in node.metadata.get('tags', [])):
                relevant_nodes.append(node_id)
        
        return relevant_nodes
    
    async def _calculate_path_confidence(self,
                                       target_node: str,
                                       evidence_nodes: List[str],
                                       max_hops: int) -> float:
        """Calculate confidence of inference path"""
        if not evidence_nodes or target_node in evidence_nodes:
            return self.nodes[target_node].confidence if target_node in self.nodes else 0.0
        
        max_confidence = 0.0
        
        for evidence_node in evidence_nodes:
            if evidence_node in self.nodes:
                try:
                    # Find shortest path
                    if nx.has_path(self.graph, evidence_node, target_node):
                        path = nx.shortest_path(self.graph, evidence_node, target_node)
                        
                        if len(path) <= max_hops + 1:
                            # Calculate path confidence
                            path_confidence = 1.0
                            for i in range(len(path) - 1):
                                edge_data = self.graph.get_edge_data(path[i], path[i + 1])
                                if edge_data:
                                    # Take first edge if multiple exist
                                    edge_confidence = list(edge_data.values())[0].get('confidence', 0.5)
                                    path_confidence *= edge_confidence
                            
                            # Apply distance decay
                            distance_decay = (self.confidence_decay ** (len(path) - 1))
                            final_confidence = path_confidence * distance_decay
                            
                            max_confidence = max(max_confidence, final_confidence)
                            
                except nx.NetworkXNoPath:
                    continue
        
        return max_confidence
    
    async def update_node_evidence(self,
                                 node_id: str,
                                 evidence_strength: float,
                                 evidence_source: str):
        """Update node confidence based on new evidence"""
        if node_id not in self.nodes:
            return
        
        node = self.nodes[node_id]
        node.update_confidence(evidence_strength)
        
        # Update graph node data
        self.graph.nodes[node_id]['confidence'] = node.confidence
        
        logger.debug(f"Updated node {node_id} confidence to {node.confidence}")
        self._save_graph()
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph"""
        node_types = {}
        edge_types = {}
        
        for node in self.nodes.values():
            node_types[node.node_type] = node_types.get(node.node_type, 0) + 1
        
        for edge in self.edges.values():
            edge_types[edge.edge_type.value] = edge_types.get(edge.edge_type.value, 0) + 1
        
        return {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'node_types': node_types,
            'edge_types': edge_types,
            'avg_node_confidence': np.mean([n.confidence for n in self.nodes.values()]) if self.nodes else 0,
            'avg_edge_confidence': np.mean([e.confidence for e in self.edges.values()]) if self.edges else 0,
            'graph_density': nx.density(self.graph),
            'storage_path': str(self.storage_path)
        }


# Backwards compatibility alias
BayesianTrustGraph = BayesianKnowledgeGraphRAG