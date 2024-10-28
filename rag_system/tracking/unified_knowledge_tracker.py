"""Unified knowledge tracking system."""

from typing import Dict, Any, List, Optional, Set
from datetime import datetime
import networkx as nx
from ..core.base_component import BaseComponent
from ..core.config import UnifiedConfig
from ..utils.embedding import get_embedding
from ..utils.named_entity_recognition import extract_entities, extract_relations
from ..utils.error_handling import log_and_handle_errors, ErrorContext

class UnifiedKnowledgeTracker(BaseComponent):
    """
    Unified system for tracking knowledge and insights across the RAG system.
    Implements knowledge graph construction, entity tracking, and insight generation.
    """
    
    def __init__(self, config: UnifiedConfig):
        """
        Initialize knowledge tracker.
        
        Args:
            config: Configuration instance
        """
        self.config = config
        self.knowledge_graph = nx.Graph()
        self.entity_embeddings = {}
        self.relation_types = set()
        self.tracked_entities = set()
        self.initialized = False
        self.tracking_stats = {
            "total_entities": 0,
            "total_relations": 0,
            "total_insights": 0,
            "avg_confidence": 0.0
        }
    
    @log_and_handle_errors()
    async def initialize(self) -> None:
        """Initialize knowledge tracker."""
        if not self.initialized:
            # Load any saved state
            if hasattr(self.config, 'knowledge_state_path'):
                try:
                    self._load_state(self.config.knowledge_state_path)
                except:
                    pass  # Use default state if file doesn't exist
            self.initialized = True
    
    @log_and_handle_errors()
    async def shutdown(self) -> None:
        """Shutdown knowledge tracker."""
        if self.initialized:
            # Save current state
            if hasattr(self.config, 'knowledge_state_path'):
                self._save_state(self.config.knowledge_state_path)
            self.initialized = False
    
    @log_and_handle_errors()
    async def get_status(self) -> Dict[str, Any]:
        """Get component status."""
        return {
            "initialized": self.initialized,
            "graph_size": len(self.knowledge_graph),
            "tracked_entities": len(self.tracked_entities),
            "relation_types": len(self.relation_types),
            "tracking_stats": self.tracking_stats
        }
    
    @log_and_handle_errors()
    async def update_config(self, config: Dict[str, Any]) -> None:
        """Update component configuration."""
        self.config = config

    async def process_text(self, text: str, source: str = "unknown") -> Dict[str, Any]:
        """
        Process text to extract and track knowledge.
        
        Args:
            text: Input text
            source: Source of the text
            
        Returns:
            Dictionary containing extracted knowledge
        """
        async with ErrorContext("KnowledgeTracker"):
            # Extract entities and relations
            entities = extract_entities(text)
            relations = extract_relations(text)
            
            # Update knowledge graph
            await self._update_graph(entities, relations, source)
            
            # Generate insights
            insights = await self._generate_insights(entities, relations)
            
            # Update statistics
            self._update_stats(entities, relations, insights)
            
            return {
                "entities": entities,
                "relations": relations,
                "insights": insights,
                "source": source,
                "timestamp": datetime.now().isoformat()
            }

    async def _update_graph(self,
                          entities: List[Dict[str, Any]],
                          relations: List[Dict[str, Any]],
                          source: str) -> None:
        """Update knowledge graph with new information."""
        # Add entities
        for entity in entities:
            if entity["text"] not in self.tracked_entities:
                # Get embedding for new entity
                embedding = get_embedding(entity["text"])
                self.entity_embeddings[entity["text"]] = embedding
                self.tracked_entities.add(entity["text"])
            
            # Add or update node
            self.knowledge_graph.add_node(
                entity["text"],
                label=entity["label"],
                description=entity["description"],
                sources=set([source])
            )
        
        # Add relations
        for relation in relations:
            self.relation_types.add(relation["relation_type"])
            
            if relation["subject"] in self.tracked_entities and \
               relation["object"] in self.tracked_entities:
                self.knowledge_graph.add_edge(
                    relation["subject"],
                    relation["object"],
                    relation_type=relation["relation_type"],
                    predicate=relation["predicate"],
                    confidence=relation["confidence"],
                    sources=set([source])
                )

    async def _generate_insights(self,
                               entities: List[Dict[str, Any]],
                               relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate insights from new information."""
        insights = []
        
        # Entity-based insights
        entity_clusters = self._cluster_entities(entities)
        for cluster in entity_clusters:
            if len(cluster) > 1:
                insights.append({
                    "type": "entity_cluster",
                    "entities": cluster,
                    "description": f"Found related entities: {', '.join(cluster)}",
                    "confidence": 0.8
                })
        
        # Relation-based insights
        relation_patterns = self._find_relation_patterns(relations)
        for pattern in relation_patterns:
            insights.append({
                "type": "relation_pattern",
                "pattern": pattern["pattern"],
                "instances": pattern["instances"],
                "description": f"Found repeated relation pattern: {pattern['pattern']}",
                "confidence": pattern["confidence"]
            })
        
        return insights

    def _cluster_entities(self, entities: List[Dict[str, Any]]) -> List[List[str]]:
        """Cluster related entities based on embeddings."""
        clusters = []
        processed = set()
        
        for entity in entities:
            if entity["text"] in processed:
                continue
            
            cluster = [entity["text"]]
            entity_embedding = self.entity_embeddings.get(entity["text"])
            
            if entity_embedding:
                # Find similar entities
                for other in entities:
                    if other["text"] != entity["text"] and \
                       other["text"] not in processed:
                        other_embedding = self.entity_embeddings.get(other["text"])
                        if other_embedding:
                            similarity = self._calculate_similarity(
                                entity_embedding,
                                other_embedding
                            )
                            if similarity > self.config.similarity_threshold:
                                cluster.append(other["text"])
                                processed.add(other["text"])
            
            if len(cluster) > 1:
                clusters.append(cluster)
            processed.add(entity["text"])
        
        return clusters

    def _find_relation_patterns(self,
                              relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find recurring patterns in relations."""
        patterns = {}
        
        for relation in relations:
            pattern = (
                relation["relation_type"],
                relation["predicate"]
            )
            
            if pattern not in patterns:
                patterns[pattern] = {
                    "pattern": f"{pattern[0]} ({pattern[1]})",
                    "instances": [],
                    "count": 0,
                    "confidence": 0.0
                }
            
            patterns[pattern]["instances"].append({
                "subject": relation["subject"],
                "object": relation["object"]
            })
            patterns[pattern]["count"] += 1
            patterns[pattern]["confidence"] = min(
                0.5 + (patterns[pattern]["count"] * 0.1),
                1.0
            )
        
        return [
            pattern for pattern in patterns.values()
            if pattern["count"] > 1
        ]

    def _calculate_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Calculate cosine similarity between embeddings."""
        import numpy as np
        emb1_np = np.array(emb1)
        emb2_np = np.array(emb2)
        return float(
            np.dot(emb1_np, emb2_np) /
            (np.linalg.norm(emb1_np) * np.linalg.norm(emb2_np))
        )

    def _update_stats(self,
                     entities: List[Dict[str, Any]],
                     relations: List[Dict[str, Any]],
                     insights: List[Dict[str, Any]]) -> None:
        """Update tracking statistics."""
        self.tracking_stats["total_entities"] += len(entities)
        self.tracking_stats["total_relations"] += len(relations)
        self.tracking_stats["total_insights"] += len(insights)
        
        # Update average confidence
        if insights:
            confidences = [insight["confidence"] for insight in insights]
            current_avg = self.tracking_stats["avg_confidence"]
            total_insights = self.tracking_stats["total_insights"]
            
            if total_insights > 1:
                self.tracking_stats["avg_confidence"] = (
                    (current_avg * (total_insights - len(insights)) +
                     sum(confidences)) / total_insights
                )
            else:
                self.tracking_stats["avg_confidence"] = sum(confidences) / len(confidences)

    def _save_state(self, path: str) -> None:
        """Save current state to file."""
        import pickle
        state = {
            "graph": self.knowledge_graph,
            "embeddings": self.entity_embeddings,
            "relations": self.relation_types,
            "entities": self.tracked_entities,
            "stats": self.tracking_stats
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def _load_state(self, path: str) -> None:
        """Load state from file."""
        import pickle
        with open(path, 'rb') as f:
            state = pickle.load(f)
            self.knowledge_graph = state["graph"]
            self.entity_embeddings = state["embeddings"]
            self.relation_types = state["relations"]
            self.tracked_entities = state["entities"]
            self.tracking_stats = state["stats"]

    async def get_entity_knowledge(self, entity: str) -> Optional[Dict[str, Any]]:
        """
        Get all knowledge about an entity.
        
        Args:
            entity: Entity to query
            
        Returns:
            Dictionary containing entity knowledge
        """
        if entity not in self.tracked_entities:
            return None
        
        # Get node data
        node_data = dict(self.knowledge_graph.nodes[entity])
        
        # Get relations
        relations = []
        for neighbor in self.knowledge_graph.neighbors(entity):
            edge_data = self.knowledge_graph.edges[entity, neighbor]
            relations.append({
                "target": neighbor,
                "type": edge_data["relation_type"],
                "predicate": edge_data["predicate"],
                "confidence": edge_data["confidence"]
            })
        
        return {
            "entity": entity,
            "metadata": node_data,
            "relations": relations,
            "embedding": self.entity_embeddings.get(entity)
        }

    async def find_paths(self,
                        start_entity: str,
                        end_entity: str,
                        max_length: int = 3) -> List[List[Dict[str, Any]]]:
        """
        Find paths between entities in knowledge graph.
        
        Args:
            start_entity: Starting entity
            end_entity: Ending entity
            max_length: Maximum path length
            
        Returns:
            List of paths, where each path is a list of edges
        """
        if not (start_entity in self.tracked_entities and \
                end_entity in self.tracked_entities):
            return []
        
        paths = []
        for path in nx.all_simple_paths(
            self.knowledge_graph,
            start_entity,
            end_entity,
            cutoff=max_length
        ):
            path_edges = []
            for i in range(len(path) - 1):
                edge_data = dict(self.knowledge_graph.edges[path[i], path[i+1]])
                path_edges.append({
                    "source": path[i],
                    "target": path[i+1],
                    "type": edge_data["relation_type"],
                    "predicate": edge_data["predicate"],
                    "confidence": edge_data["confidence"]
                })
            paths.append(path_edges)
        
        return sorted(
            paths,
            key=lambda p: sum(edge["confidence"] for edge in p),
            reverse=True
        )

    async def get_similar_entities(self,
                                 entity: str,
                                 threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Find entities similar to given entity.
        
        Args:
            entity: Entity to compare
            threshold: Similarity threshold
            
        Returns:
            List of similar entities with similarity scores
        """
        if entity not in self.tracked_entities:
            return []
        
        entity_embedding = self.entity_embeddings[entity]
        similar_entities = []
        
        for other in self.tracked_entities:
            if other != entity:
                other_embedding = self.entity_embeddings[other]
                similarity = self._calculate_similarity(
                    entity_embedding,
                    other_embedding
                )
                if similarity > threshold:
                    similar_entities.append({
                        "entity": other,
                        "similarity": similarity
                    })
        
        return sorted(
            similar_entities,
            key=lambda x: x["similarity"],
            reverse=True
        )
