"""Cognitive nexus for knowledge integration and evolution."""

from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime
from .base_component import BaseComponent
from ..utils.error_handling import log_and_handle_errors, ErrorContext

class CognitiveNexus(BaseComponent):
    """
    Cognitive nexus that manages knowledge integration and evolution.
    Maintains a knowledge graph and provides query, update, and evolution capabilities.
    """
    
    def __init__(self):
        """Initialize cognitive nexus."""
        self.knowledge_graph = {}
        self.initialized = False
        self.evolution_stats = {
            "updates": 0,
            "evolutions": 0,
            "last_evolution": None
        }
    
    @log_and_handle_errors()
    async def initialize(self) -> None:
        """Initialize nexus components."""
        if not self.initialized:
            # Initialize knowledge graph
            self.knowledge_graph = {
                "nodes": {},
                "edges": {},
                "metadata": {
                    "created": datetime.now().isoformat(),
                    "last_updated": None,
                    "version": "1.0"
                }
            }
            self.initialized = True
    
    @log_and_handle_errors()
    async def shutdown(self) -> None:
        """Shutdown nexus components."""
        self.initialized = False
    
    @log_and_handle_errors()
    async def get_status(self) -> Dict[str, Any]:
        """Get component status."""
        return {
            "initialized": self.initialized,
            "graph_size": {
                "nodes": len(self.knowledge_graph.get("nodes", {})),
                "edges": len(self.knowledge_graph.get("edges", {}))
            },
            "evolution_stats": self.evolution_stats
        }
    
    @log_and_handle_errors()
    async def update_config(self, config: Dict[str, Any]) -> None:
        """Update component configuration."""
        pass  # No configuration needed currently
    
    @log_and_handle_errors()
    async def query(self,
                   content: str,
                   embeddings: List[float],
                   entities: List[str]) -> Dict[str, Any]:
        """
        Query the cognitive nexus.
        
        Args:
            content: Query content
            embeddings: Content embeddings
            entities: Extracted entities
            
        Returns:
            Dictionary containing query results
        """
        async with ErrorContext("CognitiveNexus.query"):
            # Find relevant nodes
            relevant_nodes = await self._find_relevant_nodes(
                content,
                embeddings,
                entities
            )
            
            # Extract context
            context = await self._extract_context(
                relevant_nodes,
                embeddings
            )
            
            # Generate cognitive context
            cognitive_context = await self._generate_cognitive_context(
                content,
                context,
                entities
            )
            
            return {
                "cognitive_context": cognitive_context,
                "relevant_nodes": relevant_nodes,
                "context": context,
                "timestamp": datetime.now().isoformat()
            }
    
    @log_and_handle_errors()
    async def update(self, task: Dict[str, Any], result: Dict[str, Any]) -> None:
        """
        Update knowledge graph based on task results.
        
        Args:
            task: Task information
            result: Task results
        """
        async with ErrorContext("CognitiveNexus.update"):
            # Extract knowledge
            new_knowledge = await self._extract_knowledge(task, result)
            
            # Update graph
            await self._update_graph(new_knowledge)
            
            # Update metadata
            self.knowledge_graph["metadata"]["last_updated"] = datetime.now().isoformat()
            self.evolution_stats["updates"] += 1
    
    @log_and_handle_errors()
    async def evolve(self) -> Dict[str, Any]:
        """
        Evolve knowledge graph structure.
        
        Returns:
            Dictionary containing evolution results
        """
        async with ErrorContext("CognitiveNexus.evolve"):
            # Analyze current state
            analysis = await self._analyze_graph()
            
            # Generate improvements
            improvements = await self._generate_improvements(analysis)
            
            # Apply improvements
            evolution_results = await self._apply_improvements(improvements)
            
            # Update evolution stats
            self.evolution_stats["evolutions"] += 1
            self.evolution_stats["last_evolution"] = datetime.now().isoformat()
            
            return evolution_results
    
    async def _find_relevant_nodes(self,
                                 content: str,
                                 embeddings: List[float],
                                 entities: List[str]) -> List[Dict[str, Any]]:
        """Find relevant nodes in knowledge graph."""
        relevant_nodes = []
        
        # Find by entity match
        for entity in entities:
            if entity in self.knowledge_graph["nodes"]:
                relevant_nodes.append(self.knowledge_graph["nodes"][entity])
        
        # Find by embedding similarity
        if embeddings:
            embedding_matches = self._find_similar_embeddings(embeddings)
            relevant_nodes.extend(embedding_matches)
        
        return relevant_nodes
    
    async def _extract_context(self,
                             nodes: List[Dict[str, Any]],
                             embeddings: List[float]) -> Dict[str, Any]:
        """Extract context from relevant nodes."""
        context = {
            "facts": [],
            "relationships": [],
            "relevance_scores": []
        }
        
        for node in nodes:
            # Extract facts
            context["facts"].extend(node.get("facts", []))
            
            # Extract relationships
            node_id = node["id"]
            if node_id in self.knowledge_graph["edges"]:
                context["relationships"].extend(
                    self.knowledge_graph["edges"][node_id]
                )
            
            # Calculate relevance
            if embeddings and "embedding" in node:
                relevance = self._calculate_similarity(
                    embeddings,
                    node["embedding"]
                )
                context["relevance_scores"].append(relevance)
        
        return context
    
    async def _generate_cognitive_context(self,
                                       content: str,
                                       context: Dict[str, Any],
                                       entities: List[str]) -> Dict[str, Any]:
        """Generate cognitive context from extracted information."""
        return {
            "query": content,
            "entities": entities,
            "relevant_facts": context["facts"][:5],
            "key_relationships": context["relationships"][:5],
            "context_summary": self._generate_context_summary(context)
        }
    
    async def _extract_knowledge(self,
                               task: Dict[str, Any],
                               result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract knowledge from task results."""
        return {
            "task_type": task.get("type"),
            "entities": task.get("entities", []),
            "facts": result.get("facts", []),
            "relationships": result.get("relationships", []),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _update_graph(self, knowledge: Dict[str, Any]) -> None:
        """Update knowledge graph with new knowledge."""
        # Add new nodes
        for entity in knowledge["entities"]:
            if entity not in self.knowledge_graph["nodes"]:
                self.knowledge_graph["nodes"][entity] = {
                    "id": entity,
                    "facts": [],
                    "created": datetime.now().isoformat()
                }
            self.knowledge_graph["nodes"][entity]["facts"].extend(
                knowledge["facts"]
            )
        
        # Add new edges
        for rel in knowledge["relationships"]:
            source = rel.get("source")
            target = rel.get("target")
            if source and target:
                if source not in self.knowledge_graph["edges"]:
                    self.knowledge_graph["edges"][source] = []
                self.knowledge_graph["edges"][source].append(rel)
    
    async def _analyze_graph(self) -> Dict[str, Any]:
        """Analyze current graph state."""
        return {
            "node_count": len(self.knowledge_graph["nodes"]),
            "edge_count": len(self.knowledge_graph["edges"]),
            "density": self._calculate_graph_density(),
            "clustering": self._calculate_clustering(),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _generate_improvements(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate potential improvements."""
        improvements = []
        
        # Check density
        if analysis["density"] < 0.1:
            improvements.append({
                "type": "density",
                "action": "add_connections",
                "priority": "high"
            })
        
        # Check clustering
        if analysis["clustering"] < 0.3:
            improvements.append({
                "type": "clustering",
                "action": "merge_clusters",
                "priority": "medium"
            })
        
        return improvements
    
    async def _apply_improvements(self, improvements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply generated improvements."""
        results = {
            "applied_improvements": [],
            "failed_improvements": [],
            "timestamp": datetime.now().isoformat()
        }
        
        for improvement in improvements:
            try:
                if improvement["type"] == "density":
                    await self._improve_density()
                elif improvement["type"] == "clustering":
                    await self._improve_clustering()
                results["applied_improvements"].append(improvement)
            except Exception as e:
                results["failed_improvements"].append({
                    "improvement": improvement,
                    "error": str(e)
                })
        
        return results
    
    def _find_similar_embeddings(self, embeddings: List[float]) -> List[Dict[str, Any]]:
        """Find nodes with similar embeddings."""
        similar_nodes = []
        for node in self.knowledge_graph["nodes"].values():
            if "embedding" in node:
                similarity = self._calculate_similarity(
                    embeddings,
                    node["embedding"]
                )
                if similarity > 0.7:  # Threshold
                    similar_nodes.append(node)
        return similar_nodes
    
    def _calculate_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Calculate cosine similarity between embeddings."""
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
    
    def _generate_context_summary(self, context: Dict[str, Any]) -> str:
        """Generate summary of context information."""
        facts = context.get("facts", [])[:3]
        relationships = context.get("relationships", [])[:2]
        
        summary_parts = []
        if facts:
            summary_parts.append("Key facts: " + "; ".join(facts))
        if relationships:
            summary_parts.append("Key relationships: " + "; ".join(
                [f"{r['source']} -> {r['target']}" for r in relationships]
            ))
        
        return " | ".join(summary_parts)
    
    def _calculate_graph_density(self) -> float:
        """Calculate graph density."""
        nodes = len(self.knowledge_graph["nodes"])
        edges = sum(len(edges) for edges in self.knowledge_graph["edges"].values())
        if nodes <= 1:
            return 0.0
        return (2.0 * edges) / (nodes * (nodes - 1))
    
    def _calculate_clustering(self) -> float:
        """Calculate clustering coefficient."""
        # Placeholder implementation
        return 0.5
    
    async def _improve_density(self) -> None:
        """Improve graph density."""
        # Placeholder implementation
        pass
    
    async def _improve_clustering(self) -> None:
        """Improve graph clustering."""
        # Placeholder implementation
        pass
