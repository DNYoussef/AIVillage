"""Cognitive nexus for knowledge integration and evolution."""

from typing import Dict, Any, List, Optional
import numpy as np
import logging
from datetime import datetime
from .base_component import BaseComponent
from ..utils.error_handling import log_and_handle_errors, ErrorContext, RAGSystemError

logger = logging.getLogger(__name__)

class CognitiveNexus(BaseComponent):
    """
    Cognitive nexus that manages knowledge integration and evolution.
    Maintains a knowledge graph and provides query, update, and evolution capabilities.
    """
    
    def __init__(self):
        """Initialize cognitive nexus."""
        super().__init__()  # Call parent's init
        self.knowledge_graph = {}
        
        # Add component-specific stats
        self.stats.update({
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "total_updates": 0,
            "successful_updates": 0,
            "failed_updates": 0,
            "total_evolutions": 0,
            "successful_evolutions": 0,
            "failed_evolutions": 0,
            "avg_query_time": 0.0,
            "avg_update_time": 0.0,
            "avg_evolution_time": 0.0,
            "last_query": None,
            "last_update": None,
            "last_evolution": None,
            "memory_usage": {
                "nodes": 0,
                "edges": 0,
                "total": 0
            }
        })
        
        logger.info("Initialized CognitiveNexus")
    
    @log_and_handle_errors()
    async def initialize(self) -> None:
        """Initialize nexus components."""
        try:
            await self._pre_initialize()
            
            logger.info("Initializing CognitiveNexus...")
            
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
            
            # Update memory usage stats
            self.stats["memory_usage"].update({
                "nodes": len(self.knowledge_graph["nodes"]),
                "edges": len(self.knowledge_graph["edges"]),
                "total": self._calculate_total_memory()
            })
            
            await self._post_initialize()
            logger.info("Successfully initialized CognitiveNexus")
            
        except Exception as e:
            logger.error(f"Error initializing CognitiveNexus: {str(e)}")
            self.initialized = False
            raise RAGSystemError(f"Failed to initialize cognitive nexus: {str(e)}") from e
    
    @log_and_handle_errors()
    async def shutdown(self) -> None:
        """Shutdown nexus components."""
        try:
            await self._pre_shutdown()
            
            logger.info("Shutting down CognitiveNexus...")
            
            # Perform final evolution if needed
            if self.stats["total_updates"] > self.stats["total_evolutions"]:
                try:
                    await self.evolve()
                except Exception as e:
                    logger.warning(f"Error during final evolution: {str(e)}")
            
            # Save final state metrics
            final_metrics = await self._analyze_graph()
            logger.info(f"Final graph metrics: {final_metrics}")
            
            # Clear knowledge graph
            self.knowledge_graph = {}
            
            # Update memory usage stats
            self.stats["memory_usage"].update({
                "nodes": 0,
                "edges": 0,
                "total": 0
            })
            
            await self._post_shutdown()
            logger.info("Successfully shut down CognitiveNexus")
            
        except Exception as e:
            logger.error(f"Error shutting down CognitiveNexus: {str(e)}")
            raise RAGSystemError(f"Failed to shutdown cognitive nexus: {str(e)}") from e
    
    @log_and_handle_errors()
    async def get_status(self) -> Dict[str, Any]:
        """Get component status."""
        base_status = await self.get_base_status()
        
        component_status = {
            "graph_size": {
                "nodes": len(self.knowledge_graph.get("nodes", {})),
                "edges": len(self.knowledge_graph.get("edges", {}))
            },
            "metadata": self.knowledge_graph.get("metadata", {}),
            "performance": {
                "density": self._calculate_graph_density(),
                "clustering": self._calculate_clustering()
            },
            "memory_usage": self.stats["memory_usage"]
        }
        
        return {
            **base_status,
            **component_status
        }
    
    @log_and_handle_errors()
    async def update_config(self, config: Dict[str, Any]) -> None:
        """Update component configuration."""
        try:
            logger.info("Updating CognitiveNexus configuration...")
            # No configuration needed currently
            logger.info("Successfully updated configuration")
        except Exception as e:
            logger.error(f"Error updating configuration: {str(e)}")
            raise RAGSystemError(f"Failed to update configuration: {str(e)}") from e
    
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
        return await self._safe_operation("query", self._do_query(content, embeddings, entities))
    
    async def _do_query(self,
                       content: str,
                       embeddings: List[float],
                       entities: List[str]) -> Dict[str, Any]:
        """Internal query implementation."""
        try:
            if not self.initialized:
                await self.initialize()
            
            start_time = datetime.now()
            self.stats["total_queries"] += 1
            
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
            
            # Update stats
            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats["successful_queries"] += 1
            self.stats["avg_query_time"] = (
                (self.stats["avg_query_time"] * (self.stats["successful_queries"] - 1) + processing_time) /
                self.stats["successful_queries"]
            )
            self.stats["last_query"] = datetime.now().isoformat()
            
            return {
                "cognitive_context": cognitive_context,
                "relevant_nodes": relevant_nodes,
                "context": context,
                "timestamp": datetime.now().isoformat(),
                "performance": {
                    "query_time": processing_time,
                    "nodes_processed": len(relevant_nodes)
                }
            }
            
        except Exception as e:
            self.stats["failed_queries"] += 1
            raise RAGSystemError(f"Error in query: {str(e)}") from e
    
    @log_and_handle_errors()
    async def update(self, task: Dict[str, Any], result: Dict[str, Any]) -> None:
        """
        Update knowledge graph based on task results.
        
        Args:
            task: Task information
            result: Task results
        """
        return await self._safe_operation("update", self._do_update(task, result))
    
    async def _do_update(self, task: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Internal update implementation."""
        try:
            if not self.initialized:
                await self.initialize()
            
            start_time = datetime.now()
            self.stats["total_updates"] += 1
            
            # Extract knowledge
            new_knowledge = await self._extract_knowledge(task, result)
            
            # Update graph
            await self._update_graph(new_knowledge)
            
            # Update metadata
            self.knowledge_graph["metadata"]["last_updated"] = datetime.now().isoformat()
            
            # Update stats
            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats["successful_updates"] += 1
            self.stats["avg_update_time"] = (
                (self.stats["avg_update_time"] * (self.stats["successful_updates"] - 1) + processing_time) /
                self.stats["successful_updates"]
            )
            self.stats["last_update"] = datetime.now().isoformat()
            
            # Update memory usage stats
            self.stats["memory_usage"].update({
                "nodes": len(self.knowledge_graph["nodes"]),
                "edges": len(self.knowledge_graph["edges"]),
                "total": self._calculate_total_memory()
            })
            
        except Exception as e:
            self.stats["failed_updates"] += 1
            raise RAGSystemError(f"Error in update: {str(e)}") from e
    
    @log_and_handle_errors()
    async def evolve(self) -> Dict[str, Any]:
        """
        Evolve knowledge graph structure.
        
        Returns:
            Dictionary containing evolution results
        """
        return await self._safe_operation("evolve", self._do_evolve())
    
    async def _do_evolve(self) -> Dict[str, Any]:
        """Internal evolution implementation."""
        try:
            if not self.initialized:
                await self.initialize()
            
            start_time = datetime.now()
            self.stats["total_evolutions"] += 1
            
            # Analyze current state
            analysis = await self._analyze_graph()
            
            # Generate improvements
            improvements = await self._generate_improvements(analysis)
            
            # Apply improvements
            evolution_results = await self._apply_improvements(improvements)
            
            # Update stats
            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats["successful_evolutions"] += 1
            self.stats["avg_evolution_time"] = (
                (self.stats["avg_evolution_time"] * (self.stats["successful_evolutions"] - 1) + processing_time) /
                self.stats["successful_evolutions"]
            )
            self.stats["last_evolution"] = datetime.now().isoformat()
            
            # Update memory usage stats
            self.stats["memory_usage"].update({
                "nodes": len(self.knowledge_graph["nodes"]),
                "edges": len(self.knowledge_graph["edges"]),
                "total": self._calculate_total_memory()
            })
            
            return {
                **evolution_results,
                "analysis": analysis,
                "performance": {
                    "evolution_time": processing_time,
                    "improvements_applied": len(evolution_results["applied_improvements"])
                }
            }
            
        except Exception as e:
            self.stats["failed_evolutions"] += 1
            raise RAGSystemError(f"Error in evolution: {str(e)}") from e
    
    def _calculate_total_memory(self) -> int:
        """Calculate total memory usage."""
        try:
            # Simple estimation based on number of nodes and edges
            node_size = 1000  # Assume average node size of 1KB
            edge_size = 500   # Assume average edge size of 500B
            return (
                len(self.knowledge_graph["nodes"]) * node_size +
                len(self.knowledge_graph["edges"]) * edge_size
            )
        except Exception:
            return 0
    
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
