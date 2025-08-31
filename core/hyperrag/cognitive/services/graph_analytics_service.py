"""
Graph Analytics Service

Responsible for computing graph metrics, analytics, and insights.
Provides centrality measures, clustering analysis, and connectivity metrics.

Extracted from GraphFixer to follow single responsibility principle.
"""

import asyncio
from typing import Any, Dict, List, Tuple, Set
import numpy as np
from collections import defaultdict, deque

from ..interfaces.service_interfaces import IGraphAnalyticsService
from ..interfaces.base_service import ServiceConfig, CacheableMixin, AsyncServiceMixin


class GraphAnalyticsService(IGraphAnalyticsService, CacheableMixin, AsyncServiceMixin):
    """
    Service for computing graph analytics and metrics.
    
    Provides:
    - Centrality metrics (degree, betweenness, closeness, eigenvector)
    - Clustering analysis and community detection
    - Connectivity measurements
    - Completeness assessment
    - Performance analytics
    """
    
    def __init__(self, config: ServiceConfig):
        super().__init__(config)
        self.metrics_cache_ttl = 300  # 5 minutes cache TTL
        self.stats = {
            "metric_calculations": 0,
            "analytics_computed": 0,
            "cache_hits": 0
        }
    
    async def initialize(self) -> bool:
        """Initialize graph analytics service."""
        self.logger.info("Initializing GraphAnalyticsService...")
        
        if not self.validate_dependencies(["trust_graph"]):
            self.logger.warning("Trust graph not available - analytics will be limited")
        
        self._initialized = True
        self.logger.info("✓ GraphAnalyticsService initialized")
        return True
    
    async def cleanup(self) -> None:
        """Clean up service resources."""
        self.clear_cache()
        self._initialized = False
    
    async def compute_centrality_metrics(self) -> Dict[str, Any]:
        """
        Compute centrality metrics for all nodes in the graph.
        
        Returns:
            Dictionary containing centrality scores for all nodes
        """
        if not self.is_initialized:
            await self.initialize()
        
        cache_key = "centrality_metrics"
        cached_result = self.get_from_cache(cache_key)
        if cached_result:
            self.stats["cache_hits"] += 1
            return cached_result
        
        try:
            if not self.config.trust_graph:
                return {"error": "No trust graph available"}
            
            # Compute different centrality measures
            metrics = {
                "degree_centrality": await self._compute_degree_centrality(),
                "betweenness_centrality": await self._compute_betweenness_centrality(),
                "closeness_centrality": await self._compute_closeness_centrality(),
                "eigenvector_centrality": await self._compute_eigenvector_centrality()
            }
            
            # Add summary statistics
            metrics["summary"] = await self._compute_centrality_summary(metrics)
            
            self.set_cache(cache_key, metrics)
            self.stats["metric_calculations"] += 1
            
            return metrics
            
        except Exception as e:
            self.logger.exception(f"Centrality computation failed: {e}")
            return {"error": str(e)}
    
    async def analyze_clusters(self) -> Dict[str, Any]:
        """
        Analyze clustering patterns in the graph.
        
        Returns:
            Dictionary containing clustering analysis results
        """
        cache_key = "cluster_analysis"
        cached_result = self.get_from_cache(cache_key)
        if cached_result:
            self.stats["cache_hits"] += 1
            return cached_result
        
        try:
            if not self.config.trust_graph:
                return {"error": "No trust graph available"}
            
            analysis = {
                "clustering_coefficient": await self._compute_clustering_coefficient(),
                "communities": await self._detect_communities(),
                "modularity": await self._compute_modularity(),
                "cluster_distribution": await self._analyze_cluster_distribution()
            }
            
            self.set_cache(cache_key, analysis)
            self.stats["analytics_computed"] += 1
            
            return analysis
            
        except Exception as e:
            self.logger.exception(f"Cluster analysis failed: {e}")
            return {"error": str(e)}
    
    async def measure_connectivity(self) -> Dict[str, Any]:
        """
        Measure overall graph connectivity.
        
        Returns:
            Dictionary containing connectivity metrics
        """
        cache_key = "connectivity_metrics"
        cached_result = self.get_from_cache(cache_key)
        if cached_result:
            self.stats["cache_hits"] += 1
            return cached_result
        
        try:
            if not self.config.trust_graph:
                return {"error": "No trust graph available"}
            
            metrics = {
                "connected_components": await self._count_connected_components(),
                "graph_density": await self._compute_graph_density(),
                "average_path_length": await self._compute_average_path_length(),
                "diameter": await self._compute_graph_diameter(),
                "connectivity_score": 0.0
            }
            
            # Calculate overall connectivity score
            metrics["connectivity_score"] = await self._calculate_connectivity_score(metrics)
            
            self.set_cache(cache_key, metrics)
            self.stats["metric_calculations"] += 1
            
            return metrics
            
        except Exception as e:
            self.logger.exception(f"Connectivity measurement failed: {e}")
            return {"error": str(e)}
    
    async def analyze_completeness(self) -> Dict[str, Any]:
        """
        Analyze graph completeness and coverage.
        
        Returns:
            Dictionary containing completeness analysis
        """
        cache_key = "completeness_analysis"
        cached_result = self.get_from_cache(cache_key)
        if cached_result:
            self.stats["cache_hits"] += 1
            return cached_result
        
        try:
            if not self.config.trust_graph:
                return {"error": "No trust graph available"}
            
            analysis = {
                "structural_completeness": await self._analyze_structural_completeness(),
                "semantic_completeness": await self._analyze_semantic_completeness(),
                "trust_completeness": await self._analyze_trust_completeness(),
                "overall_completeness": 0.0
            }
            
            # Calculate overall completeness score
            structural_score = analysis["structural_completeness"].get("completeness_score", 0.0)
            semantic_score = analysis["semantic_completeness"].get("completeness_score", 0.0)
            trust_score = analysis["trust_completeness"].get("completeness_score", 0.0)
            
            analysis["overall_completeness"] = (structural_score + semantic_score + trust_score) / 3.0
            
            self.set_cache(cache_key, analysis)
            self.stats["analytics_computed"] += 1
            
            return analysis
            
        except Exception as e:
            self.logger.exception(f"Completeness analysis failed: {e}")
            return {"error": str(e)}
    
    # Private implementation methods
    
    async def _compute_degree_centrality(self) -> Dict[str, float]:
        """Compute degree centrality for all nodes."""
        centrality = {}
        
        max_possible_degree = len(self.config.trust_graph.nodes) - 1
        if max_possible_degree == 0:
            return centrality
        
        for node_id, node in self.config.trust_graph.nodes.items():
            degree = len(node.incoming_edges) + len(node.outgoing_edges)
            centrality[node_id] = degree / max_possible_degree
        
        return centrality
    
    async def _compute_betweenness_centrality(self) -> Dict[str, float]:
        """Compute betweenness centrality using Brandes algorithm (simplified)."""
        centrality = {node_id: 0.0 for node_id in self.config.trust_graph.nodes}
        
        # Simplified implementation - in practice would use full Brandes algorithm
        for source in self.config.trust_graph.nodes:
            # Single-source shortest paths
            shortest_paths = await self._compute_shortest_paths(source)
            
            # Update betweenness scores based on paths through each node
            for target in self.config.trust_graph.nodes:
                if source != target and target in shortest_paths:
                    path = shortest_paths[target]["path"]
                    if len(path) > 2:  # Path has intermediate nodes
                        for intermediate in path[1:-1]:  # Exclude source and target
                            centrality[intermediate] += 1.0
        
        # Normalize scores
        n = len(self.config.trust_graph.nodes)
        normalization_factor = (n - 1) * (n - 2) / 2.0 if n > 2 else 1.0
        
        for node_id in centrality:
            centrality[node_id] /= normalization_factor
        
        return centrality
    
    async def _compute_closeness_centrality(self) -> Dict[str, float]:
        """Compute closeness centrality for all nodes."""
        centrality = {}
        
        for node_id in self.config.trust_graph.nodes:
            distances = await self._compute_shortest_paths(node_id)
            
            if distances:
                total_distance = sum(dist["distance"] for dist in distances.values())
                reachable_nodes = len(distances)
                
                if total_distance > 0 and reachable_nodes > 1:
                    centrality[node_id] = (reachable_nodes - 1) / total_distance
                else:
                    centrality[node_id] = 0.0
            else:
                centrality[node_id] = 0.0
        
        return centrality
    
    async def _compute_eigenvector_centrality(self, max_iterations: int = 100) -> Dict[str, float]:
        """Compute eigenvector centrality using power iteration."""
        nodes = list(self.config.trust_graph.nodes.keys())
        n = len(nodes)
        
        if n == 0:
            return {}
        
        # Initialize centrality scores
        centrality = {node_id: 1.0 / n for node_id in nodes}
        
        # Power iteration
        for _ in range(max_iterations):
            new_centrality = {node_id: 0.0 for node_id in nodes}
            
            # Update each node's centrality
            for node_id in nodes:
                node = self.config.trust_graph.nodes[node_id]
                
                # Sum centrality from incoming neighbors
                for edge_id in node.incoming_edges:
                    if edge_id in self.config.trust_graph.edges:
                        edge = self.config.trust_graph.edges[edge_id]
                        source_id = edge.source_id
                        new_centrality[node_id] += centrality[source_id]
            
            # Normalize
            norm = sum(new_centrality.values())
            if norm > 0:
                for node_id in new_centrality:
                    new_centrality[node_id] /= norm
            
            centrality = new_centrality
        
        return centrality
    
    async def _compute_shortest_paths(self, source: str) -> Dict[str, Dict[str, Any]]:
        """Compute shortest paths from a source node using BFS."""
        if source not in self.config.trust_graph.nodes:
            return {}
        
        distances = {source: {"distance": 0, "path": [source]}}
        queue = deque([source])
        
        while queue:
            current = queue.popleft()
            current_distance = distances[current]["distance"]
            current_path = distances[current]["path"]
            
            if current in self.config.trust_graph.nodes:
                node = self.config.trust_graph.nodes[current]
                
                # Explore outgoing edges
                for edge_id in node.outgoing_edges:
                    if edge_id in self.config.trust_graph.edges:
                        edge = self.config.trust_graph.edges[edge_id]
                        neighbor = edge.target_id
                        
                        if neighbor not in distances:
                            distances[neighbor] = {
                                "distance": current_distance + 1,
                                "path": current_path + [neighbor]
                            }
                            queue.append(neighbor)
        
        return distances
    
    async def _compute_centrality_summary(self, metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Compute summary statistics for centrality metrics."""
        summary = {}
        
        for metric_name, scores in metrics.items():
            if isinstance(scores, dict) and scores:
                values = list(scores.values())
                summary[metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "max": max(values),
                    "min": min(values),
                    "top_nodes": sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
                }
        
        return summary
    
    async def _compute_clustering_coefficient(self) -> Dict[str, Any]:
        """Compute local and global clustering coefficients."""
        local_clustering = {}
        
        for node_id, node in self.config.trust_graph.nodes.items():
            neighbors = await self._get_neighbors(node_id)
            
            if len(neighbors) < 2:
                local_clustering[node_id] = 0.0
                continue
            
            # Count edges between neighbors
            edges_between_neighbors = 0
            for i, neighbor1 in enumerate(neighbors):
                for neighbor2 in neighbors[i+1:]:
                    if await self._nodes_connected(neighbor1, neighbor2):
                        edges_between_neighbors += 1
            
            # Calculate clustering coefficient
            possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
            local_clustering[node_id] = edges_between_neighbors / possible_edges if possible_edges > 0 else 0.0
        
        global_clustering = np.mean(list(local_clustering.values())) if local_clustering else 0.0
        
        return {
            "local": local_clustering,
            "global": global_clustering
        }
    
    async def _detect_communities(self) -> Dict[str, Any]:
        """Simple community detection using connected components."""
        visited = set()
        communities = []
        
        for node_id in self.config.trust_graph.nodes:
            if node_id not in visited:
                community = await self._explore_component(node_id, visited)
                if community:
                    communities.append(community)
        
        return {
            "count": len(communities),
            "communities": communities,
            "sizes": [len(community) for community in communities]
        }
    
    async def _explore_component(self, start_node: str, visited: Set[str]) -> List[str]:
        """Explore connected component starting from a node."""
        component = []
        stack = [start_node]
        
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            
            visited.add(current)
            component.append(current)
            
            # Add neighbors to stack
            neighbors = await self._get_neighbors(current)
            for neighbor in neighbors:
                if neighbor not in visited:
                    stack.append(neighbor)
        
        return component
    
    async def _get_neighbors(self, node_id: str) -> List[str]:
        """Get all neighbors of a node."""
        if node_id not in self.config.trust_graph.nodes:
            return []
        
        neighbors = set()
        node = self.config.trust_graph.nodes[node_id]
        
        # Add outgoing neighbors
        for edge_id in node.outgoing_edges:
            if edge_id in self.config.trust_graph.edges:
                edge = self.config.trust_graph.edges[edge_id]
                neighbors.add(edge.target_id)
        
        # Add incoming neighbors
        for edge_id in node.incoming_edges:
            if edge_id in self.config.trust_graph.edges:
                edge = self.config.trust_graph.edges[edge_id]
                neighbors.add(edge.source_id)
        
        return list(neighbors)
    
    async def _nodes_connected(self, node1: str, node2: str) -> bool:
        """Check if two nodes are directly connected."""
        for edge in self.config.trust_graph.edges.values():
            if ((edge.source_id == node1 and edge.target_id == node2) or
                (edge.source_id == node2 and edge.target_id == node1)):
                return True
        return False
    
    async def _compute_modularity(self) -> float:
        """Compute modularity score for community structure."""
        # Simplified modularity calculation
        # In practice would use the full modularity formula
        communities = await self._detect_communities()
        
        if not communities["communities"]:
            return 0.0
        
        total_edges = len(self.config.trust_graph.edges)
        if total_edges == 0:
            return 0.0
        
        # Simple approximation based on community sizes
        size_variance = np.var(communities["sizes"]) if communities["sizes"] else 0
        return max(0.0, 1.0 - (size_variance / (len(self.config.trust_graph.nodes) ** 2)))
    
    async def _analyze_cluster_distribution(self) -> Dict[str, Any]:
        """Analyze the distribution of cluster sizes."""
        communities = await self._detect_communities()
        sizes = communities["sizes"]
        
        if not sizes:
            return {"error": "No communities detected"}
        
        return {
            "mean_size": np.mean(sizes),
            "median_size": np.median(sizes),
            "largest_cluster": max(sizes),
            "smallest_cluster": min(sizes),
            "size_distribution": sizes
        }
    
    async def _count_connected_components(self) -> int:
        """Count the number of connected components."""
        communities = await self._detect_communities()
        return communities["count"]
    
    async def _compute_graph_density(self) -> float:
        """Compute graph density (actual edges / possible edges)."""
        n_nodes = len(self.config.trust_graph.nodes)
        n_edges = len(self.config.trust_graph.edges)
        
        if n_nodes <= 1:
            return 0.0
        
        max_possible_edges = n_nodes * (n_nodes - 1)  # For directed graph
        return n_edges / max_possible_edges
    
    async def _compute_average_path_length(self) -> float:
        """Compute average shortest path length."""
        total_distance = 0
        total_pairs = 0
        
        # Sample a subset of nodes for efficiency
        sample_nodes = list(self.config.trust_graph.nodes.keys())[:min(20, len(self.config.trust_graph.nodes))]
        
        for source in sample_nodes:
            distances = await self._compute_shortest_paths(source)
            for target, path_info in distances.items():
                if source != target:
                    total_distance += path_info["distance"]
                    total_pairs += 1
        
        return total_distance / total_pairs if total_pairs > 0 else 0.0
    
    async def _compute_graph_diameter(self) -> int:
        """Compute graph diameter (longest shortest path)."""
        max_distance = 0
        
        # Sample a subset for efficiency
        sample_nodes = list(self.config.trust_graph.nodes.keys())[:min(15, len(self.config.trust_graph.nodes))]
        
        for source in sample_nodes:
            distances = await self._compute_shortest_paths(source)
            if distances:
                max_dist_from_source = max(path_info["distance"] for path_info in distances.values())
                max_distance = max(max_distance, max_dist_from_source)
        
        return max_distance
    
    async def _calculate_connectivity_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall connectivity score from metrics."""
        factors = []
        
        # Density factor
        density = metrics.get("graph_density", 0.0)
        factors.append(min(1.0, density * 10))  # Scale density appropriately
        
        # Component factor
        components = metrics.get("connected_components", 1)
        if len(self.config.trust_graph.nodes) > 0:
            component_factor = 1.0 / components  # Fewer components is better
            factors.append(component_factor)
        
        # Path length factor (shorter is better)
        avg_path_length = metrics.get("average_path_length", 0.0)
        if avg_path_length > 0:
            path_factor = min(1.0, 3.0 / avg_path_length)  # 3 is ideal avg path length
            factors.append(path_factor)
        
        return sum(factors) / len(factors) if factors else 0.0
    
    async def _analyze_structural_completeness(self) -> Dict[str, Any]:
        """Analyze structural completeness of the graph."""
        total_nodes = len(self.config.trust_graph.nodes)
        total_edges = len(self.config.trust_graph.edges)
        
        if total_nodes == 0:
            return {"completeness_score": 0.0}
        
        # Simple structural metrics
        avg_degree = (2 * total_edges) / total_nodes if total_nodes > 0 else 0
        isolated_nodes = sum(
            1 for node in self.config.trust_graph.nodes.values()
            if len(node.incoming_edges) == 0 and len(node.outgoing_edges) == 0
        )
        
        completeness_score = min(1.0, avg_degree / 3.0)  # 3 is reasonable avg degree
        
        return {
            "completeness_score": completeness_score,
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "average_degree": avg_degree,
            "isolated_nodes": isolated_nodes
        }
    
    async def _analyze_semantic_completeness(self) -> Dict[str, Any]:
        """Analyze semantic completeness using available data."""
        # Count nodes with embeddings
        nodes_with_embeddings = sum(
            1 for node in self.config.trust_graph.nodes.values()
            if hasattr(node, 'embedding') and node.embedding is not None
        )
        
        total_nodes = len(self.config.trust_graph.nodes)
        embedding_coverage = nodes_with_embeddings / total_nodes if total_nodes > 0 else 0.0
        
        return {
            "completeness_score": embedding_coverage,
            "nodes_with_embeddings": nodes_with_embeddings,
            "total_nodes": total_nodes,
            "embedding_coverage": embedding_coverage
        }
    
    async def _analyze_trust_completeness(self) -> Dict[str, Any]:
        """Analyze trust score completeness and distribution."""
        trust_scores = [
            node.trust_score for node in self.config.trust_graph.nodes.values()
            if hasattr(node, 'trust_score')
        ]
        
        if not trust_scores:
            return {"completeness_score": 0.0}
        
        # Analyze trust distribution
        avg_trust = np.mean(trust_scores)
        trust_std = np.std(trust_scores)
        
        # Good trust distribution should have reasonable average and not too much variance
        completeness_score = avg_trust * (1.0 - min(1.0, trust_std))
        
        return {
            "completeness_score": completeness_score,
            "avg_trust": avg_trust,
            "trust_std": trust_std,
            "high_trust_nodes": sum(1 for score in trust_scores if score > 0.8),
            "low_trust_nodes": sum(1 for score in trust_scores if score < 0.3)
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            "metric_calculations": self.stats["metric_calculations"],
            "analytics_computed": self.stats["analytics_computed"],
            "cache_hits": self.stats["cache_hits"],
            "cache_size": len(self._cache) if hasattr(self, '_cache') else 0
        }