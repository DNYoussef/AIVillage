"""
Missing Node Detector for Knowledge Graph RAG
Identifies gaps in knowledge and suggests missing concepts or connections
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Set
import numpy as np
import networkx as nx
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

class GapType(Enum):
    """Types of knowledge gaps that can be detected"""
    MISSING_CONCEPT = "missing_concept"         # Concept that should exist
    MISSING_CONNECTION = "missing_connection"   # Connection between existing concepts
    STRUCTURAL_GAP = "structural_gap"          # Gap in graph structure
    DOMAIN_GAP = "domain_gap"                  # Missing domain knowledge
    TEMPORAL_GAP = "temporal_gap"              # Missing temporal connections
    CAUSAL_GAP = "causal_gap"                 # Missing causal relationships

@dataclass
class KnowledgeGap:
    """Represents a detected knowledge gap"""
    gap_id: str
    gap_type: GapType
    description: str
    confidence: float
    importance: float
    suggested_concepts: List[str] = field(default_factory=list)
    suggested_connections: List[Tuple[str, str]] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    context_nodes: List[str] = field(default_factory=list)

@dataclass
class GapAnalysis:
    """Analysis of knowledge gaps in a specific area"""
    analysis_id: str
    focus_area: str
    detected_gaps: List[KnowledgeGap] = field(default_factory=list)
    coverage_score: float = 0.0
    completeness_metrics: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

class MissingNodeDetector:
    """
    Detects missing nodes and connections in knowledge graphs
    Uses graph analysis and pattern recognition to identify gaps
    """
    
    def __init__(self, knowledge_graph: nx.Graph):
        self.knowledge_graph = knowledge_graph
        self.gap_counter = 0
        
        # Detection parameters
        self.structural_threshold = 0.3   # Minimum structural similarity for gap detection
        self.importance_threshold = 0.4   # Minimum importance for gap reporting
        self.confidence_threshold = 0.5   # Minimum confidence for suggestions
        
        # Analysis weights
        self.weights = {
            'structural_analysis': 0.3,
            'connectivity_analysis': 0.2,
            'pattern_analysis': 0.25,
            'semantic_analysis': 0.25
        }
    
    async def detect_missing_nodes(self,
                                 focus_concepts: Optional[List[str]] = None,
                                 analysis_scope: str = "local",
                                 gap_types: Optional[List[GapType]] = None) -> GapAnalysis:
        """
        Detect missing nodes and connections in the knowledge graph
        
        Args:
            focus_concepts: Specific concepts to analyze around
            analysis_scope: "local", "global", or "domain"
            gap_types: Specific types of gaps to detect
            
        Returns:
            GapAnalysis with detected gaps and recommendations
        """
        analysis = GapAnalysis(
            analysis_id=f"analysis_{self.gap_counter}",
            focus_area=f"{analysis_scope}_analysis"
        )
        self.gap_counter += 1
        
        if gap_types is None:
            gap_types = list(GapType)
        
        # Different detection strategies based on gap types
        for gap_type in gap_types:
            if gap_type == GapType.MISSING_CONCEPT:
                gaps = await self._detect_missing_concepts(focus_concepts, analysis_scope)
            elif gap_type == GapType.MISSING_CONNECTION:
                gaps = await self._detect_missing_connections(focus_concepts, analysis_scope)
            elif gap_type == GapType.STRUCTURAL_GAP:
                gaps = await self._detect_structural_gaps(focus_concepts, analysis_scope)
            elif gap_type == GapType.DOMAIN_GAP:
                gaps = await self._detect_domain_gaps(focus_concepts)
            elif gap_type == GapType.TEMPORAL_GAP:
                gaps = await self._detect_temporal_gaps(focus_concepts)
            elif gap_type == GapType.CAUSAL_GAP:
                gaps = await self._detect_causal_gaps(focus_concepts)
            else:
                gaps = []
            
            analysis.detected_gaps.extend(gaps)
        
        # Calculate completeness metrics
        analysis.completeness_metrics = await self._calculate_completeness_metrics(
            focus_concepts, analysis_scope
        )
        
        # Calculate overall coverage score
        analysis.coverage_score = await self._calculate_coverage_score(analysis.completeness_metrics)
        
        # Generate recommendations
        analysis.recommendations = await self._generate_recommendations(analysis.detected_gaps)
        
        # Filter and rank gaps by importance
        analysis.detected_gaps = await self._rank_gaps_by_importance(analysis.detected_gaps)
        
        return analysis
    
    async def _detect_missing_concepts(self,
                                     focus_concepts: Optional[List[str]],
                                     scope: str) -> List[KnowledgeGap]:
        """Detect missing concepts using structural pattern analysis"""
        gaps = []
        
        # If no focus concepts, analyze entire graph
        if not focus_concepts:
            focus_concepts = list(self.knowledge_graph.nodes())[:20]  # Sample for performance
        
        for concept in focus_concepts:
            if concept not in self.knowledge_graph.nodes():
                continue
            
            # Analyze neighborhood patterns
            neighbors = list(self.knowledge_graph.neighbors(concept))
            
            # Find common patterns in neighbor connections
            common_patterns = await self._find_common_neighbor_patterns(neighbors)
            
            # Detect missing concepts based on incomplete patterns
            for pattern_nodes, pattern_frequency in common_patterns.items():
                if pattern_frequency >= 3:  # Pattern appears in multiple places
                    # Check if this concept fits the pattern but is missing connections
                    missing_concept_candidates = await self._suggest_pattern_completions(
                        concept, pattern_nodes
                    )
                    
                    for candidate in missing_concept_candidates:
                        gap = KnowledgeGap(
                            gap_id=f"gap_{self.gap_counter}",
                            gap_type=GapType.MISSING_CONCEPT,
                            description=f"Missing concept that completes pattern around {concept}",
                            confidence=min(0.8, pattern_frequency / 5.0),
                            importance=0.6,
                            suggested_concepts=[candidate],
                            context_nodes=[concept] + list(pattern_nodes),
                            evidence=[f"Pattern appears {pattern_frequency} times"]
                        )
                        gaps.append(gap)
                        self.gap_counter += 1
        
        return gaps[:10]  # Limit results
    
    async def _detect_missing_connections(self,
                                        focus_concepts: Optional[List[str]],
                                        scope: str) -> List[KnowledgeGap]:
        """Detect missing connections between existing concepts"""
        gaps = []
        
        if not focus_concepts:
            focus_concepts = list(self.knowledge_graph.nodes())[:20]
        
        # Look for concepts that should be connected but aren't
        for i, concept1 in enumerate(focus_concepts):
            for concept2 in focus_concepts[i+1:]:
                if (concept1 in self.knowledge_graph.nodes() and 
                    concept2 in self.knowledge_graph.nodes() and
                    not self.knowledge_graph.has_edge(concept1, concept2)):
                    
                    # Calculate connection likelihood
                    connection_score = await self._calculate_connection_likelihood(concept1, concept2)
                    
                    if connection_score >= self.confidence_threshold:
                        gap = KnowledgeGap(
                            gap_id=f"gap_{self.gap_counter}",
                            gap_type=GapType.MISSING_CONNECTION,
                            description=f"Missing connection between {concept1} and {concept2}",
                            confidence=connection_score,
                            importance=connection_score * 0.8,
                            suggested_connections=[(concept1, concept2)],
                            context_nodes=[concept1, concept2],
                            evidence=[f"Connection likelihood: {connection_score:.2f}"]
                        )
                        gaps.append(gap)
                        self.gap_counter += 1
        
        return gaps[:15]  # Limit results
    
    async def _detect_structural_gaps(self,
                                    focus_concepts: Optional[List[str]],
                                    scope: str) -> List[KnowledgeGap]:
        """Detect structural gaps in the graph topology"""
        gaps = []
        
        # Analyze graph structure for common patterns
        # Look for bridge nodes that could indicate missing structure
        
        # Find articulation points (nodes whose removal disconnects the graph)
        try:
            articulation_points = list(nx.articulation_points(self.knowledge_graph))
            
            for bridge_node in articulation_points[:5]:  # Analyze first 5
                # This indicates a structural bottleneck
                # There might be missing nodes that could provide alternative paths
                
                # Find the components that would be disconnected
                temp_graph = self.knowledge_graph.copy()
                temp_graph.remove_node(bridge_node)
                components = list(nx.connected_components(temp_graph))
                
                if len(components) > 1:
                    gap = KnowledgeGap(
                        gap_id=f"gap_{self.gap_counter}",
                        gap_type=GapType.STRUCTURAL_GAP,
                        description=f"Structural bottleneck at {bridge_node} - missing alternative connections",
                        confidence=0.7,
                        importance=0.8,  # High importance for structural gaps
                        context_nodes=[bridge_node],
                        evidence=[f"Articulation point connects {len(components)} components"],
                        suggested_concepts=[f"bridge_concept_for_{bridge_node}"]
                    )
                    gaps.append(gap)
                    self.gap_counter += 1
                    
        except nx.NetworkXError:
            # Graph might not be connected
            pass
        
        return gaps[:5]  # Limit structural gaps
    
    async def _detect_domain_gaps(self, focus_concepts: Optional[List[str]]) -> List[KnowledgeGap]:
        """Detect missing domain knowledge"""
        gaps = []
        
        # Analyze domain coverage by looking at node types/categories
        domain_distribution = defaultdict(int)
        
        for node in self.knowledge_graph.nodes(data=True):
            node_id, node_data = node
            domain = node_data.get('domain', 'unknown')
            domain_distribution[domain] += 1
        
        # Find domains that are underrepresented
        total_nodes = len(self.knowledge_graph.nodes())
        expected_domain_size = total_nodes / max(1, len(domain_distribution))
        
        for domain, count in domain_distribution.items():
            if count < expected_domain_size * 0.3:  # Significantly underrepresented
                gap = KnowledgeGap(
                    gap_id=f"gap_{self.gap_counter}",
                    gap_type=GapType.DOMAIN_GAP,
                    description=f"Underrepresented domain: {domain}",
                    confidence=0.6,
                    importance=0.5,
                    suggested_concepts=[f"more_{domain}_concepts"],
                    evidence=[f"Domain has only {count} nodes vs expected {expected_domain_size:.1f}"]
                )
                gaps.append(gap)
                self.gap_counter += 1
        
        return gaps[:5]  # Limit domain gaps
    
    async def _detect_temporal_gaps(self, focus_concepts: Optional[List[str]]) -> List[KnowledgeGap]:
        """Detect missing temporal connections"""
        gaps = []
        
        # Look for temporal sequences that might have missing steps
        # This would be enhanced with actual temporal metadata
        
        temporal_nodes = []
        for node in self.knowledge_graph.nodes(data=True):
            node_id, node_data = node
            if 'timestamp' in node_data or 'temporal' in node_data.get('tags', []):
                temporal_nodes.append(node_id)
        
        # Find potential temporal gaps (simplified)
        if len(temporal_nodes) >= 3:
            # Look for nodes that could be in sequence but aren't connected
            for i in range(len(temporal_nodes) - 2):
                node1, node2, node3 = temporal_nodes[i], temporal_nodes[i+1], temporal_nodes[i+2]
                
                # If 1->3 connected but not 1->2->3, there might be a gap
                if (self.knowledge_graph.has_edge(node1, node3) and
                    not (self.knowledge_graph.has_edge(node1, node2) and 
                         self.knowledge_graph.has_edge(node2, node3))):
                    
                    gap = KnowledgeGap(
                        gap_id=f"gap_{self.gap_counter}",
                        gap_type=GapType.TEMPORAL_GAP,
                        description=f"Missing temporal step between {node1} and {node3}",
                        confidence=0.5,
                        importance=0.6,
                        suggested_connections=[(node1, node2), (node2, node3)],
                        context_nodes=[node1, node2, node3],
                        evidence=["Temporal sequence gap detected"]
                    )
                    gaps.append(gap)
                    self.gap_counter += 1
        
        return gaps[:3]  # Limit temporal gaps
    
    async def _detect_causal_gaps(self, focus_concepts: Optional[List[str]]) -> List[KnowledgeGap]:
        """Detect missing causal relationships"""
        gaps = []
        
        # Look for potential causal relationships that aren't explicitly modeled
        # This is simplified - would be enhanced with causal inference
        
        causal_indicators = ['cause', 'effect', 'result', 'leads_to', 'because']
        
        potential_causes = []
        potential_effects = []
        
        for node in self.knowledge_graph.nodes(data=True):
            node_id, node_data = node
            content = node_data.get('content', '').lower()
            
            if any(indicator in content for indicator in causal_indicators[:2]):
                potential_causes.append(node_id)
            if any(indicator in content for indicator in causal_indicators[2:]):
                potential_effects.append(node_id)
        
        # Find potential causal connections that don't exist
        for cause in potential_causes[:5]:
            for effect in potential_effects[:5]:
                if (cause != effect and 
                    not self.knowledge_graph.has_edge(cause, effect) and
                    not self.knowledge_graph.has_edge(effect, cause)):
                    
                    # Check if they're in reasonable proximity (could be causally related)
                    try:
                        distance = nx.shortest_path_length(self.knowledge_graph, cause, effect)
                        if distance <= 3:  # Close enough to potentially be causal
                            gap = KnowledgeGap(
                                gap_id=f"gap_{self.gap_counter}",
                                gap_type=GapType.CAUSAL_GAP,
                                description=f"Potential causal relationship: {cause} -> {effect}",
                                confidence=0.4,  # Lower confidence for causal inference
                                importance=0.7,  # High importance for causal relationships
                                suggested_connections=[(cause, effect)],
                                context_nodes=[cause, effect],
                                evidence=[f"Distance: {distance}, causal indicators present"]
                            )
                            gaps.append(gap)
                            self.gap_counter += 1
                    except nx.NetworkXNoPath:
                        pass
        
        return gaps[:3]  # Limit causal gaps
    
    async def _find_common_neighbor_patterns(self, neighbors: List[str]) -> Dict[frozenset, int]:
        """Find common patterns in neighbor connections"""
        patterns = defaultdict(int)
        
        for neighbor in neighbors:
            if neighbor in self.knowledge_graph.nodes():
                neighbor_neighbors = set(self.knowledge_graph.neighbors(neighbor))
                # Create pattern from neighbor's connections
                pattern = frozenset(neighbor_neighbors)
                patterns[pattern] += 1
        
        return dict(patterns)
    
    async def _suggest_pattern_completions(self, concept: str, pattern_nodes: frozenset) -> List[str]:
        """Suggest concepts that would complete a pattern"""
        suggestions = []
        
        # Simple suggestion based on pattern analysis
        # This would be enhanced with semantic similarity
        
        pattern_list = list(pattern_nodes)[:3]  # Limit for performance
        suggestion = f"completion_for_{concept}_with_{'_'.join(pattern_list)}"
        suggestions.append(suggestion)
        
        return suggestions
    
    async def _calculate_connection_likelihood(self, concept1: str, concept2: str) -> float:
        """Calculate likelihood that two concepts should be connected"""
        # Calculate based on common neighbors and graph structure
        
        neighbors1 = set(self.knowledge_graph.neighbors(concept1))
        neighbors2 = set(self.knowledge_graph.neighbors(concept2))
        
        # Common neighbors indicate potential connection
        common_neighbors = neighbors1 & neighbors2
        jaccard_similarity = len(common_neighbors) / len(neighbors1 | neighbors2) if (neighbors1 | neighbors2) else 0
        
        # Distance factor (closer nodes more likely to connect)
        try:
            distance = nx.shortest_path_length(self.knowledge_graph, concept1, concept2)
            distance_factor = max(0, 1 - (distance / 10))  # Normalize distance
        except nx.NetworkXNoPath:
            distance_factor = 0
        
        # Combined likelihood
        likelihood = (jaccard_similarity * 0.7) + (distance_factor * 0.3)
        
        return likelihood
    
    async def _calculate_completeness_metrics(self,
                                           focus_concepts: Optional[List[str]],
                                           scope: str) -> Dict[str, float]:
        """Calculate various completeness metrics"""
        metrics = {}
        
        # Graph density (how well connected)
        metrics['density'] = nx.density(self.knowledge_graph)
        
        # Clustering coefficient (local connectivity)
        try:
            metrics['clustering'] = nx.average_clustering(self.knowledge_graph)
        except:
            metrics['clustering'] = 0.0
        
        # Average path length (how well integrated)
        try:
            if nx.is_connected(self.knowledge_graph):
                metrics['avg_path_length'] = nx.average_shortest_path_length(self.knowledge_graph)
            else:
                # Handle disconnected graph
                largest_cc = max(nx.connected_components(self.knowledge_graph), key=len)
                subgraph = self.knowledge_graph.subgraph(largest_cc)
                metrics['avg_path_length'] = nx.average_shortest_path_length(subgraph)
        except:
            metrics['avg_path_length'] = float('inf')
        
        # Domain coverage (variety of node types)
        node_types = set()
        for _, node_data in self.knowledge_graph.nodes(data=True):
            node_types.add(node_data.get('node_type', 'unknown'))
        
        metrics['domain_diversity'] = len(node_types)
        
        return metrics
    
    async def _calculate_coverage_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall coverage score"""
        # Normalize and combine metrics
        normalized_metrics = {}
        
        # Density (0-1, higher is better)
        normalized_metrics['density'] = min(1.0, metrics.get('density', 0))
        
        # Clustering (0-1, higher is better)
        normalized_metrics['clustering'] = min(1.0, metrics.get('clustering', 0))
        
        # Average path length (normalize by max reasonable path length)
        avg_path = metrics.get('avg_path_length', float('inf'))
        if avg_path == float('inf'):
            normalized_metrics['connectivity'] = 0.0
        else:
            normalized_metrics['connectivity'] = max(0, 1 - (avg_path / 10))
        
        # Domain diversity (normalize by reasonable max diversity)
        diversity = metrics.get('domain_diversity', 0)
        normalized_metrics['diversity'] = min(1.0, diversity / 20)
        
        # Weighted average
        coverage_score = (
            normalized_metrics['density'] * 0.3 +
            normalized_metrics['clustering'] * 0.25 +
            normalized_metrics['connectivity'] * 0.25 +
            normalized_metrics['diversity'] * 0.2
        )
        
        return coverage_score
    
    async def _generate_recommendations(self, gaps: List[KnowledgeGap]) -> List[str]:
        """Generate recommendations based on detected gaps"""
        recommendations = []
        
        # Count gaps by type
        gap_counts = Counter(gap.gap_type for gap in gaps)
        
        if gap_counts[GapType.MISSING_CONCEPT] > 5:
            recommendations.append("Consider adding more diverse concepts to improve knowledge coverage")
        
        if gap_counts[GapType.MISSING_CONNECTION] > 10:
            recommendations.append("Focus on creating more connections between existing concepts")
        
        if gap_counts[GapType.STRUCTURAL_GAP] > 2:
            recommendations.append("Address structural bottlenecks to improve graph connectivity")
        
        if gap_counts[GapType.DOMAIN_GAP] > 3:
            recommendations.append("Expand knowledge in underrepresented domains")
        
        # High-confidence gaps
        high_confidence_gaps = [g for g in gaps if g.confidence >= 0.7]
        if len(high_confidence_gaps) > 5:
            recommendations.append("Prioritize addressing high-confidence gaps first")
        
        return recommendations
    
    async def _rank_gaps_by_importance(self, gaps: List[KnowledgeGap]) -> List[KnowledgeGap]:
        """Rank gaps by importance and confidence"""
        # Filter by minimum thresholds
        filtered_gaps = [
            gap for gap in gaps 
            if gap.importance >= self.importance_threshold and gap.confidence >= 0.3
        ]
        
        # Sort by combined importance and confidence
        filtered_gaps.sort(
            key=lambda g: (g.importance * 0.6 + g.confidence * 0.4),
            reverse=True
        )
        
        return filtered_gaps[:20]  # Return top 20 gaps
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get statistics about the missing node detection system"""
        return {
            'graph_size': self.knowledge_graph.number_of_nodes(),
            'graph_edges': self.knowledge_graph.number_of_edges(),
            'gap_types_supported': len(GapType),
            'structural_threshold': self.structural_threshold,
            'importance_threshold': self.importance_threshold,
            'confidence_threshold': self.confidence_threshold,
            'detection_weights': self.weights
        }