"""
Graph Analysis and Completeness Assessment

Focused module for analyzing graph completeness and quality metrics.
Provides comprehensive analysis capabilities with configurable metrics.
"""

from dataclasses import dataclass
import logging
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger(__name__)


class TrustGraphProtocol(Protocol):
    """Protocol for trust graph dependency."""

    @property
    def nodes(self) -> dict[str, Any]:
        """Node collection."""
        ...

    @property
    def edges(self) -> dict[str, Any]:
        """Edge collection."""
        ...


class VectorEngineProtocol(Protocol):
    """Protocol for vector engine dependency."""

    async def analyze_semantic_coverage(self) -> dict[str, Any]:
        """Analyze semantic coverage."""
        ...


@dataclass(frozen=True)
class AnalysisConfig:
    """Configuration for graph analysis - no magic values."""

    TARGET_AVERAGE_DEGREE: float = 3.0
    MIN_TRUST_THRESHOLD: float = 0.3
    HIGH_TRUST_THRESHOLD: float = 0.8
    MIN_CONNECTIVITY_SCORE: float = 0.5


class StructuralAnalyzer:
    """Analyzes structural properties of the graph."""

    def __init__(self, trust_graph: TrustGraphProtocol | None = None, config: AnalysisConfig | None = None):
        self._trust_graph = trust_graph
        self._config = config or AnalysisConfig()

    async def analyze_structural_completeness(self) -> dict[str, Any]:
        """Analyze structural completeness of the graph."""
        if not self._trust_graph:
            return {"completeness_score": 0.0, "error": "No trust graph available"}

        total_nodes = len(self._trust_graph.nodes)
        total_edges = len(self._trust_graph.edges)

        if total_nodes == 0:
            return {"completeness_score": 0.0, "total_nodes": 0, "total_edges": 0}

        # Calculate metrics
        avg_degree = (2 * total_edges) / total_nodes
        isolated_nodes = self._count_isolated_nodes()
        weakly_connected_nodes = self._count_weakly_connected_nodes()

        # Calculate completeness score
        completeness_score = min(1.0, avg_degree / self._config.TARGET_AVERAGE_DEGREE)

        # Penalize for isolated and weakly connected nodes
        isolation_penalty = (isolated_nodes + weakly_connected_nodes * 0.5) / total_nodes
        completeness_score = max(0.0, completeness_score - isolation_penalty)

        return {
            "completeness_score": completeness_score,
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "average_degree": avg_degree,
            "isolated_nodes": isolated_nodes,
            "weakly_connected_nodes": weakly_connected_nodes,
            "isolation_penalty": isolation_penalty,
        }

    def _count_isolated_nodes(self) -> int:
        """Count nodes with no connections."""
        if not self._trust_graph:
            return 0

        return sum(
            1
            for node in self._trust_graph.nodes.values()
            if (len(getattr(node, "incoming_edges", [])) == 0 and len(getattr(node, "outgoing_edges", [])) == 0)
        )

    def _count_weakly_connected_nodes(self) -> int:
        """Count nodes with 1-2 connections."""
        if not self._trust_graph:
            return 0

        return sum(
            1
            for node in self._trust_graph.nodes.values()
            if 1 <= (len(getattr(node, "incoming_edges", [])) + len(getattr(node, "outgoing_edges", []))) <= 2
        )


class TrustAnalyzer:
    """Analyzes trust distribution and patterns."""

    def __init__(self, trust_graph: TrustGraphProtocol | None = None, config: AnalysisConfig | None = None):
        self._trust_graph = trust_graph
        self._config = config or AnalysisConfig()

    async def analyze_trust_distribution(self) -> dict[str, Any]:
        """Analyze trust score distribution across the graph."""
        if not self._trust_graph:
            return {"avg_trust": 0.0, "error": "No trust graph available"}

        trust_scores = [getattr(node, "trust_score", 0.0) for node in self._trust_graph.nodes.values()]

        if not trust_scores:
            return {"avg_trust": 0.0, "node_count": 0}

        avg_trust = np.mean(trust_scores)
        trust_std = np.std(trust_scores)

        # Categorize nodes by trust level
        high_trust_nodes = sum(1 for score in trust_scores if score > self._config.HIGH_TRUST_THRESHOLD)
        low_trust_nodes = sum(1 for score in trust_scores if score < self._config.MIN_TRUST_THRESHOLD)
        medium_trust_nodes = len(trust_scores) - high_trust_nodes - low_trust_nodes

        # Calculate trust quality score
        trust_quality = self._calculate_trust_quality(trust_scores)

        return {
            "avg_trust": avg_trust,
            "trust_std": trust_std,
            "trust_quality": trust_quality,
            "high_trust_nodes": high_trust_nodes,
            "medium_trust_nodes": medium_trust_nodes,
            "low_trust_nodes": low_trust_nodes,
            "total_nodes": len(trust_scores),
            "trust_distribution": {
                "min": min(trust_scores),
                "max": max(trust_scores),
                "median": np.median(trust_scores),
                "percentile_25": np.percentile(trust_scores, 25),
                "percentile_75": np.percentile(trust_scores, 75),
            },
        }

    def _calculate_trust_quality(self, trust_scores: list[float]) -> float:
        """Calculate overall trust quality metric."""
        if not trust_scores:
            return 0.0

        # Higher average trust is better
        avg_component = np.mean(trust_scores)

        # Lower variance is better (more consistent trust)
        variance_component = 1.0 - min(1.0, np.var(trust_scores))

        # More high-trust nodes is better
        high_trust_ratio = sum(1 for score in trust_scores if score > self._config.HIGH_TRUST_THRESHOLD) / len(
            trust_scores
        )

        # Weighted combination
        return avg_component * 0.4 + variance_component * 0.3 + high_trust_ratio * 0.3


class ConnectivityAnalyzer:
    """Analyzes connectivity patterns and quality."""

    def __init__(self, trust_graph: TrustGraphProtocol | None = None, config: AnalysisConfig | None = None):
        self._trust_graph = trust_graph
        self._config = config or AnalysisConfig()

    async def analyze_connectivity_patterns(self) -> dict[str, Any]:
        """Analyze connectivity patterns in the graph."""
        if not self._trust_graph:
            return {"connectivity_score": 0.0, "error": "No trust graph available"}

        total_nodes = len(self._trust_graph.nodes)
        total_edges = len(self._trust_graph.edges)

        if total_nodes == 0:
            return {"connectivity_score": 0.0, "total_nodes": 0}

        # Calculate connectivity metrics
        edge_density = self._calculate_edge_density(total_nodes, total_edges)
        clustering_coefficient = await self._estimate_clustering_coefficient()
        degree_distribution = self._analyze_degree_distribution()

        # Calculate overall connectivity score
        connectivity_score = self._calculate_connectivity_score(
            edge_density, clustering_coefficient, degree_distribution
        )

        return {
            "connectivity_score": connectivity_score,
            "edge_density": edge_density,
            "clustering_coefficient": clustering_coefficient,
            "degree_distribution": degree_distribution,
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "components_estimate": 1,  # Simplified - would need full graph traversal
        }

    def _calculate_edge_density(self, total_nodes: int, total_edges: int) -> float:
        """Calculate edge density of the graph."""
        if total_nodes <= 1:
            return 0.0

        max_possible_edges = total_nodes * (total_nodes - 1) / 2
        return total_edges / max_possible_edges

    async def _estimate_clustering_coefficient(self) -> float:
        """Estimate clustering coefficient (simplified version)."""
        if not self._trust_graph:
            return 0.0

        # Simplified estimation - would need full triangle counting for accuracy
        total_nodes = len(self._trust_graph.nodes)
        total_edges = len(self._trust_graph.edges)

        if total_nodes < 3:
            return 0.0

        # Rough approximation based on edge density
        edge_density = self._calculate_edge_density(total_nodes, total_edges)
        return min(1.0, edge_density * 2)  # Heuristic approximation

    def _analyze_degree_distribution(self) -> dict[str, Any]:
        """Analyze the distribution of node degrees."""
        if not self._trust_graph:
            return {"mean": 0.0, "std": 0.0, "max": 0}

        degrees = []
        for node in self._trust_graph.nodes.values():
            degree = len(getattr(node, "incoming_edges", [])) + len(getattr(node, "outgoing_edges", []))
            degrees.append(degree)

        if not degrees:
            return {"mean": 0.0, "std": 0.0, "max": 0}

        return {
            "mean": np.mean(degrees),
            "std": np.std(degrees),
            "max": max(degrees),
            "min": min(degrees),
            "median": np.median(degrees),
        }

    def _calculate_connectivity_score(
        self, edge_density: float, clustering_coefficient: float, degree_distribution: dict[str, Any]
    ) -> float:
        """Calculate overall connectivity quality score."""
        # Edge density component (but not too dense)
        density_score = min(1.0, edge_density * 10)  # Scale appropriately
        if edge_density > 0.8:  # Penalize over-connectivity
            density_score *= 0.8

        # Clustering component
        clustering_score = clustering_coefficient

        # Degree distribution component (prefer some variance, avoid extremes)
        degree_distribution.get("std", 0.0)
        mean_degree = degree_distribution.get("mean", 0.0)
        degree_score = min(1.0, mean_degree / 5.0)  # Target ~5 average degree

        # Weighted combination
        return density_score * 0.4 + clustering_score * 0.3 + degree_score * 0.3


class SemanticAnalyzer:
    """Analyzes semantic coverage and quality."""

    def __init__(self, vector_engine: VectorEngineProtocol | None = None):
        self._vector_engine = vector_engine

    async def analyze_semantic_completeness(self) -> dict[str, Any]:
        """Analyze semantic completeness using vector analysis."""
        if not self._vector_engine:
            return {"completeness_score": 0.5, "note": "No vector engine available"}

        try:
            # Delegate to vector engine for detailed semantic analysis
            semantic_analysis = await self._vector_engine.analyze_semantic_coverage()
            return semantic_analysis
        except Exception as e:
            logger.warning(f"Semantic analysis failed: {e}")
            return {
                "completeness_score": 0.5,
                "error": str(e),
                "fallback_metrics": {
                    "semantic_clusters": 0,
                    "coverage_areas": [],
                },
            }


class GraphAnalyzer:
    """
    Comprehensive graph analysis engine.

    Coordinates different analysis components to provide
    complete assessment of graph quality and completeness.
    """

    def __init__(
        self,
        trust_graph: TrustGraphProtocol | None = None,
        vector_engine: VectorEngineProtocol | None = None,
        config: AnalysisConfig | None = None,
    ):
        self._trust_graph = trust_graph
        self._vector_engine = vector_engine
        self._config = config or AnalysisConfig()

        # Initialize specialized analyzers
        self._structural_analyzer = StructuralAnalyzer(trust_graph, config)
        self._trust_analyzer = TrustAnalyzer(trust_graph, config)
        self._connectivity_analyzer = ConnectivityAnalyzer(trust_graph, config)
        self._semantic_analyzer = SemanticAnalyzer(vector_engine)

    async def analyze_graph_completeness(self) -> dict[str, Any]:
        """Perform comprehensive graph completeness analysis."""
        try:
            # Run all analyses concurrently
            import asyncio

            structural_task = self._structural_analyzer.analyze_structural_completeness()
            trust_task = self._trust_analyzer.analyze_trust_distribution()
            connectivity_task = self._connectivity_analyzer.analyze_connectivity_patterns()
            semantic_task = self._semantic_analyzer.analyze_semantic_completeness()

            structural_metrics, trust_metrics, connectivity_metrics, semantic_metrics = await asyncio.gather(
                structural_task, trust_task, connectivity_task, semantic_task, return_exceptions=True
            )

            # Handle any exceptions
            if isinstance(structural_metrics, Exception):
                structural_metrics = {"completeness_score": 0.0, "error": str(structural_metrics)}
            if isinstance(trust_metrics, Exception):
                trust_metrics = {"avg_trust": 0.0, "error": str(trust_metrics)}
            if isinstance(connectivity_metrics, Exception):
                connectivity_metrics = {"connectivity_score": 0.0, "error": str(connectivity_metrics)}
            if isinstance(semantic_metrics, Exception):
                semantic_metrics = {"completeness_score": 0.5, "error": str(semantic_metrics)}

            # Calculate overall completeness score
            overall_score = self._calculate_overall_completeness(
                structural_metrics, trust_metrics, connectivity_metrics, semantic_metrics
            )

            return {
                "overall_completeness": overall_score,
                "structural_metrics": structural_metrics,
                "trust_metrics": trust_metrics,
                "connectivity_metrics": connectivity_metrics,
                "semantic_metrics": semantic_metrics,
                "analysis_summary": self._generate_analysis_summary(overall_score),
            }

        except Exception as e:
            logger.exception("Graph completeness analysis failed")
            return {"error": str(e), "overall_completeness": 0.0}

    def _calculate_overall_completeness(
        self, structural: dict[str, Any], trust: dict[str, Any], connectivity: dict[str, Any], semantic: dict[str, Any]
    ) -> float:
        """Calculate weighted overall completeness score."""
        # Extract individual scores with defaults
        structural_score = structural.get("completeness_score", 0.0)
        trust_score = trust.get("avg_trust", 0.0)
        connectivity_score = connectivity.get("connectivity_score", 0.0)
        semantic_score = semantic.get("completeness_score", 0.5)

        # Weighted combination - adjust weights based on importance
        weights = {
            "structural": 0.3,
            "trust": 0.25,
            "connectivity": 0.25,
            "semantic": 0.2,
        }

        overall_score = (
            structural_score * weights["structural"]
            + trust_score * weights["trust"]
            + connectivity_score * weights["connectivity"]
            + semantic_score * weights["semantic"]
        )

        return min(1.0, max(0.0, overall_score))

    def _generate_analysis_summary(self, overall_score: float) -> dict[str, Any]:
        """Generate human-readable analysis summary."""
        if overall_score >= 0.8:
            quality_level = "Excellent"
            recommendations = ["Continue monitoring for quality maintenance"]
        elif overall_score >= 0.6:
            quality_level = "Good"
            recommendations = [
                "Focus on improving weakest metrics",
                "Consider adding more connections to isolated nodes",
            ]
        elif overall_score >= 0.4:
            quality_level = "Fair"
            recommendations = [
                "Significant improvement needed",
                "Focus on structural connectivity",
                "Review trust score distribution",
            ]
        else:
            quality_level = "Poor"
            recommendations = [
                "Major restructuring recommended",
                "Add missing nodes and relationships",
                "Improve trust score quality",
                "Increase semantic coverage",
            ]

        return {
            "quality_level": quality_level,
            "score_interpretation": f"{overall_score:.1%} complete",
            "recommendations": recommendations,
        }
