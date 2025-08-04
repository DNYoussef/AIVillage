"""Memory Consolidation System.

Brain-inspired memory consolidation mimicking hippocampal-neocortical transfer.
Consolidates episodic memories into semantic knowledge during 'sleep' cycles.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import time
from typing import Any

import numpy as np

from AIVillage.src.mcp_servers.hyperag.guardian.gate import GuardianGate

from .base import (
    ConsolidationBatch,
    Edge,
    Node,
)
from .hippo_index import HippoIndex
from .hypergraph_kg import (
    HypergraphKG,
    create_hyperedge,
    create_semantic_node,
)

logger = logging.getLogger(__name__)


@dataclass
class ConsolidationConfig:
    """Configuration for memory consolidation process."""

    # Thresholds for consolidation
    confidence_threshold: float = 0.7  # Min confidence to consolidate
    evidence_threshold: int = 3  # Min evidence count for episodic edges
    frequency_threshold: int = 2  # Min occurrence count for patterns

    # Timing parameters
    consolidation_interval_hours: int = 8  # How often to run consolidation
    max_age_hours: int = 24  # Max age of episodic memories to consider
    batch_size: int = 100  # Max items per consolidation batch

    # Pattern detection
    enable_pattern_detection: bool = True
    min_pattern_support: float = 0.3  # Min support for frequent patterns
    max_pattern_length: int = 5  # Max length of sequential patterns

    # Community-based consolidation
    enable_community_consolidation: bool = True
    community_threshold: float = 0.5  # Min community coherence

    # Uncertainty handling
    uncertainty_penalty: float = 0.2  # Penalty for high uncertainty
    max_uncertainty: float = 0.8  # Max uncertainty to allow consolidation

    # Personalization
    user_specific_consolidation: bool = True
    cross_user_threshold: float = 0.9  # Min confidence for cross-user patterns


@dataclass
class ConsolidationResult:
    """Result of a consolidation operation."""

    batch_id: str
    processed_nodes: int = 0
    processed_edges: int = 0
    created_semantic_nodes: int = 0
    created_hyperedges: int = 0
    merged_duplicates: int = 0
    rejected_low_confidence: int = 0
    processing_time_ms: float = 0.0
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class PatternDetector:
    """Detects patterns in episodic memory for consolidation."""

    def __init__(self, config: ConsolidationConfig) -> None:
        self.config = config

    async def detect_frequent_relations(
        self, edges: list[Edge], min_support: float | None = None
    ) -> list[dict[str, Any]]:
        """Detect frequently occurring relation patterns."""
        if min_support is None:
            min_support = self.config.min_pattern_support

        # Count relation frequencies
        relation_counts = {}
        total_edges = len(edges)

        for edge in edges:
            key = (edge.relation, edge.source_id, edge.target_id)
            if key not in relation_counts:
                relation_counts[key] = {
                    "count": 0,
                    "confidence_sum": 0.0,
                    "evidence_sum": 0,
                    "edges": [],
                }

            relation_counts[key]["count"] += 1
            relation_counts[key]["confidence_sum"] += edge.confidence
            relation_counts[key]["evidence_sum"] += edge.evidence_count
            relation_counts[key]["edges"].append(edge)

        # Filter by support threshold
        frequent_patterns = []
        for (relation, source, target), data in relation_counts.items():
            support = data["count"] / total_edges if total_edges > 0 else 0

            if (
                support >= min_support
                and data["count"] >= self.config.frequency_threshold
            ):
                avg_confidence = data["confidence_sum"] / data["count"]
                total_evidence = data["evidence_sum"]

                frequent_patterns.append(
                    {
                        "relation": relation,
                        "source_id": source,
                        "target_id": target,
                        "frequency": data["count"],
                        "support": support,
                        "avg_confidence": avg_confidence,
                        "total_evidence": total_evidence,
                        "contributing_edges": [e.id for e in data["edges"]],
                    }
                )

        # Sort by frequency and confidence
        frequent_patterns.sort(
            key=lambda x: (x["frequency"], x["avg_confidence"]), reverse=True
        )

        logger.debug(f"Detected {len(frequent_patterns)} frequent relation patterns")
        return frequent_patterns

    async def detect_concept_clusters(
        self, nodes: list[Node], similarity_threshold: float = 0.8
    ) -> list[list[str]]:
        """Detect clusters of similar concept nodes for merging."""
        if not nodes:
            return []

        clusters = []
        processed = set()

        for i, node_a in enumerate(nodes):
            if node_a.id in processed:
                continue

            cluster = [node_a.id]
            processed.add(node_a.id)

            # Find similar nodes
            for _j, node_b in enumerate(nodes[i + 1 :], i + 1):
                if node_b.id in processed:
                    continue

                # Calculate similarity (content-based for now)
                if node_a.embedding is not None and node_b.embedding is not None:
                    similarity = np.dot(node_a.embedding, node_b.embedding) / (
                        np.linalg.norm(node_a.embedding)
                        * np.linalg.norm(node_b.embedding)
                    )
                else:
                    # Fallback to simple text similarity
                    similarity = self._text_similarity(node_a.content, node_b.content)

                if similarity >= similarity_threshold:
                    cluster.append(node_b.id)
                    processed.add(node_b.id)

            if len(cluster) > 1:
                clusters.append(cluster)

        logger.debug(f"Detected {len(clusters)} concept clusters")
        return clusters

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity based on common words."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0


class MemoryConsolidator:
    """Main consolidation system coordinating episodic to semantic memory transfer.

    Features:
    - Pattern-based consolidation
    - Community-aware processing
    - Uncertainty-guided decisions
    - User-specific adaptation
    - Incremental processing
    """

    def __init__(
        self,
        hippo_index: HippoIndex,
        hypergraph_kg: HypergraphKG,
        config: ConsolidationConfig = None,
        guardian_gate: GuardianGate | None = None,
    ) -> None:
        self.hippo = hippo_index
        self.semantic = hypergraph_kg
        self.config = config or ConsolidationConfig()
        self.pattern_detector = PatternDetector(self.config)
        self.guardian_gate = guardian_gate or GuardianGate()

        # Consolidation tracking
        self.last_consolidation: datetime | None = None
        self.pending_batches: asyncio.Queue[ConsolidationBatch] = asyncio.Queue()
        self.consolidation_stats: dict[str, Any] = {
            "guardian_approvals": 0,
            "guardian_quarantines": 0,
            "guardian_rejections": 0,
        }

        logger.info("MemoryConsolidator initialized with Guardian Gate")

    @property
    def pending_consolidations(self) -> int:
        """Number of consolidation batches waiting to be processed."""
        return self.pending_batches.qsize()

    async def run_consolidation_cycle(
        self, user_id: str | None = None, force: bool = False
    ) -> ConsolidationResult:
        """Run a complete consolidation cycle.

        Args:
            user_id: Specific user to consolidate (None for all users)
            force: Force consolidation even if interval hasn't passed

        Returns:
            ConsolidationResult with processing statistics
        """
        start_time = time.time()
        batch_id = f"consolidation_{int(start_time)}"

        logger.info(f"Starting consolidation cycle {batch_id}")

        try:
            # Check if consolidation is needed
            if not force and not await self._should_consolidate():
                return ConsolidationResult(
                    batch_id=batch_id, metadata={"skipped": "consolidation not needed"}
                )

            # Get episodic memories for consolidation
            candidate_nodes, candidate_edges = await self._get_consolidation_candidates(
                user_id
            )

            if not candidate_nodes and not candidate_edges:
                return ConsolidationResult(
                    batch_id=batch_id, metadata={"skipped": "no candidates found"}
                )

            result = ConsolidationResult(batch_id=batch_id)

            # Step 1: Detect patterns in episodic data
            if self.config.enable_pattern_detection:
                await self._consolidate_patterns(candidate_edges, result)

            # Step 2: Consolidate high-confidence nodes
            await self._consolidate_nodes(candidate_nodes, result)

            # Step 3: Merge similar concepts
            await self._merge_duplicates(result)

            # Step 4: Update semantic graph structure
            if self.config.enable_community_consolidation:
                await self._update_communities(result)

            # Step 5: Clean up consolidated episodic memories
            await self._cleanup_consolidated_memories(
                candidate_nodes, candidate_edges, result
            )

            # Update consolidation tracking
            self.last_consolidation = datetime.now()
            result.processing_time_ms = (time.time() - start_time) * 1000

            logger.info(
                f"Consolidation cycle {batch_id} completed: "
                f"{result.created_semantic_nodes} nodes, "
                f"{result.created_hyperedges} edges in "
                f"{result.processing_time_ms:.2f}ms"
            )

            return result

        except Exception as e:
            error_msg = f"Consolidation cycle failed: {e!s}"
            logger.exception(error_msg)

            result = ConsolidationResult(
                batch_id=batch_id,
                processing_time_ms=(time.time() - start_time) * 1000,
                errors=[error_msg],
            )
            return result

    async def schedule_consolidation(self, interval_hours: int | None = None) -> None:
        """Schedule periodic consolidation cycles.

        Args:
            interval_hours: Override default consolidation interval
        """
        interval = interval_hours or self.config.consolidation_interval_hours
        logger.info(f"Scheduling consolidation every {interval} hours")

        while True:
            try:
                await asyncio.sleep(interval * 3600)  # Convert to seconds
                result = await self.run_consolidation_cycle()

                if result.errors:
                    logger.warning(
                        f"Consolidation completed with errors: {result.errors}"
                    )
                else:
                    logger.info("Scheduled consolidation completed successfully")

            except asyncio.CancelledError:
                logger.info("Consolidation scheduling cancelled")
                break
            except Exception as e:
                logger.exception(f"Scheduled consolidation failed: {e!s}")
                # Continue scheduling despite errors
                await asyncio.sleep(3600)  # Wait 1 hour before retry

    async def get_consolidation_status(self) -> dict[str, Any]:
        """Get current consolidation status and statistics."""
        try:
            # Get memory stats from both systems
            hippo_stats = await self.hippo.get_memory_stats()
            semantic_stats = await self.semantic.get_memory_stats()

            return {
                "last_consolidation": self.last_consolidation.isoformat()
                if self.last_consolidation
                else None,
                "pending_consolidations": self.pending_consolidations,
                "consolidation_interval_hours": self.config.consolidation_interval_hours,
                "next_consolidation": (
                    self.last_consolidation
                    + timedelta(hours=self.config.consolidation_interval_hours)
                ).isoformat()
                if self.last_consolidation
                else "pending",
                "episodic_memory": {
                    "total_nodes": hippo_stats.total_nodes,
                    "total_edges": hippo_stats.total_edges,
                    "avg_confidence": hippo_stats.avg_confidence,
                },
                "semantic_memory": {
                    "total_nodes": semantic_stats.semantic_nodes,
                    "total_edges": semantic_stats.total_edges,
                    "avg_confidence": semantic_stats.avg_confidence,
                },
                "consolidation_stats": self.consolidation_stats,
                "config": {
                    "confidence_threshold": self.config.confidence_threshold,
                    "evidence_threshold": self.config.evidence_threshold,
                    "frequency_threshold": self.config.frequency_threshold,
                    "max_age_hours": self.config.max_age_hours,
                },
            }

        except Exception as e:
            logger.exception(f"Failed to get consolidation status: {e!s}")
            return {"error": str(e)}

    # Private helper methods

    async def _should_consolidate(self) -> bool:
        """Check if consolidation should run based on time and thresholds."""
        if self.last_consolidation is None:
            return True

        time_since_last = datetime.now() - self.last_consolidation
        return time_since_last.total_seconds() >= (
            self.config.consolidation_interval_hours * 3600
        )

    async def _get_consolidation_candidates(
        self, user_id: str | None = None
    ) -> tuple[list[Node], list[Edge]]:
        """Get episodic memories eligible for consolidation."""
        try:
            # Get recent episodic nodes with sufficient confidence
            candidate_nodes = await self.hippo.query_nodes(
                query="",  # Empty query to get all
                limit=self.config.batch_size,
                user_id=user_id,
                confidence_threshold=self.config.confidence_threshold,
                max_age_hours=self.config.max_age_hours,
            )

            # Filter by uncertainty threshold
            filtered_nodes = [
                node
                for node in candidate_nodes.nodes
                if node.uncertainty <= self.config.max_uncertainty
            ]

            # Get edges for consolidation (would need to implement in HippoIndex)
            # For now, return empty list
            candidate_edges = []

            logger.debug(
                f"Found {len(filtered_nodes)} candidate nodes, {len(candidate_edges)} candidate edges"
            )
            return filtered_nodes, candidate_edges

        except Exception as e:
            logger.exception(f"Failed to get consolidation candidates: {e!s}")
            return [], []

    async def _consolidate_patterns(
        self, edges: list[Edge], result: ConsolidationResult
    ) -> None:
        """Consolidate frequent patterns into semantic hyperedges."""
        try:
            frequent_patterns = await self.pattern_detector.detect_frequent_relations(
                edges
            )

            for pattern in frequent_patterns:
                # Skip patterns with insufficient evidence
                if pattern["total_evidence"] < self.config.evidence_threshold:
                    continue

                # Apply uncertainty penalty
                adjusted_confidence = pattern["avg_confidence"] * (
                    1
                    - self.config.uncertainty_penalty
                    * np.mean([0.1])  # Placeholder uncertainty calculation
                )

                if adjusted_confidence < self.config.confidence_threshold:
                    result.rejected_low_confidence += 1
                    continue

                # Create semantic hyperedge
                participants = [pattern["source_id"], pattern["target_id"]]

                hyperedge = create_hyperedge(
                    participants=participants,
                    relation=pattern["relation"],
                    confidence=adjusted_confidence,
                    evidence_count=pattern["total_evidence"],
                )

                # Add consolidation metadata
                hyperedge.consolidation_source = pattern["contributing_edges"]
                hyperedge.metadata.update(
                    {
                        "consolidation_batch": result.batch_id,
                        "frequency": pattern["frequency"],
                        "support": pattern["support"],
                        "pattern_type": "frequent_relation",
                    }
                )

                # Validate through Guardian Gate before storing
                guardian_approved = await self._validate_consolidation_with_guardian(
                    hyperedge, "pattern_consolidation", result
                )

                if guardian_approved:
                    # Store in semantic memory
                    success = await self.semantic.store_hyperedge(hyperedge)
                    if success:
                        result.created_hyperedges += 1
                        result.processed_edges += pattern["frequency"]
                    else:
                        result.errors.append(
                            f"Failed to store hyperedge for pattern {pattern['relation']}"
                        )
                else:
                    result.errors.append(
                        f"Guardian blocked hyperedge for pattern {pattern['relation']}"
                    )

        except Exception as e:
            error_msg = f"Pattern consolidation failed: {e!s}"
            logger.exception(error_msg)
            result.errors.append(error_msg)

    async def _consolidate_nodes(
        self, nodes: list[Node], result: ConsolidationResult
    ) -> None:
        """Consolidate high-confidence episodic nodes into semantic nodes."""
        try:
            for node in nodes:
                # Apply consolidation criteria
                if not self._meets_consolidation_criteria(node):
                    result.rejected_low_confidence += 1
                    continue

                # Create semantic node
                semantic_node = create_semantic_node(
                    content=node.content,
                    node_type="consolidated_concept",
                    confidence=node.confidence,
                )

                # Transfer relevant properties
                semantic_node.user_id = node.user_id
                semantic_node.importance_score = min(
                    1.0, node.importance_score * 1.2
                )  # Boost for consolidation
                semantic_node.consolidation_count = 1

                # Add consolidation metadata
                semantic_node.metadata.update(
                    {
                        "consolidation_batch": result.batch_id,
                        "episodic_source": node.id,
                        "original_created_at": node.created_at.isoformat(),
                        "consolidation_reason": "high_confidence_episodic",
                    }
                )

                # Validate through Guardian Gate before storing
                guardian_approved = await self._validate_consolidation_with_guardian(
                    semantic_node, "node_consolidation", result
                )

                if guardian_approved:
                    # Store in semantic memory
                    success = await self.semantic.store_semantic_node(semantic_node)
                    if success:
                        result.created_semantic_nodes += 1
                        result.processed_nodes += 1
                    else:
                        result.errors.append(
                            f"Failed to store semantic node for {node.id}"
                        )
                else:
                    result.errors.append(
                        f"Guardian blocked semantic node for {node.id}"
                    )

        except Exception as e:
            error_msg = f"Node consolidation failed: {e!s}"
            logger.exception(error_msg)
            result.errors.append(error_msg)

    async def _merge_duplicates(self, result: ConsolidationResult) -> None:
        """Detect and merge duplicate concepts in semantic memory."""
        try:
            # This would involve semantic similarity search and merging
            # Implementation would require semantic graph querying
            # For now, just log the intent
            logger.debug("Duplicate merging not yet implemented")

        except Exception as e:
            error_msg = f"Duplicate merging failed: {e!s}"
            logger.exception(error_msg)
            result.errors.append(error_msg)

    async def _update_communities(self, result: ConsolidationResult) -> None:
        """Update community structure after consolidation."""
        try:
            if hasattr(self.semantic, "detect_communities"):
                communities = await self.semantic.detect_communities()
                result.metadata["communities_detected"] = len(set(communities.values()))

        except Exception as e:
            error_msg = f"Community update failed: {e!s}"
            logger.exception(error_msg)
            result.errors.append(error_msg)

    async def _cleanup_consolidated_memories(
        self, nodes: list[Node], edges: list[Edge], result: ConsolidationResult
    ) -> None:
        """Clean up episodic memories that have been successfully consolidated."""
        try:
            # For now, just mark as consolidated rather than deleting
            # In production, might want to archive or soft-delete
            logger.debug(f"Would clean up {len(nodes)} nodes and {len(edges)} edges")

        except Exception as e:
            error_msg = f"Cleanup failed: {e!s}"
            logger.exception(error_msg)
            result.errors.append(error_msg)

    def _meets_consolidation_criteria(self, node: Node) -> bool:
        """Check if a node meets criteria for consolidation."""
        # Confidence threshold
        if node.confidence < self.config.confidence_threshold:
            return False

        # Uncertainty threshold
        if node.uncertainty > self.config.max_uncertainty:
            return False

        # Access frequency (nodes accessed multiple times are more likely to be important)
        if node.access_count < 2:
            return False

        # Age criteria (not too recent, not too old)
        if node.created_at:
            age_hours = (datetime.now() - node.created_at).total_seconds() / 3600
            if age_hours < 1 or age_hours > self.config.max_age_hours:
                return False

        return True

    async def _validate_consolidation_with_guardian(
        self, item: Any, consolidation_type: str, result: ConsolidationResult
    ) -> bool:
        """Validate consolidation items through Guardian Gate.

        Args:
            item: The hyperedge or semantic node to validate
            consolidation_type: Type of consolidation ("pattern_consolidation" or "node_consolidation")
            result: ConsolidationResult to update with Guardian stats

        Returns:
            Boolean indicating if Guardian approved the consolidation
        """
        try:
            # Create a creative bridge for Guardian validation
            from hyperag.guardian.gate import CreativeBridge

            # Extract relevant information based on item type
            if hasattr(item, "relation"):  # Hyperedge
                bridge_id = f"consolidation_{consolidation_type}_{item.relation}_{hash(str(item.participants)) % 10000}"
                confidence = getattr(item, "confidence", 0.7)
            else:  # Semantic node
                bridge_id = (
                    f"consolidation_{consolidation_type}_{hash(item.content) % 10000}"
                )
                confidence = getattr(item, "confidence", 0.7)

            bridge = CreativeBridge(
                id=bridge_id, confidence=confidence, bridge_type=consolidation_type
            )

            # Add consolidation metadata to bridge if available
            if hasattr(bridge, "metadata"):
                bridge.metadata = {
                    "consolidation_type": consolidation_type,
                    "item_type": type(item).__name__,
                    "consolidation_batch": result.batch_id,
                }

            # Validate through Guardian
            decision = await self.guardian_gate.evaluate_creative(bridge)

            # Update statistics
            if decision == "APPLY":
                self.consolidation_stats["guardian_approvals"] += 1
                logger.debug(f"Guardian approved {consolidation_type} for {bridge_id}")
                return True
            if decision == "QUARANTINE":
                self.consolidation_stats["guardian_quarantines"] += 1
                logger.warning(
                    f"Guardian quarantined {consolidation_type} for {bridge_id}"
                )
                return False  # Don't store quarantined items automatically
            # REJECT
            self.consolidation_stats["guardian_rejections"] += 1
            logger.warning(f"Guardian rejected {consolidation_type} for {bridge_id}")
            return False

        except Exception as e:
            logger.exception(f"Guardian validation failed during consolidation: {e}")
            # Default to allowing consolidation on Guardian error to avoid blocking system
            return True


# Factory function for creating consolidator


def create_memory_consolidator(
    hippo_index: HippoIndex,
    hypergraph_kg: HypergraphKG,
    config: ConsolidationConfig | None = None,
) -> MemoryConsolidator:
    """Create a memory consolidator with the given backends."""
    if config is None:
        config = ConsolidationConfig()

    return MemoryConsolidator(
        hippo_index=hippo_index, hypergraph_kg=hypergraph_kg, config=config
    )
