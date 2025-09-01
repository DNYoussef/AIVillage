"""
Knowledge Validator Service

Responsible for validating knowledge consistency, detecting conflicts,
and learning from validation feedback to improve future proposals.

Extracted from GraphFixer to follow single responsibility principle.
"""

from typing import Any, Dict, List, Union, Set
from datetime import datetime, timedelta

from ..graph_fixer import DetectedGap, ProposedNode, ProposedRelationship, GapType
from ..interfaces.service_interfaces import IKnowledgeValidatorService
from ..interfaces.base_service import ServiceConfig, CacheableMixin, AsyncServiceMixin


class KnowledgeValidatorService(IKnowledgeValidatorService, CacheableMixin, AsyncServiceMixin):
    """
    Service for validating knowledge consistency and learning from feedback.

    Capabilities:
    - Consistency validation against existing knowledge
    - Conflict detection and resolution
    - Logic verification for gaps and proposals
    - Learning from validation feedback
    - Pattern recognition for successful proposals
    """

    def __init__(self, config: ServiceConfig):
        super().__init__(config)
        self.validation_rules = [
            "no_self_references",
            "no_circular_dependencies",
            "trust_score_consistency",
            "semantic_coherence",
            "structural_validity",
        ]
        self.learning_data = {"successful_patterns": [], "failed_patterns": [], "validation_history": []}
        self.stats = {
            "validations_performed": 0,
            "conflicts_detected": 0,
            "consistency_checks": 0,
            "learning_updates": 0,
        }

    async def initialize(self) -> bool:
        """Initialize knowledge validator service."""
        self.logger.info("Initializing KnowledgeValidatorService...")

        # Load historical validation data
        await self._load_learning_data()

        self._initialized = True
        self.logger.info("✓ KnowledgeValidatorService initialized")
        return True

    async def cleanup(self) -> None:
        """Clean up service resources."""
        # Save learning data
        await self._save_learning_data()
        self.clear_cache()
        self._initialized = False

    async def validate_consistency(self, proposals: List[Union[ProposedNode, ProposedRelationship]]) -> Dict[str, bool]:
        """
        Validate consistency of proposals with existing knowledge.

        Args:
            proposals: List of proposed nodes or relationships

        Returns:
            Dictionary mapping proposal IDs to validation results
        """
        if not self.is_initialized:
            await self.initialize()

        results = {}

        try:
            for proposal in proposals:
                proposal_id = proposal.id
                is_consistent = True

                # Run consistency checks
                for rule in self.validation_rules:
                    if not await self._apply_validation_rule(proposal, rule):
                        is_consistent = False
                        self.logger.debug(f"Proposal {proposal_id} failed rule: {rule}")
                        break

                # Check against existing knowledge
                if is_consistent:
                    is_consistent = await self._validate_against_existing_knowledge(proposal)

                results[proposal_id] = is_consistent
                self.stats["consistency_checks"] += 1

            self.stats["validations_performed"] += 1

            valid_count = sum(1 for is_valid in results.values() if is_valid)
            self.logger.info(f"Validated {len(proposals)} proposals: {valid_count} consistent")

            return results

        except Exception as e:
            self.logger.exception(f"Consistency validation failed: {e}")
            return {proposal.id: False for proposal in proposals}

    async def check_conflicts(self, proposal: Union[ProposedNode, ProposedRelationship]) -> List[str]:
        """
        Check for conflicts with existing knowledge.

        Args:
            proposal: Proposed node or relationship to check

        Returns:
            List of conflict descriptions (empty if no conflicts)
        """
        conflicts = []

        try:
            if isinstance(proposal, ProposedNode):
                conflicts.extend(await self._check_node_conflicts(proposal))
            else:
                conflicts.extend(await self._check_relationship_conflicts(proposal))

            self.stats["conflicts_detected"] += len(conflicts)

            return conflicts

        except Exception as e:
            self.logger.exception(f"Conflict checking failed: {e}")
            return ["Conflict checking error occurred"]

    async def verify_logic(self, gap: DetectedGap) -> bool:
        """
        Verify the logical validity of a detected gap.

        Args:
            gap: Gap to verify

        Returns:
            True if gap logic is valid, False otherwise
        """
        try:
            # Basic logical consistency checks
            logic_checks = [
                await self._verify_gap_evidence(gap),
                await self._verify_gap_context(gap),
                await self._verify_gap_priority(gap),
                await self._verify_gap_type_consistency(gap),
            ]

            is_valid = all(logic_checks)

            if not is_valid:
                self.logger.debug(f"Gap {gap.id} failed logic verification")

            return is_valid

        except Exception as e:
            self.logger.exception(f"Gap logic verification failed: {e}")
            return False

    async def learn_from_validation(
        self, proposal: Union[ProposedNode, ProposedRelationship], is_accepted: bool
    ) -> None:
        """
        Learn from validation feedback to improve future proposals.

        Args:
            proposal: The validated proposal
            is_accepted: Whether the proposal was accepted
        """
        try:
            # Extract patterns from the proposal
            pattern = await self._extract_proposal_pattern(proposal)

            # Store learning data
            if is_accepted:
                self.learning_data["successful_patterns"].append(
                    {
                        "pattern": pattern,
                        "timestamp": datetime.now(),
                        "proposal_type": "node" if isinstance(proposal, ProposedNode) else "relationship",
                    }
                )
            else:
                self.learning_data["failed_patterns"].append(
                    {
                        "pattern": pattern,
                        "timestamp": datetime.now(),
                        "proposal_type": "node" if isinstance(proposal, ProposedNode) else "relationship",
                        "rejection_reason": proposal.validation_feedback,
                    }
                )

            # Update validation history
            self.learning_data["validation_history"].append(
                {
                    "proposal_id": proposal.id,
                    "is_accepted": is_accepted,
                    "timestamp": datetime.now(),
                    "confidence": proposal.confidence,
                    "utility_score": proposal.utility_score,
                }
            )

            # Prune old data to keep memory manageable
            await self._prune_learning_data()

            self.stats["learning_updates"] += 1

            self.logger.info(f"Learning update: proposal {'accepted' if is_accepted else 'rejected'}")

        except Exception as e:
            self.logger.exception(f"Learning from validation failed: {e}")

    # Private implementation methods

    async def _apply_validation_rule(self, proposal: Union[ProposedNode, ProposedRelationship], rule: str) -> bool:
        """Apply a specific validation rule to a proposal."""
        try:
            if rule == "no_self_references":
                return await self._check_no_self_references(proposal)
            elif rule == "no_circular_dependencies":
                return await self._check_no_circular_dependencies(proposal)
            elif rule == "trust_score_consistency":
                return await self._check_trust_score_consistency(proposal)
            elif rule == "semantic_coherence":
                return await self._check_semantic_coherence(proposal)
            elif rule == "structural_validity":
                return await self._check_structural_validity(proposal)
            else:
                self.logger.warning(f"Unknown validation rule: {rule}")
                return True

        except Exception as e:
            self.logger.exception(f"Validation rule {rule} failed: {e}")
            return False

    async def _check_no_self_references(self, proposal: Union[ProposedNode, ProposedRelationship]) -> bool:
        """Check that proposals don't create self-references."""
        if isinstance(proposal, ProposedRelationship):
            return proposal.source_id != proposal.target_id
        return True  # Nodes can't self-reference

    async def _check_no_circular_dependencies(self, proposal: Union[ProposedNode, ProposedRelationship]) -> bool:
        """Check for circular dependencies (simplified)."""
        if isinstance(proposal, ProposedRelationship):
            # Check if adding this relationship would create a cycle
            return await self._would_create_cycle(proposal)
        return True

    async def _would_create_cycle(self, proposal: ProposedRelationship) -> bool:
        """Check if a relationship proposal would create a cycle."""
        if not self.config.trust_graph:
            return True  # Allow if we can't check

        # Simple cycle detection using DFS
        # Check if there's already a path from target to source
        visited = set()
        return not await self._has_path_dfs(proposal.target_id, proposal.source_id, visited)

    async def _has_path_dfs(self, start: str, target: str, visited: Set[str]) -> bool:
        """DFS to check if path exists between nodes."""
        if start == target:
            return True

        if start in visited or start not in self.config.trust_graph.nodes:
            return False

        visited.add(start)
        node = self.config.trust_graph.nodes[start]

        # Check outgoing edges
        for edge_id in node.outgoing_edges:
            if edge_id in self.config.trust_graph.edges:
                edge = self.config.trust_graph.edges[edge_id]
                if await self._has_path_dfs(edge.target_id, target, visited):
                    return True

        return False

    async def _check_trust_score_consistency(self, proposal: Union[ProposedNode, ProposedRelationship]) -> bool:
        """Check trust score consistency."""
        if isinstance(proposal, ProposedNode):
            # Trust scores should be reasonable
            return 0.0 <= proposal.suggested_trust_score <= 1.0
        else:
            # Relationship strength should be reasonable
            return 0.0 <= proposal.relation_strength <= 1.0

    async def _check_semantic_coherence(self, proposal: Union[ProposedNode, ProposedRelationship]) -> bool:
        """Check semantic coherence of proposal."""
        # Basic checks for content quality
        if isinstance(proposal, ProposedNode):
            return proposal.content and proposal.concept and len(proposal.content.strip()) > 5
        else:
            return proposal.relation_type and len(proposal.relation_type.strip()) > 0

    async def _check_structural_validity(self, proposal: Union[ProposedNode, ProposedRelationship]) -> bool:
        """Check structural validity of proposal."""
        # Check required fields and value ranges
        if isinstance(proposal, ProposedNode):
            return (
                0.0 <= proposal.existence_probability <= 1.0
                and 0.0 <= proposal.utility_score <= 1.0
                and 0.0 <= proposal.confidence <= 1.0
            )
        else:
            return (
                proposal.source_id
                and proposal.target_id
                and 0.0 <= proposal.existence_probability <= 1.0
                and 0.0 <= proposal.utility_score <= 1.0
                and 0.0 <= proposal.confidence <= 1.0
            )

    async def _validate_against_existing_knowledge(self, proposal: Union[ProposedNode, ProposedRelationship]) -> bool:
        """Validate proposal against existing knowledge in the graph."""
        if not self.config.trust_graph:
            return True  # Can't validate without graph

        try:
            if isinstance(proposal, ProposedNode):
                return await self._validate_node_against_graph(proposal)
            else:
                return await self._validate_relationship_against_graph(proposal)

        except Exception as e:
            self.logger.exception(f"Knowledge validation failed: {e}")
            return False

    async def _validate_node_against_graph(self, proposal: ProposedNode) -> bool:
        """Validate node proposal against existing graph knowledge."""
        # Check for concept duplicates
        for existing_node in self.config.trust_graph.nodes.values():
            if hasattr(existing_node, "concept") and existing_node.concept == proposal.concept:
                return False  # Duplicate concept

        # Check if suggested relationships reference valid nodes
        if proposal.suggested_relationships:
            for rel in proposal.suggested_relationships:
                target_concept = rel.get("target_concept")
                if target_concept:
                    # Check if target concept exists
                    exists = any(
                        hasattr(node, "concept") and node.concept == target_concept
                        for node in self.config.trust_graph.nodes.values()
                    )
                    if not exists:
                        return False

        return True

    async def _validate_relationship_against_graph(self, proposal: ProposedRelationship) -> bool:
        """Validate relationship proposal against existing graph knowledge."""
        # Check if source and target nodes exist
        if (
            proposal.source_id not in self.config.trust_graph.nodes
            or proposal.target_id not in self.config.trust_graph.nodes
        ):
            return False

        # Check if relationship already exists
        for edge in self.config.trust_graph.edges.values():
            if edge.source_id == proposal.source_id and edge.target_id == proposal.target_id:
                return False  # Relationship already exists

        return True

    async def _check_node_conflicts(self, proposal: ProposedNode) -> List[str]:
        """Check for conflicts specific to node proposals."""
        conflicts = []

        if not self.config.trust_graph:
            return conflicts

        # Check for concept conflicts
        for node in self.config.trust_graph.nodes.values():
            if hasattr(node, "concept") and node.concept == proposal.concept:
                conflicts.append(f"Concept '{proposal.concept}' already exists")

        # Check for content similarity conflicts
        similar_content = await self._find_similar_content(proposal.content)
        if similar_content:
            conflicts.append(f"Similar content already exists: {similar_content[:100]}...")

        return conflicts

    async def _check_relationship_conflicts(self, proposal: ProposedRelationship) -> List[str]:
        """Check for conflicts specific to relationship proposals."""
        conflicts = []

        if not self.config.trust_graph:
            return conflicts

        # Check for existing relationships
        for edge in self.config.trust_graph.edges.values():
            if edge.source_id == proposal.source_id and edge.target_id == proposal.target_id:
                conflicts.append(f"Relationship already exists between {proposal.source_id} and {proposal.target_id}")

        # Check for conflicting relationship types
        opposite_relationship = await self._find_opposite_relationship(proposal)
        if opposite_relationship:
            conflicts.append(f"Conflicting relationship type with existing: {opposite_relationship}")

        return conflicts

    async def _find_similar_content(self, content: str) -> str:
        """Find similar content in existing nodes."""
        # Simple similarity check
        content_words = set(content.lower().split())

        for node in self.config.trust_graph.nodes.values():
            if hasattr(node, "content") and node.content:
                node_words = set(node.content.lower().split())
                overlap = len(content_words.intersection(node_words))
                total_words = len(content_words.union(node_words))

                if total_words > 0 and overlap / total_words > 0.8:  # 80% similarity
                    return node.content

        return ""

    async def _find_opposite_relationship(self, proposal: ProposedRelationship) -> str:
        """Find conflicting relationship types."""
        # Define conflicting relationship types
        conflicts = {
            "causal": ["temporal"],  # Causal relationships shouldn't be just temporal
            "hierarchical": ["associative"],  # Clear hierarchy vs loose association
        }

        for edge in self.config.trust_graph.edges.values():
            if edge.source_id == proposal.source_id and edge.target_id == proposal.target_id:

                edge_type = getattr(edge, "relation_type", "unknown")
                proposed_type = proposal.relation_type

                # Check for conflicts
                if (proposed_type in conflicts and edge_type in conflicts[proposed_type]) or (
                    edge_type in conflicts and proposed_type in conflicts[edge_type]
                ):
                    return edge_type

        return ""

    async def _verify_gap_evidence(self, gap: DetectedGap) -> bool:
        """Verify that gap evidence is logical and sufficient."""
        if not gap.evidence:
            return False  # Need some evidence

        # Check evidence quality
        for evidence in gap.evidence:
            if len(evidence.strip()) < 10:  # Too short
                return False

        return True

    async def _verify_gap_context(self, gap: DetectedGap) -> bool:
        """Verify gap context makes sense."""
        # Basic context checks
        if gap.gap_type == GapType.MISSING_RELATIONSHIP and len(gap.source_nodes) < 2:
            return False  # Need at least 2 nodes for missing relationship

        if gap.gap_type == GapType.ISOLATED_CLUSTER and not gap.source_nodes:
            return False  # Need nodes to identify isolation

        return True

    async def _verify_gap_priority(self, gap: DetectedGap) -> bool:
        """Verify gap priority is reasonable."""
        return 0.0 <= gap.priority <= 1.0 and 0.0 <= gap.confidence <= 1.0

    async def _verify_gap_type_consistency(self, gap: DetectedGap) -> bool:
        """Verify gap type matches its characteristics."""
        # Check that gap type aligns with evidence and context
        evidence_text = " ".join(gap.evidence).lower()

        if gap.gap_type == GapType.MISSING_NODE and "node" not in evidence_text:
            if "concept" not in evidence_text and "missing" not in evidence_text:
                return False

        if gap.gap_type == GapType.MISSING_RELATIONSHIP:
            if "relationship" not in evidence_text and "connection" not in evidence_text:
                return False

        return True

    async def _extract_proposal_pattern(self, proposal: Union[ProposedNode, ProposedRelationship]) -> Dict[str, Any]:
        """Extract pattern from proposal for learning."""
        if isinstance(proposal, ProposedNode):
            return {
                "type": "node",
                "existence_probability": proposal.existence_probability,
                "utility_score": proposal.utility_score,
                "confidence": proposal.confidence,
                "has_relationships": len(proposal.suggested_relationships) > 0,
                "content_length": len(proposal.content) if proposal.content else 0,
            }
        else:
            return {
                "type": "relationship",
                "relation_type": proposal.relation_type,
                "relation_strength": proposal.relation_strength,
                "existence_probability": proposal.existence_probability,
                "utility_score": proposal.utility_score,
                "confidence": proposal.confidence,
                "evidence_count": len(proposal.evidence_sources),
            }

    async def _load_learning_data(self) -> None:
        """Load historical learning data."""
        # In practice, would load from persistent storage
        # For now, initialize empty structures
        self.learning_data = {"successful_patterns": [], "failed_patterns": [], "validation_history": []}

    async def _save_learning_data(self) -> None:
        """Save learning data to persistent storage."""
        # In practice, would save to database or file
        pass

    async def _prune_learning_data(self, max_age_days: int = 90) -> None:
        """Prune old learning data to keep memory manageable."""
        cutoff_date = datetime.now() - timedelta(days=max_age_days)

        # Prune successful patterns
        self.learning_data["successful_patterns"] = [
            pattern for pattern in self.learning_data["successful_patterns"] if pattern["timestamp"] > cutoff_date
        ]

        # Prune failed patterns
        self.learning_data["failed_patterns"] = [
            pattern for pattern in self.learning_data["failed_patterns"] if pattern["timestamp"] > cutoff_date
        ]

        # Prune validation history
        self.learning_data["validation_history"] = [
            entry for entry in self.learning_data["validation_history"] if entry["timestamp"] > cutoff_date
        ]

        # Keep maximum number of entries
        max_entries = 1000
        for key in ["successful_patterns", "failed_patterns", "validation_history"]:
            if len(self.learning_data[key]) > max_entries:
                self.learning_data[key] = self.learning_data[key][-max_entries:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            "validations_performed": self.stats["validations_performed"],
            "conflicts_detected": self.stats["conflicts_detected"],
            "consistency_checks": self.stats["consistency_checks"],
            "learning_updates": self.stats["learning_updates"],
            "validation_rules": self.validation_rules,
            "learning_data_size": {
                "successful_patterns": len(self.learning_data["successful_patterns"]),
                "failed_patterns": len(self.learning_data["failed_patterns"]),
                "validation_history": len(self.learning_data["validation_history"]),
            },
        }
