"""
GraphFixer Facade

Provides backward compatibility and unified interface for the decomposed
GraphFixer services. Coordinates service interactions and maintains
the original API contract.

This facade ensures that existing code continues to work while
benefiting from the improved service architecture.
"""

import time
from typing import Any

from ..graph_fixer import DetectedGap, GapAnalysisResult, ProposedNode, ProposedRelationship
from ..interfaces.base_service import ServiceConfig
from ..services.confidence_calculator_service import ConfidenceCalculatorService
from ..services.gap_detection_service import GapDetectionService
from ..services.graph_analytics_service import GraphAnalyticsService
from ..services.knowledge_validator_service import KnowledgeValidatorService
from ..services.node_proposal_service import NodeProposalService
from ..services.relationship_analyzer_service import RelationshipAnalyzerService


class GraphFixerFacade:
    """
    Facade for the decomposed GraphFixer services.

    Maintains the original GraphFixer API while delegating to specialized
    services. Provides service coordination, caching, and unified error handling.

    Benefits of this facade:
    - Backward compatibility with existing code
    - Simplified service coordination
    - Unified configuration and initialization
    - Consistent error handling and logging
    - Performance optimizations through service coordination
    """

    def __init__(
        self,
        trust_graph=None,
        vector_engine=None,
        min_confidence_threshold: float = 0.3,
        max_proposals_per_gap: int = 3,
    ):
        # Create unified configuration
        self.config = ServiceConfig(
            trust_graph=trust_graph,
            vector_engine=vector_engine,
            min_confidence_threshold=min_confidence_threshold,
            max_proposals_per_gap=max_proposals_per_gap,
            cache_enabled=True,
            logging_level="INFO",
        )

        # Initialize services
        self.gap_detection = GapDetectionService(self.config)
        self.node_proposal = NodeProposalService(self.config)
        self.relationship_analyzer = RelationshipAnalyzerService(self.config)
        self.confidence_calculator = ConfidenceCalculatorService(self.config)
        self.graph_analytics = GraphAnalyticsService(self.config)
        self.knowledge_validator = KnowledgeValidatorService(self.config)

        # Service coordination state
        self.initialized = False
        self.stats = {"facade_calls": 0, "service_errors": 0, "total_analysis_time": 0.0}

        # Compatibility attributes (maintain original interface)
        self.trust_graph = trust_graph
        self.vector_engine = vector_engine
        self.min_confidence_threshold = min_confidence_threshold
        self.max_proposals_per_gap = max_proposals_per_gap

    async def initialize(self):
        """Initialize all services and facade."""
        if self.initialized:
            return

        try:
            # Initialize all services concurrently
            initialization_tasks = [
                self.gap_detection.initialize(),
                self.node_proposal.initialize(),
                self.relationship_analyzer.initialize(),
                self.confidence_calculator.initialize(),
                self.graph_analytics.initialize(),
                self.knowledge_validator.initialize(),
            ]

            await asyncio.gather(*initialization_tasks, return_exceptions=True)

            self.initialized = True

            # Get service availability summary
            available_services = sum(
                [
                    self.gap_detection.is_initialized,
                    self.node_proposal.is_initialized,
                    self.relationship_analyzer.is_initialized,
                    self.confidence_calculator.is_initialized,
                    self.graph_analytics.is_initialized,
                    self.knowledge_validator.is_initialized,
                ]
            )

            print(f"🔧 GraphFixer ready: {available_services}/6 services initialized")

        except Exception as e:
            print(f"GraphFixer initialization failed: {e}")
            raise

    async def detect_knowledge_gaps(
        self, query: str | None = None, retrieved_info: list[Any] | None = None, focus_area: str | None = None
    ) -> list[DetectedGap]:
        """
        Detect knowledge gaps in the graph (original API compatibility).

        Delegates to GapDetectionService with enhanced coordination.
        """
        await self._ensure_initialized()
        self.stats["facade_calls"] += 1

        try:
            gaps = await self.gap_detection.detect_gaps(query, retrieved_info, focus_area)

            # Enhance gaps with validation
            for gap in gaps:
                is_valid = await self.knowledge_validator.verify_logic(gap)
                if not is_valid:
                    gap.confidence *= 0.5  # Reduce confidence for invalid gaps

            return gaps

        except Exception as e:
            self.stats["service_errors"] += 1
            print(f"Gap detection failed: {e}")
            return []

    async def propose_solutions(
        self, gaps: list[DetectedGap], max_proposals: int | None = None
    ) -> tuple[list[ProposedNode], list[ProposedRelationship]]:
        """
        Propose solutions for detected gaps (original API compatibility).

        Coordinates NodeProposalService and RelationshipAnalyzerService.
        """
        await self._ensure_initialized()
        self.stats["facade_calls"] += 1

        try:
            # Limit gaps if specified
            limited_gaps = gaps[:max_proposals] if max_proposals else gaps

            # Generate proposals concurrently
            node_task = self.node_proposal.propose_nodes(limited_gaps)
            relationship_task = self.relationship_analyzer.propose_relationships(limited_gaps)

            proposed_nodes, proposed_relationships = await asyncio.gather(node_task, relationship_task)

            # Enhance proposals with confidence calculation
            await self._enhance_proposals_with_confidence(proposed_nodes, proposed_relationships, limited_gaps)

            # Validate proposals
            await self._validate_proposals(proposed_nodes + proposed_relationships)

            return proposed_nodes, proposed_relationships

        except Exception as e:
            self.stats["service_errors"] += 1
            print(f"Solution proposal failed: {e}")
            return [], []

    async def validate_proposal(
        self, proposal: ProposedNode | ProposedRelationship, validation_feedback: str, is_accepted: bool
    ) -> bool:
        """
        Validate a proposal (original API compatibility).

        Enhanced with learning from KnowledgeValidatorService.
        """
        await self._ensure_initialized()

        try:
            # Update proposal status
            if is_accepted:
                proposal.validation_status = "validated"
            else:
                proposal.validation_status = "rejected"

            proposal.validation_feedback = validation_feedback

            # Learn from validation
            await self.knowledge_validator.learn_from_validation(proposal, is_accepted)

            # Update confidence calculator's validation history
            if hasattr(self.confidence_calculator, "update_validation_history"):
                await self.confidence_calculator.update_validation_history(proposal, is_accepted)

            return True

        except Exception as e:
            self.stats["service_errors"] += 1
            print(f"Proposal validation failed: {e}")
            return False

    async def analyze_graph_completeness(self) -> dict[str, Any]:
        """
        Analyze graph completeness (original API compatibility).

        Enhanced with GraphAnalyticsService capabilities.
        """
        await self._ensure_initialized()
        self.stats["facade_calls"] += 1

        try:
            return await self.graph_analytics.analyze_completeness()

        except Exception as e:
            self.stats["service_errors"] += 1
            print(f"Graph completeness analysis failed: {e}")
            return {"error": str(e)}

    async def get_gap_statistics(self) -> dict[str, Any]:
        """
        Get gap statistics (original API compatibility).

        Aggregates statistics from all services.
        """
        await self._ensure_initialized()

        try:
            # Collect statistics from all services
            stats = {
                "facade": self.stats,
                "gap_detection": self.gap_detection.get_statistics(),
                "node_proposal": self.node_proposal.get_statistics(),
                "relationship_analyzer": self.relationship_analyzer.get_statistics(),
                "confidence_calculator": self.confidence_calculator.get_statistics(),
                "graph_analytics": self.graph_analytics.get_statistics(),
                "knowledge_validator": self.knowledge_validator.get_statistics(),
            }

            # Calculate aggregate metrics
            stats["aggregate"] = {
                "total_gaps_detected": stats["gap_detection"]["gaps_detected"],
                "total_proposals": (
                    stats["node_proposal"]["nodes_proposed"] + stats["relationship_analyzer"]["relationships_proposed"]
                ),
                "avg_confidence": stats["confidence_calculator"]["avg_confidence"],
                "validations_performed": stats["knowledge_validator"]["validations_performed"],
            }

            return stats

        except Exception as e:
            print(f"Statistics gathering failed: {e}")
            return {"error": str(e)}

    async def perform_comprehensive_analysis(
        self, query: str | None = None, retrieved_info: list[Any] | None = None, focus_area: str | None = None
    ) -> GapAnalysisResult:
        """
        Perform comprehensive gap analysis (enhanced API).

        Coordinates all services for complete analysis workflow.
        """
        await self._ensure_initialized()
        start_time = time.time()

        try:
            # Step 1: Detect gaps
            gaps = await self.detect_knowledge_gaps(query, retrieved_info, focus_area)

            # Step 2: Propose solutions
            proposed_nodes, proposed_relationships = await self.propose_solutions(gaps)

            # Step 3: Calculate analytics
            analytics = await self.graph_analytics.analyze_completeness()

            # Step 4: Create result
            analysis_time = (time.time() - start_time) * 1000
            self.stats["total_analysis_time"] += analysis_time

            result = GapAnalysisResult(
                gaps_detected=gaps,
                proposed_nodes=proposed_nodes,
                proposed_relationships=proposed_relationships,
                analysis_time_ms=analysis_time,
                total_gaps_found=len(gaps),
                total_proposals=len(proposed_nodes) + len(proposed_relationships),
                avg_gap_confidence=sum(g.confidence for g in gaps) / len(gaps) if gaps else 0.0,
                avg_proposal_confidence=(
                    (
                        sum(p.confidence for p in proposed_nodes + proposed_relationships)
                        / len(proposed_nodes + proposed_relationships)
                    )
                    if (proposed_nodes or proposed_relationships)
                    else 0.0
                ),
                coverage_improvement=analytics.get("overall_completeness", 0.0),
                nodes_analyzed=len(self.config.trust_graph.nodes) if self.config.trust_graph else 0,
                relationships_analyzed=len(self.config.trust_graph.edges) if self.config.trust_graph else 0,
                metadata={"services_used": 6, "analytics": analytics, "service_stats": await self.get_gap_statistics()},
            )

            return result

        except Exception as e:
            self.stats["service_errors"] += 1
            print(f"Comprehensive analysis failed: {e}")

            # Return error result
            return GapAnalysisResult(
                gaps_detected=[],
                proposed_nodes=[],
                proposed_relationships=[],
                analysis_time_ms=(time.time() - start_time) * 1000,
                metadata={"error": str(e)},
            )

    # Service coordination methods

    async def _ensure_initialized(self):
        """Ensure facade and all services are initialized."""
        if not self.initialized:
            await self.initialize()

    async def _enhance_proposals_with_confidence(
        self,
        proposed_nodes: list[ProposedNode],
        proposed_relationships: list[ProposedRelationship],
        gaps: list[DetectedGap],
    ):
        """Enhance proposals with refined confidence scores."""
        try:
            # Create gap lookup
            gap_lookup = {gap.id: gap for gap in gaps}

            # Enhance node proposals
            for proposal in proposed_nodes:
                if proposal.gap_id in gap_lookup:
                    gap = gap_lookup[proposal.gap_id]
                    evidence = gap.evidence + [f"Proposal reasoning: {proposal.reasoning}"]

                    enhanced_confidence = await self.confidence_calculator.calculate_confidence(proposal, gap, evidence)
                    proposal.confidence = enhanced_confidence

            # Enhance relationship proposals
            for proposal in proposed_relationships:
                if proposal.gap_id in gap_lookup:
                    gap = gap_lookup[proposal.gap_id]
                    evidence = gap.evidence + proposal.evidence_sources

                    enhanced_confidence = await self.confidence_calculator.calculate_confidence(proposal, gap, evidence)
                    proposal.confidence = enhanced_confidence

        except Exception as e:
            print(f"Confidence enhancement failed: {e}")

    async def _validate_proposals(self, proposals: list[ProposedNode | ProposedRelationship]):
        """Validate proposals and update their validation status."""
        try:
            validation_results = await self.knowledge_validator.validate_consistency(proposals)

            for proposal in proposals:
                is_valid = validation_results.get(proposal.id, False)

                if not is_valid:
                    # Check for conflicts
                    conflicts = await self.knowledge_validator.check_conflicts(proposal)
                    proposal.validation_feedback = f"Validation issues: {'; '.join(conflicts)}"
                    proposal.confidence *= 0.7  # Reduce confidence for validation issues

        except Exception as e:
            print(f"Proposal validation failed: {e}")

    async def cleanup(self):
        """Clean up all services and facade resources."""
        if not self.initialized:
            return

        try:
            cleanup_tasks = [
                service.cleanup()
                for service in [
                    self.gap_detection,
                    self.node_proposal,
                    self.relationship_analyzer,
                    self.confidence_calculator,
                    self.graph_analytics,
                    self.knowledge_validator,
                ]
            ]

            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            self.initialized = False

        except Exception as e:
            print(f"Cleanup failed: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()


# Import asyncio at the top of the file
import asyncio
