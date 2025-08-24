"""
Distributed RAG Coordinator - World's Biggest Knowledge Repository

Extends the existing HyperRAG system to support distributed, sharded knowledge storage
across the fog compute network. This system coordinates:

1. Local Mini-RAG systems (on-device personal knowledge)
2. Global Distributed RAG (sharded across fog network like torrenting)
3. Privacy-preserving knowledge elevation from local to global
4. MCP interface for Sage/Curator/King governance (2/3 quorum for changes)

Built on existing consolidated infrastructure:
- packages/rag/core/hyper_rag.py (HyperRAG orchestrator)
- packages/rag/graph/bayesian_trust_graph.py (trust networks)
- packages/edge/fog_compute/fog_coordinator.py (fog network)
- packages/agents/specialized/ (Sage, Curator, King agents)
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import time
from typing import Any

from ...edge.fog_compute.fog_coordinator import FogCoordinator
from ...edge.mobile.mini_rag_system import GlobalContribution, MiniRAGSystem
from ...p2p.core.transport_manager import TransportManager
from ..core.hyper_rag import HyperRAG, QueryMode
from ..graph.bayesian_trust_graph import GraphNode

logger = logging.getLogger(__name__)


class ShardingStrategy(Enum):
    """Strategies for distributing knowledge across fog network"""

    SEMANTIC_SHARDING = "semantic"  # Shard by topic/domain
    TRUST_BASED = "trust_based"  # Shard by trust/reliability
    GEOGRAPHIC = "geographic"  # Shard by location/region
    TEMPORAL = "temporal"  # Shard by time/recency
    HYBRID = "hybrid"  # Combination of strategies


class GovernanceDecision(Enum):
    """Types of governance decisions for RAG system"""

    MINOR_UPDATE = "minor_update"  # Individual agent can decide
    MAJOR_CHANGE = "major_change"  # Requires 2/3 quorum (Sage + Curator OR King)
    CRITICAL_CHANGE = "critical_change"  # Requires all 3 agents unanimous
    EMERGENCY_ACTION = "emergency_action"  # King override capability


@dataclass
class KnowledgeShard:
    """A shard of knowledge distributed across the fog network"""

    shard_id: str
    semantic_domain: str  # Topic/domain this shard covers
    node_assignments: dict[str, float]  # node_id -> responsibility_weight
    trust_threshold: float = 0.6
    knowledge_count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)

    # Replication and backup
    primary_nodes: list[str] = field(default_factory=list)
    backup_nodes: list[str] = field(default_factory=list)
    replication_factor: int = 3  # How many nodes store each piece


@dataclass
class GovernanceProposal:
    """Proposal for changes to the RAG system"""

    proposal_id: str
    decision_type: GovernanceDecision
    proposer_agent: str  # sage, curator, or king
    title: str
    description: str
    changes: dict[str, Any]  # Specific changes proposed

    # Voting
    votes_for: set[str] = field(default_factory=set)
    votes_against: set[str] = field(default_factory=set)
    required_votes: int = 2  # Default 2/3 majority

    # Status
    status: str = "pending"  # pending, approved, rejected, implemented
    created_at: datetime = field(default_factory=datetime.now)
    voting_deadline: datetime = None


class DistributedRAGCoordinator:
    """
    Coordinates distributed knowledge storage across fog compute network

    This system extends the existing HyperRAG to support:
    - Distributed/sharded knowledge storage like torrenting
    - Privacy-preserving local-to-global knowledge elevation
    - MCP governance interface for Sage/Curator/King agents
    - Auto-updates from Sage research jobs
    """

    def __init__(self, hyper_rag: HyperRAG, fog_coordinator: FogCoordinator, transport_manager: TransportManager):
        self.hyper_rag = hyper_rag
        self.fog_coordinator = fog_coordinator
        self.transport_manager = transport_manager

        # Distributed storage management
        self.knowledge_shards: dict[str, KnowledgeShard] = {}
        self.mini_rag_systems: dict[str, MiniRAGSystem] = {}  # device_id -> MiniRAG

        # Governance system
        self.governance_proposals: dict[str, GovernanceProposal] = {}
        self.authorized_agents = {"sage", "curator", "king"}  # MCP agent names

        # Performance tracking
        self.metrics = {
            "total_knowledge_pieces": 0,
            "shards_active": 0,
            "devices_participating": 0,
            "global_contributions_processed": 0,
            "governance_proposals_active": 0,
            "sage_research_updates": 0,
        }

        logger.info("Distributed RAG Coordinator initialized")

    async def initialize_distributed_system(self) -> bool:
        """Initialize the distributed RAG system"""

        try:
            # 1. Assess fog network capacity
            fog_status = self.fog_coordinator.get_system_status()
            available_nodes = fog_status["nodes"]["total"]

            if available_nodes < 3:
                logger.warning(f"Only {available_nodes} fog nodes available, need at least 3 for reliable sharding")
                return False

            # 2. Create initial semantic shards
            semantic_domains = [
                "technology",
                "science",
                "health",
                "business",
                "education",
                "entertainment",
                "lifestyle",
                "news",
                "research",
                "personal_patterns",
            ]

            for domain in semantic_domains:
                shard_id = f"shard_{domain}_{int(time.time())}"
                shard = KnowledgeShard(shard_id=shard_id, semantic_domain=domain, node_assignments={})

                # Assign nodes to this shard
                await self._assign_nodes_to_shard(shard)
                self.knowledge_shards[shard_id] = shard

            # 3. Set up governance MCP interfaces
            await self._register_governance_mcp_tools()

            self.metrics["shards_active"] = len(self.knowledge_shards)
            logger.info(f"Initialized {len(self.knowledge_shards)} knowledge shards across {available_nodes} nodes")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize distributed system: {e}")
            return False

    async def register_mini_rag(self, device_id: str, mini_rag: MiniRAGSystem) -> bool:
        """Register a device's mini-RAG system for knowledge elevation"""

        if device_id in self.mini_rag_systems:
            logger.info(f"Mini-RAG for device {device_id} already registered, updating")

        self.mini_rag_systems[device_id] = mini_rag
        self.metrics["devices_participating"] = len(self.mini_rag_systems)

        logger.info(f"Registered Mini-RAG for device {device_id}")

        # Start background task to process contributions from this device
        asyncio.create_task(self._monitor_device_contributions(device_id))

        return True

    async def process_global_contributions(self, contributions: list[GlobalContribution]) -> dict[str, bool]:
        """Process anonymized contributions from local mini-RAG systems"""

        results = {}

        for contribution in contributions:
            try:
                # 1. Validate contribution is properly anonymized
                if not self._validate_anonymization(contribution):
                    results[contribution.contribution_id] = False
                    logger.warning(f"Contribution {contribution.contribution_id} rejected - insufficient anonymization")
                    continue

                # 2. Determine target shard based on content
                target_shard = await self._determine_target_shard(contribution)
                if not target_shard:
                    results[contribution.contribution_id] = False
                    continue

                # 3. Create knowledge node in Bayesian trust graph
                knowledge_node = await self._create_global_knowledge_node(contribution)

                # 4. Add to appropriate shard
                await self._add_to_distributed_shard(target_shard, knowledge_node, contribution)

                # 5. Update global HyperRAG system
                await self._integrate_with_hyperrag(knowledge_node, contribution)

                results[contribution.contribution_id] = True
                self.metrics["global_contributions_processed"] += 1

                logger.info(f"Successfully processed global contribution {contribution.contribution_id}")

            except Exception as e:
                logger.error(f"Error processing contribution {contribution.contribution_id}: {e}")
                results[contribution.contribution_id] = False

        return results

    async def sage_research_update(self, research_results: dict[str, Any]) -> str:
        """Handle updates from Sage agent after deep research jobs"""

        update_id = f"sage_update_{int(time.time())}"

        try:
            # 1. Extract key findings from research
            findings = research_results.get("key_findings", [])
            sources = research_results.get("sources", [])
            confidence = research_results.get("confidence_score", 0.7)
            domain = research_results.get("research_domain", "general")

            # 2. Create high-trust knowledge nodes (Sage is authoritative)
            for finding in findings:
                knowledge_node = GraphNode(
                    content=finding.get("content", ""),
                    concept=finding.get("concept", ""),
                    trust_score=min(0.9, confidence + 0.2),  # Sage gets trust bonus
                    confidence=confidence,
                    belief_strength=0.8,
                    evidence_count=len(sources),
                    source_reliability=0.9,  # Sage is highly reliable
                    verification_status="sage_verified",
                    tags=["sage_research", domain, "authoritative"],
                )

                # Add to appropriate shard
                target_shard = await self._determine_target_shard_by_domain(domain)
                if target_shard:
                    await self._add_sage_knowledge_to_shard(target_shard, knowledge_node)

            # 3. Update HyperRAG with new authoritative knowledge
            await self.hyper_rag.add_document(
                content=f"Sage Research Update: {research_results.get('title', 'Research Findings')}",
                metadata={
                    "source": "sage_agent",
                    "research_id": research_results.get("research_id"),
                    "domain": domain,
                    "confidence": confidence,
                    "update_id": update_id,
                },
            )

            self.metrics["sage_research_updates"] += 1
            self.metrics["total_knowledge_pieces"] += len(findings)

            logger.info(f"Processed Sage research update {update_id} with {len(findings)} findings")
            return update_id

        except Exception as e:
            logger.error(f"Error processing Sage research update: {e}")
            return f"error_{update_id}"

    async def create_governance_proposal(
        self,
        proposer: str,
        title: str,
        description: str,
        changes: dict[str, Any],
        decision_type: GovernanceDecision = GovernanceDecision.MAJOR_CHANGE,
    ) -> str:
        """Create governance proposal for RAG system changes (MCP interface)"""

        if proposer not in self.authorized_agents:
            raise PermissionError(f"Agent {proposer} not authorized for governance proposals")

        proposal_id = f"proposal_{proposer}_{int(time.time())}"

        # Determine required votes based on decision type
        required_votes = {
            GovernanceDecision.MINOR_UPDATE: 1,
            GovernanceDecision.MAJOR_CHANGE: 2,  # 2/3 quorum
            GovernanceDecision.CRITICAL_CHANGE: 3,  # Unanimous
            GovernanceDecision.EMERGENCY_ACTION: 1,  # King override
        }

        # Emergency actions only allowed by King
        if decision_type == GovernanceDecision.EMERGENCY_ACTION and proposer != "king":
            raise PermissionError("Emergency actions can only be proposed by King agent")

        proposal = GovernanceProposal(
            proposal_id=proposal_id,
            decision_type=decision_type,
            proposer_agent=proposer,
            title=title,
            description=description,
            changes=changes,
            required_votes=required_votes[decision_type],
            voting_deadline=datetime.fromtimestamp(time.time() + 86400),  # 24 hour voting period
        )

        # Auto-approve if only 1 vote needed
        if required_votes[decision_type] == 1:
            proposal.votes_for.add(proposer)
            proposal.status = "approved"
            await self._implement_proposal(proposal)
        else:
            proposal.votes_for.add(proposer)  # Proposer auto-votes for their proposal

        self.governance_proposals[proposal_id] = proposal
        self.metrics["governance_proposals_active"] += 1

        logger.info(f"Created governance proposal {proposal_id} by {proposer}")
        return proposal_id

    async def vote_on_proposal(self, voter: str, proposal_id: str, vote: bool) -> dict[str, Any]:
        """Vote on a governance proposal (MCP interface)"""

        if voter not in self.authorized_agents:
            raise PermissionError(f"Agent {voter} not authorized to vote")

        if proposal_id not in self.governance_proposals:
            raise ValueError(f"Proposal {proposal_id} not found")

        proposal = self.governance_proposals[proposal_id]

        if proposal.status != "pending":
            raise ValueError(f"Proposal {proposal_id} is not in pending status")

        # Record vote
        if vote:
            proposal.votes_for.add(voter)
            proposal.votes_against.discard(voter)
        else:
            proposal.votes_against.add(voter)
            proposal.votes_for.discard(voter)

        # Check if proposal passes
        votes_for = len(proposal.votes_for)
        votes_against = len(proposal.votes_against)

        result = {
            "proposal_id": proposal_id,
            "voter": voter,
            "vote": vote,
            "votes_for": votes_for,
            "votes_against": votes_against,
            "required_votes": proposal.required_votes,
            "status": proposal.status,
        }

        if votes_for >= proposal.required_votes:
            proposal.status = "approved"
            await self._implement_proposal(proposal)
            result["status"] = "approved"
            result["message"] = "Proposal approved and implemented"

        elif votes_against >= (3 - proposal.required_votes + 1):  # Majority against
            proposal.status = "rejected"
            result["status"] = "rejected"
            result["message"] = "Proposal rejected"

        logger.info(f"Vote recorded: {voter} voted {vote} on proposal {proposal_id}")
        return result

    async def get_distributed_knowledge(self, query: str, mode: QueryMode = QueryMode.BALANCED) -> dict[str, Any]:
        """Query the distributed knowledge system"""

        # 1. Query local HyperRAG first (fastest)
        hyper_results = await self.hyper_rag.query(query, mode)

        # 2. Identify relevant shards for distributed search
        relevant_shards = await self._identify_relevant_shards(query)

        # 3. Query distributed shards in parallel
        shard_results = []
        if relevant_shards:
            tasks = [self._query_shard(shard_id, query) for shard_id in relevant_shards]
            shard_results = await asyncio.gather(*tasks, return_exceptions=True)

        # 4. Combine and rank results
        combined_results = await self._combine_distributed_results(hyper_results, shard_results)

        return {
            "query": query,
            "mode": mode.value,
            "local_results": hyper_results,
            "distributed_results": combined_results,
            "shards_queried": len(relevant_shards),
            "total_sources": len(combined_results.get("sources", [])),
            "confidence": combined_results.get("confidence", 0.0),
        }

    # Private helper methods

    async def _assign_nodes_to_shard(self, shard: KnowledgeShard):
        """Assign fog nodes to a knowledge shard"""

        fog_status = self.fog_coordinator.get_system_status()
        available_nodes = list(fog_status["nodes"]["by_state"].keys())

        # Simple round-robin assignment for demo
        # In production would use sophisticated placement algorithms
        for i, node_id in enumerate(available_nodes[: shard.replication_factor]):
            weight = 1.0 / shard.replication_factor
            shard.node_assignments[node_id] = weight

            if i == 0:
                shard.primary_nodes.append(node_id)
            else:
                shard.backup_nodes.append(node_id)

    async def _validate_anonymization(self, contribution: GlobalContribution) -> bool:
        """Validate that contribution is properly anonymized"""

        content = contribution.anonymized_content

        # Check for required anonymization markers
        required_fields = ["pattern_knowledge", "anonymization_applied", "privacy_preserved"]
        for field in required_fields:
            if field not in content:
                return False

        # Verify no personal identifiers in pattern knowledge
        pattern = content.get("pattern_knowledge", "").lower()
        personal_indicators = ["i ", "my ", "me ", "personal", "home address", "@", "phone"]

        for indicator in personal_indicators:
            if indicator in pattern:
                logger.warning(f"Personal identifier found in contribution: {indicator}")
                return False

        return content.get("anonymization_applied", False) and content.get("privacy_preserved", False)

    async def _determine_target_shard(self, contribution: GlobalContribution) -> str | None:
        """Determine which shard should store this contribution"""

        contribution.anonymized_content.get("knowledge_type", "")
        pattern = contribution.anonymized_content.get("pattern_knowledge", "").lower()

        # Simple keyword-based shard selection
        shard_keywords = {
            "technology": ["app", "software", "tech", "device", "programming"],
            "health": ["health", "medical", "fitness", "wellness", "exercise"],
            "business": ["work", "meeting", "business", "professional", "office"],
            "lifestyle": ["personal", "home", "family", "social", "entertainment"],
            "research": ["research", "study", "analysis", "investigation", "findings"],
        }

        for domain, keywords in shard_keywords.items():
            if any(keyword in pattern for keyword in keywords):
                # Find shard for this domain
                for shard_id, shard in self.knowledge_shards.items():
                    if shard.semantic_domain == domain:
                        return shard_id

        # Default to general knowledge shard
        for shard_id, shard in self.knowledge_shards.items():
            if shard.semantic_domain == "personal_patterns":
                return shard_id

        return None

    async def _monitor_device_contributions(self, device_id: str):
        """Background task to monitor contributions from a device"""

        while device_id in self.mini_rag_systems:
            try:
                mini_rag = self.mini_rag_systems[device_id]

                # Check for new contribution candidates
                candidates = await mini_rag.get_global_contribution_candidates()

                if candidates:
                    # Create contributions
                    contributions = await mini_rag.contribute_to_global_rag(
                        [c.knowledge_id for c in candidates[:5]]
                    )  # Limit batch size

                    if contributions:
                        # Process contributions
                        results = await self.process_global_contributions(contributions)
                        success_count = sum(1 for success in results.values() if success)
                        logger.info(
                            f"Processed {success_count}/{len(contributions)} contributions from device {device_id}"
                        )

                # Sleep between checks
                await asyncio.sleep(3600)  # Check every hour

            except Exception as e:
                logger.error(f"Error monitoring contributions from device {device_id}: {e}")
                await asyncio.sleep(1800)  # Shorter sleep on error

    def get_system_status(self) -> dict[str, Any]:
        """Get comprehensive system status"""

        return {
            "distributed_rag_coordinator": {
                "status": "operational",
                "shards_active": len(self.knowledge_shards),
                "devices_connected": len(self.mini_rag_systems),
                "governance_proposals": {
                    "active": len([p for p in self.governance_proposals.values() if p.status == "pending"]),
                    "total": len(self.governance_proposals),
                },
                "metrics": self.metrics.copy(),
            },
            "fog_network": self.fog_coordinator.get_system_status(),
            "hyperrag_status": "integrated" if self.hyper_rag else "not_available",
        }


# MCP Tools for Governance Interface
class RAGGovernanceMCP:
    """MCP (Model Control Protocol) tools for RAG system governance"""

    def __init__(self, coordinator: DistributedRAGCoordinator):
        self.coordinator = coordinator

    async def create_proposal(
        self, agent_id: str, title: str, description: str, changes: dict[str, Any], decision_type: str = "major_change"
    ) -> dict[str, Any]:
        """MCP tool for creating governance proposals"""
        try:
            decision_enum = GovernanceDecision(decision_type)
            proposal_id = await self.coordinator.create_governance_proposal(
                proposer=agent_id, title=title, description=description, changes=changes, decision_type=decision_enum
            )
            return {"success": True, "proposal_id": proposal_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def vote_proposal(self, agent_id: str, proposal_id: str, vote: bool) -> dict[str, Any]:
        """MCP tool for voting on proposals"""
        try:
            result = await self.coordinator.vote_on_proposal(agent_id, proposal_id, vote)
            return {"success": True, **result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def submit_research_update(self, research_results: dict[str, Any]) -> dict[str, Any]:
        """MCP tool for Sage to submit research updates"""
        try:
            update_id = await self.coordinator.sage_research_update(research_results)
            return {"success": True, "update_id": update_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def query_distributed_knowledge(self, query: str, mode: str = "balanced") -> dict[str, Any]:
        """MCP tool for querying distributed knowledge"""
        try:
            query_mode = QueryMode(mode)
            results = await self.coordinator.get_distributed_knowledge(query, query_mode)
            return {"success": True, **results}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_rag_metrics(self) -> dict[str, Any]:
        """MCP tool for getting RAG system metrics"""
        return {"success": True, "status": self.coordinator.get_system_status()}
