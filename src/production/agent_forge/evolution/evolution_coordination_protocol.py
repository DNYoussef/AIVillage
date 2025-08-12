"""Evolution Coordination Protocol for Future Distributed Operation."""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ....core.p2p import P2PNode
from ....core.p2p.message_protocol import (
    EvolutionMessage,
    MessagePriority,
    MessageType,
)

logger = logging.getLogger(__name__)


class CoordinationPhase(Enum):
    """Phases of evolution coordination."""

    DISCOVERY = "discovery"  # Discover suitable peers
    NEGOTIATION = "negotiation"  # Negotiate resource allocation
    PREPARATION = "preparation"  # Prepare for distributed execution
    EXECUTION = "execution"  # Execute distributed evolution
    CONSENSUS = "consensus"  # Reach consensus on results
    FINALIZATION = "finalization"  # Finalize and distribute results


class ConsensusType(Enum):
    """Types of consensus mechanisms."""

    SIMPLE_MAJORITY = "simple_majority"  # >50% agreement
    SUPERMAJORITY = "supermajority"  # >66% agreement
    UNANIMOUS = "unanimous"  # 100% agreement
    WEIGHTED_VOTING = "weighted_voting"  # Weighted by peer capability
    PROOF_OF_WORK = "proof_of_work"  # Best result wins


@dataclass
class EvolutionProposal:
    """Proposal for distributed evolution."""

    proposal_id: str
    agent_id: str
    evolution_type: str
    initiator_node_id: str
    timestamp: float

    # Resource requirements
    min_peers_required: int
    max_peers_allowed: int
    total_memory_mb_required: int
    total_cpu_percent_required: float
    estimated_duration_minutes: float

    # Quality and priority
    quality_target: float
    priority_level: int
    can_be_interrupted: bool

    # Coordination parameters
    consensus_type: ConsensusType = ConsensusType.SIMPLE_MAJORITY
    consensus_threshold: float = 0.6
    timeout_minutes: float = 10.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "agent_id": self.agent_id,
            "evolution_type": self.evolution_type,
            "initiator_node_id": self.initiator_node_id,
            "timestamp": self.timestamp,
            "min_peers_required": self.min_peers_required,
            "max_peers_allowed": self.max_peers_allowed,
            "total_memory_mb_required": self.total_memory_mb_required,
            "total_cpu_percent_required": self.total_cpu_percent_required,
            "estimated_duration_minutes": self.estimated_duration_minutes,
            "quality_target": self.quality_target,
            "priority_level": self.priority_level,
            "can_be_interrupted": self.can_be_interrupted,
            "consensus_type": self.consensus_type.value,
            "consensus_threshold": self.consensus_threshold,
            "timeout_minutes": self.timeout_minutes,
        }


@dataclass
class PeerContribution:
    """Peer's contribution to distributed evolution."""

    peer_id: str
    node_capabilities: dict[str, Any]
    offered_memory_mb: int
    offered_cpu_percent: float
    estimated_availability_minutes: float
    commitment_strength: float  # 0-1, how committed the peer is
    trust_score: float  # 0-1, trust level


@dataclass
class ConsensusVote:
    """Vote in consensus process."""

    voter_id: str
    proposal_id: str
    vote: str  # "approve", "reject", "abstain"
    confidence: float  # 0-1, confidence in the vote
    reasoning: str | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class EvolutionResult:
    """Result from distributed evolution."""

    result_id: str
    contributor_id: str
    proposal_id: str
    success: bool
    quality_score: float
    execution_time_minutes: float
    resource_usage: dict[str, float]
    result_data: dict[str, Any]
    verification_hash: str | None = None
    timestamp: float = field(default_factory=time.time)


class EvolutionCoordinationProtocol:
    """Protocol for coordinating distributed evolution across P2P network."""

    def __init__(self, p2p_node: P2PNode, node_capabilities: dict[str, Any]) -> None:
        self.p2p_node = p2p_node
        self.node_capabilities = node_capabilities

        # Active coordination state
        self.active_proposals: dict[str, EvolutionProposal] = {}
        self.peer_contributions: dict[
            str, list[PeerContribution]
        ] = {}  # proposal_id -> contributions
        self.consensus_votes: dict[
            str, list[ConsensusVote]
        ] = {}  # proposal_id -> votes
        self.evolution_results: dict[
            str, list[EvolutionResult]
        ] = {}  # proposal_id -> results

        # Coordination tracking
        self.coordination_phases: dict[str, CoordinationPhase] = {}
        self.phase_timeouts: dict[str, float] = {}

        # Callbacks for coordination events
        self.proposal_callbacks: list[callable] = []
        self.consensus_callbacks: list[callable] = []
        self.result_callbacks: list[callable] = []

        # Protocol statistics
        self.stats = {
            "proposals_initiated": 0,
            "proposals_participated": 0,
            "proposals_completed": 0,
            "proposals_failed": 0,
            "consensus_reached": 0,
            "consensus_failed": 0,
            "distributed_executions": 0,
            "coordination_timeouts": 0,
        }

        # Register P2P message handlers
        self._register_message_handlers()

    def _register_message_handlers(self) -> None:
        """Register message handlers for coordination protocol."""
        handlers = {
            "EVOLUTION_PROPOSAL": self._handle_evolution_proposal,
            "EVOLUTION_PROPOSAL_RESPONSE": self._handle_proposal_response,
            "EVOLUTION_CONTRIBUTION_OFFER": self._handle_contribution_offer,
            "EVOLUTION_CONTRIBUTION_ACCEPT": self._handle_contribution_accept,
            "EVOLUTION_CONTRIBUTION_REJECT": self._handle_contribution_reject,
            "EVOLUTION_CONSENSUS_REQUEST": self._handle_consensus_request,
            "EVOLUTION_CONSENSUS_VOTE": self._handle_consensus_vote,
            "EVOLUTION_RESULT_SUBMIT": self._handle_result_submit,
            "EVOLUTION_RESULT_VERIFY": self._handle_result_verify,
            "EVOLUTION_COORDINATION_ABORT": self._handle_coordination_abort,
        }

        for msg_type, handler in handlers.items():
            self.p2p_node.register_handler(msg_type, handler)

    async def initiate_distributed_evolution(
        self, agent_id: str, evolution_type: str, resource_requirements: dict[str, Any]
    ) -> str:
        """Initiate distributed evolution coordination."""
        proposal_id = str(uuid.uuid4())

        # Create evolution proposal
        proposal = EvolutionProposal(
            proposal_id=proposal_id,
            agent_id=agent_id,
            evolution_type=evolution_type,
            initiator_node_id=self.p2p_node.node_id,
            timestamp=time.time(),
            min_peers_required=resource_requirements.get("min_peers", 2),
            max_peers_allowed=resource_requirements.get("max_peers", 5),
            total_memory_mb_required=resource_requirements.get("memory_mb", 2048),
            total_cpu_percent_required=resource_requirements.get("cpu_percent", 200.0),
            estimated_duration_minutes=resource_requirements.get(
                "duration_minutes", 60.0
            ),
            quality_target=resource_requirements.get("quality_target", 0.8),
            priority_level=resource_requirements.get("priority", 2),
            can_be_interrupted=resource_requirements.get("can_interrupt", True),
        )

        # Store proposal
        self.active_proposals[proposal_id] = proposal
        self.coordination_phases[proposal_id] = CoordinationPhase.DISCOVERY
        self.phase_timeouts[proposal_id] = time.time() + proposal.timeout_minutes * 60

        # Broadcast proposal to peers
        await self._broadcast_proposal(proposal)

        # Start coordination process
        asyncio.create_task(self._coordinate_distributed_evolution(proposal_id))

        self.stats["proposals_initiated"] += 1
        logger.info(f"Initiated distributed evolution proposal {proposal_id}")

        return proposal_id

    async def _broadcast_proposal(self, proposal: EvolutionProposal) -> None:
        """Broadcast evolution proposal to all peers."""
        message = EvolutionMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.EVOLUTION_PROPOSAL,
            sender_id=self.p2p_node.node_id,
            evolution_id=proposal.proposal_id,
            agent_id=proposal.agent_id,
            priority=MessagePriority.HIGH,
            data=proposal.to_dict(),
        )

        # Send to all suitable peers
        suitable_peers = self.p2p_node.get_suitable_evolution_peers()
        for peer in suitable_peers:
            await self.p2p_node.send_to_peer(peer.device_id, message.to_dict())

    async def _coordinate_distributed_evolution(self, proposal_id: str) -> None:
        """Main coordination loop for distributed evolution."""
        try:
            self.active_proposals[proposal_id]

            # Phase 1: Discovery and Negotiation
            await self._discovery_phase(proposal_id)

            # Phase 2: Resource Preparation
            if self.coordination_phases[proposal_id] == CoordinationPhase.PREPARATION:
                await self._preparation_phase(proposal_id)

            # Phase 3: Distributed Execution
            if self.coordination_phases[proposal_id] == CoordinationPhase.EXECUTION:
                await self._execution_phase(proposal_id)

            # Phase 4: Consensus on Results
            if self.coordination_phases[proposal_id] == CoordinationPhase.CONSENSUS:
                await self._consensus_phase(proposal_id)

            # Phase 5: Finalization
            if self.coordination_phases[proposal_id] == CoordinationPhase.FINALIZATION:
                await self._finalization_phase(proposal_id)

            self.stats["proposals_completed"] += 1

        except asyncio.TimeoutError:
            logger.warning(f"Coordination timeout for proposal {proposal_id}")
            self.stats["coordination_timeouts"] += 1
            await self._abort_coordination(proposal_id, "timeout")

        except Exception as e:
            logger.exception(f"Error in coordination for proposal {proposal_id}: {e}")
            self.stats["proposals_failed"] += 1
            await self._abort_coordination(proposal_id, str(e))

        finally:
            # Cleanup
            await self._cleanup_coordination(proposal_id)

    async def _discovery_phase(self, proposal_id: str) -> None:
        """Discovery phase: collect peer contributions."""
        logger.info(f"Starting discovery phase for proposal {proposal_id}")

        proposal = self.active_proposals[proposal_id]
        timeout = self.phase_timeouts[proposal_id]

        # Wait for peer responses
        while time.time() < timeout:
            contributions = self.peer_contributions.get(proposal_id, [])

            if len(contributions) >= proposal.min_peers_required:
                # Sufficient peers found, move to preparation
                self.coordination_phases[proposal_id] = CoordinationPhase.PREPARATION
                logger.info(
                    f"Discovery phase completed for proposal {proposal_id} with {len(contributions)} peers"
                )
                return

            await asyncio.sleep(1.0)

        # Check if we have minimum requirements
        contributions = self.peer_contributions.get(proposal_id, [])
        if len(contributions) < proposal.min_peers_required:
            msg = f"Insufficient peers: got {len(contributions)}, need {proposal.min_peers_required}"
            raise Exception(msg)

    async def _preparation_phase(self, proposal_id: str) -> None:
        """Preparation phase: prepare resources and confirm participation."""
        logger.info(f"Starting preparation phase for proposal {proposal_id}")

        # Select best peer contributions
        contributions = self.peer_contributions.get(proposal_id, [])
        selected_peers = self._select_best_contributions(proposal_id, contributions)

        # Send preparation requests to selected peers
        for peer in selected_peers:
            await self._send_preparation_request(proposal_id, peer)

        # Wait for confirmations
        # For now, assume preparation succeeds
        self.coordination_phases[proposal_id] = CoordinationPhase.EXECUTION

    async def _execution_phase(self, proposal_id: str) -> None:
        """Execution phase: coordinate distributed evolution execution."""
        logger.info(f"Starting execution phase for proposal {proposal_id}")

        # For Sprint 6, this is a placeholder for future distributed execution
        # Currently, we'll simulate distributed execution

        proposal = self.active_proposals[proposal_id]
        contributions = self.peer_contributions.get(proposal_id, [])

        # Simulate distributed execution
        await asyncio.sleep(5.0)  # Simulate execution time

        # Create mock results
        for i, peer in enumerate(contributions[:3]):  # Use up to 3 peers
            result = EvolutionResult(
                result_id=str(uuid.uuid4()),
                contributor_id=peer.peer_id,
                proposal_id=proposal_id,
                success=True,
                quality_score=0.8 + (i * 0.05),  # Simulate quality variance
                execution_time_minutes=proposal.estimated_duration_minutes * 0.8,
                resource_usage={
                    "memory_mb": peer.offered_memory_mb * 0.9,
                    "cpu_percent": peer.offered_cpu_percent * 0.8,
                },
                result_data={"simulation": True, "peer_rank": i},
            )

            if proposal_id not in self.evolution_results:
                self.evolution_results[proposal_id] = []
            self.evolution_results[proposal_id].append(result)

        self.coordination_phases[proposal_id] = CoordinationPhase.CONSENSUS
        self.stats["distributed_executions"] += 1

    async def _consensus_phase(self, proposal_id: str) -> None:
        """Consensus phase: reach agreement on best result."""
        logger.info(f"Starting consensus phase for proposal {proposal_id}")

        self.active_proposals[proposal_id]
        results = self.evolution_results.get(proposal_id, [])

        if not results:
            msg = "No results to reach consensus on"
            raise Exception(msg)

        # Request consensus votes from peers
        await self._request_consensus_votes(proposal_id, results)

        # Wait for votes (simplified for Sprint 6)
        await asyncio.sleep(2.0)

        # Determine consensus (mock implementation)
        max(results, key=lambda r: r.quality_score)

        # For now, assume consensus is reached
        self.coordination_phases[proposal_id] = CoordinationPhase.FINALIZATION
        self.stats["consensus_reached"] += 1

    async def _finalization_phase(self, proposal_id: str) -> None:
        """Finalization phase: distribute final results."""
        logger.info(f"Starting finalization phase for proposal {proposal_id}")

        # Get consensus result
        results = self.evolution_results.get(proposal_id, [])
        if results:
            best_result = max(results, key=lambda r: r.quality_score)

            # Broadcast final result to all participants
            await self._broadcast_final_result(proposal_id, best_result)

        logger.info(f"Coordination completed for proposal {proposal_id}")

    def _select_best_contributions(
        self, proposal_id: str, contributions: list[PeerContribution]
    ) -> list[PeerContribution]:
        """Select best peer contributions for execution."""
        proposal = self.active_proposals[proposal_id]

        # Score contributions based on capability, trust, and commitment
        scored_contributions = []
        for contrib in contributions:
            score = (
                contrib.trust_score * 0.4
                + contrib.commitment_strength * 0.3
                + min(contrib.offered_memory_mb / 1000, 1.0) * 0.2
                + min(contrib.offered_cpu_percent / 100, 1.0) * 0.1
            )
            scored_contributions.append((score, contrib))

        # Sort by score and select top contributors
        scored_contributions.sort(reverse=True)
        max_peers = min(proposal.max_peers_allowed, len(scored_contributions))

        return [contrib for _, contrib in scored_contributions[:max_peers]]

    async def _send_preparation_request(
        self, proposal_id: str, peer: PeerContribution
    ) -> None:
        """Send preparation request to peer."""
        # Implementation would send actual preparation message

    async def _request_consensus_votes(
        self, proposal_id: str, results: list[EvolutionResult]
    ) -> None:
        """Request consensus votes from participating peers."""
        # Implementation would request votes from peers

    async def _broadcast_final_result(
        self, proposal_id: str, result: EvolutionResult
    ) -> None:
        """Broadcast final consensus result."""
        message = EvolutionMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.EVOLUTION_RESULTS_SHARE,
            sender_id=self.p2p_node.node_id,
            evolution_id=proposal_id,
            priority=MessagePriority.HIGH,
            data={"final_result": result.__dict__, "consensus_reached": True},
        )

        # Broadcast to all peers
        await self.p2p_node.broadcast_to_peers("EVOLUTION_FINAL_RESULT", message.data)

    async def _abort_coordination(self, proposal_id: str, reason: str) -> None:
        """Abort coordination process."""
        logger.warning(f"Aborting coordination for proposal {proposal_id}: {reason}")

        # Send abort message to all participants
        abort_message = EvolutionMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.EVOLUTION_COORDINATION_ABORT,
            sender_id=self.p2p_node.node_id,
            evolution_id=proposal_id,
            priority=MessagePriority.HIGH,
            data={"reason": reason},
        )

        await self.p2p_node.broadcast_to_peers(
            "EVOLUTION_COORDINATION_ABORT", abort_message.data
        )

    async def _cleanup_coordination(self, proposal_id: str) -> None:
        """Cleanup coordination data."""
        # Remove from active tracking
        self.active_proposals.pop(proposal_id, None)
        self.peer_contributions.pop(proposal_id, None)
        self.consensus_votes.pop(proposal_id, None)
        self.coordination_phases.pop(proposal_id, None)
        self.phase_timeouts.pop(proposal_id, None)

        # Keep results for a while for reference
        # In production, this would be moved to persistent storage

    # Message Handlers
    async def _handle_evolution_proposal(self, message: dict, writer) -> None:
        """Handle incoming evolution proposal."""
        proposal_data = message.get("data", {})
        proposal_id = proposal_data.get("proposal_id")

        logger.info(f"Received evolution proposal {proposal_id}")

        # Evaluate if we can contribute
        can_contribute = await self._evaluate_contribution_capability(proposal_data)

        if can_contribute:
            # Create contribution offer
            contribution = PeerContribution(
                peer_id=self.p2p_node.node_id,
                node_capabilities=self.node_capabilities,
                offered_memory_mb=min(
                    1024, self.node_capabilities.get("memory_available_mb", 0)
                ),
                offered_cpu_percent=min(
                    50.0, self.node_capabilities.get("cpu_available_percent", 0)
                ),
                estimated_availability_minutes=120.0,
                commitment_strength=0.8,
                trust_score=1.0,  # Self-trust
            )

            # Send contribution offer
            await self._send_contribution_offer(
                proposal_id, contribution, message["sender_id"]
            )

            self.stats["proposals_participated"] += 1

    async def _evaluate_contribution_capability(
        self, proposal_data: dict[str, Any]
    ) -> bool:
        """Evaluate if we can contribute to a proposal."""
        # Check resource availability
        required_memory = proposal_data.get(
            "total_memory_mb_required", 0
        ) / proposal_data.get("max_peers_allowed", 1)
        required_cpu = proposal_data.get(
            "total_cpu_percent_required", 0
        ) / proposal_data.get("max_peers_allowed", 1)

        available_memory = self.node_capabilities.get("memory_available_mb", 0)
        available_cpu = self.node_capabilities.get("cpu_available_percent", 0)

        # Simple capability check
        return (
            available_memory >= required_memory * 0.5
            and available_cpu >= required_cpu * 0.5
        )

    async def _send_contribution_offer(
        self, proposal_id: str, contribution: PeerContribution, initiator_id: str
    ) -> None:
        """Send contribution offer to proposal initiator."""
        message = EvolutionMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.EVOLUTION_CONTRIBUTION_OFFER,
            sender_id=self.p2p_node.node_id,
            recipient_id=initiator_id,
            evolution_id=proposal_id,
            priority=MessagePriority.NORMAL,
            data=contribution.__dict__,
        )

        await self.p2p_node.send_to_peer(initiator_id, message.to_dict())

    async def _handle_proposal_response(self, message: dict, writer) -> None:
        """Handle response to our proposal."""
        # Placeholder

    async def _handle_contribution_offer(self, message: dict, writer) -> None:
        """Handle contribution offer from peer."""
        proposal_id = message.get("evolution_id")
        contribution_data = message.get("data", {})

        if proposal_id and proposal_id in self.active_proposals:
            # Create contribution object
            contribution = PeerContribution(**contribution_data)

            # Store contribution
            if proposal_id not in self.peer_contributions:
                self.peer_contributions[proposal_id] = []
            self.peer_contributions[proposal_id].append(contribution)

            logger.info(
                f"Received contribution offer for proposal {proposal_id} from {contribution.peer_id}"
            )

    async def _handle_contribution_accept(self, message: dict, writer) -> None:
        """Handle contribution acceptance."""
        # Placeholder

    async def _handle_contribution_reject(self, message: dict, writer) -> None:
        """Handle contribution rejection."""
        # Placeholder

    async def _handle_consensus_request(self, message: dict, writer) -> None:
        """Handle consensus request."""
        # Placeholder

    async def _handle_consensus_vote(self, message: dict, writer) -> None:
        """Handle consensus vote."""
        # Placeholder

    async def _handle_result_submit(self, message: dict, writer) -> None:
        """Handle result submission."""
        # Placeholder

    async def _handle_result_verify(self, message: dict, writer) -> None:
        """Handle result verification."""
        # Placeholder

    async def _handle_coordination_abort(self, message: dict, writer) -> None:
        """Handle coordination abort."""
        proposal_id = message.get("evolution_id")
        reason = message.get("data", {}).get("reason", "unknown")

        logger.warning(f"Coordination aborted for proposal {proposal_id}: {reason}")

        # Cleanup local state
        await self._cleanup_coordination(proposal_id)

    def get_coordination_status(self) -> dict[str, Any]:
        """Get current coordination status."""
        return {
            "active_proposals": len(self.active_proposals),
            "coordination_phases": {
                proposal_id: phase.value
                for proposal_id, phase in self.coordination_phases.items()
            },
            "peer_contributions": {
                proposal_id: len(contributions)
                for proposal_id, contributions in self.peer_contributions.items()
            },
            "evolution_results": {
                proposal_id: len(results)
                for proposal_id, results in self.evolution_results.items()
            },
            "stats": self.stats.copy(),
        }

    def register_proposal_callback(self, callback: callable) -> None:
        """Register callback for proposal events."""
        self.proposal_callbacks.append(callback)

    def register_consensus_callback(self, callback: callable) -> None:
        """Register callback for consensus events."""
        self.consensus_callbacks.append(callback)

    def register_result_callback(self, callback: callable) -> None:
        """Register callback for result events."""
        self.result_callbacks.append(callback)
