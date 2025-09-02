"""
P2P Network Integration for Federated Marketplace

Integrates the enhanced fog marketplace with P2P network and federated systems:
- Resource discovery through P2P network
- Federated coordinator integration  
- Cross-network auction participation
- Distributed resource allocation
- Peer reputation and trust management

Key Features:
- P2P resource advertising and discovery
- Federated auction coordination
- Cross-network resource sharing
- Distributed consensus for allocations
- Real-time peer status monitoring
"""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum
import logging
from typing import Any
import uuid

from .auction_engine import AuctionEngine, ResourceRequirement
from .market_orchestrator import MarketOrchestrator
from .pricing_manager import DynamicPricingManager
from .resource_allocator import DynamicResourceAllocator, QoSRequirement, ResourceNode, ResourceType

logger = logging.getLogger(__name__)


class P2PMessageType(str, Enum):
    """Types of P2P messages for marketplace integration"""

    RESOURCE_ADVERTISEMENT = "resource_advertisement"
    RESOURCE_REQUEST = "resource_request"
    AUCTION_ANNOUNCEMENT = "auction_announcement"
    BID_SUBMISSION = "bid_submission"
    ALLOCATION_PROPOSAL = "allocation_proposal"
    ALLOCATION_ACCEPTANCE = "allocation_acceptance"
    PEER_STATUS_UPDATE = "peer_status_update"
    REPUTATION_UPDATE = "reputation_update"


class NetworkRole(str, Enum):
    """P2P network roles for marketplace participants"""

    RESOURCE_PROVIDER = "resource_provider"
    RESOURCE_CONSUMER = "resource_consumer"
    AUCTION_COORDINATOR = "auction_coordinator"
    MARKETPLACE_BROKER = "marketplace_broker"
    FEDERATED_TRAINER = "federated_trainer"
    INFERENCE_CLIENT = "inference_client"


@dataclass
class P2PMarketplaceMessage:
    """Standard message format for P2P marketplace communication"""

    message_id: str
    message_type: P2PMessageType
    sender_peer_id: str
    sender_role: NetworkRole

    # Message content
    payload: dict[str, Any] = field(default_factory=dict)

    # Routing and delivery
    target_peer_ids: list[str] = field(default_factory=list)  # Empty = broadcast
    network_zone: str = "global"
    priority: int = 5  # 1-10, higher = more priority

    # Message metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    ttl_seconds: int = 300  # Time to live
    requires_ack: bool = False

    # Security and validation
    signature: str = ""
    trust_level: str = "medium"  # low, medium, high, verified


@dataclass
class P2PPeerInfo:
    """Information about P2P network peer"""

    peer_id: str
    roles: list[NetworkRole]

    # Network connectivity
    network_addresses: list[str] = field(default_factory=list)
    last_seen: datetime = field(default_factory=lambda: datetime.now(UTC))
    connection_quality: float = 1.0  # 0.0 to 1.0

    # Marketplace capabilities
    available_resources: dict[str, Any] = field(default_factory=dict)
    supported_auctions: list[str] = field(default_factory=list)
    pricing_tiers: list[str] = field(default_factory=list)

    # Reputation and trust
    trust_score: float = 0.5
    reputation_score: float = 0.5
    successful_transactions: int = 0
    failed_transactions: int = 0

    # Performance metrics
    average_latency_ms: float = 100.0
    uptime_percentage: float = 95.0
    response_time_ms: float = 50.0


class P2PMarketplaceIntegration:
    """
    Integration layer between P2P network and federated marketplace

    Provides:
    - Resource discovery and advertisement through P2P network
    - Distributed auction coordination
    - Cross-network resource allocation
    - Peer reputation and trust management
    """

    def __init__(
        self,
        peer_id: str,
        p2p_network=None,
        auction_engine: AuctionEngine = None,
        pricing_manager: DynamicPricingManager = None,
        market_orchestrator: MarketOrchestrator = None,
        resource_allocator: DynamicResourceAllocator = None,
    ):
        self.peer_id = peer_id
        self.p2p_network = p2p_network

        # Marketplace components
        self.auction_engine = auction_engine
        self.pricing_manager = pricing_manager
        self.market_orchestrator = market_orchestrator
        self.resource_allocator = resource_allocator

        # P2P state
        self.network_peers: dict[str, P2PPeerInfo] = {}
        self.active_auctions: dict[str, dict[str, Any]] = {}
        self.pending_allocations: dict[str, dict[str, Any]] = {}

        # Message handling
        self.message_handlers: dict[P2PMessageType, callable] = {
            P2PMessageType.RESOURCE_ADVERTISEMENT: self._handle_resource_advertisement,
            P2PMessageType.RESOURCE_REQUEST: self._handle_resource_request,
            P2PMessageType.AUCTION_ANNOUNCEMENT: self._handle_auction_announcement,
            P2PMessageType.BID_SUBMISSION: self._handle_bid_submission,
            P2PMessageType.ALLOCATION_PROPOSAL: self._handle_allocation_proposal,
            P2PMessageType.ALLOCATION_ACCEPTANCE: self._handle_allocation_acceptance,
            P2PMessageType.PEER_STATUS_UPDATE: self._handle_peer_status_update,
            P2PMessageType.REPUTATION_UPDATE: self._handle_reputation_update,
        }

        # Configuration
        self.config = {
            "resource_advertisement_interval": 60,  # seconds
            "peer_discovery_interval": 30,  # seconds
            "auction_participation_threshold": 0.7,  # trust threshold
            "max_concurrent_auctions": 10,
            "reputation_update_interval": 300,  # 5 minutes
            "network_timeout": 30,  # seconds
        }

        # Background tasks
        self._advertisement_task: asyncio.Task | None = None
        self._peer_discovery_task: asyncio.Task | None = None
        self._reputation_task: asyncio.Task | None = None

        logger.info(f"P2P marketplace integration initialized for peer {peer_id}")

    async def start(self):
        """Start P2P marketplace integration"""

        # Register with P2P network
        if self.p2p_network:
            await self.p2p_network.register_message_handler("marketplace", self._handle_p2p_message)

        # Start background tasks
        self._advertisement_task = asyncio.create_task(self._resource_advertisement_loop())
        self._peer_discovery_task = asyncio.create_task(self._peer_discovery_loop())
        self._reputation_task = asyncio.create_task(self._reputation_update_loop())

        # Announce our presence
        await self._announce_peer_capabilities()

        logger.info("P2P marketplace integration started")

    async def stop(self):
        """Stop P2P marketplace integration"""

        # Cancel background tasks
        if self._advertisement_task:
            self._advertisement_task.cancel()
        if self._peer_discovery_task:
            self._peer_discovery_task.cancel()
        if self._reputation_task:
            self._reputation_task.cancel()

        logger.info("P2P marketplace integration stopped")

    async def discover_federated_resources(
        self,
        requirements: ResourceRequirement,
        qos_requirements: QoSRequirement,
        max_peers: int = 50,
    ) -> list[tuple[str, ResourceNode]]:
        """Discover federated resources across P2P network"""

        # Create resource request message
        request_msg = P2PMarketplaceMessage(
            message_id=f"req_{uuid.uuid4().hex[:8]}",
            message_type=P2PMessageType.RESOURCE_REQUEST,
            sender_peer_id=self.peer_id,
            sender_role=NetworkRole.RESOURCE_CONSUMER,
            payload={
                "requirements": {
                    "cpu_cores": float(requirements.cpu_cores),
                    "memory_gb": float(requirements.memory_gb),
                    "storage_gb": float(requirements.storage_gb),
                    "bandwidth_mbps": float(requirements.bandwidth_mbps),
                    "duration_hours": float(requirements.duration_hours),
                    "participants_needed": requirements.participants_needed,
                    "workload_type": requirements.workload_type,
                    "privacy_level": requirements.privacy_level,
                },
                "qos_requirements": {
                    "max_latency_ms": float(qos_requirements.max_latency_ms),
                    "min_availability_percentage": float(qos_requirements.min_availability_percentage),
                    "max_cost_per_hour": float(qos_requirements.max_cost_per_hour),
                    "privacy_level": qos_requirements.privacy_level,
                },
                "max_responses": max_peers,
            },
            requires_ack=True,
        )

        # Broadcast request to network
        if self.p2p_network:
            await self.p2p_network.broadcast_message("marketplace", request_msg.payload)

        # Wait for responses
        await asyncio.sleep(5)  # Give peers time to respond

        # Collect discovered resources
        discovered_resources = []

        for peer_id, peer_info in self.network_peers.items():
            if (
                peer_info.available_resources
                and NetworkRole.RESOURCE_PROVIDER in peer_info.roles
                and peer_info.trust_score >= 0.5
            ):

                # Create ResourceNode from peer info
                resource_node = self._create_resource_node_from_peer(peer_info)
                if resource_node and resource_node.can_handle_workload(requirements):
                    discovered_resources.append((peer_id, resource_node))

        logger.info(f"Discovered {len(discovered_resources)} federated resources from P2P network")
        return discovered_resources[:max_peers]

    async def create_federated_auction(
        self,
        auction_type: str,
        requirements: ResourceRequirement,
        reserve_price: float,
        duration_minutes: int = 30,
    ) -> str:
        """Create federated auction coordinated through P2P network"""

        if not self.auction_engine:
            raise ValueError("Auction engine not available")

        # Create local auction
        auction_id = await self.auction_engine.create_auction(
            requester_id=self.peer_id,
            requirements=requirements,
            reserve_price=Decimal(str(reserve_price)),
            duration_minutes=duration_minutes,
        )

        # Announce auction to P2P network
        announcement_msg = P2PMarketplaceMessage(
            message_id=f"auction_{uuid.uuid4().hex[:8]}",
            message_type=P2PMessageType.AUCTION_ANNOUNCEMENT,
            sender_peer_id=self.peer_id,
            sender_role=NetworkRole.AUCTION_COORDINATOR,
            payload={
                "auction_id": auction_id,
                "auction_type": auction_type,
                "requirements": {
                    "cpu_cores": float(requirements.cpu_cores),
                    "memory_gb": float(requirements.memory_gb),
                    "participants_needed": requirements.participants_needed,
                    "workload_type": requirements.workload_type,
                    "privacy_level": requirements.privacy_level,
                },
                "reserve_price": reserve_price,
                "duration_minutes": duration_minutes,
                "coordinator_peer": self.peer_id,
            },
            priority=8,  # High priority for auction announcements
        )

        if self.p2p_network:
            await self.p2p_network.broadcast_message("marketplace", announcement_msg.payload)

        # Track active auction
        self.active_auctions[auction_id] = {
            "type": auction_type,
            "requirements": requirements,
            "announced_at": datetime.now(UTC),
            "participants": set(),
            "bids_received": 0,
        }

        logger.info(f"Created federated auction {auction_id} and announced to P2P network")
        return auction_id

    async def submit_federated_bid(
        self,
        auction_id: str,
        coordinator_peer_id: str,
        bid_price: float,
        available_resources: dict[str, Any],
        trust_score: float = 0.8,
    ) -> str:
        """Submit bid to federated auction through P2P network"""

        bid_msg = P2PMarketplaceMessage(
            message_id=f"bid_{uuid.uuid4().hex[:8]}",
            message_type=P2PMessageType.BID_SUBMISSION,
            sender_peer_id=self.peer_id,
            sender_role=NetworkRole.RESOURCE_PROVIDER,
            target_peer_ids=[coordinator_peer_id],
            payload={
                "auction_id": auction_id,
                "bid_price": bid_price,
                "available_resources": available_resources,
                "trust_score": trust_score,
                "peer_capabilities": {
                    "uptime_percentage": 99.0,
                    "average_latency_ms": 50.0,
                    "successful_transactions": 100,
                },
                "commitment_duration_hours": 24,
            },
            priority=7,
            requires_ack=True,
        )

        if self.p2p_network:
            await self.p2p_network.send_message(coordinator_peer_id, "marketplace", bid_msg.payload)

        logger.info(f"Submitted federated bid to auction {auction_id} via peer {coordinator_peer_id}")
        return bid_msg.message_id

    async def allocate_federated_resources(
        self,
        allocation_request: dict[str, Any],
        target_peers: list[str],
    ) -> dict[str, Any]:
        """Allocate resources across federated network"""

        allocation_id = f"fedalloc_{uuid.uuid4().hex[:8]}"

        # Create allocation proposals for target peers
        proposals = []

        for peer_id in target_peers:
            if peer_id in self.network_peers:
                self.network_peers[peer_id]

                proposal = P2PMarketplaceMessage(
                    message_id=f"prop_{uuid.uuid4().hex[:8]}",
                    message_type=P2PMessageType.ALLOCATION_PROPOSAL,
                    sender_peer_id=self.peer_id,
                    sender_role=NetworkRole.MARKETPLACE_BROKER,
                    target_peer_ids=[peer_id],
                    payload={
                        "allocation_id": allocation_id,
                        "resource_requirements": allocation_request,
                        "estimated_duration_hours": allocation_request.get("duration_hours", 1.0),
                        "compensation": {
                            "amount": allocation_request.get("cost_per_peer", 10.0),
                            "currency": "USD",
                            "payment_method": "marketplace_credits",
                        },
                        "qos_guarantees": {
                            "max_latency_ms": 200,
                            "min_uptime_percentage": 95,
                            "response_time_sla": 100,
                        },
                        "contract_terms": {
                            "commitment_hours": 24,
                            "cancellation_policy": "2_hour_notice",
                            "penalty_rate": 0.1,
                        },
                    },
                    priority=6,
                    requires_ack=True,
                )

                proposals.append(proposal)

        # Send proposals to target peers
        accepted_peers = []

        if self.p2p_network:
            for proposal in proposals:
                peer_id = proposal.target_peer_ids[0]
                await self.p2p_network.send_message(peer_id, "marketplace", proposal.payload)

        # Wait for acceptances (simplified - would be event-driven in production)
        await asyncio.sleep(10)

        # Check for accepted proposals (would be tracked in real implementation)
        for peer_id in target_peers:
            if peer_id in self.network_peers:
                # Simulate acceptance based on peer trust score
                if self.network_peers[peer_id].trust_score > 0.6:
                    accepted_peers.append(peer_id)

        # Track allocation
        self.pending_allocations[allocation_id] = {
            "request": allocation_request,
            "target_peers": target_peers,
            "accepted_peers": accepted_peers,
            "created_at": datetime.now(UTC),
            "status": "pending" if accepted_peers else "failed",
        }

        result = {
            "allocation_id": allocation_id,
            "requested_peers": len(target_peers),
            "accepted_peers": len(accepted_peers),
            "success_rate": len(accepted_peers) / max(1, len(target_peers)),
            "allocated_resources": accepted_peers,
        }

        logger.info(f"Federated allocation {allocation_id}: {len(accepted_peers)}/{len(target_peers)} peers accepted")

        return result

    async def update_peer_reputation(self, peer_id: str, transaction_success: bool, performance_score: float):
        """Update reputation for P2P peer based on transaction outcome"""

        if peer_id not in self.network_peers:
            return

        peer_info = self.network_peers[peer_id]

        # Update transaction counts
        if transaction_success:
            peer_info.successful_transactions += 1
        else:
            peer_info.failed_transactions += 1

        # Update reputation score (weighted average)
        total_transactions = peer_info.successful_transactions + peer_info.failed_transactions
        success_rate = peer_info.successful_transactions / max(1, total_transactions)

        # Combine success rate with performance score
        new_reputation = (success_rate * 0.7) + (performance_score * 0.3)

        # Smooth update (moving average)
        peer_info.reputation_score = (peer_info.reputation_score * 0.8) + (new_reputation * 0.2)

        # Update trust score based on reputation and other factors
        peer_info.trust_score = min(1.0, peer_info.reputation_score * 1.1)

        # Broadcast reputation update to network
        reputation_msg = P2PMarketplaceMessage(
            message_id=f"rep_{uuid.uuid4().hex[:8]}",
            message_type=P2PMessageType.REPUTATION_UPDATE,
            sender_peer_id=self.peer_id,
            sender_role=NetworkRole.MARKETPLACE_BROKER,
            payload={
                "peer_id": peer_id,
                "transaction_success": transaction_success,
                "performance_score": performance_score,
                "updated_reputation": peer_info.reputation_score,
                "updated_trust": peer_info.trust_score,
                "reporter_trust": 0.9,  # Our trust level as reporter
            },
            priority=3,
        )

        if self.p2p_network:
            await self.p2p_network.broadcast_message("marketplace", reputation_msg.payload)

        logger.info(
            f"Updated reputation for peer {peer_id}: "
            f"reputation={peer_info.reputation_score:.3f}, trust={peer_info.trust_score:.3f}"
        )

    # Message handlers

    async def _handle_p2p_message(self, sender_id: str, message_data: dict[str, Any]):
        """Handle incoming P2P marketplace message"""

        try:
            # Parse message
            msg_type = P2PMessageType(message_data.get("message_type"))

            # Get handler
            handler = self.message_handlers.get(msg_type)
            if handler:
                await handler(sender_id, message_data)
            else:
                logger.warning(f"No handler for message type: {msg_type}")

        except Exception as e:
            logger.error(f"Error handling P2P message from {sender_id}: {e}")

    async def _handle_resource_advertisement(self, sender_id: str, message_data: dict[str, Any]):
        """Handle resource advertisement from peer"""

        # Update peer information
        if sender_id not in self.network_peers:
            self.network_peers[sender_id] = P2PPeerInfo(
                peer_id=sender_id,
                roles=[NetworkRole.RESOURCE_PROVIDER],
            )

        peer_info = self.network_peers[sender_id]
        peer_info.available_resources = message_data.get("resources", {})
        peer_info.last_seen = datetime.now(UTC)

        # Update capabilities
        capabilities = message_data.get("capabilities", {})
        peer_info.uptime_percentage = capabilities.get("uptime_percentage", 95.0)
        peer_info.average_latency_ms = capabilities.get("average_latency_ms", 100.0)

        logger.debug(f"Updated resource advertisement from peer {sender_id}")

    async def _handle_resource_request(self, sender_id: str, message_data: dict[str, Any]):
        """Handle resource request from peer"""

        if not self.resource_allocator:
            return

        # Check if we can fulfill request
        requirements = message_data.get("requirements", {})

        # Create simplified ResourceRequirement
        resource_req = ResourceRequirement(
            cpu_cores=Decimal(str(requirements.get("cpu_cores", 1.0))),
            memory_gb=Decimal(str(requirements.get("memory_gb", 1.0))),
            storage_gb=Decimal(str(requirements.get("storage_gb", 10.0))),
            bandwidth_mbps=Decimal(str(requirements.get("bandwidth_mbps", 10.0))),
            duration_hours=Decimal(str(requirements.get("duration_hours", 1.0))),
        )

        # Check our available resources
        our_nodes = list(self.resource_allocator.available_nodes.values())
        suitable_nodes = [node for node in our_nodes if node.can_handle_workload(resource_req)]

        if suitable_nodes:
            # Send resource offer
            best_node = max(suitable_nodes, key=lambda n: n.calculate_quality_score())

            offer_msg = P2PMarketplaceMessage(
                message_id=f"offer_{uuid.uuid4().hex[:8]}",
                message_type=P2PMessageType.RESOURCE_ADVERTISEMENT,
                sender_peer_id=self.peer_id,
                sender_role=NetworkRole.RESOURCE_PROVIDER,
                target_peer_ids=[sender_id],
                payload={
                    "request_id": message_data.get("message_id"),
                    "offer": {
                        "cpu_cores": float(best_node.cpu_cores),
                        "memory_gb": float(best_node.memory_gb),
                        "storage_gb": float(best_node.storage_gb),
                        "bandwidth_mbps": float(best_node.bandwidth_mbps),
                        "cost_per_hour": float(best_node.cost_per_hour),
                        "quality_score": float(best_node.calculate_quality_score()),
                        "trust_score": float(best_node.trust_score),
                        "latency_ms": float(best_node.latency_ms),
                    },
                    "availability": {
                        "immediate": True,
                        "max_duration_hours": 24,
                        "commitment_level": "standard",
                    },
                },
            )

            if self.p2p_network:
                await self.p2p_network.send_message(sender_id, "marketplace", offer_msg.payload)

    async def _handle_auction_announcement(self, sender_id: str, message_data: dict[str, Any]):
        """Handle auction announcement from peer"""

        auction_info = {
            "coordinator_peer": sender_id,
            "auction_id": message_data.get("auction_id"),
            "auction_type": message_data.get("auction_type"),
            "requirements": message_data.get("requirements"),
            "reserve_price": message_data.get("reserve_price"),
            "announced_at": datetime.now(UTC),
        }

        # Decide whether to participate based on our capabilities and trust
        if (
            sender_id in self.network_peers
            and self.network_peers[sender_id].trust_score >= self.config["auction_participation_threshold"]
        ):

            # Check if we have suitable resources
            auction_info["requirements"]

            if self.resource_allocator and len(self.resource_allocator.available_nodes) > 0:

                # Consider submitting bid (simplified logic)
                logger.info(f"Considering participation in auction {auction_info['auction_id']} from {sender_id}")

                # Would implement bid calculation and submission logic here

    async def _handle_bid_submission(self, sender_id: str, message_data: dict[str, Any]):
        """Handle bid submission for our auction"""

        auction_id = message_data.get("auction_id")

        if auction_id in self.active_auctions and self.auction_engine:
            # Submit bid to local auction engine
            bid_id = await self.auction_engine.submit_bid(
                auction_id=auction_id,
                bidder_id=sender_id,
                node_id=f"{sender_id}_primary",
                bid_price=Decimal(str(message_data.get("bid_price", 0))),
                trust_score=Decimal(str(message_data.get("trust_score", 0.5))),
                reputation_score=Decimal(
                    str(self.network_peers.get(sender_id, P2PPeerInfo(peer_id=sender_id)).reputation_score)
                ),
                available_resources=message_data.get("available_resources", {}),
            )

            if bid_id:
                self.active_auctions[auction_id]["participants"].add(sender_id)
                self.active_auctions[auction_id]["bids_received"] += 1

                logger.info(f"Received bid {bid_id} from {sender_id} for auction {auction_id}")

    async def _handle_allocation_proposal(self, sender_id: str, message_data: dict[str, Any]):
        """Handle resource allocation proposal"""

        allocation_id = message_data.get("allocation_id")
        requirements = message_data.get("resource_requirements", {})
        compensation = message_data.get("compensation", {})

        # Simple acceptance logic based on our availability and trust
        accept_proposal = (
            sender_id in self.network_peers
            and self.network_peers[sender_id].trust_score >= 0.6
            and compensation.get("amount", 0) >= 5.0  # Minimum compensation
        )

        if accept_proposal:
            # Send acceptance
            acceptance_msg = P2PMarketplaceMessage(
                message_id=f"accept_{uuid.uuid4().hex[:8]}",
                message_type=P2PMessageType.ALLOCATION_ACCEPTANCE,
                sender_peer_id=self.peer_id,
                sender_role=NetworkRole.RESOURCE_PROVIDER,
                target_peer_ids=[sender_id],
                payload={
                    "allocation_id": allocation_id,
                    "accepted": True,
                    "resource_commitment": {
                        "cpu_cores": requirements.get("cpu_cores", 2),
                        "memory_gb": requirements.get("memory_gb", 4),
                        "duration_hours": requirements.get("duration_hours", 1),
                    },
                    "availability_window": {
                        "start_time": (datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
                        "end_time": (datetime.now(UTC) + timedelta(hours=25)).isoformat(),
                    },
                },
            )

            if self.p2p_network:
                await self.p2p_network.send_message(sender_id, "marketplace", acceptance_msg.payload)

            logger.info(f"Accepted allocation proposal {allocation_id} from {sender_id}")

    async def _handle_allocation_acceptance(self, sender_id: str, message_data: dict[str, Any]):
        """Handle allocation acceptance from peer"""

        allocation_id = message_data.get("allocation_id")

        if allocation_id in self.pending_allocations:
            allocation_info = self.pending_allocations[allocation_id]

            if sender_id not in allocation_info.get("accepted_peers", []):
                allocation_info["accepted_peers"].append(sender_id)

            logger.info(f"Received allocation acceptance from {sender_id} for {allocation_id}")

    async def _handle_peer_status_update(self, sender_id: str, message_data: dict[str, Any]):
        """Handle peer status update"""

        if sender_id not in self.network_peers:
            self.network_peers[sender_id] = P2PPeerInfo(
                peer_id=sender_id,
                roles=[],
            )

        peer_info = self.network_peers[sender_id]

        # Update status information
        status = message_data.get("status", {})
        peer_info.connection_quality = status.get("connection_quality", 1.0)
        peer_info.uptime_percentage = status.get("uptime_percentage", 95.0)
        peer_info.average_latency_ms = status.get("average_latency_ms", 100.0)
        peer_info.last_seen = datetime.now(UTC)

        # Update roles if provided
        if "roles" in status:
            peer_info.roles = [NetworkRole(role) for role in status["roles"]]

    async def _handle_reputation_update(self, sender_id: str, message_data: dict[str, Any]):
        """Handle reputation update from peer"""

        target_peer_id = message_data.get("peer_id")
        transaction_success = message_data.get("transaction_success", True)
        performance_score = message_data.get("performance_score", 0.5)
        reporter_trust = message_data.get("reporter_trust", 0.5)

        # Update reputation if we trust the reporter
        if (
            sender_id in self.network_peers
            and self.network_peers[sender_id].trust_score >= 0.7
            and target_peer_id in self.network_peers
        ):

            target_peer = self.network_peers[target_peer_id]

            # Weight the update by reporter trust
            weighted_performance = performance_score * reporter_trust

            # Update reputation (moving average)
            if transaction_success:
                target_peer.reputation_score = (target_peer.reputation_score * 0.9) + (weighted_performance * 0.1)
            else:
                # Penalty for failed transactions
                target_peer.reputation_score = max(0.1, target_peer.reputation_score * 0.95)

            # Update trust score
            target_peer.trust_score = min(1.0, target_peer.reputation_score * 1.05)

            logger.debug(f"Updated reputation for {target_peer_id} based on report from {sender_id}")

    # Background tasks

    async def _resource_advertisement_loop(self):
        """Periodically advertise our resources to P2P network"""

        while True:
            try:
                await asyncio.sleep(self.config["resource_advertisement_interval"])

                if self.resource_allocator and self.p2p_network:
                    # Collect our available resources
                    available_resources = {}
                    total_nodes = len(self.resource_allocator.available_nodes)

                    if total_nodes > 0:
                        # Aggregate resource information
                        total_cpu = sum(node.cpu_cores for node in self.resource_allocator.available_nodes.values())
                        total_memory = sum(node.memory_gb for node in self.resource_allocator.available_nodes.values())
                        avg_trust = (
                            sum(node.trust_score for node in self.resource_allocator.available_nodes.values())
                            / total_nodes
                        )

                        available_resources = {
                            "total_cpu_cores": float(total_cpu),
                            "total_memory_gb": float(total_memory),
                            "node_count": total_nodes,
                            "average_trust_score": float(avg_trust),
                            "supported_workloads": ["inference", "training", "general"],
                            "pricing_tiers": ["small", "medium", "large"],
                        }

                        # Create advertisement message
                        ad_msg = P2PMarketplaceMessage(
                            message_id=f"ad_{uuid.uuid4().hex[:8]}",
                            message_type=P2PMessageType.RESOURCE_ADVERTISEMENT,
                            sender_peer_id=self.peer_id,
                            sender_role=NetworkRole.RESOURCE_PROVIDER,
                            payload={
                                "resources": available_resources,
                                "capabilities": {
                                    "auction_support": True,
                                    "federated_training": True,
                                    "federated_inference": True,
                                    "uptime_percentage": 99.0,
                                    "average_latency_ms": 50.0,
                                },
                                "contact_info": {
                                    "peer_id": self.peer_id,
                                    "supported_protocols": ["marketplace_v1"],
                                },
                            },
                        )

                        await self.p2p_network.broadcast_message("marketplace", ad_msg.payload)

                        logger.debug(f"Advertised {total_nodes} nodes with {float(total_cpu)} CPU cores to P2P network")

            except Exception as e:
                logger.error(f"Error in resource advertisement loop: {e}")
                await asyncio.sleep(60)

    async def _peer_discovery_loop(self):
        """Periodically discover and update peer information"""

        while True:
            try:
                await asyncio.sleep(self.config["peer_discovery_interval"])

                # Clean up stale peer information
                stale_cutoff = datetime.now(UTC) - timedelta(minutes=10)
                stale_peers = [
                    peer_id for peer_id, peer_info in self.network_peers.items() if peer_info.last_seen < stale_cutoff
                ]

                for peer_id in stale_peers:
                    del self.network_peers[peer_id]

                if stale_peers:
                    logger.debug(f"Removed {len(stale_peers)} stale peers from network")

            except Exception as e:
                logger.error(f"Error in peer discovery loop: {e}")
                await asyncio.sleep(120)

    async def _reputation_update_loop(self):
        """Periodically update and sync peer reputation"""

        while True:
            try:
                await asyncio.sleep(self.config["reputation_update_interval"])

                # Update our reputation scores based on recent interactions
                # This would integrate with actual transaction history

                # Broadcast our peer status
                await self._announce_peer_capabilities()

            except Exception as e:
                logger.error(f"Error in reputation update loop: {e}")
                await asyncio.sleep(300)

    # Helper methods

    async def _announce_peer_capabilities(self):
        """Announce our capabilities to P2P network"""

        if not self.p2p_network:
            return

        # Determine our roles
        roles = [NetworkRole.MARKETPLACE_BROKER]

        if self.resource_allocator and self.resource_allocator.available_nodes:
            roles.append(NetworkRole.RESOURCE_PROVIDER)

        if self.auction_engine:
            roles.append(NetworkRole.AUCTION_COORDINATOR)

        # Create status update message
        status_msg = P2PMarketplaceMessage(
            message_id=f"status_{uuid.uuid4().hex[:8]}",
            message_type=P2PMessageType.PEER_STATUS_UPDATE,
            sender_peer_id=self.peer_id,
            sender_role=NetworkRole.MARKETPLACE_BROKER,
            payload={
                "status": {
                    "roles": [role.value for role in roles],
                    "connection_quality": 1.0,
                    "uptime_percentage": 99.0,
                    "average_latency_ms": 30.0,
                    "supported_features": {
                        "federated_auctions": True,
                        "resource_discovery": True,
                        "reputation_tracking": True,
                        "qos_guarantees": True,
                    },
                },
                "marketplace_version": "1.0.0",
            },
        )

        await self.p2p_network.broadcast_message("marketplace", status_msg.payload)

    def _create_resource_node_from_peer(self, peer_info: P2PPeerInfo) -> ResourceNode | None:
        """Create ResourceNode from P2P peer information"""

        resources = peer_info.available_resources

        if not resources:
            return None

        try:
            return ResourceNode(
                node_id=f"{peer_info.peer_id}_aggregate",
                node_type=ResourceType.EDGE_SERVER,  # Default type
                cpu_cores=Decimal(str(resources.get("total_cpu_cores", 1))),
                memory_gb=Decimal(str(resources.get("total_memory_gb", 1))),
                storage_gb=Decimal("100.0"),  # Default
                bandwidth_mbps=Decimal("100.0"),  # Default
                trust_score=Decimal(str(peer_info.trust_score)),
                reputation_score=Decimal(str(peer_info.reputation_score)),
                availability_score=Decimal(str(peer_info.uptime_percentage / 100)),
                latency_ms=Decimal(str(peer_info.average_latency_ms)),
                cost_per_hour=Decimal("5.0"),  # Default pricing
                region="p2p_network",
            )

        except Exception as e:
            logger.error(f"Error creating resource node from peer {peer_info.peer_id}: {e}")
            return None


# Global P2P integration instance
_p2p_integration: P2PMarketplaceIntegration | None = None


def get_p2p_integration(peer_id: str = None, p2p_network=None, **marketplace_components) -> P2PMarketplaceIntegration:
    """Get global P2P marketplace integration instance"""
    global _p2p_integration

    if _p2p_integration is None:
        _p2p_integration = P2PMarketplaceIntegration(
            peer_id=peer_id or f"marketplace_{uuid.uuid4().hex[:8]}", p2p_network=p2p_network, **marketplace_components
        )

    return _p2p_integration
