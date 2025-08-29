"""
Betanet Blockchain Anchor Service

Provides blockchain anchoring for cryptographic proofs with fraud detection
and immutable proof storage on the Betanet network.
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
import time
from typing import Any

from ..bridges.betanet_integration import BetaNetFogTransport, is_betanet_available

logger = logging.getLogger(__name__)


class AnchorStatus(Enum):
    """Blockchain anchor status"""

    PENDING = "pending"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    REVOKED = "revoked"


@dataclass
class BlockchainAnchor:
    """Blockchain anchor record for proof"""

    anchor_id: str
    proof_hash: str
    merkle_root: str
    block_height: int | None = None
    transaction_id: str | None = None
    anchor_timestamp: float = field(default_factory=time.time)
    confirmation_timestamp: float | None = None
    status: AnchorStatus = AnchorStatus.PENDING
    gas_cost: float | None = None
    confirmation_count: int = 0
    fraud_checks: list[str] = field(default_factory=list)


@dataclass
class FraudAlert:
    """Fraud detection alert"""

    alert_id: str
    proof_hash: str
    anchor_id: str
    fraud_type: str  # "double_spend", "replay_attack", "invalid_signature"
    confidence: float  # 0.0 to 1.0
    evidence: dict[str, Any]
    detected_at: float = field(default_factory=time.time)
    investigator_node: str | None = None


class BetanetAnchorService:
    """
    Betanet Blockchain Anchor Service for fog computing proofs

    Features:
    - Immutable proof anchoring to Betanet blockchain
    - Fraud detection and prevention mechanisms
    - Batch anchoring for cost optimization
    - Cross-chain verification support
    - Compliance audit trail maintenance
    """

    def __init__(self, node_id: str, enable_fraud_detection: bool = True):
        self.node_id = node_id
        self.enable_fraud_detection = enable_fraud_detection
        self.transport: BetaNetFogTransport | None = None

        # Storage
        self.anchors: dict[str, BlockchainAnchor] = {}
        self.fraud_alerts: dict[str, FraudAlert] = {}
        self.pending_batches: list[dict[str, Any]] = []

        # Configuration
        self.config = {
            "batch_size": 50,  # Max proofs per batch transaction
            "confirmation_blocks": 6,  # Required confirmations
            "fraud_check_interval": 300,  # 5 minutes
            "max_gas_price": 100,  # Gas price limit
            "retry_attempts": 3,
            "anchor_timeout": 3600,  # 1 hour timeout
        }

        # Statistics
        self.stats = {
            "anchors_created": 0,
            "confirmations_received": 0,
            "fraud_alerts_generated": 0,
            "batch_transactions": 0,
            "failed_anchors": 0,
            "total_gas_spent": 0.0,
        }

        # Initialize transport if Betanet available
        self._initialize_transport()

        logger.info(f"Betanet anchor service initialized for node {node_id}")

    def _initialize_transport(self):
        """Initialize Betanet transport for blockchain communication"""
        if is_betanet_available():
            self.transport = BetaNetFogTransport(
                privacy_mode="balanced",
                enable_covert=False,  # Use standard transport for blockchain
                mobile_optimization=True,
            )
            logger.info("Betanet transport initialized for anchoring")
        else:
            logger.warning("Betanet not available - anchoring will use simulation mode")

    async def anchor_proof(self, proof_hash: str, merkle_root: str, priority: str = "normal") -> str:
        """
        Anchor a single proof to Betanet blockchain

        Args:
            proof_hash: Hash of the proof to anchor
            merkle_root: Merkle root if part of batch
            priority: Anchoring priority ("low", "normal", "high")

        Returns:
            Anchor ID for tracking
        """
        anchor_id = f"anchor_{self.node_id}_{int(time.time())}_{len(self.anchors)}"

        try:
            # Create anchor record
            anchor = BlockchainAnchor(anchor_id=anchor_id, proof_hash=proof_hash, merkle_root=merkle_root)

            # Perform fraud checks before anchoring
            if self.enable_fraud_detection:
                fraud_detected = await self._check_for_fraud(proof_hash, anchor_id)
                if fraud_detected:
                    anchor.status = AnchorStatus.FAILED
                    anchor.fraud_checks.append("pre_anchor_fraud_detected")
                    self.anchors[anchor_id] = anchor
                    return anchor_id

            # Submit to blockchain
            if priority == "high":
                # Individual transaction for high priority
                success = await self._submit_individual_anchor(anchor)
            else:
                # Add to batch for normal/low priority
                success = await self._add_to_batch(anchor, priority)

            if success:
                self.anchors[anchor_id] = anchor
                self.stats["anchors_created"] += 1
                logger.info(f"Anchored proof {proof_hash[:16]}... as {anchor_id}")
            else:
                anchor.status = AnchorStatus.FAILED
                self.anchors[anchor_id] = anchor
                self.stats["failed_anchors"] += 1
                logger.error(f"Failed to anchor proof {proof_hash}")

            return anchor_id

        except Exception as e:
            logger.error(f"Anchor creation failed: {e}")
            # Create failed anchor record for audit trail
            failed_anchor = BlockchainAnchor(
                anchor_id=anchor_id, proof_hash=proof_hash, merkle_root=merkle_root, status=AnchorStatus.FAILED
            )
            self.anchors[anchor_id] = failed_anchor
            return anchor_id

    async def _submit_individual_anchor(self, anchor: BlockchainAnchor) -> bool:
        """Submit individual anchor transaction to blockchain"""
        if not self.transport:
            # Simulation mode
            logger.info(f"SIMULATION: Anchoring {anchor.anchor_id} to blockchain")
            await asyncio.sleep(0.1)  # Simulate network delay

            anchor.status = AnchorStatus.CONFIRMED
            anchor.transaction_id = f"sim_tx_{anchor.anchor_id}"
            anchor.block_height = 1000000 + len(self.anchors)
            anchor.confirmation_timestamp = time.time()
            anchor.gas_cost = 0.01  # Simulated gas cost

            return True

        try:
            # Prepare transaction data
            tx_data = {
                "type": "proof_anchor",
                "anchor_id": anchor.anchor_id,
                "proof_hash": anchor.proof_hash,
                "merkle_root": anchor.merkle_root,
                "node_id": self.node_id,
                "timestamp": anchor.anchor_timestamp,
            }

            # Send to blockchain via Betanet transport
            tx_payload = json.dumps(tx_data).encode()
            result = await self.transport.send_job_data(tx_payload, "blockchain_node", priority="high")

            if result["success"]:
                anchor.transaction_id = f"betanet_tx_{int(time.time())}"
                self.stats["total_gas_spent"] += result.get("gas_cost", 0.01)
                logger.info(f"Submitted anchor transaction: {anchor.transaction_id}")

                # Start confirmation monitoring
                asyncio.create_task(self._monitor_confirmation(anchor))
                return True
            else:
                logger.error(f"Blockchain transaction failed: {result.get('error')}")
                return False

        except Exception as e:
            logger.error(f"Individual anchor submission failed: {e}")
            return False

    async def _add_to_batch(self, anchor: BlockchainAnchor, priority: str) -> bool:
        """Add anchor to batch for cost-efficient submission"""
        batch_entry = {"anchor": anchor, "priority": priority, "added_at": time.time()}

        self.pending_batches.append(batch_entry)

        # Check if batch is ready for submission
        if len(self.pending_batches) >= self.config["batch_size"]:
            await self._submit_batch()

        return True

    async def _submit_batch(self) -> bool:
        """Submit batch of anchors as single blockchain transaction"""
        if not self.pending_batches:
            return False

        batch = self.pending_batches[: self.config["batch_size"]]
        self.pending_batches = self.pending_batches[self.config["batch_size"] :]

        try:
            # Create batch Merkle tree from all anchors
            anchor_hashes = [entry["anchor"].proof_hash for entry in batch]
            batch_merkle_root = await self._calculate_batch_merkle_root(anchor_hashes)

            batch_id = f"batch_{self.node_id}_{int(time.time())}"

            if not self.transport:
                # Simulation mode
                logger.info(f"SIMULATION: Submitting batch {batch_id} with {len(batch)} anchors")
                await asyncio.sleep(0.2)

                # Mark all anchors in batch as confirmed
                for entry in batch:
                    anchor = entry["anchor"]
                    anchor.status = AnchorStatus.CONFIRMED
                    anchor.transaction_id = f"sim_batch_tx_{batch_id}"
                    anchor.block_height = 1000000 + len(self.anchors)
                    anchor.confirmation_timestamp = time.time()
                    anchor.gas_cost = 0.001  # Lower per-proof cost in batch

                self.stats["batch_transactions"] += 1
                return True

            # Prepare batch transaction
            batch_data = {
                "type": "proof_batch",
                "batch_id": batch_id,
                "merkle_root": batch_merkle_root,
                "anchor_count": len(batch),
                "anchors": [
                    {"anchor_id": entry["anchor"].anchor_id, "proof_hash": entry["anchor"].proof_hash}
                    for entry in batch
                ],
                "node_id": self.node_id,
                "timestamp": time.time(),
            }

            # Send batch to blockchain
            batch_payload = json.dumps(batch_data).encode()
            result = await self.transport.send_job_data(batch_payload, "blockchain_node", priority="normal")

            if result["success"]:
                batch_tx_id = f"betanet_batch_{batch_id}"
                gas_per_anchor = result.get("gas_cost", 0.01) / len(batch)

                # Update all anchors in batch
                for entry in batch:
                    anchor = entry["anchor"]
                    anchor.transaction_id = batch_tx_id
                    anchor.gas_cost = gas_per_anchor

                self.stats["batch_transactions"] += 1
                self.stats["total_gas_spent"] += result.get("gas_cost", 0.01)

                # Start confirmation monitoring for batch
                asyncio.create_task(self._monitor_batch_confirmation(batch, batch_tx_id))

                logger.info(f"Submitted batch transaction {batch_tx_id} with {len(batch)} anchors")
                return True
            else:
                logger.error(f"Batch transaction failed: {result.get('error')}")
                # Mark all anchors as failed
                for entry in batch:
                    entry["anchor"].status = AnchorStatus.FAILED
                return False

        except Exception as e:
            logger.error(f"Batch submission failed: {e}")
            return False

    async def _calculate_batch_merkle_root(self, hashes: list[str]) -> str:
        """Calculate Merkle root for batch of proof hashes"""
        if not hashes:
            return ""

        current_level = hashes[:]

        while len(current_level) > 1:
            next_level = []

            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left

                import hashlib

                combined = left + right
                parent = hashlib.sha256(combined.encode()).hexdigest()
                next_level.append(parent)

            current_level = next_level

        return current_level[0]

    async def _monitor_confirmation(self, anchor: BlockchainAnchor):
        """Monitor blockchain confirmation for individual anchor"""
        max_wait_time = self.config["anchor_timeout"]
        start_time = time.time()

        while (time.time() - start_time) < max_wait_time:
            await asyncio.sleep(30)  # Check every 30 seconds

            # Simulate confirmation checks
            if not self.transport:
                # In simulation, confirm after delay
                if time.time() - start_time > 60:  # 1 minute delay
                    anchor.status = AnchorStatus.CONFIRMED
                    anchor.confirmation_timestamp = time.time()
                    anchor.confirmation_count = self.config["confirmation_blocks"]
                    self.stats["confirmations_received"] += 1
                    logger.info(f"Anchor {anchor.anchor_id} confirmed")
                    break
            else:
                # Check with actual blockchain
                confirmed = await self._check_transaction_confirmation(anchor.transaction_id)
                if confirmed:
                    anchor.status = AnchorStatus.CONFIRMED
                    anchor.confirmation_timestamp = time.time()
                    anchor.confirmation_count = self.config["confirmation_blocks"]
                    self.stats["confirmations_received"] += 1
                    break

        # Handle timeout
        if anchor.status != AnchorStatus.CONFIRMED:
            logger.warning(f"Anchor {anchor.anchor_id} confirmation timeout")

    async def _monitor_batch_confirmation(self, batch: list[dict], batch_tx_id: str):
        """Monitor blockchain confirmation for anchor batch"""
        max_wait_time = self.config["anchor_timeout"]
        start_time = time.time()

        while (time.time() - start_time) < max_wait_time:
            await asyncio.sleep(45)  # Check every 45 seconds for batches

            if not self.transport:
                # Simulation mode
                if time.time() - start_time > 90:  # 1.5 minute delay for batches
                    for entry in batch:
                        anchor = entry["anchor"]
                        anchor.status = AnchorStatus.CONFIRMED
                        anchor.confirmation_timestamp = time.time()
                        anchor.confirmation_count = self.config["confirmation_blocks"]
                        self.stats["confirmations_received"] += 1
                    logger.info(f"Batch {batch_tx_id} confirmed with {len(batch)} anchors")
                    break
            else:
                confirmed = await self._check_transaction_confirmation(batch_tx_id)
                if confirmed:
                    for entry in batch:
                        anchor = entry["anchor"]
                        anchor.status = AnchorStatus.CONFIRMED
                        anchor.confirmation_timestamp = time.time()
                        anchor.confirmation_count = self.config["confirmation_blocks"]
                        self.stats["confirmations_received"] += 1
                    break

    async def _check_transaction_confirmation(self, tx_id: str | None) -> bool:
        """Check if transaction is confirmed on blockchain"""
        if not tx_id or not self.transport:
            return False

        try:
            # Query blockchain for transaction status
            query_data = {"type": "tx_status", "tx_id": tx_id}
            query_payload = json.dumps(query_data).encode()

            result = await self.transport.send_job_data(query_payload, "blockchain_query_node", priority="low")

            return result.get("success", False)

        except Exception as e:
            logger.error(f"Confirmation check failed: {e}")
            return False

    async def _check_for_fraud(self, proof_hash: str, anchor_id: str) -> bool:
        """Check for potential fraud in proof before anchoring"""
        if not self.enable_fraud_detection:
            return False

        fraud_indicators = []

        # Check for duplicate proof hash (replay attack)
        duplicate_count = sum(
            1
            for anchor in self.anchors.values()
            if anchor.proof_hash == proof_hash and anchor.status == AnchorStatus.CONFIRMED
        )

        if duplicate_count > 0:
            fraud_indicators.append("duplicate_proof_hash")

        # Check for rapid submission patterns (potential spam)
        recent_anchors = [
            anchor for anchor in self.anchors.values() if time.time() - anchor.anchor_timestamp < 60  # Last minute
        ]

        if len(recent_anchors) > 10:  # More than 10 in last minute
            fraud_indicators.append("rapid_submission_pattern")

        # Generate fraud alert if indicators found
        if fraud_indicators:
            alert_id = f"fraud_{anchor_id}_{int(time.time())}"
            alert = FraudAlert(
                alert_id=alert_id,
                proof_hash=proof_hash,
                anchor_id=anchor_id,
                fraud_type="|".join(fraud_indicators),
                confidence=min(1.0, len(fraud_indicators) * 0.3),
                evidence={
                    "duplicate_count": duplicate_count,
                    "recent_anchor_count": len(recent_anchors),
                    "indicators": fraud_indicators,
                },
                investigator_node=self.node_id,
            )

            self.fraud_alerts[alert_id] = alert
            self.stats["fraud_alerts_generated"] += 1

            logger.warning(
                f"Fraud detected for proof {proof_hash}: {fraud_indicators} " f"(confidence: {alert.confidence:.2f})"
            )

            return alert.confidence > 0.7  # Block if high confidence

        return False

    async def get_anchor_status(self, anchor_id: str) -> dict[str, Any] | None:
        """Get status of specific anchor"""
        if anchor_id not in self.anchors:
            return None

        anchor = self.anchors[anchor_id]
        return {
            "anchor_id": anchor.anchor_id,
            "proof_hash": anchor.proof_hash,
            "status": anchor.status.value,
            "transaction_id": anchor.transaction_id,
            "block_height": anchor.block_height,
            "confirmation_count": anchor.confirmation_count,
            "gas_cost": anchor.gas_cost,
            "anchor_timestamp": anchor.anchor_timestamp,
            "confirmation_timestamp": anchor.confirmation_timestamp,
        }

    async def force_batch_submission(self):
        """Force submission of pending batch even if not full"""
        if self.pending_batches:
            logger.info(f"Force submitting batch with {len(self.pending_batches)} anchors")
            await self._submit_batch()

    def get_fraud_alerts(self) -> list[dict[str, Any]]:
        """Get all fraud alerts"""
        return [
            {
                "alert_id": alert.alert_id,
                "proof_hash": alert.proof_hash,
                "fraud_type": alert.fraud_type,
                "confidence": alert.confidence,
                "detected_at": alert.detected_at,
                "evidence": alert.evidence,
            }
            for alert in self.fraud_alerts.values()
        ]

    def get_statistics(self) -> dict[str, Any]:
        """Get anchor service statistics"""
        return {
            **self.stats,
            "pending_batch_size": len(self.pending_batches),
            "total_anchors": len(self.anchors),
            "confirmed_anchors": len([a for a in self.anchors.values() if a.status == AnchorStatus.CONFIRMED]),
            "fraud_alerts": len(self.fraud_alerts),
            "betanet_available": is_betanet_available(),
        }
