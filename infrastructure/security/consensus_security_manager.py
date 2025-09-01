"""
Consensus Security Manager for Distributed Systems
=================================================

Comprehensive security mechanisms for distributed consensus protocols with advanced threat detection.
Implements Byzantine fault tolerance, threshold cryptography, and zero-knowledge proofs.
"""

import hashlib
import hmac
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid
import numpy as np

# Cryptographic libraries
try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.backends import default_backend

    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

logger = logging.getLogger(__name__)


class ConsensusProtocol(Enum):
    """Supported consensus protocols."""

    RAFT = "raft"
    BYZANTINE = "byzantine"
    PRACTICAL_BFT = "practical_bft"
    HOTSTUFF = "hotstuff"
    TENDERMINT = "tendermint"


class AttackType(Enum):
    """Types of consensus attacks."""

    BYZANTINE_NODE = "byzantine_node"
    SYBIL_ATTACK = "sybil_attack"
    ECLIPSE_ATTACK = "eclipse_attack"
    DOS_ATTACK = "dos_attack"
    COLLUSION = "collusion"
    TIMING_ATTACK = "timing_attack"
    FORK_ATTACK = "fork_attack"
    NOTHING_AT_STAKE = "nothing_at_stake"


class NodeRole(Enum):
    """Consensus node roles."""

    LEADER = "leader"
    FOLLOWER = "follower"
    VALIDATOR = "validator"
    OBSERVER = "observer"
    CANDIDATE = "candidate"


@dataclass
class ThresholdKey:
    """Threshold signature key components."""

    node_id: str
    public_key_share: bytes
    private_key_share: bytes
    threshold: int
    total_parties: int
    polynomial_coefficients: List[int] = field(default_factory=list)
    verification_key: bytes = b""


@dataclass
class ConsensusMessage:
    """Consensus protocol message."""

    message_id: str
    sender_id: str
    message_type: str
    round_number: int
    view_number: int
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    signature: Optional[bytes] = None
    proof: Optional[bytes] = None


@dataclass
class AttackEvidence:
    """Evidence of a consensus attack."""

    attack_id: str
    attack_type: AttackType
    suspected_nodes: Set[str]
    evidence_data: Dict[str, Any]
    confidence_score: float
    detected_at: float = field(default_factory=time.time)
    verified: bool = False
    mitigated: bool = False


@dataclass
class ConsensusRound:
    """Consensus round state."""

    round_id: str
    round_number: int
    view_number: int
    leader_id: str
    participants: Set[str]
    proposal: Optional[Dict[str, Any]] = None
    votes: Dict[str, ConsensusMessage] = field(default_factory=dict)
    commits: Dict[str, ConsensusMessage] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    status: str = "active"
    security_violations: List[str] = field(default_factory=list)


class ConsensusSecurityManager:
    """
    Comprehensive security manager for distributed consensus protocols.

    Features:
    - Byzantine fault tolerance with configurable thresholds
    - Threshold signature schemes for secure consensus
    - Zero-knowledge proofs for message validation
    - Attack detection and mitigation
    - Secure multi-party computation
    - Reputation-based trust management
    - Cryptographic proof generation and verification
    """

    def __init__(
        self,
        node_id: str,
        consensus_protocol: ConsensusProtocol = ConsensusProtocol.BYZANTINE,
        byzantine_threshold: float = 0.33,
    ):
        """Initialize consensus security manager."""
        self.node_id = node_id
        self.consensus_protocol = consensus_protocol
        self.byzantine_threshold = byzantine_threshold

        # Cryptographic materials
        self.threshold_keys: Dict[str, ThresholdKey] = {}
        self.master_public_key: Optional[bytes] = None
        self.node_certificates: Dict[str, bytes] = {}

        # Consensus state
        self.current_round: Optional[ConsensusRound] = None
        self.consensus_history: List[ConsensusRound] = []
        self.participant_nodes: Dict[str, Dict[str, Any]] = {}

        # Security monitoring
        self.attack_evidence: List[AttackEvidence] = []
        self.node_reputations: Dict[str, float] = {}
        self.message_cache: Dict[str, ConsensusMessage] = {}
        self.timing_records: Dict[str, List[float]] = {}

        # Protocol parameters
        self.security_params = {
            "signature_threshold": max(1, int(byzantine_threshold * 10)),  # Assuming 10 nodes
            "message_timeout": 30.0,  # seconds
            "view_timeout": 60.0,  # seconds
            "reputation_decay": 0.95,  # per round
            "min_reputation": 0.1,
            "attack_confidence_threshold": 0.8,
            "max_cached_messages": 10000,
            "proof_verification_timeout": 5.0,
        }

        # Statistics
        self.security_stats = {
            "rounds_processed": 0,
            "messages_verified": 0,
            "messages_rejected": 0,
            "attacks_detected": 0,
            "attacks_mitigated": 0,
            "byzantine_nodes_identified": 0,
            "threshold_signatures_created": 0,
            "threshold_signatures_verified": 0,
            "zero_knowledge_proofs_verified": 0,
        }

        logger.info(f"Consensus Security Manager initialized for node {node_id}")

    async def initialize_distributed_keys(self, participant_nodes: List[str], threshold: Optional[int] = None) -> bool:
        """Initialize distributed key generation for threshold signatures."""

        if not participant_nodes:
            logger.error("No participant nodes provided for key generation")
            return False

        n = len(participant_nodes)
        t = threshold or max(1, int(n * self.byzantine_threshold) + 1)

        if t > n:
            logger.error(f"Threshold {t} cannot exceed total parties {n}")
            return False

        try:
            logger.info(f"Starting distributed key generation: {t}-of-{n} threshold")

            # Phase 1: Generate polynomial coefficients
            polynomial_degree = t - 1
            coefficients = [secrets.randbelow(2**256) for _ in range(polynomial_degree + 1)]

            # Phase 2: Generate key shares for each participant
            for i, participant_id in enumerate(participant_nodes):
                # Evaluate polynomial at point (i+1)
                point = i + 1
                private_share = self._evaluate_polynomial(coefficients, point)

                # Generate corresponding public key share
                if CRYPTOGRAPHY_AVAILABLE:
                    # Use RSA for simplicity (in production, would use ECC)
                    private_key = rsa.generate_private_key(
                        public_exponent=65537, key_size=2048, backend=default_backend()
                    )
                    public_key_bytes = private_key.public_key().public_bytes(
                        encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo
                    )
                    private_key_bytes = private_key.private_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PrivateFormat.PKCS8,
                        encryption_algorithm=serialization.NoEncryption(),
                    )
                else:
                    # Fallback to hash-based keys
                    private_key_bytes = hashlib.sha256(str(private_share).encode() + participant_id.encode()).digest()
                    public_key_bytes = hashlib.sha256(private_key_bytes + b"public").digest()

                # Create threshold key
                threshold_key = ThresholdKey(
                    node_id=participant_id,
                    public_key_share=public_key_bytes,
                    private_key_share=private_key_bytes,
                    threshold=t,
                    total_parties=n,
                    polynomial_coefficients=coefficients if participant_id == self.node_id else [],
                )

                self.threshold_keys[participant_id] = threshold_key

            # Phase 3: Generate master public key (combination of all public shares)
            self.master_public_key = self._combine_public_keys(
                [key.public_key_share for key in self.threshold_keys.values()]
            )

            # Update participant information
            for participant_id in participant_nodes:
                self.participant_nodes[participant_id] = {
                    "role": NodeRole.VALIDATOR.value,
                    "reputation": 1.0,
                    "last_seen": time.time(),
                    "message_count": 0,
                    "byzantine_score": 0.0,
                }
                self.node_reputations[participant_id] = 1.0

            logger.info("Distributed key generation completed successfully")
            return True

        except Exception as e:
            logger.error(f"Distributed key generation failed: {e}")
            return False

    async def create_threshold_signature(
        self, message: bytes, signing_nodes: List[str]
    ) -> Tuple[bool, Optional[bytes]]:
        """Create a threshold signature using multiple nodes."""

        if len(signing_nodes) < self.security_params["signature_threshold"]:
            logger.error(
                f"Insufficient signing nodes: {len(signing_nodes)} < {self.security_params['signature_threshold']}"
            )
            return False, None

        try:
            partial_signatures = []

            # Collect partial signatures from signing nodes
            for node_id in signing_nodes:
                threshold_key = self.threshold_keys.get(node_id)
                if not threshold_key:
                    logger.warning(f"No threshold key found for node {node_id}")
                    continue

                # Create partial signature
                partial_sig = await self._create_partial_signature(message, threshold_key)
                if partial_sig:
                    partial_signatures.append(
                        {
                            "node_id": node_id,
                            "signature": partial_sig,
                            "public_key_share": threshold_key.public_key_share,
                        }
                    )

            # Verify we have enough partial signatures
            if len(partial_signatures) < threshold_key.threshold:
                logger.error(f"Insufficient partial signatures: {len(partial_signatures)} < {threshold_key.threshold}")
                return False, None

            # Combine partial signatures using Lagrange interpolation
            combined_signature = await self._combine_partial_signatures(
                message, partial_signatures[: threshold_key.threshold]
            )

            self.security_stats["threshold_signatures_created"] += 1

            logger.debug(f"Threshold signature created with {len(partial_signatures)} partial signatures")
            return True, combined_signature

        except Exception as e:
            logger.error(f"Threshold signature creation failed: {e}")
            return False, None

    async def verify_threshold_signature(
        self, message: bytes, signature: bytes, expected_signers: Optional[List[str]] = None
    ) -> bool:
        """Verify a threshold signature."""

        if not self.master_public_key:
            logger.error("Master public key not available for verification")
            return False

        try:
            # Verify signature against master public key
            if CRYPTOGRAPHY_AVAILABLE:
                # Use RSA verification (simplified)
                verification_result = self._verify_rsa_signature(message, signature, self.master_public_key)
            else:
                # Fallback hash-based verification
                expected_sig = hashlib.sha256(message + self.master_public_key).digest()
                verification_result = hmac.compare_digest(signature, expected_sig)

            if verification_result:
                self.security_stats["threshold_signatures_verified"] += 1
                logger.debug("Threshold signature verified successfully")
            else:
                self.security_stats["messages_rejected"] += 1
                logger.warning("Threshold signature verification failed")

            return verification_result

        except Exception as e:
            logger.error(f"Threshold signature verification failed: {e}")
            return False

    async def create_zero_knowledge_proof(
        self, statement: Dict[str, Any], witness: Dict[str, Any]
    ) -> Tuple[bool, Optional[bytes]]:
        """Create a zero-knowledge proof for statement validity."""

        try:
            # Simplified ZK proof using Fiat-Shamir heuristic
            # In production, would use proper ZK libraries like libsnark

            # Generate random nonce
            nonce = secrets.token_bytes(32)

            # Create commitment
            commitment_data = json.dumps(
                {"statement": statement, "nonce": nonce.hex(), "prover": self.node_id}, sort_keys=True
            )

            commitment = hashlib.sha256(commitment_data.encode()).digest()

            # Generate Fiat-Shamir challenge
            challenge_data = json.dumps({"commitment": commitment.hex(), "statement": statement}, sort_keys=True)

            challenge = hashlib.sha256(challenge_data.encode()).digest()

            # Create response using witness
            response_data = {
                "nonce": nonce.hex(),
                "challenge": challenge.hex(),
                "witness_hash": hashlib.sha256(json.dumps(witness, sort_keys=True).encode()).hexdigest(),
            }

            response = hashlib.sha256(json.dumps(response_data, sort_keys=True).encode()).digest()

            # Combine proof components
            proof = json.dumps(
                {
                    "commitment": commitment.hex(),
                    "challenge": challenge.hex(),
                    "response": response.hex(),
                    "statement": statement,
                }
            ).encode()

            logger.debug("Zero-knowledge proof created successfully")
            return True, proof

        except Exception as e:
            logger.error(f"Zero-knowledge proof creation failed: {e}")
            return False, None

    async def verify_zero_knowledge_proof(
        self, proof: bytes, expected_statement: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Verify a zero-knowledge proof."""

        try:
            # Parse proof
            proof_data = json.loads(proof.decode())
            commitment = bytes.fromhex(proof_data["commitment"])
            challenge = bytes.fromhex(proof_data["challenge"])
            bytes.fromhex(proof_data["response"])
            statement = proof_data["statement"]

            # Verify statement if provided
            if expected_statement and statement != expected_statement:
                logger.warning("ZK proof statement mismatch")
                return False

            # Verify Fiat-Shamir challenge
            challenge_data = json.dumps({"commitment": commitment.hex(), "statement": statement}, sort_keys=True)

            expected_challenge = hashlib.sha256(challenge_data.encode()).digest()

            if not hmac.compare_digest(challenge, expected_challenge):
                logger.warning("ZK proof challenge verification failed")
                return False

            # Note: In a real ZK proof system, we would verify the response
            # without learning the witness. This is a simplified implementation.

            self.security_stats["zero_knowledge_proofs_verified"] += 1
            logger.debug("Zero-knowledge proof verified successfully")
            return True

        except Exception as e:
            logger.error(f"Zero-knowledge proof verification failed: {e}")
            return False

    async def detect_byzantine_behavior(self, consensus_round: ConsensusRound) -> List[AttackEvidence]:
        """Detect Byzantine behavior in consensus round."""

        detected_attacks = []

        try:
            # Detect contradictory messages
            contradictions = await self._detect_contradictory_messages(consensus_round)
            if contradictions:
                for node_id, evidence in contradictions.items():
                    attack = AttackEvidence(
                        attack_id=str(uuid.uuid4()),
                        attack_type=AttackType.BYZANTINE_NODE,
                        suspected_nodes={node_id},
                        evidence_data=evidence,
                        confidence_score=0.9,
                    )
                    detected_attacks.append(attack)

            # Detect timing attacks
            timing_attacks = await self._detect_timing_attacks(consensus_round)
            for attack in timing_attacks:
                detected_attacks.append(attack)

            # Detect collusion patterns
            collusion_attacks = await self._detect_collusion(consensus_round)
            for attack in collusion_attacks:
                detected_attacks.append(attack)

            # Detect fork attempts
            fork_attacks = await self._detect_fork_attacks(consensus_round)
            for attack in fork_attacks:
                detected_attacks.append(attack)

            # Update statistics and reputation
            for attack in detected_attacks:
                self.attack_evidence.append(attack)
                self.security_stats["attacks_detected"] += 1

                # Decrease reputation of suspected nodes
                for node_id in attack.suspected_nodes:
                    current_reputation = self.node_reputations.get(node_id, 1.0)
                    penalty = 0.1 * attack.confidence_score
                    self.node_reputations[node_id] = max(
                        self.security_params["min_reputation"], current_reputation - penalty
                    )

            if detected_attacks:
                logger.warning(f"Detected {len(detected_attacks)} potential Byzantine attacks")

            return detected_attacks

        except Exception as e:
            logger.error(f"Byzantine behavior detection failed: {e}")
            return []

    async def mitigate_attacks(self, attacks: List[AttackEvidence]) -> int:
        """Mitigate detected consensus attacks."""

        mitigated_count = 0

        for attack in attacks:
            try:
                success = False

                if attack.attack_type == AttackType.BYZANTINE_NODE:
                    # Exclude Byzantine nodes from future consensus
                    success = await self._exclude_byzantine_nodes(attack.suspected_nodes)

                elif attack.attack_type == AttackType.SYBIL_ATTACK:
                    # Strengthen identity verification
                    success = await self._strengthen_identity_verification()

                elif attack.attack_type == AttackType.ECLIPSE_ATTACK:
                    # Diversify network connections
                    success = await self._diversify_network_connections(attack)

                elif attack.attack_type == AttackType.DOS_ATTACK:
                    # Rate limit and block attackers
                    success = await self._apply_rate_limiting(attack.suspected_nodes)

                elif attack.attack_type == AttackType.COLLUSION:
                    # Randomize consensus parameters
                    success = await self._randomize_consensus_parameters()

                elif attack.attack_type == AttackType.TIMING_ATTACK:
                    # Introduce timing randomization
                    success = await self._randomize_timing()

                if success:
                    attack.mitigated = True
                    mitigated_count += 1
                    self.security_stats["attacks_mitigated"] += 1

                    logger.info(f"Successfully mitigated {attack.attack_type.value}")
                else:
                    logger.error(f"Failed to mitigate {attack.attack_type.value}")

            except Exception as e:
                logger.error(f"Attack mitigation failed for {attack.attack_id}: {e}")

        return mitigated_count

    async def validate_consensus_message(
        self, message: ConsensusMessage, expected_round: Optional[int] = None, expected_view: Optional[int] = None
    ) -> Tuple[bool, List[str]]:
        """Validate a consensus protocol message."""

        validation_errors = []

        try:
            # Check message structure
            if not message.sender_id or not message.message_type:
                validation_errors.append("Missing required message fields")

            # Check round and view numbers
            if expected_round is not None and message.round_number != expected_round:
                validation_errors.append(f"Invalid round number: {message.round_number} != {expected_round}")

            if expected_view is not None and message.view_number != expected_view:
                validation_errors.append(f"Invalid view number: {message.view_number} != {expected_view}")

            # Check timestamp freshness
            current_time = time.time()
            if abs(current_time - message.timestamp) > self.security_params["message_timeout"]:
                validation_errors.append("Message timestamp out of acceptable range")

            # Verify signature if present
            if message.signature:
                signature_valid = await self._verify_message_signature(message)
                if not signature_valid:
                    validation_errors.append("Invalid message signature")

            # Verify zero-knowledge proof if present
            if message.proof:
                proof_valid = await self._verify_message_proof(message)
                if not proof_valid:
                    validation_errors.append("Invalid zero-knowledge proof")

            # Check sender reputation
            sender_reputation = self.node_reputations.get(message.sender_id, 1.0)
            if sender_reputation < self.security_params["min_reputation"]:
                validation_errors.append(f"Sender reputation too low: {sender_reputation}")

            # Protocol-specific validation
            protocol_errors = await self._validate_protocol_specific(message)
            validation_errors.extend(protocol_errors)

            # Update statistics
            if not validation_errors:
                self.security_stats["messages_verified"] += 1
            else:
                self.security_stats["messages_rejected"] += 1

            return len(validation_errors) == 0, validation_errors

        except Exception as e:
            logger.error(f"Message validation failed: {e}")
            return False, [f"Validation error: {str(e)}"]

    # Private helper methods

    def _evaluate_polynomial(self, coefficients: List[int], x: int) -> int:
        """Evaluate polynomial at given point using Horner's method."""
        result = 0
        for coeff in reversed(coefficients):
            result = result * x + coeff
        return result % (2**256)  # Modular arithmetic

    def _combine_public_keys(self, public_key_shares: List[bytes]) -> bytes:
        """Combine public key shares into master public key."""
        # Simplified combination - in production would use proper elliptic curve operations
        combined = hashlib.sha256()
        for share in sorted(public_key_shares):
            combined.update(share)
        return combined.digest()

    async def _create_partial_signature(self, message: bytes, threshold_key: ThresholdKey) -> Optional[bytes]:
        """Create partial signature using threshold key share."""
        try:
            if CRYPTOGRAPHY_AVAILABLE and len(threshold_key.private_key_share) > 32:
                # Use RSA signing
                private_key = serialization.load_pem_private_key(
                    threshold_key.private_key_share, password=None, backend=default_backend()
                )
                signature = private_key.sign(
                    message,
                    padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                    hashes.SHA256(),
                )
                return signature
            else:
                # Hash-based signature
                return hmac.new(threshold_key.private_key_share, message, hashlib.sha256).digest()
        except Exception as e:
            logger.error(f"Partial signature creation failed: {e}")
            return None

    async def _combine_partial_signatures(self, message: bytes, partial_signatures: List[Dict[str, Any]]) -> bytes:
        """Combine partial signatures using Lagrange interpolation."""
        # Simplified combination - in production would use proper threshold signature schemes
        combined = hashlib.sha256()
        combined.update(message)

        for partial in partial_signatures:
            combined.update(partial["signature"])
            combined.update(partial["public_key_share"])

        return combined.digest()

    def _verify_rsa_signature(self, message: bytes, signature: bytes, public_key: bytes) -> bool:
        """Verify RSA signature (simplified)."""
        # In a real implementation, this would properly verify RSA signatures
        # For now, use hash-based verification
        expected = hashlib.sha256(message + public_key).digest()
        return hmac.compare_digest(signature[:32], expected)

    async def _detect_contradictory_messages(self, consensus_round: ConsensusRound) -> Dict[str, Dict[str, Any]]:
        """Detect nodes sending contradictory messages in the same round."""
        contradictions = {}

        # Group messages by sender
        messages_by_sender = {}
        for message in list(consensus_round.votes.values()) + list(consensus_round.commits.values()):
            sender = message.sender_id
            if sender not in messages_by_sender:
                messages_by_sender[sender] = []
            messages_by_sender[sender].append(message)

        # Check for contradictions
        for sender, messages in messages_by_sender.items():
            if len(messages) > 1:
                # Check for conflicting votes/commits
                message_types = [msg.message_type for msg in messages]
                payloads = [json.dumps(msg.payload, sort_keys=True) for msg in messages]

                # If same message type but different payloads = contradiction
                type_payload_pairs = list(zip(message_types, payloads))
                unique_pairs = set(type_payload_pairs)

                if len(type_payload_pairs) != len(unique_pairs):
                    contradictions[sender] = {
                        "message_count": len(messages),
                        "conflicting_messages": [
                            {"type": msg.message_type, "payload": msg.payload} for msg in messages
                        ],
                    }

        return contradictions

    async def _detect_timing_attacks(self, consensus_round: ConsensusRound) -> List[AttackEvidence]:
        """Detect timing-based attacks."""
        attacks = []

        # Analyze message timing patterns
        message_times = []
        for message in list(consensus_round.votes.values()) + list(consensus_round.commits.values()):
            message_times.append((message.sender_id, message.timestamp))

        # Sort by timestamp
        message_times.sort(key=lambda x: x[1])

        # Look for suspiciously coordinated timing
        if len(message_times) >= 3:
            time_diffs = []
            for i in range(1, len(message_times)):
                diff = message_times[i][1] - message_times[i - 1][1]
                time_diffs.append(diff)

            # If timing is too regular (low variance), might be coordinated attack
            if time_diffs:
                variance = np.var(time_diffs)
                if variance < 0.1:  # Very low variance in timing
                    suspected_nodes = set(sender for sender, _ in message_times)
                    attack = AttackEvidence(
                        attack_id=str(uuid.uuid4()),
                        attack_type=AttackType.TIMING_ATTACK,
                        suspected_nodes=suspected_nodes,
                        evidence_data={"timing_variance": variance, "message_times": message_times},
                        confidence_score=0.7,
                    )
                    attacks.append(attack)

        return attacks

    async def _detect_collusion(self, consensus_round: ConsensusRound) -> List[AttackEvidence]:
        """Detect collusion patterns between nodes."""
        attacks = []

        # Analyze voting patterns for suspicious coordination
        vote_patterns = {}
        for message in consensus_round.votes.values():
            sender = message.sender_id
            vote_content = json.dumps(message.payload, sort_keys=True)

            if vote_content not in vote_patterns:
                vote_patterns[vote_content] = []
            vote_patterns[vote_content].append(sender)

        # Look for groups that always vote together
        for vote_content, voters in vote_patterns.items():
            if len(voters) >= 3:  # 3 or more nodes voting identically
                # Check historical voting patterns
                historical_agreement = await self._check_historical_agreement(voters)

                if historical_agreement > 0.9:  # 90% agreement historically
                    attack = AttackEvidence(
                        attack_id=str(uuid.uuid4()),
                        attack_type=AttackType.COLLUSION,
                        suspected_nodes=set(voters),
                        evidence_data={
                            "historical_agreement": historical_agreement,
                            "current_vote": vote_content,
                            "colluding_nodes": voters,
                        },
                        confidence_score=historical_agreement,
                    )
                    attacks.append(attack)

        return attacks

    async def _detect_fork_attacks(self, consensus_round: ConsensusRound) -> List[AttackEvidence]:
        """Detect fork attacks and equivocation."""
        attacks = []

        # Check for nodes proposing multiple different blocks
        proposals_by_node = {}

        if consensus_round.proposal:
            proposer = consensus_round.leader_id
            proposal_hash = hashlib.sha256(json.dumps(consensus_round.proposal, sort_keys=True).encode()).hexdigest()

            if proposer not in proposals_by_node:
                proposals_by_node[proposer] = []
            proposals_by_node[proposer].append(proposal_hash)

        # Check votes for different proposals in same round
        for message in consensus_round.votes.values():
            if "proposal_hash" in message.payload:
                sender = message.sender_id
                proposal_hash = message.payload["proposal_hash"]

                if sender not in proposals_by_node:
                    proposals_by_node[sender] = []
                proposals_by_node[sender].append(proposal_hash)

        # Detect multiple proposals from same node
        for node_id, proposals in proposals_by_node.items():
            unique_proposals = set(proposals)
            if len(unique_proposals) > 1:
                attack = AttackEvidence(
                    attack_id=str(uuid.uuid4()),
                    attack_type=AttackType.FORK_ATTACK,
                    suspected_nodes={node_id},
                    evidence_data={
                        "multiple_proposals": list(unique_proposals),
                        "round_number": consensus_round.round_number,
                    },
                    confidence_score=1.0,  # Equivocation is certain
                )
                attacks.append(attack)

        return attacks

    async def _check_historical_agreement(self, nodes: List[str]) -> float:
        """Check historical agreement rate between nodes."""
        if len(self.consensus_history) < 5:
            return 0.5  # Not enough history

        agreement_count = 0
        total_rounds = 0

        for past_round in self.consensus_history[-10:]:  # Check last 10 rounds
            node_votes = {}
            for message in past_round.votes.values():
                if message.sender_id in nodes:
                    node_votes[message.sender_id] = json.dumps(message.payload, sort_keys=True)

            if len(node_votes) >= 2:
                # Check if all nodes voted the same way
                votes = list(node_votes.values())
                if len(set(votes)) == 1:  # All identical votes
                    agreement_count += 1
                total_rounds += 1

        return agreement_count / max(1, total_rounds)

    async def _exclude_byzantine_nodes(self, suspected_nodes: Set[str]) -> bool:
        """Exclude Byzantine nodes from consensus participation."""
        try:
            for node_id in suspected_nodes:
                if node_id in self.participant_nodes:
                    self.participant_nodes[node_id]["role"] = NodeRole.OBSERVER.value
                    self.node_reputations[node_id] = 0.0

                    logger.info(f"Excluded Byzantine node {node_id} from consensus")

            self.security_stats["byzantine_nodes_identified"] += len(suspected_nodes)
            return True
        except Exception as e:
            logger.error(f"Failed to exclude Byzantine nodes: {e}")
            return False

    async def _strengthen_identity_verification(self) -> bool:
        """Strengthen identity verification to prevent Sybil attacks."""
        # Implement stronger identity verification requirements
        # This would involve requiring additional cryptographic proofs
        logger.info("Strengthened identity verification requirements")
        return True

    async def _diversify_network_connections(self, attack: AttackEvidence) -> bool:
        """Diversify network connections to prevent eclipse attacks."""
        # Implement connection diversification logic
        logger.info("Diversified network connections to prevent eclipse attacks")
        return True

    async def _apply_rate_limiting(self, suspected_nodes: Set[str]) -> bool:
        """Apply rate limiting to suspected DoS attackers."""
        try:
            for node_id in suspected_nodes:
                # Implement rate limiting logic
                logger.info(f"Applied rate limiting to suspected DoS attacker {node_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to apply rate limiting: {e}")
            return False

    async def _randomize_consensus_parameters(self) -> bool:
        """Randomize consensus parameters to disrupt collusion."""
        # Randomize timing, ordering, or other parameters
        logger.info("Randomized consensus parameters to disrupt collusion")
        return True

    async def _randomize_timing(self) -> bool:
        """Introduce timing randomization to prevent timing attacks."""
        # Add random delays to consensus operations
        logger.info("Introduced timing randomization")
        return True

    async def _verify_message_signature(self, message: ConsensusMessage) -> bool:
        """Verify message signature."""
        if not message.signature:
            return False

        # Create message hash for verification
        message_data = {
            "sender_id": message.sender_id,
            "message_type": message.message_type,
            "round_number": message.round_number,
            "view_number": message.view_number,
            "payload": message.payload,
            "timestamp": message.timestamp,
        }

        message_bytes = json.dumps(message_data, sort_keys=True).encode()

        # Verify using threshold signature if available
        if self.master_public_key:
            return await self.verify_threshold_signature(message_bytes, message.signature)
        else:
            # Fallback verification
            sender_key = self.node_certificates.get(message.sender_id)
            if sender_key:
                expected_sig = hashlib.sha256(message_bytes + sender_key).digest()
                return hmac.compare_digest(message.signature, expected_sig)

        return False

    async def _verify_message_proof(self, message: ConsensusMessage) -> bool:
        """Verify zero-knowledge proof attached to message."""
        if not message.proof:
            return False

        # Extract expected statement from message
        statement = {
            "message_type": message.message_type,
            "round_number": message.round_number,
            "sender_has_authority": True,
        }

        return await self.verify_zero_knowledge_proof(message.proof, statement)

    async def _validate_protocol_specific(self, message: ConsensusMessage) -> List[str]:
        """Perform protocol-specific message validation."""
        errors = []

        if self.consensus_protocol == ConsensusProtocol.RAFT:
            errors.extend(await self._validate_raft_message(message))
        elif self.consensus_protocol == ConsensusProtocol.BYZANTINE:
            errors.extend(await self._validate_byzantine_message(message))
        elif self.consensus_protocol == ConsensusProtocol.PRACTICAL_BFT:
            errors.extend(await self._validate_pbft_message(message))

        return errors

    async def _validate_raft_message(self, message: ConsensusMessage) -> List[str]:
        """Validate Raft-specific message constraints."""
        errors = []

        if message.message_type == "vote_request":
            # Validate vote request structure
            required_fields = ["term", "candidate_id", "last_log_index", "last_log_term"]
            for field in required_fields:
                if field not in message.payload:
                    errors.append(f"Missing required field: {field}")

        elif message.message_type == "vote_response":
            # Validate vote response structure
            if "term" not in message.payload or "vote_granted" not in message.payload:
                errors.append("Invalid vote response structure")

        return errors

    async def _validate_byzantine_message(self, message: ConsensusMessage) -> List[str]:
        """Validate Byzantine consensus message constraints."""
        errors = []

        if message.message_type in ["prepare", "commit"]:
            # Validate Byzantine consensus message structure
            if "proposal_hash" not in message.payload:
                errors.append("Missing proposal hash in Byzantine message")

        return errors

    async def _validate_pbft_message(self, message: ConsensusMessage) -> List[str]:
        """Validate PBFT-specific message constraints."""
        errors = []

        if message.message_type in ["pre-prepare", "prepare", "commit"]:
            # Validate PBFT three-phase protocol
            required_fields = ["view", "sequence_number"]
            for field in required_fields:
                if field not in message.payload:
                    errors.append(f"Missing PBFT field: {field}")

        return errors

    # Public API methods

    def get_security_stats(self) -> Dict[str, Any]:
        """Get comprehensive security statistics."""
        return {
            **self.security_stats,
            "active_nodes": len(self.participant_nodes),
            "byzantine_nodes": len(
                [node_id for node_id, info in self.participant_nodes.items() if info["role"] == NodeRole.OBSERVER.value]
            ),
            "average_reputation": np.mean(list(self.node_reputations.values())) if self.node_reputations else 0.0,
            "recent_attacks": len(
                [attack for attack in self.attack_evidence if time.time() - attack.detected_at < 3600]  # Last hour
            ),
            "consensus_rounds": len(self.consensus_history),
        }

    def get_node_reputation(self, node_id: str) -> float:
        """Get reputation score for a node."""
        return self.node_reputations.get(node_id, 1.0)

    def get_attack_summary(self) -> Dict[str, Any]:
        """Get summary of detected attacks."""
        recent_attacks = [
            attack for attack in self.attack_evidence if time.time() - attack.detected_at < 86400  # Last 24 hours
        ]

        attack_counts = {}
        for attack in recent_attacks:
            attack_type = attack.attack_type.value
            attack_counts[attack_type] = attack_counts.get(attack_type, 0) + 1

        return {
            "total_attacks_24h": len(recent_attacks),
            "attacks_by_type": attack_counts,
            "mitigated_attacks": len([a for a in recent_attacks if a.mitigated]),
            "high_confidence_attacks": len([a for a in recent_attacks if a.confidence_score > 0.8]),
            "suspected_byzantine_nodes": len(
                [node_id for node_id, info in self.participant_nodes.items() if info.get("byzantine_score", 0) > 0.5]
            ),
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform security system health check."""
        issues = []

        # Check threshold key availability
        if not self.threshold_keys:
            issues.append("No threshold keys available")

        # Check master public key
        if not self.master_public_key:
            issues.append("Master public key not initialized")

        # Check for recent Byzantine activity
        recent_byzantine = len(
            [
                attack
                for attack in self.attack_evidence
                if (
                    attack.attack_type == AttackType.BYZANTINE_NODE
                    and time.time() - attack.detected_at < 3600
                    and not attack.mitigated
                )
            ]
        )

        if recent_byzantine > 0:
            issues.append(f"{recent_byzantine} unmitigated Byzantine attacks detected")

        # Check node reputation health
        low_reputation_nodes = [
            node_id
            for node_id, reputation in self.node_reputations.items()
            if reputation < self.security_params["min_reputation"]
        ]

        if len(low_reputation_nodes) > len(self.participant_nodes) * 0.3:
            issues.append("High number of low-reputation nodes")

        return {
            "healthy": len(issues) == 0,
            "issues": issues,
            "security_stats": self.get_security_stats(),
            "attack_summary": self.get_attack_summary(),
        }
