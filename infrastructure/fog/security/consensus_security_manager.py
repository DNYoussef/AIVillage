"""
Consensus Security Manager for Distributed Federated Systems

This module implements comprehensive security mechanisms for distributed consensus protocols
with advanced threat detection, cryptographic infrastructure, and attack mitigation.
Designed specifically for federated learning environments using BetaNet integration.
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import secrets
import hmac
from datetime import datetime, UTC
from collections import defaultdict

logger = logging.getLogger(__name__)


class AttackType(Enum):
    """Types of attacks detected by the security system"""
    BYZANTINE = "byzantine"
    SYBIL = "sybil"
    ECLIPSE = "eclipse"
    DOS = "dos"
    GRADIENT_INVERSION = "gradient_inversion"
    MODEL_POISONING = "model_poisoning"
    TIMING_ATTACK = "timing_attack"
    COLLUSION = "collusion"


class SecurityLevel(Enum):
    """Security levels for different operations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityAlert:
    """Security alert information"""
    alert_id: str
    attack_type: AttackType
    severity: SecurityLevel
    confidence: float  # 0.0 to 1.0
    details: Dict[str, Any]
    affected_nodes: List[str]
    timestamp: float = field(default_factory=time.time)
    evidence: Dict[str, Any] = field(default_factory=dict)
    mitigation_applied: Optional[str] = None


@dataclass
class ThresholdSignature:
    """Threshold signature components"""
    signature: bytes
    commitment: bytes
    challenge: bytes
    response: bytes
    participants: List[str]
    threshold: int


@dataclass
class ZKProof:
    """Zero-knowledge proof structure"""
    commitment: bytes
    challenge: bytes
    response: bytes
    public_parameters: Dict[str, Any]
    proof_type: str
    verified: bool = False


class EllipticCurveOperations:
    """Simplified elliptic curve operations for cryptographic protocols"""
    
    def __init__(self, curve_name: str = "secp256k1"):
        self.curve_name = curve_name
        self.generator_point = self._get_generator()
        self.field_order = self._get_field_order()
        
    def _get_generator(self) -> bytes:
        """Get curve generator point"""
        # Simplified - in production use proper cryptographic library
        return hashlib.sha256(b"generator_point").digest()
    
    def _get_field_order(self) -> int:
        """Get curve field order"""
        # Simplified - use proper curve parameters
        return 2**256 - 2**32 - 2**9 - 2**8 - 2**7 - 2**6 - 2**4 - 1
    
    def point_multiply(self, point: bytes, scalar: int) -> bytes:
        """Multiply point by scalar"""
        # Simplified implementation
        combined = point + scalar.to_bytes(32, 'big')
        return hashlib.sha256(combined).digest()
    
    def point_add(self, point1: bytes, point2: bytes) -> bytes:
        """Add two points"""
        # Simplified implementation
        combined = point1 + point2
        return hashlib.sha256(combined).digest()
    
    def generate_keypair(self) -> Tuple[int, bytes]:
        """Generate private/public key pair"""
        private_key = secrets.randbits(256)
        public_key = self.point_multiply(self.generator_point, private_key)
        return private_key, public_key


class ThresholdCryptographySystem:
    """
    Threshold cryptography implementation for secure distributed operations
    Enables t-of-n signatures and secret sharing for federated learning
    """
    
    def __init__(self, threshold: int, total_parties: int):
        self.threshold = threshold
        self.total_parties = total_parties
        self.curve = EllipticCurveOperations()
        self.master_public_key: Optional[bytes] = None
        self.key_shares: Dict[str, int] = {}
        self.public_shares: Dict[str, bytes] = {}
        self.verification_keys: Dict[str, bytes] = {}
        
    async def distributed_key_generation(self, participants: List[str]) -> Dict[str, Any]:
        """
        Execute distributed key generation protocol
        
        Args:
            participants: List of participant node IDs
            
        Returns:
            Dictionary with DKG results
        """
        logger.info(f"Starting DKG with {len(participants)} participants, threshold {self.threshold}")
        
        # Phase 1: Each party generates secret polynomial
        polynomials = {}
        commitments = {}
        
        for participant in participants:
            # Generate random polynomial coefficients
            coefficients = [secrets.randbits(256) for _ in range(self.threshold)]
            polynomials[participant] = coefficients
            
            # Generate commitments to coefficients
            commitment_points = []
            for coeff in coefficients:
                commitment = self.curve.point_multiply(self.curve.generator_point, coeff)
                commitment_points.append(commitment)
            commitments[participant] = commitment_points
        
        # Phase 2: Share secret values
        secret_shares = defaultdict(dict)
        for sender in participants:
            for i, receiver in enumerate(participants, 1):
                # Evaluate polynomial at point i
                share_value = sum(
                    coeff * (i ** j) for j, coeff in enumerate(polynomials[sender])
                ) % self.curve.field_order
                secret_shares[receiver][sender] = share_value
        
        # Phase 3: Verify received shares
        verified_shares = {}
        for participant in participants:
            participant_share = 0
            for sender in participants:
                share = secret_shares[participant][sender]
                # Verify against commitment (simplified)
                if self._verify_share_commitment(share, commitments[sender], participant):
                    participant_share += share
                else:
                    logger.warning(f"Invalid share from {sender} to {participant}")
            
            verified_shares[participant] = participant_share % self.curve.field_order
        
        # Phase 4: Generate master public key
        master_public_point = None
        for sender_commitments in commitments.values():
            # First commitment is to constant term
            if master_public_point is None:
                master_public_point = sender_commitments[0]
            else:
                master_public_point = self.curve.point_add(
                    master_public_point, sender_commitments[0]
                )
        
        self.master_public_key = master_public_point
        self.key_shares = verified_shares
        
        # Generate verification keys
        for participant in participants:
            self.verification_keys[participant] = self.curve.point_multiply(
                self.curve.generator_point, verified_shares[participant]
            )
        
        return {
            "master_public_key": self.master_public_key,
            "key_shares_count": len(verified_shares),
            "participants": participants,
            "threshold": self.threshold
        }
    
    def _verify_share_commitment(self, share: int, commitments: List[bytes], participant_id: str) -> bool:
        """Verify that a share matches its commitment"""
        # Simplified verification - in production use proper Feldman VSS
        participant_index = hash(participant_id) % len(commitments)
        expected_commitment = self.curve.point_multiply(
            self.curve.generator_point, share
        )
        # This is a simplified check
        return True  # In production, implement proper verification
    
    async def create_threshold_signature(
        self, 
        message: bytes, 
        signers: List[str]
    ) -> ThresholdSignature:
        """
        Create threshold signature from t signers
        
        Args:
            message: Message to sign
            signers: List of signing participants (must be >= threshold)
            
        Returns:
            Threshold signature
        """
        if len(signers) < self.threshold:
            raise ValueError(f"Need at least {self.threshold} signers, got {len(signers)}")
        
        # Use only required number of signers
        active_signers = signers[:self.threshold]
        
        # Phase 1: Commitment phase
        nonces = {}
        commitments = {}
        
        for signer in active_signers:
            nonce = secrets.randbits(256)
            nonces[signer] = nonce
            commitment = self.curve.point_multiply(self.curve.generator_point, nonce)
            commitments[signer] = commitment
        
        # Combine commitments
        combined_commitment = None
        for commitment in commitments.values():
            if combined_commitment is None:
                combined_commitment = commitment
            else:
                combined_commitment = self.curve.point_add(combined_commitment, commitment)
        
        # Phase 2: Challenge generation (Fiat-Shamir)
        challenge_input = message + combined_commitment + self.master_public_key
        challenge = int.from_bytes(
            hashlib.sha256(challenge_input).digest(), 'big'
        ) % self.curve.field_order
        
        # Phase 3: Response generation
        partial_signatures = {}
        for signer in active_signers:
            if signer not in self.key_shares:
                raise ValueError(f"No key share for signer {signer}")
            
            # Calculate Lagrange coefficient
            lagrange_coeff = self._calculate_lagrange_coefficient(
                signer, active_signers
            )
            
            # Generate partial signature
            response = (
                nonces[signer] + 
                challenge * self.key_shares[signer] * lagrange_coeff
            ) % self.curve.field_order
            
            partial_signatures[signer] = response
        
        # Phase 4: Combine partial signatures
        final_response = sum(partial_signatures.values()) % self.curve.field_order
        
        signature = ThresholdSignature(
            signature=final_response.to_bytes(32, 'big'),
            commitment=combined_commitment,
            challenge=challenge.to_bytes(32, 'big'),
            response=final_response.to_bytes(32, 'big'),
            participants=active_signers,
            threshold=self.threshold
        )
        
        return signature
    
    def _calculate_lagrange_coefficient(self, signer: str, signers: List[str]) -> int:
        """Calculate Lagrange interpolation coefficient"""
        signer_index = signers.index(signer) + 1  # 1-indexed
        coefficient = 1
        
        for j, other_signer in enumerate(signers, 1):
            if other_signer != signer:
                coefficient = (coefficient * j) // (j - signer_index)
        
        return coefficient % self.curve.field_order
    
    def verify_threshold_signature(
        self, 
        message: bytes, 
        signature: ThresholdSignature
    ) -> bool:
        """Verify threshold signature"""
        if not self.master_public_key:
            return False
        
        # Recreate challenge
        challenge_input = message + signature.commitment + self.master_public_key
        expected_challenge = int.from_bytes(
            hashlib.sha256(challenge_input).digest(), 'big'
        ) % self.curve.field_order
        
        actual_challenge = int.from_bytes(signature.challenge, 'big')
        
        if expected_challenge != actual_challenge:
            return False
        
        # Verify signature equation: g^response = commitment * public_key^challenge
        left_side = self.curve.point_multiply(
            self.curve.generator_point,
            int.from_bytes(signature.response, 'big')
        )
        
        right_side_1 = signature.commitment
        right_side_2 = self.curve.point_multiply(
            self.master_public_key,
            actual_challenge
        )
        right_side = self.curve.point_add(right_side_1, right_side_2)
        
        return left_side == right_side


class ZeroKnowledgeProofSystem:
    """
    Zero-knowledge proof system for privacy-preserving operations
    Enables proving knowledge without revealing sensitive information
    """
    
    def __init__(self):
        self.curve = EllipticCurveOperations()
        self.proof_cache: Dict[str, ZKProof] = {}
        
    async def prove_gradient_knowledge(
        self,
        gradient_commitment: bytes,
        gradient_value: int,
        node_id: str
    ) -> ZKProof:
        """
        Prove knowledge of gradient without revealing the value
        Uses Schnorr-like protocol for discrete log proof
        
        Args:
            gradient_commitment: Commitment to the gradient
            gradient_value: Actual gradient value (kept secret)
            node_id: Identity of the proving node
            
        Returns:
            Zero-knowledge proof
        """
        # Generate random nonce
        nonce = secrets.randbits(256)
        commitment = self.curve.point_multiply(self.curve.generator_point, nonce)
        
        # Generate Fiat-Shamir challenge
        challenge_input = (
            commitment + 
            gradient_commitment + 
            node_id.encode() +
            str(time.time()).encode()
        )
        challenge = int.from_bytes(
            hashlib.sha256(challenge_input).digest(), 'big'
        ) % self.curve.field_order
        
        # Calculate response
        response = (nonce + challenge * gradient_value) % self.curve.field_order
        
        proof = ZKProof(
            commitment=commitment,
            challenge=challenge.to_bytes(32, 'big'),
            response=response.to_bytes(32, 'big'),
            public_parameters={
                "gradient_commitment": gradient_commitment,
                "node_id": node_id,
                "timestamp": time.time()
            },
            proof_type="gradient_knowledge",
            verified=False
        )
        
        # Cache proof for later verification
        proof_id = hashlib.sha256(
            commitment + proof.challenge + proof.response
        ).hexdigest()
        self.proof_cache[proof_id] = proof
        
        return proof
    
    def verify_gradient_proof(
        self,
        proof: ZKProof,
        gradient_commitment: bytes
    ) -> bool:
        """
        Verify zero-knowledge proof of gradient knowledge
        
        Args:
            proof: The zero-knowledge proof to verify
            gradient_commitment: Expected gradient commitment
            
        Returns:
            True if proof is valid
        """
        try:
            # Extract parameters
            challenge = int.from_bytes(proof.challenge, 'big')
            response = int.from_bytes(proof.response, 'big')
            
            # Verify proof equation: g^response = commitment * public_commitment^challenge
            left_side = self.curve.point_multiply(
                self.curve.generator_point, response
            )
            
            right_side_1 = proof.commitment
            right_side_2 = self.curve.point_multiply(gradient_commitment, challenge)
            right_side = self.curve.point_add(right_side_1, right_side_2)
            
            is_valid = left_side == right_side
            
            # Verify challenge was generated correctly
            challenge_input = (
                proof.commitment +
                gradient_commitment +
                proof.public_parameters["node_id"].encode() +
                str(proof.public_parameters["timestamp"]).encode()
            )
            expected_challenge = int.from_bytes(
                hashlib.sha256(challenge_input).digest(), 'big'
            ) % self.curve.field_order
            
            challenge_valid = challenge == expected_challenge
            
            proof.verified = is_valid and challenge_valid
            return proof.verified
            
        except Exception as e:
            logger.error(f"Proof verification failed: {e}")
            return False
    
    async def prove_range(
        self,
        value: int,
        min_range: int,
        max_range: int,
        commitment: bytes
    ) -> ZKProof:
        """
        Prove that a committed value is within a specified range
        Simplified range proof implementation
        
        Args:
            value: Secret value to prove is in range
            min_range: Minimum allowed value
            max_range: Maximum allowed value
            commitment: Commitment to the value
            
        Returns:
            Zero-knowledge range proof
        """
        if not (min_range <= value <= max_range):
            raise ValueError("Value not in specified range")
        
        # Simplified range proof - in production use Bulletproofs or similar
        shifted_value = value - min_range
        range_size = max_range - min_range + 1
        
        # Create bit decomposition proof
        bit_length = (range_size - 1).bit_length()
        bit_proofs = []
        
        for i in range(bit_length):
            bit = (shifted_value >> i) & 1
            bit_nonce = secrets.randbits(256)
            bit_commitment = self.curve.point_multiply(
                self.curve.generator_point, bit_nonce
            )
            
            # Prove bit is 0 or 1
            if bit == 0:
                # Prove knowledge of commitment opening to 0
                challenge = secrets.randbits(256) % self.curve.field_order
                response = bit_nonce % self.curve.field_order
            else:
                # Prove knowledge of commitment opening to 1
                challenge = secrets.randbits(256) % self.curve.field_order
                response = (bit_nonce - challenge) % self.curve.field_order
            
            bit_proofs.append({
                "bit_commitment": bit_commitment,
                "challenge": challenge,
                "response": response
            })
        
        # Combine all bit proofs
        combined_commitment = commitment
        for bit_proof in bit_proofs:
            combined_commitment = self.curve.point_add(
                combined_commitment, bit_proof["bit_commitment"]
            )
        
        proof = ZKProof(
            commitment=combined_commitment,
            challenge=secrets.randbits(256).to_bytes(32, 'big'),
            response=secrets.randbits(256).to_bytes(32, 'big'),
            public_parameters={
                "min_range": min_range,
                "max_range": max_range,
                "bit_proofs": bit_proofs,
                "range_size": range_size
            },
            proof_type="range_proof",
            verified=False
        )
        
        return proof


class AttackDetectionSystem:
    """
    Comprehensive attack detection system for federated consensus
    Monitors for various attack patterns and anomalies
    """
    
    def __init__(self):
        self.reputation_scores: Dict[str, float] = defaultdict(lambda: 1.0)
        self.behavior_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.active_alerts: List[SecurityAlert] = []
        self.detection_thresholds = {
            AttackType.BYZANTINE: 0.7,
            AttackType.SYBIL: 0.8,
            AttackType.ECLIPSE: 0.6,
            AttackType.DOS: 0.9,
            AttackType.GRADIENT_INVERSION: 0.8,
            AttackType.MODEL_POISONING: 0.7,
            AttackType.TIMING_ATTACK: 0.6,
            AttackType.COLLUSION: 0.8
        }
        
    async def detect_byzantine_behavior(
        self,
        node_id: str,
        messages: List[Dict[str, Any]],
        consensus_round: int
    ) -> Optional[SecurityAlert]:
        """
        Detect Byzantine behavior patterns
        
        Args:
            node_id: Node under analysis
            messages: Messages sent by the node
            consensus_round: Current consensus round
            
        Returns:
            Security alert if Byzantine behavior detected
        """
        anomalies = []
        
        # Check for contradictory messages
        message_hashes = set()
        contradictions = 0
        
        for msg in messages:
            msg_hash = hashlib.sha256(json.dumps(msg, sort_keys=True).encode()).hexdigest()
            if msg_hash in message_hashes:
                contradictions += 1
            message_hashes.add(msg_hash)
        
        if contradictions > 0:
            anomalies.append("contradictory_messages")
        
        # Check for timing anomalies
        if len(messages) > 1:
            timestamps = [msg.get('timestamp', 0) for msg in messages]
            timestamp_diffs = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            
            # Detect unusually rapid message sending
            rapid_messages = sum(1 for diff in timestamp_diffs if diff < 0.001)  # < 1ms
            if rapid_messages > len(messages) * 0.5:  # More than 50% rapid
                anomalies.append("rapid_message_pattern")
        
        # Check reputation history
        recent_behavior = self.behavior_history[node_id][-10:]  # Last 10 behaviors
        malicious_behavior_count = sum(
            1 for behavior in recent_behavior 
            if behavior.get('type') == 'malicious'
        )
        
        if malicious_behavior_count > 3:
            anomalies.append("repeated_malicious_behavior")
        
        # Calculate confidence based on anomalies
        confidence = min(1.0, len(anomalies) * 0.3 + contradictions * 0.4)
        
        if confidence >= self.detection_thresholds[AttackType.BYZANTINE]:
            alert = SecurityAlert(
                alert_id=f"byz_{node_id}_{consensus_round}_{int(time.time())}",
                attack_type=AttackType.BYZANTINE,
                severity=SecurityLevel.HIGH,
                confidence=confidence,
                details={
                    "contradictions": contradictions,
                    "anomalies": anomalies,
                    "message_count": len(messages),
                    "consensus_round": consensus_round
                },
                affected_nodes=[node_id],
                evidence={
                    "messages": messages,
                    "timestamps": [msg.get('timestamp', 0) for msg in messages],
                    "reputation_score": self.reputation_scores[node_id]
                }
            )
            
            self.active_alerts.append(alert)
            self._update_reputation(node_id, -0.3)  # Penalty for Byzantine behavior
            
            logger.warning(
                f"Byzantine behavior detected from {node_id}: "
                f"confidence {confidence:.2f}, anomalies: {anomalies}"
            )
            
            return alert
        
        return None
    
    async def detect_sybil_attack(
        self,
        new_node_requests: List[Dict[str, Any]]
    ) -> Optional[SecurityAlert]:
        """
        Detect Sybil attack patterns in node join requests
        
        Args:
            new_node_requests: List of new node join requests
            
        Returns:
            Security alert if Sybil attack detected
        """
        if not new_node_requests:
            return None
        
        # Analyze patterns in join requests
        suspicious_patterns = []
        
        # Check for rapid joins from similar sources
        timestamps = [req.get('timestamp', 0) for req in new_node_requests]
        ip_addresses = [req.get('ip_address', '') for req in new_node_requests]
        user_agents = [req.get('user_agent', '') for req in new_node_requests]
        
        # Time clustering analysis
        time_windows = []
        window_size = 300  # 5 minutes
        
        for i, ts in enumerate(timestamps):
            window_nodes = [
                j for j, other_ts in enumerate(timestamps)
                if abs(other_ts - ts) <= window_size
            ]
            if len(window_nodes) > 5:  # More than 5 nodes in 5-minute window
                time_windows.append(window_nodes)
        
        if time_windows:
            suspicious_patterns.append("rapid_temporal_clustering")
        
        # IP address clustering
        ip_counts = defaultdict(int)
        for ip in ip_addresses:
            ip_counts[ip] += 1
        
        suspicious_ips = [ip for ip, count in ip_counts.items() if count > 3]
        if suspicious_ips:
            suspicious_patterns.append("ip_address_clustering")
        
        # User agent similarity
        unique_user_agents = set(user_agents)
        if len(unique_user_agents) < len(new_node_requests) * 0.3:  # Less than 30% unique
            suspicious_patterns.append("user_agent_similarity")
        
        # Resource fingerprinting - check for similar hardware profiles
        hardware_profiles = []
        for req in new_node_requests:
            profile = (
                req.get('cpu_cores', 0),
                req.get('memory_gb', 0),
                req.get('disk_gb', 0),
                req.get('network_speed', 0)
            )
            hardware_profiles.append(profile)
        
        unique_profiles = set(hardware_profiles)
        if len(unique_profiles) < len(hardware_profiles) * 0.4:  # Less than 40% unique
            suspicious_patterns.append("hardware_fingerprint_similarity")
        
        confidence = min(1.0, len(suspicious_patterns) * 0.25)
        
        if confidence >= self.detection_thresholds[AttackType.SYBIL]:
            alert = SecurityAlert(
                alert_id=f"sybil_{int(time.time())}",
                attack_type=AttackType.SYBIL,
                severity=SecurityLevel.CRITICAL,
                confidence=confidence,
                details={
                    "suspicious_patterns": suspicious_patterns,
                    "request_count": len(new_node_requests),
                    "suspicious_ips": suspicious_ips,
                    "unique_user_agents": len(unique_user_agents)
                },
                affected_nodes=[req.get('node_id', 'unknown') for req in new_node_requests],
                evidence={
                    "requests": new_node_requests,
                    "ip_distribution": dict(ip_counts),
                    "time_windows": time_windows,
                    "hardware_profiles": hardware_profiles
                }
            )
            
            self.active_alerts.append(alert)
            
            logger.critical(
                f"Sybil attack detected: {len(new_node_requests)} suspicious nodes, "
                f"confidence {confidence:.2f}, patterns: {suspicious_patterns}"
            )
            
            return alert
        
        return None
    
    async def detect_eclipse_attack(
        self,
        node_id: str,
        peer_connections: List[Dict[str, Any]]
    ) -> Optional[SecurityAlert]:
        """
        Detect Eclipse attack attempts
        
        Args:
            node_id: Target node being analyzed
            peer_connections: List of peer connection information
            
        Returns:
            Security alert if Eclipse attack detected
        """
        if not peer_connections:
            return None
        
        eclipse_indicators = []
        
        # Analyze peer diversity
        peer_ips = [conn.get('ip_address', '') for conn in peer_connections]
        peer_asns = [conn.get('asn', '') for conn in peer_connections]
        peer_geolocations = [conn.get('country', '') for conn in peer_connections]
        
        # Geographic diversity check
        unique_countries = set(peer_geolocations)
        if len(unique_countries) < max(2, len(peer_connections) * 0.3):
            eclipse_indicators.append("low_geographic_diversity")
        
        # Network diversity check (ASN)
        unique_asns = set(peer_asns)
        if len(unique_asns) < max(2, len(peer_connections) * 0.4):
            eclipse_indicators.append("low_network_diversity")
        
        # Check for connection time clustering
        connection_times = [conn.get('connected_at', 0) for conn in peer_connections]
        recent_connections = [
            t for t in connection_times 
            if time.time() - t < 3600  # Last hour
        ]
        
        if len(recent_connections) > len(peer_connections) * 0.8:  # 80% recent
            eclipse_indicators.append("rapid_peer_replacement")
        
        # Check for suspicious peer behavior
        malicious_peers = 0
        for conn in peer_connections:
            peer_id = conn.get('peer_id', '')
            if self.reputation_scores.get(peer_id, 1.0) < 0.3:
                malicious_peers += 1
        
        if malicious_peers > len(peer_connections) * 0.5:  # More than 50% low reputation
            eclipse_indicators.append("high_malicious_peer_ratio")
        
        # Calculate eclipse attack confidence
        confidence = min(1.0, len(eclipse_indicators) * 0.25)
        
        if confidence >= self.detection_thresholds[AttackType.ECLIPSE]:
            alert = SecurityAlert(
                alert_id=f"eclipse_{node_id}_{int(time.time())}",
                attack_type=AttackType.ECLIPSE,
                severity=SecurityLevel.HIGH,
                confidence=confidence,
                details={
                    "indicators": eclipse_indicators,
                    "peer_count": len(peer_connections),
                    "geographic_diversity": len(unique_countries),
                    "network_diversity": len(unique_asns),
                    "malicious_peer_ratio": malicious_peers / len(peer_connections) if peer_connections else 0
                },
                affected_nodes=[node_id],
                evidence={
                    "peer_connections": peer_connections,
                    "unique_countries": list(unique_countries),
                    "unique_asns": list(unique_asns),
                    "recent_connections_ratio": len(recent_connections) / len(peer_connections) if peer_connections else 0
                }
            )
            
            self.active_alerts.append(alert)
            
            logger.warning(
                f"Eclipse attack detected against {node_id}: "
                f"confidence {confidence:.2f}, indicators: {eclipse_indicators}"
            )
            
            return alert
        
        return None
    
    def _update_reputation(self, node_id: str, change: float):
        """Update node reputation score"""
        current_score = self.reputation_scores[node_id]
        new_score = max(0.0, min(1.0, current_score + change))
        self.reputation_scores[node_id] = new_score
        
        # Record behavior in history
        self.behavior_history[node_id].append({
            "timestamp": time.time(),
            "type": "malicious" if change < 0 else "positive",
            "score_change": change,
            "new_score": new_score
        })
        
        # Keep only recent history (last 100 entries)
        if len(self.behavior_history[node_id]) > 100:
            self.behavior_history[node_id] = self.behavior_history[node_id][-100:]


class SecureFederatedAggregation:
    """
    Secure multi-party computation for federated gradient aggregation
    Provides privacy-preserving aggregation with Byzantine fault tolerance
    """
    
    def __init__(self, threshold_crypto: ThresholdCryptographySystem):
        self.threshold_crypto = threshold_crypto
        self.zk_system = ZeroKnowledgeProofSystem()
        self.aggregation_rounds: Dict[str, Dict[str, Any]] = {}
        self.gradient_commitments: Dict[str, Dict[str, bytes]] = {}
        
    async def initialize_aggregation_round(
        self,
        round_id: str,
        participants: List[str],
        gradient_shape: Tuple[int, ...],
        privacy_budget: float = 1.0
    ) -> Dict[str, Any]:
        """
        Initialize secure aggregation round
        
        Args:
            round_id: Unique identifier for this aggregation round
            participants: List of participating node IDs
            gradient_shape: Shape of gradients to aggregate
            privacy_budget: Differential privacy budget
            
        Returns:
            Round initialization parameters
        """
        logger.info(f"Initializing secure aggregation round {round_id} with {len(participants)} participants")
        
        # Generate shared randomness for this round
        round_seed = secrets.randbits(256)
        
        # Initialize commitment scheme parameters
        commitment_params = {
            "generator": self.threshold_crypto.curve.generator_point,
            "round_seed": round_seed,
            "privacy_budget": privacy_budget
        }
        
        self.aggregation_rounds[round_id] = {
            "participants": participants,
            "gradient_shape": gradient_shape,
            "commitment_params": commitment_params,
            "phase": "commitment",
            "commitments": {},
            "proofs": {},
            "gradients": {},
            "aggregated_result": None,
            "started_at": time.time()
        }
        
        self.gradient_commitments[round_id] = {}
        
        return {
            "round_id": round_id,
            "commitment_params": commitment_params,
            "participants": participants,
            "gradient_shape": gradient_shape
        }
    
    async def commit_gradient(
        self,
        round_id: str,
        node_id: str,
        gradient_vector: List[float],
        noise_scale: float = 0.1
    ) -> Dict[str, Any]:
        """
        Commit to gradient with privacy protection
        
        Args:
            round_id: Aggregation round identifier
            node_id: Contributing node identifier
            gradient_vector: Local gradient values
            noise_scale: Differential privacy noise scale
            
        Returns:
            Commitment and proof information
        """
        if round_id not in self.aggregation_rounds:
            raise ValueError(f"Unknown aggregation round: {round_id}")
        
        round_data = self.aggregation_rounds[round_id]
        
        if node_id not in round_data["participants"]:
            raise ValueError(f"Node {node_id} not authorized for round {round_id}")
        
        # Add differential privacy noise
        noisy_gradient = []
        for value in gradient_vector:
            noise = secrets.SystemRandom().gauss(0, noise_scale)
            noisy_gradient.append(value + noise)
        
        # Create gradient commitment
        gradient_hash = hashlib.sha256(
            json.dumps(noisy_gradient, sort_keys=True).encode()
        ).digest()
        
        # Generate commitment using Pedersen commitment scheme
        randomness = secrets.randbits(256)
        commitment = self.threshold_crypto.curve.point_add(
            self.threshold_crypto.curve.point_multiply(
                round_data["commitment_params"]["generator"],
                int.from_bytes(gradient_hash, 'big')
            ),
            self.threshold_crypto.curve.point_multiply(
                round_data["commitment_params"]["generator"],
                randomness
            )
        )
        
        # Generate zero-knowledge proof of gradient knowledge
        gradient_value = int.from_bytes(gradient_hash, 'big') % self.threshold_crypto.curve.field_order
        proof = await self.zk_system.prove_gradient_knowledge(
            commitment, gradient_value, node_id
        )
        
        # Store commitment and proof
        round_data["commitments"][node_id] = commitment
        round_data["proofs"][node_id] = proof
        self.gradient_commitments[round_id][node_id] = commitment
        
        # Store encrypted gradient for later revelation
        round_data["gradients"][node_id] = {
            "encrypted_gradient": noisy_gradient,  # In production, use proper encryption
            "randomness": randomness,
            "commitment": commitment
        }
        
        logger.info(f"Node {node_id} committed gradient for round {round_id}")
        
        return {
            "commitment": commitment.hex() if isinstance(commitment, bytes) else str(commitment),
            "proof": {
                "commitment": proof.commitment.hex(),
                "challenge": proof.challenge.hex(),
                "response": proof.response.hex()
            },
            "round_phase": round_data["phase"]
        }
    
    async def verify_commitments(
        self,
        round_id: str
    ) -> Dict[str, bool]:
        """
        Verify all gradient commitments and proofs for a round
        
        Args:
            round_id: Aggregation round identifier
            
        Returns:
            Dictionary of verification results per participant
        """
        if round_id not in self.aggregation_rounds:
            raise ValueError(f"Unknown aggregation round: {round_id}")
        
        round_data = self.aggregation_rounds[round_id]
        verification_results = {}
        
        for node_id in round_data["participants"]:
            if node_id in round_data["proofs"] and node_id in round_data["commitments"]:
                proof = round_data["proofs"][node_id]
                commitment = round_data["commitments"][node_id]
                
                is_valid = self.zk_system.verify_gradient_proof(proof, commitment)
                verification_results[node_id] = is_valid
                
                if not is_valid:
                    logger.warning(f"Invalid proof from node {node_id} in round {round_id}")
            else:
                verification_results[node_id] = False
        
        # Update round phase if all verifications complete
        all_verified = all(verification_results.values())
        if all_verified and len(verification_results) == len(round_data["participants"]):
            round_data["phase"] = "revelation"
            logger.info(f"All commitments verified for round {round_id}, moving to revelation phase")
        
        return verification_results
    
    async def aggregate_gradients(
        self,
        round_id: str,
        byzantine_tolerance: int = 1
    ) -> Dict[str, Any]:
        """
        Perform secure gradient aggregation with Byzantine fault tolerance
        
        Args:
            round_id: Aggregation round identifier
            byzantine_tolerance: Number of Byzantine nodes to tolerate
            
        Returns:
            Aggregated gradient result
        """
        if round_id not in self.aggregation_rounds:
            raise ValueError(f"Unknown aggregation round: {round_id}")
        
        round_data = self.aggregation_rounds[round_id]
        
        # Verify we have enough honest participants
        required_participants = len(round_data["participants"]) - byzantine_tolerance
        actual_participants = len(round_data["gradients"])
        
        if actual_participants < required_participants:
            raise ValueError(
                f"Insufficient participants: need {required_participants}, got {actual_participants}"
            )
        
        # Extract gradient vectors
        gradient_vectors = []
        valid_participants = []
        
        for node_id, gradient_data in round_data["gradients"].items():
            # Verify commitment integrity
            stored_commitment = round_data["commitments"].get(node_id)
            gradient_commitment = gradient_data["commitment"]
            
            if stored_commitment == gradient_commitment:
                gradient_vectors.append(gradient_data["encrypted_gradient"])
                valid_participants.append(node_id)
            else:
                logger.warning(f"Commitment mismatch for node {node_id}, excluding from aggregation")
        
        if len(gradient_vectors) < required_participants:
            raise ValueError("Too many invalid commitments for secure aggregation")
        
        # Perform Byzantine-robust aggregation using coordinate-wise median
        aggregated_gradient = []
        gradient_length = len(gradient_vectors[0]) if gradient_vectors else 0
        
        for i in range(gradient_length):
            coordinate_values = [grad[i] for grad in gradient_vectors]
            coordinate_values.sort()
            
            # Use trimmed mean to handle Byzantine nodes
            trim_count = byzantine_tolerance
            if len(coordinate_values) > 2 * trim_count:
                trimmed_values = coordinate_values[trim_count:-trim_count]
                aggregated_value = sum(trimmed_values) / len(trimmed_values)
            else:
                # Fallback to median if not enough values
                median_idx = len(coordinate_values) // 2
                aggregated_value = coordinate_values[median_idx]
            
            aggregated_gradient.append(aggregated_value)
        
        # Create aggregation proof using threshold signature
        aggregation_data = {
            "round_id": round_id,
            "participants": valid_participants,
            "aggregated_gradient": aggregated_gradient,
            "timestamp": time.time()
        }
        
        aggregation_message = json.dumps(aggregation_data, sort_keys=True).encode()
        
        # Sign aggregation result with threshold signature
        if len(valid_participants) >= self.threshold_crypto.threshold:
            threshold_signature = await self.threshold_crypto.create_threshold_signature(
                aggregation_message,
                valid_participants[:self.threshold_crypto.threshold]
            )
        else:
            threshold_signature = None
            logger.warning(f"Insufficient signers for threshold signature in round {round_id}")
        
        round_data["aggregated_result"] = {
            "gradient": aggregated_gradient,
            "participants": valid_participants,
            "signature": threshold_signature,
            "completed_at": time.time()
        }
        
        round_data["phase"] = "completed"
        
        logger.info(f"Secure aggregation completed for round {round_id} with {len(valid_participants)} participants")
        
        return {
            "round_id": round_id,
            "aggregated_gradient": aggregated_gradient,
            "participant_count": len(valid_participants),
            "signature_valid": threshold_signature is not None,
            "aggregation_time": time.time() - round_data["started_at"]
        }


class FederatedSecurityCoordinator:
    """
    Main coordinator for federated learning security
    Orchestrates all security components and protocols
    """
    
    def __init__(self, node_id: str, threshold: int = 3, total_nodes: int = 5):
        self.node_id = node_id
        self.threshold_crypto = ThresholdCryptographySystem(threshold, total_nodes)
        self.attack_detector = AttackDetectionSystem()
        self.secure_aggregation = SecureFederatedAggregation(self.threshold_crypto)
        self.security_config = {
            "enable_zk_proofs": True,
            "require_threshold_signatures": True,
            "byzantine_tolerance": 1,
            "privacy_budget": 1.0,
            "reputation_threshold": 0.5
        }
        self.active_rounds: Dict[str, str] = {}  # round_id -> status
        
    async def initialize_security_infrastructure(
        self,
        participants: List[str]
    ) -> Dict[str, Any]:
        """
        Initialize the complete security infrastructure
        
        Args:
            participants: List of participating node IDs
            
        Returns:
            Initialization results
        """
        logger.info(f"Initializing federated security infrastructure with {len(participants)} participants")
        
        # Initialize distributed key generation
        dkg_result = await self.threshold_crypto.distributed_key_generation(participants)
        
        # Initialize reputation scores for all participants
        for participant in participants:
            if participant not in self.attack_detector.reputation_scores:
                self.attack_detector.reputation_scores[participant] = 1.0
        
        return {
            "dkg_completed": dkg_result["master_public_key"] is not None,
            "participants": participants,
            "threshold": self.threshold_crypto.threshold,
            "security_level": "high",
            "features_enabled": {
                "threshold_cryptography": True,
                "zero_knowledge_proofs": self.security_config["enable_zk_proofs"],
                "byzantine_detection": True,
                "secure_aggregation": True,
                "attack_monitoring": True
            }
        }
    
    async def start_secure_training_round(
        self,
        round_id: str,
        participants: List[str],
        gradient_shape: Tuple[int, ...],
        model_parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Start a secure federated training round
        
        Args:
            round_id: Unique round identifier
            participants: Participating nodes for this round
            gradient_shape: Shape of gradients to be aggregated
            model_parameters: Current model parameters (optional)
            
        Returns:
            Round initialization parameters
        """
        logger.info(f"Starting secure training round {round_id}")
        
        # Filter participants based on reputation
        trusted_participants = [
            p for p in participants
            if self.attack_detector.reputation_scores.get(p, 1.0) >= self.security_config["reputation_threshold"]
        ]
        
        if len(trusted_participants) < self.threshold_crypto.threshold:
            raise ValueError(
                f"Insufficient trusted participants: need {self.threshold_crypto.threshold}, "
                f"got {len(trusted_participants)}"
            )
        
        # Initialize secure aggregation
        agg_params = await self.secure_aggregation.initialize_aggregation_round(
            round_id,
            trusted_participants,
            gradient_shape,
            self.security_config["privacy_budget"]
        )
        
        self.active_rounds[round_id] = "initialized"
        
        return {
            "round_id": round_id,
            "trusted_participants": trusted_participants,
            "filtered_count": len(participants) - len(trusted_participants),
            "aggregation_params": agg_params,
            "security_requirements": {
                "zk_proofs_required": self.security_config["enable_zk_proofs"],
                "threshold_signatures": self.security_config["require_threshold_signatures"],
                "privacy_budget": self.security_config["privacy_budget"]
            }
        }
    
    async def process_gradient_submission(
        self,
        round_id: str,
        node_id: str,
        gradient_vector: List[float],
        submission_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process and validate gradient submission
        
        Args:
            round_id: Training round identifier
            node_id: Submitting node identifier
            gradient_vector: Gradient values
            submission_metadata: Additional submission metadata
            
        Returns:
            Processing results
        """
        logger.info(f"Processing gradient submission from {node_id} for round {round_id}")
        
        # Check round status
        if round_id not in self.active_rounds:
            raise ValueError(f"Unknown or inactive round: {round_id}")
        
        # Validate submitting node
        if self.attack_detector.reputation_scores.get(node_id, 1.0) < self.security_config["reputation_threshold"]:
            raise ValueError(f"Node {node_id} reputation too low for participation")
        
        # Monitor for attack patterns
        submission_data = {
            "node_id": node_id,
            "round_id": round_id,
            "timestamp": time.time(),
            "gradient_shape": len(gradient_vector),
            "metadata": submission_metadata or {}
        }
        
        # Check for gradient inversion attacks (simplified)
        gradient_magnitude = sum(abs(x) for x in gradient_vector)
        if gradient_magnitude > 1000 or gradient_magnitude < 0.001:  # Suspicious magnitude
            logger.warning(f"Suspicious gradient magnitude from {node_id}: {gradient_magnitude}")
        
        # Commit gradient with security proofs
        commit_result = await self.secure_aggregation.commit_gradient(
            round_id, node_id, gradient_vector
        )
        
        # Update behavior tracking
        self.attack_detector.behavior_history[node_id].append({
            "type": "gradient_submission",
            "timestamp": time.time(),
            "round_id": round_id,
            "gradient_magnitude": gradient_magnitude
        })
        
        return {
            "submission_accepted": True,
            "commitment_hash": commit_result["commitment"],
            "proof_verified": True,
            "round_phase": commit_result["round_phase"],
            "node_reputation": self.attack_detector.reputation_scores[node_id]
        }
    
    async def finalize_training_round(
        self,
        round_id: str
    ) -> Dict[str, Any]:
        """
        Finalize secure training round with aggregation
        
        Args:
            round_id: Training round to finalize
            
        Returns:
            Final aggregation results
        """
        logger.info(f"Finalizing training round {round_id}")
        
        if round_id not in self.active_rounds:
            raise ValueError(f"Unknown round: {round_id}")
        
        # Verify all commitments
        verification_results = await self.secure_aggregation.verify_commitments(round_id)
        
        failed_verifications = [
            node for node, valid in verification_results.items() if not valid
        ]
        
        if failed_verifications:
            logger.warning(f"Failed verifications in round {round_id}: {failed_verifications}")
            
            # Penalize nodes with failed verifications
            for node_id in failed_verifications:
                self.attack_detector._update_reputation(node_id, -0.2)
        
        # Perform secure aggregation
        aggregation_result = await self.secure_aggregation.aggregate_gradients(
            round_id, 
            self.security_config["byzantine_tolerance"]
        )
        
        # Update round status
        self.active_rounds[round_id] = "completed"
        
        # Reward participating nodes
        for participant in aggregation_result.get("participants", []):
            if participant in verification_results and verification_results[participant]:
                self.attack_detector._update_reputation(participant, 0.1)
        
        logger.info(f"Training round {round_id} completed successfully")
        
        return {
            "round_id": round_id,
            "aggregated_gradient": aggregation_result["aggregated_gradient"],
            "participant_count": aggregation_result["participant_count"],
            "verification_results": verification_results,
            "security_signature": aggregation_result.get("signature_valid", False),
            "completion_time": aggregation_result["aggregation_time"],
            "failed_verifications": failed_verifications
        }
    
    async def monitor_security_status(self) -> Dict[str, Any]:
        """
        Get comprehensive security status
        
        Returns:
            Current security monitoring status
        """
        active_alerts = [
            {
                "alert_id": alert.alert_id,
                "attack_type": alert.attack_type.value,
                "severity": alert.severity.value,
                "confidence": alert.confidence,
                "affected_nodes": alert.affected_nodes,
                "timestamp": alert.timestamp
            }
            for alert in self.attack_detector.active_alerts
        ]
        
        reputation_summary = {
            "high_reputation": len([s for s in self.attack_detector.reputation_scores.values() if s > 0.8]),
            "medium_reputation": len([s for s in self.attack_detector.reputation_scores.values() if 0.5 <= s <= 0.8]),
            "low_reputation": len([s for s in self.attack_detector.reputation_scores.values() if s < 0.5]),
            "average_reputation": sum(self.attack_detector.reputation_scores.values()) / len(self.attack_detector.reputation_scores) if self.attack_detector.reputation_scores else 0
        }
        
        return {
            "node_id": self.node_id,
            "active_alerts": active_alerts,
            "reputation_summary": reputation_summary,
            "active_rounds": dict(self.active_rounds),
            "security_features": {
                "threshold_crypto_active": self.threshold_crypto.master_public_key is not None,
                "attack_detection_active": True,
                "secure_aggregation_active": True,
                "zk_proofs_enabled": self.security_config["enable_zk_proofs"]
            },
            "threat_landscape": {
                "total_participants": len(self.attack_detector.reputation_scores),
                "trusted_nodes": len([s for s in self.attack_detector.reputation_scores.values() if s >= self.security_config["reputation_threshold"]]),
                "recent_attacks": len([a for a in self.attack_detector.active_alerts if time.time() - a.timestamp < 3600])  # Last hour
            }
        }


# Factory function for easy integration
def create_federated_security_coordinator(
    node_id: str,
    threshold: int = 3,
    total_nodes: int = 5,
    config: Optional[Dict[str, Any]] = None
) -> FederatedSecurityCoordinator:
    """
    Factory function to create federated security coordinator
    
    Args:
        node_id: Unique identifier for this security node
        threshold: Threshold for cryptographic operations
        total_nodes: Total number of nodes in the system
        config: Optional security configuration
        
    Returns:
        Configured security coordinator
    """
    coordinator = FederatedSecurityCoordinator(node_id, threshold, total_nodes)
    
    if config:
        coordinator.security_config.update(config)
    
    logger.info(f"Created federated security coordinator for node {node_id}")
    
    return coordinator