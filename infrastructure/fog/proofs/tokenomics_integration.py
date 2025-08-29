"""
Proof System Tokenomics Integration

Integrates cryptographic proofs with the fog tokenomics system:
- Proof-based reward calculation
- Token distribution for valid proofs
- Quality bonuses and penalties
- Consensus participation rewards
"""

from dataclasses import dataclass
from decimal import Decimal
import logging
from typing import Any

from .proof_generator import CryptographicProof, ProofOfAudit, ProofOfExecution, ProofOfSLA, ProofType
from .proof_verifier import VerificationReport, VerificationResult

logger = logging.getLogger(__name__)


@dataclass
class ProofReward:
    """Reward calculation for a cryptographic proof"""

    proof_id: str
    proof_type: ProofType
    base_reward: Decimal
    quality_bonus: Decimal
    verification_bonus: Decimal
    consensus_bonus: Decimal
    total_reward: Decimal
    penalty_amount: Decimal
    net_reward: Decimal
    reward_factors: dict[str, float]


class ProofTokenomicsIntegration:
    """
    Integration between proof system and tokenomics

    Handles:
    - Reward calculation based on proof type and quality
    - Token distribution for proof generation and verification
    - Quality bonuses for high-performance proofs
    - Penalties for invalid or low-quality proofs
    - Consensus participation rewards
    """

    def __init__(self, fog_token_system, reward_config: dict[str, Any] = None):
        self.fog_token_system = fog_token_system

        # Default reward configuration
        self.reward_config = {
            "base_rewards": {
                ProofType.PROOF_OF_EXECUTION.value: Decimal("10.0"),  # 10 FOG per execution proof
                ProofType.PROOF_OF_AUDIT.value: Decimal("5.0"),  # 5 FOG per audit proof
                ProofType.PROOF_OF_SLA.value: Decimal("15.0"),  # 15 FOG per SLA proof
                ProofType.MERKLE_BATCH.value: Decimal("25.0"),  # 25 FOG per batch proof
            },
            "quality_multipliers": {
                "high_efficiency": 1.5,  # 50% bonus for high resource efficiency
                "perfect_sla": 1.3,  # 30% bonus for perfect SLA compliance
                "strong_consensus": 1.2,  # 20% bonus for strong audit consensus
                "fast_execution": 1.1,  # 10% bonus for fast task completion
            },
            "verification_rewards": {
                "proof_verification": Decimal("1.0"),  # 1 FOG per proof verification
                "batch_verification": Decimal("5.0"),  # 5 FOG per batch verification
            },
            "consensus_rewards": {
                "audit_participation": Decimal("2.0"),  # 2 FOG per audit participation
                "consensus_leader": Decimal("5.0"),  # 5 FOG for consensus leadership
            },
            "penalties": {
                "invalid_proof": Decimal("5.0"),  # 5 FOG penalty for invalid proofs
                "sla_breach": Decimal("10.0"),  # 10 FOG penalty for SLA breaches
                "consensus_failure": Decimal("3.0"),  # 3 FOG penalty for consensus failures
            },
            "minimum_reward": Decimal("0.1"),  # Minimum reward to prevent zero payouts
            "maximum_bonus": Decimal("100.0"),  # Maximum bonus multiplier
        }

        # Update with custom config
        if reward_config:
            self.reward_config.update(reward_config)

        # Statistics
        self.stats = {
            "total_rewards_calculated": 0,
            "total_rewards_distributed": Decimal("0.0"),
            "total_penalties_applied": Decimal("0.0"),
            "rewards_by_type": {},
            "quality_bonuses_given": 0,
            "verification_rewards_given": 0,
        }

        logger.info("Proof tokenomics integration initialized")

    async def calculate_proof_reward(
        self, proof: CryptographicProof, verification_report: VerificationReport | None = None
    ) -> ProofReward:
        """
        Calculate reward for a cryptographic proof

        Args:
            proof: The cryptographic proof
            verification_report: Optional verification report

        Returns:
            ProofReward object with detailed breakdown
        """
        try:
            # Get base reward for proof type
            base_reward = self.reward_config["base_rewards"].get(proof.proof_type.value, Decimal("1.0"))

            # Calculate quality bonus
            quality_bonus = await self._calculate_quality_bonus(proof)

            # Calculate verification bonus
            verification_bonus = self._calculate_verification_bonus(verification_report)

            # Calculate consensus bonus
            consensus_bonus = await self._calculate_consensus_bonus(proof)

            # Calculate penalties
            penalty_amount = await self._calculate_penalties(proof, verification_report)

            # Apply bonuses
            total_bonus = quality_bonus + verification_bonus + consensus_bonus
            total_reward = base_reward + (base_reward * total_bonus)

            # Apply minimum reward
            total_reward = max(total_reward, self.reward_config["minimum_reward"])

            # Apply penalties
            net_reward = max(Decimal("0.0"), total_reward - penalty_amount)

            # Create reward factors summary
            reward_factors = {
                "base_reward_factor": float(base_reward),
                "quality_bonus_factor": float(quality_bonus),
                "verification_bonus_factor": float(verification_bonus),
                "consensus_bonus_factor": float(consensus_bonus),
                "penalty_factor": float(penalty_amount),
            }

            reward = ProofReward(
                proof_id=proof.proof_id,
                proof_type=proof.proof_type,
                base_reward=base_reward,
                quality_bonus=quality_bonus * base_reward,
                verification_bonus=verification_bonus * base_reward,
                consensus_bonus=consensus_bonus * base_reward,
                total_reward=total_reward,
                penalty_amount=penalty_amount,
                net_reward=net_reward,
                reward_factors=reward_factors,
            )

            # Update statistics
            self.stats["total_rewards_calculated"] += 1
            proof_type_key = proof.proof_type.value
            if proof_type_key not in self.stats["rewards_by_type"]:
                self.stats["rewards_by_type"][proof_type_key] = {"count": 0, "total_amount": Decimal("0.0")}
            self.stats["rewards_by_type"][proof_type_key]["count"] += 1
            self.stats["rewards_by_type"][proof_type_key]["total_amount"] += net_reward

            logger.debug(
                f"Calculated reward for proof {proof.proof_id}: "
                f"{net_reward} FOG (base: {base_reward}, bonuses: {total_bonus:.2%})"
            )

            return reward

        except Exception as e:
            logger.error(f"Error calculating proof reward: {e}")
            raise

    async def _calculate_quality_bonus(self, proof: CryptographicProof) -> Decimal:
        """Calculate quality bonus based on proof type and metadata"""
        quality_bonus = Decimal("0.0")

        try:
            if isinstance(proof, ProofOfExecution):
                # Execution efficiency bonus
                efficiency = proof.metadata.get("resource_efficiency", 0.5)
                if efficiency > 0.8:
                    quality_bonus += Decimal(str(self.reward_config["quality_multipliers"]["high_efficiency"] - 1))
                    self.stats["quality_bonuses_given"] += 1

                # Fast execution bonus
                duration = proof.metadata.get("execution_duration", 3600)
                if duration < 300:  # Less than 5 minutes
                    quality_bonus += Decimal(str(self.reward_config["quality_multipliers"]["fast_execution"] - 1))
                    self.stats["quality_bonuses_given"] += 1

            elif isinstance(proof, ProofOfAudit):
                # Strong consensus bonus
                consensus = proof.achieved_consensus
                if consensus > 0.9:  # > 90% consensus
                    quality_bonus += Decimal(str(self.reward_config["quality_multipliers"]["strong_consensus"] - 1))
                    self.stats["quality_bonuses_given"] += 1

            elif isinstance(proof, ProofOfSLA):
                # Perfect SLA compliance bonus
                compliance = proof.metadata.get("compliance_percentage", 95.0)
                if compliance >= 100.0:
                    quality_bonus += Decimal(str(self.reward_config["quality_multipliers"]["perfect_sla"] - 1))
                    self.stats["quality_bonuses_given"] += 1

            # Cap quality bonus
            quality_bonus = min(quality_bonus, Decimal("2.0"))  # Max 200% bonus

        except Exception as e:
            logger.error(f"Error calculating quality bonus: {e}")

        return quality_bonus

    def _calculate_verification_bonus(self, verification_report: VerificationReport | None) -> Decimal:
        """Calculate bonus for proof verification"""
        if not verification_report:
            return Decimal("0.0")

        try:
            if verification_report.result == VerificationResult.VALID:
                # Base verification bonus
                bonus = Decimal("0.1")  # 10% bonus for verified proofs

                # Fast verification bonus
                if verification_report.verification_time_ms < 1000:  # Less than 1 second
                    bonus += Decimal("0.05")  # Additional 5% bonus

                self.stats["verification_rewards_given"] += 1
                return bonus

        except Exception as e:
            logger.error(f"Error calculating verification bonus: {e}")

        return Decimal("0.0")

    async def _calculate_consensus_bonus(self, proof: CryptographicProof) -> Decimal:
        """Calculate bonus for consensus participation"""
        consensus_bonus = Decimal("0.0")

        try:
            if isinstance(proof, ProofOfAudit):
                # Bonus for audit consensus participation
                auditor_count = len(proof.audit_evidence)
                if auditor_count >= 3:  # Minimum viable consensus
                    consensus_bonus += Decimal("0.1")  # 10% bonus

                # Additional bonus for strong consensus
                if proof.achieved_consensus > 0.8:
                    consensus_bonus += Decimal("0.05")  # Additional 5% bonus

            elif proof.proof_type == ProofType.MERKLE_BATCH:
                # Bonus for batch proof coordination
                batch_size = proof.metadata.get("batch_size", 1)
                if batch_size >= 10:
                    consensus_bonus += Decimal("0.15")  # 15% bonus for large batches
                elif batch_size >= 5:
                    consensus_bonus += Decimal("0.08")  # 8% bonus for medium batches

        except Exception as e:
            logger.error(f"Error calculating consensus bonus: {e}")

        return consensus_bonus

    async def _calculate_penalties(
        self, proof: CryptographicProof, verification_report: VerificationReport | None
    ) -> Decimal:
        """Calculate penalties for invalid or poor quality proofs"""
        penalty_amount = Decimal("0.0")

        try:
            # Invalid proof penalty
            if verification_report and verification_report.result != VerificationResult.VALID:
                penalty_amount += self.reward_config["penalties"]["invalid_proof"]
                self.stats["total_penalties_applied"] += penalty_amount

            # Type-specific penalties
            if isinstance(proof, ProofOfSLA):
                compliance = proof.metadata.get("compliance_percentage", 100.0)
                if compliance < 95.0:  # Below 95% compliance
                    penalty_amount += self.reward_config["penalties"]["sla_breach"]
                    self.stats["total_penalties_applied"] += penalty_amount

            elif isinstance(proof, ProofOfAudit):
                if proof.achieved_consensus < proof.consensus_threshold:
                    penalty_amount += self.reward_config["penalties"]["consensus_failure"]
                    self.stats["total_penalties_applied"] += penalty_amount

        except Exception as e:
            logger.error(f"Error calculating penalties: {e}")

        return penalty_amount

    async def distribute_proof_reward(self, proof_reward: ProofReward, recipient_account: str) -> bool:
        """
        Distribute calculated reward to recipient account

        Args:
            proof_reward: Calculated proof reward
            recipient_account: Account ID to receive the reward

        Returns:
            True if distribution successful
        """
        try:
            if proof_reward.net_reward <= 0:
                logger.info(f"No reward to distribute for proof {proof_reward.proof_id}")
                return True

            # Record the reward as a contribution
            contribution_metrics = {
                "proof_type": proof_reward.proof_type.value,
                "proof_id": proof_reward.proof_id,
                "base_reward": float(proof_reward.base_reward),
                "quality_bonus": float(proof_reward.quality_bonus),
                "verification_bonus": float(proof_reward.verification_bonus),
                "consensus_bonus": float(proof_reward.consensus_bonus),
                "penalty_amount": float(proof_reward.penalty_amount),
                "net_reward": float(proof_reward.net_reward),
                "reward_factors": proof_reward.reward_factors,
            }

            # Use the fog token system to record contribution and distribute reward
            await self.fog_token_system.record_contribution(
                contributor_id=recipient_account,
                device_id=f"proof_device_{recipient_account}",
                metrics={
                    "compute_hours": float(proof_reward.net_reward) / 10.0,  # Convert to compute hours equivalent
                    "tasks_completed": 1,
                    "uptime_percent": 100.0,
                    "success_rate": 100.0,
                    "proof_reward": contribution_metrics,
                },
            )

            # Update statistics
            self.stats["total_rewards_distributed"] += proof_reward.net_reward

            logger.info(
                f"Distributed {proof_reward.net_reward} FOG to {recipient_account} "
                f"for proof {proof_reward.proof_id}"
            )

            return True

        except Exception as e:
            logger.error(f"Error distributing proof reward: {e}")
            return False

    async def distribute_verification_reward(
        self, verifier_account: str, proof_type: ProofType, verification_successful: bool
    ) -> bool:
        """Distribute reward for proof verification"""
        try:
            if not verification_successful:
                logger.debug(f"No verification reward for failed verification by {verifier_account}")
                return True

            # Calculate verification reward
            if proof_type == ProofType.MERKLE_BATCH:
                reward_amount = self.reward_config["verification_rewards"]["batch_verification"]
            else:
                reward_amount = self.reward_config["verification_rewards"]["proof_verification"]

            # Distribute reward
            success = await self.fog_token_system.transfer(
                from_account="system",
                to_account=verifier_account,
                amount=float(reward_amount),
                description=f"Proof verification reward - {proof_type.value}",
            )

            if success:
                self.stats["verification_rewards_given"] += 1
                self.stats["total_rewards_distributed"] += reward_amount

                logger.debug(f"Distributed {reward_amount} FOG verification reward to {verifier_account}")

            return success

        except Exception as e:
            logger.error(f"Error distributing verification reward: {e}")
            return False

    async def distribute_consensus_reward(
        self, participants: list[str], consensus_type: str = "audit_participation"
    ) -> int:
        """Distribute rewards for consensus participation"""
        successful_distributions = 0

        try:
            reward_amount = self.reward_config["consensus_rewards"].get(
                consensus_type, self.reward_config["consensus_rewards"]["audit_participation"]
            )

            for participant in participants:
                try:
                    success = await self.fog_token_system.transfer(
                        from_account="system",
                        to_account=participant,
                        amount=float(reward_amount),
                        description=f"Consensus participation reward - {consensus_type}",
                    )

                    if success:
                        successful_distributions += 1
                        self.stats["total_rewards_distributed"] += reward_amount

                        logger.debug(f"Distributed {reward_amount} FOG consensus reward to {participant}")

                except Exception as e:
                    logger.error(f"Error distributing consensus reward to {participant}: {e}")

        except Exception as e:
            logger.error(f"Error distributing consensus rewards: {e}")

        return successful_distributions

    def get_reward_statistics(self) -> dict[str, Any]:
        """Get reward distribution statistics"""
        return {
            **self.stats,
            "reward_config": {
                "base_rewards": {k: float(v) for k, v in self.reward_config["base_rewards"].items()},
                "quality_multipliers": self.reward_config["quality_multipliers"],
                "verification_rewards": {k: float(v) for k, v in self.reward_config["verification_rewards"].items()},
                "consensus_rewards": {k: float(v) for k, v in self.reward_config["consensus_rewards"].items()},
                "penalties": {k: float(v) for k, v in self.reward_config["penalties"].items()},
            },
        }

    def update_reward_config(self, new_config: dict[str, Any]):
        """Update reward configuration"""
        self.reward_config.update(new_config)
        logger.info("Updated proof reward configuration")

    async def calculate_batch_reward(
        self, proofs: list[CryptographicProof], verification_reports: list[VerificationReport]
    ) -> list[ProofReward]:
        """Calculate rewards for a batch of proofs"""
        rewards = []

        for i, proof in enumerate(proofs):
            verification_report = verification_reports[i] if i < len(verification_reports) else None

            try:
                reward = await self.calculate_proof_reward(proof, verification_report)
                rewards.append(reward)
            except Exception as e:
                logger.error(f"Error calculating reward for proof {proof.proof_id}: {e}")
                # Create zero reward for failed calculations
                zero_reward = ProofReward(
                    proof_id=proof.proof_id,
                    proof_type=proof.proof_type,
                    base_reward=Decimal("0.0"),
                    quality_bonus=Decimal("0.0"),
                    verification_bonus=Decimal("0.0"),
                    consensus_bonus=Decimal("0.0"),
                    total_reward=Decimal("0.0"),
                    penalty_amount=Decimal("0.0"),
                    net_reward=Decimal("0.0"),
                    reward_factors={},
                )
                rewards.append(zero_reward)

        return rewards
