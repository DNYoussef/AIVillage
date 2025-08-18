import logging
from dataclasses import dataclass

from .credit_system import VILLAGECreditSystem

logger = logging.getLogger(__name__)

REWARD_PER_OP = 0.001
UNDERSERVED_REGIONS = {"Sub-Saharan Africa", "South Asia"}


@dataclass
class ComputeSession:
    user_id: str
    operations: int
    duration: float
    model_id: str
    proof: str
    used_green_energy: bool
    device_location: str


class ComputeMiningSystem:
    def __init__(self, credit_system: VILLAGECreditSystem) -> None:
        """Create a compute mining system tied to a credit system."""
        self.credit_system = credit_system

    def verify_computation(self, proof: str) -> bool:
        return bool(proof)

    def detect_gaming(self, device_id: str, session: ComputeSession) -> bool:
        if session.operations <= 0 or session.duration <= 0:
            logger.warning("Invalid session from %s: %s", device_id, session)
            return True
        return False

    def track_compute_contribution(self, device_id: str, session: ComputeSession) -> int:
        if not self.verify_computation(session.proof):
            return 0
        if self.detect_gaming(device_id, session):
            return 0
        base_reward = session.operations * REWARD_PER_OP
        if session.used_green_energy:
            base_reward *= 1.5
        if session.device_location in UNDERSERVED_REGIONS:
            base_reward *= 2
        metadata: dict[str, str] = {
            "operations": session.operations,
            "duration": session.duration,
            "model_trained": session.model_id,
        }
        earned = self.credit_system.earn_credits(
            user_id=session.user_id,
            action="COMPUTE_CONTRIBUTION",
            metadata=metadata,
        )
        logger.info("Device %s awarded %d credits", device_id, earned)
        return earned
