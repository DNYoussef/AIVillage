"""
Validation Manager for Proposal Validation

Handles validation of proposed nodes and relationships,
learning from feedback to improve future proposals.
"""

import logging
from typing import Any

from .graph_types import ProposedNode, ProposedRelationship

logger = logging.getLogger(__name__)


class ValidationManager:
    """
    Manages validation of proposals and learning from feedback.

    Follows single responsibility principle for validation concerns.
    """

    def __init__(self):
        self._validation_history: list[dict[str, Any]] = []
        self._learning_metrics = {
            "total_validations": 0,
            "accepted_proposals": 0,
            "rejected_proposals": 0,
            "acceptance_rate": 0.0,
        }

    async def validate_proposal(
        self, proposal: ProposedNode | ProposedRelationship, validation_feedback: str, is_accepted: bool
    ) -> bool:
        """
        Validate a proposed node or relationship.

        Args:
            proposal: The proposal to validate
            validation_feedback: Human feedback on the proposal
            is_accepted: Whether the proposal was accepted

        Returns:
            True if validation was successful
        """
        try:
            # Update proposal status
            if is_accepted:
                proposal.validation_status = "validated"
                self._learning_metrics["accepted_proposals"] += 1
            else:
                proposal.validation_status = "rejected"
                self._learning_metrics["rejected_proposals"] += 1

            proposal.validation_feedback = validation_feedback

            # Record validation for learning
            self._record_validation(proposal, is_accepted, validation_feedback)

            # Update metrics
            self._learning_metrics["total_validations"] += 1
            self._update_acceptance_rate()

            # Learn from validation
            await self._learn_from_validation(proposal, is_accepted)

            return True

        except Exception as e:
            logger.exception(f"Proposal validation failed: {e}")
            return False

    def get_validation_stats(self) -> dict[str, Any]:
        """Get validation statistics."""
        return {
            "validation_metrics": self._learning_metrics.copy(),
            "total_history_entries": len(self._validation_history),
            "recent_validations": self._validation_history[-10:] if self._validation_history else [],
        }

    def _record_validation(
        self, proposal: ProposedNode | ProposedRelationship, is_accepted: bool, feedback: str
    ) -> None:
        """Record validation details for learning."""
        validation_record = {
            "proposal_id": proposal.id,
            "proposal_type": type(proposal).__name__,
            "gap_id": proposal.gap_id,
            "is_accepted": is_accepted,
            "feedback": feedback,
            "confidence": proposal.confidence,
            "utility_score": proposal.utility_score,
            "existence_probability": proposal.existence_probability,
        }

        self._validation_history.append(validation_record)

        # Keep history manageable
        if len(self._validation_history) > 1000:
            self._validation_history = self._validation_history[-500:]

    async def _learn_from_validation(self, proposal: ProposedNode | ProposedRelationship, is_accepted: bool) -> None:
        """Learn from validation feedback to improve future proposals."""
        # Log the learning event
        logger.info(f"Validation feedback: {proposal.id} " f"{'accepted' if is_accepted else 'rejected'}")

        # Future improvements could include:
        # - Adjusting confidence scoring algorithms based on success patterns
        # - Learning which gap types have higher acceptance rates
        # - Identifying common patterns in rejected proposals
        # - Updating detection method weights based on proposal success

    def _update_acceptance_rate(self) -> None:
        """Update the overall acceptance rate metric."""
        total = self._learning_metrics["total_validations"]
        if total > 0:
            accepted = self._learning_metrics["accepted_proposals"]
            self._learning_metrics["acceptance_rate"] = accepted / total
