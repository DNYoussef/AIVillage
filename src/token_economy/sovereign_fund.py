from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class TreasuryManager:
    total_value: float = 0.0
    distributions: dict[str, float] = field(default_factory=dict)

    def get_total_value(self) -> float:
        return self.total_value

    def distribute(self, wallet: str, amount: float) -> None:
        logger.info("Distributing %.2f to %s", amount, wallet)
        self.distributions[wallet] = self.distributions.get(wallet, 0) + amount
        self.total_value -= amount


@dataclass
class InvestmentManager:
    conservative: dict[str, float] = field(default_factory=dict)
    aggressive: dict[str, float] = field(default_factory=dict)

    def allocate_conservative(self, allocations: dict[str, float]) -> None:
        logger.info("Conservative allocation: %s", allocations)
        self.conservative.update(allocations)

    def allocate_aggressive(self, allocations: dict[str, float]) -> None:
        logger.info("Aggressive allocation: %s", allocations)
        self.aggressive.update(allocations)


class DigitalSovereignWealthFund:
    def __init__(self) -> None:
        """Initialise the sovereign wealth fund with treasury and investments."""
        self.treasury = TreasuryManager()
        self.investments = InvestmentManager()

    def implement_barbell_strategy(self) -> None:
        total_funds = self.treasury.get_total_value()
        conservative_allocation = total_funds * 0.8
        self.investments.allocate_conservative(
            {
                "stablecoins": conservative_allocation * 0.5,
                "government_bonds": conservative_allocation * 0.3,
                "gold_backed": conservative_allocation * 0.2,
            }
        )
        aggressive_allocation = total_funds * 0.2
        self.investments.allocate_aggressive(
            {
                "ai_startups": aggressive_allocation * 0.4,
                "compute_infrastructure": aggressive_allocation * 0.3,
                "research_grants": aggressive_allocation * 0.3,
            }
        )
        logger.info("Barbell strategy implemented for %.2f funds", total_funds)

    def calculate_monthly_yield(self) -> float:
        return self.treasury.get_total_value() * 0.02

    @dataclass
    class User:
        """Eligible user for UBI distribution."""

        wallet: str

    def get_eligible_users(self) -> list["DigitalSovereignWealthFund.User"]:
        return []

    def calculate_contribution_multiplier(
        self, _user: "DigitalSovereignWealthFund.User"
    ) -> float:
        return 1.0

    def distribute_ubi(self) -> None:
        monthly_yield = self.calculate_monthly_yield()
        eligible_users = self.get_eligible_users()
        if not eligible_users:
            logger.warning("No eligible users for UBI distribution")
            return
        base_amount = (monthly_yield * 0.5) / len(eligible_users)
        for user in eligible_users:
            amount = base_amount * self.calculate_contribution_multiplier(user)
            self.treasury.distribute(user.wallet, amount)
