"""
Fog Tokenomics Service

Manages token economics and reward distribution including:
- Token account management
- Reward distribution for contributors
- Staking and governance mechanisms
- Economic incentive alignment
"""

import asyncio
from datetime import UTC, datetime
from typing import Any

from ...tokenomics.fog_token_system import FogTokenSystem
from ..interfaces.base_service import BaseFogService, ServiceHealthCheck, ServiceStatus


class FogTokenomicsService(BaseFogService):
    """Service for managing fog computing token economics"""

    def __init__(self, service_name: str, config: dict[str, Any], event_bus):
        super().__init__(service_name, config, event_bus)

        # Core components
        self.token_system: FogTokenSystem | None = None

        # Tokenomics configuration
        self.token_config = config.get("tokens", {})

        # Service metrics
        self.metrics = {
            "total_accounts": 0,
            "total_supply": 0.0,
            "total_rewards_distributed": 0.0,
            "total_staked": 0.0,
            "governance_proposals": 0,
            "successful_transfers": 0,
            "failed_transfers": 0,
            "reward_distributions": 0,
        }

    async def initialize(self) -> bool:
        """Initialize the tokenomics service"""
        try:
            # Initialize token system
            self.token_system = FogTokenSystem(
                initial_supply=self.token_config.get("initial_supply", 1000000000),
                reward_rate_per_hour=self.token_config.get("reward_rate_per_hour", 10),
                staking_apy=self.token_config.get("staking_apy", 0.05),
                governance_threshold=self.token_config.get("governance_threshold", 1000000),
            )

            # Create system accounts
            system_key = b"system_key_placeholder"  # In production, use proper crypto
            await self.token_system.create_account("system", system_key, 0)
            await self.token_system.create_account(
                "treasury", system_key, self.token_config.get("initial_supply", 1000000000) * 0.1
            )

            self.metrics["total_accounts"] = 2  # system and treasury
            self.metrics["total_supply"] = self.token_config.get("initial_supply", 1000000000)

            # Subscribe to relevant events
            self.subscribe_to_events("device_registered", self._handle_device_registered)
            self.subscribe_to_events("task_completed", self._handle_task_completed)
            self.subscribe_to_events("token_transfer_request", self._handle_token_transfer)
            self.subscribe_to_events("reward_distribution", self._handle_reward_distribution)

            # Start background tasks
            self.add_background_task(self._reward_distribution_task(), "reward_distribution")
            self.add_background_task(self._staking_rewards_task(), "staking_rewards")
            self.add_background_task(self._governance_processing_task(), "governance")

            self.logger.info("Fog tokenomics service initialized")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize tokenomics service: {e}")
            return False

    async def cleanup(self) -> bool:
        """Cleanup tokenomics service resources"""
        try:
            # Process any pending rewards
            if self.token_system:
                # Implementation would handle pending reward distributions
                pass

            self.logger.info("Fog tokenomics service cleaned up")
            return True

        except Exception as e:
            self.logger.error(f"Error cleaning up tokenomics service: {e}")
            return False

    async def health_check(self) -> ServiceHealthCheck:
        """Perform health check on tokenomics service"""
        try:
            error_messages = []

            # Check token system
            if not self.token_system:
                error_messages.append("Token system not initialized")
            else:
                # Check system account balances
                self.token_system.accounts.get("system", {}).get("balance", 0)
                treasury_balance = self.token_system.accounts.get("treasury", {}).get("balance", 0)

                if treasury_balance <= 0:
                    error_messages.append("Treasury account is empty")

            # Check transfer failure rate
            total_transfers = self.metrics["successful_transfers"] + self.metrics["failed_transfers"]
            if total_transfers > 0:
                failure_rate = self.metrics["failed_transfers"] / total_transfers
                if failure_rate > 0.05:  # More than 5% failure rate
                    error_messages.append(f"High transfer failure rate: {failure_rate:.2%}")

            status = ServiceStatus.RUNNING if not error_messages else ServiceStatus.ERROR

            return ServiceHealthCheck(
                service_name=self.service_name,
                status=status,
                last_check=datetime.now(UTC),
                error_message="; ".join(error_messages) if error_messages else None,
                metrics=self.metrics.copy(),
            )

        except Exception as e:
            return ServiceHealthCheck(
                service_name=self.service_name,
                status=ServiceStatus.ERROR,
                last_check=datetime.now(UTC),
                error_message=f"Health check failed: {e}",
                metrics=self.metrics.copy(),
            )

    async def create_account(self, account_id: str, public_key: bytes, initial_balance: float = 0.0) -> bool:
        """Create a new token account"""
        try:
            if not self.token_system:
                return False

            success = await self.token_system.create_account(account_id, public_key, initial_balance)

            if success:
                self.metrics["total_accounts"] += 1

                # Publish account creation event
                await self.publish_event(
                    "account_created",
                    {
                        "account_id": account_id,
                        "initial_balance": initial_balance,
                        "timestamp": datetime.now(UTC).isoformat(),
                    },
                )

                self.logger.info(f"Created token account: {account_id}")

            return success

        except Exception as e:
            self.logger.error(f"Failed to create account {account_id}: {e}")
            return False

    async def transfer_tokens(self, from_account: str, to_account: str, amount: float, description: str = "") -> bool:
        """Transfer tokens between accounts"""
        try:
            if not self.token_system:
                return False

            success = await self.token_system.transfer(from_account, to_account, amount, description)

            if success:
                self.metrics["successful_transfers"] += 1

                # Publish transfer event
                await self.publish_event(
                    "token_transfer",
                    {
                        "from_account": from_account,
                        "to_account": to_account,
                        "amount": amount,
                        "description": description,
                        "timestamp": datetime.now(UTC).isoformat(),
                    },
                )
            else:
                self.metrics["failed_transfers"] += 1

            return success

        except Exception as e:
            self.logger.error(f"Failed to transfer tokens: {e}")
            self.metrics["failed_transfers"] += 1
            return False

    async def distribute_reward(self, account_id: str, amount: float, reason: str) -> bool:
        """Distribute reward tokens to an account"""
        try:
            # Transfer from treasury to recipient
            success = await self.transfer_tokens("treasury", account_id, amount, f"Reward: {reason}")

            if success:
                self.metrics["total_rewards_distributed"] += amount
                self.metrics["reward_distributions"] += 1

                # Publish reward distribution event
                await self.publish_event(
                    "reward_distributed",
                    {
                        "account_id": account_id,
                        "amount": amount,
                        "reason": reason,
                        "timestamp": datetime.now(UTC).isoformat(),
                    },
                )

                self.logger.info(f"Distributed {amount} tokens to {account_id}: {reason}")

            return success

        except Exception as e:
            self.logger.error(f"Failed to distribute reward: {e}")
            return False

    async def stake_tokens(self, account_id: str, amount: float) -> bool:
        """Stake tokens for governance and rewards"""
        try:
            if not self.token_system:
                return False

            success = await self.token_system.stake_tokens(account_id, amount)

            if success:
                self.metrics["total_staked"] += amount

                # Publish staking event
                await self.publish_event(
                    "tokens_staked",
                    {"account_id": account_id, "amount": amount, "timestamp": datetime.now(UTC).isoformat()},
                )

            return success

        except Exception as e:
            self.logger.error(f"Failed to stake tokens: {e}")
            return False

    async def get_account_balance(self, account_id: str) -> float:
        """Get account balance"""
        try:
            if not self.token_system or account_id not in self.token_system.accounts:
                return 0.0

            return self.token_system.accounts[account_id].get("balance", 0.0)

        except Exception as e:
            self.logger.error(f"Failed to get account balance: {e}")
            return 0.0

    async def get_tokenomics_stats(self) -> dict[str, Any]:
        """Get comprehensive tokenomics statistics"""
        try:
            stats = self.metrics.copy()

            if self.token_system:
                token_stats = self.token_system.get_network_stats()
                stats.update(
                    {
                        "token_system_stats": token_stats,
                        "treasury_balance": self.token_system.accounts.get("treasury", {}).get("balance", 0),
                        "system_balance": self.token_system.accounts.get("system", {}).get("balance", 0),
                    }
                )

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get tokenomics stats: {e}")
            return self.metrics.copy()

    async def _handle_device_registered(self, event):
        """Handle device registration - create token account"""
        device_id = event.data.get("device_id")
        if device_id:
            # Create account for new device
            device_key = f"device_{device_id}".encode()
            await self.create_account(device_id, device_key, 0.0)

    async def _handle_task_completed(self, event):
        """Handle task completion - distribute rewards"""
        device_id = event.data.get("device_id")
        compute_time = event.data.get("compute_time_hours", 0.0)
        success = event.data.get("success", False)

        if device_id and success and compute_time > 0:
            # Calculate reward based on compute time
            reward_amount = compute_time * self.token_config.get("reward_rate_per_hour", 10)
            await self.distribute_reward(device_id, reward_amount, "Task completion")

    async def _handle_token_transfer(self, event):
        """Handle token transfer requests"""
        from_account = event.data.get("from_account")
        to_account = event.data.get("to_account")
        amount = event.data.get("amount")
        description = event.data.get("description", "")

        success = await self.transfer_tokens(from_account, to_account, amount, description)

        # Publish response
        await self.publish_event(
            "token_transfer_response", {"request_id": event.data.get("request_id"), "success": success}
        )

    async def _handle_reward_distribution(self, event):
        """Handle reward distribution requests"""
        account_id = event.data.get("account_id")
        amount = event.data.get("amount")
        reason = event.data.get("reason", "Manual distribution")

        success = await self.distribute_reward(account_id, amount, reason)

        await self.publish_event(
            "reward_distribution_response", {"request_id": event.data.get("request_id"), "success": success}
        )

    async def _reward_distribution_task(self):
        """Background task to process pending reward distributions"""
        while not self._shutdown_event.is_set():
            try:
                # This would implement batched reward processing
                # For now, just update metrics
                if self.token_system:
                    stats = self.token_system.get_network_stats()
                    self.metrics["total_rewards_distributed"] = stats.get("total_rewards_distributed", 0)

                await asyncio.sleep(3600)  # Process every hour

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Reward distribution task error: {e}")
                await asyncio.sleep(600)

    async def _staking_rewards_task(self):
        """Background task to distribute staking rewards"""
        while not self._shutdown_event.is_set():
            try:
                if self.token_system:
                    # Calculate and distribute staking rewards
                    staking_apy = self.token_config.get("staking_apy", 0.05)

                    # Get all accounts with staked tokens
                    for account_id, account_data in self.token_system.accounts.items():
                        staked_amount = account_data.get("staked", 0.0)
                        if staked_amount > 0:
                            # Calculate hourly staking reward
                            hourly_reward = staked_amount * (staking_apy / 8760)  # 8760 hours in year

                            if hourly_reward > 0.001:  # Minimum reward threshold
                                await self.distribute_reward(account_id, hourly_reward, "Staking reward")

                await asyncio.sleep(3600)  # Distribute every hour

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Staking rewards task error: {e}")
                await asyncio.sleep(1800)

    async def _governance_processing_task(self):
        """Background task to process governance proposals"""
        while not self._shutdown_event.is_set():
            try:
                # This would implement governance proposal processing
                # For now, just track metrics
                if self.token_system:
                    # Count eligible governance participants
                    threshold = self.token_config.get("governance_threshold", 1000000)
                    eligible_voters = 0

                    for account_data in self.token_system.accounts.values():
                        staked = account_data.get("staked", 0.0)
                        if staked >= threshold:
                            eligible_voters += 1

                    # Update governance metrics
                    self.metrics["eligible_governance_voters"] = eligible_voters

                await asyncio.sleep(86400)  # Process daily

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Governance processing task error: {e}")
                await asyncio.sleep(3600)
