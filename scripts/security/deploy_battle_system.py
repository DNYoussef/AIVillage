#!/usr/bin/env python3
"""
Deploy Sword/Shield Security Battle System

Initializes and deploys the daily mock security battle system:
- Creates and configures Sword and Shield agents
- Sets up Battle Orchestrator with daily scheduling
- Configures secure sandbox environment
- Starts automated daily battle cycle
- Sets up monitoring and reporting to King Agent

Usage:
    python deploy_battle_system.py [--battle-time HH:MM] [--dry-run]
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, time

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from software.meta_agents import BattleOrchestrator, ShieldAgent, SwordAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("battle_system.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class BattleSystemDeployer:
    """
    Deploys and manages the Sword/Shield security battle system.
    """

    def __init__(self, battle_time: time = time(2, 0), dry_run: bool = False):
        self.battle_time = battle_time
        self.dry_run = dry_run

        # Agent instances
        self.sword_agent: SwordAgent = None
        self.shield_agent: ShieldAgent = None
        self.battle_orchestrator: BattleOrchestrator = None

        logger.info(f"Battle System Deployer initialized - Daily battles at {battle_time}")
        if dry_run:
            logger.info("DRY RUN MODE: No actual deployment will occur")

    async def deploy_battle_system(self) -> bool:
        """
        Deploy the complete battle system.

        Returns:
            bool: True if deployment successful, False otherwise
        """
        try:
            logger.info("🚀 Starting Sword/Shield Security Battle System deployment...")

            # Step 1: Initialize agents
            if not await self._initialize_agents():
                return False

            # Step 2: Configure sandbox environment
            if not await self._configure_sandbox():
                return False

            # Step 3: Validate agent capabilities
            if not await self._validate_agent_capabilities():
                return False

            # Step 4: Test battle orchestration
            if not await self._test_battle_orchestration():
                return False

            # Step 5: Start daily battle scheduling (unless dry run)
            if not self.dry_run:
                if not await self._start_battle_scheduling():
                    return False
            else:
                logger.info("DRY RUN: Skipping battle scheduling startup")

            logger.info("✅ Sword/Shield Security Battle System deployment completed successfully!")
            return True

        except Exception as e:
            logger.error(f"❌ Battle system deployment failed: {e}")
            return False

    async def _initialize_agents(self) -> bool:
        """Initialize Sword, Shield, and Battle Orchestrator agents."""
        try:
            logger.info("Initializing security agents...")

            # Initialize Sword Agent (offensive security specialist)
            self.sword_agent = SwordAgent("production_sword")
            logger.info(f"✓ Sword Agent initialized: {self.sword_agent.agent_id}")

            # Initialize Shield Agent (defensive security specialist)
            self.shield_agent = ShieldAgent("production_shield")
            logger.info(f"✓ Shield Agent initialized: {self.shield_agent.agent_id}")

            # Initialize Battle Orchestrator
            self.battle_orchestrator = BattleOrchestrator(
                "production_battle_orchestrator",
                battle_time=self.battle_time,
                king_agent_id="king",
            )
            logger.info(f"✓ Battle Orchestrator initialized: {self.battle_orchestrator.agent_id}")

            # Register agents with orchestrator
            await self.battle_orchestrator.initialize_agents()
            logger.info("✓ Agents registered with battle orchestrator")

            return True

        except Exception as e:
            logger.error(f"Agent initialization failed: {e}")
            return False

    async def _configure_sandbox(self) -> bool:
        """Configure secure sandbox environment for battles."""
        try:
            logger.info("Configuring secure sandbox environment...")

            # Validate sandbox configuration
            sandbox_config = self.battle_orchestrator.sandbox_config

            required_config = [
                "isolated_network",
                "virtual_machines",
                "simulated_services",
                "monitoring_enabled",
            ]

            for key in required_config:
                if key not in sandbox_config:
                    logger.error(f"Missing sandbox configuration: {key}")
                    return False

            logger.info(f"✓ Sandbox configured with {sandbox_config['virtual_machines']} VMs")
            logger.info(f"✓ Simulated services: {', '.join(sandbox_config['simulated_services'])}")
            logger.info(f"✓ Network isolation: {sandbox_config['isolated_network']}")
            logger.info(f"✓ Monitoring enabled: {sandbox_config['monitoring_enabled']}")

            return True

        except Exception as e:
            logger.error(f"Sandbox configuration failed: {e}")
            return False

    async def _validate_agent_capabilities(self) -> bool:
        """Validate that agents have required capabilities."""
        try:
            logger.info("Validating agent capabilities...")

            # Test Sword Agent capabilities
            sword_status = self.sword_agent.get_offensive_status()
            if sword_status["status"] != "active":
                logger.error("Sword Agent not in active status")
                return False

            if sword_status["attack_techniques_count"] == 0:
                logger.error("Sword Agent has no attack techniques loaded")
                return False

            logger.info(f"✓ Sword Agent: {sword_status['attack_techniques_count']} attack techniques ready")

            # Test Shield Agent capabilities
            shield_status = self.shield_agent.get_defensive_status()
            if shield_status["status"] != "active":
                logger.error("Shield Agent not in active status")
                return False

            if shield_status["defensive_patterns_count"] == 0:
                logger.error("Shield Agent has no defensive patterns loaded")
                return False

            logger.info(f"✓ Shield Agent: {shield_status['defensive_patterns_count']} defensive patterns ready")

            # Test Battle Orchestrator capabilities
            orchestrator_status = self.battle_orchestrator.get_orchestrator_status()
            if orchestrator_status["status"] != "active":
                logger.error("Battle Orchestrator not in active status")
                return False

            if orchestrator_status["available_scenarios"] == 0:
                logger.error("Battle Orchestrator has no battle scenarios")
                return False

            logger.info(f"✓ Battle Orchestrator: {orchestrator_status['available_scenarios']} scenarios available")

            return True

        except Exception as e:
            logger.error(f"Agent capability validation failed: {e}")
            return False

    async def _test_battle_orchestration(self) -> bool:
        """Test battle orchestration without full battle execution."""
        try:
            logger.info("Testing battle orchestration capabilities...")

            # Test scenario selection
            scenario = self.battle_orchestrator._select_battle_scenario()
            logger.info(f"✓ Scenario selection working: {scenario.name} ({scenario.difficulty_level})")

            # Test sandbox setup
            sandbox_status = await self.battle_orchestrator._setup_sandbox_environment(scenario)
            if not sandbox_status["isolation_confirmed"]:
                logger.error("Sandbox isolation test failed")
                return False
            logger.info("✓ Sandbox environment test passed")

            # Test agent preparation (mock mode)
            sword_prep = await self.sword_agent.prepare_battle_attack(
                {
                    "scenario": scenario.__dict__,
                    "target_systems": scenario.target_systems,
                    "test_mode": True,
                }
            )
            if "strategy_id" not in sword_prep:
                logger.error("Sword battle preparation test failed")
                return False
            logger.info("✓ Sword preparation test passed")

            shield_prep = await self.shield_agent.prepare_battle_defense(
                {
                    "scenario": scenario.__dict__,
                    "expected_attacks": scenario.attack_vectors,
                    "test_mode": True,
                }
            )
            if "strategy_id" not in shield_prep:
                logger.error("Shield battle preparation test failed")
                return False
            logger.info("✓ Shield preparation test passed")

            logger.info("✓ Battle orchestration test completed successfully")
            return True

        except Exception as e:
            logger.error(f"Battle orchestration test failed: {e}")
            return False

    async def _start_battle_scheduling(self) -> bool:
        """Start the automated daily battle scheduling."""
        try:
            logger.info("Starting automated daily battle scheduling...")

            # Create background task for battle scheduling
            battle_task = asyncio.create_task(self.battle_orchestrator.schedule_daily_battles())

            # Wait a moment to ensure task started successfully
            await asyncio.sleep(1)

            if battle_task.done() and battle_task.exception():
                logger.error(f"Battle scheduling task failed immediately: {battle_task.exception()}")
                return False

            logger.info(f"✓ Daily battle scheduling active - Next battle at {self.battle_time}")
            logger.info("Battle system is now running in background...")

            # Keep the system running (in production, this would be managed by a service manager)
            try:
                await battle_task  # This will run indefinitely
            except KeyboardInterrupt:
                logger.info("Battle system shutdown requested")
                battle_task.cancel()
                try:
                    await battle_task
                except asyncio.CancelledError:
                    pass
                logger.info("Battle system shutdown complete")

            return True

        except Exception as e:
            logger.error(f"Battle scheduling startup failed: {e}")
            return False

    def get_deployment_status(self) -> dict:
        """Get current deployment status."""
        status = {
            "deployment_time": datetime.now().isoformat(),
            "battle_time_configured": self.battle_time.strftime("%H:%M"),
            "dry_run_mode": self.dry_run,
            "agents_initialized": {
                "sword": self.sword_agent is not None,
                "shield": self.shield_agent is not None,
                "orchestrator": self.battle_orchestrator is not None,
            },
        }

        if self.battle_orchestrator:
            orchestrator_status = self.battle_orchestrator.get_orchestrator_status()
            status["orchestrator_status"] = orchestrator_status

        return status


async def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(
        description="Deploy Sword/Shield Security Battle System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python deploy_battle_system.py                    # Deploy with default 2:00 AM battles
  python deploy_battle_system.py --battle-time 03:30 # Deploy with 3:30 AM battles
  python deploy_battle_system.py --dry-run          # Test deployment without starting battles
        """,
    )

    parser.add_argument(
        "--battle-time",
        type=str,
        default="02:00",
        help="Daily battle time in HH:MM format (default: 02:00)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test deployment without starting battle scheduling",
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse battle time
    try:
        hour, minute = map(int, args.battle_time.split(":"))
        battle_time = time(hour, minute)
    except ValueError:
        logger.error(f"Invalid battle time format: {args.battle_time}. Use HH:MM format.")
        return 1

    # Create and run deployer
    deployer = BattleSystemDeployer(battle_time=battle_time, dry_run=args.dry_run)

    success = await deployer.deploy_battle_system()

    if success:
        logger.info("📊 Final deployment status:")
        status = deployer.get_deployment_status()
        for key, value in status.items():
            logger.info(f"  {key}: {value}")

        if not args.dry_run:
            logger.info("🛡️⚔️ Sword/Shield Security Battle System is now active!")
            logger.info(f"Daily battles will occur at {args.battle_time}")
            logger.info("System is running... Press Ctrl+C to shutdown")

        return 0
    else:
        logger.error("❌ Deployment failed - see errors above")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Deployment interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
