"""
SLO Recovery Router - Integration Adapter
Integration with Flake Detector and GitHub Orchestrator data feeds
Real-time data integration and routing coordination
"""

from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Any

import aiohttp

from .slo_recovery_router import RoutingDecision, SLORecoveryRouter


@dataclass
class FlakeDetectorData:
    """Data structure for Flake Detector integration"""

    detection_id: str
    flake_patterns: list[dict]
    failure_probability: float
    affected_tests: list[str]
    root_cause_analysis: dict
    confidence_score: float
    timestamp: datetime


@dataclass
class GitHubOrchestratorData:
    """Data structure for GitHub Orchestrator integration"""

    orchestration_id: str
    repository: str
    workflow_failures: list[dict]
    deployment_status: str
    branch: str
    commit_sha: str
    failure_context: dict
    timestamp: datetime


class DataFeedAdapter(ABC):
    """Abstract base class for data feed adapters"""

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the data feed"""
        pass

    @abstractmethod
    async def poll_data(self) -> Any | None:
        """Poll for new data"""
        pass

    @abstractmethod
    def transform_to_failure_data(self, raw_data: Any) -> dict:
        """Transform raw data to failure data format"""
        pass


class FlakeDetectorAdapter(DataFeedAdapter):
    """Adapter for Flake Detector data feed integration"""

    def __init__(self, api_endpoint: str, api_key: str):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.session = None
        self.logger = logging.getLogger(__name__)
        self.last_poll_time = datetime.now()

    async def connect(self) -> bool:
        """Connect to Flake Detector API"""
        try:
            self.session = aiohttp.ClientSession(headers={"Authorization": f"Bearer {self.api_key}"})

            # Test connection
            async with self.session.get(f"{self.api_endpoint}/health") as response:
                if response.status == 200:
                    self.logger.info("Connected to Flake Detector successfully")
                    return True
                else:
                    self.logger.error(f"Flake Detector connection failed: {response.status}")
                    return False

        except Exception as e:
            self.logger.error(f"Failed to connect to Flake Detector: {e}")
            return False

    async def poll_data(self) -> FlakeDetectorData | None:
        """Poll for new flake detection data"""
        if not self.session:
            await self.connect()

        try:
            # Poll for new detections since last poll
            params = {"since": self.last_poll_time.isoformat(), "status": "active", "severity": "high,critical"}

            async with self.session.get(f"{self.api_endpoint}/detections", params=params) as response:

                if response.status == 200:
                    data = await response.json()
                    self.last_poll_time = datetime.now()

                    if data.get("detections"):
                        # Get the most recent critical detection
                        latest = data["detections"][0]
                        return FlakeDetectorData(
                            detection_id=latest["id"],
                            flake_patterns=latest.get("patterns", []),
                            failure_probability=latest.get("failure_probability", 0.0),
                            affected_tests=latest.get("affected_tests", []),
                            root_cause_analysis=latest.get("root_cause", {}),
                            confidence_score=latest.get("confidence", 0.0),
                            timestamp=datetime.fromisoformat(latest["timestamp"]),
                        )
                else:
                    self.logger.warning(f"Flake Detector poll failed: {response.status}")

        except Exception as e:
            self.logger.error(f"Error polling Flake Detector: {e}")

        return None

    def transform_to_failure_data(self, flake_data: FlakeDetectorData) -> dict:
        """Transform flake detection data to failure data format"""

        error_message = f"Flake detected in tests: {', '.join(flake_data.affected_tests[:3])}"
        if len(flake_data.affected_tests) > 3:
            error_message += f" and {len(flake_data.affected_tests) - 3} more"

        # Extract logs from patterns
        logs = []
        for pattern in flake_data.flake_patterns:
            logs.append(f"Pattern: {pattern.get('description', '')}")
            logs.append(f"Frequency: {pattern.get('frequency', 0)}")

        # Build context
        context = {
            "detection_id": flake_data.detection_id,
            "source": "flake_detector",
            "affected_tests": flake_data.affected_tests,
            "failure_probability": flake_data.failure_probability,
            "root_cause": flake_data.root_cause_analysis,
            "patterns": flake_data.flake_patterns,
        }

        return {
            "error_message": error_message,
            "logs": logs,
            "context": context,
            "timestamp": flake_data.timestamp.isoformat(),
            "source": "flake_detector",
            "confidence": flake_data.confidence_score,
        }


class GitHubOrchestratorAdapter(DataFeedAdapter):
    """Adapter for GitHub Orchestrator data feed integration"""

    def __init__(self, webhook_url: str, api_token: str):
        self.webhook_url = webhook_url
        self.api_token = api_token
        self.session = None
        self.logger = logging.getLogger(__name__)
        self.last_poll_time = datetime.now()

    async def connect(self) -> bool:
        """Connect to GitHub Orchestrator"""
        try:
            self.session = aiohttp.ClientSession(headers={"Authorization": f"token {self.api_token}"})

            # Test connection
            async with self.session.get(f"{self.webhook_url}/status") as response:
                if response.status == 200:
                    self.logger.info("Connected to GitHub Orchestrator successfully")
                    return True
                else:
                    self.logger.error(f"GitHub Orchestrator connection failed: {response.status}")
                    return False

        except Exception as e:
            self.logger.error(f"Failed to connect to GitHub Orchestrator: {e}")
            return False

    async def poll_data(self) -> GitHubOrchestratorData | None:
        """Poll for GitHub workflow failures"""
        if not self.session:
            await self.connect()

        try:
            # Poll for workflow failures
            params = {"since": self.last_poll_time.isoformat(), "status": "failure", "conclusion": "failure"}

            async with self.session.get(f"{self.webhook_url}/workflow-runs", params=params) as response:

                if response.status == 200:
                    data = await response.json()
                    self.last_poll_time = datetime.now()

                    if data.get("workflow_runs"):
                        # Get the most recent failure
                        latest = data["workflow_runs"][0]
                        return GitHubOrchestratorData(
                            orchestration_id=latest["id"],
                            repository=latest.get("repository", {}).get("full_name", ""),
                            workflow_failures=latest.get("failures", []),
                            deployment_status=latest.get("conclusion", ""),
                            branch=latest.get("head_branch", ""),
                            commit_sha=latest.get("head_sha", ""),
                            failure_context=latest.get("context", {}),
                            timestamp=datetime.fromisoformat(latest["created_at"]),
                        )
                else:
                    self.logger.warning(f"GitHub Orchestrator poll failed: {response.status}")

        except Exception as e:
            self.logger.error(f"Error polling GitHub Orchestrator: {e}")

        return None

    def transform_to_failure_data(self, github_data: GitHubOrchestratorData) -> dict:
        """Transform GitHub orchestrator data to failure data format"""

        error_message = f"GitHub workflow failure in {github_data.repository}"
        if github_data.branch:
            error_message += f" on branch {github_data.branch}"

        # Extract logs from workflow failures
        logs = []
        for failure in github_data.workflow_failures:
            logs.append(f"Job: {failure.get('job_name', 'unknown')}")
            logs.append(f"Step: {failure.get('step_name', 'unknown')}")
            logs.append(f"Error: {failure.get('error_message', '')}")

        # Build context
        context = {
            "orchestration_id": github_data.orchestration_id,
            "source": "github_orchestrator",
            "repository": github_data.repository,
            "branch": github_data.branch,
            "commit_sha": github_data.commit_sha,
            "deployment_status": github_data.deployment_status,
            "workflow_failures": github_data.workflow_failures,
            "failure_context": github_data.failure_context,
        }

        return {
            "error_message": error_message,
            "logs": logs,
            "context": context,
            "timestamp": github_data.timestamp.isoformat(),
            "source": "github_orchestrator",
            "confidence": 0.8,  # Default confidence for GitHub failures
        }


class IntegrationCoordinator:
    """Coordinates data feed integration with SLO Recovery Router"""

    def __init__(self, slo_router: SLORecoveryRouter):
        self.slo_router = slo_router
        self.adapters = {}
        self.logger = logging.getLogger(__name__)
        self.polling_interval = 30  # seconds
        self.running = False

    def add_adapter(self, name: str, adapter: DataFeedAdapter):
        """Add a data feed adapter"""
        self.adapters[name] = adapter
        self.logger.info(f"Added adapter: {name}")

    async def start_integration(self):
        """Start the integration polling loop"""
        self.logger.info("Starting integration coordinator")
        self.running = True

        # Connect all adapters
        for name, adapter in self.adapters.items():
            success = await adapter.connect()
            if not success:
                self.logger.error(f"Failed to connect adapter: {name}")

        # Start polling loop
        await self._polling_loop()

    async def stop_integration(self):
        """Stop the integration"""
        self.logger.info("Stopping integration coordinator")
        self.running = False

    async def _polling_loop(self):
        """Main polling loop for data feeds"""
        while self.running:
            try:
                # Poll all adapters concurrently
                tasks = []
                for name, adapter in self.adapters.items():
                    tasks.append(self._poll_and_route(name, adapter))

                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

                # Wait before next poll
                await asyncio.sleep(self.polling_interval)

            except Exception as e:
                self.logger.error(f"Error in polling loop: {e}")
                await asyncio.sleep(5)  # Brief pause before retry

    async def _poll_and_route(self, adapter_name: str, adapter: DataFeedAdapter):
        """Poll adapter and route data through SLO router"""
        try:
            raw_data = await adapter.poll_data()
            if raw_data:
                self.logger.info(f"Received data from {adapter_name}")

                # Transform to failure data format
                failure_data = adapter.transform_to_failure_data(raw_data)

                # Route through SLO Recovery Router
                routing_decision = await self.slo_router.route_to_remedies(failure_data)

                # Log routing decision
                self.logger.info(
                    f"Routing decision for {adapter_name}: "
                    f"{routing_decision.strategy_selection.selected_strategy.name} "
                    f"(confidence: {routing_decision.routing_confidence:.3f})"
                )

                # Execute coordination plan if high confidence
                if routing_decision.routing_confidence > 0.75:
                    await self._execute_coordination_plan(routing_decision)
                else:
                    self.logger.warning(
                        f"Low routing confidence ({routing_decision.routing_confidence:.3f}) "
                        f"for {adapter_name} - escalating for human review"
                    )

        except Exception as e:
            self.logger.error(f"Error processing {adapter_name}: {e}")

    async def _execute_coordination_plan(self, routing_decision: RoutingDecision):
        """Execute the coordination plan from routing decision"""
        try:
            coordination_plan = routing_decision.coordination_plan

            # Execute coordination plan
            execution_results = self.slo_router.parallel_coordinator.execute_coordination_plan(coordination_plan)

            # Log execution results
            success_rate = (
                sum(1 for result in execution_results["agent_results"].values() if result.get("success", False))
                / len(execution_results["agent_results"])
                if execution_results["agent_results"]
                else 0
            )

            self.logger.info(
                f"Coordination plan executed: {coordination_plan.plan_id} " f"- Success rate: {success_rate:.2%}"
            )

            # Handle escalations if any
            if routing_decision.escalation_events:
                self.logger.warning(f"Escalation events triggered: {len(routing_decision.escalation_events)}")

        except Exception as e:
            self.logger.error(f"Error executing coordination plan: {e}")

    def get_integration_status(self) -> dict:
        """Get status of all integrations"""
        status = {
            "running": self.running,
            "polling_interval": self.polling_interval,
            "adapters": {},
            "last_update": datetime.now().isoformat(),
        }

        for name, adapter in self.adapters.items():
            status["adapters"][name] = {
                "connected": hasattr(adapter, "session") and adapter.session is not None,
                "last_poll": getattr(adapter, "last_poll_time", None),
                "adapter_type": type(adapter).__name__,
            }

        return status


# Factory function for creating configured integration coordinator
def create_integration_coordinator(
    slo_router: SLORecoveryRouter,
    flake_detector_config: dict | None = None,
    github_orchestrator_config: dict | None = None,
) -> IntegrationCoordinator:
    """Factory function to create configured integration coordinator"""

    coordinator = IntegrationCoordinator(slo_router)

    # Add Flake Detector adapter if configured
    if flake_detector_config:
        flake_adapter = FlakeDetectorAdapter(
            api_endpoint=flake_detector_config["api_endpoint"], api_key=flake_detector_config["api_key"]
        )
        coordinator.add_adapter("flake_detector", flake_adapter)

    # Add GitHub Orchestrator adapter if configured
    if github_orchestrator_config:
        github_adapter = GitHubOrchestratorAdapter(
            webhook_url=github_orchestrator_config["webhook_url"], api_token=github_orchestrator_config["api_token"]
        )
        coordinator.add_adapter("github_orchestrator", github_adapter)

    return coordinator


# Export for use by other components
__all__ = [
    "IntegrationCoordinator",
    "FlakeDetectorAdapter",
    "GitHubOrchestratorAdapter",
    "create_integration_coordinator",
]
