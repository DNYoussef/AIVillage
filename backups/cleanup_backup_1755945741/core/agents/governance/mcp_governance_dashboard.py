"""
Unified MCP Governance Dashboard

Central governance interface for the complete AIVillage digital twin and meta-agent ecosystem.
Provides unified Model Control Protocol (MCP) tools for managing:

1. Digital Twin Concierge systems (on-device personal AI)
2. Meta-agent sharding across fog compute network
3. Distributed RAG system governance (Sage/Curator/King voting)
4. P2P network coordination and resource management
5. Privacy-preserving data flows and compliance monitoring

This dashboard integrates all the systems we've built:
- packages/edge/mobile/digital_twin_concierge.py
- packages/agents/distributed/meta_agent_sharding_coordinator.py
- packages/rag/distributed/distributed_rag_coordinator.py
- packages/p2p/core/transport_manager.py
- packages/edge/fog_compute/fog_coordinator.py

Architecture:
- MCP Server providing unified tools for all AI agents
- Real-time monitoring and control of distributed systems
- Privacy audit trails and compliance reporting
- Resource optimization across edge-to-fog spectrum
- Democratic governance with agent voting systems
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from ...agents.distributed.meta_agent_sharding_coordinator import MetaAgentShardingCoordinator
from ...edge.fog_compute.fog_coordinator import FogCoordinator
from ...edge.mobile.digital_twin_concierge import DigitalTwinConcierge
from ...p2p.core.transport_manager import UnifiedTransportManager
from ...rag.distributed.distributed_rag_coordinator import DistributedRAGCoordinator, RAGGovernanceMCP

logger = logging.getLogger(__name__)


class SystemComponent(Enum):
    """Major system components under governance"""

    DIGITAL_TWINS = "digital_twins"
    META_AGENTS = "meta_agents"
    DISTRIBUTED_RAG = "distributed_rag"
    P2P_NETWORK = "p2p_network"
    FOG_COMPUTE = "fog_compute"
    MOBILE_OPTIMIZATION = "mobile_optimization"


class GovernanceLevel(Enum):
    """Levels of governance access"""

    READ_ONLY = "read_only"  # View status and metrics
    OPERATOR = "operator"  # Start/stop services, adjust configs
    COORDINATOR = "coordinator"  # Resource allocation, deployment decisions
    GOVERNANCE = "governance"  # Voting on major changes
    EMERGENCY = "emergency"  # Override capabilities (King agent only)


@dataclass
class SystemStatus:
    """Comprehensive system status across all components"""

    component: SystemComponent
    status: str  # operational, degraded, offline, maintenance
    health_score: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

    # Component-specific metrics
    metrics: dict[str, Any] = field(default_factory=dict)
    alerts: list[str] = field(default_factory=list)
    resource_usage: dict[str, float] = field(default_factory=dict)

    # Performance indicators
    latency_ms: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0


@dataclass
class GovernanceAction:
    """Governance action for audit trail"""

    action_id: str = field(default_factory=lambda: str(uuid4()))
    agent_id: str = ""
    action_type: str = ""
    target_system: SystemComponent = SystemComponent.DIGITAL_TWINS
    description: str = ""

    # Authorization
    governance_level: GovernanceLevel = GovernanceLevel.READ_ONLY
    requires_approval: bool = False
    approved_by: set[str] = field(default_factory=set)

    # Execution
    status: str = "pending"  # pending, approved, executing, completed, failed
    timestamp: datetime = field(default_factory=datetime.now)
    result: dict[str, Any] = field(default_factory=dict)


class UnifiedMCPGovernanceDashboard:
    """
    Unified MCP Governance Dashboard

    Central command center for the entire AIVillage ecosystem providing:
    - Unified monitoring of all system components
    - Democratic governance with agent voting
    - Resource optimization and allocation
    - Privacy compliance and audit trails
    - Emergency response and system recovery
    """

    def __init__(
        self,
        digital_twin_concierge: DigitalTwinConcierge | None = None,
        meta_agent_coordinator: MetaAgentShardingCoordinator | None = None,
        distributed_rag: DistributedRAGCoordinator | None = None,
        transport_manager: UnifiedTransportManager | None = None,
        fog_coordinator: FogCoordinator | None = None,
    ):
        # Core system components
        self.digital_twin_concierge = digital_twin_concierge
        self.meta_agent_coordinator = meta_agent_coordinator
        self.distributed_rag = distributed_rag
        self.transport_manager = transport_manager
        self.fog_coordinator = fog_coordinator

        # Governance systems
        self.rag_governance = RAGGovernanceMCP(distributed_rag) if distributed_rag else None
        self.authorized_agents = {"sage", "curator", "king"}  # Voting agents

        # System monitoring
        self.system_status: dict[SystemComponent, SystemStatus] = {}
        self.governance_actions: dict[str, GovernanceAction] = {}
        self.system_alerts: list[dict[str, Any]] = []

        # Performance tracking
        self.metrics = {
            "total_digital_twins_active": 0,
            "total_meta_agents_deployed": 0,
            "distributed_rag_queries_per_hour": 0.0,
            "p2p_network_nodes": 0,
            "fog_compute_utilization": 0.0,
            "privacy_violations": 0,
            "governance_proposals_active": 0,
            "system_uptime_hours": 0.0,
        }

        # Privacy and compliance
        self.privacy_audit_trail: list[dict[str, Any]] = []
        self.compliance_status = {
            "data_protection": True,
            "local_data_only": True,
            "differential_privacy": True,
            "consent_management": True,
            "data_retention_policy": True,
        }

        logger.info("Unified MCP Governance Dashboard initialized")

    async def initialize_dashboard(self) -> bool:
        """Initialize the unified governance dashboard"""

        try:
            # Initialize monitoring for each component
            await self._initialize_component_monitoring()

            # Set up periodic system health checks
            asyncio.create_task(self._periodic_health_checks())

            # Start privacy compliance monitoring
            asyncio.create_task(self._privacy_compliance_monitor())

            # Initialize emergency response system
            await self._initialize_emergency_response()

            logger.info("🎛️ Unified MCP Governance Dashboard operational")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize governance dashboard: {e}")
            return False

    # === MCP GOVERNANCE TOOLS ===

    async def mcp_get_system_overview(self, agent_id: str) -> dict[str, Any]:
        """MCP Tool: Get comprehensive system overview"""

        if not await self._authorize_action(agent_id, GovernanceLevel.READ_ONLY):
            return {"error": "Insufficient permissions"}

        try:
            overview = {
                "system_health": await self._calculate_overall_system_health(),
                "component_status": {comp.value: status.__dict__ for comp, status in self.system_status.items()},
                "key_metrics": self.metrics.copy(),
                "active_alerts": len(self.system_alerts),
                "governance_status": {
                    "active_proposals": len([a for a in self.governance_actions.values() if a.status == "pending"]),
                    "authorized_agents": list(self.authorized_agents),
                },
                "privacy_compliance": self.compliance_status.copy(),
                "timestamp": datetime.now().isoformat(),
            }

            await self._log_governance_action(agent_id, "view_system_overview", "system_overview")
            return {"success": True, "overview": overview}

        except Exception as e:
            logger.error(f"Error getting system overview: {e}")
            return {"error": str(e)}

    async def mcp_manage_digital_twins(
        self, agent_id: str, action: str, device_ids: list[str] = None, config: dict[str, Any] = None
    ) -> dict[str, Any]:
        """MCP Tool: Manage digital twin concierge systems"""

        if not await self._authorize_action(agent_id, GovernanceLevel.OPERATOR):
            return {"error": "Insufficient permissions - requires Operator level"}

        if not self.digital_twin_concierge:
            return {"error": "Digital twin concierge not available"}

        try:
            results = {}

            if action == "get_status":
                privacy_report = self.digital_twin_concierge.get_privacy_report()
                learning_metrics = self.digital_twin_concierge.get_learning_metrics()

                results = {
                    "privacy_status": privacy_report,
                    "learning_metrics": learning_metrics,
                    "surprise_threshold": self.digital_twin_concierge.surprise_threshold,
                    "data_sources_active": len(self.digital_twin_concierge.data_sources),
                }

            elif action == "update_preferences":
                if config:
                    success = await self.digital_twin_concierge.update_user_preferences(config)
                    results = {"preferences_updated": success, "config": config}
                else:
                    return {"error": "Config required for preference updates"}

            elif action == "trigger_learning_evaluation":
                evaluation = await self.digital_twin_concierge.evaluate_surprise_learning()
                results = {"evaluation": evaluation}

            elif action == "privacy_audit":
                audit = await self.digital_twin_concierge.conduct_privacy_audit()
                results = {"privacy_audit": audit}

            else:
                return {"error": f"Unknown action: {action}"}

            await self._log_governance_action(agent_id, f"digital_twin_{action}", "digital_twins", results)
            return {"success": True, "results": results}

        except Exception as e:
            logger.error(f"Error managing digital twins: {e}")
            return {"error": str(e)}

    async def mcp_coordinate_meta_agents(
        self, agent_id: str, action: str, target_agents: list[str] = None, deployment_config: dict[str, Any] = None
    ) -> dict[str, Any]:
        """MCP Tool: Coordinate meta-agent deployment and sharding"""

        if not await self._authorize_action(agent_id, GovernanceLevel.COORDINATOR):
            return {"error": "Insufficient permissions - requires Coordinator level"}

        if not self.meta_agent_coordinator:
            return {"error": "Meta-agent coordinator not available"}

        try:
            results = {}

            if action == "get_deployment_status":
                status = await self.meta_agent_coordinator.get_deployment_status()
                results = status

            elif action == "create_deployment_plan":
                plan = await self.meta_agent_coordinator.create_deployment_plan(
                    target_agents=target_agents,
                    force_local=deployment_config.get("force_local", False) if deployment_config else False,
                )
                results = {
                    "plan_id": plan.plan_id,
                    "local_agents": len(plan.local_agents),
                    "fog_agents": len(plan.fog_agents),
                    "sharding_plans": list(plan.sharding_plans.keys()),
                }

            elif action == "deploy_agents":
                if (
                    not hasattr(self.meta_agent_coordinator, "current_deployment")
                    or not self.meta_agent_coordinator.current_deployment
                ):
                    return {"error": "No deployment plan available - create plan first"}

                deployment_results = await self.meta_agent_coordinator.deploy_agents(
                    self.meta_agent_coordinator.current_deployment
                )
                results = {"deployment_results": deployment_results}

            elif action == "get_privacy_report":
                privacy_report = self.meta_agent_coordinator.get_privacy_report()
                results = {"privacy_report": privacy_report}

            else:
                return {"error": f"Unknown action: {action}"}

            await self._log_governance_action(agent_id, f"meta_agent_{action}", "meta_agents", results)
            return {"success": True, "results": results}

        except Exception as e:
            logger.error(f"Error coordinating meta-agents: {e}")
            return {"error": str(e)}

    async def mcp_govern_distributed_rag(
        self, agent_id: str, action: str, proposal_data: dict[str, Any] = None, query: str = None
    ) -> dict[str, Any]:
        """MCP Tool: Governance interface for distributed RAG system"""

        if not self.rag_governance:
            return {"error": "RAG governance not available"}

        try:
            results = {}

            if action == "create_proposal":
                if not proposal_data:
                    return {"error": "Proposal data required"}

                result = await self.rag_governance.create_proposal(
                    agent_id=agent_id,
                    title=proposal_data.get("title", ""),
                    description=proposal_data.get("description", ""),
                    changes=proposal_data.get("changes", {}),
                    decision_type=proposal_data.get("decision_type", "major_change"),
                )
                results = result

            elif action == "vote_proposal":
                if not proposal_data:
                    return {"error": "Proposal ID and vote required"}

                result = await self.rag_governance.vote_proposal(
                    agent_id=agent_id,
                    proposal_id=proposal_data.get("proposal_id", ""),
                    vote=proposal_data.get("vote", False),
                )
                results = result

            elif action == "submit_research_update":
                if not proposal_data:
                    return {"error": "Research results required"}

                result = await self.rag_governance.submit_research_update(proposal_data)
                results = result

            elif action == "query_distributed_knowledge":
                if not query:
                    return {"error": "Query required"}

                result = await self.rag_governance.query_distributed_knowledge(
                    query=query, mode=proposal_data.get("mode", "balanced") if proposal_data else "balanced"
                )
                results = result

            elif action == "get_rag_metrics":
                result = await self.rag_governance.get_rag_metrics()
                results = result

            else:
                return {"error": f"Unknown action: {action}"}

            await self._log_governance_action(agent_id, f"rag_{action}", "distributed_rag", results)
            return {"success": True, "results": results}

        except Exception as e:
            logger.error(f"Error in RAG governance: {e}")
            return {"error": str(e)}

    async def mcp_manage_p2p_network(self, agent_id: str, action: str, config: dict[str, Any] = None) -> dict[str, Any]:
        """MCP Tool: Manage P2P network and transport systems"""

        if not await self._authorize_action(agent_id, GovernanceLevel.OPERATOR):
            return {"error": "Insufficient permissions - requires Operator level"}

        if not self.transport_manager:
            return {"error": "Transport manager not available"}

        try:
            results = {}

            if action == "get_network_status":
                stats = await self.transport_manager.get_stats()
                active_transports = await self.transport_manager.get_active_transports()

                results = {
                    "transport_stats": stats,
                    "active_transports": active_transports,
                    "primary_transport": getattr(self.transport_manager, "primary_transport", "unknown"),
                }

            elif action == "optimize_routing":
                # Trigger routing optimization based on current conditions
                if hasattr(self.transport_manager, "optimize_transport_routing"):
                    optimization_result = await self.transport_manager.optimize_transport_routing()
                    results = {"optimization_result": optimization_result}
                else:
                    results = {"message": "Routing optimization not supported"}

            elif action == "update_transport_config":
                if config:
                    # Update transport configuration
                    success = await self._update_transport_config(config)
                    results = {"config_updated": success, "config": config}
                else:
                    return {"error": "Config required for transport updates"}

            else:
                return {"error": f"Unknown action: {action}"}

            await self._log_governance_action(agent_id, f"p2p_{action}", "p2p_network", results)
            return {"success": True, "results": results}

        except Exception as e:
            logger.error(f"Error managing P2P network: {e}")
            return {"error": str(e)}

    async def mcp_emergency_override(
        self, agent_id: str, override_action: str, target_system: str, justification: str
    ) -> dict[str, Any]:
        """MCP Tool: Emergency override capabilities (King agent only)"""

        if agent_id != "king":
            return {"error": "Emergency override only available to King agent"}

        if not await self._authorize_action(agent_id, GovernanceLevel.EMERGENCY):
            return {"error": "Emergency authorization failed"}

        try:
            results = {}
            emergency_id = str(uuid4())

            # Log emergency action immediately
            emergency_log = {
                "emergency_id": emergency_id,
                "agent_id": agent_id,
                "action": override_action,
                "target": target_system,
                "justification": justification,
                "timestamp": datetime.now().isoformat(),
            }

            self.system_alerts.append(
                {
                    "type": "EMERGENCY_OVERRIDE",
                    "severity": "CRITICAL",
                    "message": f"King agent emergency override: {override_action} on {target_system}",
                    "details": emergency_log,
                }
            )

            # Execute emergency action based on target system
            if target_system == "all_systems":
                if override_action == "emergency_shutdown":
                    results = await self._emergency_shutdown_all_systems()
                elif override_action == "force_restart":
                    results = await self._force_restart_all_systems()
                else:
                    return {"error": f"Unknown emergency action: {override_action}"}

            elif target_system == "digital_twins":
                if override_action == "privacy_lockdown":
                    results = await self._emergency_privacy_lockdown()
                else:
                    return {"error": f"Unknown emergency action for digital twins: {override_action}"}

            else:
                return {"error": f"Unknown target system: {target_system}"}

            await self._log_governance_action(
                agent_id,
                f"emergency_{override_action}",
                SystemComponent(target_system) if target_system != "all_systems" else SystemComponent.DIGITAL_TWINS,
                {"emergency_id": emergency_id, "results": results},
            )

            return {"success": True, "emergency_id": emergency_id, "results": results}

        except Exception as e:
            logger.error(f"Emergency override failed: {e}")
            return {"error": str(e)}

    async def mcp_privacy_audit_report(self, agent_id: str, audit_scope: str = "full") -> dict[str, Any]:
        """MCP Tool: Generate comprehensive privacy audit report"""

        if not await self._authorize_action(agent_id, GovernanceLevel.READ_ONLY):
            return {"error": "Insufficient permissions"}

        try:
            audit_report = {
                "audit_id": str(uuid4()),
                "generated_by": agent_id,
                "timestamp": datetime.now().isoformat(),
                "scope": audit_scope,
                "compliance_status": self.compliance_status.copy(),
                "privacy_violations": self.metrics["privacy_violations"],
                "audit_trail_entries": len(self.privacy_audit_trail),
            }

            # Component-specific privacy audits
            if audit_scope in ["full", "digital_twins"] and self.digital_twin_concierge:
                digital_twin_privacy = self.digital_twin_concierge.get_privacy_report()
                audit_report["digital_twin_privacy"] = digital_twin_privacy

            if audit_scope in ["full", "meta_agents"] and self.meta_agent_coordinator:
                meta_agent_privacy = self.meta_agent_coordinator.get_privacy_report()
                audit_report["meta_agent_privacy"] = meta_agent_privacy

            if audit_scope in ["full", "rag_system"] and self.distributed_rag:
                rag_status = self.distributed_rag.get_system_status()
                audit_report["rag_privacy"] = {
                    "distributed_nodes": len(rag_status.get("fog_network", {}).get("nodes", {})),
                    "governance_active": rag_status.get("distributed_rag_coordinator", {}).get("status")
                    == "operational",
                }

            # Recent privacy audit events
            recent_events = [
                event
                for event in self.privacy_audit_trail[-50:]  # Last 50 events
                if (datetime.now() - datetime.fromisoformat(event.get("timestamp", "2020-01-01T00:00:00"))).days <= 30
            ]
            audit_report["recent_privacy_events"] = recent_events

            await self._log_governance_action(agent_id, "privacy_audit_report", "privacy_audit")
            return {"success": True, "audit_report": audit_report}

        except Exception as e:
            logger.error(f"Privacy audit failed: {e}")
            return {"error": str(e)}

    # === PRIVATE HELPER METHODS ===

    async def _initialize_component_monitoring(self):
        """Initialize monitoring for all system components"""

        components = [
            SystemComponent.DIGITAL_TWINS,
            SystemComponent.META_AGENTS,
            SystemComponent.DISTRIBUTED_RAG,
            SystemComponent.P2P_NETWORK,
            SystemComponent.FOG_COMPUTE,
            SystemComponent.MOBILE_OPTIMIZATION,
        ]

        for component in components:
            status = SystemStatus(
                component=component, status="initializing", health_score=0.0, metrics={}, alerts=[], resource_usage={}
            )

            self.system_status[component] = status

        logger.info("Component monitoring initialized for all systems")

    async def _periodic_health_checks(self):
        """Periodic system health monitoring"""

        while True:
            try:
                await asyncio.sleep(60)  # Check every minute

                # Update component health
                for component in self.system_status.keys():
                    await self._update_component_health(component)

                # Update system metrics
                await self._update_system_metrics()

                # Check for alerts
                await self._check_system_alerts()

            except Exception as e:
                logger.error(f"Health check failed: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error

    async def _update_component_health(self, component: SystemComponent):
        """Update health status for a specific component"""

        try:
            if component == SystemComponent.DIGITAL_TWINS and self.digital_twin_concierge:
                privacy_report = self.digital_twin_concierge.get_privacy_report()
                metrics = self.digital_twin_concierge.get_learning_metrics()

                self.system_status[component].status = "operational"
                self.system_status[component].health_score = 0.9
                self.system_status[component].metrics = {
                    "privacy_status": privacy_report.get("status", "unknown"),
                    "learning_effectiveness": metrics.get("learning_effectiveness", 0.0),
                    "data_sources_active": len(self.digital_twin_concierge.data_sources),
                }

            elif component == SystemComponent.META_AGENTS and self.meta_agent_coordinator:
                deployment_status = await self.meta_agent_coordinator.get_deployment_status()

                self.system_status[component].status = (
                    "operational" if deployment_status.get("local_agents", 0) > 0 else "standby"
                )
                self.system_status[component].health_score = 0.85
                self.system_status[component].metrics = deployment_status.get("metrics", {})

            elif component == SystemComponent.DISTRIBUTED_RAG and self.distributed_rag:
                rag_status = self.distributed_rag.get_system_status()

                self.system_status[component].status = rag_status.get("distributed_rag_coordinator", {}).get(
                    "status", "unknown"
                )
                self.system_status[component].health_score = 0.8
                self.system_status[component].metrics = rag_status.get("distributed_rag_coordinator", {}).get(
                    "metrics", {}
                )

            elif component == SystemComponent.P2P_NETWORK and self.transport_manager:
                transport_stats = await self.transport_manager.get_stats()

                self.system_status[component].status = "operational"
                self.system_status[component].health_score = 0.75
                self.system_status[component].metrics = transport_stats

            elif component == SystemComponent.FOG_COMPUTE and self.fog_coordinator:
                fog_status = self.fog_coordinator.get_system_status()

                self.system_status[component].status = "operational"
                self.system_status[component].health_score = 0.7
                self.system_status[component].metrics = fog_status

            else:
                # Component not available
                self.system_status[component].status = "unavailable"
                self.system_status[component].health_score = 0.0
                self.system_status[component].metrics = {}

            self.system_status[component].last_updated = datetime.now()

        except Exception as e:
            logger.error(f"Failed to update health for {component.value}: {e}")
            self.system_status[component].status = "error"
            self.system_status[component].health_score = 0.0

    async def _calculate_overall_system_health(self) -> float:
        """Calculate overall system health score"""

        if not self.system_status:
            return 0.0

        health_scores = [status.health_score for status in self.system_status.values()]
        return sum(health_scores) / len(health_scores) if health_scores else 0.0

    async def _authorize_action(self, agent_id: str, required_level: GovernanceLevel) -> bool:
        """Authorize agent for governance action"""

        # Basic authorization logic
        if required_level == GovernanceLevel.READ_ONLY:
            return True  # All agents can read

        if required_level == GovernanceLevel.EMERGENCY:
            return agent_id == "king"  # Only King can do emergency actions

        if required_level in [GovernanceLevel.GOVERNANCE, GovernanceLevel.COORDINATOR, GovernanceLevel.OPERATOR]:
            return agent_id in self.authorized_agents

        return False

    async def _log_governance_action(self, agent_id: str, action_type: str, target: str, result: dict[str, Any] = None):
        """Log governance action for audit trail"""

        action = GovernanceAction(
            agent_id=agent_id,
            action_type=action_type,
            target_system=SystemComponent(target) if isinstance(target, str) else target,
            description=f"{agent_id} performed {action_type} on {target}",
            status="completed",
            result=result or {},
        )

        self.governance_actions[action.action_id] = action

        # Also add to privacy audit trail if relevant
        if any(keyword in action_type.lower() for keyword in ["privacy", "data", "audit"]):
            privacy_event = {
                "event_id": action.action_id,
                "agent_id": agent_id,
                "event_type": action_type,
                "timestamp": action.timestamp.isoformat(),
                "details": result or {},
            }
            self.privacy_audit_trail.append(privacy_event)

    async def _privacy_compliance_monitor(self):
        """Monitor privacy compliance across all systems"""

        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes

                # Check digital twin privacy compliance
                if self.digital_twin_concierge:
                    privacy_report = self.digital_twin_concierge.get_privacy_report()

                    if privacy_report.get("status") != "all_local":
                        self._raise_privacy_alert("Digital twin data leakage detected")

                # Check meta-agent privacy
                if self.meta_agent_coordinator:
                    privacy_report = self.meta_agent_coordinator.get_privacy_report()

                    if (
                        not privacy_report.get("privacy_guarantees", {}).get("digital_twin")
                        == "all_data_local_never_shared"
                    ):
                        self._raise_privacy_alert("Meta-agent privacy violation detected")

                # Update compliance status
                self.compliance_status["last_check"] = datetime.now().isoformat()

            except Exception as e:
                logger.error(f"Privacy compliance monitoring failed: {e}")
                await asyncio.sleep(600)  # Wait longer on error

    def _raise_privacy_alert(self, message: str):
        """Raise privacy compliance alert"""

        alert = {
            "type": "PRIVACY_VIOLATION",
            "severity": "HIGH",
            "message": message,
            "timestamp": datetime.now().isoformat(),
        }

        self.system_alerts.append(alert)
        self.metrics["privacy_violations"] += 1

        logger.warning(f"Privacy alert: {message}")

    def get_governance_status(self) -> dict[str, Any]:
        """Get current governance system status"""

        return {
            "dashboard_operational": True,
            "authorized_agents": list(self.authorized_agents),
            "system_components": [comp.value for comp in self.system_status.keys()],
            "overall_health": asyncio.create_task(self._calculate_overall_system_health()),
            "active_alerts": len(self.system_alerts),
            "governance_actions_total": len(self.governance_actions),
            "privacy_compliance": self.compliance_status.copy(),
            "last_updated": datetime.now().isoformat(),
        }


# Example usage and integration
async def demo_unified_governance():
    """Demonstrate the unified governance dashboard"""

    print("🎛️ Unified MCP Governance Dashboard Demo")
    print("=" * 60)

    # Mock components for demo
    class MockDigitalTwin:
        def get_privacy_report(self):
            return {"status": "all_local", "data_sources": ["conversations", "location"]}

        def get_learning_metrics(self):
            return {"learning_effectiveness": 0.85, "surprise_threshold": 0.3}

        @property
        def data_sources(self):
            return ["conversations", "location", "app_usage"]

    class MockMetaAgentCoordinator:
        async def get_deployment_status(self):
            return {"local_agents": 2, "fog_agents": 5, "metrics": {"deployment_time": 45.2}}

        def get_privacy_report(self):
            return {
                "privacy_guarantees": {
                    "digital_twin": "all_data_local_never_shared",
                    "meta_agents": "inference_only_no_personal_data",
                }
            }

    class MockDistributedRAG:
        def get_system_status(self):
            return {
                "distributed_rag_coordinator": {
                    "status": "operational",
                    "metrics": {"total_knowledge_pieces": 15000, "shards_active": 8},
                }
            }

    class MockTransportManager:
        async def get_stats(self):
            return {"messages_sent": 1250, "bytes_transferred": 5500000, "active_connections": 12}

    class MockFogCoordinator:
        def get_system_status(self):
            return {"nodes": {"total": 6}, "utilization": 0.65}

    # Create dashboard with mock components
    dashboard = UnifiedMCPGovernanceDashboard(
        digital_twin_concierge=MockDigitalTwin(),
        meta_agent_coordinator=MockMetaAgentCoordinator(),
        distributed_rag=MockDistributedRAG(),
        transport_manager=MockTransportManager(),
        fog_coordinator=MockFogCoordinator(),
    )

    # Initialize dashboard
    await dashboard.initialize_dashboard()

    # Test MCP tools
    print("\n📊 Testing System Overview...")
    overview = await dashboard.mcp_get_system_overview("sage")
    print(f"System health: {overview.get('overview', {}).get('system_health')}")
    print(f"Active components: {len(overview.get('overview', {}).get('component_status', {}))}")

    print("\n🤖 Testing Digital Twin Management...")
    dt_status = await dashboard.mcp_manage_digital_twins("curator", "get_status")
    print(f"Digital twin status: {dt_status.get('success', False)}")

    print("\n🧠 Testing Meta-Agent Coordination...")
    ma_status = await dashboard.mcp_coordinate_meta_agents("king", "get_deployment_status")
    print(f"Meta-agent coordination: {ma_status.get('success', False)}")

    print("\n📚 Testing RAG Governance...")
    rag_metrics = await dashboard.mcp_govern_distributed_rag("sage", "get_rag_metrics")
    print(f"RAG governance: {rag_metrics.get('success', False)}")

    print("\n🔒 Testing Privacy Audit...")
    privacy_audit = await dashboard.mcp_privacy_audit_report("curator", "full")
    print(f"Privacy audit completed: {privacy_audit.get('success', False)}")

    # Show governance status
    print("\n📋 Governance Status:")
    status = dashboard.get_governance_status()
    print(f"Dashboard operational: {status.get('dashboard_operational')}")
    print(f"Authorized agents: {status.get('authorized_agents')}")
    print(f"System components: {len(status.get('system_components', []))}")

    print("\n✅ Unified governance dashboard demo completed!")


if __name__ == "__main__":
    asyncio.run(demo_unified_governance())
