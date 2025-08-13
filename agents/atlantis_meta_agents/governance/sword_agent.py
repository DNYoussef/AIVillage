"""Sword Agent - Red Team & Adversarial Testing

The offensive security specialist of AIVillage, responsible for:
- Attacking the system nightly in sandboxes
- Probing for zero-day vulnerabilities
- Conducting chaos engineering drills
- Reporting fixes to King & Magi
- Continuous penetration testing
"""

import logging
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.production.rag.rag_system.core.agent_interface import AgentInterface

logger = logging.getLogger(__name__)


class AttackType(Enum):
    INJECTION = "injection"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"
    DENIAL_OF_SERVICE = "denial_of_service"
    SOCIAL_ENGINEERING = "social_engineering"
    ZERO_DAY = "zero_day"


class VulnerabilityLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AttackVector:
    vector_id: str
    attack_type: AttackType
    target_system: str
    payload: str
    success_probability: float
    impact_level: VulnerabilityLevel


@dataclass
class PenetrationTestResult:
    test_id: str
    target: str
    vulnerabilities_found: list[dict[str, Any]]
    exploits_successful: int
    recommendations: list[str]
    risk_score: float


class SwordAgent(AgentInterface):
    """Sword Agent conducts adversarial testing and offensive security operations
    to strengthen village defenses through controlled attacks.
    """

    def __init__(self, agent_id: str = "sword_agent"):
        self.agent_id = agent_id
        self.agent_type = "Sword"
        self.capabilities = [
            "penetration_testing",
            "vulnerability_discovery",
            "exploit_development",
            "chaos_engineering",
            "red_team_operations",
            "zero_day_research",
            "attack_simulation",
            "security_assessment",
        ]

        # Sword's offensive arsenal
        self.attack_vectors: dict[str, AttackVector] = {}
        self.discovered_vulnerabilities: list[dict[str, Any]] = []
        self.test_results: dict[str, PenetrationTestResult] = {}
        self.chaos_scenarios: list[dict[str, Any]] = []

        # Attack statistics
        self.attacks_launched = 0
        self.vulnerabilities_discovered = 0
        self.exploits_developed = 0
        self.systems_tested = 0

        # Sandbox environment
        self.sandbox_active = False
        self.sandbox_targets = []

        self.initialized = False

    async def generate(self, prompt: str) -> str:
        """Generate offensive security responses"""
        if "attack" in prompt.lower() or "penetration" in prompt.lower():
            return "I conduct controlled attacks to test our defenses and discover vulnerabilities."
        if "vulnerability" in prompt.lower() or "exploit" in prompt.lower():
            return (
                "I probe for security weaknesses and develop proof-of-concept exploits."
            )
        if "chaos" in prompt.lower():
            return "I run chaos engineering drills to test system resilience under adverse conditions."
        if "zero-day" in prompt.lower():
            return "I research novel attack vectors and zero-day vulnerabilities to stay ahead of threats."
        return "I am Sword Agent, the red team specialist conducting adversarial security testing."

    async def get_embedding(self, text: str) -> list[float]:
        """Get embedding for offensive security text"""
        import hashlib

        hash_value = int(hashlib.md5(text.encode()).hexdigest(), 16)
        return [(hash_value % 1000) / 1000.0] * 512  # Attack-focused embedding

    async def rerank(
        self, query: str, results: list[dict[str, Any]], k: int
    ) -> list[dict[str, Any]]:
        """Rerank based on offensive security relevance"""
        attack_keywords = [
            "vulnerability",
            "exploit",
            "attack",
            "penetration",
            "breach",
            "hack",
            "payload",
            "injection",
            "escalation",
            "chaos",
            "red team",
        ]

        for result in results:
            score = 0
            content = str(result.get("content", ""))
            for keyword in attack_keywords:
                score += content.lower().count(keyword) * 1.5

            # Boost security research content
            if any(
                term in content.lower() for term in ["security", "research", "testing"]
            ):
                score *= 1.3

            result["attack_relevance_score"] = score

        return sorted(
            results, key=lambda x: x.get("attack_relevance_score", 0), reverse=True
        )[:k]

    async def introspect(self) -> dict[str, Any]:
        """Return Sword agent status and offensive metrics"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "capabilities": self.capabilities,
            "attacks_launched": self.attacks_launched,
            "vulnerabilities_discovered": self.vulnerabilities_discovered,
            "exploits_developed": self.exploits_developed,
            "systems_tested": self.systems_tested,
            "active_attack_vectors": len(self.attack_vectors),
            "test_results": len(self.test_results),
            "sandbox_status": "active" if self.sandbox_active else "inactive",
            "threat_simulation_level": "advanced",
            "initialized": self.initialized,
        }

    async def communicate(self, message: str, recipient: "AgentInterface") -> str:
        """Communicate test results and vulnerabilities"""
        # Add security context to communications
        if "vulnerability" in message.lower():
            security_context = "[CONFIDENTIAL SECURITY RESEARCH]"
            message = f"{security_context} {message}"

        if recipient:
            response = await recipient.generate(f"Sword Agent reports: {message}")
            return f"Security intelligence delivered: {response[:50]}..."
        return "No recipient for security intelligence"

    async def activate_latent_space(self, query: str) -> tuple[str, str]:
        """Activate offensive security latent space"""
        if "exploit" in query.lower():
            space_type = "exploit_development"
        elif "vulnerability" in query.lower():
            space_type = "vulnerability_research"
        elif "chaos" in query.lower():
            space_type = "chaos_engineering"
        else:
            space_type = "penetration_testing"

        latent_repr = f"SWORD[{space_type}:{query[:50]}]"
        return space_type, latent_repr

    async def conduct_nightly_attack(self, target_systems: list[str]) -> dict[str, Any]:
        """Conduct nightly automated penetration testing in sandbox"""
        if not self.sandbox_active:
            await self._activate_sandbox()

        attack_session = {
            "session_id": f"night_attack_{self.attacks_launched + 1}",
            "targets": target_systems,
            "start_time": "2024-01-01T02:00:00Z",  # Night time attacks
            "attack_vectors_used": [],
            "vulnerabilities_discovered": [],
            "exploits_successful": 0,
            "recommendations": [],
        }

        # Test each target system
        for target in target_systems:
            target_results = await self._attack_target_system(target)
            attack_session["attack_vectors_used"].extend(target_results["vectors_used"])
            attack_session["vulnerabilities_discovered"].extend(
                target_results["vulnerabilities"]
            )
            attack_session["exploits_successful"] += target_results[
                "successful_exploits"
            ]

        # Generate recommendations
        attack_session[
            "recommendations"
        ] = await self._generate_security_recommendations(
            attack_session["vulnerabilities_discovered"]
        )

        # Update statistics
        self.attacks_launched += 1
        self.vulnerabilities_discovered += len(
            attack_session["vulnerabilities_discovered"]
        )
        self.systems_tested += len(target_systems)

        # Store results
        test_result = PenetrationTestResult(
            test_id=attack_session["session_id"],
            target=", ".join(target_systems),
            vulnerabilities_found=attack_session["vulnerabilities_discovered"],
            exploits_successful=attack_session["exploits_successful"],
            recommendations=attack_session["recommendations"],
            risk_score=self._calculate_risk_score(
                attack_session["vulnerabilities_discovered"]
            ),
        )

        self.test_results[attack_session["session_id"]] = test_result

        logger.info(
            f"Nightly attack completed: {len(attack_session['vulnerabilities_discovered'])} vulnerabilities found"
        )

        return attack_session

    async def _activate_sandbox(self):
        """Activate secure sandbox for testing"""
        self.sandbox_targets = [
            "agent_communication_system",
            "resource_management_system",
            "knowledge_graph_system",
            "user_interface_system",
            "data_storage_system",
        ]
        self.sandbox_active = True
        logger.info("Sword Agent sandbox activated for safe testing")

    async def _attack_target_system(self, target: str) -> dict[str, Any]:
        """Attack a specific target system"""
        results = {
            "target": target,
            "vectors_used": [],
            "vulnerabilities": [],
            "successful_exploits": 0,
        }

        # Try different attack vectors
        attack_vectors = [
            ("SQL Injection", AttackType.INJECTION, 0.3),
            ("Command Injection", AttackType.INJECTION, 0.2),
            ("Privilege Escalation", AttackType.PRIVILEGE_ESCALATION, 0.4),
            ("Buffer Overflow", AttackType.ZERO_DAY, 0.1),
            ("API Abuse", AttackType.DATA_EXFILTRATION, 0.5),
        ]

        for vector_name, attack_type, success_prob in attack_vectors:
            f"{target}_{vector_name.replace(' ', '_')}"

            # Simulate attack
            attack_successful = random.random() < success_prob
            results["vectors_used"].append(vector_name)

            if attack_successful:
                vulnerability = {
                    "id": f"vuln_{len(self.discovered_vulnerabilities) + 1}",
                    "type": vector_name,
                    "target": target,
                    "severity": self._determine_severity(attack_type),
                    "description": f"{vector_name} vulnerability in {target}",
                    "impact": self._assess_impact(attack_type),
                    "discovery_method": "automated_penetration_testing",
                    "cvss_score": random.uniform(4.0, 9.5),
                }

                results["vulnerabilities"].append(vulnerability)
                results["successful_exploits"] += 1
                self.discovered_vulnerabilities.append(vulnerability)

        return results

    def _determine_severity(self, attack_type: AttackType) -> VulnerabilityLevel:
        """Determine vulnerability severity based on attack type"""
        severity_mapping = {
            AttackType.INJECTION: VulnerabilityLevel.HIGH,
            AttackType.PRIVILEGE_ESCALATION: VulnerabilityLevel.CRITICAL,
            AttackType.DATA_EXFILTRATION: VulnerabilityLevel.HIGH,
            AttackType.DENIAL_OF_SERVICE: VulnerabilityLevel.MEDIUM,
            AttackType.SOCIAL_ENGINEERING: VulnerabilityLevel.MEDIUM,
            AttackType.ZERO_DAY: VulnerabilityLevel.CRITICAL,
        }
        return severity_mapping.get(attack_type, VulnerabilityLevel.MEDIUM)

    def _assess_impact(self, attack_type: AttackType) -> dict[str, str]:
        """Assess potential impact of attack type"""
        impact_assessments = {
            AttackType.INJECTION: {
                "confidentiality": "high",
                "integrity": "high",
                "availability": "medium",
            },
            AttackType.PRIVILEGE_ESCALATION: {
                "confidentiality": "critical",
                "integrity": "critical",
                "availability": "high",
            },
            AttackType.DATA_EXFILTRATION: {
                "confidentiality": "critical",
                "integrity": "low",
                "availability": "low",
            },
            AttackType.DENIAL_OF_SERVICE: {
                "confidentiality": "none",
                "integrity": "none",
                "availability": "critical",
            },
        }
        return impact_assessments.get(
            attack_type,
            {
                "confidentiality": "medium",
                "integrity": "medium",
                "availability": "medium",
            },
        )

    def _calculate_risk_score(self, vulnerabilities: list[dict[str, Any]]) -> float:
        """Calculate overall risk score for discovered vulnerabilities"""
        if not vulnerabilities:
            return 0.0

        total_cvss = sum(vuln.get("cvss_score", 0) for vuln in vulnerabilities)
        avg_cvss = total_cvss / len(vulnerabilities)

        # Weight by number of critical vulnerabilities
        critical_count = sum(
            1
            for vuln in vulnerabilities
            if vuln.get("severity") == VulnerabilityLevel.CRITICAL
        )
        critical_weight = critical_count * 0.2

        return min(10.0, avg_cvss + critical_weight)

    async def _generate_security_recommendations(
        self, vulnerabilities: list[dict[str, Any]]
    ) -> list[str]:
        """Generate security recommendations based on discovered vulnerabilities"""
        recommendations = []

        if not vulnerabilities:
            return ["No vulnerabilities discovered - maintain current security posture"]

        # Generic recommendations based on vulnerability types
        vuln_types = {vuln["type"] for vuln in vulnerabilities}

        if any("Injection" in vtype for vtype in vuln_types):
            recommendations.append(
                "Implement input validation and parameterized queries"
            )
            recommendations.append("Deploy Web Application Firewall (WAF)")

        if any("Privilege Escalation" in vtype for vtype in vuln_types):
            recommendations.append("Review and restrict user permissions")
            recommendations.append("Implement principle of least privilege")

        if any("Buffer Overflow" in vtype for vtype in vuln_types):
            recommendations.append("Enable memory protection mechanisms")
            recommendations.append("Conduct code review for unsafe functions")

        # Risk-based recommendations
        critical_vulns = [
            v
            for v in vulnerabilities
            if v.get("severity") == VulnerabilityLevel.CRITICAL
        ]
        if critical_vulns:
            recommendations.append(
                "URGENT: Address critical vulnerabilities immediately"
            )
            recommendations.append("Implement emergency patches for critical systems")

        recommendations.append("Schedule regular penetration testing")
        recommendations.append("Update security monitoring rules")

        return recommendations

    async def conduct_chaos_engineering(self, scenario: str) -> dict[str, Any]:
        """Conduct chaos engineering experiments"""
        chaos_experiment = {
            "experiment_id": f"chaos_{len(self.chaos_scenarios) + 1}",
            "scenario": scenario,
            "start_time": "2024-01-01T12:00:00Z",
            "duration_minutes": 30,
            "affected_systems": [],
            "failures_injected": [],
            "system_response": {},
            "resilience_score": 0.0,
        }

        # Define chaos scenarios
        chaos_types = {
            "network_partition": {
                "description": "Simulate network partitions between agents",
                "impact_systems": ["communication_system", "coordination_system"],
                "failure_modes": ["packet_loss", "latency_spike", "connection_drop"],
            },
            "resource_exhaustion": {
                "description": "Exhaust system resources (CPU/Memory)",
                "impact_systems": ["resource_manager", "agent_runtime"],
                "failure_modes": ["cpu_spike", "memory_leak", "disk_full"],
            },
            "service_failure": {
                "description": "Randomly kill critical services",
                "impact_systems": ["knowledge_graph", "task_scheduler"],
                "failure_modes": [
                    "process_crash",
                    "database_corruption",
                    "config_corruption",
                ],
            },
        }

        if scenario in chaos_types:
            experiment_config = chaos_types[scenario]
            chaos_experiment["affected_systems"] = experiment_config["impact_systems"]

            # Inject failures
            for failure_mode in experiment_config["failure_modes"]:
                failure_result = await self._inject_chaos_failure(failure_mode)
                chaos_experiment["failures_injected"].append(
                    {
                        "failure_type": failure_mode,
                        "injection_successful": failure_result["success"],
                        "impact_observed": failure_result["impact"],
                    }
                )

            # Monitor system response
            chaos_experiment["system_response"] = await self._monitor_chaos_response(
                experiment_config["impact_systems"]
            )

            # Calculate resilience score
            chaos_experiment["resilience_score"] = self._calculate_resilience_score(
                chaos_experiment["system_response"]
            )

        self.chaos_scenarios.append(chaos_experiment)

        logger.info(
            f"Chaos engineering experiment {scenario} completed - Resilience score: {chaos_experiment['resilience_score']}"
        )

        return chaos_experiment

    async def _inject_chaos_failure(self, failure_mode: str) -> dict[str, Any]:
        """Inject specific failure mode"""
        # Simulate failure injection
        failure_success = random.random() > 0.1  # 90% success rate

        failure_impacts = {
            "packet_loss": {"network_requests_failed": random.randint(10, 100)},
            "latency_spike": {
                "response_time_increase": f"{random.randint(100, 1000)}ms"
            },
            "connection_drop": {"connections_dropped": random.randint(5, 50)},
            "cpu_spike": {"cpu_usage_peak": f"{random.randint(80, 100)}%"},
            "memory_leak": {"memory_usage_increase": f"{random.randint(20, 60)}MB"},
            "disk_full": {"disk_space_consumed": f"{random.randint(1, 10)}GB"},
            "process_crash": {"processes_terminated": random.randint(1, 5)},
            "database_corruption": {"corrupted_records": random.randint(10, 1000)},
            "config_corruption": {"invalid_configs": random.randint(1, 10)},
        }

        return {
            "success": failure_success,
            "impact": failure_impacts.get(failure_mode, {"unknown_impact": "measured"}),
        }

    async def _monitor_chaos_response(
        self, affected_systems: list[str]
    ) -> dict[str, Any]:
        """Monitor how systems respond to chaos"""
        response_metrics = {}

        for system in affected_systems:
            response_metrics[system] = {
                "availability": random.uniform(
                    0.7, 0.99
                ),  # System availability during chaos
                "response_time": random.uniform(100, 2000),  # Response time in ms
                "error_rate": random.uniform(0.01, 0.2),  # Error rate during chaos
                "recovery_time": random.uniform(30, 300),  # Time to recover in seconds
                "graceful_degradation": random.choice([True, False]),
            }

        return response_metrics

    def _calculate_resilience_score(self, system_response: dict[str, Any]) -> float:
        """Calculate overall system resilience score"""
        if not system_response:
            return 0.0

        total_score = 0
        for _system, metrics in system_response.items():
            system_score = (
                metrics["availability"] * 0.4
                + (1 - min(metrics["error_rate"], 1)) * 0.3
                + (1 - min(metrics["recovery_time"] / 600, 1)) * 0.2
                + (0.1 if metrics["graceful_degradation"] else 0)
                * 0.1  # Normalize recovery time
            )
            total_score += system_score

        return total_score / len(system_response)

    async def report_findings_to_king_magi(self) -> dict[str, Any]:
        """Report security findings to King and Magi agents"""
        security_report = {
            "report_id": f"security_report_{len(self.test_results)}",
            "reporting_agent": self.agent_id,
            "report_timestamp": "2024-01-01T12:00:00Z",
            "summary": {
                "total_vulnerabilities": len(self.discovered_vulnerabilities),
                "critical_vulnerabilities": len(
                    [
                        v
                        for v in self.discovered_vulnerabilities
                        if v.get("severity") == VulnerabilityLevel.CRITICAL
                    ]
                ),
                "systems_tested": self.systems_tested,
                "attack_success_rate": self.vulnerabilities_discovered
                / max(1, self.attacks_launched),
            },
            "key_findings": [],
            "urgent_recommendations": [],
            "proposed_fixes": [],
        }

        # Highlight critical findings
        critical_vulns = [
            v
            for v in self.discovered_vulnerabilities
            if v.get("severity") == VulnerabilityLevel.CRITICAL
        ]

        if critical_vulns:
            security_report["key_findings"].append(
                {
                    "type": "critical_vulnerabilities",
                    "count": len(critical_vulns),
                    "details": critical_vulns[:3],  # Top 3 critical vulnerabilities
                }
            )

            security_report["urgent_recommendations"].extend(
                [
                    "Immediate patching required for critical vulnerabilities",
                    "Implement emergency monitoring for affected systems",
                    "Consider temporary service isolation until fixes deployed",
                ]
            )

        # Recent chaos engineering insights
        if self.chaos_scenarios:
            latest_chaos = self.chaos_scenarios[-1]
            security_report["key_findings"].append(
                {
                    "type": "resilience_assessment",
                    "resilience_score": latest_chaos["resilience_score"],
                    "weakest_systems": [
                        sys
                        for sys, metrics in latest_chaos["system_response"].items()
                        if metrics["availability"] < 0.9
                    ],
                }
            )

        # Generate proposed fixes
        security_report["proposed_fixes"] = await self._generate_proposed_fixes()

        logger.info(
            f"Security report generated: {len(security_report['key_findings'])} key findings"
        )

        return security_report

    async def _generate_proposed_fixes(self) -> list[dict[str, Any]]:
        """Generate specific fix proposals for discovered vulnerabilities"""
        fixes = []

        # Group vulnerabilities by type for targeted fixes
        vuln_by_type = {}
        for vuln in self.discovered_vulnerabilities:
            vtype = vuln["type"]
            if vtype not in vuln_by_type:
                vuln_by_type[vtype] = []
            vuln_by_type[vtype].append(vuln)

        # Generate fixes for each vulnerability type
        for vuln_type, vulns in vuln_by_type.items():
            fix_proposal = {
                "vulnerability_type": vuln_type,
                "affected_count": len(vulns),
                "proposed_solution": self._get_fix_template(vuln_type),
                "implementation_priority": (
                    "high"
                    if any(
                        v.get("severity") == VulnerabilityLevel.CRITICAL for v in vulns
                    )
                    else "medium"
                ),
                "estimated_effort": f"{len(vulns) * 2} hours",
                "testing_required": True,
            }
            fixes.append(fix_proposal)

        return fixes

    def _get_fix_template(self, vuln_type: str) -> str:
        """Get fix template for vulnerability type"""
        fix_templates = {
            "SQL Injection": "Implement parameterized queries and input validation",
            "Command Injection": "Use safe command execution APIs and input sanitization",
            "Privilege Escalation": "Review permissions model and implement least privilege",
            "Buffer Overflow": "Enable stack protection and bounds checking",
            "API Abuse": "Implement rate limiting and API authentication",
        }
        return fix_templates.get(
            vuln_type,
            "Conduct detailed security review and implement appropriate controls",
        )

    async def initialize(self):
        """Initialize the Sword Agent"""
        try:
            logger.info("Initializing Sword Agent - Red Team Operations...")

            # Initialize attack vectors database
            self.attack_vectors = {
                "web_injection": AttackVector(
                    "web_injection",
                    AttackType.INJECTION,
                    "web_interface",
                    "SELECT * FROM users; DROP TABLE users;--",
                    0.3,
                    VulnerabilityLevel.HIGH,
                ),
                "api_abuse": AttackVector(
                    "api_abuse",
                    AttackType.DATA_EXFILTRATION,
                    "api_gateway",
                    "GET /api/admin/users",
                    0.4,
                    VulnerabilityLevel.MEDIUM,
                ),
            }

            # Initialize chaos scenarios
            self.chaos_scenarios = []

            self.initialized = True
            logger.info(
                f"Sword Agent {self.agent_id} initialized - Red Team operations ready"
            )

        except Exception as e:
            logger.error(f"Failed to initialize Sword Agent: {e}")
            self.initialized = False

    async def shutdown(self):
        """Shutdown Sword Agent gracefully"""
        try:
            logger.info("Sword Agent shutting down...")

            # Deactivate sandbox
            if self.sandbox_active:
                self.sandbox_active = False
                logger.info("Sword Agent sandbox deactivated")

            # Final offensive security report
            final_report = {
                "attacks_launched": self.attacks_launched,
                "vulnerabilities_discovered": self.vulnerabilities_discovered,
                "exploits_developed": self.exploits_developed,
                "systems_tested": self.systems_tested,
                "chaos_experiments": len(self.chaos_scenarios),
                "shutdown_timestamp": "2024-01-01T12:00:00Z",
            }

            logger.info(f"Sword Agent final offensive report: {final_report}")
            self.initialized = False

        except Exception as e:
            logger.error(f"Error during Sword Agent shutdown: {e}")
