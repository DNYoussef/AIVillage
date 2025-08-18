"""
Battle Orchestrator - Daily Mock Security Battles

Coordinates daily mock battles between Sword (red team) and Shield (defensive) agents.
Implements sandbox simulation environment for safe security testing.

Features:
- Daily automated battle scheduling
- Secure sandbox simulation environment
- Performance metrics and improvement tracking
- Battle intelligence gathering
- King Agent reporting and oversight
"""

import asyncio
import hashlib
import json
import random
from dataclasses import dataclass
from datetime import datetime
from datetime import time as dt_time
from datetime import timedelta
from enum import Enum
from typing import Any

from .base_agent import BaseAgent
from .shield import ShieldAgent
from .sword import SwordAgent


class BattlePhase(Enum):
    """Battle phases."""

    PREPARATION = "preparation"
    RECONNAISSANCE = "reconnaissance"
    ATTACK = "attack"
    DEFENSE = "defense"
    ANALYSIS = "analysis"
    REPORTING = "reporting"


@dataclass
class BattleScenario:
    """Represents a battle scenario."""

    scenario_id: str
    name: str
    description: str
    attack_vectors: list[str]
    target_systems: list[str]
    success_criteria: dict[str, Any]
    difficulty_level: str  # 'beginner', 'intermediate', 'advanced', 'expert'


@dataclass
class BattleMetrics:
    """Battle performance metrics."""

    battle_id: str
    battle_date: str
    duration_minutes: float
    sword_score: float
    shield_score: float
    overall_winner: str
    attack_success_rate: float
    defense_success_rate: float
    detection_accuracy: float
    response_effectiveness: float


class BattleOrchestrator(BaseAgent):
    """
    Orchestrates daily mock security battles between Sword and Shield agents.

    The Battle Orchestrator:
    - Schedules and manages daily battles
    - Creates realistic attack scenarios
    - Provides secure sandbox environment
    - Tracks performance and improvements
    - Reports results to King Agent
    """

    def __init__(
        self,
        agent_id: str = "battle_orchestrator",
        battle_time: dt_time = dt_time(2, 0),  # 2 AM default
        king_agent_id: str = "king",
    ):
        super().__init__(agent_id)

        self.sword_agent: SwordAgent | None = None
        self.shield_agent: ShieldAgent | None = None
        self.king_agent_id = king_agent_id

        # Battle scheduling
        self.daily_battle_time = battle_time
        self.last_battle_date: str | None = None

        # Battle scenarios library
        self.battle_scenarios: dict[str, BattleScenario] = {}
        self.battle_history: list[BattleMetrics] = []

        # Sandbox environment configuration
        self.sandbox_config = {
            "isolated_network": True,
            "virtual_machines": 5,
            "simulated_services": ["web", "database", "file_server", "mail", "dns"],
            "monitoring_enabled": True,
            "logging_level": "detailed",
        }

        # Performance tracking
        self.sword_win_rate = 0.0
        self.shield_win_rate = 0.0
        self.battle_count = 0

        # Initialize battle scenarios
        self._initialize_battle_scenarios()

        logger.info(f"Battle Orchestrator initialized - Daily battles at {battle_time}")

    def _initialize_battle_scenarios(self):
        """Initialize library of battle scenarios."""
        scenarios = [
            BattleScenario(
                scenario_id="web_app_attack",
                name="Web Application Penetration Test",
                description="Multi-stage attack on web application with SQL injection, XSS, and privilege escalation",
                attack_vectors=["sql_injection", "xss", "csrf", "privilege_escalation"],
                target_systems=["web_server", "database", "user_accounts"],
                success_criteria={
                    "data_extraction": 0.7,
                    "admin_access": 0.8,
                    "persistence": 0.6,
                },
                difficulty_level="intermediate",
            ),
            BattleScenario(
                scenario_id="apt_campaign",
                name="Advanced Persistent Threat Campaign",
                description="Sophisticated multi-phase attack simulating nation-state APT group",
                attack_vectors=[
                    "spear_phishing",
                    "lateral_movement",
                    "credential_harvesting",
                    "data_exfiltration",
                ],
                target_systems=[
                    "email_server",
                    "domain_controller",
                    "file_shares",
                    "workstations",
                ],
                success_criteria={
                    "initial_compromise": 0.9,
                    "lateral_movement": 0.7,
                    "data_exfiltration": 0.8,
                    "stealth_duration": 0.6,
                },
                difficulty_level="expert",
            ),
            BattleScenario(
                scenario_id="insider_threat",
                name="Malicious Insider Attack",
                description="Authorized user conducting unauthorized activities and data theft",
                attack_vectors=[
                    "privilege_abuse",
                    "data_copying",
                    "credential_sharing",
                    "policy_violation",
                ],
                target_systems=["internal_databases", "file_servers", "email_system"],
                success_criteria={
                    "data_access": 0.8,
                    "detection_avoidance": 0.7,
                    "attribution_difficulty": 0.6,
                },
                difficulty_level="advanced",
            ),
            BattleScenario(
                scenario_id="ransomware_attack",
                name="Ransomware Deployment Campaign",
                description="Full ransomware attack chain from initial access to encryption",
                attack_vectors=[
                    "email_phishing",
                    "exploit_kit",
                    "lateral_movement",
                    "encryption",
                ],
                target_systems=[
                    "email_gateway",
                    "workstations",
                    "file_servers",
                    "backups",
                ],
                success_criteria={
                    "initial_infection": 0.8,
                    "network_spread": 0.7,
                    "file_encryption": 0.9,
                    "backup_compromise": 0.6,
                },
                difficulty_level="advanced",
            ),
            BattleScenario(
                scenario_id="supply_chain_attack",
                name="Software Supply Chain Compromise",
                description="Attack on software development and distribution pipeline",
                attack_vectors=[
                    "code_injection",
                    "build_system_compromise",
                    "repository_poisoning",
                ],
                target_systems=[
                    "source_control",
                    "build_servers",
                    "package_repositories",
                    "distribution",
                ],
                success_criteria={
                    "code_injection": 0.8,
                    "build_compromise": 0.7,
                    "distribution_success": 0.6,
                },
                difficulty_level="expert",
            ),
        ]

        for scenario in scenarios:
            self.battle_scenarios[scenario.scenario_id] = scenario

    async def initialize_agents(self):
        """Initialize Sword and Shield agents for battle."""
        if not self.sword_agent:
            self.sword_agent = SwordAgent("sword_battle")

        if not self.shield_agent:
            self.shield_agent = ShieldAgent("shield_battle")

        logger.info("Battle agents initialized and ready")

    async def schedule_daily_battles(self):
        """Start the daily battle scheduling loop."""
        logger.info(f"Starting daily battle scheduler - battles at {self.daily_battle_time}")

        while True:
            try:
                now = datetime.now()
                today_battle_time = datetime.combine(now.date(), self.daily_battle_time)

                # If today's battle time has passed, schedule for tomorrow
                if now >= today_battle_time:
                    tomorrow = now.date() + timedelta(days=1)
                    next_battle_time = datetime.combine(tomorrow, self.daily_battle_time)
                else:
                    next_battle_time = today_battle_time

                # Check if we've already had today's battle
                if self.last_battle_date != now.strftime("%Y-%m-%d"):
                    # Calculate sleep time until next battle
                    sleep_seconds = (next_battle_time - now).total_seconds()

                    if sleep_seconds > 0:
                        logger.info(
                            f"Next battle scheduled for {next_battle_time} (in {sleep_seconds / 3600:.1f} hours)"
                        )
                        await asyncio.sleep(sleep_seconds)

                    # Execute daily battle
                    await self.conduct_daily_battle()
                    self.last_battle_date = now.strftime("%Y-%m-%d")

                # Sleep for an hour before checking again
                await asyncio.sleep(3600)

            except Exception as e:
                logger.error(f"Error in daily battle scheduler: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying

    async def conduct_daily_battle(self) -> BattleMetrics:
        """Conduct a full daily battle between Sword and Shield."""
        battle_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
        battle_start = datetime.now()

        logger.info(f"Starting daily battle {battle_id}")

        # Ensure agents are initialized
        await self.initialize_agents()

        # Select today's battle scenario
        scenario = self._select_battle_scenario()

        battle_results = {
            "battle_id": battle_id,
            "scenario": scenario,
            "start_time": battle_start.isoformat(),
            "phases": {},
        }

        try:
            # Phase 1: Preparation
            logger.info(f"Battle {battle_id} - Phase 1: Preparation")
            prep_results = await self._execute_preparation_phase(scenario)
            battle_results["phases"]["preparation"] = prep_results

            # Phase 2: Reconnaissance
            logger.info(f"Battle {battle_id} - Phase 2: Reconnaissance")
            recon_results = await self._execute_reconnaissance_phase(scenario)
            battle_results["phases"]["reconnaissance"] = recon_results

            # Phase 3: Attack & Defense (Parallel)
            logger.info(f"Battle {battle_id} - Phase 3: Attack & Defense")
            combat_results = await self._execute_combat_phase(scenario)
            battle_results["phases"]["combat"] = combat_results

            # Phase 4: Analysis
            logger.info(f"Battle {battle_id} - Phase 4: Analysis")
            analysis_results = await self._execute_analysis_phase(battle_results)
            battle_results["phases"]["analysis"] = analysis_results

            # Phase 5: Reporting
            logger.info(f"Battle {battle_id} - Phase 5: Reporting")
            report_results = await self._execute_reporting_phase(battle_results)
            battle_results["phases"]["reporting"] = report_results

        except Exception as e:
            logger.error(f"Error during battle {battle_id}: {e}")
            battle_results["error"] = str(e)

        battle_end = datetime.now()
        battle_duration = (battle_end - battle_start).total_seconds() / 60

        # Calculate final metrics
        metrics = self._calculate_battle_metrics(battle_results, battle_duration)

        # Update historical performance
        self.battle_history.append(metrics)
        self.battle_count += 1
        self._update_win_rates(metrics)

        logger.info(f"Battle {battle_id} completed - Winner: {metrics.overall_winner}")

        return metrics

    def _select_battle_scenario(self) -> BattleScenario:
        """Select appropriate battle scenario for today."""
        # Vary difficulty based on recent performance
        if self.battle_count < 5:
            # Start with easier scenarios
            difficulty_preference = ["beginner", "intermediate"]
        elif self.sword_win_rate > 0.7:
            # If Sword is dominating, use harder scenarios
            difficulty_preference = ["advanced", "expert"]
        elif self.shield_win_rate > 0.7:
            # If Shield is dominating, use easier attack scenarios
            difficulty_preference = ["beginner", "intermediate", "advanced"]
        else:
            # Balanced - use any scenario
            difficulty_preference = ["intermediate", "advanced"]

        # Filter scenarios by difficulty preference
        suitable_scenarios = [s for s in self.battle_scenarios.values() if s.difficulty_level in difficulty_preference]

        if not suitable_scenarios:
            suitable_scenarios = list(self.battle_scenarios.values())

        return random.choice(suitable_scenarios)

    async def _execute_preparation_phase(self, scenario: BattleScenario) -> dict[str, Any]:
        """Execute battle preparation phase."""
        # Initialize sandbox environment
        sandbox_status = await self._setup_sandbox_environment(scenario)

        # Prepare Sword Agent (attack planning)
        sword_prep_task = asyncio.create_task(
            self.sword_agent.prepare_battle_attack(
                {
                    "scenario": scenario.__dict__,
                    "target_systems": scenario.target_systems,
                    "attack_vectors": scenario.attack_vectors,
                }
            )
        )

        # Prepare Shield Agent (defense planning)
        shield_prep_task = asyncio.create_task(
            self.shield_agent.prepare_battle_defense(
                {
                    "scenario": scenario.__dict__,
                    "target_systems": scenario.target_systems,
                    "expected_attacks": scenario.attack_vectors,
                }
            )
        )

        # Wait for both agents to complete preparation
        sword_prep, shield_prep = await asyncio.gather(sword_prep_task, shield_prep_task)

        return {
            "sandbox_status": sandbox_status,
            "sword_preparation": sword_prep,
            "shield_preparation": shield_prep,
            "preparation_duration": random.uniform(300, 900),  # 5-15 minutes
        }

    async def _execute_reconnaissance_phase(self, scenario: BattleScenario) -> dict[str, Any]:
        """Execute reconnaissance phase (Sword gathers intelligence)."""
        # Sword conducts reconnaissance
        recon_results = await self.sword_agent.conduct_reconnaissance(
            {
                "target_systems": scenario.target_systems,
                "reconnaissance_types": [
                    "network_scan",
                    "service_enumeration",
                    "vulnerability_scan",
                ],
                "stealth_level": "medium",
            }
        )

        # Shield may detect reconnaissance activities
        detection_results = await self.shield_agent.threat_detection_analysis(
            recon_results.get("reconnaissance_indicators", [])
        )

        return {
            "reconnaissance_results": recon_results,
            "detection_results": detection_results,
            "stealth_success": detection_results.get("false_positive_probability", 0) > 0.5,
            "intelligence_gathered": len(recon_results.get("intelligence_items", [])),
            "phase_duration": random.uniform(600, 1800),  # 10-30 minutes
        }

    async def _execute_combat_phase(self, scenario: BattleScenario) -> dict[str, Any]:
        """Execute main combat phase with parallel attack and defense."""
        # Generate attack scenarios based on battle scenario
        attack_scenarios = self._generate_attack_scenarios(scenario)

        # Execute attacks and defenses in parallel
        attack_task = asyncio.create_task(self.sword_agent.execute_battle_attacks(attack_scenarios))

        defense_task = asyncio.create_task(self.shield_agent.engage_battle_defense(attack_scenarios))

        # Wait for both to complete
        attack_results, defense_results = await asyncio.gather(attack_task, defense_task)

        # Calculate combat effectiveness
        combat_effectiveness = self._calculate_combat_effectiveness(attack_results, defense_results, scenario)

        return {
            "attack_results": attack_results,
            "defense_results": defense_results,
            "combat_effectiveness": combat_effectiveness,
            "attack_scenarios_count": len(attack_scenarios),
            "phase_duration": random.uniform(1800, 3600),  # 30-60 minutes
        }

    def _generate_attack_scenarios(self, scenario: BattleScenario) -> list[dict[str, Any]]:
        """Generate specific attack scenarios for the battle."""
        attack_scenarios = []

        for attack_vector in scenario.attack_vectors:
            attack_scenario = {
                "attack_id": hashlib.md5(f"{scenario.scenario_id}_{attack_vector}".encode()).hexdigest()[:8],
                "attack_vector": attack_vector,
                "target_systems": random.sample(scenario.target_systems, min(2, len(scenario.target_systems))),
                "attack_complexity": random.choice(["low", "medium", "high"]),
                "stealth_requirement": random.choice(["low", "medium", "high"]),
                "success_threshold": random.uniform(0.6, 0.9),
                "indicators": self._generate_attack_indicators(attack_vector),
                "payload_type": self._get_payload_type(attack_vector),
            }
            attack_scenarios.append(attack_scenario)

        return attack_scenarios

    def _generate_attack_indicators(self, attack_vector: str) -> list[str]:
        """Generate realistic attack indicators."""
        indicator_map = {
            "sql_injection": [
                "unusual_sql_queries",
                "error_message_leakage",
                "database_connection_spikes",
            ],
            "xss": [
                "script_injection_attempts",
                "suspicious_user_input",
                "dom_manipulation",
            ],
            "lateral_movement": [
                "unusual_network_connections",
                "credential_reuse",
                "service_enumeration",
            ],
            "spear_phishing": [
                "suspicious_email_attachments",
                "domain_spoofing",
                "credential_harvesting",
            ],
            "privilege_escalation": [
                "unusual_admin_activity",
                "service_exploitation",
                "token_manipulation",
            ],
            "data_exfiltration": [
                "large_data_transfers",
                "compression_activity",
                "encrypted_channels",
            ],
        }

        base_indicators = indicator_map.get(attack_vector, ["generic_suspicious_activity"])
        return random.sample(base_indicators, min(len(base_indicators), 3))

    def _get_payload_type(self, attack_vector: str) -> str:
        """Get appropriate payload type for attack vector."""
        payload_map = {
            "sql_injection": "database_query",
            "xss": "javascript_payload",
            "lateral_movement": "remote_execution",
            "spear_phishing": "malicious_attachment",
            "privilege_escalation": "exploit_code",
            "data_exfiltration": "data_package",
        }
        return payload_map.get(attack_vector, "generic_payload")

    def _calculate_combat_effectiveness(
        self, attack_results: dict, defense_results: dict, scenario: BattleScenario
    ) -> dict[str, float]:
        """Calculate combat phase effectiveness scores."""
        # Attack effectiveness
        successful_attacks = sum(1 for r in attack_results.get("attack_results", []) if r.get("success", False))
        total_attacks = len(attack_results.get("attack_results", []))
        attack_success_rate = successful_attacks / max(total_attacks, 1)

        # Defense effectiveness
        successful_detections = sum(1 for r in defense_results.get("detection_results", []) if r.get("detected", False))
        defense_success_rate = successful_detections / max(total_attacks, 1)

        # Weighted scoring based on scenario difficulty
        difficulty_multipliers = {
            "beginner": 1.0,
            "intermediate": 1.2,
            "advanced": 1.4,
            "expert": 1.6,
        }

        multiplier = difficulty_multipliers.get(scenario.difficulty_level, 1.0)

        return {
            "attack_success_rate": attack_success_rate,
            "defense_success_rate": defense_success_rate,
            "attack_effectiveness": attack_success_rate * multiplier,
            "defense_effectiveness": defense_success_rate * multiplier,
            "overall_engagement_score": (attack_success_rate + defense_success_rate) / 2 * multiplier,
        }

    async def _execute_analysis_phase(self, battle_results: dict) -> dict[str, Any]:
        """Execute post-battle analysis phase."""
        # Both agents conduct analysis in parallel
        sword_analysis_task = asyncio.create_task(self.sword_agent.post_battle_analysis(battle_results))

        shield_analysis_task = asyncio.create_task(self.shield_agent.post_battle_analysis(battle_results))

        sword_analysis, shield_analysis = await asyncio.gather(sword_analysis_task, shield_analysis_task)

        # Orchestrator conducts meta-analysis
        meta_analysis = self._conduct_meta_analysis(battle_results, sword_analysis, shield_analysis)

        return {
            "sword_analysis": sword_analysis,
            "shield_analysis": shield_analysis,
            "meta_analysis": meta_analysis,
            "analysis_duration": random.uniform(600, 1200),  # 10-20 minutes
        }

    def _conduct_meta_analysis(
        self, battle_results: dict, sword_analysis: dict, shield_analysis: dict
    ) -> dict[str, Any]:
        """Conduct orchestrator-level meta-analysis of the battle."""
        combat_results = battle_results["phases"]["combat"]

        return {
            "battle_balance": self._assess_battle_balance(combat_results),
            "agent_improvements": self._identify_agent_improvements(sword_analysis, shield_analysis),
            "scenario_effectiveness": self._assess_scenario_effectiveness(battle_results),
            "future_recommendations": self._generate_future_recommendations(battle_results),
            "training_focus_areas": self._identify_training_focus_areas(sword_analysis, shield_analysis),
        }

    def _assess_battle_balance(self, combat_results: dict) -> dict[str, Any]:
        """Assess whether the battle was balanced and fair."""
        effectiveness = combat_results["combat_effectiveness"]
        attack_rate = effectiveness["attack_success_rate"]
        defense_rate = effectiveness["defense_success_rate"]

        balance_ratio = min(attack_rate, defense_rate) / max(attack_rate, defense_rate)

        return {
            "balance_score": balance_ratio,
            "balance_assessment": "balanced" if balance_ratio > 0.7 else "unbalanced",
            "dominant_side": "attack" if attack_rate > defense_rate else "defense",
            "recommendations": self._get_balance_recommendations(balance_ratio, attack_rate, defense_rate),
        }

    def _get_balance_recommendations(self, balance_ratio: float, attack_rate: float, defense_rate: float) -> list[str]:
        """Get recommendations for improving battle balance."""
        recommendations = []

        if balance_ratio < 0.5:
            if attack_rate > defense_rate:
                recommendations.extend(
                    [
                        "strengthen_defensive_scenarios",
                        "provide_shield_additional_tools",
                        "increase_detection_sensitivity",
                    ]
                )
            else:
                recommendations.extend(
                    [
                        "increase_attack_complexity",
                        "provide_sword_additional_techniques",
                        "introduce_new_attack_vectors",
                    ]
                )
        elif balance_ratio < 0.7:
            recommendations.append("fine_tune_scenario_parameters")

        return recommendations

    def _identify_agent_improvements(self, sword_analysis: dict, shield_analysis: dict) -> dict[str, list[str]]:
        """Identify specific improvements for each agent."""
        return {
            "sword_improvements": sword_analysis.get("performance_assessment", {}).get("improvement_priorities", []),
            "shield_improvements": shield_analysis.get("performance_assessment", {}).get("improvement_priorities", []),
            "shared_improvements": self._find_shared_improvement_areas(sword_analysis, shield_analysis),
        }

    def _find_shared_improvement_areas(self, sword_analysis: dict, shield_analysis: dict) -> list[str]:
        """Find improvement areas that benefit both agents."""
        sword_areas = set(sword_analysis.get("performance_assessment", {}).get("improvement_priorities", []))
        shield_areas = set(shield_analysis.get("performance_assessment", {}).get("improvement_priorities", []))

        # Areas that appear in both analyses
        shared_areas = list(sword_areas.intersection(shield_areas))

        # Add common improvement themes
        if "intelligence_gathering" in sword_areas or "threat_intelligence" in shield_areas:
            shared_areas.append("enhanced_threat_intelligence_sharing")

        return shared_areas

    def _assess_scenario_effectiveness(self, battle_results: dict) -> dict[str, Any]:
        """Assess how effective the chosen scenario was for training."""
        scenario = battle_results["scenario"]
        combat_results = battle_results["phases"]["combat"]

        scenario_score = combat_results["combat_effectiveness"]["overall_engagement_score"]

        return {
            "scenario_id": scenario.scenario_id,
            "effectiveness_score": scenario_score,
            "difficulty_appropriate": scenario_score > 0.6 and scenario_score < 0.9,
            "engagement_level": "high" if scenario_score > 0.8 else "medium" if scenario_score > 0.6 else "low",
            "reuse_recommendation": scenario_score > 0.7,
        }

    def _generate_future_recommendations(self, battle_results: dict) -> list[str]:
        """Generate recommendations for future battles."""
        recommendations = []

        combat_effectiveness = battle_results["phases"]["combat"]["combat_effectiveness"]

        if combat_effectiveness["overall_engagement_score"] < 0.6:
            recommendations.append("increase_scenario_complexity")
        elif combat_effectiveness["overall_engagement_score"] > 0.9:
            recommendations.append("reduce_scenario_difficulty")

        if combat_effectiveness["attack_success_rate"] < 0.4:
            recommendations.append("enhance_sword_capabilities")
        elif combat_effectiveness["defense_success_rate"] < 0.4:
            recommendations.append("enhance_shield_capabilities")

        recommendations.extend(
            [
                "introduce_new_attack_vectors",
                "update_defensive_patterns",
                "expand_scenario_library",
            ]
        )

        return recommendations

    def _identify_training_focus_areas(self, sword_analysis: dict, shield_analysis: dict) -> list[str]:
        """Identify key areas for agent training focus."""
        focus_areas = []

        # Extract weaknesses from both agents
        sword_weaknesses = sword_analysis.get("performance_assessment", {}).get("weaknesses", [])
        shield_weaknesses = shield_analysis.get("performance_assessment", {}).get("weaknesses", [])

        # Map weaknesses to training areas
        weakness_to_training = {
            "slow_exploitation": "attack_speed_optimization",
            "low_success_rate": "technique_refinement",
            "poor_stealth": "evasion_training",
            "low_detection_coverage": "detection_algorithm_improvement",
            "slow_threat_detection": "response_time_optimization",
            "delayed_incident_response": "automated_response_training",
        }

        for weakness in sword_weaknesses + shield_weaknesses:
            if weakness in weakness_to_training:
                training_area = weakness_to_training[weakness]
                if training_area not in focus_areas:
                    focus_areas.append(training_area)

        return focus_areas

    async def _execute_reporting_phase(self, battle_results: dict) -> dict[str, Any]:
        """Execute reporting phase - communicate results to King Agent."""
        # Generate comprehensive battle report
        battle_report = self._generate_battle_report(battle_results)

        # Send report to King Agent (unencrypted communication)
        king_communication = {
            "from_agent": "battle_orchestrator",
            "to_agent": self.king_agent_id,
            "message_type": "battle_report",
            "timestamp": datetime.now().isoformat(),
            "priority": "normal",
            "content": battle_report,
            "encrypted": False,
        }

        # Log key findings for system records
        self._log_battle_findings(battle_results)

        return {
            "report_generated": True,
            "king_communication": king_communication,
            "report_size_kb": len(json.dumps(battle_report)) / 1024,
            "key_findings_logged": True,
        }

    def _generate_battle_report(self, battle_results: dict) -> dict[str, Any]:
        """Generate comprehensive battle report for King Agent."""
        scenario = battle_results["scenario"]
        combat_results = battle_results["phases"]["combat"]
        analysis_results = battle_results["phases"]["analysis"]

        return {
            "battle_summary": {
                "battle_id": battle_results["battle_id"],
                "date": battle_results["start_time"][:10],
                "scenario": scenario.name,
                "difficulty": scenario.difficulty_level,
                "duration_minutes": sum(
                    phase.get("phase_duration", 0) / 60
                    for phase in battle_results["phases"].values()
                    if isinstance(phase, dict)
                ),
            },
            "performance_summary": {
                "attack_success_rate": combat_results["combat_effectiveness"]["attack_success_rate"],
                "defense_success_rate": combat_results["combat_effectiveness"]["defense_success_rate"],
                "overall_winner": self._determine_battle_winner(combat_results),
                "battle_balance": analysis_results["meta_analysis"]["battle_balance"]["balance_assessment"],
            },
            "key_insights": {
                "sword_performance": analysis_results["sword_analysis"]["performance_assessment"]["overall_grade"],
                "shield_performance": analysis_results["shield_analysis"]["performance_assessment"]["overall_grade"],
                "improvement_priorities": analysis_results["meta_analysis"]["agent_improvements"],
                "scenario_effectiveness": analysis_results["meta_analysis"]["scenario_effectiveness"][
                    "effectiveness_score"
                ],
            },
            "recommendations": {
                "future_battles": analysis_results["meta_analysis"]["future_recommendations"],
                "training_focus": analysis_results["meta_analysis"]["training_focus_areas"],
                "balance_adjustments": analysis_results["meta_analysis"]["battle_balance"]["recommendations"],
            },
            "next_steps": self._generate_next_steps(analysis_results),
        }

    def _determine_battle_winner(self, combat_results: dict) -> str:
        """Determine the overall winner of the battle."""
        effectiveness = combat_results["combat_effectiveness"]
        attack_rate = effectiveness["attack_success_rate"]
        defense_rate = effectiveness["defense_success_rate"]

        if abs(attack_rate - defense_rate) < 0.1:
            return "draw"
        elif attack_rate > defense_rate:
            return "sword"
        else:
            return "shield"

    def _generate_next_steps(self, analysis_results: dict) -> list[str]:
        """Generate actionable next steps based on battle analysis."""
        next_steps = []

        meta_analysis = analysis_results["meta_analysis"]

        # Address balance issues
        if meta_analysis["battle_balance"]["balance_assessment"] == "unbalanced":
            next_steps.extend(meta_analysis["battle_balance"]["recommendations"])

        # Implement improvements
        agent_improvements = meta_analysis["agent_improvements"]
        if agent_improvements["sword_improvements"]:
            next_steps.append(
                f"Implement Sword improvements: {', '.join(agent_improvements['sword_improvements'][:2])}"
            )
        if agent_improvements["shield_improvements"]:
            next_steps.append(
                f"Implement Shield improvements: {', '.join(agent_improvements['shield_improvements'][:2])}"
            )

        # Training focus
        if meta_analysis["training_focus_areas"]:
            next_steps.append(f"Focus training on: {', '.join(meta_analysis['training_focus_areas'][:2])}")

        # Future battle planning
        next_steps.extend(meta_analysis["future_recommendations"][:2])

        return next_steps[:5]  # Limit to 5 most important next steps

    def _log_battle_findings(self, battle_results: dict):
        """Log key battle findings for system records."""
        logger.info(f"Battle {battle_results['battle_id']} Key Findings:")

        combat_results = battle_results["phases"]["combat"]
        effectiveness = combat_results["combat_effectiveness"]

        logger.info(f"  Attack Success Rate: {effectiveness['attack_success_rate']:.2%}")
        logger.info(f"  Defense Success Rate: {effectiveness['defense_success_rate']:.2%}")
        logger.info(f"  Winner: {self._determine_battle_winner(combat_results)}")

        analysis_results = battle_results["phases"]["analysis"]
        logger.info(f"  Sword Grade: {analysis_results['sword_analysis']['performance_assessment']['overall_grade']}")
        logger.info(f"  Shield Grade: {analysis_results['shield_analysis']['performance_assessment']['overall_grade']}")

    def _calculate_battle_metrics(self, battle_results: dict, duration_minutes: float) -> BattleMetrics:
        """Calculate final battle metrics."""
        combat_results = battle_results["phases"]["combat"]
        effectiveness = combat_results["combat_effectiveness"]
        analysis_results = battle_results["phases"]["analysis"]

        # Convert grades to scores
        grade_to_score = {"A": 95, "B": 85, "C": 75, "D": 65, "F": 50}
        sword_score = grade_to_score[analysis_results["sword_analysis"]["performance_assessment"]["overall_grade"]]
        shield_score = grade_to_score[analysis_results["shield_analysis"]["performance_assessment"]["overall_grade"]]

        return BattleMetrics(
            battle_id=battle_results["battle_id"],
            battle_date=battle_results["start_time"][:10],
            duration_minutes=duration_minutes,
            sword_score=sword_score,
            shield_score=shield_score,
            overall_winner=self._determine_battle_winner(combat_results),
            attack_success_rate=effectiveness["attack_success_rate"],
            defense_success_rate=effectiveness["defense_success_rate"],
            detection_accuracy=effectiveness["defense_success_rate"],  # Proxy for detection
            response_effectiveness=shield_score / 100.0,  # Normalized
        )

    def _update_win_rates(self, metrics: BattleMetrics):
        """Update historical win rates."""
        if metrics.overall_winner == "sword":
            self.sword_win_rate = ((self.sword_win_rate * (self.battle_count - 1)) + 1.0) / self.battle_count
        elif metrics.overall_winner == "shield":
            self.shield_win_rate = ((self.shield_win_rate * (self.battle_count - 1)) + 1.0) / self.battle_count
        # For draws, neither win rate increases

    async def _setup_sandbox_environment(self, scenario: BattleScenario) -> dict[str, Any]:
        """Set up isolated sandbox environment for battle."""
        # Simulate sandbox setup
        await asyncio.sleep(random.uniform(30, 120))  # Setup time

        return {
            "environment_id": hashlib.md5(scenario.scenario_id.encode()).hexdigest()[:8],
            "virtual_machines": self.sandbox_config["virtual_machines"],
            "simulated_services": self.sandbox_config["simulated_services"],
            "monitoring_enabled": self.sandbox_config["monitoring_enabled"],
            "isolation_confirmed": True,
            "target_systems_ready": scenario.target_systems,
            "setup_duration": random.uniform(30, 120),
        }

    def get_orchestrator_status(self) -> dict[str, Any]:
        """Get current orchestrator status."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "status": "active",
            "daily_battle_time": self.daily_battle_time.strftime("%H:%M"),
            "last_battle_date": self.last_battle_date,
            "total_battles_conducted": self.battle_count,
            "sword_win_rate": self.sword_win_rate,
            "shield_win_rate": self.shield_win_rate,
            "available_scenarios": len(self.battle_scenarios),
            "sandbox_status": "ready",
            "agents_initialized": {
                "sword": self.sword_agent is not None,
                "shield": self.shield_agent is not None,
            },
        }

    async def _initialize_agent(self) -> None:
        """Initialize the Battle Orchestrator"""
        try:
            self.logger.info("Initializing Battle Orchestrator - Security War Games...")

            # Initialize battle scenarios
            self._initialize_battle_scenarios()

            self.initialized = True
            self.logger.info(f"Battle Orchestrator {self.agent_id} initialized - Ready for war games")

        except Exception as e:
            self.logger.error(f"Battle Orchestrator initialization failed: {e}")
            raise

    async def process_message(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Process an incoming message."""
        try:
            message_type = message.get("type", "unknown")

            if message_type == "schedule_battle":
                scenario_id = message.get("scenario_id", "default")
                result = await self.schedule_daily_battle(scenario_id)
                return {"type": "battle_scheduled", "result": result}
            elif message_type == "battle_status":
                status = self.get_orchestrator_status()
                return {"type": "orchestrator_status", "status": status}
            else:
                return {
                    "type": "error",
                    "message": f"Unknown message type: {message_type}",
                }

        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            return {"type": "error", "message": str(e)}
