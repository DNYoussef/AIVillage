"""
Shield Agent - Defensive Security Specialist

Research-backed defensive security agent implementing:
- Intelligence at the Edge of Chaos for adaptive defense
- Quiet-STaR reasoning with encrypted thought bubbles
- Self-modeling for improved defensive predictions
- Daily mock battles against Sword Agent

Shield is cloned from Magi with specialized defensive security capabilities.
All thought bubbles are encrypted except when communicating with King Agent.
"""

import hashlib
import json
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .base_agent import BaseAgent


@dataclass
class DefensivePattern:
    """Represents a defensive security pattern or technique."""

    name: str
    category: str  # 'detection', 'prevention', 'response', 'forensics'
    effectiveness_score: float
    implementation_complexity: float
    false_positive_rate: float
    resource_requirements: dict[str, Any]
    countermeasures: list[str]


@dataclass
class ThreatIntelligence:
    """Threat intelligence data structure."""

    threat_id: str
    threat_type: str  # 'malware', 'apt', 'vulnerability', 'technique'
    severity: str  # 'low', 'medium', 'high', 'critical'
    indicators: list[str]
    mitigations: list[str]
    last_seen: datetime
    confidence: float


@dataclass
class SecurityIncident:
    """Security incident data structure."""

    incident_id: str
    incident_type: str
    severity: str
    detected_at: datetime
    source: str
    indicators: list[str]
    response_actions: list[str]
    status: str  # 'open', 'investigating', 'contained', 'resolved'


class ShieldAgent(BaseAgent):
    """
    Defensive Security Specialist Agent

    Shield focuses on:
    - Defensive security research and implementation
    - Threat detection and incident response
    - Security monitoring and forensics
    - Mock battle defense against Sword Agent
    """

    def __init__(self, agent_id: str = "shield", king_public_key: str | None = None):
        super().__init__(agent_id)

        # Initialize core reasoning systems (simplified for integration testing)
        self.quiet_star = None  # Placeholder - would be QuietSTaRReasoning
        self.thought_encryption = None  # Placeholder - would be ThoughtBubbleEncryption

        # Initialize belief system for defensive knowledge (simplified)
        self.belief_engine = None  # Placeholder - would be BayesianBeliefEngine

        # Defensive security databases
        self.defensive_patterns: dict[str, DefensivePattern] = {}
        self.threat_intelligence: dict[str, ThreatIntelligence] = {}
        self.security_incidents: dict[str, SecurityIncident] = {}

        # Battle tracking
        self.battle_history: list[dict] = []
        self.defense_strategies: dict[str, Any] = {}

        # Performance metrics
        self.detection_rate = 0.0
        self.false_positive_rate = 0.0
        self.response_time = 0.0

        # Initialize with core defensive patterns
        self._initialize_core_defenses()

        self.logger.info("Shield Agent initialized with defensive security capabilities")

    def _initialize_core_defenses(self):
        """Initialize core defensive security patterns."""
        core_patterns = [
            DefensivePattern(
                name="Anomaly-Based Intrusion Detection",
                category="detection",
                effectiveness_score=0.85,
                implementation_complexity=0.7,
                false_positive_rate=0.15,
                resource_requirements={
                    "cpu": "medium",
                    "memory": "high",
                    "network": "medium",
                },
                countermeasures=["traffic_normalization", "behavioral_mimicry"],
            ),
            DefensivePattern(
                name="Zero Trust Architecture",
                category="prevention",
                effectiveness_score=0.90,
                implementation_complexity=0.8,
                false_positive_rate=0.05,
                resource_requirements={
                    "cpu": "low",
                    "memory": "medium",
                    "network": "high",
                },
                countermeasures=["credential_stuffing", "lateral_movement"],
            ),
            DefensivePattern(
                name="Automated Incident Response",
                category="response",
                effectiveness_score=0.75,
                implementation_complexity=0.6,
                false_positive_rate=0.20,
                resource_requirements={
                    "cpu": "high",
                    "memory": "medium",
                    "network": "low",
                },
                countermeasures=["false_flag_attacks", "resource_exhaustion"],
            ),
            DefensivePattern(
                name="Memory Forensics Analysis",
                category="forensics",
                effectiveness_score=0.95,
                implementation_complexity=0.9,
                false_positive_rate=0.02,
                resource_requirements={
                    "cpu": "very_high",
                    "memory": "very_high",
                    "network": "low",
                },
                countermeasures=["memory_obfuscation", "rootkit_hiding"],
            ),
        ]

        for pattern in core_patterns:
            self.defensive_patterns[pattern.name] = pattern

        # Add beliefs about defensive effectiveness (placeholder for integration testing)
        if self.belief_engine:  # Only if belief engine is available
            for pattern in core_patterns:
                belief_id = f"effectiveness_{pattern.name.lower().replace(' ', '_')}"
                self.belief_engine.add_belief(
                    belief_id,
                    f"Defensive pattern {pattern.name} has {pattern.effectiveness_score:.0%} effectiveness",
                    pattern.effectiveness_score,
                )

    async def think_encrypted(self, prompt: str, context: dict = None) -> str:
        """
        Generate encrypted thought bubble using Quiet-STaR reasoning.
        Only King Agent can decrypt these thoughts.
        """
        # Simplified implementation for integration testing
        if self.quiet_star:
            # Generate reasoning with START/END tokens
            thought = await self.quiet_star.generate_reasoning(
                prompt, context or {}, reasoning_type="defensive_analysis"
            )
        else:
            # Placeholder implementation
            thought = f"DEFENSIVE_ANALYSIS: {prompt} (Simplified for testing)"

        # Encrypt thought bubble (King can decrypt)
        if self.thought_encryption:
            encrypted_thought = self.thought_encryption.encrypt_thought(thought)
        else:
            # Placeholder encryption
            import base64

            encrypted_thought = base64.b64encode(thought.encode()).decode()

        return encrypted_thought

    async def daily_defensive_research(self) -> dict[str, Any]:
        """
        Conduct daily defensive security research.
        Focus on new threats, defensive techniques, and countermeasures.
        """
        research_prompt = """
        <START>
        Daily defensive security research objectives:
        1. Analyze emerging threat landscape
        2. Research new defensive technologies
        3. Evaluate existing defense effectiveness
        4. Plan countermeasures for known attack vectors
        5. Update threat intelligence database

        Focus areas for today's research:
        - Advanced Persistent Threats (APTs)
        - Zero-day vulnerability defense
        - AI/ML-based attack detection
        - Quantum-resistant cryptography
        - Supply chain security
        <END>
        """

        # Generate encrypted research thoughts
        research_thoughts = await self.think_encrypted(
            research_prompt,
            {"date": datetime.now().isoformat(), "research_type": "daily_defensive"},
        )

        # Simulate defensive research activities
        research_results = {
            "research_date": datetime.now().isoformat(),
            "encrypted_thoughts": research_thoughts,
            "new_threats_analyzed": random.randint(5, 15),
            "defensive_patterns_updated": random.randint(2, 8),
            "countermeasures_developed": random.randint(3, 10),
            "effectiveness_improvements": {
                pattern: random.uniform(0.01, 0.05)
                for pattern in random.sample(list(self.defensive_patterns.keys()), 3)
            },
        }

        # Update defensive patterns with research insights
        for pattern_name, improvement in research_results["effectiveness_improvements"].items():
            if pattern_name in self.defensive_patterns:
                old_score = self.defensive_patterns[pattern_name].effectiveness_score
                new_score = min(0.99, old_score + improvement)
                self.defensive_patterns[pattern_name].effectiveness_score = new_score

                # Update belief in Bayesian engine
                belief_id = f"effectiveness_{pattern_name.lower().replace(' ', '_')}"
                self.belief_engine.update_belief_probability(belief_id, new_score)

        return research_results

    async def threat_detection_analysis(self, indicators: list[str]) -> dict[str, Any]:
        """
        Analyze potential threats using defensive intelligence.
        """
        detection_prompt = f"""
        <START>
        Threat detection analysis for indicators:
        {json.dumps(indicators, indent=2)}

        Defensive analysis objectives:
        1. Classify threat type and severity
        2. Identify attack patterns and TTPs
        3. Assess potential impact and scope
        4. Recommend containment measures
        5. Plan incident response actions

        Apply defensive reasoning to determine:
        - Is this a false positive?
        - What defensive patterns should activate?
        - What response priority should be assigned?
        - What forensic data should be collected?
        <END>
        """

        # Generate encrypted analysis
        analysis_thoughts = await self.think_encrypted(
            detection_prompt,
            {"indicators": indicators, "analysis_type": "threat_detection"},
        )

        # Simulate threat classification
        threat_types = [
            "malware",
            "apt",
            "insider_threat",
            "vulnerability_exploit",
            "reconnaissance",
        ]
        severities = ["low", "medium", "high", "critical"]

        threat_analysis = {
            "analysis_id": hashlib.md5(str(indicators).encode()).hexdigest()[:8],
            "threat_type": random.choice(threat_types),
            "severity": random.choice(severities),
            "confidence": random.uniform(0.6, 0.95),
            "false_positive_probability": random.uniform(0.05, 0.30),
            "encrypted_analysis": analysis_thoughts,
            "recommended_actions": self._generate_response_actions(),
            "forensic_priorities": self._generate_forensic_priorities(),
            "containment_measures": self._generate_containment_measures(),
        }

        # Create security incident if severity is medium or higher
        if threat_analysis["severity"] in ["medium", "high", "critical"]:
            incident = SecurityIncident(
                incident_id=threat_analysis["analysis_id"],
                incident_type=threat_analysis["threat_type"],
                severity=threat_analysis["severity"],
                detected_at=datetime.now(),
                source="shield_detection",
                indicators=indicators,
                response_actions=threat_analysis["recommended_actions"],
                status="open",
            )
            self.security_incidents[incident.incident_id] = incident

        return threat_analysis

    def _generate_response_actions(self) -> list[str]:
        """Generate incident response actions."""
        actions = [
            "isolate_affected_systems",
            "collect_forensic_evidence",
            "analyze_network_traffic",
            "review_system_logs",
            "update_detection_rules",
            "notify_stakeholders",
            "preserve_evidence_chain",
            "document_timeline",
        ]
        return random.sample(actions, random.randint(3, 6))

    def _generate_forensic_priorities(self) -> list[str]:
        """Generate forensic investigation priorities."""
        priorities = [
            "memory_dump_analysis",
            "disk_image_acquisition",
            "network_pcap_analysis",
            "registry_analysis",
            "file_system_timeline",
            "malware_reverse_engineering",
            "behavioral_analysis",
            "attribution_investigation",
        ]
        return random.sample(priorities, random.randint(2, 5))

    def _generate_containment_measures(self) -> list[str]:
        """Generate containment measures."""
        measures = [
            "network_segmentation",
            "access_revocation",
            "system_quarantine",
            "traffic_blocking",
            "account_suspension",
            "service_isolation",
            "communication_monitoring",
            "backup_verification",
        ]
        return random.sample(measures, random.randint(2, 4))

    async def prepare_battle_defense(self, sword_intelligence: dict = None) -> dict[str, Any]:
        """
        Prepare defensive strategies for mock battle against Sword Agent.
        """
        defense_prompt = f"""
        <START>
        Mock battle defense preparation against Sword Agent:

        Known Sword capabilities from intelligence:
        {json.dumps(sword_intelligence or {}, indent=2)}

        Defensive preparation objectives:
        1. Analyze Sword's likely attack vectors
        2. Prepare countermeasures for each vector
        3. Set up detection and monitoring systems
        4. Plan incident response procedures
        5. Establish defensive perimeter

        Key defensive considerations:
        - What are Sword's favorite attack techniques?
        - Which defensive patterns are most effective?
        - How to minimize false positives during battle?
        - What forensic evidence should be preserved?
        - How to maintain business continuity?
        <END>
        """

        # Generate encrypted defense strategy
        defense_thoughts = await self.think_encrypted(
            defense_prompt,
            {
                "sword_intelligence": sword_intelligence,
                "preparation_type": "battle_defense",
            },
        )

        # Select defensive patterns for battle
        battle_defenses = random.sample(list(self.defensive_patterns.keys()), min(4, len(self.defensive_patterns)))

        defense_strategy = {
            "strategy_id": hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8],
            "preparation_time": datetime.now().isoformat(),
            "encrypted_strategy": defense_thoughts,
            "active_defenses": battle_defenses,
            "detection_systems": [
                "anomaly_detection",
                "signature_matching",
                "behavioral_analysis",
                "threat_intelligence_correlation",
            ],
            "response_playbooks": [
                "immediate_containment",
                "evidence_preservation",
                "stakeholder_notification",
                "system_recovery",
            ],
            "success_metrics": {
                "detection_rate_target": 0.90,
                "false_positive_rate_limit": 0.10,
                "response_time_target": 300,  # 5 minutes
                "containment_time_target": 900,  # 15 minutes
            },
        }

        self.defense_strategies[defense_strategy["strategy_id"]] = defense_strategy
        return defense_strategy

    async def engage_battle_defense(self, attack_scenarios: list[dict]) -> dict[str, Any]:
        """
        Execute defensive responses during mock battle with Sword Agent.
        """
        battle_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]

        battle_results = {
            "battle_id": battle_id,
            "battle_start": datetime.now().isoformat(),
            "attack_scenarios_count": len(attack_scenarios),
            "detection_results": [],
            "response_actions": [],
            "battle_metrics": {},
        }

        for i, scenario in enumerate(attack_scenarios):
            # Simulate detection attempt
            detection_success = random.uniform(0, 1) < self.detection_rate + 0.1
            detection_time = random.uniform(30, 300)  # 30 seconds to 5 minutes

            if detection_success:
                # Generate defensive response
                response_prompt = f"""
                <START>
                Detected attack scenario in mock battle:
                {json.dumps(scenario, indent=2)}

                Immediate defensive response required:
                1. Classify attack type and severity
                2. Activate appropriate countermeasures
                3. Contain potential damage
                4. Preserve evidence for analysis
                5. Report to King Agent if critical
                <END>
                """

                response_thoughts = await self.think_encrypted(
                    response_prompt, {"scenario": scenario, "battle_id": battle_id}
                )

                # Execute response actions
                response_time = random.uniform(60, 600)  # 1-10 minutes
                containment_success = random.uniform(0, 1) < 0.85

                detection_result = {
                    "scenario_id": i,
                    "detected": True,
                    "detection_time": detection_time,
                    "response_time": response_time,
                    "containment_success": containment_success,
                    "encrypted_response": response_thoughts,
                }
            else:
                # Missed detection
                detection_result = {
                    "scenario_id": i,
                    "detected": False,
                    "missed_indicators": scenario.get("indicators", []),
                    "failure_reason": random.choice(
                        [
                            "false_negative",
                            "detection_gap",
                            "signature_evasion",
                            "timing_issue",
                        ]
                    ),
                }

            battle_results["detection_results"].append(detection_result)

        # Calculate battle performance metrics
        detections = [r for r in battle_results["detection_results"] if r["detected"]]
        detection_rate = len(detections) / len(attack_scenarios)
        avg_detection_time = sum(r["detection_time"] for r in detections) / max(len(detections), 1)
        avg_response_time = sum(r["response_time"] for r in detections if "response_time" in r) / max(
            len(detections), 1
        )

        battle_results["battle_metrics"] = {
            "detection_rate": detection_rate,
            "average_detection_time": avg_detection_time,
            "average_response_time": avg_response_time,
            "successful_containments": sum(1 for r in detections if r.get("containment_success")),
            "battle_duration": random.uniform(1800, 7200),  # 30 minutes to 2 hours
        }

        battle_results["battle_end"] = datetime.now().isoformat()

        # Update performance metrics
        self.detection_rate = 0.7 * self.detection_rate + 0.3 * detection_rate

        # Store battle history
        self.battle_history.append(battle_results)

        return battle_results

    async def post_battle_analysis(self, battle_results: dict) -> dict[str, Any]:
        """
        Conduct post-battle analysis to improve defensive capabilities.
        """
        analysis_prompt = f"""
        <START>
        Post-battle defensive analysis:

        Battle Performance:
        - Detection Rate: {battle_results["battle_metrics"]["detection_rate"]:.2%}
        - Average Detection Time: {battle_results["battle_metrics"]["average_detection_time"]:.1f}s
        - Average Response Time: {battle_results["battle_metrics"]["average_response_time"]:.1f}s

        Key questions for improvement:
        1. Which attack vectors were missed and why?
        2. How can detection accuracy be improved?
        3. What response procedures need optimization?
        4. Which defensive patterns performed best/worst?
        5. What intelligence should be gathered about Sword's tactics?

        Learning objectives:
        - Identify defensive gaps and vulnerabilities
        - Improve detection rule effectiveness
        - Optimize response time and accuracy
        - Develop better threat intelligence
        <END>
        """

        analysis_thoughts = await self.think_encrypted(
            analysis_prompt,
            {"battle_results": battle_results, "analysis_type": "post_battle"},
        )

        # Generate improvement recommendations
        missed_detections = [r for r in battle_results["detection_results"] if not r["detected"]]
        improvement_areas = []

        if len(missed_detections) > 2:
            improvement_areas.append("detection_coverage")
        if battle_results["battle_metrics"]["average_detection_time"] > 180:
            improvement_areas.append("detection_speed")
        if battle_results["battle_metrics"]["average_response_time"] > 300:
            improvement_areas.append("response_efficiency")

        analysis_results = {
            "analysis_id": hashlib.md5(str(battle_results).encode()).hexdigest()[:8],
            "battle_id": battle_results["battle_id"],
            "analysis_time": datetime.now().isoformat(),
            "encrypted_analysis": analysis_thoughts,
            "performance_assessment": {
                "overall_grade": self._calculate_performance_grade(battle_results["battle_metrics"]),
                "strengths": self._identify_strengths(battle_results),
                "weaknesses": self._identify_weaknesses(battle_results),
                "improvement_priorities": improvement_areas,
            },
            "tactical_intelligence": {
                "sword_attack_patterns": self._analyze_attack_patterns(battle_results),
                "effective_countermeasures": self._identify_effective_countermeasures(battle_results),
                "defensive_gaps": missed_detections,
            },
            "next_battle_preparations": self._plan_next_battle_improvements(),
        }

        # Update beliefs based on battle performance
        self._update_defensive_beliefs(analysis_results)

        return analysis_results

    def _calculate_performance_grade(self, metrics: dict) -> str:
        """Calculate overall performance grade."""
        detection_score = metrics["detection_rate"] * 40
        speed_score = max(0, (300 - metrics["average_detection_time"]) / 300 * 30)
        response_score = max(0, (600 - metrics["average_response_time"]) / 600 * 30)

        total_score = detection_score + speed_score + response_score

        if total_score >= 85:
            return "A"
        elif total_score >= 75:
            return "B"
        elif total_score >= 65:
            return "C"
        elif total_score >= 55:
            return "D"
        else:
            return "F"

    def _identify_strengths(self, battle_results: dict) -> list[str]:
        """Identify defensive strengths from battle."""
        strengths = []
        metrics = battle_results["battle_metrics"]

        if metrics["detection_rate"] > 0.8:
            strengths.append("high_detection_accuracy")
        if metrics["average_detection_time"] < 120:
            strengths.append("rapid_threat_detection")
        if metrics["average_response_time"] < 240:
            strengths.append("fast_incident_response")
        if metrics["successful_containments"] > len(battle_results["detection_results"]) * 0.8:
            strengths.append("effective_containment")

        return strengths

    def _identify_weaknesses(self, battle_results: dict) -> list[str]:
        """Identify defensive weaknesses from battle."""
        weaknesses = []
        metrics = battle_results["battle_metrics"]

        if metrics["detection_rate"] < 0.6:
            weaknesses.append("low_detection_coverage")
        if metrics["average_detection_time"] > 240:
            weaknesses.append("slow_threat_detection")
        if metrics["average_response_time"] > 480:
            weaknesses.append("delayed_incident_response")

        missed_count = len([r for r in battle_results["detection_results"] if not r["detected"]])
        if missed_count > len(battle_results["detection_results"]) * 0.3:
            weaknesses.append("significant_detection_gaps")

        return weaknesses

    def _analyze_attack_patterns(self, battle_results: dict) -> list[str]:
        """Analyze Sword Agent's attack patterns."""
        # Simulate pattern recognition
        patterns = [
            "multi_stage_attacks",
            "credential_harvesting",
            "lateral_movement",
            "data_exfiltration",
            "persistence_mechanisms",
            "evasion_techniques",
        ]
        return random.sample(patterns, random.randint(2, 4))

    def _identify_effective_countermeasures(self, battle_results: dict) -> list[str]:
        """Identify which defensive measures were most effective."""
        effective_measures = []

        for result in battle_results["detection_results"]:
            if result["detected"] and result.get("containment_success"):
                effective_measures.extend(["anomaly_detection", "behavioral_analysis", "network_monitoring"])

        return list(set(effective_measures))

    def _plan_next_battle_improvements(self) -> list[str]:
        """Plan improvements for next battle."""
        improvements = [
            "enhance_detection_rules",
            "improve_response_automation",
            "expand_threat_intelligence",
            "optimize_alert_correlation",
            "strengthen_incident_playbooks",
        ]
        return random.sample(improvements, random.randint(2, 4))

    def _update_defensive_beliefs(self, analysis_results: dict):
        """Update Bayesian beliefs based on battle analysis."""
        performance_grade = analysis_results["performance_assessment"]["overall_grade"]

        # Convert grade to probability update
        grade_to_prob = {"A": 0.9, "B": 0.8, "C": 0.7, "D": 0.6, "F": 0.5}
        performance_prob = grade_to_prob[performance_grade]

        # Update belief in defensive effectiveness
        self.belief_engine.update_belief_probability("defensive_effectiveness", performance_prob)

        # Update beliefs about specific defensive patterns
        for strength in analysis_results["performance_assessment"]["strengths"]:
            if strength in self.belief_engine.beliefs:
                current_prob = self.belief_engine.beliefs[strength].probability
                new_prob = min(0.95, current_prob + 0.05)
                self.belief_engine.update_belief_probability(strength, new_prob)

    async def communicate_with_king(self, message: str, priority: str = "normal") -> dict[str, Any]:
        """
        Communicate defensive intelligence to King Agent.
        These communications are unencrypted as specified.
        """
        communication = {
            "from_agent": "shield",
            "to_agent": "king",
            "timestamp": datetime.now().isoformat(),
            "priority": priority,
            "message_type": "defensive_intelligence",
            "content": message,
            "encrypted": False,  # King communications are unencrypted
        }

        logger.info(f"Shield communicating with King: {priority} priority")
        return communication

    def get_defensive_status(self) -> dict[str, Any]:
        """Get current defensive status and capabilities."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "status": "active",
            "defensive_patterns_count": len(self.defensive_patterns),
            "threat_intelligence_count": len(self.threat_intelligence),
            "active_incidents": len([i for i in self.security_incidents.values() if i.status != "resolved"]),
            "battle_history_count": len(self.battle_history),
            "performance_metrics": {
                "detection_rate": self.detection_rate,
                "false_positive_rate": self.false_positive_rate,
                "response_time": self.response_time,
            },
            "last_research": self.last_activity.get("daily_research"),
            "next_battle_readiness": len(self.defense_strategies) > 0,
        }

    async def _initialize_agent(self) -> None:
        """Initialize the Shield Agent"""
        try:
            logger.info("Initializing Shield Agent - Defensive Security Specialist...")

            # Initialize defensive patterns
            self._initialize_default_patterns()

            self.initialized = True
            logger.info(f"Shield Agent {self.agent_id} initialized - Ready for defensive operations")

        except Exception as e:
            logger.error(f"Shield Agent initialization failed: {e}")
            raise

    def _initialize_default_patterns(self):
        """Initialize default defensive patterns."""
        default_pattern = DefensivePattern(
            id="default_001",
            name="Basic Threat Detection",
            description="Standard threat detection and response pattern",
            attack_vectors=[
                AttackVector.WEB_APPLICATION,
                AttackVector.NETWORK_PENETRATION,
            ],
            detection_rules=[
                "Unusual network activity",
                "Failed authentication attempts",
            ],
            response_procedures=["Alert security team", "Block suspicious IP"],
            effectiveness_score=0.75,
            last_updated=datetime.now(),
        )
        self.defensive_patterns[default_pattern.id] = default_pattern

    async def process_message(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Process an incoming message."""
        try:
            message_type = message.get("type", "unknown")

            if message_type == "threat_analysis":
                indicators = message.get("indicators", [])
                analysis = await self.threat_detection_analysis(indicators)
                return {"type": "threat_response", "analysis": analysis}
            elif message_type == "battle_preparation":
                intel = message.get("intel", {})
                strategy = await self.prepare_battle_defense(intel)
                return {"type": "defense_strategy", "strategy": strategy}
            else:
                return {
                    "type": "error",
                    "message": f"Unknown message type: {message_type}",
                }

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {"type": "error", "message": str(e)}
