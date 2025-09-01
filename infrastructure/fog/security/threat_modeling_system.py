"""
Comprehensive Threat Modeling System for Federated Learning

This module provides advanced threat modeling capabilities for federated learning systems,
including attack scenario generation, risk assessment, and security recommendation engine.
Integrates with BetaNet infrastructure for distributed security analysis.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import secrets
import statistics
from datetime import datetime

logger = logging.getLogger(__name__)


class ThreatCategory(Enum):
    """Categories of security threats"""

    CONFIDENTIALITY = "confidentiality"
    INTEGRITY = "integrity"
    AVAILABILITY = "availability"
    AUTHENTICITY = "authenticity"
    NON_REPUDIATION = "non_repudiation"
    PRIVACY = "privacy"


class AttackVector(Enum):
    """Attack vector classifications"""

    NETWORK = "network"
    MODEL = "model"
    DATA = "data"
    SYSTEM = "system"
    SOCIAL = "social"
    PHYSICAL = "physical"
    CRYPTOGRAPHIC = "cryptographic"


class RiskLevel(Enum):
    """Risk level classifications"""

    VERY_LOW = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5


class MitigationStatus(Enum):
    """Status of threat mitigations"""

    NOT_IMPLEMENTED = "not_implemented"
    PARTIALLY_IMPLEMENTED = "partially_implemented"
    IMPLEMENTED = "implemented"
    VERIFIED = "verified"


@dataclass
class ThreatActor:
    """Definition of threat actors"""

    actor_id: str
    name: str
    motivation: str  # financial, espionage, disruption, etc.
    capabilities: List[str]  # technical skills and resources
    resources: str  # nation-state, organized crime, individual, etc.
    typical_targets: List[str]
    attack_patterns: List[str]
    sophistication_level: int  # 1-5 scale


@dataclass
class Asset:
    """System assets that need protection"""

    asset_id: str
    name: str
    description: str
    asset_type: str  # data, model, service, infrastructure
    criticality: RiskLevel
    confidentiality_req: RiskLevel
    integrity_req: RiskLevel
    availability_req: RiskLevel
    owners: List[str]
    dependencies: List[str]
    location: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Vulnerability:
    """Security vulnerabilities"""

    vuln_id: str
    name: str
    description: str
    affected_assets: List[str]
    cwe_id: Optional[str] = None  # Common Weakness Enumeration
    cvss_score: Optional[float] = None
    exploitability: RiskLevel = RiskLevel.MEDIUM
    impact: RiskLevel = RiskLevel.MEDIUM
    likelihood: float = 0.5  # 0.0 to 1.0
    discovery_date: float = field(default_factory=time.time)
    remediation_effort: str = "medium"  # low, medium, high
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreatScenario:
    """Complete threat scenario definition"""

    scenario_id: str
    name: str
    description: str
    threat_actor: str
    attack_vector: AttackVector
    threat_category: ThreatCategory
    target_assets: List[str]
    exploited_vulnerabilities: List[str]
    attack_steps: List[Dict[str, Any]]
    prerequisites: List[str]
    indicators_of_compromise: List[str]
    impact_assessment: Dict[str, Any]
    likelihood_score: float
    risk_score: float
    mitigations: List[str]
    residual_risk: float
    last_updated: float = field(default_factory=time.time)


@dataclass
class SecurityControl:
    """Security control/mitigation"""

    control_id: str
    name: str
    description: str
    control_type: str  # preventive, detective, corrective, compensating
    implementation_status: MitigationStatus
    effectiveness: float  # 0.0 to 1.0
    cost: str  # low, medium, high
    complexity: str  # low, medium, high
    applicable_threats: List[str]
    prerequisites: List[str]
    maintenance_required: bool
    testing_frequency: str  # continuous, daily, weekly, monthly, quarterly
    last_tested: Optional[float] = None
    test_results: Dict[str, Any] = field(default_factory=dict)


class FederatedThreatDatabase:
    """
    Database of known threats specific to federated learning
    """

    def __init__(self):
        self.threat_actors = self._initialize_threat_actors()
        self.common_vulnerabilities = self._initialize_vulnerabilities()
        self.attack_patterns = self._initialize_attack_patterns()

    def _initialize_threat_actors(self) -> Dict[str, ThreatActor]:
        """Initialize database of known threat actors"""
        return {
            "malicious_participant": ThreatActor(
                actor_id="malicious_participant",
                name="Malicious Federated Learning Participant",
                motivation="data_theft, model_manipulation, disruption",
                capabilities=["machine_learning", "data_poisoning", "model_inversion"],
                resources="individual_to_small_group",
                typical_targets=["training_data", "model_updates", "aggregated_models"],
                attack_patterns=["gradient_inversion", "membership_inference", "model_poisoning"],
                sophistication_level=3,
            ),
            "adversarial_coordinator": ThreatActor(
                actor_id="adversarial_coordinator",
                name="Compromised Coordination Server",
                motivation="espionage, data_harvesting",
                capabilities=["privileged_access", "aggregation_manipulation", "participant_impersonation"],
                resources="organized_group",
                typical_targets=["all_participant_data", "model_architecture", "training_metadata"],
                attack_patterns=["byzantine_attack", "eclipse_attack", "aggregation_manipulation"],
                sophistication_level=4,
            ),
            "nation_state": ThreatActor(
                actor_id="nation_state",
                name="Advanced Persistent Threat (Nation-State)",
                motivation="strategic_intelligence, economic_espionage",
                capabilities=["zero_day_exploits", "supply_chain_attacks", "advanced_cryptanalysis"],
                resources="nation_state",
                typical_targets=["critical_infrastructure_models", "economic_data", "citizen_data"],
                attack_patterns=["supply_chain_compromise", "advanced_model_extraction"],
                sophistication_level=5,
            ),
            "insider_threat": ThreatActor(
                actor_id="insider_threat",
                name="Malicious Insider",
                motivation="financial_gain, revenge, espionage",
                capabilities=["privileged_access", "system_knowledge", "social_engineering"],
                resources="individual",
                typical_targets=["sensitive_training_data", "model_parameters", "system_credentials"],
                attack_patterns=["data_exfiltration", "credential_abuse", "backdoor_insertion"],
                sophistication_level=3,
            ),
        }

    def _initialize_vulnerabilities(self) -> Dict[str, Vulnerability]:
        """Initialize common federated learning vulnerabilities"""
        return {
            "unencrypted_gradients": Vulnerability(
                vuln_id="FL-001",
                name="Unencrypted Gradient Transmission",
                description="Model gradients transmitted without encryption",
                affected_assets=["gradient_updates", "model_parameters"],
                exploitability=RiskLevel.HIGH,
                impact=RiskLevel.HIGH,
                likelihood=0.8,
                remediation_effort="medium",
            ),
            "weak_aggregation": Vulnerability(
                vuln_id="FL-002",
                name="Insufficient Aggregation Security",
                description="Lack of Byzantine-fault tolerance in aggregation",
                affected_assets=["aggregation_server", "global_model"],
                exploitability=RiskLevel.MEDIUM,
                impact=RiskLevel.CRITICAL,
                likelihood=0.6,
                remediation_effort="high",
            ),
            "participant_authentication": Vulnerability(
                vuln_id="FL-003",
                name="Weak Participant Authentication",
                description="Insufficient verification of participant identity",
                affected_assets=["participant_nodes", "training_network"],
                exploitability=RiskLevel.HIGH,
                impact=RiskLevel.HIGH,
                likelihood=0.7,
                remediation_effort="medium",
            ),
            "model_inversion_exposure": Vulnerability(
                vuln_id="FL-004",
                name="Model Inversion Attack Susceptibility",
                description="Model updates reveal training data information",
                affected_assets=["training_data", "participant_privacy"],
                exploitability=RiskLevel.MEDIUM,
                impact=RiskLevel.HIGH,
                likelihood=0.5,
                remediation_effort="high",
            ),
            "differential_privacy_bypass": Vulnerability(
                vuln_id="FL-005",
                name="Inadequate Privacy Protection",
                description="Insufficient differential privacy implementation",
                affected_assets=["individual_data_records", "privacy_guarantees"],
                exploitability=RiskLevel.LOW,
                impact=RiskLevel.CRITICAL,
                likelihood=0.3,
                remediation_effort="high",
            ),
        }

    def _initialize_attack_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize federated learning attack patterns"""
        return {
            "gradient_inversion": {
                "name": "Gradient Inversion Attack",
                "description": "Reconstruct training data from shared gradients",
                "steps": [
                    "Intercept gradient updates",
                    "Apply gradient inversion algorithm",
                    "Reconstruct approximate training samples",
                    "Extract sensitive information",
                ],
                "prerequisites": ["access_to_gradients", "computational_resources"],
                "indicators": ["unusual_gradient_analysis", "reconstruction_attempts"],
                "impact": {"confidentiality": "high", "privacy": "critical"},
            },
            "model_poisoning": {
                "name": "Model Poisoning Attack",
                "description": "Corrupt global model through malicious updates",
                "steps": [
                    "Join federated training as legitimate participant",
                    "Generate adversarial model updates",
                    "Submit poisoned gradients during training",
                    "Influence global model behavior",
                ],
                "prerequisites": ["participant_access", "adversarial_ml_knowledge"],
                "indicators": ["abnormal_gradient_patterns", "model_performance_degradation"],
                "impact": {"integrity": "critical", "availability": "medium"},
            },
            "membership_inference": {
                "name": "Membership Inference Attack",
                "description": "Determine if specific data was used in training",
                "steps": [
                    "Obtain access to trained model",
                    "Query model with target data samples",
                    "Analyze prediction confidence patterns",
                    "Infer membership of specific records",
                ],
                "prerequisites": ["model_access", "statistical_analysis_capability"],
                "indicators": ["systematic_model_querying", "confidence_analysis"],
                "impact": {"privacy": "high", "confidentiality": "medium"},
            },
            "byzantine_attack": {
                "name": "Byzantine Fault Attack",
                "description": "Coordinate malicious participants to disrupt consensus",
                "steps": [
                    "Establish multiple malicious participants",
                    "Coordinate to exceed Byzantine threshold",
                    "Submit conflicting or malicious updates",
                    "Disrupt aggregation process",
                ],
                "prerequisites": ["multiple_compromised_participants", "coordination_capability"],
                "indicators": ["coordinated_malicious_behavior", "consensus_disruption"],
                "impact": {"availability": "critical", "integrity": "high"},
            },
        }


class ThreatModelingEngine:
    """
    Core engine for threat modeling and risk assessment
    """

    def __init__(self, threat_db: FederatedThreatDatabase):
        self.threat_db = threat_db
        self.assets: Dict[str, Asset] = {}
        self.vulnerabilities: Dict[str, Vulnerability] = {}
        self.threat_scenarios: Dict[str, ThreatScenario] = {}
        self.security_controls: Dict[str, SecurityControl] = {}
        self.risk_appetite = {
            "confidentiality": RiskLevel.LOW,
            "integrity": RiskLevel.VERY_LOW,
            "availability": RiskLevel.MEDIUM,
            "privacy": RiskLevel.VERY_LOW,
        }

    async def register_asset(self, asset: Asset) -> str:
        """Register an asset for threat modeling"""
        self.assets[asset.asset_id] = asset
        logger.info(f"Registered asset: {asset.name} ({asset.asset_id})")

        # Automatically identify applicable vulnerabilities
        await self._identify_asset_vulnerabilities(asset)

        return asset.asset_id

    async def _identify_asset_vulnerabilities(self, asset: Asset):
        """Identify vulnerabilities applicable to an asset"""
        applicable_vulns = []

        # Check each known vulnerability for applicability
        for vuln_id, vuln in self.threat_db.common_vulnerabilities.items():
            if await self._is_vulnerability_applicable(asset, vuln):
                # Create asset-specific vulnerability instance
                asset_vuln = Vulnerability(
                    vuln_id=f"{asset.asset_id}_{vuln_id}",
                    name=f"{vuln.name} ({asset.name})",
                    description=vuln.description,
                    affected_assets=[asset.asset_id],
                    exploitability=vuln.exploitability,
                    impact=self._calculate_asset_impact(asset, vuln),
                    likelihood=self._calculate_asset_likelihood(asset, vuln),
                    remediation_effort=vuln.remediation_effort,
                )

                self.vulnerabilities[asset_vuln.vuln_id] = asset_vuln
                applicable_vulns.append(asset_vuln.vuln_id)

        logger.info(f"Identified {len(applicable_vulns)} vulnerabilities for asset {asset.name}")

    async def _is_vulnerability_applicable(self, asset: Asset, vuln: Vulnerability) -> bool:
        """Determine if vulnerability applies to asset"""
        # Check asset type compatibility
        if asset.asset_type in ["model", "gradient"] and "gradient" in vuln.name.lower():
            return True
        if asset.asset_type == "data" and "data" in vuln.name.lower():
            return True
        if asset.asset_type == "service" and "aggregation" in vuln.name.lower():
            return True
        if asset.asset_type == "infrastructure" and "authentication" in vuln.name.lower():
            return True

        return False

    def _calculate_asset_impact(self, asset: Asset, vuln: Vulnerability) -> RiskLevel:
        """Calculate impact level based on asset criticality"""
        base_impact = vuln.impact.value
        asset_criticality = asset.criticality.value

        # Adjust impact based on asset criticality
        adjusted_impact = min(5, max(1, base_impact + asset_criticality - 3))
        return RiskLevel(adjusted_impact)

    def _calculate_asset_likelihood(self, asset: Asset, vuln: Vulnerability) -> float:
        """Calculate likelihood based on asset exposure and vulnerability"""
        base_likelihood = vuln.likelihood

        # Adjust based on asset location and exposure
        if asset.location == "public":
            base_likelihood *= 1.5
        elif asset.location == "private":
            base_likelihood *= 0.7

        return min(1.0, base_likelihood)

    async def generate_threat_scenarios(self, asset_id: str) -> List[ThreatScenario]:
        """Generate threat scenarios for a specific asset"""
        if asset_id not in self.assets:
            raise ValueError(f"Asset {asset_id} not found")

        asset = self.assets[asset_id]
        scenarios = []

        # Find vulnerabilities affecting this asset
        asset_vulnerabilities = [v for v in self.vulnerabilities.values() if asset_id in v.affected_assets]

        # Generate scenarios for each threat actor and attack pattern combination
        for actor_id, actor in self.threat_db.threat_actors.items():
            for pattern_id, pattern in self.threat_db.attack_patterns.items():
                # Check if this combination is realistic
                if pattern_id in actor.attack_patterns:
                    scenario = await self._create_threat_scenario(
                        asset, actor, pattern_id, pattern, asset_vulnerabilities
                    )
                    scenarios.append(scenario)
                    self.threat_scenarios[scenario.scenario_id] = scenario

        logger.info(f"Generated {len(scenarios)} threat scenarios for asset {asset.name}")
        return scenarios

    async def _create_threat_scenario(
        self,
        asset: Asset,
        actor: ThreatActor,
        pattern_id: str,
        pattern: Dict[str, Any],
        vulnerabilities: List[Vulnerability],
    ) -> ThreatScenario:
        """Create detailed threat scenario"""
        scenario_id = f"TS_{asset.asset_id}_{actor.actor_id}_{pattern_id}_{secrets.token_hex(4)}"

        # Filter relevant vulnerabilities for this attack pattern
        relevant_vulns = [v for v in vulnerabilities if self._is_vulnerability_relevant_to_pattern(v, pattern)]

        # Calculate likelihood based on actor capability and vulnerability exploitability
        likelihood = self._calculate_scenario_likelihood(actor, relevant_vulns, asset)

        # Assess impact
        impact_assessment = self._assess_scenario_impact(asset, pattern, relevant_vulns)

        # Calculate overall risk score
        risk_score = likelihood * max(impact_assessment.values())

        scenario = ThreatScenario(
            scenario_id=scenario_id,
            name=f"{pattern['name']} against {asset.name} by {actor.name}",
            description=f"{pattern['description']} targeting {asset.name}",
            threat_actor=actor.actor_id,
            attack_vector=self._determine_attack_vector(pattern),
            threat_category=self._determine_threat_category(pattern),
            target_assets=[asset.asset_id],
            exploited_vulnerabilities=[v.vuln_id for v in relevant_vulns],
            attack_steps=self._expand_attack_steps(pattern["steps"], asset),
            prerequisites=pattern["prerequisites"],
            indicators_of_compromise=pattern["indicators"],
            impact_assessment=impact_assessment,
            likelihood_score=likelihood,
            risk_score=risk_score,
            mitigations=[],  # Will be populated by mitigation analysis
            residual_risk=risk_score,  # Initial residual risk equals full risk
        )

        return scenario

    def _is_vulnerability_relevant_to_pattern(self, vulnerability: Vulnerability, pattern: Dict[str, Any]) -> bool:
        """Check if vulnerability is relevant to attack pattern"""
        pattern_name = pattern["name"].lower()
        vuln_name = vulnerability.name.lower()

        # Simple keyword matching - in production, use more sophisticated matching
        if "gradient" in pattern_name and "gradient" in vuln_name:
            return True
        if "model" in pattern_name and ("model" in vuln_name or "aggregation" in vuln_name):
            return True
        if "inference" in pattern_name and "privacy" in vuln_name:
            return True

        return False

    def _calculate_scenario_likelihood(
        self, actor: ThreatActor, vulnerabilities: List[Vulnerability], asset: Asset
    ) -> float:
        """Calculate likelihood of threat scenario"""
        if not vulnerabilities:
            return 0.1  # Very low likelihood without exploitable vulnerabilities

        # Base likelihood from actor sophistication and motivation alignment
        actor_capability_factor = actor.sophistication_level / 5.0

        # Asset attractiveness factor
        asset_value_factor = asset.criticality.value / 5.0

        # Vulnerability exploitability factor
        avg_exploitability = statistics.mean([v.exploitability.value for v in vulnerabilities]) / 5.0

        # Combine factors
        likelihood = actor_capability_factor * 0.4 + asset_value_factor * 0.3 + avg_exploitability * 0.3

        return min(1.0, likelihood)

    def _assess_scenario_impact(
        self, asset: Asset, pattern: Dict[str, Any], vulnerabilities: List[Vulnerability]
    ) -> Dict[str, float]:
        """Assess impact of threat scenario"""
        impact_assessment = {"confidentiality": 0.0, "integrity": 0.0, "availability": 0.0, "privacy": 0.0}

        # Extract impact from attack pattern
        pattern_impact = pattern.get("impact", {})
        for category, level in pattern_impact.items():
            if category in impact_assessment:
                # Convert level to numeric score
                if level == "critical":
                    impact_assessment[category] = 1.0
                elif level == "high":
                    impact_assessment[category] = 0.8
                elif level == "medium":
                    impact_assessment[category] = 0.5
                elif level == "low":
                    impact_assessment[category] = 0.2

        # Adjust based on asset requirements
        if hasattr(asset, "confidentiality_req"):
            impact_assessment["confidentiality"] *= asset.confidentiality_req.value / 5.0
        if hasattr(asset, "integrity_req"):
            impact_assessment["integrity"] *= asset.integrity_req.value / 5.0
        if hasattr(asset, "availability_req"):
            impact_assessment["availability"] *= asset.availability_req.value / 5.0

        return impact_assessment

    def _determine_attack_vector(self, pattern: Dict[str, Any]) -> AttackVector:
        """Determine primary attack vector from pattern"""
        pattern_name = pattern["name"].lower()

        if "gradient" in pattern_name or "model" in pattern_name:
            return AttackVector.MODEL
        elif "membership" in pattern_name or "privacy" in pattern_name:
            return AttackVector.DATA
        elif "byzantine" in pattern_name:
            return AttackVector.NETWORK
        else:
            return AttackVector.SYSTEM

    def _determine_threat_category(self, pattern: Dict[str, Any]) -> ThreatCategory:
        """Determine primary threat category from pattern"""
        impact = pattern.get("impact", {})

        if "confidentiality" in impact and impact["confidentiality"] in ["high", "critical"]:
            return ThreatCategory.CONFIDENTIALITY
        elif "integrity" in impact and impact["integrity"] in ["high", "critical"]:
            return ThreatCategory.INTEGRITY
        elif "privacy" in impact and impact["privacy"] in ["high", "critical"]:
            return ThreatCategory.PRIVACY
        elif "availability" in impact and impact["availability"] in ["high", "critical"]:
            return ThreatCategory.AVAILABILITY
        else:
            return ThreatCategory.AUTHENTICITY

    def _expand_attack_steps(self, base_steps: List[str], asset: Asset) -> List[Dict[str, Any]]:
        """Expand attack steps with asset-specific details"""
        expanded_steps = []

        for i, step in enumerate(base_steps):
            expanded_step = {
                "step_number": i + 1,
                "description": step,
                "target_asset": asset.asset_id,
                "required_access": "network" if i == 0 else "elevated",
                "detection_difficulty": "medium",
                "mitigation_opportunities": [],
            }
            expanded_steps.append(expanded_step)

        return expanded_steps

    async def recommend_mitigations(self, scenario_id: str) -> List[SecurityControl]:
        """Recommend security controls for a threat scenario"""
        if scenario_id not in self.threat_scenarios:
            raise ValueError(f"Threat scenario {scenario_id} not found")

        scenario = self.threat_scenarios[scenario_id]
        recommendations = []

        # Get relevant security controls
        control_candidates = self._get_applicable_controls(scenario)

        # Prioritize controls by effectiveness and feasibility
        prioritized_controls = self._prioritize_controls(control_candidates, scenario)

        # Select top recommendations
        for control in prioritized_controls[:5]:  # Top 5 recommendations
            recommendations.append(control)

        # Update scenario with mitigation recommendations
        scenario.mitigations = [c.control_id for c in recommendations]
        scenario.residual_risk = self._calculate_residual_risk(scenario, recommendations)

        logger.info(f"Generated {len(recommendations)} mitigation recommendations for {scenario.name}")
        return recommendations

    def _get_applicable_controls(self, scenario: ThreatScenario) -> List[SecurityControl]:
        """Get security controls applicable to threat scenario"""
        # Define standard security controls for federated learning
        standard_controls = [
            SecurityControl(
                control_id="FL_CTRL_001",
                name="End-to-End Gradient Encryption",
                description="Encrypt gradient updates during transmission and storage",
                control_type="preventive",
                implementation_status=MitigationStatus.NOT_IMPLEMENTED,
                effectiveness=0.9,
                cost="medium",
                complexity="medium",
                applicable_threats=["gradient_inversion", "data_reconstruction"],
                prerequisites=["cryptographic_infrastructure"],
                maintenance_required=True,
                testing_frequency="weekly",
            ),
            SecurityControl(
                control_id="FL_CTRL_002",
                name="Byzantine-Fault Tolerant Aggregation",
                description="Implement robust aggregation resistant to Byzantine attacks",
                control_type="preventive",
                implementation_status=MitigationStatus.NOT_IMPLEMENTED,
                effectiveness=0.8,
                cost="high",
                complexity="high",
                applicable_threats=["model_poisoning", "byzantine_attack"],
                prerequisites=["consensus_algorithm", "threshold_cryptography"],
                maintenance_required=True,
                testing_frequency="daily",
            ),
            SecurityControl(
                control_id="FL_CTRL_003",
                name="Differential Privacy Implementation",
                description="Add calibrated noise to protect individual privacy",
                control_type="preventive",
                implementation_status=MitigationStatus.NOT_IMPLEMENTED,
                effectiveness=0.7,
                cost="medium",
                complexity="high",
                applicable_threats=["membership_inference", "privacy_breach"],
                prerequisites=["privacy_budget_allocation"],
                maintenance_required=True,
                testing_frequency="monthly",
            ),
            SecurityControl(
                control_id="FL_CTRL_004",
                name="Multi-Factor Participant Authentication",
                description="Strong authentication for federated learning participants",
                control_type="preventive",
                implementation_status=MitigationStatus.NOT_IMPLEMENTED,
                effectiveness=0.85,
                cost="low",
                complexity="medium",
                applicable_threats=["impersonation", "unauthorized_participation"],
                prerequisites=["identity_management_system"],
                maintenance_required=False,
                testing_frequency="quarterly",
            ),
            SecurityControl(
                control_id="FL_CTRL_005",
                name="Continuous Security Monitoring",
                description="Real-time monitoring of federated learning activities",
                control_type="detective",
                implementation_status=MitigationStatus.NOT_IMPLEMENTED,
                effectiveness=0.6,
                cost="medium",
                complexity="medium",
                applicable_threats=["anomaly_detection", "attack_early_warning"],
                prerequisites=["monitoring_infrastructure"],
                maintenance_required=True,
                testing_frequency="continuous",
            ),
        ]

        # Filter controls applicable to this scenario
        applicable_controls = []
        attack_pattern = scenario.name.lower()

        for control in standard_controls:
            if any(threat in attack_pattern for threat in control.applicable_threats):
                applicable_controls.append(control)

        return applicable_controls

    def _prioritize_controls(self, controls: List[SecurityControl], scenario: ThreatScenario) -> List[SecurityControl]:
        """Prioritize security controls by effectiveness and feasibility"""

        def calculate_priority_score(control: SecurityControl) -> float:
            # Effectiveness weight (0.5)
            effectiveness_score = control.effectiveness * 0.5

            # Cost consideration (0.2) - lower cost is better
            cost_scores = {"low": 1.0, "medium": 0.7, "high": 0.4}
            cost_score = cost_scores.get(control.cost, 0.5) * 0.2

            # Complexity consideration (0.2) - lower complexity is better
            complexity_scores = {"low": 1.0, "medium": 0.7, "high": 0.4}
            complexity_score = complexity_scores.get(control.complexity, 0.5) * 0.2

            # Implementation status bonus (0.1)
            status_bonus = 0.0
            if control.implementation_status == MitigationStatus.PARTIALLY_IMPLEMENTED:
                status_bonus = 0.05
            elif control.implementation_status == MitigationStatus.IMPLEMENTED:
                status_bonus = 0.1

            return effectiveness_score + cost_score + complexity_score + status_bonus

        # Sort by priority score (descending)
        prioritized = sorted(controls, key=calculate_priority_score, reverse=True)
        return prioritized

    def _calculate_residual_risk(self, scenario: ThreatScenario, mitigations: List[SecurityControl]) -> float:
        """Calculate residual risk after applying mitigations"""
        if not mitigations:
            return scenario.risk_score

        # Calculate combined effectiveness of mitigations
        # Use formula: combined = 1 - (1 - eff1) * (1 - eff2) * ... * (1 - effN)
        combined_effectiveness = 1.0
        for mitigation in mitigations:
            combined_effectiveness *= 1.0 - mitigation.effectiveness

        combined_effectiveness = 1.0 - combined_effectiveness

        # Apply to original risk
        residual_risk = scenario.risk_score * (1.0 - combined_effectiveness)
        return max(0.0, residual_risk)

    async def generate_threat_model_report(self, scope: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate comprehensive threat modeling report"""
        # Filter scope if specified
        if scope:
            relevant_scenarios = {
                k: v for k, v in self.threat_scenarios.items() if any(asset_id in scope for asset_id in v.target_assets)
            }
        else:
            relevant_scenarios = self.threat_scenarios

        # Risk analysis
        risk_summary = self._analyze_risk_landscape(relevant_scenarios)

        # Vulnerability analysis
        vuln_summary = self._analyze_vulnerabilities()

        # Mitigation coverage
        coverage_analysis = self._analyze_mitigation_coverage(relevant_scenarios)

        # Recommendations
        recommendations = await self._generate_strategic_recommendations(relevant_scenarios)

        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "scope": scope or "all_assets",
                "total_scenarios": len(relevant_scenarios),
                "total_assets": len(self.assets),
                "total_vulnerabilities": len(self.vulnerabilities),
            },
            "executive_summary": {
                "overall_risk_level": risk_summary["overall_risk_level"],
                "high_risk_scenarios": risk_summary["high_risk_count"],
                "critical_vulnerabilities": vuln_summary["critical_count"],
                "mitigation_coverage": coverage_analysis["overall_coverage"],
                "key_recommendations": recommendations[:3],
            },
            "risk_analysis": risk_summary,
            "vulnerability_analysis": vuln_summary,
            "threat_scenarios": [
                {
                    "scenario_id": scenario.scenario_id,
                    "name": scenario.name,
                    "risk_score": scenario.risk_score,
                    "likelihood": scenario.likelihood_score,
                    "impact_summary": max(scenario.impact_assessment.values()),
                    "mitigation_count": len(scenario.mitigations),
                    "residual_risk": scenario.residual_risk,
                }
                for scenario in relevant_scenarios.values()
            ],
            "mitigation_analysis": coverage_analysis,
            "recommendations": recommendations,
        }

        return report

    def _analyze_risk_landscape(self, scenarios: Dict[str, ThreatScenario]) -> Dict[str, Any]:
        """Analyze overall risk landscape"""
        if not scenarios:
            return {"overall_risk_level": "VERY_LOW", "high_risk_count": 0}

        risk_scores = [s.risk_score for s in scenarios.values()]
        residual_risks = [s.residual_risk for s in scenarios.values()]

        # Risk distribution
        high_risk_count = len([r for r in risk_scores if r >= 0.7])
        medium_risk_count = len([r for r in risk_scores if 0.4 <= r < 0.7])
        low_risk_count = len([r for r in risk_scores if r < 0.4])

        # Overall risk level
        avg_risk = statistics.mean(risk_scores)
        max_risk = max(risk_scores)

        if max_risk >= 0.8 or avg_risk >= 0.6:
            overall_level = "CRITICAL"
        elif max_risk >= 0.6 or avg_risk >= 0.4:
            overall_level = "HIGH"
        elif max_risk >= 0.4 or avg_risk >= 0.2:
            overall_level = "MEDIUM"
        else:
            overall_level = "LOW"

        return {
            "overall_risk_level": overall_level,
            "average_risk_score": avg_risk,
            "maximum_risk_score": max_risk,
            "high_risk_count": high_risk_count,
            "medium_risk_count": medium_risk_count,
            "low_risk_count": low_risk_count,
            "risk_distribution": {"high": high_risk_count, "medium": medium_risk_count, "low": low_risk_count},
            "mitigation_impact": {
                "average_original_risk": avg_risk,
                "average_residual_risk": statistics.mean(residual_risks),
                "risk_reduction": avg_risk - statistics.mean(residual_risks),
            },
        }

    def _analyze_vulnerabilities(self) -> Dict[str, Any]:
        """Analyze vulnerability landscape"""
        if not self.vulnerabilities:
            return {"critical_count": 0, "high_count": 0}

        vulns = list(self.vulnerabilities.values())

        # Count by impact level
        critical_count = len([v for v in vulns if v.impact == RiskLevel.CRITICAL])
        high_count = len([v for v in vulns if v.impact == RiskLevel.HIGH])
        medium_count = len([v for v in vulns if v.impact == RiskLevel.MEDIUM])
        low_count = len([v for v in vulns if v.impact in [RiskLevel.LOW, RiskLevel.VERY_LOW]])

        # Exploitability analysis
        highly_exploitable = len([v for v in vulns if v.exploitability in [RiskLevel.HIGH, RiskLevel.CRITICAL]])

        return {
            "total_vulnerabilities": len(vulns),
            "critical_count": critical_count,
            "high_count": high_count,
            "medium_count": medium_count,
            "low_count": low_count,
            "highly_exploitable": highly_exploitable,
            "impact_distribution": {
                "critical": critical_count,
                "high": high_count,
                "medium": medium_count,
                "low": low_count,
            },
        }

    def _analyze_mitigation_coverage(self, scenarios: Dict[str, ThreatScenario]) -> Dict[str, Any]:
        """Analyze mitigation coverage across scenarios"""
        if not scenarios:
            return {"overall_coverage": 0.0}

        # Calculate coverage statistics
        total_scenarios = len(scenarios)
        scenarios_with_mitigations = len([s for s in scenarios.values() if s.mitigations])

        coverage_percentage = scenarios_with_mitigations / total_scenarios if total_scenarios > 0 else 0.0

        # Risk reduction analysis
        original_risks = [s.risk_score for s in scenarios.values()]
        residual_risks = [s.residual_risk for s in scenarios.values()]

        avg_risk_reduction = statistics.mean(original_risks) - statistics.mean(residual_risks)

        return {
            "overall_coverage": coverage_percentage,
            "scenarios_with_mitigations": scenarios_with_mitigations,
            "total_scenarios": total_scenarios,
            "average_risk_reduction": avg_risk_reduction,
            "mitigation_effectiveness": {
                "high": len([s for s in scenarios.values() if s.risk_score - s.residual_risk > 0.3]),
                "medium": len([s for s in scenarios.values() if 0.1 <= s.risk_score - s.residual_risk <= 0.3]),
                "low": len([s for s in scenarios.values() if s.risk_score - s.residual_risk < 0.1]),
            },
        }

    async def _generate_strategic_recommendations(self, scenarios: Dict[str, ThreatScenario]) -> List[Dict[str, Any]]:
        """Generate strategic security recommendations"""
        recommendations = []

        # Analyze risk patterns
        high_risk_scenarios = [s for s in scenarios.values() if s.risk_score >= 0.6]

        # Recommendation 1: Address highest risks first
        if high_risk_scenarios:
            recommendations.append(
                {
                    "priority": "HIGH",
                    "category": "Risk Management",
                    "title": "Address Critical Risk Scenarios",
                    "description": f"Implement immediate mitigations for {len(high_risk_scenarios)} high-risk scenarios",
                    "impact": "Significant risk reduction",
                    "effort": "High",
                    "timeline": "30-60 days",
                }
            )

        # Recommendation 2: Strengthen authentication
        auth_scenarios = [s for s in scenarios.values() if "authentication" in s.name.lower()]
        if auth_scenarios:
            recommendations.append(
                {
                    "priority": "MEDIUM",
                    "category": "Access Control",
                    "title": "Implement Strong Authentication",
                    "description": "Deploy multi-factor authentication for all federated participants",
                    "impact": "Prevent unauthorized access",
                    "effort": "Medium",
                    "timeline": "15-30 days",
                }
            )

        # Recommendation 3: Privacy protection
        privacy_scenarios = [
            s for s in scenarios.values() if "privacy" in s.name.lower() or "inference" in s.name.lower()
        ]
        if privacy_scenarios:
            recommendations.append(
                {
                    "priority": "HIGH",
                    "category": "Privacy Protection",
                    "title": "Implement Differential Privacy",
                    "description": "Add differential privacy mechanisms to protect participant data",
                    "impact": "Strong privacy guarantees",
                    "effort": "High",
                    "timeline": "60-90 days",
                }
            )

        # Recommendation 4: Monitoring and detection
        recommendations.append(
            {
                "priority": "MEDIUM",
                "category": "Security Operations",
                "title": "Deploy Security Monitoring",
                "description": "Implement continuous monitoring for attack detection",
                "impact": "Early threat detection",
                "effort": "Medium",
                "timeline": "30-45 days",
            }
        )

        # Recommendation 5: Regular security assessment
        recommendations.append(
            {
                "priority": "LOW",
                "category": "Governance",
                "title": "Regular Security Reviews",
                "description": "Establish periodic threat model reviews and updates",
                "impact": "Maintain security posture",
                "effort": "Low",
                "timeline": "Ongoing",
            }
        )

        return recommendations


# Factory function for creating threat modeling system
def create_threat_modeling_system() -> ThreatModelingEngine:
    """
    Factory function to create threat modeling system

    Returns:
        Configured threat modeling engine
    """
    threat_db = FederatedThreatDatabase()
    engine = ThreatModelingEngine(threat_db)

    logger.info("Created threat modeling system with federated learning threat database")

    return engine
