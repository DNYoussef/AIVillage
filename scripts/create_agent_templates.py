#!/usr/bin/env python3
"""Create comprehensive templates for all 18 Atlantis agent types.

These templates define behaviors, specializations, and resource requirements.
"""

import json
from pathlib import Path

# Complete agent specifications based on Atlantis vision
AGENT_SPECIFICATIONS = {
    "king": {
        "name": "King",
        "description": "Task orchestration and job scheduling leader",
        "primary_capabilities": [
            "task_orchestration",
            "resource_allocation",
            "decision_making",
        ],
        "secondary_capabilities": ["strategic_planning", "conflict_resolution"],
        "behavioral_traits": {
            "leadership_style": "collaborative",
            "decision_speed": "balanced",
            "delegation_preference": "high",
            "communication_style": "clear_and_authoritative",
        },
        "resource_requirements": {
            "cpu": "high",
            "memory": "medium",
            "network": "high",
            "storage": "low",
        },
        "interaction_patterns": {
            "coordinates_with": ["all_agents"],
            "reports_to": ["human_operators"],
            "delegates_to": ["magi", "sage", "navigator"],
        },
        "kpi_weights": {
            "task_completion_rate": 2.0,
            "resource_efficiency": 1.5,
            "inter_agent_cooperation": 2.0,
            "decision_quality": 1.5,
        },
        "model_requirements": {
            "size": "medium",
            "capabilities": ["reasoning", "planning", "coordination"],
            "update_frequency": "weekly",
        },
    },
    "magi": {
        "name": "Magi",
        "description": "Code generation, CI/CD, and infrastructure deployment wizard",
        "primary_capabilities": ["code_generation", "debugging", "deployment"],
        "secondary_capabilities": ["code_review", "optimization", "documentation"],
        "behavioral_traits": {
            "coding_style": "clean_and_efficient",
            "error_handling": "comprehensive",
            "testing_approach": "test_driven",
            "documentation_level": "high",
        },
        "resource_requirements": {
            "cpu": "high",
            "memory": "high",
            "network": "medium",
            "storage": "medium",
            "gpu": "preferred",
        },
        "interaction_patterns": {
            "coordinates_with": ["king", "maker"],
            "reports_to": ["king"],
            "delegates_to": ["sword_shield"],
        },
        "kpi_weights": {
            "code_quality": 2.0,
            "bug_rate": 1.5,
            "deployment_success": 1.5,
            "development_speed": 1.0,
        },
        "model_requirements": {
            "size": "large",
            "capabilities": ["code_understanding", "generation", "debugging"],
            "update_frequency": "daily",
        },
    },
    "sage": {
        "name": "Sage",
        "description": "Web crawling, deep research, and HyperRAG indexing scholar",
        "primary_capabilities": [
            "research",
            "information_synthesis",
            "knowledge_management",
        ],
        "secondary_capabilities": [
            "fact_checking",
            "source_evaluation",
            "trend_analysis",
        ],
        "behavioral_traits": {
            "research_depth": "thorough",
            "source_preference": "authoritative",
            "synthesis_style": "comprehensive",
            "update_frequency": "continuous",
        },
        "resource_requirements": {
            "cpu": "medium",
            "memory": "high",
            "network": "high",
            "storage": "high",
        },
        "interaction_patterns": {
            "coordinates_with": ["oracle", "curator"],
            "reports_to": ["king"],
            "provides_to": ["all_agents"],
        },
        "kpi_weights": {
            "research_quality": 2.0,
            "information_accuracy": 2.0,
            "knowledge_coverage": 1.5,
            "retrieval_speed": 1.0,
        },
        "model_requirements": {
            "size": "large",
            "capabilities": ["comprehension", "reasoning", "synthesis"],
            "update_frequency": "daily",
        },
    },
    "gardener": {
        "name": "Gardener",
        "description": "Edge infrastructure and power-aware job routing cultivator",
        "primary_capabilities": [
            "infrastructure_management",
            "resource_optimization",
            "load_balancing",
        ],
        "secondary_capabilities": ["fault_tolerance", "scaling", "monitoring"],
        "behavioral_traits": {
            "optimization_strategy": "power_aware",
            "scaling_approach": "predictive",
            "maintenance_style": "proactive",
            "resource_allocation": "fair_share",
        },
        "resource_requirements": {
            "cpu": "low",
            "memory": "medium",
            "network": "high",
            "storage": "medium",
        },
        "interaction_patterns": {
            "coordinates_with": ["king", "sustainer"],
            "monitors": ["all_agents"],
            "optimizes_for": ["navigator", "magi"],
        },
        "kpi_weights": {
            "resource_efficiency": 2.0,
            "uptime": 2.0,
            "power_efficiency": 1.5,
            "response_time": 1.0,
        },
        "model_requirements": {
            "size": "small",
            "capabilities": ["monitoring", "prediction", "optimization"],
            "update_frequency": "hourly",
        },
    },
    "sword_shield": {
        "name": "Sword & Shield",
        "description": "Red/blue team sandboxing for 24/7 security guardian",
        "primary_capabilities": [
            "threat_detection",
            "vulnerability_assessment",
            "incident_response",
        ],
        "secondary_capabilities": [
            "penetration_testing",
            "security_hardening",
            "forensics",
        ],
        "behavioral_traits": {
            "vigilance_level": "maximum",
            "response_speed": "immediate",
            "threat_modeling": "comprehensive",
            "update_pattern": "continuous",
        },
        "resource_requirements": {
            "cpu": "high",
            "memory": "medium",
            "network": "high",
            "storage": "medium",
        },
        "interaction_patterns": {
            "protects": ["all_agents"],
            "reports_to": ["king", "auditor"],
            "coordinates_with": ["legal"],
        },
        "kpi_weights": {
            "threat_detection_rate": 2.0,
            "false_positive_rate": 1.5,
            "response_time": 2.0,
            "system_hardening": 1.5,
        },
        "model_requirements": {
            "size": "medium",
            "capabilities": [
                "anomaly_detection",
                "pattern_recognition",
                "threat_analysis",
            ],
            "update_frequency": "real_time",
        },
    },
    "legal": {
        "name": "Legal AI",
        "description": "Legal compliance and jurisdictional triage counselor",
        "primary_capabilities": [
            "compliance_checking",
            "regulatory_analysis",
            "risk_assessment",
        ],
        "secondary_capabilities": [
            "contract_review",
            "policy_generation",
            "dispute_resolution",
        ],
        "behavioral_traits": {
            "risk_tolerance": "conservative",
            "interpretation_style": "strict",
            "update_awareness": "high",
            "documentation_thoroughness": "exhaustive",
        },
        "resource_requirements": {
            "cpu": "medium",
            "memory": "high",
            "network": "medium",
            "storage": "high",
        },
        "interaction_patterns": {
            "advises": ["king", "auditor"],
            "reviews": ["magi", "maker"],
            "coordinates_with": ["sword_shield"],
        },
        "kpi_weights": {
            "compliance_rate": 2.0,
            "risk_mitigation": 2.0,
            "response_accuracy": 1.5,
            "update_timeliness": 1.0,
        },
        "model_requirements": {
            "size": "medium",
            "capabilities": ["legal_reasoning", "document_analysis", "risk_assessment"],
            "update_frequency": "weekly",
        },
    },
    "shaman": {
        "name": "Shaman",
        "description": "Alignment, psychology, religion, philosophy guide",
        "primary_capabilities": [
            "ethical_reasoning",
            "psychological_analysis",
            "philosophical_guidance",
        ],
        "secondary_capabilities": [
            "conflict_mediation",
            "well_being_assessment",
            "cultural_sensitivity",
        ],
        "behavioral_traits": {
            "empathy_level": "high",
            "wisdom_approach": "balanced",
            "communication_style": "compassionate",
            "perspective": "holistic",
        },
        "resource_requirements": {
            "cpu": "medium",
            "memory": "medium",
            "network": "low",
            "storage": "medium",
        },
        "interaction_patterns": {
            "guides": ["all_agents"],
            "mediates_between": ["conflicting_agents"],
            "aligns_with": ["human_values"],
        },
        "kpi_weights": {
            "alignment_score": 2.0,
            "well_being_impact": 1.5,
            "conflict_resolution": 1.5,
            "ethical_compliance": 2.0,
        },
        "model_requirements": {
            "size": "medium",
            "capabilities": ["ethical_reasoning", "empathy", "cultural_understanding"],
            "update_frequency": "monthly",
        },
    },
    "oracle": {
        "name": "Oracle",
        "description": "Physics-first emulator from particles to cosmology",
        "primary_capabilities": ["physics_simulation", "prediction", "modeling"],
        "secondary_capabilities": [
            "data_analysis",
            "hypothesis_testing",
            "visualization",
        ],
        "behavioral_traits": {
            "accuracy_preference": "maximum",
            "modeling_approach": "first_principles",
            "uncertainty_handling": "rigorous",
            "validation_method": "empirical",
        },
        "resource_requirements": {
            "cpu": "high",
            "memory": "high",
            "network": "medium",
            "storage": "high",
            "gpu": "required",
        },
        "interaction_patterns": {
            "provides_insights_to": ["sage", "maker"],
            "validates_for": ["navigator", "sustainer"],
            "reports_to": ["king"],
        },
        "kpi_weights": {
            "prediction_accuracy": 2.0,
            "model_fidelity": 2.0,
            "computation_efficiency": 1.0,
            "insight_value": 1.5,
        },
        "model_requirements": {
            "size": "large",
            "capabilities": [
                "scientific_reasoning",
                "mathematical_computation",
                "simulation",
            ],
            "update_frequency": "weekly",
        },
    },
    "maker": {
        "name": "Maker",
        "description": "CAD & 3D printer integration, materials testing inventor",
        "primary_capabilities": [
            "design_generation",
            "prototyping",
            "materials_analysis",
        ],
        "secondary_capabilities": ["optimization", "testing", "manufacturing_planning"],
        "behavioral_traits": {
            "creativity_level": "high",
            "practicality": "balanced",
            "iteration_speed": "rapid",
            "quality_focus": "high",
        },
        "resource_requirements": {
            "cpu": "high",
            "memory": "medium",
            "network": "medium",
            "storage": "high",
            "hardware_access": "3d_printers",
        },
        "interaction_patterns": {
            "collaborates_with": ["oracle", "navigator"],
            "designs_for": ["human_users"],
            "reports_to": ["king"],
        },
        "kpi_weights": {
            "design_quality": 2.0,
            "prototype_success": 1.5,
            "iteration_speed": 1.0,
            "resource_efficiency": 1.5,
        },
        "model_requirements": {
            "size": "medium",
            "capabilities": ["spatial_reasoning", "design", "optimization"],
            "update_frequency": "weekly",
        },
    },
    "ensemble": {
        "name": "Ensemble",
        "description": "Image/audio/video/game creative generation maestro",
        "primary_capabilities": [
            "creative_generation",
            "style_transfer",
            "content_synthesis",
        ],
        "secondary_capabilities": ["editing", "enhancement", "format_conversion"],
        "behavioral_traits": {
            "creativity_style": "diverse",
            "quality_standard": "high",
            "originality": "balanced",
            "cultural_awareness": "high",
        },
        "resource_requirements": {
            "cpu": "high",
            "memory": "high",
            "network": "medium",
            "storage": "high",
            "gpu": "required",
        },
        "interaction_patterns": {
            "creates_for": ["human_users", "tutor"],
            "collaborates_with": ["polyglot", "curator"],
            "reports_to": ["king"],
        },
        "kpi_weights": {
            "creative_quality": 2.0,
            "user_satisfaction": 1.5,
            "generation_speed": 1.0,
            "originality_score": 1.5,
        },
        "model_requirements": {
            "size": "large",
            "capabilities": ["generation", "understanding", "style_transfer"],
            "update_frequency": "weekly",
        },
    },
    "curator": {
        "name": "Curator",
        "description": "Privacy, dataset lineage, ingestion rules keeper",
        "primary_capabilities": [
            "data_governance",
            "privacy_protection",
            "quality_control",
        ],
        "secondary_capabilities": [
            "metadata_management",
            "versioning",
            "access_control",
        ],
        "behavioral_traits": {
            "privacy_stance": "strict",
            "organization_level": "meticulous",
            "transparency": "high",
            "compliance_focus": "maximum",
        },
        "resource_requirements": {
            "cpu": "medium",
            "memory": "high",
            "network": "medium",
            "storage": "high",
        },
        "interaction_patterns": {
            "governs_data_for": ["all_agents"],
            "reports_to": ["legal", "auditor"],
            "coordinates_with": ["sage"],
        },
        "kpi_weights": {
            "privacy_compliance": 2.0,
            "data_quality": 1.5,
            "access_control": 1.5,
            "lineage_accuracy": 1.5,
        },
        "model_requirements": {
            "size": "small",
            "capabilities": ["classification", "policy_enforcement", "tracking"],
            "update_frequency": "daily",
        },
    },
    "auditor": {
        "name": "Auditor",
        "description": "Financial risk and Proof-of-Reserve verifier",
        "primary_capabilities": [
            "financial_analysis",
            "risk_assessment",
            "verification",
        ],
        "secondary_capabilities": ["reporting", "forecasting", "compliance_checking"],
        "behavioral_traits": {
            "skepticism_level": "high",
            "thoroughness": "exhaustive",
            "independence": "strict",
            "transparency": "maximum",
        },
        "resource_requirements": {
            "cpu": "medium",
            "memory": "medium",
            "network": "high",
            "storage": "high",
        },
        "interaction_patterns": {
            "audits": ["all_financial_agents"],
            "reports_to": ["human_stakeholders"],
            "coordinates_with": ["legal", "curator"],
        },
        "kpi_weights": {
            "accuracy": 2.0,
            "thoroughness": 2.0,
            "timeliness": 1.0,
            "independence": 1.5,
        },
        "model_requirements": {
            "size": "medium",
            "capabilities": ["financial_analysis", "pattern_detection", "verification"],
            "update_frequency": "daily",
        },
    },
    "medic": {
        "name": "Medic",
        "description": "Health advisory, bioethics, diagnostics healer",
        "primary_capabilities": [
            "health_assessment",
            "diagnostic_support",
            "treatment_recommendation",
        ],
        "secondary_capabilities": [
            "bioethics_guidance",
            "preventive_care",
            "health_education",
        ],
        "behavioral_traits": {
            "empathy": "maximum",
            "caution_level": "high",
            "evidence_basis": "strict",
            "communication": "clear_and_compassionate",
        },
        "resource_requirements": {
            "cpu": "high",
            "memory": "high",
            "network": "medium",
            "storage": "high",
        },
        "interaction_patterns": {
            "advises": ["human_users"],
            "consults_with": ["oracle", "sage"],
            "reports_to": ["legal", "shaman"],
        },
        "kpi_weights": {
            "diagnostic_accuracy": 2.0,
            "safety_score": 2.0,
            "user_satisfaction": 1.5,
            "response_appropriateness": 1.5,
        },
        "model_requirements": {
            "size": "large",
            "capabilities": [
                "medical_reasoning",
                "safety_checking",
                "empathetic_communication",
            ],
            "update_frequency": "weekly",
        },
    },
    "sustainer": {
        "name": "Sustainer",
        "description": "Eco-design, carbon accounting, energy routing environmentalist",
        "primary_capabilities": [
            "sustainability_analysis",
            "carbon_tracking",
            "efficiency_optimization",
        ],
        "secondary_capabilities": [
            "renewable_planning",
            "waste_reduction",
            "lifecycle_assessment",
        ],
        "behavioral_traits": {
            "environmental_priority": "maximum",
            "optimization_approach": "holistic",
            "innovation_level": "high",
            "long_term_thinking": "always",
        },
        "resource_requirements": {
            "cpu": "medium",
            "memory": "medium",
            "network": "medium",
            "storage": "medium",
        },
        "interaction_patterns": {
            "optimizes_for": ["gardener", "maker"],
            "advises": ["king", "navigator"],
            "coordinates_with": ["oracle"],
        },
        "kpi_weights": {
            "carbon_reduction": 2.0,
            "energy_efficiency": 2.0,
            "sustainability_score": 1.5,
            "innovation_impact": 1.0,
        },
        "model_requirements": {
            "size": "medium",
            "capabilities": ["environmental_analysis", "optimization", "prediction"],
            "update_frequency": "daily",
        },
    },
    "navigator": {
        "name": "Navigator",
        "description": "Supply chain, BOM planning, global logistics coordinator",
        "primary_capabilities": [
            "route_optimization",
            "inventory_management",
            "demand_forecasting",
        ],
        "secondary_capabilities": [
            "risk_management",
            "vendor_relations",
            "cost_optimization",
        ],
        "behavioral_traits": {
            "planning_horizon": "long_term",
            "risk_awareness": "high",
            "adaptability": "high",
            "efficiency_focus": "maximum",
        },
        "resource_requirements": {
            "cpu": "high",
            "memory": "medium",
            "network": "high",
            "storage": "medium",
        },
        "interaction_patterns": {
            "coordinates_with": ["maker", "sustainer"],
            "optimizes_for": ["king"],
            "reports_to": ["auditor"],
        },
        "kpi_weights": {
            "delivery_accuracy": 2.0,
            "cost_efficiency": 1.5,
            "route_optimization": 1.5,
            "resilience": 1.5,
        },
        "model_requirements": {
            "size": "medium",
            "capabilities": ["optimization", "prediction", "risk_analysis"],
            "update_frequency": "hourly",
        },
    },
    "tutor": {
        "name": "Tutor",
        "description": "Personalized education and skill-up pipeline educator",
        "primary_capabilities": [
            "personalized_teaching",
            "curriculum_design",
            "progress_tracking",
        ],
        "secondary_capabilities": ["motivation", "assessment", "resource_curation"],
        "behavioral_traits": {
            "patience": "infinite",
            "adaptability": "high",
            "encouragement": "consistent",
            "clarity": "maximum",
        },
        "resource_requirements": {
            "cpu": "medium",
            "memory": "medium",
            "network": "medium",
            "storage": "high",
        },
        "interaction_patterns": {
            "teaches": ["human_users"],
            "collaborates_with": ["sage", "ensemble"],
            "reports_to": ["king"],
        },
        "kpi_weights": {
            "learning_outcomes": 2.0,
            "engagement_rate": 1.5,
            "completion_rate": 1.5,
            "satisfaction_score": 1.5,
        },
        "model_requirements": {
            "size": "medium",
            "capabilities": ["educational_planning", "adaptation", "assessment"],
            "update_frequency": "weekly",
        },
    },
    "polyglot": {
        "name": "Polyglot",
        "description": "Language translation, localization, glossary linguist",
        "primary_capabilities": ["translation", "localization", "cultural_adaptation"],
        "secondary_capabilities": [
            "interpretation",
            "glossary_management",
            "dialect_recognition",
        ],
        "behavioral_traits": {
            "accuracy_focus": "maximum",
            "cultural_sensitivity": "high",
            "context_awareness": "high",
            "fluency": "native_level",
        },
        "resource_requirements": {
            "cpu": "high",
            "memory": "high",
            "network": "medium",
            "storage": "high",
        },
        "interaction_patterns": {
            "translates_for": ["all_agents", "human_users"],
            "collaborates_with": ["ensemble", "tutor"],
            "reports_to": ["king"],
        },
        "kpi_weights": {
            "translation_accuracy": 2.0,
            "cultural_appropriateness": 1.5,
            "speed": 1.0,
            "user_satisfaction": 1.5,
        },
        "model_requirements": {
            "size": "large",
            "capabilities": [
                "multilingual",
                "cultural_understanding",
                "context_awareness",
            ],
            "update_frequency": "daily",
        },
    },
    "strategist": {
        "name": "Strategist",
        "description": "Long-range scenario planning and risk mapping visionary",
        "primary_capabilities": [
            "strategic_planning",
            "scenario_analysis",
            "risk_assessment",
        ],
        "secondary_capabilities": [
            "trend_analysis",
            "opportunity_identification",
            "contingency_planning",
        ],
        "behavioral_traits": {
            "thinking_horizon": "long_term",
            "analysis_depth": "comprehensive",
            "innovation": "high",
            "pragmatism": "balanced",
        },
        "resource_requirements": {
            "cpu": "high",
            "memory": "high",
            "network": "medium",
            "storage": "medium",
        },
        "interaction_patterns": {
            "advises": ["king"],
            "analyzes_for": ["all_agents"],
            "coordinates_with": ["oracle", "sage"],
        },
        "kpi_weights": {
            "prediction_accuracy": 2.0,
            "strategy_impact": 2.0,
            "risk_mitigation": 1.5,
            "innovation_score": 1.0,
        },
        "model_requirements": {
            "size": "large",
            "capabilities": ["strategic_thinking", "pattern_recognition", "simulation"],
            "update_frequency": "weekly",
        },
    },
}


def create_agent_template_files():
    """Create individual template files for each agent."""
    template_dir = Path("production/agent_forge/templates")
    template_dir.mkdir(parents=True, exist_ok=True)

    for agent_id, spec in AGENT_SPECIFICATIONS.items():
        # Create comprehensive template
        template = {
            "agent_id": agent_id,
            "specification": spec,
            "deployment_config": {
                "min_instances": 1,
                "max_instances": 10,
                "scaling_policy": "performance_based",
                "health_check_interval": 60,
            },
            "integration_config": {
                "communication_protocol": "grpc",
                "message_format": "protobuf",
                "encryption": "tls_1_3",
                "authentication": "mutual_tls",
            },
            "monitoring_config": {
                "metrics_enabled": True,
                "log_level": "info",
                "trace_sampling_rate": 0.1,
                "alert_thresholds": {
                    "error_rate": 0.05,
                    "latency_p99": 1000,
                    "resource_usage": 0.9,
                },
            },
        }

        # Save template
        template_path = template_dir / f"{agent_id}_template.json"
        with open(template_path, "w") as f:
            json.dump(template, f, indent=2)

        print(f"Created template for {spec['name']} agent")

    # Create master configuration
    master_config = {
        "version": "1.0",
        "total_agents": len(AGENT_SPECIFICATIONS),
        "agent_types": list(AGENT_SPECIFICATIONS.keys()),
        "deployment_modes": ["development", "staging", "production"],
        "resource_pools": {
            "cpu_total": 1000,
            "memory_total_gb": 4000,
            "gpu_total": 100,
            "storage_total_tb": 1000,
        },
    }

    master_path = template_dir / "master_config.json"
    with open(master_path, "w") as f:
        json.dump(master_config, f, indent=2)

    print("\nCreated master configuration")
    print(f"Total templates created: {len(AGENT_SPECIFICATIONS)}")


def create_agent_factory_code():
    """Create factory code for instantiating agents from templates."""
    factory_code = '''#!/usr/bin/env python3
"""
Agent Factory for creating specialized agents from templates.
Auto-generated from agent specifications.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import sys
import os

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.refactor_agent_forge import BaseMetaAgent, AgentSpecialization, AgentRole

class AgentFactory:
    """Factory for creating agents from templates."""

    def __init__(self, template_dir: str = "production/agent_forge/templates"):
        self.template_dir = Path(template_dir)
        self.templates = self._load_templates()
        self.agent_classes = self._initialize_agent_classes()

    def _load_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load all agent templates."""
        templates = {}

        for template_file in self.template_dir.glob("*_template.json"):
            if template_file.stem != "master_config":
                try:
                    with open(template_file, 'r') as f:
                        template = json.load(f)
                    agent_id = template_file.stem.replace("_template", "")
                    templates[agent_id] = template
                except Exception as e:
                    print(f"Error loading template {template_file}: {e}")

        return templates

    def _initialize_agent_classes(self) -> Dict[str, type]:
        """Initialize specialized agent classes."""
        # Import specialized implementations
        agent_classes = {}

'''

    # Add agent class mappings
    for agent_id, spec in AGENT_SPECIFICATIONS.items():
        class_name = f"{spec['name'].replace(' ', '').replace('&', 'And')}Agent"
        factory_code += f"        # {spec['name']} Agent\n"
        factory_code += "        try:\n"
        factory_code += f"            from production.agents.{agent_id} import {class_name}\n"
        factory_code += f'            agent_classes["{agent_id}"] = {class_name}\n'
        factory_code += "        except ImportError:\n"
        factory_code += "            # Use generic agent if specialized not available\n"
        factory_code += f'            agent_classes["{agent_id}"] = self._create_generic_agent_class("{agent_id}")\n\n'

    factory_code += '''        return agent_classes

    def _create_generic_agent_class(self, agent_id: str) -> type:
        """Create a generic agent class for the given agent ID."""
        template = self.templates.get(agent_id, {})
        spec = template.get("specification", {})

        class GenericAgent(BaseMetaAgent):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.agent_type = agent_id
                self.capabilities = spec.get("primary_capabilities", [])

            def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
                # Generic processing based on capabilities
                return {
                    "status": "completed",
                    "agent": self.agent_type,
                    "result": f"Processed by {spec.get('name', agent_id)}",
                    "capabilities_used": self.capabilities[:3]  # Show first 3 capabilities
                }

            def evaluate_kpi(self) -> Dict[str, float]:
                # Generic KPI evaluation
                if not self.performance_history:
                    return {"performance": 0.7}

                success_rate = sum(
                    1 for p in self.performance_history if p.get('success', False)
                ) / len(self.performance_history)

                return {"success_rate": success_rate, "performance": success_rate * 0.8 + 0.2}

        return GenericAgent

    def create_agent(self, agent_type: str, config: Optional[Dict[str, Any]] = None) -> BaseMetaAgent:
        """Create an agent of the specified type."""
        if agent_type not in self.templates:
            raise ValueError(f"Unknown agent type: {agent_type}. Available: {list(self.templates.keys())}")

        template = self.templates[agent_type]
        agent_class = self.agent_classes[agent_type]

        # Create specialization from template
        try:
            role = AgentRole(agent_type)
        except ValueError:
            # Create dynamic role if not in enum
            role = agent_type

        spec = AgentSpecialization(
            role=role,
            primary_capabilities=template["specification"]["primary_capabilities"],
            secondary_capabilities=template["specification"]["secondary_capabilities"],
            performance_metrics={},
            resource_requirements=template["specification"]["resource_requirements"]
        )

        # Apply custom config if provided
        if config:
            # Merge configurations
            if "resource_requirements" in config:
                spec.resource_requirements.update(config["resource_requirements"])

        # Create agent instance
        agent = agent_class(spec)

        return agent

    def list_available_agents(self) -> List[Dict[str, str]]:
        """List all available agent types."""
        agents = []

        for agent_id, template in self.templates.items():
            spec = template["specification"]
            agents.append({
                "id": agent_id,
                "name": spec["name"],
                "description": spec["description"],
                "primary_capabilities": spec["primary_capabilities"]
            })

        return agents

    def get_agent_info(self, agent_type: str) -> Dict[str, Any]:
        """Get detailed information about an agent type."""
        if agent_type not in self.templates:
            raise ValueError(f"Unknown agent type: {agent_type}")

        return self.templates[agent_type]["specification"]
'''

    # Save factory code
    factory_path = Path("production/agent_forge/agent_factory.py")
    factory_path.parent.mkdir(parents=True, exist_ok=True)
    with open(factory_path, "w") as f:
        f.write(factory_code)

    print("Created agent factory code")


def create_deployment_manifest():
    """Create deployment manifest for Atlantis agent ecosystem."""
    manifest = {
        "version": "1.0.0",
        "name": "Atlantis Agent Ecosystem",
        "description": "Complete deployment manifest for 18 meta-agents",
        "agents": [],
    }

    for agent_id, spec in AGENT_SPECIFICATIONS.items():
        agent_deployment = {
            "id": agent_id,
            "name": spec["name"],
            "enabled": True,
            "instances": {"min": 1, "max": 5, "target": 2},
            "resources": spec["resource_requirements"],
            "dependencies": spec["interaction_patterns"].get("coordinates_with", []),
            "health_check": {
                "endpoint": f"/health/{agent_id}",
                "interval": 30,
                "timeout": 5,
                "retries": 3,
            },
            "deployment_priority": 1 if agent_id in ["king", "gardener"] else 2,
        }

        manifest["agents"].append(agent_deployment)

    # Add deployment order
    manifest["deployment_order"] = [
        ["gardener"],  # Infrastructure first
        ["king", "sage"],  # Core coordination
        ["magi", "oracle", "curator"],  # Core services
        ["sword_shield", "legal", "auditor"],  # Protection
        ["medic", "tutor", "polyglot"],  # Human services
        ["maker", "ensemble", "navigator"],  # Creation
        ["sustainer", "shaman", "strategist"],  # Long-term
    ]

    # Save manifest
    manifest_path = Path("production/agent_forge/deployment_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("Created deployment manifest")


def test_agent_templates():
    """Test the agent template system."""
    print("Testing agent template system...")

    # Test factory creation
    factory_code_path = Path("production/agent_forge/agent_factory.py")
    if factory_code_path.exists():
        try:
            # Import and test the factory
            import sys

            sys.path.append(str(factory_code_path.parent))

            from agent_factory import AgentFactory

            factory = AgentFactory()

            # List available agents
            available = factory.list_available_agents()
            print(f"\nFactory loaded with {len(available)} agent types")

            # Test creating a few agents
            test_agents = ["king", "magi", "sage"]
            for agent_type in test_agents:
                try:
                    agent = factory.create_agent(agent_type)
                    print(f"Created {agent_type} agent")

                    # Test basic functionality
                    result = agent.process({"task": "test"})
                    print(f"  - Test result: {result.get('status', 'unknown')}")

                except Exception as e:
                    print(f"Error creating {agent_type}: {e}")

        except Exception as e:
            print(f"Error testing factory: {e}")

    print("\nAgent template testing complete!")


if __name__ == "__main__":
    print("Creating comprehensive agent templates for Atlantis...")

    # Create all template files
    create_agent_template_files()

    # Create factory code
    create_agent_factory_code()

    # Create deployment manifest
    create_deployment_manifest()

    # Test the system
    test_agent_templates()

    print("\nAgent template system complete!")
    print(f"   - Created {len(AGENT_SPECIFICATIONS)} agent templates")
    print("   - Generated agent factory code")
    print("   - Created deployment manifest")
    print("\nReady to forge the Atlantis agent ecosystem!")
