"""Atlantis Meta-Agents - Complete AIVillage Agent Architecture

This module contains the full roster of 25 Atlantis meta-agents organized by domain:

Governance, Safety, and Policy:
- King (Orchestrator & Job Scheduler)
- Shield (Blue Team & Constitutional Enforcement)
- Sword (Red Team & Adversarial Testing)
- Legal/Judiciar (Law, Policy, Contracts)
- Auditor (Receipts, Risk & Compliance)

Knowledge, Research, and Reasoning:
- Sage (Research Lead & Knowledge Graph)
- Curator (Content Organization)
- Oracle (Science & Simulation)
- Shaman (Human Factors & Patterns)
- Strategist (Long-Horizon Planning)

Language, Education, and Health:
- Polyglot (Translation & Linguistics)
- Tutor (Learning & Assessment)
- Medic (Telehealth Triage & Guidance)

Economy, Markets, and Finance:
- Merchant (Marketplace & Credits)
- Banker/Economist (Treasury & Investment)

Build, Infrastructure, and Ops:
- Magi (Engineering & Model R&D)
- Navigator (Routing & Data Movement)
- Gardener (System Upkeep & Topology)
- Sustainer (Capacity & Efficiency)
- Coordinator (Multi-Agent Workflow)

Making, Culture, and Local Value:
- Maker (Makerspaces & Production)
- Ensemble (Creative Media)
- Horticulturist (Agro/Permaculture)

Plus 8 Specialized Sub-Agents:
- DataScienceAgent, DevOpsAgent, FinancialAgent, CreativeAgent,
- SocialAgent, TranslatorAgent, ArchitectAgent, TesterAgent
"""

# Q1 MVP - Import only implemented agents
from .culture_making import HorticulturistAgent
from .governance import AuditorAgent, KingAgent, ShieldAgent
from .infrastructure import MagiAgent, NavigatorAgent, SustainerAgent
from .language_education_health import PolyglotAgent, TutorAgent

__all__ = [
    # Q1 MVP - Implemented Agents Only
    "KingAgent",
    "ShieldAgent",
    "AuditorAgent",  # Governance
    "MagiAgent",
    "NavigatorAgent",
    "SustainerAgent",  # Infrastructure
    "PolyglotAgent",
    "TutorAgent",  # Language & Education
    "HorticulturistAgent",  # Culture & Making
]

# Q1 MVP Agent organization by domain
GOVERNANCE_AGENTS = ["KingAgent", "ShieldAgent", "AuditorAgent"]
INFRASTRUCTURE_AGENTS = ["MagiAgent", "NavigatorAgent", "SustainerAgent"]
LANGUAGE_HEALTH_AGENTS = ["PolyglotAgent", "TutorAgent"]
CULTURE_MAKING_AGENTS = ["HorticulturistAgent"]

# Q1 MVP Total agent count
TOTAL_MVP_AGENTS = 9  # Q1 MVP implemented agents


def get_agent_roster():
    """Get Q1 MVP agent roster organized by domain"""
    return {
        "governance": GOVERNANCE_AGENTS,
        "infrastructure": INFRASTRUCTURE_AGENTS,
        "language_health": LANGUAGE_HEALTH_AGENTS,
        "culture_making": CULTURE_MAKING_AGENTS,
        "totals": {"total_mvp_agents": TOTAL_MVP_AGENTS},
    }


def get_agent_by_capability(capability: str):
    """Find Q1 MVP agents that have a specific capability"""
    capability_map = {
        # Orchestration & Leadership
        "task_orchestration": ["KingAgent"],
        "strategic_planning": ["KingAgent"],
        # Security & Safety
        "security_enforcement": ["ShieldAgent"],
        "risk_assessment": ["AuditorAgent"],
        "compliance_monitoring": ["AuditorAgent"],
        # Communication & Culture
        "translation": ["PolyglotAgent"],
        "multilingual_support": ["PolyglotAgent"],
        # Technical & Infrastructure
        "system_engineering": ["MagiAgent"],
        "routing_networking": ["NavigatorAgent"],
        "resource_monitoring": ["SustainerAgent"],
        "capacity_management": ["SustainerAgent"],
        # Learning & Agriculture
        "education": ["TutorAgent"],
        "personalized_learning": ["TutorAgent"],
        "agricultural_systems": ["HorticulturistAgent"],
        "permaculture": ["HorticulturistAgent"],
        # Audit & Compliance
        "receipt_management": ["AuditorAgent"],
        "financial_tracking": ["AuditorAgent"],
    }

    return capability_map.get(capability, [])
