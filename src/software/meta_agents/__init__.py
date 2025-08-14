"""
Meta-Agents Architecture

Each meta-agent has its own:
- Architecture and model
- Tool access and capabilities
- Memory and journal system
- Communication via BitChat/BetaNet
- Quiet Star self-learning
- Encrypted thought bubbles (except King - public thoughts)
"""

from .battle_orchestrator import BattleMetrics, BattleOrchestrator, BattleScenario
from .king import KingAgent
from .shield import ShieldAgent
from .sword import SwordAgent

__all__ = [
    "KingAgent",
    "SwordAgent",
    "ShieldAgent",
    "BattleOrchestrator",
    "BattleMetrics",
    "BattleScenario",
]
