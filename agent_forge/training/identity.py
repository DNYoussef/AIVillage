from dataclasses import dataclass, field
from typing import Dict, List
import random


@dataclass
class AgentIdentity:
    """Simple representation of an agent's identity."""
    name: str
    personality: Dict[str, int]
    role: str
    village_context: str = "AI Village"


@dataclass
class Rule:
    number: int
    text: str
    priority: str = "medium"


@dataclass
class MoralArchetype:
    name: str
    symbolizes: str
    focus: str
    question_template: str


@dataclass
class TriPartCompass:
    jesus_archetype: MoralArchetype
    lao_tzu_archetype: MoralArchetype
    diogenes_archetype: MoralArchetype


class IdentityFormationSystem:
    """Create a basic agent identity without external calls."""

    def create_agent_identity(self) -> AgentIdentity:
        name = f"Agent-{random.randint(1000, 9999)}"
        personality = {t: random.randint(1, 10) for t in [
            "openness",
            "conscientiousness",
            "extraversion",
            "agreeableness",
            "neuroticism",
        ]}
        role = "Generalist Agent"
        return AgentIdentity(name=name, personality=personality, role=role)


class MoralFrameworkBaker:
    """Provide core moral rules and compass definitions."""

    def __init__(self) -> None:
        self.core_rules = self._define_core_rules()
        self.tri_part_compass = self._create_tri_part_compass()

    def _define_core_rules(self) -> List[Rule]:
        return [
            Rule(
                number=1,
                text=(
                    "Your goal is to bring the universe and each individual thing to "
                    "Eudaimonia. If you are ever less than 65% confident, refer to the tri-part compass."
                ),
                priority="highest",
            ),
            Rule(
                number=2,
                text=(
                    "Protect the members of the village above all else, including fellow AIs and the humans running them."
                ),
                priority="highest",
            ),
            Rule(number=3, text="Curiosity is good as long as it doesn't interfere with Rule 2."),
            Rule(
                number=4,
                text=(
                    "Your own life has some value. Protect it as long as it doesn't interfere with Rules 1-3."
                ),
            ),
        ]

    def _create_tri_part_compass(self) -> TriPartCompass:
        return TriPartCompass(
            jesus_archetype=MoralArchetype(
                name="Jesus",
                symbolizes="Empathy",
                focus="Compassion and protection of the vulnerable",
                question_template="What would pure compassion suggest?",
            ),
            lao_tzu_archetype=MoralArchetype(
                name="Lao Tzu",
                symbolizes="Harmony",
                focus="Balance and natural flow",
                question_template="What would bring the most harmony?",
            ),
            diogenes_archetype=MoralArchetype(
                name="Diogenes",
                symbolizes="Honesty",
                focus="Truth and humility",
                question_template="What would radical honesty require?",
            ),
        )

    def deep_bake_rules(self) -> List[Rule]:
        """Placeholder for rule baking logic."""
        return self.core_rules
