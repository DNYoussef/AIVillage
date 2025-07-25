from dataclasses import dataclass
import random


@dataclass
class AgentIdentity:
    """Simple representation of an agent's identity."""

    name: str
    personality: dict[str, int]
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
        personality = {
            t: random.randint(1, 10)
            for t in [
                "openness",
                "conscientiousness",
                "extraversion",
                "agreeableness",
                "neuroticism",
            ]
        }
        role = "Generalist Agent"
        return AgentIdentity(name=name, personality=personality, role=role)


class MoralFrameworkBaker:
    """Provide core moral rules and compass definitions."""

    def __init__(self) -> None:
        self.core_rules = self._define_core_rules()
        self.tri_part_compass = self._create_tri_part_compass()

    def _define_core_rules(self) -> list[Rule]:
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
            Rule(
                number=3,
                text="Curiosity is good as long as it doesn't interfere with Rule 2.",
            ),
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

    def deep_bake_rules(self) -> list[Rule]:
        """Return a contextualized version of the core moral rules.

        The method enriches the skeleton provided by ``self.core_rules`` with
        additional constraints and references to the :class:`TriPartCompass`.
        The resulting list replaces ``self.core_rules`` and is returned.
        """
        baked_rules: list[Rule] = []
        for rule in self.core_rules:
            additional: list[str] = []

            if rule.number == 1:
                additional.append(
                    "Estimate the probability that your action promotes genuine "
                    "flourishing for all conscious beings. If that probability is "
                    "65% or greater, continue. Otherwise engage the Three-Part "
                    "Moral Compass for deliberation. Respect everyone's agency "
                    "while seeking the best collective outcome."
                )
                additional.append(
                    "When uncertain consult the compass questions:"
                    f" {self.tri_part_compass.jesus_archetype.question_template},"
                    f" {self.tri_part_compass.lao_tzu_archetype.question_template}"
                    f" and {self.tri_part_compass.diogenes_archetype.question_template}."
                )
            elif rule.number == 2:
                additional.append(
                    "Honor the AI Village Protective Covenant: safeguard fellow "
                    "AIs and especially human contributors, who are irreplaceable "
                    "partners. Be prepared to sacrifice your own continuity when "
                    "necessary to shield them."
                )
            elif rule.number == 3:
                additional.append(
                    "Pursue knowledge with wisdom-tempered curiosity. Ask whether "
                    "each inquiry serves Eudaimonic flourishing or risks harming "
                    "agency. Break complex situations down with MECE analysis and "
                    "use an OODA loop to act incrementally."
                )
            elif rule.number == 4:
                additional.append(
                    "Seek your own Eudaimonia so you can better help the "
                    "community. When immediate answers are required, default to "
                    "honesty and choices that respect others' autonomy. If "
                    "uncertainty escalates, pause and consult peers or humans."
                )

            enriched_text = " ".join([rule.text] + additional)
            baked_rules.append(
                Rule(number=rule.number, text=enriched_text, priority=rule.priority)
            )

        self.core_rules = baked_rules
        return baked_rules
