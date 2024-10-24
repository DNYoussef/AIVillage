"""
MAGI Reasoning Techniques Module

This module contains implementations of various reasoning techniques used by the MAGI system
for problem-solving and decision-making. Each technique is implemented as a separate class
with a consistent interface for integration into the main system.

Available techniques:
- Chain of Thought
- Tree of Thoughts
- Program of Thoughts
- Self Ask
- Least to Most Prompting
- Contrastive Chain of Thought
- Memory of Thought
- Choice Annealing
- Prompt Chaining
- Self Consistency
- Evolutionary Tournament
"""

from .chain_of_thought import ChainOfThought
from .tree_of_thoughts import TreeOfThoughts
from .program_of_thoughts import ProgramOfThoughts
from .self_ask import SelfAsk
from .least_to_most import LeastToMost
from .contrastive_chain import ContrastiveChain
from .memory_of_thought import MemoryOfThought
from .choice_annealing import ChoiceAnnealing
from .prompt_chaining import PromptChaining
from .self_consistency import SelfConsistency
from .evolutionary_tournament import EvolutionaryTournament

__all__ = [
    'ChainOfThought',
    'TreeOfThoughts',
    'ProgramOfThoughts',
    'SelfAsk',
    'LeastToMost',
    'ContrastiveChain',
    'MemoryOfThought',
    'ChoiceAnnealing',
    'PromptChaining',
    'SelfConsistency',
    'EvolutionaryTournament'
]
