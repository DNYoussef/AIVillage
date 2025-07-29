"""Production evolution components organized by Sprint 2."""

# Export main evolution classes
try:
    from .evomerge.config import Config
    from .evomerge.evolutionary_tournament import EvolutionaryTournament

    __all__ = ["Config", "EvolutionaryTournament"]
except ImportError:
    # Handle missing dependencies gracefully
    EvolutionaryTournament = None
    Config = None
    __all__ = []
