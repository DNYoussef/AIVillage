"""Production evolution components organized by Sprint 2."""

# Export main evolution classes
try:
    from .evomerge.evolutionary_tournament import EvolutionaryTournament
    from .evomerge.config import Config
    
    __all__ = ['EvolutionaryTournament', 'Config']
except ImportError:
    # Handle missing dependencies gracefully
    EvolutionaryTournament = None
    Config = None
    __all__ = []
