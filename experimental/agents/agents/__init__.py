from .orchestration import main

try:  # pragma: no cover - optional dependency
    from .unified_base_agent import (
        SelfEvolvingSystem,
        UnifiedAgentConfig,
        UnifiedBaseAgent,
        create_agent,
    )
except ModuleNotFoundError:  # pragma: no cover - core package not installed
    pass

__all__ = [
    "main",
]
