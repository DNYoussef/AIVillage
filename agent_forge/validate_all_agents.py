"""CLI wrapper for production agent validation."""

from src.production.agent_forge.validate_all_agents import main, validate_all_agents

__all__ = ["main", "validate_all_agents"]

if __name__ == "__main__":  # pragma: no cover - script entry
    main()
