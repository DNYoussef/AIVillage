"""
Feature Flag System for AIVillage

This module provides runtime feature toggles with canary rollouts and kill-switch capabilities.
Designed for production safety and gradual feature deployment.
"""

import os
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

try:
    pass
except ImportError:
    pass


class FlagState(Enum):
    """Feature flag states."""

    ENABLED = "enabled"
    DISABLED = "disabled"
    CANARY = "canary"
    KILL_SWITCH = "kill_switch"


class FeatureFlag:
    """Individual feature flag configuration."""

    def __init__(
        self,
        key: str,
        state: FlagState = FlagState.DISABLED,
        description: str = "",
        canary_percentage: float = 0.0,
        canary_users: set[str] | None = None,
        environments: list[str] | None = None,
        kill_switch_reason: str = "",
    ):
        self.key = key
        self.state = state
        self.description = description
        self.canary_percentage = max(0.0, min(100.0, canary_percentage))
        self.canary_users = canary_users or set()
        self.environments = environments or ["all"]
        self.kill_switch_reason = kill_switch_reason
        self.created_at = time.time()
        self.updated_at = time.time()

    def is_enabled_for_user(self, user_id: str | None = None, environment: str = "production") -> bool:
        """Check if flag is enabled for a specific user/environment."""
        # Check environment filter
        if self.environments != ["all"] and environment not in self.environments:
            return False

        # Handle kill switch
        if self.state == FlagState.KILL_SWITCH:
            return False

        # Handle disabled state
        if self.state == FlagState.DISABLED:
            return False

        # Handle enabled state
        if self.state == FlagState.ENABLED:
            return True

        # Handle canary state
        if self.state == FlagState.CANARY:
            # Check if user is in explicit canary list
            if user_id and user_id in self.canary_users:
                return True

            # Check percentage-based canary
            if user_id and self.canary_percentage > 0:
                # Simple hash-based percentage check
                user_hash = hash(f"{self.key}:{user_id}") % 100
                return user_hash < self.canary_percentage

            return False

        return False

    def to_dict(self) -> dict[str, Any]:
        """Convert flag to dictionary representation."""
        return {
            "key": self.key,
            "state": self.state.value,
            "description": self.description,
            "canary_percentage": self.canary_percentage,
            "canary_users": list(self.canary_users),
            "environments": self.environments,
            "kill_switch_reason": self.kill_switch_reason,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class FeatureFlagManager:
    """Centralized feature flag management with runtime configuration."""

    def __init__(self, config_path: str | None = None):
        self.config_path = config_path or self._get_default_config_path()
        self.flags: dict[str, FeatureFlag] = {}
        self._lock = threading.RLock()
        self._environment = os.environ.get("AIVILLAGE_ENV", "production")
        self._last_reload = 0
        self._reload_interval = 60  # seconds

        self.load_flags()

    def _get_default_config_path(self) -> str:
        """Get default config path."""
        project_root = Path(__file__).parent.parent.parent.parent
        return str(project_root / "config" / "flags.yaml")

    def load_flags(self) -> None:
        """Load flags from configuration file."""
        with self._lock:
            try:
                if os.path.exists(self.config_path):
                    with open(self.config_path) as f:
                        config = yaml.safe_load(f) or {}

                    self.flags = {}
                    for flag_data in config.get("flags", []):
                        flag = FeatureFlag(
                            key=flag_data["key"],
                            state=FlagState(flag_data.get("state", "disabled")),
                            description=flag_data.get("description", ""),
                            canary_percentage=flag_data.get("canary_percentage", 0.0),
                            canary_users=set(flag_data.get("canary_users", [])),
                            environments=flag_data.get("environments", ["all"]),
                            kill_switch_reason=flag_data.get("kill_switch_reason", ""),
                        )
                        self.flags[flag.key] = flag

                    self._last_reload = time.time()
                else:
                    # Create default config if it doesn't exist
                    self._create_default_config()

            except Exception as e:
                print(f"Warning: Failed to load feature flags from {self.config_path}: {e}")
                # Continue with empty flags rather than failing

    def _create_default_config(self) -> None:
        """Create default configuration file."""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

        default_config = {
            "flags": [
                {
                    "key": "advanced_rag_features",
                    "state": "canary",
                    "description": "Enable advanced RAG features like Bayesian trust networks",
                    "canary_percentage": 10.0,
                    "environments": ["development", "staging"],
                },
                {
                    "key": "agent_forge_v2",
                    "state": "disabled",
                    "description": "Enable Agent Forge v2 pipeline with enhanced compression",
                    "environments": ["development"],
                },
                {
                    "key": "p2p_mesh_networking",
                    "state": "enabled",
                    "description": "Enable BitChat P2P mesh networking",
                    "environments": ["all"],
                },
                {
                    "key": "experimental_compression",
                    "state": "canary",
                    "description": "Enable experimental compression algorithms",
                    "canary_percentage": 5.0,
                    "canary_users": ["admin", "tester"],
                    "environments": ["development", "staging"],
                },
            ]
        }

        with open(self.config_path, "w") as f:
            yaml.dump(default_config, f, default_flow_style=False)

    def is_enabled(self, flag_key: str, user_id: str | None = None) -> bool:
        """Check if a feature flag is enabled."""
        # Auto-reload config if needed
        if time.time() - self._last_reload > self._reload_interval:
            self.load_flags()

        with self._lock:
            flag = self.flags.get(flag_key)
            if not flag:
                # Unknown flags default to disabled for safety
                return False

            return flag.is_enabled_for_user(user_id, self._environment)

    def get_flag(self, flag_key: str) -> FeatureFlag | None:
        """Get feature flag configuration."""
        with self._lock:
            return self.flags.get(flag_key)

    def set_flag_state(self, flag_key: str, state: FlagState, reason: str = "") -> bool:
        """Set flag state (runtime override)."""
        with self._lock:
            flag = self.flags.get(flag_key)
            if not flag:
                return False

            flag.state = state
            flag.updated_at = time.time()

            if state == FlagState.KILL_SWITCH:
                flag.kill_switch_reason = reason

            return True

    def kill_switch(self, flag_key: str, reason: str) -> bool:
        """Emergency kill switch for a feature."""
        return self.set_flag_state(flag_key, FlagState.KILL_SWITCH, reason)

    def list_flags(self) -> dict[str, dict[str, Any]]:
        """List all feature flags and their current state."""
        with self._lock:
            return {key: flag.to_dict() for key, flag in self.flags.items()}

    def get_enabled_flags(self, user_id: str | None = None) -> list[str]:
        """Get list of enabled flag keys for a user."""
        enabled = []
        with self._lock:
            for key, flag in self.flags.items():
                if flag.is_enabled_for_user(user_id, self._environment):
                    enabled.append(key)
        return enabled


# Global feature flag manager instance
_flag_manager: FeatureFlagManager | None = None


def get_flag_manager() -> FeatureFlagManager:
    """Get global feature flag manager instance."""
    global _flag_manager
    if _flag_manager is None:
        _flag_manager = FeatureFlagManager()
    return _flag_manager


def is_enabled(flag_key: str, user_id: str | None = None) -> bool:
    """Check if a feature flag is enabled (convenience function)."""
    return get_flag_manager().is_enabled(flag_key, user_id)


def kill_switch(flag_key: str, reason: str) -> bool:
    """Emergency kill switch for a feature (convenience function)."""
    return get_flag_manager().kill_switch(flag_key, reason)


def feature_flag(flag_key: str, user_id: str | None = None):
    """Decorator to conditionally execute functions based on feature flags."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            if is_enabled(flag_key, user_id):
                return func(*args, **kwargs)
            else:
                # Return None or raise NotImplementedError based on context
                return None

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = f"[Feature Flag: {flag_key}] {func.__doc__ or ''}"
        return wrapper

    return decorator


# Context manager for feature flag testing
class FeatureFlagContext:
    """Context manager for temporarily enabling/disabling flags in tests."""

    def __init__(self, flag_overrides: dict[str, bool]):
        self.overrides = flag_overrides
        self.manager = get_flag_manager()
        self.original_states = {}

    def __enter__(self):
        with self.manager._lock:
            for flag_key, enabled in self.overrides.items():
                if flag_key in self.manager.flags:
                    flag = self.manager.flags[flag_key]
                    self.original_states[flag_key] = flag.state
                    flag.state = FlagState.ENABLED if enabled else FlagState.DISABLED
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        with self.manager._lock:
            for flag_key, original_state in self.original_states.items():
                if flag_key in self.manager.flags:
                    self.manager.flags[flag_key].state = original_state


# Usage examples and utilities
if __name__ == "__main__":
    # Example usage
    manager = FeatureFlagManager()

    # Check if feature is enabled
    if is_enabled("advanced_rag_features", "user123"):
        print("Advanced RAG features enabled for user")

    # Use decorator
    @feature_flag("experimental_compression")
    def experimental_compress_data(data):
        return f"Compressed: {data}"

    result = experimental_compress_data("test data")
    print(f"Result: {result}")

    # Emergency kill switch
    kill_switch("experimental_compression", "Production incident #1234")

    # List all flags
    all_flags = manager.list_flags()
    print(f"All flags: {all_flags}")
