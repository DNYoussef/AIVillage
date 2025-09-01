#!/usr/bin/env python3
"""
Enhanced Module Bridge for agent_forge (hyphenated directory) to agent_forge (underscore import)
This allows importing agent_forge.phases.cognate_pretrain even though the directory is agent_forge

Handles all submodules and provides proper error messages for debugging.
"""

import importlib.util
import logging
from pathlib import Path
import sys

# Set up logging for debugging imports
logger = logging.getLogger(__name__)

# Get the actual agent_forge directory
_agent_forge_dir = Path(__file__).parent / "agent_forge"


def _load_hyphenated_module(name, path, submodule_name=None):
    """Load a module from a hyphenated directory with enhanced error handling."""
    try:
        init_file = path / "__init__.py"
        if not init_file.exists():
            logger.warning(f"No __init__.py found in {path}")
            return None

        spec = importlib.util.spec_from_file_location(name, init_file)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)

            # Add the module to sys.modules BEFORE executing to handle circular imports
            sys.modules[name] = module

            try:
                spec.loader.exec_module(module)
                logger.debug(f"Successfully loaded module: {name}")
                return module
            except Exception as e:
                logger.error(f"Failed to execute module {name}: {e}")
                # Remove from sys.modules if execution failed
                if name in sys.modules:
                    del sys.modules[name]
                return None
        else:
            logger.error(f"Could not create spec for module {name} at {path}")
            return None
    except Exception as e:
        logger.error(f"Error loading module {name}: {e}")
        return None


def _load_all_phases():
    """Load all phase modules dynamically."""
    phases_dir = _agent_forge_dir / "phases"
    if not phases_dir.exists():
        logger.warning(f"Phases directory not found: {phases_dir}")
        return None

    # Load the phases module
    phases_module = _load_hyphenated_module("agent_forge.phases", phases_dir)
    if not phases_module:
        return None

    # Dynamically load all phase subdirectories
    loaded_phases = []
    for phase_path in phases_dir.iterdir():
        if phase_path.is_dir() and (phase_path / "__init__.py").exists():
            phase_name = phase_path.name.replace("-", "_")  # Convert hyphenated names
            module_name = f"agent_forge.phases.{phase_name}"

            phase_module = _load_hyphenated_module(module_name, phase_path)
            if phase_module:
                # Add as attribute to phases module
                setattr(phases_module, phase_name, phase_module)
                loaded_phases.append(phase_name)
                logger.debug(f"Loaded phase module: {phase_name}")

    logger.info(f"Loaded {len(loaded_phases)} phase modules: {loaded_phases}")
    return phases_module


def _load_core_modules():
    """Load core modules (phase_controller, etc.)."""
    core_dir = _agent_forge_dir / "core"
    if not core_dir.exists():
        logger.warning(f"Core directory not found: {core_dir}")
        return {}

    loaded_core = {}
    for core_file in core_dir.glob("*.py"):
        if core_file.name == "__init__.py":
            continue

        module_name = f"agent_forge.core.{core_file.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, core_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                loaded_core[core_file.stem] = module
                logger.debug(f"Loaded core module: {module_name}")
        except Exception as e:
            logger.error(f"Failed to load core module {module_name}: {e}")

    return loaded_core


# Main module loading logic
_import_success = False
_import_errors = []

if _agent_forge_dir.exists():
    try:
        # Load the main agent_forge module
        _agent_forge_module = _load_hyphenated_module("agent_forge", _agent_forge_dir)
        if _agent_forge_module:
            # Load core modules
            core_modules = _load_core_modules()
            for name, module in core_modules.items():
                setattr(_agent_forge_module, name, module)

            # Load all phases
            phases_module = _load_all_phases()
            if phases_module:
                _agent_forge_module.phases = phases_module

            _import_success = True
            logger.info("Agent Forge bridge module loaded successfully")
        else:
            _import_errors.append("Failed to load main agent_forge module")

    except Exception as e:
        _import_errors.append(f"Critical error loading agent_forge: {e}")
        logger.error(f"Critical error in bridge module: {e}")
else:
    _import_errors.append(f"Agent forge directory not found: {_agent_forge_dir}")


# Provide diagnostics function
def get_import_status():
    """Get the current import status for debugging."""
    return {
        "success": _import_success,
        "errors": _import_errors,
        "agent_forge_dir": str(_agent_forge_dir),
        "loaded_modules": [name for name in sys.modules.keys() if name.startswith("agent_forge")],
    }


# Re-export the module for convenience
if _import_success:
    try:
        # Only import if the bridge was successful
        from agent_forge import *
    except ImportError as e:
        logger.warning(f"Could not re-export agent_forge: {e}")
        _import_errors.append(f"Re-export failed: {e}")
else:
    logger.error("Agent Forge bridge failed to load properly")
