#!/usr/bin/env python3
"""
Import Helper for Agent Forge System

Provides standardized import functions for tests and external modules.
Handles the hyphenated directory structure and provides clear error messages.
"""

import logging
from pathlib import Path
import sys
from typing import Any

# Set up logging
logger = logging.getLogger(__name__)


def setup_agent_forge_paths():
    """
    Set up Python paths for Agent Forge imports.
    Call this before importing any agent_forge modules.
    """
    # Get project root
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent  # Go up from core/agent_forge/

    # Add paths in correct order
    paths_to_add = [
        str(project_root),  # Root for general imports
        str(project_root / "core"),  # For agent_forge bridge
        str(project_root / "core" / "agent_forge"),  # Direct access to agent_forge
    ]

    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)

    return project_root


def import_agent_forge_module(module_path: str, required: bool = True) -> Any | None:
    """
    Import an agent_forge module with proper error handling.

    Args:
        module_path: Dot-separated module path (e.g., 'agent_forge.phases.cognate_pretrain')
        required: If True, raises ImportError on failure. If False, returns None.

    Returns:
        The imported module or None if import failed and not required.
    """
    try:
        # Ensure paths are set up
        setup_agent_forge_paths()

        # Try importing through the bridge module first
        if module_path == "agent_forge":
            import agent_forge

            return agent_forge

        # Split the path to handle submodules
        parts = module_path.split(".")
        if len(parts) < 2 or parts[0] != "agent_forge":
            raise ImportError(f"Invalid agent_forge module path: {module_path}")

        # Import the main agent_forge module first
        import agent_forge

        # Navigate to the specific submodule
        current_module = agent_forge
        for part in parts[1:]:
            if hasattr(current_module, part):
                current_module = getattr(current_module, part)
            else:
                raise ImportError(f"Module {module_path} not found - missing attribute '{part}'")

        return current_module

    except ImportError as e:
        error_msg = f"Failed to import {module_path}: {e}"
        logger.error(error_msg)

        if required:
            raise ImportError(error_msg) from e
        else:
            logger.warning(f"Optional import failed: {module_path}")
            return None
    except Exception as e:
        error_msg = f"Unexpected error importing {module_path}: {e}"
        logger.error(error_msg)

        if required:
            raise ImportError(error_msg) from e
        else:
            return None


def get_cognate_modules() -> dict[str, Any]:
    """
    Import all Cognate-related modules.

    Returns:
        Dictionary mapping module names to imported modules.
    """
    modules = {}
    cognate_imports = [
        ("cognate_pretrain", "agent_forge.phases.cognate_pretrain"),
        ("model_factory", "agent_forge.phases.cognate_pretrain.model_factory"),
        ("cognate_creator", "agent_forge.phases.cognate_pretrain.cognate_creator"),
        ("phase_integration", "agent_forge.phases.cognate_pretrain.phase_integration"),
    ]

    for name, module_path in cognate_imports:
        try:
            modules[name] = import_agent_forge_module(module_path, required=False)
        except:
            logger.warning(f"Could not import {name} from {module_path}")

    return {k: v for k, v in modules.items() if v is not None}


def get_evomerge_modules() -> dict[str, Any]:
    """
    Import all EvoMerge-related modules.

    Returns:
        Dictionary mapping module names to imported modules.
    """
    modules = {}
    evomerge_imports = [
        ("evomerge", "agent_forge.phases.evomerge"),
        ("evomerge_phase", "agent_forge.phases.evomerge"),
    ]

    for name, module_path in evomerge_imports:
        try:
            modules[name] = import_agent_forge_module(module_path, required=False)
        except:
            logger.warning(f"Could not import {name} from {module_path}")

    return {k: v for k, v in modules.items() if v is not None}


def get_core_modules() -> dict[str, Any]:
    """
    Import core Agent Forge modules.

    Returns:
        Dictionary mapping module names to imported modules.
    """
    modules = {}
    core_imports = [
        ("phase_controller", "agent_forge.core.phase_controller"),
        ("unified_pipeline", "agent_forge.unified_pipeline"),
    ]

    for name, module_path in core_imports:
        try:
            modules[name] = import_agent_forge_module(module_path, required=False)
        except:
            logger.warning(f"Could not import {name} from {module_path}")

    return {k: v for k, v in modules.items() if v is not None}


def validate_agent_forge_installation() -> dict[str, Any]:
    """
    Validate that Agent Forge modules can be imported.

    Returns:
        Dictionary with validation results.
    """
    results = {
        "success": False,
        "errors": [],
        "warnings": [],
        "imported_modules": [],
        "cognate_available": False,
        "evomerge_available": False,
        "core_available": False,
    }

    try:
        # Test basic import
        import_agent_forge_module("agent_forge", required=True)
        results["imported_modules"].append("agent_forge")

        # Test Cognate modules
        cognate_modules = get_cognate_modules()
        if cognate_modules:
            results["cognate_available"] = True
            results["imported_modules"].extend(cognate_modules.keys())

        # Test EvoMerge modules
        evomerge_modules = get_evomerge_modules()
        if evomerge_modules:
            results["evomerge_available"] = True
            results["imported_modules"].extend(evomerge_modules.keys())

        # Test core modules
        core_modules = get_core_modules()
        if core_modules:
            results["core_available"] = True
            results["imported_modules"].extend(core_modules.keys())

        # Check if we have the minimum required modules
        if results["cognate_available"] or results["evomerge_available"]:
            results["success"] = True
        else:
            results["errors"].append("Neither Cognate nor EvoMerge modules could be imported")

    except Exception as e:
        results["errors"].append(f"Critical import error: {e}")

    return results


def print_import_status():
    """Print a detailed import status report for debugging."""
    print("=" * 60)
    print("AGENT FORGE IMPORT STATUS")
    print("=" * 60)

    # Set up paths
    project_root = setup_agent_forge_paths()
    print(f"Project root: {project_root}")
    print(f"Python paths: {len(sys.path)} paths configured")

    # Validate installation
    status = validate_agent_forge_installation()

    print(f"\nImport Status: {'✅ SUCCESS' if status['success'] else '❌ FAILED'}")
    print(f"Cognate Available: {'✅ YES' if status['cognate_available'] else '❌ NO'}")
    print(f"EvoMerge Available: {'✅ YES' if status['evomerge_available'] else '❌ NO'}")
    print(f"Core Available: {'✅ YES' if status['core_available'] else '❌ NO'}")

    if status["imported_modules"]:
        print(f"\nImported Modules ({len(status['imported_modules'])}):")
        for module in status["imported_modules"]:
            print(f"  ✓ {module}")

    if status["errors"]:
        print(f"\nErrors ({len(status['errors'])}):")
        for error in status["errors"]:
            print(f"  ❌ {error}")

    if status["warnings"]:
        print(f"\nWarnings ({len(status['warnings'])}):")
        for warning in status["warnings"]:
            print(f"  ⚠️ {warning}")

    print("=" * 60)


if __name__ == "__main__":
    # Run diagnostic when called directly
    print_import_status()
