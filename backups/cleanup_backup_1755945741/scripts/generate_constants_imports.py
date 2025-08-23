#!/usr/bin/env python3
"""Generate automatic constants imports for magic literal replacement.

This script analyzes Python files and suggests import statements for
replacing magic literals with constants from our constants modules.
"""

import ast
import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConstantsMapper:
    """Maps magic literals to appropriate constants."""

    def __init__(self):
        self.constants_map = self._build_constants_map()

    def _build_constants_map(self) -> dict[str, dict[str, str]]:
        """Build a map of constants from our constants modules."""

        # Map common patterns to constants
        patterns = {
            # Security patterns
            r"(password|auth|token|secret|crypto).*\d+": "packages.core.security.constants",
            r"jwt.*algorithm.*hs256": "packages.core.security.constants.JWT_ALGORITHM",
            r"session.*timeout.*\d+": "packages.core.security.constants.SESSION_TIMEOUT_MINUTES",
            # Training patterns
            r"50.*million|50m|parameters.*1_000_000": "packages.core.training.constants.PARAMETERS_PER_MILLION",
            r"model.*size.*50.*million": "packages.core.training.constants.HRRM_MODEL_SIZE_50M",
            r"device.*cuda": "packages.core.training.constants.LogMessages.DEVICE_SELECTED",
            # Backup patterns
            r"backup.*\d+.*day": "packages.core.backup.constants.BACKUP_RETENTION_DAYS",
            r"sleep.*5": "packages.core.backup.constants.BACKUP_STATUS_CHECK_INTERVAL_SECONDS",
            # Cost management patterns
            r"cost.*hour|hour.*cost": "packages.core.cost_management.constants.COST_UPDATE_INTERVAL_MINUTES",
            r"budget.*\d+": "packages.core.cost_management.constants.DEFAULT_MONTHLY_BUDGET_USD",
            # Agent patterns
            r"agent.*timeout.*\d+": "packages.agents.core.components.constants.AGENT_STARTUP_TIMEOUT_SECONDS",
            r"heartbeat.*\d+": "packages.agents.core.components.constants.AGENT_HEARTBEAT_INTERVAL_SECONDS",
            # Common patterns
            r"timeout.*30|30.*second": "packages.core.common.constants.HTTP_TIMEOUT_SECONDS",
            r"port.*8080|8080": "packages.core.common.constants.DEFAULT_PORT",
        }

        return patterns

    def suggest_constant(self, literal_value: str, context: str) -> list[str]:
        """Suggest constants for a magic literal."""
        suggestions = []
        context_lower = context.lower()

        for pattern, constant in self.constants_map.items():
            if re.search(pattern, context_lower):
                suggestions.append(constant)

        # Fallback suggestions based on value patterns
        if not suggestions:
            if isinstance(literal_value, int | float):
                if "timeout" in context_lower:
                    suggestions.append("packages.core.common.constants.HTTP_TIMEOUT_SECONDS")
                elif "port" in context_lower:
                    suggestions.append("packages.core.common.constants.DEFAULT_PORT")
                elif "interval" in context_lower:
                    suggestions.append("packages.core.common.constants.CACHE_TTL_SECONDS")

        return suggestions


def generate_import_suggestions(file_path: Path) -> list[str]:
    """Generate import suggestions for a Python file."""
    mapper = ConstantsMapper()
    suggestions = []

    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
            lines = content.splitlines()

        tree = ast.parse(content, filename=str(file_path))

        # Analyze AST for magic literals
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, int | float | str):
                if should_analyze_literal(node.value):
                    line_content = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                    constants = mapper.suggest_constant(node.value, line_content)

                    for constant in constants:
                        suggestion = f"# Replace {node.value} with {constant}"
                        if suggestion not in suggestions:
                            suggestions.append(suggestion)

    except Exception as e:
        logger.warning(f"Error analyzing {file_path}: {e}")

    return suggestions


def should_analyze_literal(value) -> bool:
    """Determine if a literal should be analyzed."""
    # Skip common non-magic values
    if value in (0, 1, -1, True, False, None, "", "utf-8"):
        return False

    if isinstance(value, str) and len(value) <= 2:
        return False

    if isinstance(value, int | float) and abs(value) >= 2:
        return True

    if isinstance(value, str) and len(value) > 2:
        return True

    return False


def main():
    """Main function to generate import suggestions."""
    logger.info("Generating constants import suggestions...")

    # Analyze high-priority files
    priority_patterns = [
        "packages/core/security/*.py",
        "packages/core/training/scripts/*.py",
        "packages/core/backup/*.py",
        "packages/core/cost_management/*.py",
        "packages/agents/**/*.py",
    ]

    all_suggestions = {}

    for pattern in priority_patterns:
        for file_path in Path(".").glob(pattern):
            if file_path.name == "constants.py":
                continue

            suggestions = generate_import_suggestions(file_path)
            if suggestions:
                all_suggestions[str(file_path)] = suggestions

    # Write suggestions to file
    output_path = Path("quality_reports/constants_import_suggestions.txt")
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w") as f:
        f.write("CONSTANTS IMPORT SUGGESTIONS\n")
        f.write("=" * 50 + "\n\n")

        for file_path, suggestions in all_suggestions.items():
            f.write(f"FILE: {file_path}\n")
            f.write("-" * len(file_path) + "\n")

            for suggestion in suggestions:
                f.write(f"  {suggestion}\n")

            f.write("\n")

    logger.info(f"Generated suggestions for {len(all_suggestions)} files")
    logger.info(f"Suggestions saved to: {output_path}")


if __name__ == "__main__":
    main()
