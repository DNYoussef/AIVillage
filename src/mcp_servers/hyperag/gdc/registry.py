"""GDC Registry

Loads and manages Graph Denial Constraint specifications from YAML configuration.
"""

import logging
import pathlib

import yaml

from .specs import GDCSpec

logger = logging.getLogger(__name__)

# Default path to GDC rules configuration
_DEFAULT_GDC_YAML = (
    pathlib.Path(__file__).parent.parent.parent.parent / "config" / "gdc_rules.yaml"
)


def load_gdc_registry(config_path: pathlib.Path | None = None) -> dict[str, GDCSpec]:
    """Load GDC specifications from YAML file

    Args:
        config_path: Path to GDC rules YAML file (optional)

    Returns:
        Dictionary mapping GDC IDs to GDCSpec objects

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If GDC specification is invalid
    """
    if config_path is None:
        config_path = _DEFAULT_GDC_YAML

    if not config_path.exists():
        raise FileNotFoundError(f"GDC rules file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as f:
            raw_specs = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in GDC rules file: {e}")

    if not isinstance(raw_specs, list):
        raise ValueError("GDC rules file must contain a list of specifications")

    registry = {}

    for spec_data in raw_specs:
        try:
            # Validate required fields
            required_fields = [
                "id",
                "description",
                "cypher",
                "severity",
                "suggested_action",
            ]
            for field in required_fields:
                if field not in spec_data:
                    raise ValueError(
                        f"Missing required field '{field}' in GDC spec: {spec_data}"
                    )

            # Create GDCSpec object
            spec = GDCSpec(**spec_data)

            # Check for duplicate IDs
            if spec.id in registry:
                raise ValueError(f"Duplicate GDC ID: {spec.id}")

            registry[spec.id] = spec
            logger.debug(f"Loaded GDC: {spec.id} - {spec.description}")

        except (ValueError, TypeError) as e:
            logger.error(f"Failed to load GDC spec: {e}")
            raise ValueError(f"Invalid GDC specification: {e}")

    logger.info(f"Loaded {len(registry)} GDC specifications from {config_path}")
    return registry


def get_gdcs_by_category(registry: dict[str, GDCSpec], category: str) -> list[GDCSpec]:
    """Get all GDCs in a specific category"""
    return [spec for spec in registry.values() if spec.category == category]


def get_gdcs_by_severity(registry: dict[str, GDCSpec], severity: str) -> list[GDCSpec]:
    """Get all GDCs with a specific severity level"""
    return [spec for spec in registry.values() if spec.severity == severity]


def get_enabled_gdcs(registry: dict[str, GDCSpec]) -> list[GDCSpec]:
    """Get all enabled GDCs"""
    return [spec for spec in registry.values() if spec.enabled]


# Global registry - loaded on import
try:
    GDC_REGISTRY = load_gdc_registry()
except FileNotFoundError:
    logger.warning("GDC rules file not found. Creating empty registry.")
    GDC_REGISTRY = {}
except Exception as e:
    logger.error(f"Failed to load GDC registry: {e}")
    GDC_REGISTRY = {}


def reload_registry(config_path: pathlib.Path | None = None) -> None:
    """Reload the global GDC registry"""
    global GDC_REGISTRY
    GDC_REGISTRY = load_gdc_registry(config_path)


def validate_registry(registry: dict[str, GDCSpec]) -> list[str]:
    """Validate a GDC registry for common issues

    Returns:
        List of validation warnings/errors
    """
    issues = []

    # Check for missing categories
    categories = {spec.category for spec in registry.values()}
    if "critical" not in categories:
        issues.append("No critical GDCs defined")

    # Check for empty Cypher queries
    for spec in registry.values():
        if not spec.cypher.strip():
            issues.append(f"Empty Cypher query in {spec.id}")

        # Basic Cypher validation
        cypher_lower = spec.cypher.lower()
        if (
            "create " in cypher_lower
            or "delete " in cypher_lower
            or "merge " in cypher_lower
        ):
            issues.append(
                f"GDC {spec.id} contains write operation - should be read-only"
            )

    # Check severity distribution
    severities = [spec.severity for spec in registry.values()]
    if severities.count("high") == 0:
        issues.append("No high-severity GDCs defined")

    return issues
