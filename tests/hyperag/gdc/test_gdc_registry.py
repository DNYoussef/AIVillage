"""
Unit tests for GDC Registry

Tests the GDC specification loading and management system.
"""

from pathlib import Path
import sys
import tempfile

import pytest
import yaml

from mcp_servers.hyperag.gdc.registry import (
    get_enabled_gdcs,
    get_gdcs_by_category,
    get_gdcs_by_severity,
    load_gdc_registry,
    validate_registry,
)
from mcp_servers.hyperag.gdc.specs import GDCSpec

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


class TestGDCRegistry:
    """Test suite for GDC registry functionality"""

    @pytest.fixture
    def sample_gdc_yaml(self):
        """Sample GDC YAML configuration for testing"""
        return [
            {
                "id": "GDC_CONFIDENCE_VIOLATION",
                "description": "Node confidence outside valid range",
                "cypher": "MATCH (n) WHERE n.confidence < 0 OR n.confidence > 1 RETURN n",
                "severity": "high",
                "suggested_action": "normalize_confidence",
                "category": "data_quality",
                "enabled": True,
            },
            {
                "id": "GDC_ORPHANED_HYPEREDGE",
                "description": "Hyperedge with insufficient participants",
                "cypher": "MATCH (h:Hyperedge) WHERE size(h.participants) < 2 RETURN h",
                "severity": "medium",
                "suggested_action": "delete_hyperedge",
                "category": "structural",
                "enabled": True,
            },
            {
                "id": "GDC_DISABLED_TEST",
                "description": "Disabled test GDC",
                "cypher": "MATCH (n) RETURN n",
                "severity": "low",
                "suggested_action": "test_action",
                "category": "test",
                "enabled": False,
            },
        ]

    @pytest.fixture
    def temp_yaml_file(self, sample_gdc_yaml):
        """Create temporary YAML file with sample GDC configuration"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(sample_gdc_yaml, f)
            return Path(f.name)

    def test_load_gdc_registry_success(self, temp_yaml_file):
        """Test successful loading of GDC registry"""
        registry = load_gdc_registry(temp_yaml_file)

        assert len(registry) == 3
        assert "GDC_CONFIDENCE_VIOLATION" in registry
        assert "GDC_ORPHANED_HYPEREDGE" in registry
        assert "GDC_DISABLED_TEST" in registry

        # Check specific GDC
        gdc = registry["GDC_CONFIDENCE_VIOLATION"]
        assert isinstance(gdc, GDCSpec)
        assert gdc.severity == "high"
        assert gdc.category == "data_quality"
        assert gdc.enabled is True

        # Cleanup
        temp_yaml_file.unlink()

    def test_load_gdc_registry_file_not_found(self):
        """Test handling of missing GDC file"""
        with pytest.raises(FileNotFoundError):
            load_gdc_registry(Path("/nonexistent/file.yaml"))

    def test_load_gdc_registry_invalid_yaml(self):
        """Test handling of invalid YAML content"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_file = Path(f.name)

        with pytest.raises(ValueError, match="Invalid YAML"):
            load_gdc_registry(temp_file)

        temp_file.unlink()

    def test_load_gdc_registry_not_list(self):
        """Test handling of YAML that's not a list"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"not": "a list"}, f)
            temp_file = Path(f.name)

        with pytest.raises(ValueError, match="must contain a list"):
            load_gdc_registry(temp_file)

        temp_file.unlink()

    def test_load_gdc_registry_missing_required_field(self):
        """Test handling of GDC specs with missing required fields"""
        incomplete_spec = [
            {
                "id": "GDC_INCOMPLETE",
                "description": "Incomplete GDC spec",
                # Missing cypher, severity, suggested_action
            }
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(incomplete_spec, f)
            temp_file = Path(f.name)

        with pytest.raises(ValueError, match="Missing required field"):
            load_gdc_registry(temp_file)

        temp_file.unlink()

    def test_load_gdc_registry_duplicate_ids(self):
        """Test handling of duplicate GDC IDs"""
        duplicate_specs = [
            {
                "id": "GDC_DUPLICATE",
                "description": "First duplicate",
                "cypher": "MATCH (n) RETURN n",
                "severity": "low",
                "suggested_action": "action1",
            },
            {
                "id": "GDC_DUPLICATE",
                "description": "Second duplicate",
                "cypher": "MATCH (m) RETURN m",
                "severity": "high",
                "suggested_action": "action2",
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(duplicate_specs, f)
            temp_file = Path(f.name)

        with pytest.raises(ValueError, match="Duplicate GDC ID"):
            load_gdc_registry(temp_file)

        temp_file.unlink()

    def test_get_gdcs_by_category(self, temp_yaml_file):
        """Test filtering GDCs by category"""
        registry = load_gdc_registry(temp_yaml_file)

        data_quality_gdcs = get_gdcs_by_category(registry, "data_quality")
        structural_gdcs = get_gdcs_by_category(registry, "structural")
        test_gdcs = get_gdcs_by_category(registry, "test")

        assert len(data_quality_gdcs) == 1
        assert len(structural_gdcs) == 1
        assert len(test_gdcs) == 1

        assert data_quality_gdcs[0].id == "GDC_CONFIDENCE_VIOLATION"
        assert structural_gdcs[0].id == "GDC_ORPHANED_HYPEREDGE"
        assert test_gdcs[0].id == "GDC_DISABLED_TEST"

        temp_yaml_file.unlink()

    def test_get_gdcs_by_severity(self, temp_yaml_file):
        """Test filtering GDCs by severity"""
        registry = load_gdc_registry(temp_yaml_file)

        high_gdcs = get_gdcs_by_severity(registry, "high")
        medium_gdcs = get_gdcs_by_severity(registry, "medium")
        low_gdcs = get_gdcs_by_severity(registry, "low")

        assert len(high_gdcs) == 1
        assert len(medium_gdcs) == 1
        assert len(low_gdcs) == 1

        assert high_gdcs[0].severity == "high"
        assert medium_gdcs[0].severity == "medium"
        assert low_gdcs[0].severity == "low"

        temp_yaml_file.unlink()

    def test_get_enabled_gdcs(self, temp_yaml_file):
        """Test filtering enabled GDCs"""
        registry = load_gdc_registry(temp_yaml_file)

        enabled_gdcs = get_enabled_gdcs(registry)

        assert len(enabled_gdcs) == 2  # Two enabled, one disabled
        enabled_ids = {gdc.id for gdc in enabled_gdcs}
        assert "GDC_CONFIDENCE_VIOLATION" in enabled_ids
        assert "GDC_ORPHANED_HYPEREDGE" in enabled_ids
        assert "GDC_DISABLED_TEST" not in enabled_ids

        temp_yaml_file.unlink()

    def test_validate_registry_success(self, temp_yaml_file):
        """Test registry validation with valid configuration"""
        registry = load_gdc_registry(temp_yaml_file)
        issues = validate_registry(registry)

        # Should have some expected warnings but no critical errors
        assert isinstance(issues, list)

        temp_yaml_file.unlink()

    def test_validate_registry_write_operations(self):
        """Test registry validation detects write operations"""
        registry = {
            "GDC_WRITE_TEST": GDCSpec(
                id="GDC_WRITE_TEST",
                description="GDC with write operation",
                cypher="CREATE (n:BadNode) RETURN n",
                severity="high",
                suggested_action="test",
            )
        }

        issues = validate_registry(registry)

        # Should detect write operation
        write_issues = [issue for issue in issues if "write operation" in issue]
        assert len(write_issues) > 0

    def test_validate_registry_empty_cypher(self):
        """Test registry validation detects empty Cypher queries"""
        registry = {
            "GDC_EMPTY_CYPHER": GDCSpec(
                id="GDC_EMPTY_CYPHER",
                description="GDC with empty cypher",
                cypher="   ",  # Empty/whitespace only
                severity="medium",
                suggested_action="test",
            )
        }

        issues = validate_registry(registry)

        # Should detect empty Cypher
        empty_cypher_issues = [issue for issue in issues if "Empty Cypher" in issue]
        assert len(empty_cypher_issues) > 0

    def test_validate_registry_no_high_severity(self):
        """Test registry validation detects missing high-severity GDCs"""
        registry = {
            "GDC_LOW_ONLY": GDCSpec(
                id="GDC_LOW_ONLY",
                description="Only low severity GDC",
                cypher="MATCH (n) RETURN n",
                severity="low",
                suggested_action="test",
            )
        }

        issues = validate_registry(registry)

        # Should detect missing high-severity GDCs
        high_severity_issues = [
            issue for issue in issues if "No high-severity" in issue
        ]
        assert len(high_severity_issues) > 0

    def test_registry_reload_functionality(self, temp_yaml_file):
        """Test registry reload functionality"""
        from mcp_servers.hyperag.gdc import registry

        # Mock the global registry
        original_registry = registry.GDC_REGISTRY.copy()

        try:
            # Reload with test file
            registry.reload_registry(temp_yaml_file)

            # Check that global registry was updated
            assert len(registry.GDC_REGISTRY) == 3
            assert "GDC_CONFIDENCE_VIOLATION" in registry.GDC_REGISTRY

        finally:
            # Restore original registry
            registry.GDC_REGISTRY = original_registry
            temp_yaml_file.unlink()


class TestGDCSpecValidation:
    """Test suite for GDC specification validation"""

    def test_valid_gdc_spec_creation(self):
        """Test creation of valid GDC spec"""
        spec = GDCSpec(
            id="GDC_VALID_TEST",
            description="Valid test specification",
            cypher="MATCH (n:TestNode) RETURN n",
            severity="medium",
            suggested_action="test_action",
            category="test",
            enabled=True,
        )

        assert spec.id == "GDC_VALID_TEST"
        assert spec.severity == "medium"
        assert spec.category == "test"
        assert spec.enabled is True

    def test_gdc_spec_defaults(self):
        """Test default values in GDC spec"""
        spec = GDCSpec(
            id="GDC_DEFAULTS_TEST",
            description="Test defaults",
            cypher="MATCH (n) RETURN n",
            severity="low",
            suggested_action="test",
        )

        # Check default values
        assert spec.category == "general"
        assert spec.enabled is True
        assert spec.performance_hint == ""

    def test_gdc_spec_invalid_id(self):
        """Test GDC spec validation for invalid ID"""
        with pytest.raises(ValueError, match="GDC ID must start with 'GDC_'"):
            GDCSpec(
                id="INVALID_ID_FORMAT",
                description="Invalid ID",
                cypher="MATCH (n) RETURN n",
                severity="low",
                suggested_action="test",
            )

    def test_gdc_spec_invalid_severity(self):
        """Test GDC spec validation for invalid severity"""
        with pytest.raises(ValueError, match="Invalid severity"):
            GDCSpec(
                id="GDC_INVALID_SEVERITY",
                description="Invalid severity",
                cypher="MATCH (n) RETURN n",
                severity="extreme",
                suggested_action="test",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
