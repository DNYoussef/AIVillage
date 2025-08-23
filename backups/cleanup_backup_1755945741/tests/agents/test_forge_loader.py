"""
Agent Forge Template Loader Tests - Validate template loading and agent creation

Tests the fixes for Prompt 4:
- Backward-compatible template path resolution
- TemplateNotFound error with remediation hints
- Validation of all 18 required agent types
- Proper agent creation and instantiation
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from production.agent_forge.agent_factory import AgentFactory, TemplateNotFoundError


class TestAgentForgeTemplateLoader:
    """Test suite for Agent Forge template loading functionality."""

    def create_test_template(self, agent_type: str) -> dict:
        """Create a test template for an agent type."""
        return {
            "name": f"{agent_type.title()} Agent",
            "role": f"Specialized {agent_type} agent for testing",
            "default_params": {"enabled": True, "timeout": 30, "max_retries": 3},
            "capabilities": [f"{agent_type}_primary", "communication", "monitoring"],
            "requirements": {"memory_mb": 256, "cpu_cores": 1},
        }

    def create_master_config(self, agent_types: list[str]) -> dict:
        """Create master config with specified agent types."""
        return {
            "version": "1.0",
            "total_agents": len(agent_types),
            "agent_types": agent_types,
            "deployment_modes": ["development", "staging", "production"],
        }

    @pytest.fixture
    def temp_templates_dir(self):
        """Create temporary template directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            template_dir = Path(temp_dir)

            # Create expected 18 agent types
            agent_types = [
                "king",
                "magi",
                "sage",
                "gardener",
                "sword_shield",
                "legal",
                "shaman",
                "oracle",
                "maker",
                "ensemble",
                "curator",
                "auditor",
                "medic",
                "sustainer",
                "navigator",
                "tutor",
                "polyglot",
                "strategist",
            ]

            # Create templates in current location (templates/*_template.json)
            for agent_type in agent_types:
                template_file = template_dir / f"{agent_type}_template.json"
                with open(template_file, "w") as f:
                    json.dump(self.create_test_template(agent_type), f, indent=2)

            # Create master config
            master_config = template_dir / "master_config.json"
            with open(master_config, "w") as f:
                json.dump(self.create_master_config(agent_types), f, indent=2)

            # Create empty agents directory for legacy compatibility
            (template_dir / "agents").mkdir(exist_ok=True)

            yield template_dir

    @pytest.fixture
    def legacy_templates_dir(self):
        """Create legacy template directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            template_dir = Path(temp_dir)
            agents_dir = template_dir / "agents"
            agents_dir.mkdir()

            # Create templates in legacy location (templates/agents/*.json)
            agent_types = ["king", "magi", "sage"]
            for agent_type in agent_types:
                template_file = agents_dir / f"{agent_type}.json"
                with open(template_file, "w") as f:
                    json.dump(self.create_test_template(agent_type), f, indent=2)

            yield template_dir

    def test_template_loading_current_location(self, temp_templates_dir):
        """Test loading templates from current location (*_template.json)."""
        with (
            patch("src.production.agent_forge.agent_factory.BaseMetaAgent"),
            patch("src.production.agent_forge.agent_factory.AgentRole"),
            patch("src.production.agent_forge.agent_factory.AgentSpecialization"),
        ):
            factory = AgentFactory(temp_templates_dir)

            # Verify all 18 templates loaded
            assert len(factory.templates) == 18

            # Verify specific templates
            assert "king" in factory.templates
            assert "magi" in factory.templates
            assert "strategist" in factory.templates

            # Verify template structure
            king_template = factory.templates["king"]
            assert king_template["name"] == "King Agent"
            assert "default_params" in king_template
            assert king_template["default_params"]["enabled"] is True

    def test_template_loading_legacy_location(self, legacy_templates_dir):
        """Test loading templates from legacy location (agents/*.json)."""
        with (
            patch("src.production.agent_forge.agent_factory.BaseMetaAgent"),
            patch("src.production.agent_forge.agent_factory.AgentRole"),
            patch("src.production.agent_forge.agent_factory.AgentSpecialization"),
        ):
            factory = AgentFactory(legacy_templates_dir)

            # Should load 3 templates from legacy location
            assert len(factory.templates) >= 3
            assert "king" in factory.templates
            assert "magi" in factory.templates
            assert "sage" in factory.templates

    def test_backward_compatibility_search_order(self):
        """Test that current location takes precedence over legacy."""
        with tempfile.TemporaryDirectory() as temp_dir:
            template_dir = Path(temp_dir)
            agents_dir = template_dir / "agents"
            agents_dir.mkdir()

            # Create template in legacy location
            legacy_template = agents_dir / "king.json"
            with open(legacy_template, "w") as f:
                json.dump({"name": "Legacy King", "version": "1.0"}, f)

            # Create template in current location
            current_template = template_dir / "king_template.json"
            with open(current_template, "w") as f:
                json.dump({"name": "Current King", "version": "2.0"}, f)

            with (
                patch("src.production.agent_forge.agent_factory.BaseMetaAgent"),
                patch("src.production.agent_forge.agent_factory.AgentRole"),
                patch("src.production.agent_forge.agent_factory.AgentSpecialization"),
            ):
                factory = AgentFactory(template_dir)

                # Current location should take precedence
                assert factory.templates["king"]["name"] == "Current King"
                assert factory.templates["king"]["version"] == "2.0"

    def test_template_not_found_error(self):
        """Test TemplateNotFoundError with proper remediation hints."""
        with tempfile.TemporaryDirectory() as temp_dir:
            template_dir = Path(temp_dir)

            # Create only a few templates (not all 18 required)
            template_file = template_dir / "king_template.json"
            with open(template_file, "w") as f:
                json.dump(self.create_test_template("king"), f)

            # Create master config expecting 18 agents
            master_config = template_dir / "master_config.json"
            with open(master_config, "w") as f:
                json.dump(
                    {
                        "version": "1.0",
                        "total_agents": 18,
                        "agent_types": ["king", "magi", "sage"],
                    },
                    f,
                )

            with (
                patch("src.production.agent_forge.agent_factory.BaseMetaAgent"),
                patch("src.production.agent_forge.agent_factory.AgentRole"),
                patch("src.production.agent_forge.agent_factory.AgentSpecialization"),
            ):
                with pytest.raises(TemplateNotFoundError) as exc_info:
                    AgentFactory(template_dir)

                error = exc_info.value
                assert "magi" in str(error)
                assert "sage" in str(error)
                assert "Available agent types: king" in str(error)
                assert "Remediation hints" in str(error)
                assert "_template.json" in str(error)

    def test_agent_creation_success(self, temp_templates_dir):
        """Test successful agent creation using templates."""
        with (
            patch("src.production.agent_forge.agent_factory.BaseMetaAgent") as MockBaseAgent,
            patch("src.production.agent_forge.agent_factory.AgentRole") as MockRole,
            patch("src.production.agent_forge.agent_factory.AgentSpecialization") as MockSpec,
        ):
            # Mock the base agent
            mock_agent = MagicMock()
            MockBaseAgent.return_value = mock_agent
            MockRole.return_value = "king"
            MockSpec.return_value = MagicMock()

            factory = AgentFactory(temp_templates_dir)

            # Test creating an agent
            agent = factory.create_agent("king")

            # Verify agent was created
            assert agent is not None
            assert hasattr(agent, "config")
            assert hasattr(agent, "name")
            assert agent.name == "King Agent"

    def test_agent_creation_with_config(self, temp_templates_dir):
        """Test agent creation with custom configuration."""
        with (
            patch("src.production.agent_forge.agent_factory.BaseMetaAgent") as MockBaseAgent,
            patch("src.production.agent_forge.agent_factory.AgentRole") as MockRole,
            patch("src.production.agent_forge.agent_factory.AgentSpecialization") as MockSpec,
        ):
            mock_agent = MagicMock()
            MockBaseAgent.return_value = mock_agent
            MockRole.return_value = "king"
            MockSpec.return_value = MagicMock()

            factory = AgentFactory(temp_templates_dir)

            # Test creating agent with custom config
            custom_config = {"custom_param": "test_value", "timeout": 60}
            agent = factory.create_agent("king", config=custom_config)

            # Verify custom config was applied
            assert agent.config["custom_param"] == "test_value"
            assert agent.config["timeout"] == 60
            # Default params should still be present
            assert agent.config["enabled"] is True

    def test_list_available_agents(self, temp_templates_dir):
        """Test listing available agent types."""
        with (
            patch("src.production.agent_forge.agent_factory.BaseMetaAgent"),
            patch("src.production.agent_forge.agent_factory.AgentRole"),
            patch("src.production.agent_forge.agent_factory.AgentSpecialization"),
        ):
            factory = AgentFactory(temp_templates_dir)
            available = factory.list_available_agents()

            # Should return all 18 agents
            assert len(available) == 18

            # Check structure
            king_info = next(a for a in available if a["id"] == "king")
            assert king_info["name"] == "King Agent"
            assert king_info["role"] == "Specialized king agent for testing"

    def test_agent_info_retrieval(self, temp_templates_dir):
        """Test getting detailed agent information."""
        with (
            patch("src.production.agent_forge.agent_factory.BaseMetaAgent"),
            patch("src.production.agent_forge.agent_factory.AgentRole"),
            patch("src.production.agent_forge.agent_factory.AgentSpecialization"),
        ):
            factory = AgentFactory(temp_templates_dir)

            king_info = factory.get_agent_info("king")

            assert king_info["name"] == "King Agent"
            assert "default_params" in king_info
            assert "capabilities" in king_info
            assert king_info["default_params"]["timeout"] == 30

    def test_required_agent_types_from_master_config(self, temp_templates_dir):
        """Test reading required agent types from master config."""
        with (
            patch("src.production.agent_forge.agent_factory.BaseMetaAgent"),
            patch("src.production.agent_forge.agent_factory.AgentRole"),
            patch("src.production.agent_forge.agent_factory.AgentSpecialization"),
        ):
            factory = AgentFactory(temp_templates_dir)
            required = factory.required_agent_types()

            assert len(required) == 18
            assert "king" in required
            assert "magi" in required
            assert "strategist" in required

    def test_invalid_agent_type_creation(self, temp_templates_dir):
        """Test error handling for invalid agent types."""
        with (
            patch("src.production.agent_forge.agent_factory.BaseMetaAgent"),
            patch("src.production.agent_forge.agent_factory.AgentRole"),
            patch("src.production.agent_forge.agent_factory.AgentSpecialization"),
        ):
            factory = AgentFactory(temp_templates_dir)

            with pytest.raises(ValueError) as exc_info:
                factory.create_agent("invalid_agent_type")

            assert "Unknown agent type: invalid_agent_type" in str(exc_info.value)
            assert "Available:" in str(exc_info.value)


class TestTemplateNotFoundError:
    """Test the TemplateNotFoundError exception specifically."""

    def test_error_message_formatting(self):
        """Test that error message is properly formatted."""
        agent_type = "missing_agent"
        searched_paths = [Path("/path/1"), Path("/path/2")]
        available_types = ["king", "magi", "sage"]

        error = TemplateNotFoundError(agent_type, searched_paths, available_types)

        error_str = str(error)
        assert "missing_agent" in error_str
        assert "/path/1" in error_str
        assert "/path/2" in error_str
        assert "king, magi, sage" in error_str
        assert "Remediation hints" in error_str
        assert "_template.json" in error_str

    def test_error_attributes(self):
        """Test that error attributes are set correctly."""
        agent_type = "test_agent"
        searched_paths = [Path("/test")]
        available_types = ["king"]

        error = TemplateNotFoundError(agent_type, searched_paths, available_types)

        assert error.agent_type == agent_type
        assert error.searched_paths == searched_paths
        assert error.available_types == available_types


if __name__ == "__main__":
    # Run basic smoke test
    print("🧪 Running Agent Forge Template Loader smoke test...")

    # Create minimal test setup
    with tempfile.TemporaryDirectory() as temp_dir:
        template_dir = Path(temp_dir)

        # Create a single template for testing
        king_template = template_dir / "king_template.json"
        with open(king_template, "w") as f:
            json.dump(
                {
                    "name": "King Agent",
                    "role": "Test king agent",
                    "default_params": {"enabled": True},
                },
                f,
            )

        try:
            with (
                patch("src.production.agent_forge.agent_factory.BaseMetaAgent"),
                patch("src.production.agent_forge.agent_factory.AgentRole"),
                patch("src.production.agent_forge.agent_factory.AgentSpecialization"),
            ):
                # This should fail due to missing templates
                factory = AgentFactory(template_dir)
                print("❌ Expected TemplateNotFoundError was not raised")

        except TemplateNotFoundError as e:
            print("✅ TemplateNotFoundError correctly raised:")
            print(f"   {str(e)[:200]}...")

        except Exception as e:
            print(f"❌ Unexpected error: {e}")

    print("🧪 Smoke test completed!")
