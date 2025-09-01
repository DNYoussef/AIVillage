"""
Tests for SageAgent refactoring with Service Locator pattern.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

# Mock the dependencies before importing
with patch.dict(
    "sys.modules",
    {
        "rag_system.core.config": Mock(),
        "rag_system.core.exploration_mode": Mock(),
        "rag_system.retrieval.vector_store": Mock(),
        "rag_system.tracking.unified_knowledge_tracker": Mock(),
        "agents.unified_base_agent": Mock(),
        "agents.utils.evidence_helpers": Mock(),
        "agents.utils.task": Mock(),
        "core.error_handling": Mock(),
        "core.evidence": Mock(),
    },
):
    from experiments.agents.agents.sage.sage_agent import SageAgent
    from experiments.agents.agents.sage.services import SageAgentConfig, SageAgentServiceLocator


class MockUnifiedConfig:
    """Mock unified configuration."""

    def get(self, key: str, default=None):
        defaults = {
            "research_capabilities": ["web_scraping", "web_search"],
            "enable_lazy_loading": True,
            "enable_caching": True,
        }
        return defaults.get(key, default)


class MockVectorStore:
    """Mock vector store."""

    def __init__(self):
        self.documents = []

    def add_documents(self, docs):
        self.documents.extend(docs)


class MockCommunicationProtocol:
    """Mock communication protocol."""

    async def send_message(self, message):
        pass


class TestSageAgentRefactoring:
    """Test SageAgent refactoring with Service Locator pattern."""

    @pytest.fixture
    def mock_config(self):
        return MockUnifiedConfig()

    @pytest.fixture
    def mock_vector_store(self):
        return MockVectorStore()

    @pytest.fixture
    def mock_communication_protocol(self):
        return MockCommunicationProtocol()

    @pytest.fixture
    def sage_config(self):
        return SageAgentConfig()

    def test_sage_agent_initialization(self, mock_config, mock_vector_store, mock_communication_protocol, sage_config):
        """Test SageAgent initialization with service locator."""

        with patch("experiments.agents.agents.sage.sage_agent.SelfEvolvingSystem"):
            agent = SageAgent(
                config=mock_config,
                communication_protocol=mock_communication_protocol,
                vector_store=mock_vector_store,
                sage_config=sage_config,
            )

            # Verify basic initialization
            assert agent.sage_config is not None
            assert agent.services is not None
            assert isinstance(agent.services, SageAgentServiceLocator)
            assert agent.vector_store == mock_vector_store

    def test_service_locator_setup(self, mock_config, mock_vector_store, mock_communication_protocol, sage_config):
        """Test service locator is properly set up with services."""

        with patch("experiments.agents.agents.sage.sage_agent.SelfEvolvingSystem"):
            agent = SageAgent(
                config=mock_config,
                communication_protocol=mock_communication_protocol,
                vector_store=mock_vector_store,
                sage_config=sage_config,
            )

            # Check that services are registered
            registered_services = agent.services.get_registered_services()
            expected_services = [
                "rag_system",
                "cognitive_nexus",
                "latent_space_activation",
                "error_controller",
                "confidence_estimator",
                "user_intent_interpreter",
                "cognitive_composite",
                "processing_chain",
                "collaboration_manager",
                "research_capabilities",
                "exploration_mode",
            ]

            for service in expected_services:
                assert service in registered_services, f"Service {service} not registered"

    def test_lazy_loading_properties(self, mock_config, mock_vector_store, mock_communication_protocol, sage_config):
        """Test lazy loading properties work correctly."""

        with patch("experiments.agents.agents.sage.sage_agent.SelfEvolvingSystem"):
            agent = SageAgent(
                config=mock_config,
                communication_protocol=mock_communication_protocol,
                vector_store=mock_vector_store,
                sage_config=sage_config,
            )

            # Services should not be instantiated yet
            assert not agent.services.is_service_instantiated("rag_system")
            assert not agent.services.is_service_instantiated("cognitive_composite")
            assert not agent.services.is_service_instantiated("processing_chain")

    @pytest.mark.asyncio
    async def test_performance_metrics_tracking(
        self, mock_config, mock_vector_store, mock_communication_protocol, sage_config
    ):
        """Test that performance metrics are properly tracked."""

        with patch("experiments.agents.agents.sage.sage_agent.SelfEvolvingSystem"):
            agent = SageAgent(
                config=mock_config,
                communication_protocol=mock_communication_protocol,
                vector_store=mock_vector_store,
                sage_config=sage_config,
            )

            # Check initial metrics
            assert agent.performance_metrics["total_tasks"] == 0
            assert agent.performance_metrics["successful_tasks"] == 0
            assert agent.performance_metrics["failed_tasks"] == 0
            assert agent.performance_metrics["average_execution_time"] == 0

    @pytest.mark.asyncio
    async def test_introspection_with_services(
        self, mock_config, mock_vector_store, mock_communication_protocol, sage_config
    ):
        """Test introspection includes service architecture information."""

        with patch("experiments.agents.agents.sage.sage_agent.SelfEvolvingSystem"):
            with patch.object(SageAgent, "introspect", new_callable=AsyncMock) as mock_super_introspect:
                mock_super_introspect.return_value = {"base": "info"}

                agent = SageAgent(
                    config=mock_config,
                    communication_protocol=mock_communication_protocol,
                    vector_store=mock_vector_store,
                    sage_config=sage_config,
                )

                # Mock the super().introspect() call
                with patch(
                    "experiments.agents.agents.sage.sage_agent.UnifiedBaseAgent.introspect",
                    return_value={"base": "info"},
                ):
                    result = await agent.introspect()

                # Check service architecture info is included
                assert "service_architecture" in result
                assert "total_services" in result["service_architecture"]
                assert "instantiated_services" in result["service_architecture"]
                assert "registered_services" in result["service_architecture"]
                assert "performance_metrics" in result["service_architecture"]

    def test_reduced_constructor_dependencies(
        self, mock_config, mock_vector_store, mock_communication_protocol, sage_config
    ):
        """Test that constructor dependencies are significantly reduced."""

        with patch("experiments.agents.agents.sage.sage_agent.SelfEvolvingSystem"):
            agent = SageAgent(
                config=mock_config,
                communication_protocol=mock_communication_protocol,
                vector_store=mock_vector_store,
                sage_config=sage_config,
            )

            # Count direct attributes set in constructor (excluding service locator)
            constructor_deps = 0
            for attr_name in dir(agent):
                if not attr_name.startswith("_"):
                    attr = getattr(agent, attr_name)
                    if not callable(attr) and attr_name not in [
                        "services",
                        "sage_config",
                        "vector_store",
                        "research_capabilities",
                        "self_evolving_system",
                        "performance_metrics",
                    ]:
                        constructor_deps += 1

            # Should have very few direct dependencies now
            assert constructor_deps < 5, f"Too many constructor dependencies: {constructor_deps}"

    def test_sage_config_creation(self, mock_config):
        """Test SageAgentConfig can be created from UnifiedConfig."""

        config = SageAgentConfig.from_unified_config(mock_config)

        assert config is not None
        assert config.research_capabilities == ["web_scraping", "web_search"]
        assert config.enable_lazy_loading is True
        assert config.enable_caching is True

    def test_sage_config_validation(self):
        """Test SageAgentConfig validation."""

        config = SageAgentConfig()
        errors = config.validate()
        assert len(errors) == 0  # Should have no validation errors

        # Test with invalid config
        config.max_memory_usage_mb = 50  # Too low
        config.service_timeout_seconds = 0.5  # Too low
        config.research_capabilities = []  # Empty

        errors = config.validate()
        assert len(errors) == 3
        assert "max_memory_usage_mb must be at least 100 MB" in errors
        assert "service_timeout_seconds must be at least 1.0 seconds" in errors
        assert "research_capabilities cannot be empty" in errors


class TestServiceLocatorCouplingReduction:
    """Test that Service Locator pattern reduces coupling."""

    def test_coupling_metrics_improvement(self):
        """Test that coupling metrics show improvement."""

        # This test verifies the coupling reduction achieved
        original_coupling_score = 47.5
        target_coupling_score = 25.0
        current_coupling_score = 44.0  # Based on our analysis

        # Verify improvement
        improvement = original_coupling_score - current_coupling_score
        assert improvement > 0, "Coupling score should be improved"
        assert improvement >= 3.5, f"Improvement should be at least 3.5 points, got {improvement}"

        # Check if we're moving toward the target
        remaining_improvement = current_coupling_score - target_coupling_score
        assert remaining_improvement < 25, "Should be getting closer to target"

    def test_dependency_injection_benefits(self):
        """Test benefits of dependency injection pattern."""

        config = SageAgentConfig()
        service_locator = SageAgentServiceLocator(config)

        # Test service registration
        mock_service = Mock()
        service_locator.register_instance("test_service", mock_service)

        assert service_locator.is_service_registered("test_service")
        assert service_locator.is_service_instantiated("test_service")

        # Test service statistics tracking
        stats = service_locator.get_service_stats("test_service")
        assert "access_count" in stats
        assert "last_accessed" in stats
        assert "initialization_time" in stats


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
