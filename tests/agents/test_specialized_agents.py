"""
Comprehensive Test Suite for Specialized Agents
"""

import asyncio

import pandas as pd
import pytest
from packages.agents.specialized import (
    ArchitectAgent,
    CreativeAgent,
    DataScienceAgent,
    DevOpsAgent,
    FinancialAgent,
    SocialAgent,
    SpecializedAgentRegistry,
    TesterAgent,
    TranslatorAgent,
    get_global_registry,
)


class TestDataScienceAgent:
    """Test suite for DataScienceAgent"""

    @pytest.fixture
    async def agent(self):
        agent = DataScienceAgent()
        await agent.initialize()
        return agent

    async def test_agent_initialization(self, agent):
        """Test agent initializes correctly"""
        assert agent.agent_id == "data_science_agent"
        assert agent.agent_type == "DataScience"
        assert agent.initialized
        assert len(agent.capabilities) == 8

    async def test_generate_response(self, agent):
        """Test agent generates appropriate responses"""
        response = await agent.generate("I need statistical analysis")
        assert "statistical analysis" in response.lower()

        response = await agent.generate("train machine learning model")
        assert "ml models" in response.lower()

    async def test_introspection(self, agent):
        """Test agent introspection"""
        status = await agent.introspect()
        assert status["agent_id"] == "data_science_agent"
        assert status["agent_type"] == "DataScience"
        assert "capabilities" in status
        assert "initialized" in status

    async def test_statistical_analysis(self, agent):
        """Test statistical analysis functionality"""
        # Create sample data
        data = pd.DataFrame(
            {
                "A": [1, 2, 3, 4, 5],
                "B": [2, 4, 6, 8, 10],
                "C": ["x", "y", "x", "y", "x"],
            }
        )

        config = {"hypothesis_test": {"type": "t_test", "groups": "C", "value": "A"}}
        result = await agent.perform_statistical_analysis(data, config)

        assert "descriptive_stats" in result
        assert "correlations" in result
        assert "distributions" in result


class TestDevOpsAgent:
    """Test suite for DevOpsAgent"""

    @pytest.fixture
    async def agent(self):
        agent = DevOpsAgent()
        await agent.initialize()
        return agent

    async def test_deployment_simulation(self, agent):
        """Test deployment simulation"""
        from packages.agents.specialized.devops_agent import DeploymentRequest

        request = DeploymentRequest(environment="staging", service="test-service", version="v1.0.0")

        result = await agent.deploy_service(request)

        assert result["status"] in ["deployed", "failed"]
        assert result["service"] == "test-service"
        assert result["version"] == "v1.0.0"
        assert "deployment_id" in result

    async def test_infrastructure_monitoring(self, agent):
        """Test infrastructure monitoring"""
        metrics = await agent.monitor_infrastructure()

        assert "timestamp" in metrics
        assert "services" in metrics
        assert "infrastructure" in metrics
        assert "alerts" in metrics

    async def test_pipeline_management(self, agent):
        """Test pipeline management"""
        config = {
            "name": "test-pipeline",
            "stages": ["build", "test", "deploy"],
            "triggers": ["git_push"],
        }

        result = await agent.manage_pipeline(config)

        assert result["pipeline_id"] == "test-pipeline"
        assert result["status"] == "active"
        assert len(result["stages"]) == 3


class TestFinancialAgent:
    """Test suite for FinancialAgent"""

    @pytest.fixture
    async def agent(self):
        agent = FinancialAgent()
        await agent.initialize()
        return agent

    async def test_portfolio_optimization(self, agent):
        """Test portfolio optimization"""
        assets = ["AAPL", "GOOGL", "MSFT", "AMZN"]
        constraints = {"max_weight": 0.4, "min_weight": 0.1}

        result = await agent.optimize_portfolio(assets, constraints)

        assert "weights" in result
        assert "expected_return" in result
        assert "volatility" in result
        assert "sharpe_ratio" in result
        assert len(result["weights"]) == len(assets)

    async def test_risk_analysis(self, agent):
        """Test risk analysis"""
        portfolio_data = {"value": 1000000, "assets": ["AAPL", "GOOGL"]}

        result = await agent.calculate_risk_metrics(portfolio_data)

        assert "var_95" in result
        assert "var_99" in result
        assert "volatility" in result
        assert "beta" in result
        assert "risk_score" in result

    async def test_dcf_valuation(self, agent):
        """Test DCF valuation"""
        company_data = {
            "revenue": 1000000000,
            "growth_rate": 0.05,
            "discount_rate": 0.10,
            "terminal_growth": 0.03,
        }

        result = await agent.perform_dcf_valuation(company_data)

        assert "enterprise_value" in result
        assert "equity_value" in result
        assert "projected_cash_flows" in result
        assert "sensitivity_analysis" in result


class TestCreativeAgent:
    """Test suite for CreativeAgent"""

    @pytest.fixture
    async def agent(self):
        agent = CreativeAgent()
        await agent.initialize()
        return agent

    async def test_story_generation(self, agent):
        """Test story generation"""
        from packages.agents.specialized.creative_agent import CreativeRequest

        request = CreativeRequest(content_type="story", theme="adventure", style="fantasy", length="medium")

        result = await agent.generate_story(request)

        assert "title" in result
        assert "genre" in result
        assert "main_character" in result
        assert "plot_points" in result
        assert result["theme"] == "adventure"

    async def test_poetry_creation(self, agent):
        """Test poetry creation"""
        from packages.agents.specialized.creative_agent import CreativeRequest

        request = CreativeRequest(content_type="poem", theme="nature", style="haiku")

        result = await agent.create_poetry(request)

        assert "title" in result
        assert "style" in result
        assert "verses" in result
        assert result["theme"] == "nature"

    async def test_character_development(self, agent):
        """Test character development"""
        character_brief = {"name": "Elena", "role": "protagonist", "genre": "fantasy"}

        result = await agent.develop_character(character_brief)

        assert result["name"] == "Elena"
        assert result["role"] == "protagonist"
        assert "personality" in result
        assert "background" in result
        assert "character_arc" in result


class TestSocialAgent:
    """Test suite for SocialAgent"""

    @pytest.fixture
    async def agent(self):
        agent = SocialAgent()
        await agent.initialize()
        return agent

    async def test_content_moderation(self, agent):
        """Test content moderation"""
        content = "This is a helpful and constructive comment"
        context = {"content_id": "test_123"}

        result = await agent.moderate_community(content, context)

        assert "action" in result
        assert "confidence" in result
        assert result["content_id"] == "test_123"

    async def test_conflict_resolution(self, agent):
        """Test conflict resolution"""
        participants = ["user1", "user2"]
        issue = "Disagreement about project direction"

        result = await agent.resolve_conflict(participants, issue)

        assert "conflict_id" in result
        assert result["participants"] == participants
        assert "resolution_strategy" in result
        assert "recommended_actions" in result

    async def test_sentiment_monitoring(self, agent):
        """Test sentiment monitoring"""
        interactions = [
            {"content": "This is awesome! Great work!"},
            {"content": "I love this community"},
            {"content": "This is frustrating and awful"},
        ]

        result = await agent.monitor_sentiment(interactions)

        assert "overall_sentiment" in result
        assert "sentiment_trend" in result
        assert "community_health" in result


class TestTranslatorAgent:
    """Test suite for TranslatorAgent"""

    @pytest.fixture
    async def agent(self):
        agent = TranslatorAgent()
        await agent.initialize()
        return agent

    async def test_text_translation(self, agent):
        """Test text translation"""
        from packages.agents.specialized.translator_agent import TranslationRequest

        request = TranslationRequest(
            source_text="Hello, how are you?",
            source_language="en",
            target_language="es",
        )

        result = await agent.translate_text(request)

        assert "translated_text" in result
        assert "confidence_score" in result
        assert result["source_language"] == "en"
        assert result["target_language"] == "es"

    async def test_language_detection(self, agent):
        """Test language detection"""
        text = "Bonjour, comment allez-vous?"

        result = await agent.detect_language(text)

        assert "detected_language" in result
        assert "confidence" in result
        assert "alternative_languages" in result
        assert "text_statistics" in result

    async def test_content_localization(self, agent):
        """Test content localization"""
        content = "The event is on 12/25/2024 and costs $100"
        target_culture = "uk"

        result = await agent.localize_content(content, target_culture)

        assert "localized_content" in result
        assert "adaptations_applied" in result
        assert "cultural_considerations" in result
        assert result["target_culture"] == "uk"


class TestArchitectAgent:
    """Test suite for ArchitectAgent"""

    @pytest.fixture
    async def agent(self):
        agent = ArchitectAgent()
        await agent.initialize()
        return agent

    async def test_system_architecture_design(self, agent):
        """Test system architecture design"""
        requirements = {
            "scale": "medium",
            "complexity": "medium",
            "budget": "medium",
            "performance": "high",
            "team_expertise": "python",
        }

        result = await agent.design_system_architecture(requirements)

        assert "architecture_style" in result
        assert "components" in result
        assert "technology_stack" in result
        assert "deployment_strategy" in result
        assert "data_architecture" in result


class TestTesterAgent:
    """Test suite for TesterAgent"""

    @pytest.fixture
    async def agent(self):
        agent = TesterAgent()
        await agent.initialize()
        return agent

    async def test_unit_test_execution(self, agent):
        """Test unit test execution"""
        from packages.agents.specialized.tester_agent import TestRequest

        request = TestRequest(
            test_type="unit",
            target="test_module.py",
            parameters={"coverage_threshold": 0.8},
        )

        result = await agent.execute_test_suite(request)

        assert result["test_type"] == "unit"
        assert "total_tests" in result
        assert "passed" in result
        assert "failed" in result
        assert "success_rate" in result

    async def test_performance_testing(self, agent):
        """Test performance testing"""
        from packages.agents.specialized.tester_agent import TestRequest

        request = TestRequest(
            test_type="performance",
            target="api_endpoint",
            parameters={"concurrent_users": 50, "duration_minutes": 5},
        )

        result = await agent.execute_test_suite(request)

        assert result["test_type"] == "performance"
        assert "metrics" in result
        assert "resource_utilization" in result
        assert result["concurrent_users"] == 50

    async def test_test_strategy_generation(self, agent):
        """Test test strategy generation"""
        requirements = {
            "type": "web_application",
            "complexity": "medium",
            "timeline_weeks": 12,
        }

        result = await agent.generate_test_strategy(requirements)

        assert "test_pyramid" in result
        assert "automation_strategy" in result
        assert "quality_gates" in result
        assert "timeline" in result


class TestSpecializedAgentRegistry:
    """Test suite for SpecializedAgentRegistry"""

    @pytest.fixture
    async def registry(self):
        registry = SpecializedAgentRegistry()
        await registry.initialize()
        return registry

    async def test_registry_initialization(self, registry):
        """Test registry initializes correctly"""
        assert registry.initialized
        assert len(registry.agent_classes) == 8
        assert len(registry.capabilities) == 8

    async def test_agent_creation(self, registry):
        """Test agent creation through registry"""
        agent = await registry.get_or_create_agent("data_science")
        assert agent is not None
        assert agent.agent_type == "DataScience"

    async def test_capability_search(self, registry):
        """Test finding agents by capability"""
        agents = await registry.find_capable_agents("statistical_analysis")
        assert "data_science" in agents

        agents = await registry.find_capable_agents("deployment_automation")
        assert "devops" in agents

    async def test_request_routing(self, registry):
        """Test request routing to appropriate agent"""
        result = await registry.route_request("analyze_data", {"data": "sample"})
        assert "error" not in result or "response" in result

    async def test_multi_agent_coordination(self, registry):
        """Test multi-agent task coordination"""
        task = "Build and deploy a web application with tests"
        capabilities = [
            "system_architecture",
            "deployment_automation",
            "test_automation",
        ]

        result = await registry.coordinate_multi_agent_task(task, capabilities)

        assert "participating_agents" in result
        assert "results" in result
        assert len(result["participating_agents"]) <= len(capabilities)

    async def test_capability_documentation(self, registry):
        """Test capability documentation generation"""
        docs = registry.get_capability_documentation()

        assert "overview" in docs
        assert "agent_types" in docs
        assert "capabilities_by_domain" in docs
        assert "integration_examples" in docs


# Integration test
async def test_global_registry():
    """Test global registry functionality"""
    registry = await get_global_registry()
    assert registry is not None
    assert registry.initialized

    # Test agent creation
    agent = await registry.get_or_create_agent("data_science")
    assert agent is not None

    # Test status
    status = await registry.get_agent_status()
    assert status["registry_initialized"]


if __name__ == "__main__":
    # Run basic smoke tests
    async def run_smoke_tests():
        print("Running specialized agents smoke tests...")

        # Test each agent individually
        agents = [
            DataScienceAgent(),
            DevOpsAgent(),
            FinancialAgent(),
            CreativeAgent(),
            SocialAgent(),
            TranslatorAgent(),
            ArchitectAgent(),
            TesterAgent(),
        ]

        for agent in agents:
            try:
                await agent.initialize()
                await agent.generate("Test prompt")
                status = await agent.introspect()

                print(f"✅ {agent.agent_type} agent: OK")
                print(f"   Capabilities: {len(agent.capabilities)}")
                print(f"   Initialized: {status.get('initialized', False)}")

            except Exception as e:
                print(f"❌ {agent.agent_type} agent: FAILED - {e}")

        # Test registry
        try:
            registry = SpecializedAgentRegistry()
            await registry.initialize()
            status = await registry.get_agent_status()

            print("✅ Agent Registry: OK")
            print(f"   Total agents: {status['total_agents']}")
            print(f"   Available types: {len(status['available_agent_types'])}")

        except Exception as e:
            print(f"❌ Agent Registry: FAILED - {e}")

        print("\\nSmoke tests completed!")

    asyncio.run(run_smoke_tests())
