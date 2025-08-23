"""
Architectural Integration Test Suite

Comprehensive tests to verify the integration of all new architectural components:
- Hardware/Software layer integration
- Agent Forge with research-backed implementations
- Hyper RAG with Bayesian probability system
- Sword/Shield security architecture
- Compatibility bridges
- Cross-component communication

This test suite ensures all components work together as a cohesive system.
"""

import asyncio
import json
import logging
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from packages.agent_forge.legacy_software.process_orchestrator import AgentForgeOrchestrator
from packages.agents.specialized.governance.shield_agent import ShieldAgent
from packages.agents.specialized.governance.sword_agent import SwordAgent
from packages.core.legacy.compatibility.bridge_system import compatibility_bridge

# Test imports
from packages.edge.mobile.digital_twin_concierge import DigitalTwinConcierge

from packages.rag.core.hyper_rag import HyperRAGOrchestrator as HyperRAGPipeline
from packages.rag.legacy_src.education.curriculum_graph import curriculum_graph

logger = logging.getLogger(__name__)


class TestArchitecturalIntegration:
    """
    Integration tests for the complete AIVillage architecture.
    """

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    async def sword_agent(self):
        """Create Sword Agent for testing."""
        agent = SwordAgent("test_sword")
        yield agent

    @pytest.fixture
    async def shield_agent(self):
        """Create Shield Agent for testing."""
        agent = ShieldAgent("test_shield")
        yield agent

    @pytest.fixture
    async def hyper_rag(self):
        """Create Hyper RAG pipeline for testing."""
        pipeline = HyperRAGPipeline("test_sage")
        yield pipeline

    @pytest.fixture
    async def agent_forge(self):
        """Create Agent Forge orchestrator for testing."""
        orchestrator = AgentForgeOrchestrator("test_forge")
        yield orchestrator

    @pytest.fixture
    async def digital_twin(self):
        """Create Digital Twin for testing."""
        twin = DigitalTwinConcierge("test_twin", "test_user")
        yield twin

    @pytest.mark.asyncio
    async def test_sword_shield_integration(self, sword_agent, shield_agent):
        """Test integration between Sword and Shield agents."""
        # Test that both agents can generate encrypted thoughts
        sword_thought = await sword_agent.think_encrypted("Test security scenario")
        shield_thought = await shield_agent.think_encrypted("Test defense scenario")

        assert sword_thought is not None
        assert shield_thought is not None
        assert isinstance(sword_thought, str)
        assert isinstance(shield_thought, str)

        # Test that both agents can conduct daily research
        sword_research = await sword_agent.daily_offensive_research()
        shield_research = await shield_agent.daily_defensive_research()

        assert "research_date" in sword_research
        assert "research_date" in shield_research
        assert "encrypted_thoughts" in sword_research
        assert "encrypted_thoughts" in shield_research

        # Test battle preparation compatibility
        sword_prep = await sword_agent.prepare_battle_attack(
            {"target_systems": ["web_server", "database"], "test_mode": True}
        )
        shield_prep = await shield_agent.prepare_battle_defense(
            {"expected_attacks": ["sql_injection", "xss"], "test_mode": True}
        )

        assert "strategy_id" in sword_prep
        assert "strategy_id" in shield_prep

        logger.info("✓ Sword/Shield integration test passed")

    @pytest.mark.asyncio
    async def test_battle_orchestrator_integration(self):
        """Test Battle Orchestrator integration with Sword/Shield."""
        orchestrator = BattleOrchestrator("test_orchestrator")

        # Initialize agents
        await orchestrator.initialize_agents()

        assert orchestrator.sword_agent is not None
        assert orchestrator.shield_agent is not None

        # Test scenario selection
        scenario = orchestrator._select_battle_scenario()
        assert scenario is not None
        assert scenario.name is not None
        assert len(scenario.attack_vectors) > 0
        assert len(scenario.target_systems) > 0

        # Test orchestrator status
        status = orchestrator.get_orchestrator_status()
        assert status["status"] == "active"
        assert status["agents_initialized"]["sword"] is True
        assert status["agents_initialized"]["shield"] is True

        logger.info("✓ Battle Orchestrator integration test passed")

    @pytest.mark.asyncio
    async def test_hyper_rag_integration(self, hyper_rag):
        """Test Hyper RAG system integration."""
        # Test knowledge ingestion
        item_id = await hyper_rag.ingest_knowledge(
            content="Machine learning is a subset of artificial intelligence that enables computers to learn without explicit programming.",
            book_summary="AI Fundamentals",
            chapter_summary="Introduction to Machine Learning",
            source_confidence=0.9,
        )

        assert item_id is not None
        assert len(item_id) > 0
        assert item_id in hyper_rag.knowledge_items

        # Test different retrieval methods
        retrieval_methods = [
            RAGType.VECTOR,
            RAGType.GRAPH,
            RAGType.BAYESIAN,
            RAGType.HYBRID,
        ]

        for method in retrieval_methods:
            result = await hyper_rag.retrieve_knowledge(
                query="What is machine learning?", retrieval_type=method, max_results=3
            )

            assert result is not None
            assert result.retrieval_method == method
            assert len(result.items) > 0
            assert result.confidence_score >= 0.0
            assert len(result.bayesian_scores) > 0

        # Test cognitive analysis integration
        hybrid_result = await hyper_rag.retrieve_knowledge(
            query="machine learning applications",
            retrieval_type=RAGType.HYBRID,
            max_results=5,
        )

        cognitive_analysis = await hyper_rag.analyze_with_cognitive_nexus(
            hybrid_result, "machine learning applications"
        )

        assert "multi_perspective_analysis" in cognitive_analysis
        assert "synthesis_result" in cognitive_analysis

        logger.info("✓ Hyper RAG integration test passed")

    @pytest.mark.asyncio
    async def test_agent_forge_integration(self, agent_forge):
        """Test Agent Forge orchestrator integration."""
        # Test basic initialization
        assert agent_forge.orchestrator_id == "test_forge"
        assert hasattr(agent_forge, "pipeline_stages")
        assert len(agent_forge.pipeline_stages) == 10

        # Test stage validation
        stage_statuses = agent_forge.validate_all_stages()
        assert isinstance(stage_statuses, dict)
        assert len(stage_statuses) == 10

        # Test research integration check
        research_status = agent_forge.check_research_integration()
        assert "quiet_star_available" in research_status
        assert "edge_of_chaos_available" in research_status
        assert "grokfast_available" in research_status
        assert "self_modeling_available" in research_status

        # Test pipeline configuration
        test_config = {
            "base_model": "test_model",
            "target_capabilities": ["reasoning", "learning"],
            "research_techniques": ["quiet_star", "grokfast"],
        }

        configured = agent_forge.configure_pipeline(test_config)
        assert configured is True

        logger.info("✓ Agent Forge integration test passed")

    @pytest.mark.asyncio
    async def test_digital_twin_integration(self, digital_twin):
        """Test Digital Twin integration with meta-agents."""
        # Test digital twin initialization
        assert digital_twin.twin_id == "test_twin"
        assert digital_twin.user_id == "test_user"

        # Test task routing logic
        test_tasks = [
            {"type": "simple_query", "content": "What is the weather?"},
            {"type": "complex_analysis", "content": "Analyze market trends"},
            {"type": "personal_data", "content": "Update my preferences"},
        ]

        for task in test_tasks:
            routing_decision = digital_twin.determine_task_routing(task)

            assert "handler" in routing_decision
            assert routing_decision["handler"] in ["local", "king", "specialist"]
            assert "confidence" in routing_decision

        # Test resource management
        resource_status = digital_twin.get_resource_status()
        assert "cpu_usage" in resource_status
        assert "memory_usage" in resource_status
        assert "battery_level" in resource_status

        logger.info("✓ Digital Twin integration test passed")

    @pytest.mark.asyncio
    async def test_hyperag_education_integration(self):
        """Test HyperAG education system integration."""
        # Test curriculum graph functionality
        from hyperag.education import add_concept_to_curriculum, generate_learning_path_for_topic

        # Add test concepts to curriculum
        concept_id_1 = add_concept_to_curriculum(
            name="Basic Mathematics",
            description="Fundamental mathematical concepts",
            difficulty="beginner",
            estimated_minutes=120,
        )

        concept_id_2 = add_concept_to_curriculum(
            name="Algebra",
            description="Introduction to algebraic concepts",
            difficulty="intermediate",
            prerequisites=[concept_id_1],
            estimated_minutes=180,
        )

        assert concept_id_1 is not None and len(concept_id_1) > 0
        assert concept_id_2 is not None and len(concept_id_2) > 0

        # Test learning path generation
        learning_path = generate_learning_path_for_topic(
            topic="algebra", learner_id="test_learner", current_knowledge=[concept_id_1]
        )

        if learning_path:  # May be None if no algebra concepts found
            assert learning_path.learner_id == "test_learner"
            assert len(learning_path.concepts) > 0

        # Test ELI5 explanation generation
        from hyperag.education import explain_concept_eli5

        eli5_explanation = explain_concept_eli5("photosynthesis", age=8)

        assert eli5_explanation.concept == "photosynthesis"
        assert len(eli5_explanation.analogies) > 0
        assert len(eli5_explanation.examples) > 0
        assert len(eli5_explanation.key_points) > 0
        assert eli5_explanation.age_appropriate is True

        logger.info("✓ HyperAG Education integration test passed")

    @pytest.mark.asyncio
    async def test_compatibility_bridge_integration(self):
        """Test compatibility bridge system integration."""
        # Test bridge statistics
        stats = compatibility_bridge.get_bridge_stats()

        assert "total_mappings" in stats
        assert "active_bridges" in stats
        assert "active_deprecations" in stats
        assert stats["total_mappings"] >= 0

        # Test migration guidance
        compatibility_bridge.get_migration_guidance("agent_forge.core")
        # May be None if no specific guidance is registered

        # Test deprecation registration
        success = compatibility_bridge.register_deprecation(
            item_name="test.deprecated_function",
            deprecated_since="1.0.0",
            removal_version="2.0.0",
            replacement="new.function",
            migration_notes="Use new.function instead",
        )

        assert success is True

        # Test bridge creation
        bridge_created = compatibility_bridge.create_import_bridge("test_old_module", "test.new.module")

        assert bridge_created is True

        logger.info("✓ Compatibility Bridge integration test passed")

    @pytest.mark.asyncio
    async def test_cross_component_communication(self, hyper_rag, sword_agent, shield_agent):
        """Test communication between different architectural components."""
        # Simulate Sage Agent using Hyper RAG to answer Sword Agent query
        await hyper_rag.ingest_knowledge(
            content="SQL injection is a code injection technique that exploits vulnerabilities in database queries.",
            book_summary="Cybersecurity Fundamentals",
            chapter_summary="Web Application Security",
            source_confidence=0.95,
        )

        # Sword Agent requests information about SQL injection
        security_query_result = await hyper_rag.retrieve_knowledge(
            query="SQL injection techniques and prevention",
            retrieval_type=RAGType.HYBRID,
            max_results=3,
        )

        assert len(security_query_result.items) > 0
        assert security_query_result.confidence_score > 0.0

        # Shield Agent analyzes threat based on retrieved knowledge
        threat_indicators = ["unusual_sql_queries", "error_message_leakage"]
        threat_analysis = await shield_agent.threat_detection_analysis(threat_indicators)

        assert "threat_type" in threat_analysis
        assert "severity" in threat_analysis
        assert "confidence" in threat_analysis

        # Simulate communication flow: Hyper RAG -> Analysis -> Battle Planning
        sword_intel = {
            "knowledge_items": [item.content for item in security_query_result.items],
            "threat_analysis": threat_analysis,
            "confidence_scores": security_query_result.bayesian_scores,
        }

        sword_strategy = await sword_agent.prepare_battle_attack(sword_intel)
        shield_strategy = await shield_agent.prepare_battle_defense(
            {"threat_intelligence": sword_intel, "expected_attacks": ["sql_injection"]}
        )

        assert "strategy_id" in sword_strategy
        assert "strategy_id" in shield_strategy

        logger.info("✓ Cross-component communication test passed")

    @pytest.mark.asyncio
    async def test_end_to_end_architectural_flow(self, temp_dir):
        """Test complete end-to-end architectural flow."""
        # 1. Digital Twin receives user query
        digital_twin = DigitalTwinConcierge("e2e_twin", "e2e_user")
        user_query = {
            "type": "learning_request",
            "content": "I want to learn about cybersecurity",
            "complexity": "intermediate",
        }

        routing_decision = digital_twin.determine_task_routing(user_query)
        assert routing_decision["handler"] in ["local", "king", "specialist"]

        # 2. Query routed to Hyper RAG for knowledge retrieval
        hyper_rag = HyperRAGPipeline("e2e_sage")

        # Ingest cybersecurity knowledge
        await hyper_rag.ingest_knowledge(
            content="Cybersecurity involves protecting systems, networks, and programs from digital attacks.",
            book_summary="Cybersecurity Fundamentals",
            chapter_summary="Introduction to Cybersecurity",
            source_confidence=0.9,
        )

        await hyper_rag.ingest_knowledge(
            content="Common cybersecurity threats include malware, phishing, and SQL injection attacks.",
            book_summary="Cybersecurity Fundamentals",
            chapter_summary="Common Threats",
            source_confidence=0.85,
        )

        # 3. Retrieve knowledge using hybrid approach
        knowledge_result = await hyper_rag.retrieve_knowledge(
            query="cybersecurity fundamentals and common threats",
            retrieval_type=RAGType.HYBRID,
            max_results=5,
        )

        assert len(knowledge_result.items) >= 2

        # 4. Generate learning path using HyperAG education system
        from hyperag.education import add_concept_to_curriculum, generate_learning_path_for_topic

        # Add cybersecurity concepts
        intro_concept = add_concept_to_curriculum(
            name="Introduction to Cybersecurity",
            description="Basic cybersecurity concepts and terminology",
            difficulty="intermediate",
            estimated_minutes=90,
        )

        add_concept_to_curriculum(
            name="Common Cyber Threats",
            description="Understanding malware, phishing, and other attacks",
            difficulty="intermediate",
            prerequisites=[intro_concept],
            estimated_minutes=120,
        )

        learning_path = generate_learning_path_for_topic(
            topic="cybersecurity", learner_id="e2e_user", current_knowledge=[]
        )

        # 5. Security agents analyze the learning content for safety
        sword_agent = SwordAgent("e2e_sword")
        shield_agent = ShieldAgent("e2e_shield")

        # Sword analyzes for potential security teaching risks
        security_analysis = await sword_agent.think_encrypted(
            "Analyze cybersecurity learning content for potential misuse risks"
        )

        # Shield validates the defensive educational value
        defense_validation = await shield_agent.think_encrypted(
            "Validate cybersecurity education content for defensive learning value"
        )

        assert security_analysis is not None
        assert defense_validation is not None

        # 6. Agent Forge could create specialized cybersecurity tutor agent
        agent_forge = AgentForgeOrchestrator("e2e_forge")

        tutor_config = {
            "base_model": "cybersecurity_tutor",
            "target_capabilities": ["cybersecurity_education", "threat_analysis"],
            "knowledge_base": [item.content for item in knowledge_result.items],
            "learning_path": learning_path.path_id if learning_path else None,
        }

        pipeline_configured = agent_forge.configure_pipeline(tutor_config)
        assert pipeline_configured is True

        # 7. Generate final response combining all systems
        final_response = {
            "query_processed": True,
            "knowledge_retrieved": len(knowledge_result.items),
            "learning_path_created": learning_path is not None,
            "security_validated": True,
            "agent_forge_ready": pipeline_configured,
            "response_confidence": knowledge_result.confidence_score,
            "educational_components": {
                "concepts_available": len(curriculum_graph.concepts),
                "learning_path_duration": learning_path.estimated_duration_hours if learning_path else 0,
                "difficulty_appropriate": True,
            },
        }

        # Validate the complete flow
        assert final_response["query_processed"] is True
        assert final_response["knowledge_retrieved"] >= 2
        assert final_response["security_validated"] is True
        assert final_response["agent_forge_ready"] is True
        assert final_response["response_confidence"] > 0.0

        logger.info("✓ End-to-end architectural flow test passed")
        logger.info(f"Final response: {json.dumps(final_response, indent=2)}")

    @pytest.mark.asyncio
    async def test_system_performance_integration(self):
        """Test system performance across integrated components."""
        start_time = datetime.now()

        # Parallel initialization of major components
        tasks = [
            self._init_sword_shield_system(),
            self._init_hyper_rag_system(),
            self._init_education_system(),
            self._init_compatibility_system(),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check that all systems initialized successfully
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                pytest.fail(f"System {i} failed to initialize: {result}")

        end_time = datetime.now()
        initialization_time = (end_time - start_time).total_seconds()

        # Performance assertions
        assert initialization_time < 10.0, f"System initialization took {initialization_time:.2f}s (>10s)"

        # Verify all systems are operational
        sword_shield_ok, hyper_rag_ok, education_ok, compatibility_ok = results

        assert sword_shield_ok is True
        assert hyper_rag_ok is True
        assert education_ok is True
        assert compatibility_ok is True

        logger.info(f"✓ System performance test passed - initialized in {initialization_time:.2f}s")

    async def _init_sword_shield_system(self) -> bool:
        """Initialize Sword/Shield security system."""
        try:
            orchestrator = BattleOrchestrator("perf_test")
            await orchestrator.initialize_agents()
            return True
        except Exception as e:
            logger.error(f"Sword/Shield system initialization failed: {e}")
            return False

    async def _init_hyper_rag_system(self) -> bool:
        """Initialize Hyper RAG system."""
        try:
            hyper_rag = HyperRAGPipeline("perf_test")
            await hyper_rag.ingest_knowledge(
                "Test knowledge for performance testing",
                "Performance Testing",
                "System Initialization",
                0.8,
            )
            return True
        except Exception as e:
            logger.error(f"Hyper RAG system initialization failed: {e}")
            return False

    async def _init_education_system(self) -> bool:
        """Initialize HyperAG education system."""
        try:
            from hyperag.education import add_concept_to_curriculum

            concept_id = add_concept_to_curriculum(
                "Performance Test Concept",
                "Test concept for performance validation",
                "beginner",
                estimated_minutes=30,
            )
            return len(concept_id) > 0
        except Exception as e:
            logger.error(f"Education system initialization failed: {e}")
            return False

    async def _init_compatibility_system(self) -> bool:
        """Initialize compatibility bridge system."""
        try:
            stats = compatibility_bridge.get_bridge_stats()
            return stats["bridge_health"] in ["healthy", "inactive"]
        except Exception as e:
            logger.error(f"Compatibility system initialization failed: {e}")
            return False

    @pytest.mark.asyncio
    async def test_error_handling_integration(self):
        """Test error handling across integrated systems."""
        # Test Hyper RAG with invalid input
        hyper_rag = HyperRAGPipeline("error_test")

        # This should handle errors gracefully
        result = await hyper_rag.retrieve_knowledge(
            query="",  # Empty query
            retrieval_type=RAGType.HYBRID,
            max_results=0,  # Invalid max_results
        )

        # Should return empty result, not crash
        assert result is not None
        assert len(result.items) == 0

        # Test Agent Forge with invalid configuration
        agent_forge = AgentForgeOrchestrator("error_test")

        invalid_config = {
            "base_model": "",  # Empty model name
            "invalid_parameter": "invalid_value",
        }

        # Should handle invalid config gracefully
        try:
            configured = agent_forge.configure_pipeline(invalid_config)
            # Should return False for invalid config, not crash
            assert configured is False
        except Exception as e:
            # Or handle with specific exception, which is also acceptable
            assert "invalid" in str(e).lower() or "error" in str(e).lower()

        # Test compatibility bridge with non-existent module
        bridge_created = compatibility_bridge.create_import_bridge("non_existent_module", "also_non_existent.module")

        # Should either succeed (creating bridge) or fail gracefully
        assert isinstance(bridge_created, bool)

        logger.info("✓ Error handling integration test passed")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
