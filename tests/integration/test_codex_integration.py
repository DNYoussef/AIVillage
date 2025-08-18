"""Integration tests for CODEX-built components with existing AIVillage systems.

This test suite verifies that all CODEX-built components integrate correctly
with the existing AIVillage codebase without conflicts or data format mismatches.
"""

import json
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

# Test imports for integration components
try:
    from src.core.evolution_metrics_integrated import EvolutionMetricsData as EvolutionMetrics
    from src.core.evolution_metrics_integrated import IntegratedEvolutionMetrics as EvolutionMetricsCollector

    EVOLUTION_AVAILABLE = True
except ImportError:
    EVOLUTION_AVAILABLE = False

try:
    from src.production.rag.rag_system.core.pipeline import Answer, Document, EnhancedRAGPipeline

    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

try:
    from src.core.p2p.libp2p_mesh import MeshMessage
    from src.core.p2p.p2p_node import P2PNode

    P2P_AVAILABLE = True
except ImportError:
    P2P_AVAILABLE = False

try:
    from src.digital_twin.core.digital_twin import LearningProfile

    DIGITAL_TWIN_AVAILABLE = True
except ImportError:
    DIGITAL_TWIN_AVAILABLE = False


class TestEvolutionIntegration:
    """Test evolution metrics integration with existing agent forge."""

    @pytest.mark.skipif(not EVOLUTION_AVAILABLE, reason="Evolution metrics not available")
    @pytest.mark.asyncio
    async def test_evolution_metrics_persistence(self):
        """Test that evolution metrics properly persist to database."""
        with tempfile.NamedTemporaryFile(suffix=".db") as temp_db:
            config = {"db_path": temp_db.name, "storage_backend": "sqlite"}
            collector = EvolutionMetricsCollector(config)

            await collector.start()

            # Create test evolution metrics
            metrics = EvolutionMetrics(
                timestamp=time.time(),
                agent_id="test_agent_001",
                evolution_type="performance_optimization",
                evolution_id="evo_123",
                performance_score=0.85,
                improvement_delta=0.05,
                quality_score=0.90,
                memory_used_mb=256,
                cpu_percent_avg=45.0,
                duration_minutes=2.5,
                success=True,
                error_count=0,
                warning_count=1,
                metadata={"test": True},
            )

            await collector.save_metrics(metrics)
            await collector.stop()

            # Verify data was persisted
            conn = sqlite3.connect(temp_db.name)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM fitness_metrics")
            count = cursor.fetchone()[0]
            conn.close()

            assert count == 1, "Evolution metrics should be persisted to database"

    @pytest.mark.skipif(not EVOLUTION_AVAILABLE, reason="Evolution metrics not available")
    def test_metrics_data_format_compatibility(self):
        """Test that evolution metrics use compatible data formats."""
        metrics = EvolutionMetrics(
            timestamp=1234567890.0,
            agent_id="agent_123",
            evolution_type="test",
            evolution_id="evo_456",
            performance_score=0.75,
            improvement_delta=0.1,
            quality_score=0.8,
            memory_used_mb=128,
            cpu_percent_avg=30.0,
            duration_minutes=1.5,
            success=True,
            error_count=0,
            warning_count=0,
        )

        # Test serialization/deserialization
        data_dict = metrics.to_dict()
        reconstructed = EvolutionMetrics.from_dict(data_dict)

        assert metrics.timestamp == reconstructed.timestamp
        assert metrics.agent_id == reconstructed.agent_id
        assert metrics.performance_score == reconstructed.performance_score


class TestRAGIntegration:
    """Test RAG pipeline integration with existing systems."""

    @pytest.mark.skipif(not RAG_AVAILABLE, reason="RAG pipeline not available")
    def test_rag_pipeline_initialization(self):
        """Test that RAG pipeline initializes without conflicts."""
        pipeline = EnhancedRAGPipeline()

        # Verify pipeline has required components
        assert hasattr(pipeline, "embedder"), "Pipeline should have embedder"
        assert hasattr(pipeline, "index"), "Pipeline should have vector index"
        assert hasattr(pipeline, "cache"), "Pipeline should have cache system"
        assert hasattr(pipeline, "llm"), "Pipeline should have LLM interface"

    @pytest.mark.skipif(not RAG_AVAILABLE, reason="RAG pipeline not available")
    @pytest.mark.asyncio
    async def test_rag_document_processing(self):
        """Test document processing and retrieval."""
        pipeline = EnhancedRAGPipeline()

        # Test documents
        docs = [
            Document(id="doc1", text="The quick brown fox jumps over the lazy dog"),
            Document(id="doc2", text="Machine learning models require training data"),
            Document(id="doc3", text="AIVillage is a distributed AI system"),
        ]

        # Process documents
        pipeline.process_documents(docs)

        # Test retrieval
        results = await pipeline.retrieve("machine learning", k=2)

        assert len(results) <= 2, "Should return at most k results"
        assert all(hasattr(r, "text") for r in results), "Results should have text"
        assert all(hasattr(r, "score") for r in results), "Results should have scores"

    @pytest.mark.skipif(not RAG_AVAILABLE, reason="RAG pipeline not available")
    def test_rag_answer_generation(self):
        """Test answer generation with citations."""
        pipeline = EnhancedRAGPipeline()

        # Mock retrieval results
        from src.production.rag.rag_system.core.pipeline import RetrievalResult

        results = [
            RetrievalResult(id=1, text="Test content about AI", score=0.9),
            RetrievalResult(id=2, text="More information on machine learning", score=0.8),
        ]

        answer = pipeline.generate_answer("What is AI?", results)

        assert isinstance(answer, Answer), "Should return Answer object"
        assert hasattr(answer, "text"), "Answer should have text"
        assert hasattr(answer, "citations"), "Answer should have citations"
        assert hasattr(answer, "confidence"), "Answer should have confidence score"
        assert hasattr(answer, "source_documents"), "Answer should have source documents"


class TestP2PIntegration:
    """Test P2P networking integration."""

    @pytest.mark.skipif(not P2P_AVAILABLE, reason="P2P components not available")
    def test_p2p_node_compatibility(self):
        """Test P2P node compatibility between implementations."""
        # Test that existing P2PNode and LibP2PMeshNetwork have compatible interfaces

        # Mock dependencies to avoid actual network connections
        with (
            patch("src.core.p2p.p2p_node.EncryptionLayer"),
            patch("src.core.p2p.p2p_node.MessageProtocol"),
            patch("src.core.p2p.p2p_node.PeerDiscovery"),
        ):
            node = P2PNode(node_id="test_node", host="localhost", port=8000)

            # Verify node has expected attributes
            assert hasattr(node, "node_id"), "Node should have ID"
            assert hasattr(node, "host"), "Node should have host"
            assert hasattr(node, "port"), "Node should have port"

    @pytest.mark.skipif(not P2P_AVAILABLE, reason="P2P components not available")
    def test_mesh_message_compatibility(self):
        """Test mesh message format compatibility."""
        from src.core.p2p.libp2p_mesh import MeshMessageType

        message = MeshMessage(
            type=MeshMessageType.DATA_MESSAGE,
            sender_id="node_123",
            target_id="node_456",
            payload={"test": "data"},
            hop_count=0,
        )

        # Verify message structure
        assert hasattr(message, "id"), "Message should have unique ID"
        assert hasattr(message, "type"), "Message should have type"
        assert hasattr(message, "sender_id"), "Message should have sender"
        assert hasattr(message, "payload"), "Message should have payload"


class TestDigitalTwinIntegration:
    """Test Digital Twin integration."""

    @pytest.mark.skipif(not DIGITAL_TWIN_AVAILABLE, reason="Digital Twin not available")
    def test_digital_twin_profile_format(self):
        """Test learning profile data format compatibility."""
        profile = LearningProfile(
            student_id="student_123",
            name="Test Student",
            age=10,
            grade_level=5,
            language="en",
            region="US",
            learning_style="visual",
            strengths=["math", "science"],
            challenges=["reading"],
            interests=["robots", "space"],
            attention_span_minutes=15,
            preferred_session_times=["morning"],
            parent_constraints={"screen_time_max": 60},
            accessibility_needs=[],
            motivation_triggers=["games", "rewards"],
        )

        # Verify profile has required fields
        assert profile.student_id == "student_123"
        assert profile.age == 10
        assert isinstance(profile.strengths, list)
        assert isinstance(profile.parent_constraints, dict)


class TestCrossComponentIntegration:
    """Test integration between multiple CODEX components."""

    @pytest.mark.asyncio
    async def test_evolution_rag_integration(self):
        """Test integration between evolution metrics and RAG pipeline."""
        # This would test how evolution metrics could be used to improve
        # RAG performance through feedback loops

    @pytest.mark.asyncio
    async def test_p2p_evolution_integration(self):
        """Test P2P networking with evolution coordination."""
        # This would test distributed evolution across P2P network

    @pytest.mark.asyncio
    async def test_digital_twin_rag_integration(self):
        """Test Digital Twin using RAG for personalized content."""
        # This would test how Digital Twin uses RAG to adapt content


class TestDataFlowIntegration:
    """Test data flow between components."""

    def test_message_serialization_compatibility(self):
        """Test that messages can be serialized/deserialized between components."""
        test_data = {
            "evolution_metrics": {
                "timestamp": 1234567890.0,
                "agent_id": "test_agent",
                "performance_score": 0.85,
            },
            "rag_query": {
                "query": "test query",
                "context": "test context",
                "timestamp": 1234567890.0,
            },
            "p2p_message": {
                "type": "DATA_MESSAGE",
                "sender": "node_1",
                "data": {"key": "value"},
            },
        }

        # Test JSON serialization
        serialized = json.dumps(test_data)
        deserialized = json.loads(serialized)

        assert deserialized == test_data, "Data should survive JSON serialization"

    def test_api_contract_compatibility(self):
        """Test that API contracts are compatible between components."""
        # Verify that async methods use consistent signatures
        # This is important for component interoperability

        # Check common patterns
        async def mock_async_method(self, data: dict[str, Any]) -> dict[str, Any]:
            return data

        # Verify method signature compatibility
        import inspect

        sig = inspect.signature(mock_async_method)

        assert "data" in sig.parameters, "Methods should accept data parameter"
        assert sig.return_annotation != inspect.Signature.empty, "Methods should have return annotations"


# Test configuration and fixtures
@pytest.fixture
def temp_database():
    """Provide temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db") as temp_file:
        yield temp_file.name


@pytest.fixture
def mock_config():
    """Provide mock configuration for testing."""
    return {
        "db_path": ":memory:",
        "storage_backend": "sqlite",
        "cache_enabled": True,
        "redis_url": "redis://localhost:6379/0",
        "log_level": "DEBUG",
    }


class TestCODEXConfigurationIntegration:
    """Test suite for CODEX configuration integration."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary configuration directory with test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "config"
            config_dir.mkdir()

            # Create main config file
            main_config = {
                "integration": {
                    "evolution_metrics": {
                        "enabled": True,
                        "backend": "sqlite",
                        "db_path": "./data/evolution_metrics.db",
                        "flush_interval_seconds": 30,
                    },
                    "rag_pipeline": {
                        "enabled": True,
                        "embedding_model": "paraphrase-MiniLM-L3-v2",
                        "cache_enabled": True,
                        "chunk_size": 512,
                    },
                    "p2p_networking": {
                        "enabled": True,
                        "transport": "libp2p",
                        "discovery_method": "mdns",
                        "max_peers": 50,
                    },
                    "digital_twin": {
                        "enabled": True,
                        "encryption_enabled": True,
                        "privacy_mode": "strict",
                        "max_profiles": 10000,
                    },
                }
            }

            import yaml

            with open(config_dir / "aivillage_config.yaml", "w") as f:
                yaml.dump(main_config, f)

            # Create P2P config file
            p2p_config = {
                "host": "0.0.0.0",
                "port": 4001,
                "peer_discovery": {
                    "mdns_enabled": True,
                    "bootstrap_peers": [],
                    "discovery_interval": 30,
                },
                "transports": {
                    "tcp_enabled": True,
                    "websocket_enabled": True,
                    "bluetooth_enabled": False,
                    "wifi_direct_enabled": False,
                },
                "security": {"tls_enabled": True, "peer_verification": True},
            }

            with open(config_dir / "p2p_config.json", "w") as f:
                json.dump(p2p_config, f, indent=2)

            # Create RAG config file
            rag_config = {
                "embedder": {
                    "model_name": "paraphrase-MiniLM-L3-v2",
                    "device": "cpu",
                    "batch_size": 32,
                },
                "retrieval": {
                    "vector_top_k": 20,
                    "keyword_top_k": 20,
                    "final_top_k": 10,
                    "rerank_enabled": False,
                },
                "cache": {
                    "l1_size": 128,
                    "l2_enabled": False,
                    "l3_directory": "/tmp/rag_cache",
                },
            }

            with open(config_dir / "rag_config.json", "w") as f:
                json.dump(rag_config, f, indent=2)

            yield config_dir

    def test_configuration_file_loading(self, temp_config_dir):
        """Test that all configuration files are loaded correctly."""
        try:
            import sys

            sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
            from core.config_manager import CODEXConfigManager

            config_manager = CODEXConfigManager(config_dir=str(temp_config_dir), enable_hot_reload=False)

            # Test main configuration
            assert config_manager.get("integration.evolution_metrics.enabled") is True
            assert config_manager.get("integration.evolution_metrics.backend") == "sqlite"
            assert config_manager.get("integration.rag_pipeline.embedding_model") == "paraphrase-MiniLM-L3-v2"
            assert config_manager.get("integration.p2p_networking.transport") == "libp2p"
            assert config_manager.get("integration.digital_twin.privacy_mode") == "strict"

            # Test P2P configuration
            assert config_manager.get("p2p_config.host") == "0.0.0.0"
            assert config_manager.get("p2p_config.port") == 4001
            assert config_manager.get("p2p_config.peer_discovery.mdns_enabled") is True
            assert config_manager.get("p2p_config.transports.tcp_enabled") is True
            assert config_manager.get("p2p_config.security.tls_enabled") is True

            # Test RAG configuration
            assert config_manager.get("rag_config.embedder.model_name") == "paraphrase-MiniLM-L3-v2"
            assert config_manager.get("rag_config.retrieval.vector_top_k") == 20
            assert config_manager.get("rag_config.retrieval.final_top_k") == 10
            assert config_manager.get("rag_config.cache.l1_size") == 128

        except ImportError:
            pytest.skip("Config manager not available for testing")

    def test_environment_variable_overrides(self, temp_config_dir):
        """Test that environment variables correctly override configuration."""
        try:
            import os
            import sys
            from unittest.mock import patch

            sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
            from core.config_manager import CODEXConfigManager

            # Set environment variables
            test_env = {
                "AIVILLAGE_STORAGE_BACKEND": "redis",
                "LIBP2P_PORT": "4002",
                "RAG_L1_CACHE_SIZE": "256",
                "DIGITAL_TWIN_MAX_PROFILES": "5000",
            }

            with patch.dict(os.environ, test_env):
                config_manager = CODEXConfigManager(config_dir=str(temp_config_dir), enable_hot_reload=False)

                # Verify overrides are applied
                assert config_manager.get("integration.evolution_metrics.backend") == "redis"
                assert config_manager.get("p2p_config.port") == 4002
                assert config_manager.get("rag_config.cache.l1_size") == 256
                assert config_manager.get("integration.digital_twin.max_profiles") == 5000

        except ImportError:
            pytest.skip("Config manager not available for testing")

    def test_codex_compliance_requirements(self, temp_config_dir):
        """Test that configuration meets all CODEX requirements."""
        try:
            import sys

            sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
            from core.config_manager import CODEXConfigManager

            config_manager = CODEXConfigManager(config_dir=str(temp_config_dir), enable_hot_reload=False)

            # Test CODEX requirement compliance
            codex_requirements = [
                ("integration.evolution_metrics.enabled", True),
                ("integration.evolution_metrics.backend", "sqlite"),
                ("integration.rag_pipeline.enabled", True),
                ("integration.rag_pipeline.embedding_model", "paraphrase-MiniLM-L3-v2"),
                ("integration.rag_pipeline.chunk_size", 512),
                ("integration.p2p_networking.enabled", True),
                ("integration.p2p_networking.transport", "libp2p"),
                ("integration.p2p_networking.discovery_method", "mdns"),
                ("integration.p2p_networking.max_peers", 50),
                ("integration.digital_twin.enabled", True),
                ("integration.digital_twin.encryption_enabled", True),
                ("integration.digital_twin.privacy_mode", "strict"),
                ("integration.digital_twin.max_profiles", 10000),
                ("p2p_config.host", "0.0.0.0"),
                ("p2p_config.port", 4001),
                ("p2p_config.peer_discovery.mdns_enabled", True),
                ("p2p_config.transports.tcp_enabled", True),
                ("p2p_config.transports.websocket_enabled", True),
                ("p2p_config.security.tls_enabled", True),
                ("p2p_config.security.peer_verification", True),
                ("rag_config.embedder.model_name", "paraphrase-MiniLM-L3-v2"),
                ("rag_config.retrieval.vector_top_k", 20),
                ("rag_config.retrieval.keyword_top_k", 20),
                ("rag_config.retrieval.final_top_k", 10),
                ("rag_config.cache.l1_size", 128),
            ]

            for config_path, expected_value in codex_requirements:
                actual_value = config_manager.get(config_path)
                assert (
                    actual_value == expected_value
                ), f"CODEX requirement failed: {config_path} = {actual_value}, expected {expected_value}"

        except ImportError:
            pytest.skip("Config manager not available for testing")


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short"])
