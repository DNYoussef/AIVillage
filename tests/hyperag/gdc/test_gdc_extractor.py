"""
Unit tests for GDC Extractor

Tests the Graph Denial Constraint violation detection engine.
"""

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_servers.hyperag.gdc.extractor import GDCExtractor, GDCExtractorContext
from mcp_servers.hyperag.gdc.specs import GDCSpec, Violation

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


class TestGDCExtractor:
    """Test suite for GDCExtractor"""

    @pytest.fixture
    def mock_neo4j_driver(self):
        """Mock Neo4j driver for testing"""
        driver = AsyncMock()
        session = AsyncMock()
        driver.session.return_value.__aenter__.return_value = session
        driver.session.return_value.__aexit__.return_value = None
        return driver, session

    @pytest.fixture
    def sample_gdc_spec(self):
        """Sample GDC specification for testing"""
        return GDCSpec(
            id="GDC_TEST_VIOLATION",
            description="Test violation for unit testing",
            cypher="MATCH (n:TestNode) WHERE n.invalid = true RETURN n",
            severity="high",
            suggested_action="fix_test_violation",
            category="test",
        )

    @pytest.fixture
    def mock_neo4j_node(self):
        """Mock Neo4j Node object"""
        node = MagicMock()
        node.labels = ["TestNode"]
        node.id = 123
        node.items.return_value = [("id", "test-node-1"), ("invalid", True)]
        return node

    @pytest.fixture
    def mock_neo4j_relationship(self):
        """Mock Neo4j Relationship object"""
        rel = MagicMock()
        rel.type = "TEST_REL"
        rel.id = 456
        rel.start_node.id = 123
        rel.end_node.id = 789
        rel.items.return_value = [("weight", 0.5)]
        return rel

    def test_extractor_initialization(self):
        """Test GDCExtractor initialization"""
        extractor = GDCExtractor(
            neo4j_uri="bolt://localhost:7687",
            neo4j_auth=("neo4j", "password"),
            max_concurrent_queries=3,
            default_limit=500,
        )

        assert extractor.neo4j_uri == "bolt://localhost:7687"
        assert extractor.neo4j_auth == ("neo4j", "password")
        assert extractor.max_concurrent_queries == 3
        assert extractor.default_limit == 500
        assert extractor.driver is None

    @pytest.mark.asyncio
    async def test_extractor_connection_management(self, mock_neo4j_driver):
        """Test connection initialization and cleanup"""
        driver, session = mock_neo4j_driver

        with patch("mcp_servers.hyperag.gdc.extractor.AsyncGraphDatabase") as mock_db:
            mock_db.driver.return_value = driver

            extractor = GDCExtractor("bolt://localhost:7687", ("neo4j", "password"))

            # Test initialization
            await extractor.initialize()
            assert extractor.driver is not None
            mock_db.driver.assert_called_once_with(
                "bolt://localhost:7687", auth=("neo4j", "password")
            )

            # Test connection test
            session.run.return_value = AsyncMock()

            # Test cleanup
            await extractor.close()
            driver.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_record_to_violation_conversion(
        self, mock_neo4j_driver, sample_gdc_spec, mock_neo4j_node
    ):
        """Test conversion of Neo4j records to Violation objects"""
        driver, session = mock_neo4j_driver

        extractor = GDCExtractor("bolt://localhost:7687", ("neo4j", "password"))
        extractor.driver = driver

        # Mock record with a node
        record = {"n": mock_neo4j_node}

        violation = await extractor._record_to_violation(record, sample_gdc_spec)

        assert isinstance(violation, Violation)
        assert violation.gdc_id == "GDC_TEST_VIOLATION"
        assert violation.severity == "high"
        assert violation.suggested_repair == "fix_test_violation"
        assert len(violation.nodes) == 1
        assert violation.nodes[0]["id"] == "test-node-1"
        assert violation.nodes[0]["_labels"] == ["TestNode"]
        assert violation.nodes[0]["_neo4j_id"] == 123

    @pytest.mark.asyncio
    async def test_single_gdc_scan(
        self, mock_neo4j_driver, sample_gdc_spec, mock_neo4j_node
    ):
        """Test scanning for violations of a single GDC"""
        driver, session = mock_neo4j_driver

        # Mock query result
        mock_result = AsyncMock()
        mock_result.data.return_value = [{"n": mock_neo4j_node}]
        session.run.return_value = mock_result

        extractor = GDCExtractor("bolt://localhost:7687", ("neo4j", "password"))
        extractor.driver = driver

        violations = await extractor._scan_single_gdc(sample_gdc_spec, 100)

        assert len(violations) == 1
        assert violations[0].gdc_id == "GDC_TEST_VIOLATION"

        # Verify Cypher query was called correctly
        expected_cypher = "MATCH (n:TestNode) WHERE n.invalid = true RETURN n LIMIT 100"
        session.run.assert_called_once_with(expected_cypher)

    @pytest.mark.asyncio
    async def test_scan_specific_gdc(self, mock_neo4j_driver):
        """Test scanning a specific GDC by ID"""
        driver, session = mock_neo4j_driver

        # Mock empty result
        mock_result = AsyncMock()
        mock_result.data.return_value = []
        session.run.return_value = mock_result

        with patch.dict(
            "mcp_servers.hyperag.gdc.registry.GDC_REGISTRY",
            {
                "GDC_TEST": GDCSpec(
                    id="GDC_TEST",
                    description="Test GDC",
                    cypher="MATCH (n) RETURN n",
                    severity="low",
                    suggested_action="test_action",
                )
            },
        ):
            extractor = GDCExtractor("bolt://localhost:7687", ("neo4j", "password"))
            extractor.driver = driver

            violations = await extractor.scan_gdc("GDC_TEST", 50)

            assert violations == []
            session.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_scan_unknown_gdc(self, mock_neo4j_driver):
        """Test scanning an unknown GDC ID raises ValueError"""
        driver, session = mock_neo4j_driver

        extractor = GDCExtractor("bolt://localhost:7687", ("neo4j", "password"))
        extractor.driver = driver

        with pytest.raises(ValueError, match="Unknown GDC ID"):
            await extractor.scan_gdc("GDC_NONEXISTENT")

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, mock_neo4j_driver):
        """Test health check with healthy connection"""
        driver, session = mock_neo4j_driver

        # Mock successful health check
        mock_result = AsyncMock()
        mock_record = {"health": 1}
        mock_result.single.return_value = mock_record
        session.run.return_value = mock_result

        extractor = GDCExtractor("bolt://localhost:7687", ("neo4j", "password"))
        extractor.driver = driver

        health = await extractor.health_check()

        assert health["status"] == "healthy"
        assert health["neo4j_uri"] == "bolt://localhost:7687"

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self):
        """Test health check without driver"""
        extractor = GDCExtractor("bolt://localhost:7687", ("neo4j", "password"))

        health = await extractor.health_check()

        assert health["status"] == "disconnected"
        assert "error" in health

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_neo4j_driver):
        """Test GDCExtractorContext context manager"""
        driver, session = mock_neo4j_driver

        with patch("mcp_servers.hyperag.gdc.extractor.AsyncGraphDatabase") as mock_db:
            mock_db.driver.return_value = driver

            async with GDCExtractorContext(
                "bolt://localhost:7687", ("neo4j", "password")
            ) as extractor:
                assert isinstance(extractor, GDCExtractor)
                assert extractor.driver is not None

            # Verify cleanup was called
            driver.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_with_limit_addition(self, mock_neo4j_driver, sample_gdc_spec):
        """Test that LIMIT is properly added to Cypher queries"""
        driver, session = mock_neo4j_driver

        mock_result = AsyncMock()
        mock_result.data.return_value = []
        session.run.return_value = mock_result

        extractor = GDCExtractor("bolt://localhost:7687", ("neo4j", "password"))
        extractor.driver = driver

        await extractor._scan_single_gdc(sample_gdc_spec, 500)

        # Verify LIMIT was added
        call_args = session.run.call_args[0][0]
        assert "LIMIT 500" in call_args

    @pytest.mark.asyncio
    async def test_scan_all_gdcs(self, mock_neo4j_driver):
        """Test scanning all GDCs"""
        driver, session = mock_neo4j_driver

        # Mock empty results for all queries
        mock_result = AsyncMock()
        mock_result.data.return_value = []
        session.run.return_value = mock_result

        # Create test registry
        test_gdcs = {
            "GDC_TEST1": GDCSpec(
                id="GDC_TEST1",
                description="Test 1",
                cypher="MATCH (n) RETURN n",
                severity="high",
                suggested_action="action1",
                enabled=True,
            ),
            "GDC_TEST2": GDCSpec(
                id="GDC_TEST2",
                description="Test 2",
                cypher="MATCH (n) RETURN n",
                severity="low",
                suggested_action="action2",
                enabled=True,
            ),
        }

        with patch.dict("mcp_servers.hyperag.gdc.registry.GDC_REGISTRY", test_gdcs):
            extractor = GDCExtractor("bolt://localhost:7687", ("neo4j", "password"))
            extractor.driver = driver

            violations = await extractor.scan_all(limit=100)

            assert violations == []
            # Should have called run() for each GDC
            assert session.run.call_count == 2

    @pytest.mark.asyncio
    async def test_error_handling_in_scan(self, mock_neo4j_driver, sample_gdc_spec):
        """Test error handling during Cypher query execution"""
        driver, session = mock_neo4j_driver

        # Mock query failure
        from neo4j.exceptions import Neo4jError

        session.run.side_effect = Neo4jError("Test error")

        extractor = GDCExtractor("bolt://localhost:7687", ("neo4j", "password"))
        extractor.driver = driver

        # Should return empty list on error, not raise exception
        violations = await extractor._scan_single_gdc(sample_gdc_spec, 100)
        assert violations == []


class TestViolationObject:
    """Test suite for Violation data structure"""

    def test_violation_creation(self):
        """Test basic Violation object creation"""
        violation = Violation(
            gdc_id="GDC_TEST",
            nodes=[{"id": "node1", "type": "TestNode"}],
            severity="high",
            suggested_repair="test_repair",
        )

        assert violation.gdc_id == "GDC_TEST"
        assert len(violation.nodes) == 1
        assert violation.severity == "high"
        assert violation.confidence_score == 1.0
        assert isinstance(violation.detected_at, datetime)

    def test_violation_validation(self):
        """Test Violation object validation"""
        # Test invalid confidence score
        with pytest.raises(ValueError):
            Violation(confidence_score=1.5)

        # Test invalid severity
        with pytest.raises(ValueError):
            Violation(severity="extreme")

    def test_violation_to_dict(self):
        """Test Violation serialization to dictionary"""
        violation = Violation(
            gdc_id="GDC_TEST", nodes=[{"id": "node1"}], severity="medium"
        )

        data = violation.to_dict()

        assert data["gdc_id"] == "GDC_TEST"
        assert data["severity"] == "medium"
        assert "detected_at" in data
        assert isinstance(data["detected_at"], str)  # Should be ISO format

    def test_violation_from_dict(self):
        """Test Violation deserialization from dictionary"""
        data = {
            "gdc_id": "GDC_TEST",
            "nodes": [{"id": "node1"}],
            "severity": "low",
            "detected_at": "2025-07-22T12:00:00",
            "confidence_score": 0.8,
        }

        violation = Violation.from_dict(data)

        assert violation.gdc_id == "GDC_TEST"
        assert violation.severity == "low"
        assert violation.confidence_score == 0.8
        assert isinstance(violation.detected_at, datetime)

    def test_get_affected_ids(self):
        """Test extraction of affected node/edge IDs"""
        violation = Violation(
            nodes=[
                {"id": "node1", "type": "TestNode"},
                {"id": "node2", "type": "TestNode"},
            ],
            edges=[{"id": "edge1", "type": "TEST_REL"}],
        )

        node_ids = violation.get_affected_node_ids()
        edge_ids = violation.get_affected_edge_ids()

        assert node_ids == ["node1", "node2"]
        assert edge_ids == ["edge1"]

    def test_add_context(self):
        """Test adding contextual information to violation"""
        violation = Violation()

        violation.add_context("scan_timestamp", "2025-07-22T12:00:00")
        violation.add_context("graph_size", 1000)

        assert violation.metadata["scan_timestamp"] == "2025-07-22T12:00:00"
        assert violation.metadata["graph_size"] == 1000


class TestGDCSpec:
    """Test suite for GDC specification validation"""

    def test_valid_gdc_spec(self):
        """Test creation of valid GDC specification"""
        spec = GDCSpec(
            id="GDC_VALID_TEST",
            description="Valid test GDC",
            cypher="MATCH (n) RETURN n",
            severity="medium",
            suggested_action="test_action",
        )

        assert spec.id == "GDC_VALID_TEST"
        assert spec.severity == "medium"
        assert spec.enabled is True  # Default value

    def test_invalid_gdc_id(self):
        """Test GDC ID validation"""
        with pytest.raises(ValueError, match="GDC ID must start with 'GDC_'"):
            GDCSpec(
                id="INVALID_ID",
                description="Invalid ID test",
                cypher="MATCH (n) RETURN n",
                severity="low",
                suggested_action="test",
            )

    def test_invalid_severity(self):
        """Test severity validation"""
        with pytest.raises(ValueError, match="Invalid severity"):
            GDCSpec(
                id="GDC_INVALID_SEVERITY",
                description="Invalid severity test",
                cypher="MATCH (n) RETURN n",
                severity="extreme",
                suggested_action="test",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
