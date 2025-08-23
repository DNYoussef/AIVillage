"""
Tests for MemoryInterface

Behavioral tests ensuring interface compliance and contract validation.
"""

from abc import ABC
from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from core.rag.interfaces.memory_interface import (
    AccessPattern,
    MemoryContext,
    MemoryEntry,
    MemoryInterface,
    MemoryMetadata,
    MemorySearchResult,
    MemoryType,
    RetentionPolicy,
)


class MockMemoryInterface(MemoryInterface):
    """Mock implementation for testing interface compliance"""

    async def store(self, content, memory_type, context, ttl=None, importance=0.5):
        return self._store_mock

    async def retrieve(self, memory_id, update_access=True):
        return self._retrieve_mock

    async def search(self, query, memory_types=None, context=None, max_results=10, similarity_threshold=0.7):
        return self._search_mock

    async def associate(self, memory_id1, memory_id2, association_strength=1.0, association_type="related"):
        return self._associate_mock

    async def get_associations(self, memory_id, association_types=None, max_associations=10):
        return self._get_associations_mock

    async def update(self, memory_id, content=None, importance=None, ttl=None, tags=None):
        return self._update_mock

    async def delete(self, memory_id, cascade_associations=False):
        return self._delete_mock

    async def cleanup_expired(self, memory_types=None, force_cleanup=False):
        return self._cleanup_expired_mock

    async def get_memory_usage(self, context=None):
        return self._get_memory_usage_mock

    async def optimize_storage(self, memory_types=None, optimization_strategy="balanced"):
        return self._optimize_storage_mock

    def __init__(self):
        self._store_mock = None
        self._retrieve_mock = None
        self._search_mock = None
        self._associate_mock = None
        self._get_associations_mock = None
        self._update_mock = None
        self._delete_mock = None
        self._cleanup_expired_mock = None
        self._get_memory_usage_mock = None
        self._optimize_storage_mock = None


@pytest.fixture
def memory_interface():
    """Fixture providing mock memory interface"""
    return MockMemoryInterface()


@pytest.fixture
def sample_context():
    """Sample memory context for testing"""
    return MemoryContext(
        user_id="test_user",
        session_id="test_session",
        domain="testing",
        retention_policy=RetentionPolicy.TIME_BASED,
        access_pattern=AccessPattern.SEQUENTIAL,
        encryption_required=True,
    )


@pytest.fixture
def sample_metadata():
    """Sample memory metadata for testing"""
    return MemoryMetadata(
        created_at=datetime.now(),
        last_accessed=datetime.now(),
        access_count=5,
        importance_score=0.8,
        tags=["test", "important"],
        associations=["related_memory_1"],
        encryption_key_id="key_123",
    )


@pytest.fixture
def sample_memory_entry(sample_metadata):
    """Sample memory entry for testing"""
    return MemoryEntry(
        memory_id="mem_123",
        content={"text": "Test memory content", "value": 42},
        memory_type=MemoryType.SHORT_TERM,
        metadata=sample_metadata,
        expires_at=datetime.now() + timedelta(hours=1),
        is_encrypted=True,
    )


class TestMemoryInterface:
    """Test memory interface contract and behavior"""

    def test_is_abstract_base_class(self):
        """Test that MemoryInterface is properly abstract"""
        assert issubclass(MemoryInterface, ABC)

        # Should not be able to instantiate directly
        with pytest.raises(TypeError):
            MemoryInterface()

    @pytest.mark.asyncio
    async def test_store_contract(self, memory_interface, sample_context):
        """Test store method contract"""
        expected_id = "mem_12345"
        memory_interface.store.return_value = expected_id

        result = await memory_interface.store(
            content="Test content",
            memory_type=MemoryType.SHORT_TERM,
            context=sample_context,
            ttl=timedelta(hours=1),
            importance=0.8,
        )

        memory_interface.store.assert_called_once()
        assert result == expected_id
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_retrieve_contract(self, memory_interface, sample_memory_entry):
        """Test retrieve method contract"""
        memory_interface.retrieve.return_value = sample_memory_entry

        result = await memory_interface.retrieve(memory_id="mem_123", update_access=True)

        memory_interface.retrieve.assert_called_once_with("mem_123", True)
        assert result == sample_memory_entry
        assert isinstance(result, MemoryEntry) or result is None

    @pytest.mark.asyncio
    async def test_search_contract(self, memory_interface, sample_memory_entry, sample_context):
        """Test search method contract"""
        expected_result = MemorySearchResult(
            entries=[sample_memory_entry],
            total_matches=1,
            search_latency_ms=50.0,
            associations_found=["related_1"],
            metadata={"search_type": "semantic"},
        )
        memory_interface.search.return_value = expected_result

        result = await memory_interface.search(
            query="test query",
            memory_types=[MemoryType.SHORT_TERM],
            context=sample_context,
            max_results=10,
            similarity_threshold=0.7,
        )

        memory_interface.search.assert_called_once()
        assert result == expected_result
        assert isinstance(result, MemorySearchResult)

    @pytest.mark.asyncio
    async def test_associate_contract(self, memory_interface):
        """Test associate method contract"""
        memory_interface.associate.return_value = True

        result = await memory_interface.associate(
            memory_id1="mem_1", memory_id2="mem_2", association_strength=0.8, association_type="semantic"
        )

        memory_interface.associate.assert_called_once()
        assert result is True
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_get_associations_contract(self, memory_interface, sample_memory_entry):
        """Test get_associations method contract"""
        expected_associations = [sample_memory_entry]
        memory_interface.get_associations.return_value = expected_associations

        result = await memory_interface.get_associations(
            memory_id="mem_123", association_types=["semantic", "temporal"], max_associations=5
        )

        memory_interface.get_associations.assert_called_once()
        assert result == expected_associations
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_update_contract(self, memory_interface):
        """Test update method contract"""
        memory_interface.update.return_value = True

        result = await memory_interface.update(
            memory_id="mem_123",
            content="Updated content",
            importance=0.9,
            ttl=timedelta(hours=2),
            tags=["updated", "important"],
        )

        memory_interface.update.assert_called_once()
        assert result is True
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_delete_contract(self, memory_interface):
        """Test delete method contract"""
        memory_interface.delete.return_value = True

        result = await memory_interface.delete(memory_id="mem_123", cascade_associations=True)

        memory_interface.delete.assert_called_once_with("mem_123", True)
        assert result is True
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_cleanup_expired_contract(self, memory_interface):
        """Test cleanup_expired method contract"""
        expected_stats = {"short_term_cleaned": 10, "working_cleaned": 5, "total_cleaned": 15}
        memory_interface.cleanup_expired.return_value = expected_stats

        result = await memory_interface.cleanup_expired(
            memory_types=[MemoryType.SHORT_TERM, MemoryType.WORKING], force_cleanup=True
        )

        memory_interface.cleanup_expired.assert_called_once()
        assert result == expected_stats
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_get_memory_usage_contract(self, memory_interface, sample_context):
        """Test get_memory_usage method contract"""
        expected_usage = {
            "total_entries": 1000,
            "memory_usage_mb": 50.5,
            "average_access_time_ms": 2.3,
            "system_health": "healthy",
        }
        memory_interface.get_memory_usage.return_value = expected_usage

        result = await memory_interface.get_memory_usage(context=sample_context)

        memory_interface.get_memory_usage.assert_called_once_with(sample_context)
        assert result == expected_usage
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_optimize_storage_contract(self, memory_interface):
        """Test optimize_storage method contract"""
        expected_optimization = {
            "optimization_applied": "balanced",
            "space_saved_mb": 25.0,
            "performance_improvement": 0.15,
            "entries_reorganized": 500,
        }
        memory_interface.optimize_storage.return_value = expected_optimization

        result = await memory_interface.optimize_storage(
            memory_types=[MemoryType.LONG_TERM], optimization_strategy="balanced"
        )

        memory_interface.optimize_storage.assert_called_once()
        assert result == expected_optimization
        assert isinstance(result, dict)


class TestMemoryDataClasses:
    """Test memory data classes and enums"""

    def test_memory_type_enum(self):
        """Test MemoryType enum values"""
        assert MemoryType.SHORT_TERM.value == "short_term"
        assert MemoryType.WORKING.value == "working"
        assert MemoryType.LONG_TERM.value == "long_term"
        assert MemoryType.EPISODIC.value == "episodic"
        assert MemoryType.SEMANTIC.value == "semantic"

    def test_access_pattern_enum(self):
        """Test AccessPattern enum values"""
        assert AccessPattern.SEQUENTIAL.value == "sequential"
        assert AccessPattern.RANDOM.value == "random"
        assert AccessPattern.TEMPORAL.value == "temporal"
        assert AccessPattern.ASSOCIATIVE.value == "associative"
        assert AccessPattern.HIERARCHICAL.value == "hierarchical"

    def test_retention_policy_enum(self):
        """Test RetentionPolicy enum values"""
        assert RetentionPolicy.TIME_BASED.value == "time_based"
        assert RetentionPolicy.ACCESS_BASED.value == "access_based"
        assert RetentionPolicy.CAPACITY_BASED.value == "capacity_based"
        assert RetentionPolicy.IMPORTANCE_BASED.value == "importance_based"
        assert RetentionPolicy.MANUAL.value == "manual"

    def test_memory_context_creation(self):
        """Test MemoryContext dataclass creation"""
        context = MemoryContext(user_id="test", retention_policy=RetentionPolicy.ACCESS_BASED, encryption_required=True)

        assert context.user_id == "test"
        assert context.retention_policy == RetentionPolicy.ACCESS_BASED
        assert context.encryption_required is True
        assert context.session_id is None  # Default value
        assert context.access_pattern == AccessPattern.RANDOM  # Default value

    def test_memory_metadata_creation(self, sample_metadata):
        """Test MemoryMetadata dataclass creation"""
        metadata = sample_metadata

        assert isinstance(metadata.created_at, datetime)
        assert isinstance(metadata.last_accessed, datetime)
        assert metadata.access_count == 5
        assert metadata.importance_score == 0.8
        assert metadata.tags == ["test", "important"]
        assert metadata.associations == ["related_memory_1"]
        assert metadata.encryption_key_id == "key_123"

    def test_memory_entry_creation(self, sample_memory_entry, sample_metadata):
        """Test MemoryEntry dataclass creation"""
        entry = sample_memory_entry

        assert entry.memory_id == "mem_123"
        assert entry.content == {"text": "Test memory content", "value": 42}
        assert entry.memory_type == MemoryType.SHORT_TERM
        assert entry.metadata == sample_metadata
        assert isinstance(entry.expires_at, datetime)
        assert entry.is_encrypted is True

    def test_memory_search_result_creation(self, sample_memory_entry):
        """Test MemorySearchResult dataclass creation"""
        result = MemorySearchResult(
            entries=[sample_memory_entry],
            total_matches=1,
            search_latency_ms=25.5,
            associations_found=["assoc_1"],
            metadata={"query_type": "semantic"},
        )

        assert result.entries == [sample_memory_entry]
        assert result.total_matches == 1
        assert result.search_latency_ms == 25.5
        assert result.associations_found == ["assoc_1"]
        assert result.metadata == {"query_type": "semantic"}


@pytest.mark.parametrize(
    "memory_type",
    [MemoryType.SHORT_TERM, MemoryType.WORKING, MemoryType.LONG_TERM, MemoryType.EPISODIC, MemoryType.SEMANTIC],
)
def test_memory_types_parametrized(memory_type):
    """Parametrized test for all memory types"""
    assert isinstance(memory_type, MemoryType)
    assert isinstance(memory_type.value, str)


@pytest.mark.parametrize(
    "pattern",
    [
        AccessPattern.SEQUENTIAL,
        AccessPattern.RANDOM,
        AccessPattern.TEMPORAL,
        AccessPattern.ASSOCIATIVE,
        AccessPattern.HIERARCHICAL,
    ],
)
def test_access_patterns_parametrized(pattern):
    """Parametrized test for all access patterns"""
    assert isinstance(pattern, AccessPattern)
    assert isinstance(pattern.value, str)


@pytest.mark.parametrize(
    "policy",
    [
        RetentionPolicy.TIME_BASED,
        RetentionPolicy.ACCESS_BASED,
        RetentionPolicy.CAPACITY_BASED,
        RetentionPolicy.IMPORTANCE_BASED,
        RetentionPolicy.MANUAL,
    ],
)
def test_retention_policies_parametrized(policy):
    """Parametrized test for all retention policies"""
    assert isinstance(policy, RetentionPolicy)
    assert isinstance(policy.value, str)
