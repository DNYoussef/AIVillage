"""Comprehensive tests for SQLite WAL tokenomics receipts."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from .tokenomics_receipts import (
    TokenomicsConfig,
    TokenomicsReceiptManager,
)


@pytest.fixture
async def temp_config():
    """Create a temporary configuration for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = TokenomicsConfig(
            database_path=str(Path(temp_dir) / "test_receipts.db"),
            busy_timeout_ms=5000,
            max_retries=3,
            receipt_retention_days=1,
        )
        yield config


@pytest.fixture
async def receipt_manager(temp_config):
    """Create a tokenomics receipt manager for testing."""
    manager = TokenomicsReceiptManager(temp_config)
    await manager.initialize()
    yield manager
    await manager.shutdown()


@pytest.mark.asyncio
async def test_tensor_transfer_receipt_creation(receipt_manager):
    """Test creating tensor transfer receipts."""
    node_id = "test_node_1"
    peer_id = "test_peer_1"
    tensor_id = "test_tensor_123"
    bytes_transferred = 1024 * 1024  # 1 MB
    metadata = {"compression": "lz4", "model_name": "test_model"}

    receipt_id = await receipt_manager.create_tensor_transfer_receipt(
        node_id=node_id,
        peer_id=peer_id,
        tensor_id=tensor_id,
        bytes_transferred=bytes_transferred,
        metadata=metadata,
    )

    assert receipt_id is not None
    assert len(receipt_id) > 0

    # Verify receipt was stored correctly
    receipt = await receipt_manager.get_receipt(receipt_id)
    assert receipt is not None
    assert receipt.node_id == node_id
    assert receipt.peer_id == peer_id
    assert receipt.tensor_id == tensor_id
    assert receipt.bytes_transferred == bytes_transferred
    assert receipt.transaction_type == "tensor_transfer"
    assert receipt.metadata == metadata
    assert receipt.amount > 0  # Should have calculated cost
    assert receipt.status == "pending"


@pytest.mark.asyncio
async def test_compute_credit_receipt_creation(receipt_manager):
    """Test creating compute credit receipts."""
    node_id = "test_node_2"
    peer_id = "test_peer_2"
    compute_time_ms = 5000.0  # 5 seconds
    metadata = {"model_layers": 12, "inference_type": "generation"}

    receipt_id = await receipt_manager.create_compute_credit_receipt(
        node_id=node_id,
        peer_id=peer_id,
        compute_time_ms=compute_time_ms,
        metadata=metadata,
    )

    assert receipt_id is not None

    # Verify receipt
    receipt = await receipt_manager.get_receipt(receipt_id)
    assert receipt is not None
    assert receipt.node_id == node_id
    assert receipt.peer_id == peer_id
    assert receipt.compute_time_ms == compute_time_ms
    assert receipt.transaction_type == "compute_credit"
    assert receipt.metadata == metadata


@pytest.mark.asyncio
async def test_bandwidth_usage_receipt_creation(receipt_manager):
    """Test creating bandwidth usage receipts."""
    node_id = "test_node_3"
    peer_id = None  # Can be None for local usage
    bytes_transferred = 10 * 1024 * 1024 * 1024  # 10 GB
    metadata = {"protocol": "tcp", "compression": "enabled"}

    receipt_id = await receipt_manager.create_bandwidth_usage_receipt(
        node_id=node_id,
        peer_id=peer_id,
        bytes_transferred=bytes_transferred,
        metadata=metadata,
    )

    assert receipt_id is not None

    # Verify receipt
    receipt = await receipt_manager.get_receipt(receipt_id)
    assert receipt is not None
    assert receipt.node_id == node_id
    assert receipt.peer_id is None
    assert receipt.bytes_transferred == bytes_transferred
    assert receipt.transaction_type == "bandwidth_usage"
    assert receipt.amount > 0  # Should be significant for 10GB


@pytest.mark.asyncio
async def test_receipt_confirmation(receipt_manager):
    """Test receipt confirmation process."""
    node_id = "test_node_confirm"

    # Create a receipt
    receipt_id = await receipt_manager.create_tensor_transfer_receipt(
        node_id=node_id,
        peer_id="peer_confirm",
        tensor_id="tensor_confirm",
        bytes_transferred=1024,
    )

    # Initial receipt should be pending
    receipt = await receipt_manager.get_receipt(receipt_id)
    assert receipt.status == "pending"
    assert receipt.confirmations == 0

    # Add some confirmations
    success = await receipt_manager.confirm_receipt(receipt_id, confirmations=2)
    assert success is False  # Should not be confirmed yet (needs 3)

    receipt = await receipt_manager.get_receipt(receipt_id)
    assert receipt.confirmations == 2
    assert receipt.status == "pending"

    # Add final confirmation to reach threshold
    success = await receipt_manager.confirm_receipt(receipt_id, confirmations=1)
    assert success is True

    receipt = await receipt_manager.get_receipt(receipt_id)
    assert receipt.confirmations == 3
    assert receipt.status == "confirmed"


@pytest.mark.asyncio
async def test_get_node_receipts(receipt_manager):
    """Test retrieving receipts for a specific node."""
    node_id = "test_node_query"

    # Create multiple receipts
    receipt_ids = []
    for i in range(5):
        receipt_id = await receipt_manager.create_tensor_transfer_receipt(
            node_id=node_id,
            peer_id=f"peer_{i}",
            tensor_id=f"tensor_{i}",
            bytes_transferred=1024 * (i + 1),
        )
        receipt_ids.append(receipt_id)

    # Confirm some receipts
    await receipt_manager.confirm_receipt(receipt_ids[0], confirmations=3)
    await receipt_manager.confirm_receipt(receipt_ids[1], confirmations=3)

    # Get all receipts for node
    all_receipts = await receipt_manager.get_node_receipts(node_id)
    assert len(all_receipts) == 5

    # Get only confirmed receipts
    confirmed_receipts = await receipt_manager.get_node_receipts(
        node_id, status="confirmed"
    )
    assert len(confirmed_receipts) == 2

    # Get only pending receipts
    pending_receipts = await receipt_manager.get_node_receipts(
        node_id, status="pending"
    )
    assert len(pending_receipts) == 3

    # Test pagination
    page1 = await receipt_manager.get_node_receipts(node_id, limit=2, offset=0)
    assert len(page1) == 2

    page2 = await receipt_manager.get_node_receipts(node_id, limit=2, offset=2)
    assert len(page2) == 2


@pytest.mark.asyncio
async def test_tokenomics_summary(receipt_manager):
    """Test tokenomics summary generation."""
    node_id = "test_node_summary"

    # Create receipts of different types
    await receipt_manager.create_tensor_transfer_receipt(
        node_id=node_id,
        peer_id="peer1",
        tensor_id="tensor1",
        bytes_transferred=1024 * 1024,
    )

    await receipt_manager.create_compute_credit_receipt(
        node_id=node_id,
        peer_id="peer2",
        compute_time_ms=10000.0,
    )

    await receipt_manager.create_bandwidth_usage_receipt(
        node_id=node_id,
        peer_id="peer3",
        bytes_transferred=5 * 1024 * 1024 * 1024,
    )

    # Get summary for specific node
    summary = await receipt_manager.get_tokenomics_summary(node_id)

    assert "node_id" in summary
    assert summary["node_id"] == node_id
    assert "transaction_summary" in summary
    assert "total_credits" in summary
    assert "total_transactions" in summary
    assert "statistics" in summary

    assert summary["total_transactions"] == 3
    assert summary["total_credits"] > 0

    # Get global summary
    global_summary = await receipt_manager.get_tokenomics_summary()
    assert global_summary["node_id"] is None
    assert global_summary["total_transactions"] >= 3


@pytest.mark.asyncio
async def test_concurrent_receipt_creation(receipt_manager):
    """Test concurrent receipt creation to verify WAL mode handling."""
    node_id = "test_node_concurrent"

    async def create_receipt(i):
        return await receipt_manager.create_tensor_transfer_receipt(
            node_id=node_id,
            peer_id=f"peer_{i}",
            tensor_id=f"tensor_{i}",
            bytes_transferred=1024 * i,
        )

    # Create 10 receipts concurrently
    tasks = [create_receipt(i) for i in range(10)]
    receipt_ids = await asyncio.gather(*tasks)

    assert len(receipt_ids) == 10
    assert all(receipt_id is not None for receipt_id in receipt_ids)
    assert len(set(receipt_ids)) == 10  # All should be unique

    # Verify all were stored
    receipts = await receipt_manager.get_node_receipts(node_id)
    assert len(receipts) == 10


@pytest.mark.asyncio
async def test_receipt_metadata_serialization(receipt_manager):
    """Test proper serialization of complex metadata."""
    node_id = "test_node_metadata"
    complex_metadata = {
        "nested": {"key": "value", "number": 42},
        "list": [1, 2, 3, "string"],
        "boolean": True,
        "null": None,
        "unicode": "测试",
    }

    receipt_id = await receipt_manager.create_tensor_transfer_receipt(
        node_id=node_id,
        peer_id="peer_metadata",
        tensor_id="tensor_metadata",
        bytes_transferred=1024,
        metadata=complex_metadata,
    )

    # Retrieve and verify metadata
    receipt = await receipt_manager.get_receipt(receipt_id)
    assert receipt is not None
    assert receipt.metadata == complex_metadata


@pytest.mark.asyncio
async def test_database_busy_handling(temp_config):
    """Test handling of database busy scenarios."""
    # Create manager with very short busy timeout for testing
    config = temp_config
    config.busy_timeout_ms = 100  # Very short timeout
    config.max_retries = 2

    manager = TokenomicsReceiptManager(config)
    await manager.initialize()

    try:
        # This should still work even with short timeout
        receipt_id = await manager.create_tensor_transfer_receipt(
            node_id="test_busy",
            peer_id="peer_busy",
            tensor_id="tensor_busy",
            bytes_transferred=1024,
        )

        assert receipt_id is not None

        # Verify stats tracking
        assert manager.stats["receipts_created"] == 1

    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_error_handling(receipt_manager):
    """Test error handling in various scenarios."""
    # Test getting non-existent receipt
    non_existent = await receipt_manager.get_receipt("non-existent-id")
    assert non_existent is None

    # Test confirming non-existent receipt
    success = await receipt_manager.confirm_receipt("non-existent-id")
    assert success is False

    # Test getting receipts for non-existent node
    receipts = await receipt_manager.get_node_receipts("non-existent-node")
    assert len(receipts) == 0


if __name__ == "__main__":
    # Run a simple test
    async def main():
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TokenomicsConfig(database_path=str(Path(temp_dir) / "test.db"))
            manager = TokenomicsReceiptManager(config)
            await manager.initialize()

            receipt_id = await manager.create_tensor_transfer_receipt(
                node_id="test_node",
                peer_id="test_peer",
                tensor_id="test_tensor",
                bytes_transferred=1024 * 1024,
                metadata={"test": True},
            )

            print(f"Created receipt: {receipt_id}")

            receipt = await manager.get_receipt(receipt_id)
            print(f"Retrieved receipt: {receipt.node_id}, {receipt.amount} credits")

            await manager.shutdown()
            print("Test completed successfully!")

    asyncio.run(main())
