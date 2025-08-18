"""SQLite WAL receipts for tokenomics integration in distributed inference.

This module provides SQLite-based receipts for tracking compute credits,
tensor streaming costs, and bandwidth usage in the distributed inference system.
"""

import asyncio
import json
import logging
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TokenomicsReceipt:
    """Represents a tokenomics transaction receipt."""

    receipt_id: str
    transaction_type: str  # 'tensor_transfer', 'compute_credit', 'bandwidth_usage'
    node_id: str
    peer_id: str | None = None

    # Transaction details
    amount: float = 0.0  # Credits/cost amount
    currency: str = "CREDITS"  # Credits currency type
    tensor_id: str | None = None
    bytes_transferred: int = 0
    compute_time_ms: float = 0.0

    # Timing
    timestamp: float = field(default_factory=time.time)
    block_timestamp: float | None = None

    # Status
    status: str = "pending"  # 'pending', 'confirmed', 'failed', 'disputed'
    confirmations: int = 0

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    signature: str | None = None
    hash: str | None = None


@dataclass
class TokenomicsConfig:
    """Configuration for tokenomics receipts."""

    database_path: str = "tokenomics_receipts.db"
    wal_mode: bool = True
    busy_timeout_ms: int = 30000  # 30 seconds
    max_retries: int = 5
    backup_interval_hours: float = 24.0

    # Pricing configuration
    credit_per_mb_transferred: float = 0.01
    credit_per_compute_hour: float = 1.0
    credit_per_gb_bandwidth: float = 0.1

    # Receipt settings
    receipt_retention_days: int = 90
    auto_confirm_threshold: int = 3  # Confirmations needed
    receipt_compression: bool = True


class TokenomicsReceiptManager:
    """Manages SQLite WAL-based receipts for distributed inference tokenomics."""

    def __init__(self, config: TokenomicsConfig | None = None) -> None:
        self.config = config or TokenomicsConfig()
        self.db_path = Path(self.config.database_path)

        # Connection pool for handling busy database
        self._connection_locks: dict[str, asyncio.Lock] = {}
        self._connection_lock = asyncio.Lock()

        # Background tasks
        self._cleanup_task: asyncio.Task | None = None
        self._backup_task: asyncio.Task | None = None

        # Statistics
        self.stats = {
            "receipts_created": 0,
            "receipts_confirmed": 0,
            "receipts_failed": 0,
            "total_credits_processed": 0.0,
            "database_size_mb": 0.0,
            "last_backup": 0.0,
            "busy_timeouts": 0,
        }

        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("TokenomicsReceiptManager initialized")

    async def initialize(self) -> None:
        """Initialize the database and start background tasks."""
        await self._setup_database()
        await self._start_background_tasks()
        logger.info("TokenomicsReceiptManager ready")

    async def shutdown(self) -> None:
        """Shutdown the receipt manager and cleanup resources."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._backup_task:
            self._backup_task.cancel()
        logger.info("TokenomicsReceiptManager shut down")

    async def _setup_database(self) -> None:
        """Setup SQLite database with WAL mode and proper schema."""
        try:
            async with self._get_connection() as conn:
                # Enable WAL mode for better concurrency
                if self.config.wal_mode:
                    await conn.execute("PRAGMA journal_mode=WAL")

                # Set busy timeout to handle concurrent access
                await conn.execute(f"PRAGMA busy_timeout={self.config.busy_timeout_ms}")

                # Enable foreign keys
                await conn.execute("PRAGMA foreign_keys=ON")

                # Create receipts table
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS receipts (
                        receipt_id TEXT PRIMARY KEY,
                        transaction_type TEXT NOT NULL,
                        node_id TEXT NOT NULL,
                        peer_id TEXT,
                        amount REAL NOT NULL DEFAULT 0.0,
                        currency TEXT NOT NULL DEFAULT 'CREDITS',
                        tensor_id TEXT,
                        bytes_transferred INTEGER DEFAULT 0,
                        compute_time_ms REAL DEFAULT 0.0,
                        timestamp REAL NOT NULL,
                        block_timestamp REAL,
                        status TEXT NOT NULL DEFAULT 'pending',
                        confirmations INTEGER DEFAULT 0,
                        metadata TEXT,  -- JSON
                        signature TEXT,
                        hash TEXT,
                        created_at REAL DEFAULT (strftime('%s', 'now')),
                        updated_at REAL DEFAULT (strftime('%s', 'now'))
                    )
                """
                )

                # Create indexes for efficient queries
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_receipts_node_id ON receipts(node_id)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_receipts_timestamp ON receipts(timestamp)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_receipts_status ON receipts(status)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_receipts_tensor_id ON receipts(tensor_id)")

                # Create summary view for statistics
                await conn.execute(
                    """
                    CREATE VIEW IF NOT EXISTS receipt_summary AS
                    SELECT
                        transaction_type,
                        status,
                        COUNT(*) as count,
                        SUM(amount) as total_amount,
                        AVG(amount) as avg_amount,
                        SUM(bytes_transferred) as total_bytes,
                        AVG(compute_time_ms) as avg_compute_time
                    FROM receipts
                    GROUP BY transaction_type, status
                """
                )

                await conn.commit()
                logger.info("Database schema initialized successfully")

        except Exception as e:
            logger.exception(f"Failed to setup database: {e}")
            raise

    @asynccontextmanager
    async def _get_connection(self):
        """Get a database connection with proper error handling."""
        conn = None
        retry_count = 0

        while retry_count < self.config.max_retries:
            try:
                # Use aiosqlite for async operations (would need to be added to requirements)
                # For now, using sync sqlite3 with asyncio thread executor
                conn = await asyncio.get_event_loop().run_in_executor(None, sqlite3.connect, str(self.db_path))

                # Configure connection
                conn.row_factory = sqlite3.Row
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    conn.execute,
                    f"PRAGMA busy_timeout={self.config.busy_timeout_ms}",
                )

                yield conn
                break

            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) or "database is busy" in str(e):
                    retry_count += 1
                    self.stats["busy_timeouts"] += 1
                    wait_time = min(2**retry_count, 10)  # Exponential backoff, max 10s
                    logger.warning(f"Database busy, retrying in {wait_time}s (attempt {retry_count})")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.exception(f"Database error: {e}")
                    raise
            except Exception as e:
                logger.exception(f"Unexpected database error: {e}")
                raise
            finally:
                if conn:
                    try:
                        await asyncio.get_event_loop().run_in_executor(None, conn.close)
                    except Exception as e:
                        logger.warning(f"Error closing connection: {e}")

        if retry_count >= self.config.max_retries:
            raise sqlite3.OperationalError("Max retries exceeded for database access")

    async def create_tensor_transfer_receipt(
        self,
        node_id: str,
        peer_id: str,
        tensor_id: str,
        bytes_transferred: int,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create a receipt for tensor transfer."""
        receipt_id = str(uuid.uuid4())

        # Calculate cost based on bytes transferred
        amount = bytes_transferred * self.config.credit_per_mb_transferred / (1024 * 1024)

        receipt = TokenomicsReceipt(
            receipt_id=receipt_id,
            transaction_type="tensor_transfer",
            node_id=node_id,
            peer_id=peer_id,
            amount=amount,
            tensor_id=tensor_id,
            bytes_transferred=bytes_transferred,
            metadata=metadata or {},
        )

        await self._store_receipt(receipt)
        self.stats["receipts_created"] += 1
        self.stats["total_credits_processed"] += amount

        logger.info(f"Created tensor transfer receipt {receipt_id}: {bytes_transferred} bytes, {amount:.4f} credits")
        return receipt_id

    async def create_compute_credit_receipt(
        self,
        node_id: str,
        peer_id: str | None,
        compute_time_ms: float,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create a receipt for compute credits."""
        receipt_id = str(uuid.uuid4())

        # Calculate cost based on compute time
        compute_hours = compute_time_ms / (1000 * 3600)
        amount = compute_hours * self.config.credit_per_compute_hour

        receipt = TokenomicsReceipt(
            receipt_id=receipt_id,
            transaction_type="compute_credit",
            node_id=node_id,
            peer_id=peer_id,
            amount=amount,
            compute_time_ms=compute_time_ms,
            metadata=metadata or {},
        )

        await self._store_receipt(receipt)
        self.stats["receipts_created"] += 1
        self.stats["total_credits_processed"] += amount

        logger.info(f"Created compute credit receipt {receipt_id}: {compute_time_ms}ms, {amount:.4f} credits")
        return receipt_id

    async def create_bandwidth_usage_receipt(
        self,
        node_id: str,
        peer_id: str | None,
        bytes_transferred: int,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create a receipt for bandwidth usage."""
        receipt_id = str(uuid.uuid4())

        # Calculate cost based on bandwidth usage
        gb_transferred = bytes_transferred / (1024 * 1024 * 1024)
        amount = gb_transferred * self.config.credit_per_gb_bandwidth

        receipt = TokenomicsReceipt(
            receipt_id=receipt_id,
            transaction_type="bandwidth_usage",
            node_id=node_id,
            peer_id=peer_id,
            amount=amount,
            bytes_transferred=bytes_transferred,
            metadata=metadata or {},
        )

        await self._store_receipt(receipt)
        self.stats["receipts_created"] += 1
        self.stats["total_credits_processed"] += amount

        logger.info(f"Created bandwidth usage receipt {receipt_id}: {bytes_transferred} bytes, {amount:.4f} credits")
        return receipt_id

    async def _store_receipt(self, receipt: TokenomicsReceipt) -> None:
        """Store a receipt in the database."""
        try:
            async with self._get_connection() as conn:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    conn.execute,
                    """
                    INSERT INTO receipts (
                        receipt_id, transaction_type, node_id, peer_id,
                        amount, currency, tensor_id, bytes_transferred,
                        compute_time_ms, timestamp, block_timestamp,
                        status, confirmations, metadata, signature, hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        receipt.receipt_id,
                        receipt.transaction_type,
                        receipt.node_id,
                        receipt.peer_id,
                        receipt.amount,
                        receipt.currency,
                        receipt.tensor_id,
                        receipt.bytes_transferred,
                        receipt.compute_time_ms,
                        receipt.timestamp,
                        receipt.block_timestamp,
                        receipt.status,
                        receipt.confirmations,
                        json.dumps(receipt.metadata) if receipt.metadata else None,
                        receipt.signature,
                        receipt.hash,
                    ),
                )
                await asyncio.get_event_loop().run_in_executor(None, conn.commit)

        except Exception as e:
            logger.exception(f"Failed to store receipt {receipt.receipt_id}: {e}")
            raise

    async def get_receipt(self, receipt_id: str) -> TokenomicsReceipt | None:
        """Get a receipt by ID."""
        try:
            async with self._get_connection() as conn:
                cursor = await asyncio.get_event_loop().run_in_executor(
                    None,
                    conn.execute,
                    "SELECT * FROM receipts WHERE receipt_id = ?",
                    (receipt_id,),
                )
                row = await asyncio.get_event_loop().run_in_executor(None, cursor.fetchone)

                if row:
                    return self._row_to_receipt(row)
                return None

        except Exception as e:
            logger.exception(f"Failed to get receipt {receipt_id}: {e}")
            return None

    async def confirm_receipt(self, receipt_id: str, confirmations: int = 1) -> bool:
        """Confirm a receipt by adding confirmations."""
        try:
            async with self._get_connection() as conn:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    conn.execute,
                    """
                    UPDATE receipts
                    SET confirmations = confirmations + ?,
                        status = CASE
                            WHEN confirmations + ? >= ? THEN 'confirmed'
                            ELSE status
                        END,
                        updated_at = strftime('%s', 'now')
                    WHERE receipt_id = ?
                    """,
                    (
                        confirmations,
                        confirmations,
                        self.config.auto_confirm_threshold,
                        receipt_id,
                    ),
                )
                await asyncio.get_event_loop().run_in_executor(None, conn.commit)

                # Check if receipt was confirmed
                cursor = await asyncio.get_event_loop().run_in_executor(
                    None,
                    conn.execute,
                    "SELECT status FROM receipts WHERE receipt_id = ?",
                    (receipt_id,),
                )
                row = await asyncio.get_event_loop().run_in_executor(None, cursor.fetchone)

                if row and row["status"] == "confirmed":
                    self.stats["receipts_confirmed"] += 1
                    logger.info(f"Receipt {receipt_id} confirmed")
                    return True

                return False

        except Exception as e:
            logger.exception(f"Failed to confirm receipt {receipt_id}: {e}")
            return False

    async def get_node_receipts(
        self, node_id: str, limit: int = 100, offset: int = 0, status: str | None = None
    ) -> list[TokenomicsReceipt]:
        """Get receipts for a specific node."""
        try:
            query = "SELECT * FROM receipts WHERE node_id = ?"
            params = [node_id]

            if status:
                query += " AND status = ?"
                params.append(status)

            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            async with self._get_connection() as conn:
                cursor = await asyncio.get_event_loop().run_in_executor(None, conn.execute, query, params)
                rows = await asyncio.get_event_loop().run_in_executor(None, cursor.fetchall)

                return [self._row_to_receipt(row) for row in rows]

        except Exception as e:
            logger.exception(f"Failed to get receipts for node {node_id}: {e}")
            return []

    async def get_tokenomics_summary(self, node_id: str | None = None) -> dict[str, Any]:
        """Get tokenomics summary statistics."""
        try:
            async with self._get_connection() as conn:
                if node_id:
                    cursor = await asyncio.get_event_loop().run_in_executor(
                        None,
                        conn.execute,
                        """
                        SELECT
                            transaction_type,
                            status,
                            COUNT(*) as count,
                            SUM(amount) as total_amount,
                            AVG(amount) as avg_amount,
                            SUM(bytes_transferred) as total_bytes,
                            AVG(compute_time_ms) as avg_compute_time
                        FROM receipts
                        WHERE node_id = ?
                        GROUP BY transaction_type, status
                        """,
                        (node_id,),
                    )
                else:
                    cursor = await asyncio.get_event_loop().run_in_executor(
                        None,
                        conn.execute,
                        "SELECT * FROM receipt_summary",
                    )

                rows = await asyncio.get_event_loop().run_in_executor(None, cursor.fetchall)

                summary = {
                    "node_id": node_id,
                    "transaction_summary": [],
                    "total_credits": 0.0,
                    "total_transactions": 0,
                    "statistics": self.stats.copy(),
                }

                for row in rows:
                    row_dict = dict(row)
                    summary["transaction_summary"].append(row_dict)
                    summary["total_credits"] += row_dict.get("total_amount", 0) or 0
                    summary["total_transactions"] += row_dict.get("count", 0) or 0

                return summary

        except Exception as e:
            logger.exception(f"Failed to get tokenomics summary: {e}")
            return {"error": str(e)}

    def _row_to_receipt(self, row: sqlite3.Row) -> TokenomicsReceipt:
        """Convert database row to TokenomicsReceipt."""
        metadata = {}
        if row["metadata"]:
            try:
                metadata = json.loads(row["metadata"])
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON metadata for receipt {row['receipt_id']}")

        return TokenomicsReceipt(
            receipt_id=row["receipt_id"],
            transaction_type=row["transaction_type"],
            node_id=row["node_id"],
            peer_id=row["peer_id"],
            amount=row["amount"],
            currency=row["currency"],
            tensor_id=row["tensor_id"],
            bytes_transferred=row["bytes_transferred"],
            compute_time_ms=row["compute_time_ms"],
            timestamp=row["timestamp"],
            block_timestamp=row["block_timestamp"],
            status=row["status"],
            confirmations=row["confirmations"],
            metadata=metadata,
            signature=row["signature"],
            hash=row["hash"],
        )

    async def _start_background_tasks(self) -> None:
        """Start background maintenance tasks."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._backup_task = asyncio.create_task(self._backup_loop())

    async def _cleanup_loop(self) -> None:
        """Background task for cleaning up old receipts."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self._cleanup_old_receipts()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in cleanup loop: {e}")

    async def _backup_loop(self) -> None:
        """Background task for database backups."""
        while True:
            try:
                await asyncio.sleep(self.config.backup_interval_hours * 3600)
                await self._backup_database()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in backup loop: {e}")

    async def _cleanup_old_receipts(self) -> None:
        """Clean up old receipts based on retention policy."""
        try:
            cutoff_time = time.time() - (self.config.receipt_retention_days * 24 * 3600)

            async with self._get_connection() as conn:
                cursor = await asyncio.get_event_loop().run_in_executor(
                    None,
                    conn.execute,
                    "DELETE FROM receipts WHERE timestamp < ? AND status = 'confirmed'",
                    (cutoff_time,),
                )
                deleted_count = cursor.rowcount
                await asyncio.get_event_loop().run_in_executor(None, conn.commit)

                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} old receipts")

        except Exception as e:
            logger.exception(f"Failed to cleanup old receipts: {e}")

    async def _backup_database(self) -> None:
        """Create a backup of the database."""
        try:
            backup_path = self.db_path.with_suffix(f".backup.{int(time.time())}.db")

            async with self._get_connection() as conn:
                # Simple backup by copying database
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: conn.execute(f"VACUUM INTO '{backup_path}'").fetchone(),
                )

            self.stats["last_backup"] = time.time()
            logger.info(f"Database backup created: {backup_path}")

        except Exception as e:
            logger.exception(f"Failed to backup database: {e}")
