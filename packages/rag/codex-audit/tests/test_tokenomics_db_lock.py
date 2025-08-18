import asyncio
import sqlite3

import pytest

from src.production.distributed_inference.tokenomics_receipts import TokenomicsConfig, TokenomicsReceiptManager


def test_db_lock_retry(tmp_path):
    async def run():
        cfg = TokenomicsConfig(database_path=str(tmp_path / "test.db"), max_retries=1, busy_timeout_ms=10)
        mgr = TokenomicsReceiptManager(cfg)

        # Hold an exclusive lock using a separate connection
        lock_conn = sqlite3.connect(cfg.database_path)
        lock_conn.execute("BEGIN EXCLUSIVE")
        try:
            with pytest.raises(sqlite3.OperationalError):
                async with mgr._get_connection() as conn:
                    await asyncio.get_event_loop().run_in_executor(None, conn.execute, "CREATE TABLE test(id INTEGER)")
        finally:
            lock_conn.rollback()
            lock_conn.close()

    asyncio.run(run())
