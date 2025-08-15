#!/usr/bin/env python3
"""Unit tests for RedisFallbackStorage memory backend."""

# isort: skip_file

import asyncio
import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from core.database.redis_manager import RedisFallbackStorage


@pytest.mark.asyncio
async def test_memory_expiration():
    """Values should expire and be purged from memory storage."""
    store = RedisFallbackStorage("memory")
    await store.set("temp", "value", ex=1)
    assert await store.get("temp") == "value"
    await asyncio.sleep(1.1)
    assert await store.get("temp") is None
    assert "temp" not in store._memory_store


@pytest.mark.asyncio
async def test_memory_concurrent_access():
    """Concurrent set/get operations should work without error."""
    store = RedisFallbackStorage("memory")

    async def worker(idx: int):
        key = f"key{idx}"
        val = f"value{idx}"
        await store.set(key, val, ex=2)
        return await store.get(key)

    results = await asyncio.gather(*(worker(i) for i in range(10)))
    assert results == [f"value{i}" for i in range(10)]
