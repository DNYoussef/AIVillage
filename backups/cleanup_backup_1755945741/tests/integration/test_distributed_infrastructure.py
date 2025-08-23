#!/usr/bin/env python3
"""
Integration tests for distributed infrastructure.
Run with: python -m pytest tests/ -v
"""

import asyncio

import pytest

from scripts.create_integration_tests import run_integration_tests


@pytest.mark.asyncio
async def test_distributed_infrastructure():
    """Test complete distributed infrastructure."""
    result = await run_integration_tests()
    assert result, "Integration tests failed"


if __name__ == "__main__":
    asyncio.run(run_integration_tests())
