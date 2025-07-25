#!/usr/bin/env python3
"""
Simple test to verify OpenRouter connectivity.
"""

import asyncio
import logging
import os
from agent_forge.orchestration.openrouter_client import OpenRouterClient
from agent_forge.orchestration.model_config import TaskType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_simple_connection():
    """Test basic OpenRouter connection."""
    
    # Check if API key is available
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY not found in environment")
        return False
    
    logger.info(f"API key found: {'*' * 20}{api_key[-10:]}")
    
    try:
        # Create client
        client = OpenRouterClient()
        
        # Simple test request
        response = await client.complete(
            task_type=TaskType.EVALUATION_GRADING,
            messages=[{"role": "user", "content": "What is 2+2? Answer with just the number."}],
            max_tokens=10,
            temperature=0.1
        )
        
        logger.info("✅ OpenRouter connection successful!")
        logger.info(f"Response: {response.content}")
        logger.info(f"Model: {response.model_used}")
        logger.info(f"Cost: ${response.cost:.4f}")
        logger.info(f"Latency: {response.latency:.2f}s")
        
        # Clean up
        await client.close()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ OpenRouter connection failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_simple_connection())
    exit(0 if success else 1)