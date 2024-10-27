import os
from typing import Dict, Any, List
import logging
from openai import AsyncOpenAI
from ..utils.task import Task

logger = logging.getLogger(__name__)

class OpenRouterAgent:
    """Agent for interacting with OpenRouter API."""
    
    def __init__(self, api_key: str, model: str, base_url: str = "https://openrouter.ai/api/v1"):
        """
        Initialize OpenRouterAgent.
        
        Args:
            api_key: OpenRouter API key
            model: Model identifier (e.g. "nvidia/llama-3.1-nemotron-70b-instruct")
            base_url: OpenRouter API base URL
        """
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/yourusername/AIVillage",  # Update with your repo
                "X-Title": "AI Village"
            }
        )
        self.model = model
        self._setup_rate_limiting()
    
    def _setup_rate_limiting(self):
        """Setup rate limiting parameters."""
        self.requests_per_minute = 50  # Adjust based on your tier
        self.request_timestamps: List[float] = []
        self.retry_delay = 1  # seconds
        self.max_retries = 3
    
    async def _handle_rate_limit(self):
        """Handle rate limiting."""
        import time
        current_time = time.time()
        minute_ago = current_time - 60
        
        # Remove timestamps older than 1 minute
        self.request_timestamps = [ts for ts in self.request_timestamps if ts > minute_ago]
        
        if len(self.request_timestamps) >= self.requests_per_minute:
            sleep_time = 60 - (current_time - self.request_timestamps[0])
            if sleep_time > 0:
                logger.info(f"Rate limit reached, waiting {sleep_time:.2f} seconds")
                await asyncio.sleep(sleep_time)
        
        self.request_timestamps.append(current_time)

    async def process_task(self, task: Task) -> Dict[str, Any]:
        """
        Process a task using the OpenRouter API.
        
        Args:
            task: Task object containing the prompt and other metadata
            
        Returns:
            Dictionary containing the response and metadata
        """
        for attempt in range(self.max_retries):
            try:
                await self._handle_rate_limit()
                
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": task.system_prompt} if task.system_prompt else None,
                        {"role": "user", "content": task.content}
                    ],
                    temperature=task.temperature if task.temperature is not None else 0.7,
                    max_tokens=task.max_tokens if task.max_tokens is not None else 1000,
                    top_p=task.top_p if task.top_p is not None else 1.0,
                    presence_penalty=task.presence_penalty if task.presence_penalty is not None else 0.0,
                    frequency_penalty=task.frequency_penalty if task.frequency_penalty is not None else 0.0,
                )
                
                result = {
                    "content": response.choices[0].message.content,
                    "model": response.model,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    },
                    "finish_reason": response.choices[0].finish_reason
                }
                
                return result
                
            except Exception as e:
                logger.error(f"Error processing task (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise
    
    async def is_complex_task(self, task: Task) -> bool:
        """
        Determine if a task is complex enough to require the API model.
        
        Args:
            task: Task to evaluate
            
        Returns:
            Boolean indicating if task is complex
        """
        # Implement complexity evaluation logic
        # For now, assume all tasks are complex
        return True
