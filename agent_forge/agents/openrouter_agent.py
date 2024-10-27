import os
from typing import Dict, Any, List, Optional
import logging
import asyncio
import json
import time
from openai import AsyncOpenAI
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class AgentInteraction:
    """Record of an agent's interaction using the frontier model."""
    prompt: str
    response: str
    model: str
    timestamp: float
    metadata: Dict[str, Any]
    performance_metrics: Optional[Dict[str, float]] = None

class OpenRouterAgent:
    """
    Model-agnostic API layer for connecting agents to frontier models via OpenRouter.
    Tracks interactions for DPO analysis and local model training.
    """
    
    def __init__(self, api_key: str, model: str, local_model: str):
        """
        Initialize OpenRouterAgent.
        
        Args:
            api_key: OpenRouter API key
            model: Frontier model identifier (e.g. "nvidia/llama-3.1-nemotron-70b-instruct")
            local_model: Local model to train (e.g. "Qwen/Qwen2.5-3B-Instruct")
        """
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/yourusername/AIVillage",
                "X-Title": "AI Village"
            }
        )
        self.model = model
        self.local_model = local_model
        
        # Track interactions for DPO and training
        self.interactions: List[AgentInteraction] = []
        
        # Rate limiting
        self.requests_per_minute = 50
        self.request_timestamps: List[float] = []
        self.retry_delay = 1
        self.max_retries = 3
    
    async def _handle_rate_limit(self):
        """Handle rate limiting."""
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

    async def generate_response(self, 
                              prompt: str, 
                              system_prompt: Optional[str] = None,
                              temperature: float = 0.7,
                              max_tokens: int = 1000) -> AgentInteraction:
        """
        Generate a response using the frontier model and track the interaction.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt for context
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            AgentInteraction containing the response and metadata
        """
        for attempt in range(self.max_retries):
            try:
                await self._handle_rate_limit()
                
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                interaction = AgentInteraction(
                    prompt=prompt,
                    response=response.choices[0].message.content,
                    model=self.model,
                    timestamp=time.time(),
                    metadata={
                        "system_prompt": system_prompt,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "usage": {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens
                        }
                    }
                )
                
                # Store interaction for DPO and training
                self.interactions.append(interaction)
                
                return interaction
                
            except Exception as e:
                logger.error(f"Error generating response (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise

    def record_performance(self, interaction: AgentInteraction, metrics: Dict[str, float]):
        """
        Record performance metrics for an interaction for DPO analysis.
        
        Args:
            interaction: The interaction to update
            metrics: Performance metrics (e.g. accuracy, relevance scores)
        """
        interaction.performance_metrics = metrics
    
    def get_training_data(self) -> List[Dict[str, Any]]:
        """
        Get training data for the local model from tracked interactions.
        
        Returns:
            List of training examples with prompts, responses, and performance metrics
        """
        return [
            {
                "prompt": interaction.prompt,
                "response": interaction.response,
                "frontier_model": interaction.model,
                "local_model": self.local_model,
                "performance": interaction.performance_metrics,
                "metadata": interaction.metadata
            }
            for interaction in self.interactions
            if interaction.performance_metrics  # Only use interactions with performance data
        ]
    
    def get_dpo_metrics(self) -> Dict[str, Any]:
        """
        Get DPO analysis metrics from tracked interactions.
        
        Returns:
            Dictionary of DPO metrics and statistics
        """
        if not self.interactions:
            return {"error": "No interactions recorded"}
            
        metrics = {}
        
        # Calculate average performance metrics
        performance_metrics = [
            i.performance_metrics for i in self.interactions 
            if i.performance_metrics
        ]
        
        if performance_metrics:
            # Combine all unique metric keys
            metric_keys = set()
            for m in performance_metrics:
                metric_keys.update(m.keys())
                
            # Calculate averages for each metric
            for key in metric_keys:
                values = [m[key] for m in performance_metrics if key in m]
                metrics[f"avg_{key}"] = sum(values) / len(values)
                
            # Add total interactions and other stats
            metrics.update({
                "total_interactions": len(self.interactions),
                "interactions_with_metrics": len(performance_metrics),
                "total_tokens": sum(
                    i.metadata["usage"]["total_tokens"] 
                    for i in self.interactions
                )
            })
            
        return metrics
