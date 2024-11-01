"""Enhanced OpenRouter API integration with improved error handling and rate limiting."""

import os
import logging
import asyncio
import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import numpy as np
from openai import AsyncOpenAI
from config.unified_config import UnifiedConfig, ModelConfig

logger = logging.getLogger(__name__)

@dataclass
class AgentInteraction:
    """Enhanced record of an agent's interaction using the frontier model."""
    prompt: str
    response: str
    model: str
    timestamp: float
    metadata: Dict[str, Any]
    performance_metrics: Optional[Dict[str, float]] = None
    error_count: int = 0
    retry_count: int = 0
    duration: float = 0.0
    token_usage: Dict[str, int] = None
    success: bool = True

class OpenRouterAgent:
    """
    Enhanced model-agnostic API layer for connecting agents to frontier models.
    Includes improved error handling, rate limiting, and performance tracking.
    """
    
    def __init__(self, api_key: str, model: str, local_model: str, config: Optional[UnifiedConfig] = None):
        """
        Initialize OpenRouterAgent with enhanced configuration.
        
        Args:
            api_key: OpenRouter API key
            model: Frontier model identifier
            local_model: Local model identifier
            config: Optional UnifiedConfig instance
        """
        self.api_key = api_key
        self.model = model
        self.local_model = local_model
        self.config = config
        
        # Load configuration
        if config:
            api_config = config.config['api']
            self.requests_per_minute = api_config['requests_per_minute']
            self.retry_delay = api_config['retry_delay']
            self.max_retries = api_config['max_retries']
            self.timeout = api_config['timeout']
        else:
            self.requests_per_minute = 50
            self.retry_delay = 1
            self.max_retries = 3
            self.timeout = 30
        
        # Initialize rate limiting
        self.request_timestamps: List[float] = []
        self.last_request_time: float = 0
        
        # Track interactions for DPO and training
        self.interactions: List[AgentInteraction] = []
        
        # Performance tracking
        self.performance_metrics: Dict[str, float] = {
            "success_rate": 1.0,
            "average_latency": 0.0,
            "token_efficiency": 1.0,
            "error_rate": 0.0
        }
        
        logger.info(f"Initialized OpenRouterAgent with model: {model}")
    
    async def initialize(self):
        """Initialize the OpenRouter client and verify API access."""
        try:
            # Initialize client with default headers exactly as shown in OpenRouter docs
            self.client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key,
                default_headers={
                    "HTTP-Referer": "https://github.com/yourusername/AIVillage",
                    "X-Title": "AI Village"
                }
            )
            
            # Test API access with a simple request
            test_messages = [{"role": "user", "content": "test"}]
            await self.client.chat.completions.create(
                model=self.model,
                messages=test_messages,
                max_tokens=1
            )
            
            logger.info("Successfully initialized OpenRouter client and verified API access")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter client: {str(e)}")
            raise
    
    async def _handle_rate_limit(self):
        """Enhanced rate limit handling with adaptive delays."""
        current_time = time.time()
        minute_ago = current_time - 60
        
        # Remove timestamps older than 1 minute
        self.request_timestamps = [ts for ts in self.request_timestamps if ts > minute_ago]
        
        # Calculate current request rate
        current_rate = len(self.request_timestamps)
        
        if current_rate >= self.requests_per_minute:
            # Calculate adaptive delay based on current rate
            rate_excess = (current_rate - self.requests_per_minute + 1) / self.requests_per_minute
            adaptive_delay = self.retry_delay * (1 + rate_excess)
            
            logger.info(f"Rate limit approached. Waiting {adaptive_delay:.2f} seconds")
            await asyncio.sleep(adaptive_delay)
        
        # Ensure minimum delay between requests
        time_since_last = current_time - self.last_request_time
        if time_since_last < 0.1:  # Minimum 100ms between requests
            await asyncio.sleep(0.1 - time_since_last)
        
        self.request_timestamps.append(current_time)
        self.last_request_time = current_time
    
    async def generate_response(self, 
                              prompt: str, 
                              system_prompt: Optional[str] = None,
                              temperature: float = 0.7,
                              max_tokens: int = 1000,
                              stream: bool = False) -> AgentInteraction:
        """
        Generate a response with enhanced error handling and monitoring.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            
        Returns:
            AgentInteraction containing the response and metadata
        """
        start_time = time.time()
        error_count = 0
        retry_count = 0
        
        for attempt in range(self.max_retries):
            try:
                await self._handle_rate_limit()
                
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=stream
                    ),
                    timeout=self.timeout
                )
                
                # Handle streaming response
                if stream:
                    collected_messages = []
                    async for chunk in response:
                        if chunk.choices[0].delta.content:
                            collected_messages.append(chunk.choices[0].delta.content)
                    response_text = "".join(collected_messages)
                else:
                    response_text = response.choices[0].message.content
                
                # Calculate performance metrics
                duration = time.time() - start_time
                token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
                
                # Create interaction record
                interaction = AgentInteraction(
                    prompt=prompt,
                    response=response_text,
                    model=self.model,
                    timestamp=start_time,
                    metadata={
                        "system_prompt": system_prompt,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "usage": token_usage,
                        "response_time": duration,
                        "stream_mode": stream
                    },
                    error_count=error_count,
                    retry_count=retry_count,
                    duration=duration,
                    token_usage=token_usage,
                    success=True
                )
                
                # Update performance metrics
                self._update_performance_metrics(interaction)
                
                # Store interaction
                self.interactions.append(interaction)
                
                return interaction
                
            except asyncio.TimeoutError:
                error_count += 1
                logger.warning(f"Request timeout (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    retry_count += 1
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise
                
            except Exception as e:
                error_count += 1
                logger.error(f"Error generating response (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    retry_count += 1
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    # Create failed interaction record
                    interaction = AgentInteraction(
                        prompt=prompt,
                        response="",
                        model=self.model,
                        timestamp=start_time,
                        metadata={
                            "error": str(e),
                            "system_prompt": system_prompt,
                            "temperature": temperature,
                            "max_tokens": max_tokens
                        },
                        error_count=error_count,
                        retry_count=retry_count,
                        duration=time.time() - start_time,
                        token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                        success=False
                    )
                    self.interactions.append(interaction)
                    self._update_performance_metrics(interaction)
                    raise
    
    def _update_performance_metrics(self, interaction: AgentInteraction):
        """Update performance metrics based on interaction results."""
        # Calculate success rate
        total_interactions = len(self.interactions)
        successful_interactions = sum(1 for i in self.interactions if i.success)
        self.performance_metrics["success_rate"] = successful_interactions / total_interactions
        
        # Calculate average latency
        total_duration = sum(i.duration for i in self.interactions)
        self.performance_metrics["average_latency"] = total_duration / total_interactions
        
        # Calculate token efficiency
        if interaction.success and interaction.token_usage:
            prompt_tokens = interaction.token_usage["prompt_tokens"]
            completion_tokens = interaction.token_usage["completion_tokens"]
            if prompt_tokens > 0:
                efficiency = completion_tokens / prompt_tokens
                # Update rolling average
                current_efficiency = self.performance_metrics["token_efficiency"]
                self.performance_metrics["token_efficiency"] = (current_efficiency * 0.9 + efficiency * 0.1)
        
        # Calculate error rate
        total_errors = sum(i.error_count for i in self.interactions)
        self.performance_metrics["error_rate"] = total_errors / total_interactions
    
    def record_performance(self, interaction: AgentInteraction, metrics: Dict[str, float]):
        """
        Record performance metrics for an interaction.
        
        Args:
            interaction: The interaction to update
            metrics: Performance metrics
        """
        interaction.performance_metrics = metrics
        self._update_performance_metrics(interaction)
    
    def get_training_data(self) -> List[Dict[str, Any]]:
        """
        Get training data for the local model.
        
        Returns:
            List of training examples
        """
        return [
            {
                "prompt": interaction.prompt,
                "response": interaction.response,
                "frontier_model": self.model,
                "local_model": self.local_model,
                "performance": interaction.performance_metrics,
                "metadata": interaction.metadata
            }
            for interaction in self.interactions
            if interaction.success and interaction.performance_metrics
        ]
    
    def get_dpo_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive DPO analysis metrics.
        
        Returns:
            Dictionary of DPO metrics and statistics
        """
        if not self.interactions:
            return {"error": "No interactions recorded"}
        
        metrics = {
            "performance": self.performance_metrics.copy(),
            "interaction_stats": {
                "total_interactions": len(self.interactions),
                "successful_interactions": sum(1 for i in self.interactions if i.success),
                "total_tokens": sum(
                    i.token_usage["total_tokens"] 
                    for i in self.interactions 
                    if i.success and i.token_usage
                )
            }
        }
        
        # Calculate quality metrics if available
        quality_scores = [
            i.performance_metrics.get("quality", 0)
            for i in self.interactions
            if i.success and i.performance_metrics and "quality" in i.performance_metrics
        ]
        
        if quality_scores:
            metrics["quality_metrics"] = {
                "average_quality": sum(quality_scores) / len(quality_scores),
                "quality_variance": np.var(quality_scores) if len(quality_scores) > 1 else 0,
                "samples_with_quality": len(quality_scores)
            }
        
        return metrics
