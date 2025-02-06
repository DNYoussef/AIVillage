"""Enhanced local model agent with improved model management and performance tracking."""

import json
import os
import time
import logging
import torch
import numpy as np
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from datetime import datetime
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)
from langroid import ChatAgent, ChatAgentConfig
from langroid.language_models.openai_gpt import OpenAIGPTConfig

from config.unified_config import UnifiedConfig, ModelConfig
from ..bakedquietiot.deepbaking import DeepSystemBakerTask
from ..bakedquietiot.quiet_star import QuietSTaRTask
from ..model_compression.model_compression import CompressedModel, CompressionConfig
from ..model_compression.hypercompression import FinalCompressionConfig, FinalCompressor
from .openrouter_agent import AgentInteraction

logger = logging.getLogger(__name__)

class ModelCheckpoint:
    """Manages model checkpoints with metadata."""
    
    def __init__(self, 
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 metadata: Dict[str, Any],
                 timestamp: float):
        self.model = model
        self.tokenizer = tokenizer
        self.metadata = metadata
        self.timestamp = timestamp

class LocalAgent:
    """
    Enhanced agent for managing local HuggingFace models with improved
    model management, checkpointing, and performance tracking.
    """
    
    def __init__(self, 
                 model_config: ModelConfig,
                 config: UnifiedConfig,
                 device: str = "cpu"):
        """
        Initialize LocalAgent with unified configuration.
        
        Args:
            model_config: Configuration for the local model
            config: Unified configuration instance
            device: Device to run model on ("cuda" or "cpu")
        """
        self.model_config = model_config
        self.config = config
        self.device = device
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.talk_head = None
        
        # Performance tracking
        self.performance_history: List[Dict[str, float]] = []
        self.current_checkpoint: Optional[ModelCheckpoint] = None
        self.checkpoint_history: List[ModelCheckpoint] = []
        
        # Initialize ChatAgent for bakedquietiot
        chat_config = ChatAgentConfig(
            name="LocalAgent",
            llm=OpenAIGPTConfig(chat_model="gpt-3.5-turbo")
        )
        self.chat_agent = ChatAgent(chat_config)
        
        logger.info(f"Initialized LocalAgent with model: {model_config.name} on device: {device}")
    
    async def initialize(self):
        """Initialize the agent by loading and baking the model."""
        await self._load_and_bake_model()
    
    async def _load_and_bake_model(self):
        """Load and enhance the model with deep baking system."""
        try:
            logger.info(f"Loading model: {self.model_config.name}")
            
            # Check for compressed model
            compressed_path = f"compressed_{self.model_config.name.split('/')[-1].lower()}"
            if not Path(compressed_path).exists():
                logger.info("Compressed model not found. Running compression pipeline...")
                await self._run_compression_pipeline()
            
            # Load compressed model
            logger.info("Loading compressed model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                compressed_path,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.tokenizer = AutoTokenizer.from_pretrained(compressed_path)
            
            # Initialize deep baker with compressed model
            baker = DeepSystemBakerTask(
                agent=self.chat_agent,
                model_name=compressed_path,
                device=self.device
            )
            
            # Run deep baking process
            baking_result = await baker.deep_bake_system(
                max_iterations=20,
                consistency_threshold=0.9
            )
            
            # Initialize QuietSTaR with baked model
            quiet_star = QuietSTaRTask(
                agent=self.chat_agent,
                model_path="deep_baked_model"
            )
            
            # Store model components
            self.model = quiet_star.model
            self.tokenizer = quiet_star.tokenizer
            self.talk_head = quiet_star.talk_head
            
            # Create initial checkpoint
            self._create_checkpoint(
                metadata={
                    "baking_result": baking_result,
                    "initial_load": True
                }
            )
            
            logger.info("Successfully loaded and baked model")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    async def _run_compression_pipeline(self):
        """Run the full compression pipeline."""
        try:
            # Stage 1: Load original model
            model = AutoModelForCausalLM.from_pretrained(
                self.model_config.name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )
            tokenizer = AutoTokenizer.from_pretrained(self.model_config.name)
            
            # Stage 2: VPTQ + BitNet compression
            compression_config = CompressionConfig(
                vector_size=4,
                codebook_size=128,
                group_size=64,
                lambda_warmup=500,
                lambda_schedule='linear',
                batch_size=4,
                learning_rate=1e-5,
                epochs=1,
                device=self.device,
                mixed_precision=True,
                num_workers=4
            )
            
            # Create dummy data for compression
            dummy_data = [(torch.randn(1, 512), torch.zeros(1))]
            train_loader = torch.utils.data.DataLoader(dummy_data, batch_size=1)
            val_loader = torch.utils.data.DataLoader(dummy_data, batch_size=1)
            
            compressed_model = CompressedModel(model, compression_config)
            # Removed compress_and_train as it does not exist
            # If compress_and_train is necessary, it should be implemented accordingly
            
            # Stage 3: HyperCompression + SeedLM
            final_config = FinalCompressionConfig(
                block_size=128,
                theta_max=500000,
                chunk_size=500000,
                lfsr_length=12,
                lfsr_polynomial=0x1100B,
                num_threads=4,
                device=self.device,
                enable_mixed_precision=True
            )
            
            final_compressor = FinalCompressor(final_config)
            compressed_state = final_compressor.compress_model(compressed_model)
            
            # Save compressed model
            output_dir = f"compressed_{self.model_config.name.split('/')[-1].lower()}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Save compressed state and tokenizer
            torch.save(compressed_state, os.path.join(output_dir, "compressed_state.pt"))
            tokenizer.save_pretrained(output_dir)
            
            logger.info(f"Saved compressed model to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error in compression pipeline: {str(e)}")
            raise
    
    def _create_checkpoint(self, metadata: Dict[str, Any]):
        """Create a new model checkpoint."""
        checkpoint = ModelCheckpoint(
            model=self.model,
            tokenizer=self.tokenizer,
            metadata=metadata,
            timestamp=time.time()
        )
        
        self.current_checkpoint = checkpoint
        self.checkpoint_history.append(checkpoint)
        
        # Keep only last 5 checkpoints to manage memory
        if len(self.checkpoint_history) > 5:
            self.checkpoint_history.pop(0)
    
    async def generate_response(self, prompt: str, 
                                system_prompt: Optional[str] = None,
                                max_tokens: int = 1000,
                                temperature: float = 0.7,
                                stream: bool = False) -> Dict[str, Any]:
        """
        Generate a response using the local model with enhanced monitoring.
        
        Args:
            prompt: The input prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream the response
            
        Returns:
            Dictionary containing the response and metadata
        """
        try:
            start_time = time.time()
            
            # Create QuietSTaR task
            quiet_star = QuietSTaRTask(
                agent=self.chat_agent,
                model_path="deep_baked_model"
            )
            
            # Prepare input
            input_text = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            
            # Generate response
            if stream:
                response_chunks = []
                async for chunk in quiet_star.process_query_stream(input_text):
                    response_chunks.append(chunk)
                response = "".join(response_chunks)
            else:
                response = await quiet_star.process_query(input_text)
            
            # Calculate performance metrics
            duration = time.time() - start_time
            input_tokens = len(self.generator.tokenizer.encode(input_text))
            output_tokens = len(self.generator.tokenizer.encode(response))
            
            # Record performance
            performance_record = {
                "timestamp": start_time,
                "duration": duration,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "tokens_per_second": (input_tokens + output_tokens) / duration
            }
            self.performance_history.append(performance_record)
            
            # Keep performance history manageable
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
            
            return {
                "response": response,
                "model": self.model_config.name,
                "metadata": {
                    "device": self.device,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "system_prompt_used": bool(system_prompt),
                    "stream_mode": stream,
                    "performance": performance_record
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    def _update_performance_metrics(self, interaction: AgentInteraction):
        """Update comprehensive performance metrics based on interaction results."""
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
                "frontier_model": self.model_name,
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
