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
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
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
        
        # Load and bake model
        self._load_and_bake_model()
        
        logger.info(f"Initialized LocalAgent with model: {model_config.name} on device: {device}")
    
    async def _load_and_bake_model(self):
        """Load and enhance the model with deep baking system."""
        try:
            logger.info(f"Loading model: {self.model_config.name}")
            
            # Initialize deep baker
            baker = DeepSystemBakerTask(
                agent=self.chat_agent,
                model_name=self.model_config.name,
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
    
    async def generate_response(self, 
                              prompt: str, 
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
            input_tokens = len(self.tokenizer.encode(input_text))
            output_tokens = len(self.tokenizer.encode(response))
            
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
    
    def record_performance(self, metrics: Dict[str, float]):
        """
        Record performance metrics relative to frontier model.
        
        Args:
            metrics: Performance metrics
        """
        record = {
            "timestamp": time.time(),
            **metrics
        }
        self.performance_history.append(record)
        
        # Analyze performance for potential checkpoint creation
        recent_performance = self.performance_history[-100:]
        avg_performance = np.mean([
            record.get("response_similarity", 0)
            for record in recent_performance
        ])
        
        # Create checkpoint if performance improved significantly
        if (avg_performance > 0.8 and 
            (not self.current_checkpoint or 
             time.time() - self.current_checkpoint.timestamp > 3600)):  # At least 1 hour between checkpoints
            self._create_checkpoint(
                metadata={
                    "performance_trigger": True,
                    "average_performance": avg_performance,
                    "samples_evaluated": len(recent_performance)
                }
            )
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get comprehensive performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.performance_history:
            return {}
        
        recent_history = self.performance_history[-100:]
        
        # Calculate basic metrics
        metrics = {
            "average_duration": np.mean([r["duration"] for r in recent_history]),
            "tokens_per_second": np.mean([r["tokens_per_second"] for r in recent_history]),
            "average_total_tokens": np.mean([r["total_tokens"] for r in recent_history])
        }
        
        # Calculate similarity metrics if available
        similarities = [
            r.get("response_similarity", None) 
            for r in recent_history 
            if "response_similarity" in r
        ]
        if similarities:
            metrics["average_similarity"] = np.mean(similarities)
            metrics["similarity_std"] = np.std(similarities)
        
        # Add checkpoint information
        if self.current_checkpoint:
            metrics["last_checkpoint_age"] = time.time() - self.current_checkpoint.timestamp
            metrics["total_checkpoints"] = len(self.checkpoint_history)
        
        return metrics
    
    async def save_checkpoint(self, path: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Save a model checkpoint with metadata.
        
        Args:
            path: Path to save checkpoint
            metadata: Optional additional metadata
        """
        checkpoint_dir = Path(path)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        try:
            # Save model and tokenizer
            self.model.save_pretrained(checkpoint_dir)
            self.tokenizer.save_pretrained(checkpoint_dir)
            
            # Save TalkHead state
            torch.save(
                self.talk_head.state_dict(),
                checkpoint_dir / "talk_head.pt"
            )
            
            # Save metadata
            checkpoint_metadata = {
                "timestamp": time.time(),
                "model_name": self.model_config.name,
                "performance_metrics": self.get_performance_metrics(),
                **(metadata or {})
            }
            
            with open(checkpoint_dir / "metadata.json", 'w') as f:
                json.dump(checkpoint_metadata, f, indent=2)
            
            logger.info(f"Saved checkpoint to: {path}")
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            raise
    
    async def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """
        Load a model checkpoint.
        
        Args:
            path: Path to load checkpoint from
            
        Returns:
            Dictionary containing the checkpoint metadata
        """
        checkpoint_dir = Path(path)
        
        try:
            # Load model and tokenizer
            self.model = AutoModelForCausalLM.from_pretrained(
                checkpoint_dir,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
            
            # Load TalkHead state
            talk_head_path = checkpoint_dir / "talk_head.pt"
            if talk_head_path.exists():
                self.talk_head.load_state_dict(torch.load(talk_head_path))
            
            # Move to device if needed
            if self.device == "cpu":
                self.model = self.model.to(self.device)
                self.talk_head = self.talk_head.to(self.device)
            
            # Load metadata
            metadata_path = checkpoint_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {"timestamp": time.time()}
            
            # Create checkpoint record
            self._create_checkpoint(metadata)
            
            logger.info(f"Loaded checkpoint from: {path}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            raise
    
    def get_checkpoint_history(self) -> List[Dict[str, Any]]:
        """
        Get history of model checkpoints.
        
        Returns:
            List of checkpoint metadata
        """
        return [
            {
                "timestamp": cp.timestamp,
                "metadata": cp.metadata
            }
            for cp in self.checkpoint_history
        ]
