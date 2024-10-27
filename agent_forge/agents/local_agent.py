import os
import time
from typing import Dict, Any, Optional, List
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from ..bakedquietiot.deepbaking import DeepSystemBakerTask
from ..bakedquietiot.quiet_star import QuietSTaRTask
from langroid import ChatAgent, ChatAgentConfig
from langroid.language_models.openai_gpt import OpenAIGPTConfig

logger = logging.getLogger(__name__)

class LocalAgent:
    """
    Agent for managing local HuggingFace models that learn from frontier model interactions.
    Uses bakedquietiot system for enhanced reasoning capabilities.
    """
    
    def __init__(self, model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize LocalAgent.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on ("cuda" or "cpu")
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.performance_history: List[Dict[str, float]] = []
        
        # Initialize ChatAgent for bakedquietiot
        config = ChatAgentConfig(
            name="LocalAgent",
            llm=OpenAIGPTConfig(chat_model="gpt-3.5-turbo")
        )
        self.chat_agent = ChatAgent(config)
        
        # Load and bake model
        self._load_and_bake_model()
        
        logger.info(f"Initialized LocalAgent with model: {model_name} on device: {device}")
    
    def _load_and_bake_model(self):
        """Load the model and bake in the IOT quiet star system."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # First, deep bake the system
            baker = DeepSystemBakerTask(
                agent=self.chat_agent,
                model_name=self.model_name,
                device=self.device
            )
            
            # Run deep baking process
            import asyncio
            asyncio.run(baker.deep_bake_system(
                max_iterations=20,  # Adjust based on needs
                consistency_threshold=0.9
            ))
            
            # Now initialize QuietSTaR with the baked model
            quiet_star = QuietSTaRTask(
                agent=self.chat_agent,
                model_path="deep_baked_model"  # Path where baker saved the model
            )
            
            # Store the enhanced model and tokenizer
            self.model = quiet_star.model
            self.tokenizer = quiet_star.tokenizer
            
            # Initialize TalkHead for thought mixing
            self.talk_head = quiet_star.talk_head
            
            logger.info(f"Successfully loaded and baked model")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    async def generate_response(self, 
                              prompt: str, 
                              system_prompt: Optional[str] = None,
                              max_tokens: int = 1000,
                              temperature: float = 0.7) -> Dict[str, Any]:
        """
        Generate a response using the local model with QuietSTaR enhancement.
        
        Args:
            prompt: The input prompt
            system_prompt: Optional system prompt for context
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Dictionary containing the response and metadata
        """
        try:
            # Create QuietSTaR task for this generation
            quiet_star = QuietSTaRTask(
                agent=self.chat_agent,
                model_path="deep_baked_model"
            )
            
            # Prepare input
            if system_prompt:
                input_text = f"{system_prompt}\n\n{prompt}"
            else:
                input_text = prompt
            
            # Generate enhanced response using QuietSTaR
            response = await quiet_star.process_query(input_text)
            
            return {
                "response": response,
                "model": self.model_name,
                "metadata": {
                    "device": self.device,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "enhanced": True
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    def record_performance(self, metrics: Dict[str, float]):
        """
        Record performance metrics relative to frontier model.
        
        Args:
            metrics: Performance metrics (e.g. accuracy, similarity to frontier model)
        """
        record = {
            "timestamp": time.time(),
            **metrics
        }
        self.performance_history.append(record)
        
        # Keep last 1000 performance records
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get average performance metrics over recent history.
        
        Returns:
            Dictionary of averaged performance metrics
        """
        if not self.performance_history:
            return {}
            
        # Get all metric keys
        metric_keys = set()
        for record in self.performance_history:
            metric_keys.update(record.keys())
        metric_keys.remove("timestamp")
        
        # Calculate averages
        averages = {}
        for key in metric_keys:
            values = [
                record[key] 
                for record in self.performance_history 
                if key in record
            ]
            if values:
                averages[key] = sum(values) / len(values)
                
        return averages
    
    def save_checkpoint(self, path: str):
        """
        Save a model checkpoint.
        
        Args:
            path: Path to save checkpoint to
        """
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        # Save TalkHead state
        torch.save(self.talk_head.state_dict(), os.path.join(path, "talk_head.pt"))
        logger.info(f"Saved checkpoint to: {path}")
    
    def load_checkpoint(self, path: str):
        """
        Load a model checkpoint.
        
        Args:
            path: Path to load checkpoint from
        """
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        
        # Load TalkHead state
        talk_head_path = os.path.join(path, "talk_head.pt")
        if os.path.exists(talk_head_path):
            self.talk_head.load_state_dict(torch.load(talk_head_path))
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
            self.talk_head = self.talk_head.to(self.device)
            
        logger.info(f"Loaded checkpoint from: {path}")
