"""Embedding utilities for RAG system."""

from typing import List, Union, Optional, Dict, Any
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, PreTrainedModel, PreTrainedTokenizer
from ..core.base_component import BaseComponent
from ..core.config import UnifiedConfig
from ..utils.error_handling import log_and_handle_errors, ErrorContext

class BERTEmbeddingModel(BaseComponent):
    """BERT-based embedding model."""
    
    def __init__(self, config: UnifiedConfig):
        """
        Initialize BERT embedding model.
        
        Args:
            config: Configuration instance
        """
        self.config = config
        self.model_name = config.embedding_model or "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.model: Optional[PreTrainedModel] = None
        self.initialized = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    @log_and_handle_errors()
    async def initialize(self) -> None:
        """Initialize model and tokenizer."""
        if not self.initialized:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.initialized = True
    
    @log_and_handle_errors()
    async def shutdown(self) -> None:
        """Shutdown model."""
        if self.initialized:
            # Clear CUDA cache if using GPU
            if self.device == "cuda":
                torch.cuda.empty_cache()
            self.initialized = False
    
    @log_and_handle_errors()
    async def get_status(self) -> Dict[str, Any]:
        """Get component status."""
        return {
            "initialized": self.initialized,
            "model_name": self.model_name,
            "device": self.device,
            "tokenizer_loaded": self.tokenizer is not None,
            "model_loaded": self.model is not None
        }
    
    @log_and_handle_errors()
    async def update_config(self, config: Dict[str, Any]) -> None:
        """Update component configuration."""
        self.config = config
        if 'embedding_model' in config:
            self.model_name = config['embedding_model']
            # Reinitialize with new model if already initialized
            if self.initialized:
                await self.shutdown()
                await self.initialize()

    async def get_embedding(
        self,
        text: Union[str, List[str]],
        pooling_strategy: str = "mean"
    ) -> Union[List[float], List[List[float]]]:
        """
        Get embeddings for text.
        
        Args:
            text: Input text or list of texts
            pooling_strategy: Pooling strategy ("mean", "max", or "cls")
            
        Returns:
            List of embeddings or list of lists for multiple texts
        """
        async with ErrorContext("BERTEmbeddingModel"):
            if not self.initialized:
                await self.initialize()
            
            # Convert single text to list
            if isinstance(text, str):
                text = [text]
            
            # Tokenize
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Get model outputs
            with torch.no_grad():
                outputs = self.model(**inputs)
                hidden_states = outputs.last_hidden_state
                
                # Apply pooling
                if pooling_strategy == "mean":
                    pooled = hidden_states.mean(dim=1)
                elif pooling_strategy == "max":
                    pooled = hidden_states.max(dim=1)[0]
                elif pooling_strategy == "cls":
                    pooled = hidden_states[:, 0]
                else:
                    raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")
            
            # Convert to numpy and normalize
            embeddings = pooled.cpu().numpy()
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized_embeddings = embeddings / norms
            
            # Return single embedding or list of embeddings
            if len(text) == 1:
                return normalized_embeddings[0].tolist()
            return [emb.tolist() for emb in normalized_embeddings]

    async def get_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """
        Calculate cosine similarity between embeddings.
        
        Args:
            emb1: First embedding
            emb2: Second embedding
            
        Returns:
            Similarity score between 0 and 1
        """
        emb1_np = np.array(emb1)
        emb2_np = np.array(emb2)
        return float(
            np.dot(emb1_np, emb2_np) /
            (np.linalg.norm(emb1_np) * np.linalg.norm(emb2_np))
        )

    async def batch_get_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32,
        pooling_strategy: str = "mean"
    ) -> List[List[float]]:
        """
        Get embeddings for a batch of texts.
        
        Args:
            texts: List of input texts
            batch_size: Size of batches to process
            pooling_strategy: Pooling strategy
            
        Returns:
            List of embeddings
        """
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = await self.get_embedding(batch, pooling_strategy)
            embeddings.extend(batch_embeddings)
        return embeddings

    async def get_contextual_embeddings(
        self,
        text: str,
        window_size: int = 128,
        stride: int = 64,
        pooling_strategy: str = "mean"
    ) -> List[List[float]]:
        """
        Get embeddings for overlapping windows of text.
        
        Args:
            text: Input text
            window_size: Size of each window
            stride: Stride between windows
            pooling_strategy: Pooling strategy
            
        Returns:
            List of embeddings for each window
        """
        # Tokenize text into words
        words = text.split()
        
        # Create windows
        windows = []
        for i in range(0, len(words), stride):
            window = words[i:i + window_size]
            if len(window) >= window_size // 2:  # Only keep substantial windows
                windows.append(" ".join(window))
        
        # Get embeddings for each window
        return await self.get_embedding(windows, pooling_strategy)

# Create default embedding model instance
embedding_model = BERTEmbeddingModel(UnifiedConfig())

def get_embedding(
    text: Union[str, List[str]],
    model_name: Optional[str] = None,
    config: Optional[UnifiedConfig] = None
) -> Union[List[float], List[List[float]]]:
    """
    Get embeddings for text using specified model.
    
    Args:
        text: Input text or list of texts
        model_name: Optional model name to use
        config: Optional configuration
        
    Returns:
        List of embeddings or list of lists for multiple texts
    """
    # Use config model if provided
    if config and not model_name:
        model_name = config.embedding_model
    
    # Default to a lightweight model
    if not model_name:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Convert single text to list
    if isinstance(text, str):
        text = [text]
    
    # Tokenize and get embeddings
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    
    # Convert to numpy and normalize
    embeddings = embeddings.numpy()
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms
    
    # Return single embedding or list of embeddings
    if len(text) == 1:
        return normalized_embeddings[0].tolist()
    return [emb.tolist() for emb in normalized_embeddings]
