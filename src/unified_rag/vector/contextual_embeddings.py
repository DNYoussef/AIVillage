"""
Contextual Embedding Engine for Dual Context Vector RAG
Generates embeddings with dual context awareness (content + semantic context)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

@dataclass
class ContextualEmbedding:
    """Embedding with dual context information"""
    content_embedding: np.ndarray
    semantic_embedding: np.ndarray  
    combined_embedding: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    context_tags: List[str] = field(default_factory=list)

class ContextualEmbeddingEngine:
    """
    Generates embeddings with dual context awareness
    Combines content embeddings with semantic context embeddings
    """
    
    def __init__(self, 
                 content_model: str = "all-MiniLM-L6-v2",
                 semantic_model: str = "all-MiniLM-L6-v2",
                 embedding_dim: int = 384):
        """
        Initialize contextual embedding engine
        
        Args:
            content_model: Model for content embeddings
            semantic_model: Model for semantic context embeddings  
            embedding_dim: Dimension of final embeddings
        """
        self.embedding_dim = embedding_dim
        
        # Initialize models
        try:
            self.content_encoder = SentenceTransformer(content_model)
            self.semantic_encoder = SentenceTransformer(semantic_model)
            logger.info(f"Initialized embedding models: {content_model}, {semantic_model}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding models: {e}")
            raise
        
        # Context weight settings
        self.content_weight = 0.7
        self.semantic_weight = 0.3
    
    async def embed_with_context(self, 
                                text: str,
                                semantic_context: Optional[str] = None,
                                context_tags: Optional[List[str]] = None,
                                metadata: Optional[Dict[str, Any]] = None) -> ContextualEmbedding:
        """
        Generate contextual embedding for text with semantic context
        
        Args:
            text: Primary text content to embed
            semantic_context: Additional semantic context 
            context_tags: List of context tags
            metadata: Additional metadata
            
        Returns:
            ContextualEmbedding with dual embeddings
        """
        try:
            # Generate content embedding
            content_embedding = self.content_encoder.encode(text)
            
            # Generate semantic context embedding
            if semantic_context:
                semantic_text = f"{text} [CONTEXT] {semantic_context}"
                semantic_embedding = self.semantic_encoder.encode(semantic_text)
            else:
                semantic_embedding = content_embedding.copy()
            
            # Combine embeddings with weights
            combined_embedding = (
                self.content_weight * content_embedding + 
                self.semantic_weight * semantic_embedding
            )
            
            # Normalize combined embedding
            combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)
            
            return ContextualEmbedding(
                content_embedding=content_embedding,
                semantic_embedding=semantic_embedding,
                combined_embedding=combined_embedding,
                metadata=metadata or {},
                context_tags=context_tags or []
            )
            
        except Exception as e:
            logger.error(f"Error generating contextual embedding: {e}")
            raise
    
    async def embed_batch_with_context(self,
                                     texts: List[str],
                                     contexts: Optional[List[str]] = None,
                                     context_tags: Optional[List[List[str]]] = None) -> List[ContextualEmbedding]:
        """
        Generate contextual embeddings for batch of texts
        
        Args:
            texts: List of texts to embed
            contexts: Optional list of semantic contexts
            context_tags: Optional list of context tag lists
            
        Returns:
            List of ContextualEmbedding objects
        """
        if contexts is None:
            contexts = [None] * len(texts)
        if context_tags is None:
            context_tags = [None] * len(texts)
        
        embeddings = []
        
        # Process in batches for efficiency
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_contexts = contexts[i:i+batch_size]
            batch_tags = context_tags[i:i+batch_size]
            
            batch_embeddings = await asyncio.gather(*[
                self.embed_with_context(text, context, tags)
                for text, context, tags in zip(batch_texts, batch_contexts, batch_tags)
            ])
            
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def compute_similarity(self, 
                         embedding1: ContextualEmbedding,
                         embedding2: ContextualEmbedding,
                         use_combined: bool = True) -> float:
        """
        Compute similarity between two contextual embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding  
            use_combined: Whether to use combined or content-only embeddings
            
        Returns:
            Cosine similarity score
        """
        try:
            if use_combined:
                vec1 = embedding1.combined_embedding
                vec2 = embedding2.combined_embedding
            else:
                vec1 = embedding1.content_embedding
                vec2 = embedding2.content_embedding
            
            # Cosine similarity
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error computing embedding similarity: {e}")
            return 0.0
    
    def search_similar_embeddings(self,
                                query_embedding: ContextualEmbedding,
                                candidate_embeddings: List[ContextualEmbedding],
                                top_k: int = 10,
                                similarity_threshold: float = 0.5) -> List[Tuple[int, float]]:
        """
        Search for similar embeddings
        
        Args:
            query_embedding: Query embedding
            candidate_embeddings: List of candidate embeddings
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of (index, similarity_score) tuples
        """
        similarities = []
        
        for i, candidate in enumerate(candidate_embeddings):
            similarity = self.compute_similarity(query_embedding, candidate)
            if similarity >= similarity_threshold:
                similarities.append((i, similarity))
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def adjust_context_weights(self, content_weight: float, semantic_weight: float):
        """
        Adjust the weights for combining content and semantic embeddings
        
        Args:
            content_weight: Weight for content embeddings
            semantic_weight: Weight for semantic embeddings
        """
        total_weight = content_weight + semantic_weight
        self.content_weight = content_weight / total_weight
        self.semantic_weight = semantic_weight / total_weight
        
        logger.info(f"Updated context weights: content={self.content_weight:.2f}, semantic={self.semantic_weight:.2f}")
    
    def get_embedding_stats(self, embeddings: List[ContextualEmbedding]) -> Dict[str, Any]:
        """
        Get statistics about a set of embeddings
        
        Args:
            embeddings: List of contextual embeddings
            
        Returns:
            Dictionary with embedding statistics
        """
        if not embeddings:
            return {}
        
        combined_embeddings = np.array([emb.combined_embedding for emb in embeddings])
        
        return {
            'count': len(embeddings),
            'embedding_dim': combined_embeddings.shape[1],
            'mean_norm': float(np.mean(np.linalg.norm(combined_embeddings, axis=1))),
            'std_norm': float(np.std(np.linalg.norm(combined_embeddings, axis=1))),
            'context_weight': self.content_weight,
            'semantic_weight': self.semantic_weight
        }