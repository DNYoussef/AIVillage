import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

@dataclass
class ActivationConfig:
    """Configuration for latent space activation."""
    attention_heads: int = 4
    activation_threshold: float = 0.5
    context_window_size: int = 5
    pca_components: int = 50

class LatentSpaceActivation:
    """
    Implements latent space activation mechanism for enhanced retrieval.
    """
    
    def __init__(self, config: Optional[ActivationConfig] = None):
        self.config = config or ActivationConfig()
        self.pca = PCA(n_components=self.config.pca_components)
        self.scaler = StandardScaler()
        self.activation_history: List[np.ndarray] = []
        
    async def activate(self, query_embedding: np.ndarray, context: Dict[str, Any]) -> np.ndarray:
        """
        Activate relevant regions in latent space.

        :param query_embedding: The query embedding vector.
        :param context: Additional context for activation.
        :return: The activated embedding vector.
        """
        # Initialize activation map
        activation_map = self._initialize_activation_map(query_embedding)
        
        # Apply context-aware activation
        context_activation = self._compute_context_activation(context)
        activation_map = self._combine_activations(activation_map, context_activation)
        
        # Apply multi-head attention
        attended_activation = self._apply_attention(activation_map, query_embedding)
        
        # Update activation history
        self._update_activation_history(attended_activation)
        
        return attended_activation
        
    def _initialize_activation_map(self, query_embedding: np.ndarray) -> np.ndarray:
        """
        Initialize the activation map based on query embedding.

        :param query_embedding: The query embedding vector.
        :return: Initial activation map.
        """
        # Normalize the query embedding
        normalized_embedding = self.scaler.fit_transform(query_embedding.reshape(1, -1))
        
        # Project to latent space using PCA
        latent_representation = self.pca.fit_transform(normalized_embedding)
        
        # Create initial activation map
        activation_map = np.zeros_like(query_embedding)
        activation_map[np.abs(normalized_embedding) > self.config.activation_threshold] = 1
        
        return activation_map
        
    def _compute_context_activation(self, context: Dict[str, Any]) -> np.ndarray:
        """
        Compute activation based on context.

        :param context: Context information.
        :return: Context-based activation map.
        """
        # Extract relevant features from context
        context_features = self._extract_context_features(context)
        
        # Create context activation map
        context_activation = np.zeros(self.config.pca_components)
        
        if context_features:
            # Normalize context features
            normalized_features = self.scaler.transform(np.array(context_features).reshape(1, -1))
            
            # Project to latent space
            context_latent = self.pca.transform(normalized_features)
            
            # Create activation based on context
            context_activation[np.abs(context_latent) > self.config.activation_threshold] = 1
            
        return context_activation
        
    def _combine_activations(self, base_activation: np.ndarray, context_activation: np.ndarray) -> np.ndarray:
        """
        Combine multiple activation maps.

        :param base_activation: Base activation map.
        :param context_activation: Context-based activation map.
        :return: Combined activation map.
        """
        # Weighted combination of activations
        combined = 0.7 * base_activation + 0.3 * context_activation
        
        # Apply non-linear activation
        combined = self._apply_nonlinear_activation(combined)
        
        return combined
        
    def _apply_attention(self, activation_map: np.ndarray, query_embedding: np.ndarray) -> np.ndarray:
        """
        Apply multi-head attention to activation map.

        :param activation_map: Current activation map.
        :param query_embedding: Original query embedding.
        :return: Attention-weighted activation map.
        """
        attention_weights = []
        
        # Multi-head attention
        for _ in range(self.config.attention_heads):
            # Generate attention weights for this head
            head_weights = self._compute_attention_weights(activation_map, query_embedding)
            attention_weights.append(head_weights)
            
        # Combine attention heads
        combined_attention = np.mean(attention_weights, axis=0)
        
        # Apply attention to activation map
        attended_activation = activation_map * combined_attention
        
        return attended_activation
        
    def _compute_attention_weights(self, activation_map: np.ndarray, query_embedding: np.ndarray) -> np.ndarray:
        """
        Compute attention weights for a single attention head.

        :param activation_map: Current activation map.
        :param query_embedding: Original query embedding.
        :return: Attention weights.
        """
        # Compute similarity between activation map and query
        similarity = np.dot(activation_map, query_embedding.T)
        
        # Apply softmax to get attention weights
        attention = np.exp(similarity) / np.sum(np.exp(similarity))
        
        return attention
        
    def _apply_nonlinear_activation(self, activation: np.ndarray) -> np.ndarray:
        """
        Apply non-linear activation function.

        :param activation: Input activation values.
        :return: Activated values.
        """
        # ReLU activation
        activation = np.maximum(0, activation)
        
        # Normalize
        if np.max(activation) > 0:
            activation = activation / np.max(activation)
            
        return activation
        
    def _extract_context_features(self, context: Dict[str, Any]) -> List[float]:
        """
        Extract numerical features from context.

        :param context: Context dictionary.
        :return: List of context features.
        """
        features = []
        
        # Extract temporal features if available
        if 'timestamp' in context:
            features.append(context['timestamp'].timestamp())
            
        # Extract relevance scores if available
        if 'relevance_scores' in context:
            features.extend(context['relevance_scores'])
            
        # Extract confidence scores if available
        if 'confidence_score' in context:
            features.append(context['confidence_score'])
            
        return features
        
    def _update_activation_history(self, activation: np.ndarray):
        """
        Update activation history for temporal analysis.

        :param activation: Current activation map.
        """
        self.activation_history.append(activation)
        
        # Keep only recent history
        if len(self.activation_history) > self.config.context_window_size:
            self.activation_history.pop(0)
            
    def get_activation_statistics(self) -> Dict[str, float]:
        """
        Get statistics about recent activations.

        :return: Dictionary of activation statistics.
        """
        if not self.activation_history:
            return {}
            
        recent_activations = np.array(self.activation_history)
        
        return {
            'mean_activation': float(np.mean(recent_activations)),
            'std_activation': float(np.std(recent_activations)),
            'max_activation': float(np.max(recent_activations)),
            'min_activation': float(np.min(recent_activations))
        }
        
    def reset(self):
        """Reset the activation state."""
        self.activation_history = []
        self.pca = PCA(n_components=self.config.pca_components)
        self.scaler = StandardScaler()
