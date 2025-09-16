"""
Quiet-STaR Implementation: Core Components for Thought Generation and Coherence Validation

This module implements the core Quiet-STaR (Self-Taught Reasoner) architecture including:
- ThoughtGenerator: Parallel thought generation with special token handling
- Thought Injection System: Coherent thought integration into model processing
- Coherence Validator: Thought quality assessment and filtering

Reference: Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking
https://arxiv.org/abs/2403.09629
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
import logging
from transformers import AutoTokenizer, AutoModel
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ThoughtConfig:
    """Configuration for Quiet-STaR thought generation"""
    num_thoughts: int = 4  # Number of parallel thoughts to generate
    thought_length: int = 32  # Tokens per thought
    coherence_threshold: float = 0.6  # Minimum coherence score to accept thought
    temperature: float = 0.8  # Temperature for thought generation
    top_p: float = 0.9  # Top-p sampling for thought diversity
    special_tokens: Dict[str, str] = None  # Special tokens for thought boundaries

    def __post_init__(self):
        if self.special_tokens is None:
            self.special_tokens = {
                'start_thought': '<|startofthought|>',
                'end_thought': '<|endofthought|>',
                'thought_sep': '<|thoughtsep|>'
            }


class ThoughtGenerator(nn.Module):
    """
    Generates multiple parallel thoughts for input sequences.

    Implements the core thought generation mechanism from Quiet-STaR,
    creating diverse reasoning paths that can be evaluated for coherence.
    """

    def __init__(self, model, tokenizer, config: ThoughtConfig):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        # Add special tokens to tokenizer if not present
        self._add_special_tokens()

        # Thought projection layers for parallel generation
        self.thought_projector = nn.Linear(
            model.config.hidden_size,
            model.config.hidden_size * config.num_thoughts
        )

        # Attention mechanism for thought selection
        self.thought_attention = nn.MultiheadAttention(
            embed_dim=model.config.hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        logger.info(f"ThoughtGenerator initialized with {config.num_thoughts} parallel thoughts")

    def _add_special_tokens(self):
        """Add special tokens for thought boundaries to tokenizer"""
        special_tokens = list(self.config.special_tokens.values())
        num_added = self.tokenizer.add_special_tokens({
            'additional_special_tokens': special_tokens
        })

        if num_added > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))
            logger.info(f"Added {num_added} special tokens to tokenizer")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate parallel thoughts for input sequences

        Args:
            input_ids: Input token sequences [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Dictionary containing generated thoughts and metadata
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Get model hidden states
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # Use last token representation for thought generation
        last_hidden = hidden_states[:, -1, :]  # [batch_size, hidden_size]

        # Project to thought space
        thought_features = self.thought_projector(last_hidden)  # [batch_size, hidden_size * num_thoughts]
        thought_features = thought_features.view(
            batch_size, self.config.num_thoughts, -1
        )  # [batch_size, num_thoughts, hidden_size]

        # Generate thoughts for each parallel branch
        all_thoughts = []
        thought_scores = []

        for thought_idx in range(self.config.num_thoughts):
            thought_hidden = thought_features[:, thought_idx, :]  # [batch_size, hidden_size]

            # Generate thought sequence
            thought_tokens, thought_score = self._generate_thought_sequence(
                input_ids, thought_hidden, thought_idx
            )

            all_thoughts.append(thought_tokens)
            thought_scores.append(thought_score)

        return {
            'thoughts': all_thoughts,  # List of [batch_size, thought_length] tensors
            'thought_scores': torch.stack(thought_scores),  # [num_thoughts, batch_size]
            'thought_features': thought_features,  # [batch_size, num_thoughts, hidden_size]
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

    def _generate_thought_sequence(self, input_ids: torch.Tensor, thought_hidden: torch.Tensor,
                                 thought_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a single thought sequence with special token handling"""
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Start with special start token
        start_token_id = self.tokenizer.convert_tokens_to_ids(self.config.special_tokens['start_thought'])
        current_ids = torch.full((batch_size, 1), start_token_id, device=device)

        # Track generation scores
        generation_scores = []

        # Generate thought tokens
        for step in range(self.config.thought_length - 2):  # -2 for start/end tokens
            # Concatenate input and current thought
            full_input = torch.cat([input_ids, current_ids], dim=1)

            # Get next token logits
            with torch.no_grad():
                outputs = self.model(input_ids=full_input)
                next_token_logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]

            # Apply temperature and top-p sampling
            next_token_logits = next_token_logits / self.config.temperature

            # Top-p filtering
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > self.config.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Apply filtering
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = float('-inf')

            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [batch_size, 1]

            # Track generation quality
            token_scores = torch.gather(probs, 1, next_token)
            generation_scores.append(token_scores.squeeze())

            # Append to current sequence
            current_ids = torch.cat([current_ids, next_token], dim=1)

        # Add end token
        end_token_id = self.tokenizer.convert_tokens_to_ids(self.config.special_tokens['end_thought'])
        end_token = torch.full((batch_size, 1), end_token_id, device=device)
        current_ids = torch.cat([current_ids, end_token], dim=1)

        # Calculate average generation score
        if generation_scores:
            avg_score = torch.stack(generation_scores).mean(dim=0)  # [batch_size]
        else:
            avg_score = torch.ones(batch_size, device=device) * 0.5

        return current_ids, avg_score


class ThoughtInjectionSystem(nn.Module):
    """
    Injects generated thoughts into model processing while maintaining sequence coherence.

    Handles the integration of parallel thoughts into the main reasoning process,
    ensuring proper attention flow and maintaining batch processing efficiency.
    """

    def __init__(self, model, config: ThoughtConfig):
        super().__init__()
        self.model = model
        self.config = config

        # Thought fusion layers
        self.thought_fusion = nn.MultiheadAttention(
            embed_dim=model.config.hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # Gating mechanism for thought integration
        self.thought_gate = nn.Sequential(
            nn.Linear(model.config.hidden_size * 2, model.config.hidden_size),
            nn.Sigmoid()
        )

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(model.config.hidden_size)

        logger.info("ThoughtInjectionSystem initialized")

    def inject_thoughts(self, base_hidden: torch.Tensor, thought_hidden: torch.Tensor,
                       attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Inject thoughts into base hidden states

        Args:
            base_hidden: Base model hidden states [batch_size, seq_len, hidden_size]
            thought_hidden: Thought representations [batch_size, num_thoughts, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Enhanced hidden states with injected thoughts
        """
        batch_size, seq_len, hidden_size = base_hidden.shape

        # Prepare thought representations for fusion
        thought_queries = base_hidden  # Use base hidden as queries
        thought_keys = thought_hidden  # Thoughts as keys
        thought_values = thought_hidden  # Thoughts as values

        # Apply attention-based fusion
        fused_thoughts, attention_weights = self.thought_fusion(
            query=thought_queries,
            key=thought_keys,
            value=thought_values,
            need_weights=True
        )

        # Gating mechanism to control thought integration
        gate_input = torch.cat([base_hidden, fused_thoughts], dim=-1)
        gate_weights = self.thought_gate(gate_input)  # [batch_size, seq_len, hidden_size]

        # Combine base and thought representations
        enhanced_hidden = gate_weights * fused_thoughts + (1 - gate_weights) * base_hidden

        # Apply layer normalization
        enhanced_hidden = self.layer_norm(enhanced_hidden)

        return enhanced_hidden, attention_weights

    def process_batch(self, input_ids: torch.Tensor, thoughts: List[torch.Tensor],
                     attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process a batch with thought injection

        Args:
            input_ids: Input token sequences [batch_size, seq_len]
            thoughts: List of thought sequences
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Enhanced model outputs with thought integration
        """
        # Get base model representations
        base_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        base_hidden = base_outputs.last_hidden_state

        # Process thoughts into hidden representations
        thought_representations = []
        for thought_batch in thoughts:
            thought_outputs = self.model(input_ids=thought_batch)
            thought_hidden = thought_outputs.last_hidden_state.mean(dim=1)  # Pool over sequence
            thought_representations.append(thought_hidden)

        # Stack thought representations
        thought_hidden = torch.stack(thought_representations, dim=1)  # [batch_size, num_thoughts, hidden_size]

        # Inject thoughts into base representations
        enhanced_hidden, attention_weights = self.inject_thoughts(
            base_hidden, thought_hidden, attention_mask
        )

        return {
            'enhanced_hidden': enhanced_hidden,
            'base_hidden': base_hidden,
            'thought_hidden': thought_hidden,
            'attention_weights': attention_weights,
            'base_outputs': base_outputs
        }


class CoherenceValidator:
    """
    Validates thought coherence and scores thoughts on a 0-1 scale.

    Implements multiple coherence metrics including semantic consistency,
    logical flow, and relevance to the input context.
    """

    def __init__(self, model, tokenizer, config: ThoughtConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        # Initialize coherence metrics
        self.metrics = {
            'semantic_similarity': self._semantic_similarity,
            'logical_consistency': self._logical_consistency,
            'relevance_score': self._relevance_score,
            'fluency_score': self._fluency_score
        }

        logger.info("CoherenceValidator initialized with 4 metric types")

    def validate_thoughts(self, input_ids: torch.Tensor, thoughts: List[torch.Tensor],
                         attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Validate coherence of generated thoughts

        Args:
            input_ids: Original input sequences [batch_size, seq_len]
            thoughts: Generated thought sequences
            attention_mask: Attention mask for input

        Returns:
            Coherence scores and filtering decisions
        """
        batch_size = input_ids.shape[0]
        num_thoughts = len(thoughts)

        # Initialize score storage
        coherence_scores = torch.zeros(batch_size, num_thoughts)
        metric_scores = {metric: torch.zeros(batch_size, num_thoughts) for metric in self.metrics}

        # Evaluate each thought
        for batch_idx in range(batch_size):
            for thought_idx, thought_batch in enumerate(thoughts):
                input_seq = input_ids[batch_idx:batch_idx+1]
                thought_seq = thought_batch[batch_idx:batch_idx+1]

                # Calculate individual metric scores
                scores = {}
                for metric_name, metric_func in self.metrics.items():
                    score = metric_func(input_seq, thought_seq)
                    scores[metric_name] = score
                    metric_scores[metric_name][batch_idx, thought_idx] = score

                # Compute weighted coherence score
                coherence_score = self._compute_weighted_score(scores)
                coherence_scores[batch_idx, thought_idx] = coherence_score

        # Filter thoughts based on coherence threshold
        valid_thoughts = coherence_scores >= self.config.coherence_threshold

        return {
            'coherence_scores': coherence_scores,
            'metric_scores': metric_scores,
            'valid_thoughts': valid_thoughts,
            'filtered_thoughts': self._filter_thoughts(thoughts, valid_thoughts)
        }

    def _semantic_similarity(self, input_ids: torch.Tensor, thought_ids: torch.Tensor) -> float:
        """Calculate semantic similarity between input and thought"""
        with torch.no_grad():
            # Get embeddings for input and thought
            input_outputs = self.model(input_ids=input_ids)
            thought_outputs = self.model(input_ids=thought_ids)

            # Use mean pooling for sequence representation
            input_repr = input_outputs.last_hidden_state.mean(dim=1)
            thought_repr = thought_outputs.last_hidden_state.mean(dim=1)

            # Calculate cosine similarity
            similarity = F.cosine_similarity(input_repr, thought_repr, dim=-1)
            return similarity.item()

    def _logical_consistency(self, input_ids: torch.Tensor, thought_ids: torch.Tensor) -> float:
        """Evaluate logical consistency of thought progression"""
        # Simple heuristic: measure consistency of token transitions
        with torch.no_grad():
            thought_outputs = self.model(input_ids=thought_ids)
            logits = thought_outputs.logits[0]  # [seq_len, vocab_size]

            # Calculate entropy of token distributions
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)

            # Lower entropy indicates more confident/consistent predictions
            consistency_score = 1.0 - (entropy.mean() / torch.log(torch.tensor(len(self.tokenizer))))
            return max(0.0, consistency_score.item())

    def _relevance_score(self, input_ids: torch.Tensor, thought_ids: torch.Tensor) -> float:
        """Calculate relevance of thought to input context"""
        # Extract key tokens from input and thought
        input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        thought_tokens = self.tokenizer.convert_ids_to_tokens(thought_ids[0])

        # Remove special tokens
        input_tokens = [t for t in input_tokens if not t.startswith('<|')]
        thought_tokens = [t for t in thought_tokens if not t.startswith('<|')]

        if not input_tokens or not thought_tokens:
            return 0.0

        # Calculate token overlap
        input_set = set(input_tokens)
        thought_set = set(thought_tokens)

        overlap = len(input_set.intersection(thought_set))
        total_unique = len(input_set.union(thought_set))

        relevance = overlap / total_unique if total_unique > 0 else 0.0
        return relevance

    def _fluency_score(self, input_ids: torch.Tensor, thought_ids: torch.Tensor) -> float:
        """Evaluate fluency of generated thought"""
        with torch.no_grad():
            # Calculate perplexity as fluency measure
            thought_outputs = self.model(input_ids=thought_ids, labels=thought_ids)
            perplexity = torch.exp(thought_outputs.loss)

            # Convert perplexity to 0-1 score (lower perplexity = higher fluency)
            fluency = 1.0 / (1.0 + perplexity.item() / 10.0)  # Normalize
            return min(1.0, max(0.0, fluency))

    def _compute_weighted_score(self, scores: Dict[str, float]) -> float:
        """Compute weighted coherence score from individual metrics"""
        weights = {
            'semantic_similarity': 0.3,
            'logical_consistency': 0.3,
            'relevance_score': 0.25,
            'fluency_score': 0.15
        }

        weighted_score = sum(weights[metric] * score for metric, score in scores.items())
        return weighted_score

    def _filter_thoughts(self, thoughts: List[torch.Tensor],
                        valid_mask: torch.Tensor) -> List[torch.Tensor]:
        """Filter thoughts based on validity mask"""
        filtered_thoughts = []

        for thought_idx, thought_batch in enumerate(thoughts):
            # Create mask for this thought across all batches
            batch_mask = valid_mask[:, thought_idx]

            if batch_mask.any():  # Keep thought if valid for any batch item
                filtered_thoughts.append(thought_batch)

        return filtered_thoughts


class QuietSTaR(nn.Module):
    """
    Main Quiet-STaR orchestrator that coordinates all components.

    Implements the complete Quiet-STaR pipeline:
    1. Generate parallel thoughts
    2. Validate coherence
    3. Inject valid thoughts into processing
    4. Return enhanced outputs
    """

    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", config: ThoughtConfig = None):
        super().__init__()

        # Initialize configuration
        self.config = config or ThoughtConfig()

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize components
        self.thought_generator = ThoughtGenerator(self.model, self.tokenizer, self.config)
        self.injection_system = ThoughtInjectionSystem(self.model, self.config)
        self.coherence_validator = CoherenceValidator(self.model, self.tokenizer, self.config)

        # Performance tracking
        self.metrics = {
            'total_thoughts_generated': 0,
            'valid_thoughts_count': 0,
            'avg_coherence_score': 0.0,
            'processing_time': 0.0
        }

        logger.info(f"QuietSTaR initialized with model: {model_name}")
        logger.info(f"Configuration: {self.config}")

    def forward(self, input_text: str, return_thoughts: bool = True) -> Dict[str, Any]:
        """
        Main forward pass with thought generation and injection

        Args:
            input_text: Input text to process
            return_thoughts: Whether to return generated thoughts

        Returns:
            Dictionary containing enhanced outputs and metadata
        """
        start_time = time.time()

        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        try:
            # Step 1: Generate parallel thoughts
            thought_outputs = self.thought_generator(input_ids, attention_mask)
            self.metrics['total_thoughts_generated'] += len(thought_outputs['thoughts'])

            # Step 2: Validate thought coherence
            validation_results = self.coherence_validator.validate_thoughts(
                input_ids, thought_outputs['thoughts'], attention_mask
            )

            # Update metrics
            valid_count = validation_results['valid_thoughts'].sum().item()
            self.metrics['valid_thoughts_count'] += valid_count
            self.metrics['avg_coherence_score'] = validation_results['coherence_scores'].mean().item()

            # Step 3: Inject valid thoughts
            if validation_results['filtered_thoughts']:
                injection_results = self.injection_system.process_batch(
                    input_ids, validation_results['filtered_thoughts'], attention_mask
                )
            else:
                # Fallback to base processing if no valid thoughts
                base_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                injection_results = {
                    'enhanced_hidden': base_outputs.last_hidden_state,
                    'base_hidden': base_outputs.last_hidden_state,
                    'thought_hidden': None,
                    'attention_weights': None,
                    'base_outputs': base_outputs
                }

            # Calculate processing time
            processing_time = time.time() - start_time
            self.metrics['processing_time'] = processing_time

            # Prepare return dictionary
            result = {
                'enhanced_hidden': injection_results['enhanced_hidden'],
                'base_hidden': injection_results['base_hidden'],
                'coherence_scores': validation_results['coherence_scores'],
                'valid_thoughts_mask': validation_results['valid_thoughts'],
                'metrics': self.metrics.copy(),
                'processing_time': processing_time
            }

            if return_thoughts:
                result.update({
                    'raw_thoughts': thought_outputs['thoughts'],
                    'filtered_thoughts': validation_results['filtered_thoughts'],
                    'thought_features': thought_outputs['thought_features'],
                    'attention_weights': injection_results.get('attention_weights'),
                    'metric_scores': validation_results['metric_scores']
                })

            return result

        except Exception as e:
            logger.error(f"Error in QuietSTaR forward pass: {e}")
            # Return fallback result
            base_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return {
                'enhanced_hidden': base_outputs.last_hidden_state,
                'base_hidden': base_outputs.last_hidden_state,
                'error': str(e),
                'processing_time': time.time() - start_time
            }

    async def async_forward(self, input_texts: List[str], return_thoughts: bool = True) -> List[Dict[str, Any]]:
        """
        Asynchronous processing for multiple inputs

        Args:
            input_texts: List of input texts to process
            return_thoughts: Whether to return generated thoughts

        Returns:
            List of result dictionaries
        """
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(self.forward, text, return_thoughts)
                for text in input_texts
            ]

            results = []
            for future in futures:
                try:
                    result = future.result(timeout=30)  # 30 second timeout
                    results.append(result)
                except Exception as e:
                    logger.error(f"Async processing error: {e}")
                    results.append({'error': str(e)})

            return results

    def get_thought_text(self, thought_ids: torch.Tensor) -> str:
        """Convert thought token IDs back to readable text"""
        return self.tokenizer.decode(thought_ids[0], skip_special_tokens=False)

    def reset_metrics(self):
        """Reset performance tracking metrics"""
        self.metrics = {
            'total_thoughts_generated': 0,
            'valid_thoughts_count': 0,
            'avg_coherence_score': 0.0,
            'processing_time': 0.0
        }

    def get_statistics(self) -> Dict[str, float]:
        """Get current performance statistics"""
        total_thoughts = self.metrics['total_thoughts_generated']
        valid_thoughts = self.metrics['valid_thoughts_count']

        return {
            'thought_validity_rate': valid_thoughts / total_thoughts if total_thoughts > 0 else 0.0,
            'avg_coherence_score': self.metrics['avg_coherence_score'],
            'avg_processing_time': self.metrics['processing_time'],
            'total_thoughts_generated': total_thoughts,
            'valid_thoughts_count': valid_thoughts
        }


# Example usage and testing functions
def create_demo_config() -> ThoughtConfig:
    """Create a demonstration configuration"""
    return ThoughtConfig(
        num_thoughts=4,
        thought_length=32,
        coherence_threshold=0.6,
        temperature=0.8,
        top_p=0.9
    )


def demo_quietstar():
    """Demonstration of Quiet-STaR functionality"""
    print("Initializing Quiet-STaR Demo...")

    # Create configuration
    config = create_demo_config()

    # Initialize Quiet-STaR
    quietstar = QuietSTaR(config=config)

    # Test input
    test_input = "The key to solving this problem is to think step by step about the underlying principles."

    print(f"\nProcessing input: {test_input}")

    # Process input
    result = quietstar(test_input)

    # Display results
    print("\n=== Quiet-STaR Results ===")
    print(f"Processing time: {result['processing_time']:.3f}s")
    print(f"Average coherence score: {result['coherence_scores'].mean():.3f}")
    print(f"Valid thoughts: {result['valid_thoughts_mask'].sum()}/{len(result['raw_thoughts'])}")

    # Show statistics
    stats = quietstar.get_statistics()
    print(f"\n=== Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value:.3f}")

    return result


if __name__ == "__main__":
    # Run demonstration
    demo_result = demo_quietstar()
    print("\nQuiet-STaR demonstration completed successfully!")