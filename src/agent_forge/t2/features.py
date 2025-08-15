"""
Feature extraction for Transformer² dispatch decisions.
Extracts prompt statistics, activation sketches, and logits entropy for expert routing.
"""

import logging
import re
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts features for expert dispatch decisions in Transformer².

    Supports three main feature types:
    1. prompt_stats: Static analysis of input prompts
    2. logits_entropy: Entropy of output logits distribution
    3. activation_sketch: Summary statistics of internal activations
    """

    def __init__(self):
        # Compile regex patterns for efficiency
        self.code_patterns = {
            "function_def": re.compile(r"\b(?:def|function|func)\s+\w+", re.IGNORECASE),
            "class_def": re.compile(r"\bclass\s+\w+", re.IGNORECASE),
            "api_calls": re.compile(r"\w+\.[a-zA-Z_]\w*\([^)]*\)"),
            "imports": re.compile(r"\b(?:import|from)\s+\w+", re.IGNORECASE),
            "control_flow": re.compile(
                r"\b(?:if|for|while|try|except|with)\b", re.IGNORECASE
            ),
            "data_structures": re.compile(
                r"\b(?:list|dict|tuple|set|array)\b", re.IGNORECASE
            ),
        }

        self.math_patterns = {
            "equations": re.compile(r"[=]\s*[^=]"),
            "operators": re.compile(r"[+\-*/^√∑∏∫]"),
            "numbers": re.compile(r"\b\d+\.?\d*\b"),
            "variables": re.compile(r"\b[a-zA-Z]\b(?!\w)"),
            "math_functions": re.compile(
                r"\b(?:sin|cos|tan|log|exp|sqrt|max|min|sum|mean)\b", re.IGNORECASE
            ),
        }

    def extract_prompt_stats(self, prompt: str) -> dict[str, float]:
        """
        Extract statistical features from input prompt.

        Args:
            prompt: Input text prompt

        Returns:
            Dictionary of prompt statistics
        """
        if not prompt or not isinstance(prompt, str):
            return self._empty_prompt_stats()

        prompt_len = len(prompt)
        if prompt_len == 0:
            return self._empty_prompt_stats()

        # Basic text statistics
        word_count = len(prompt.split())
        char_count = prompt_len
        line_count = len(prompt.splitlines())
        avg_word_len = (
            np.mean([len(word) for word in prompt.split()]) if word_count > 0 else 0
        )

        # Code-related features
        code_features = self._extract_code_features(prompt)

        # Mathematical content features
        math_features = self._extract_math_features(prompt)

        # Linguistic features
        linguistic_features = self._extract_linguistic_features(prompt)

        # Combine all features
        stats = {
            # Basic statistics
            "size": min(char_count / 1000, 2.0),  # Normalize to [0, 2]
            "word_count": min(word_count / 500, 2.0),  # Normalize
            "line_count": min(line_count / 50, 2.0),
            "avg_word_length": min(avg_word_len / 10, 1.0),
            # Code features
            "code_ratio": code_features["code_ratio"],
            "function_density": code_features["function_density"],
            "api_density": code_features["api_density"],
            "control_complexity": code_features["control_complexity"],
            # Math features
            "math_ratio": math_features["math_ratio"],
            "equation_density": math_features["equation_density"],
            "number_density": math_features["number_density"],
            # Linguistic features
            "question_ratio": linguistic_features["question_ratio"],
            "imperative_ratio": linguistic_features["imperative_ratio"],
            "complexity_score": linguistic_features["complexity_score"],
        }

        # Extract API keywords for routing
        stats["api_keywords"] = self._extract_api_keywords(prompt)

        return stats

    def _extract_code_features(self, prompt: str) -> dict[str, float]:
        """Extract coding-related features."""
        prompt_len = len(prompt)
        if prompt_len == 0:
            return {
                "code_ratio": 0,
                "function_density": 0,
                "api_density": 0,
                "control_complexity": 0,
            }

        # Count code patterns
        code_indicators = 0
        for pattern in self.code_patterns.values():
            code_indicators += len(pattern.findall(prompt))

        # Specific pattern densities
        function_count = len(self.code_patterns["function_def"].findall(prompt))
        api_count = len(self.code_patterns["api_calls"].findall(prompt))
        control_count = len(self.code_patterns["control_flow"].findall(prompt))

        return {
            "code_ratio": min(code_indicators / (prompt_len / 100), 1.0),
            "function_density": min(function_count / (prompt_len / 200), 1.0),
            "api_density": min(api_count / (prompt_len / 150), 1.0),
            "control_complexity": min(control_count / (prompt_len / 100), 1.0),
        }

    def _extract_math_features(self, prompt: str) -> dict[str, float]:
        """Extract mathematics-related features."""
        prompt_len = len(prompt)
        if prompt_len == 0:
            return {"math_ratio": 0, "equation_density": 0, "number_density": 0}

        # Count math patterns
        math_indicators = 0
        for pattern in self.math_patterns.values():
            math_indicators += len(pattern.findall(prompt))

        equation_count = len(self.math_patterns["equations"].findall(prompt))
        number_count = len(self.math_patterns["numbers"].findall(prompt))

        return {
            "math_ratio": min(math_indicators / (prompt_len / 50), 1.0),
            "equation_density": min(equation_count / (prompt_len / 100), 1.0),
            "number_density": min(number_count / (prompt_len / 50), 1.0),
        }

    def _extract_linguistic_features(self, prompt: str) -> dict[str, float]:
        """Extract linguistic and structural features."""
        if not prompt:
            return {"question_ratio": 0, "imperative_ratio": 0, "complexity_score": 0}

        sentences = re.split(r"[.!?]+", prompt)
        if not sentences:
            return {"question_ratio": 0, "imperative_ratio": 0, "complexity_score": 0}

        question_count = prompt.count("?")
        imperative_patterns = len(
            re.findall(
                r"\b(?:write|implement|create|build|make|generate|solve)\b",
                prompt,
                re.IGNORECASE,
            )
        )

        # Complexity heuristic based on sentence structure
        avg_sentence_len = np.mean([len(s.split()) for s in sentences if s.strip()])
        nested_depth = prompt.count("(") + prompt.count("[") + prompt.count("{")

        complexity = min((avg_sentence_len / 20 + nested_depth / 10) / 2, 1.0)

        return {
            "question_ratio": min(question_count / len(sentences), 1.0),
            "imperative_ratio": min(imperative_patterns / len(sentences), 1.0),
            "complexity_score": complexity,
        }

    def _extract_api_keywords(self, prompt: str) -> list[str]:
        """Extract API-related keywords for specialized routing."""
        # Common API patterns and frameworks
        api_patterns = {
            "web": r"\b(?:http|url|api|rest|json|xml|html|css|javascript|react|vue|angular)\b",
            "data": r"\b(?:pandas|numpy|sklearn|tensorflow|pytorch|data|csv|sql|database)\b",
            "system": r"\b(?:file|path|os|sys|subprocess|threading|async|await)\b",
            "math": r"\b(?:math|numpy|scipy|statistics|algorithm|sort|search|graph)\b",
            "ml": r"\b(?:model|train|predict|feature|classification|regression|neural|network)\b",
        }

        keywords = []
        prompt_lower = prompt.lower()

        for category, pattern in api_patterns.items():
            matches = re.findall(pattern, prompt_lower, re.IGNORECASE)
            if matches:
                keywords.extend(
                    [f"{category}:{match}" for match in set(matches[:3])]
                )  # Limit per category

        return keywords[:10]  # Limit total keywords

    def _empty_prompt_stats(self) -> dict[str, float]:
        """Return empty/default prompt statistics."""
        return {
            "size": 0,
            "word_count": 0,
            "line_count": 0,
            "avg_word_length": 0,
            "code_ratio": 0,
            "function_density": 0,
            "api_density": 0,
            "control_complexity": 0,
            "math_ratio": 0,
            "equation_density": 0,
            "number_density": 0,
            "question_ratio": 0,
            "imperative_ratio": 0,
            "complexity_score": 0,
            "api_keywords": [],
        }

    def extract_logits_entropy(self, logits: torch.Tensor) -> dict[str, float]:
        """
        Extract entropy features from model logits.

        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size]

        Returns:
            Dictionary of entropy-based features
        """
        if logits.numel() == 0:
            return {"h0": 0.0, "h_mean": 0.0, "h_std": 0.0, "h_max": 0.0}

        with torch.no_grad():
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)

            # Compute entropy: H = -sum(p * log(p))
            log_probs = F.log_softmax(logits, dim=-1)
            entropy = -torch.sum(probs * log_probs, dim=-1)  # [batch_size, seq_len]

            # Extract statistics
            h0 = entropy[0, 0].item() if entropy.numel() > 0 else 0.0
            h_mean = entropy.mean().item()
            h_std = entropy.std().item() if entropy.numel() > 1 else 0.0
            h_max = entropy.max().item()

            return {
                "h0": h0,  # First token entropy
                "h_mean": h_mean,  # Mean entropy across sequence
                "h_std": h_std,  # Entropy standard deviation
                "h_max": h_max,  # Maximum entropy
            }

    def extract_activation_sketch(
        self,
        activations: dict[int, torch.Tensor],
        layer_subset: list[int] | None = None,
    ) -> dict[str, float]:
        """
        Extract activation summary statistics for dispatch.

        Args:
            activations: Dictionary mapping layer indices to activation tensors
            layer_subset: Optional subset of layers to analyze

        Returns:
            Dictionary of activation-based features
        """
        if not activations:
            return {}

        sketch = {}
        layers_to_analyze = layer_subset or list(activations.keys())

        for layer_id in layers_to_analyze:
            if layer_id not in activations:
                continue

            acts = activations[layer_id]
            if acts.numel() == 0:
                continue

            with torch.no_grad():
                # Flatten to [batch * seq, features] for analysis
                acts_flat = acts.view(-1, acts.size(-1))

                # Basic statistics
                mean_norm = torch.norm(acts_flat, p=2, dim=1).mean().item()
                activation_sparsity = (acts_flat == 0).float().mean().item()

                # Activation distribution
                acts_std = acts_flat.std(dim=0).mean().item()
                acts_kurtosis = self._compute_kurtosis(acts_flat).item()

                sketch[f"layer_{layer_id}"] = {
                    "mean_norm": mean_norm,
                    "sparsity": activation_sparsity,
                    "std": acts_std,
                    "kurtosis": acts_kurtosis,
                }

        return sketch

    def _compute_kurtosis(self, x: torch.Tensor) -> torch.Tensor:
        """Compute kurtosis of tensor (measure of tail heaviness)."""
        if x.numel() < 2:
            return torch.tensor(0.0)

        # Standardize
        x_mean = x.mean(dim=0)
        x_std = x.std(dim=0) + 1e-8
        x_standardized = (x - x_mean) / x_std

        # Fourth moment
        fourth_moment = (x_standardized**4).mean(dim=0)

        # Excess kurtosis (normal distribution has kurtosis = 3)
        kurtosis = fourth_moment - 3

        return kurtosis.mean()  # Average across features

    def extract_all_features(
        self,
        prompt: str,
        logits: torch.Tensor | None = None,
        activations: dict[int, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        """
        Extract all feature types in a single call.

        Args:
            prompt: Input prompt text
            logits: Optional model output logits
            activations: Optional internal activations

        Returns:
            Combined feature dictionary
        """
        features = {
            "prompt_stats": self.extract_prompt_stats(prompt),
        }

        if logits is not None:
            features["logits_entropy"] = self.extract_logits_entropy(logits)

        if activations is not None:
            features["activation_sketch"] = self.extract_activation_sketch(activations)

        return features

    def get_feature_description(self) -> dict[str, str]:
        """Get human-readable descriptions of all features."""
        return {
            "prompt_stats": "Static analysis of input prompt (size, code ratio, complexity)",
            "logits_entropy": "Entropy of model output distribution (uncertainty measure)",
            "activation_sketch": "Summary statistics of internal layer activations",
        }
