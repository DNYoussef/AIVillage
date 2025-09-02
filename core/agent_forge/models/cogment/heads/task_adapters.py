"""Task-specific output adapters for different task types."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskAdapter(nn.Module, ABC):
    """Base class for task-specific output adapters."""

    def __init__(self, d_model: int, task_type: str):
        super().__init__()
        self.d_model = d_model
        self.task_type = task_type

    @abstractmethod
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Convert hidden states to task-specific outputs."""
        pass

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ClassificationAdapter(TaskAdapter):
    """
    Classification task adapter.

    Converts hidden states to class probabilities.
    Budget: <50K parameters for most configurations.
    """

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        dropout: float = 0.1,
        use_pooling: bool = True,
        pooling_strategy: str = "mean",  # "mean", "max", "cls", "last"
    ):
        super().__init__(d_model, "classification")

        self.num_classes = num_classes
        self.use_pooling = use_pooling
        self.pooling_strategy = pooling_strategy

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

        # Optional normalization
        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        """Initialize classifier weights."""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def _pool_sequence(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Pool sequence to single representation."""
        if not self.use_pooling:
            return hidden_states

        if self.pooling_strategy == "mean":
            if attention_mask is not None:
                # Masked mean pooling
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                return sum_embeddings / sum_mask
            else:
                return hidden_states.mean(dim=1)

        elif self.pooling_strategy == "max":
            if attention_mask is not None:
                hidden_states = hidden_states.masked_fill(~attention_mask.unsqueeze(-1).bool(), -1e9)
            return hidden_states.max(dim=1)[0]

        elif self.pooling_strategy == "cls":
            return hidden_states[:, 0]  # Use first token (CLS-like)

        elif self.pooling_strategy == "last":
            if attention_mask is not None:
                # Get last non-padded token
                seq_lengths = attention_mask.sum(dim=1) - 1
                return hidden_states[torch.arange(hidden_states.size(0)), seq_lengths]
            else:
                return hidden_states[:, -1]  # Use last token

        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Convert hidden states to class logits.

        Args:
            hidden_states: Hidden states [B, seq_len, d_model] or [B, d_model]
            attention_mask: Attention mask [B, seq_len] (optional)

        Returns:
            logits: Classification logits [B, num_classes]
        """
        # Handle different input shapes
        if hidden_states.dim() == 3:
            # Sequence input: pool to single representation
            pooled = self._pool_sequence(hidden_states, attention_mask)
        elif hidden_states.dim() == 2:
            # Already pooled input
            pooled = hidden_states
        else:
            raise ValueError(f"Unsupported input shape: {hidden_states.shape}")

        # Normalize and classify
        pooled = self.norm(pooled)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        return logits


class RegressionAdapter(TaskAdapter):
    """
    Regression task adapter.

    Converts hidden states to continuous values.
    Budget: <30K parameters for most configurations.
    """

    def __init__(
        self,
        d_model: int,
        output_dim: int = 1,
        activation: str = "linear",  # "linear", "sigmoid", "tanh", "relu"
        dropout: float = 0.1,
        use_pooling: bool = True,
        pooling_strategy: str = "mean",
    ):
        super().__init__(d_model, "regression")

        self.output_dim = output_dim
        self.activation = activation
        self.use_pooling = use_pooling
        self.pooling_strategy = pooling_strategy

        # Regression head
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(d_model, output_dim)

        # Normalization
        self.norm = nn.LayerNorm(d_model)

        # Activation function
        if activation == "sigmoid":
            self.output_activation = nn.Sigmoid()
        elif activation == "tanh":
            self.output_activation = nn.Tanh()
        elif activation == "relu":
            self.output_activation = nn.ReLU()
        else:
            self.output_activation = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        """Initialize regressor weights."""
        nn.init.xavier_uniform_(self.regressor.weight)
        nn.init.zeros_(self.regressor.bias)

    def _pool_sequence(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Pool sequence to single representation (shared with ClassificationAdapter)."""
        if not self.use_pooling:
            return hidden_states

        if self.pooling_strategy == "mean":
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                return sum_embeddings / sum_mask
            else:
                return hidden_states.mean(dim=1)
        elif self.pooling_strategy == "max":
            if attention_mask is not None:
                hidden_states = hidden_states.masked_fill(~attention_mask.unsqueeze(-1).bool(), -1e9)
            return hidden_states.max(dim=1)[0]
        elif self.pooling_strategy == "last":
            if attention_mask is not None:
                seq_lengths = attention_mask.sum(dim=1) - 1
                return hidden_states[torch.arange(hidden_states.size(0)), seq_lengths]
            else:
                return hidden_states[:, -1]
        else:
            return hidden_states[:, 0]  # Default to first token

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Convert hidden states to regression outputs.

        Args:
            hidden_states: Hidden states [B, seq_len, d_model] or [B, d_model]
            attention_mask: Attention mask [B, seq_len] (optional)

        Returns:
            outputs: Regression outputs [B, output_dim]
        """
        # Handle different input shapes
        if hidden_states.dim() == 3:
            pooled = self._pool_sequence(hidden_states, attention_mask)
        elif hidden_states.dim() == 2:
            pooled = hidden_states
        else:
            raise ValueError(f"Unsupported input shape: {hidden_states.shape}")

        # Normalize and regress
        pooled = self.norm(pooled)
        pooled = self.dropout(pooled)
        outputs = self.regressor(pooled)
        outputs = self.output_activation(outputs)

        return outputs


class ARCTaskAdapter(TaskAdapter):
    """
    ARC-specific task adapter for grid reasoning.

    Specialized for ARC tasks:
    - Grid reconstruction
    - Pattern completion
    - Rule application
    Budget: <100K parameters
    """

    def __init__(
        self,
        d_model: int,
        max_grid_size: int = 30,
        num_colors: int = 10,
        output_format: str = "grid",  # "grid", "sequence", "both"
        dropout: float = 0.1,
    ):
        super().__init__(d_model, "arc")

        self.max_grid_size = max_grid_size
        self.num_colors = num_colors
        self.output_format = output_format

        # Grid reconstruction head
        self.grid_projector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, max_grid_size * max_grid_size),
        )

        # Color prediction head
        self.color_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_colors),
        )

        # Optional: Pattern similarity head for rule matching
        self.pattern_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 16),  # 16 common pattern types
        )

        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        """Initialize ARC adapter weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self, hidden_states: torch.Tensor, target_grid_size: tuple[int, int] | None = None
    ) -> dict[str, torch.Tensor]:
        """
        Convert hidden states to ARC outputs.

        Args:
            hidden_states: Hidden states [B, seq_len, d_model] or [B, d_model]
            target_grid_size: Target output grid size (H, W)

        Returns:
            outputs: Dictionary with ARC predictions
        """
        # Pool to single representation if sequence input
        if hidden_states.dim() == 3:
            # Use mean pooling for grid-level reasoning
            pooled = hidden_states.mean(dim=1)  # [B, d_model]
        else:
            pooled = hidden_states

        pooled = self.norm(pooled)

        outputs = {}

        # Grid structure prediction
        if self.output_format in ["grid", "both"]:
            grid_logits = self.grid_projector(pooled)  # [B, max_grid_size^2]
            grid_logits = grid_logits.view(-1, self.max_grid_size, self.max_grid_size)

            # Resize if target size specified
            if target_grid_size is not None:
                H, W = target_grid_size
                if H != self.max_grid_size or W != self.max_grid_size:
                    grid_logits = F.interpolate(
                        grid_logits.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
                    ).squeeze(1)

            outputs["grid_structure"] = grid_logits

        # Color prediction per cell
        if self.output_format in ["sequence", "both"]:
            color_logits = self.color_classifier(pooled)  # [B, num_colors]
            outputs["color_distribution"] = color_logits

        # Pattern recognition
        pattern_logits = self.pattern_head(pooled)  # [B, 16]
        outputs["pattern_type"] = pattern_logits

        return outputs


class TextGenerationAdapter(TaskAdapter):
    """
    Text generation adapter using vocabulary heads.

    Lightweight adapter that works with optimized vocabulary heads.
    Budget: <20K parameters
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        tie_with_embeddings: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__(d_model, "text_generation")

        self.vocab_size = vocab_size
        self.tie_with_embeddings = tie_with_embeddings

        if not tie_with_embeddings:
            # Independent output projection
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
            self._init_weights()
        else:
            # Will be tied with input embeddings externally
            self.lm_head = None

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def _init_weights(self):
        """Initialize generation head."""
        if self.lm_head is not None:
            nn.init.xavier_uniform_(self.lm_head.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Convert hidden states to vocabulary logits.

        Args:
            hidden_states: Hidden states [B, seq_len, d_model]

        Returns:
            logits: Vocabulary logits [B, seq_len, vocab_size]
        """
        # Normalize
        hidden_states = self.norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Generate logits
        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            raise RuntimeError("LM head not set. Use tie_with_embeddings or set lm_head manually.")

        return logits

    def tie_weights(self, embedding_weight: torch.Tensor):
        """Tie output weights with input embeddings."""
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.lm_head.weight = embedding_weight


class MathTaskAdapter(TaskAdapter):
    """
    Mathematical reasoning task adapter.

    Specialized for math problems:
    - Numerical answer extraction
    - Step-by-step reasoning
    - Expression evaluation
    Budget: <75K parameters
    """

    def __init__(
        self,
        d_model: int,
        max_number_range: float = 1000.0,
        num_reasoning_steps: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__(d_model, "math")

        self.max_number_range = max_number_range
        self.num_reasoning_steps = num_reasoning_steps

        # Numerical answer prediction
        self.number_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Tanh(),  # Normalize to [-1, 1] then scale
        )

        # Operation type classification
        self.operation_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 8),  # +, -, *, /, ^, sqrt, log, etc.
        )

        # Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

        # Step-by-step reasoning indicator
        self.reasoning_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_reasoning_steps),
            nn.Sigmoid(),  # Which steps are needed
        )

        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        """Initialize math adapter weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Convert hidden states to math outputs.

        Args:
            hidden_states: Hidden states [B, seq_len, d_model] or [B, d_model]

        Returns:
            outputs: Dictionary with math predictions
        """
        # Pool to single representation if sequence
        if hidden_states.dim() == 3:
            pooled = hidden_states.mean(dim=1)
        else:
            pooled = hidden_states

        pooled = self.norm(pooled)

        # Numerical answer (scaled to target range)
        number_logit = self.number_head(pooled)  # [B, 1]
        numerical_answer = number_logit * self.max_number_range  # Scale to range

        # Operation classification
        operation_logits = self.operation_head(pooled)  # [B, 8]

        # Confidence score
        confidence = self.confidence_head(pooled)  # [B, 1]

        # Reasoning steps needed
        reasoning_steps = self.reasoning_head(pooled)  # [B, num_reasoning_steps]

        return {
            "numerical_answer": numerical_answer,
            "operation_type": operation_logits,
            "confidence": confidence,
            "reasoning_steps": reasoning_steps,
        }


def create_task_adapter(task_type: str, d_model: int, **kwargs) -> TaskAdapter:
    """
    Factory function for creating task adapters.

    Args:
        task_type: Type of adapter ("classification", "regression", "arc", "text_gen", "math")
        d_model: Model dimension
        **kwargs: Task-specific arguments

    Returns:
        TaskAdapter: Configured task adapter
    """
    if task_type == "classification":
        return ClassificationAdapter(d_model, **kwargs)
    elif task_type == "regression":
        return RegressionAdapter(d_model, **kwargs)
    elif task_type == "arc":
        return ARCTaskAdapter(d_model, **kwargs)
    elif task_type == "text_generation":
        return TextGenerationAdapter(d_model, **kwargs)
    elif task_type == "math":
        return MathTaskAdapter(d_model, **kwargs)
    else:
        raise ValueError(f"Unknown task type: {task_type}")


class MultiTaskAdapter(nn.Module):
    """
    Multi-task adapter that can handle multiple task types simultaneously.

    Useful for models that need to handle different tasks with shared representations.
    """

    def __init__(self, d_model: int, task_configs: dict[str, dict]):
        super().__init__()

        self.d_model = d_model
        self.task_configs = task_configs

        # Create adapters for each task
        self.adapters = nn.ModuleDict()
        for task_name, config in task_configs.items():
            task_type = config.pop("task_type")
            self.adapters[task_name] = create_task_adapter(task_type, d_model, **config)

    def forward(self, hidden_states: torch.Tensor, task_name: str, **kwargs) -> torch.Tensor:
        """Forward pass for specific task."""
        if task_name not in self.adapters:
            raise ValueError(f"Unknown task: {task_name}")

        return self.adapters[task_name](hidden_states, **kwargs)

    def get_all_outputs(self, hidden_states: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        """Get outputs for all tasks."""
        outputs = {}
        for task_name, adapter in self.adapters.items():
            try:
                outputs[task_name] = adapter(hidden_states, **kwargs)
            except Exception as e:
                outputs[task_name] = f"Error: {e}"
        return outputs

    def count_parameters(self) -> int:
        """Count total parameters across all adapters."""
        return sum(adapter.count_parameters() for adapter in self.adapters.values())


if __name__ == "__main__":
    # Test different adapters
    d_model = 320
    batch_size = 2
    seq_len = 10

    # Test input
    hidden_states = torch.randn(batch_size, seq_len, d_model)
    pooled_states = torch.randn(batch_size, d_model)

    print("=== Task Adapter Testing ===")

    # Classification adapter
    cls_adapter = ClassificationAdapter(d_model, num_classes=5)
    cls_output = cls_adapter(hidden_states)
    print(f"Classification - Input: {hidden_states.shape}, Output: {cls_output.shape}")
    print(f"Classification parameters: {cls_adapter.count_parameters():,}")

    # ARC adapter
    arc_adapter = ARCTaskAdapter(d_model)
    arc_output = arc_adapter(pooled_states)
    print(f"ARC - Input: {pooled_states.shape}")
    for key, value in arc_output.items():
        print(f"  {key}: {value.shape}")
    print(f"ARC parameters: {arc_adapter.count_parameters():,}")

    # Math adapter
    math_adapter = MathTaskAdapter(d_model)
    math_output = math_adapter(pooled_states)
    print(f"Math - Input: {pooled_states.shape}")
    for key, value in math_output.items():
        print(f"  {key}: {value.shape}")
    print(f"Math parameters: {math_adapter.count_parameters():,}")

    # Multi-task adapter
    task_configs = {
        "classification": {"task_type": "classification", "num_classes": 10},
        "regression": {"task_type": "regression", "output_dim": 3},
        "arc": {"task_type": "arc", "max_grid_size": 20},
    }

    multi_adapter = MultiTaskAdapter(d_model, task_configs)
    print(f"Multi-task parameters: {multi_adapter.count_parameters():,}")

    # Test single task
    cls_result = multi_adapter(hidden_states, "classification")
    print(f"Multi-task classification: {cls_result.shape}")

    # Test all tasks
    all_outputs = multi_adapter.get_all_outputs(pooled_states)
    print("Multi-task all outputs:")
    for task, output in all_outputs.items():
        if isinstance(output, torch.Tensor):
            print(f"  {task}: {output.shape}")
        elif isinstance(output, dict):
            print(f"  {task}: {list(output.keys())}")
        else:
            print(f"  {task}: {output}")
