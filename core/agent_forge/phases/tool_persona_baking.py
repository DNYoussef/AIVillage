"""
Phase 5: Tool & Persona Baking with Grokfast
Comprehensive tool integration and persona optimization phase that:
- Bakes tool usage patterns into model weights using Grokfast acceleration
- Optimizes persona responses through iterative training
- Integrates specialized capabilities through prompt baking
- Tests and validates tool-calling behavior
- Creates agent-specific personality patterns

Consolidates implementations from:
- packages/agent_forge/legacy_src/tool_baking.py (basic structure)
- packages/agent_forge/legacy_src/prompt_baking.py (A/B testing and weight baking)
- packages/agent_forge/phases/quietstar.py (iterative baking techniques)
"""

import ast
from collections import defaultdict
from dataclasses import dataclass, field
import json
import logging
import operator
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:  # Optional DSPy integration
    from src.coordination.dspy_integration import DSPyAgentOptimizer
except Exception:  # pragma: no cover - DSPy optional
    DSPyAgentOptimizer = None  # type: ignore[misc]

# Try to import PhaseController, with fallback for direct imports
try:
    from ..core.phase_controller import PhaseController, PhaseResult
except (ImportError, ValueError):
    # Fallback for direct imports - create minimal base classes
    from abc import ABC, abstractmethod
    from dataclasses import dataclass
    from datetime import datetime
    from typing import Any

    import torch.nn as nn

    @dataclass
    class PhaseResult:
        success: bool
        model: nn.Module
        phase_name: str = None
        metrics: dict = None
        duration_seconds: float = 0.0
        artifacts: dict = None
        config: dict = None
        error: str = None
        start_time: datetime = None
        end_time: datetime = None

        def __post_init__(self):
            if self.end_time is None:
                self.end_time = datetime.now()
            if self.start_time is None:
                self.start_time = self.end_time

    class PhaseController(ABC):
        def __init__(self, config: Any):
            self.config = config

        @abstractmethod
        async def run(self, model: nn.Module) -> PhaseResult:
            pass


logger = logging.getLogger(__name__)

ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


@dataclass
class PhaseConfig:
    """Base configuration class for Agent Forge phases."""

    pass


@dataclass
class ToolPersonaBakingConfig(PhaseConfig):
    """Configuration for Tool & Persona Baking phase."""

    # Model paths
    model_path: str = ""
    output_path: str = ""
    tokenizer_path: str | None = None

    # Tool integration configuration
    available_tools: list[str] = field(
        default_factory=lambda: [
            "calculator",
            "search",
            "code_executor",
            "file_manager",
            "web_scraper",
            "data_analyzer",
            "text_processor",
        ]
    )

    # Tool usage patterns to bake
    tool_patterns: dict[str, list[str]] = field(
        default_factory=lambda: {
            "calculator": [
                "Let me calculate that: {expression}",
                "I'll compute this for you: {expression}",
                "Using the calculator: {expression}",
                "Mathematical calculation: {expression}",
            ],
            "search": [
                "I'll search for information about: {query}",
                "Let me look up: {query}",
                "Searching for: {query}",
                "Finding information on: {query}",
            ],
            "code_executor": [
                "I'll run this code: {code}",
                "Let me execute: {code}",
                "Running the code: {code}",
                "Code execution: {code}",
            ],
        }
    )

    # Persona configuration
    agent_personas: list[str] = field(
        default_factory=lambda: [
            "helpful_assistant",
            "expert_researcher",
            "creative_writer",
            "technical_analyst",
            "problem_solver",
            "teacher",
        ]
    )

    persona_templates: dict[str, str] = field(
        default_factory=lambda: {
            "helpful_assistant": "I'm here to help you with {task}. Let me provide a clear and useful response.",
            "expert_researcher": "As a researcher, I'll analyze {task} systematically and provide evidence-based insights.",
            "creative_writer": "Let me approach {task} with creativity and imagination.",
            "technical_analyst": "I'll provide a technical analysis of {task} with detailed explanations.",
            "problem_solver": "Let me break down {task} into manageable steps and solve it methodically.",
            "teacher": "I'll explain {task} in a way that's easy to understand and learn from.",
        }
    )

    # Baking configuration with Grokfast
    enable_grokfast: bool = True
    grokfast_ema_alpha: float = 0.98
    grokfast_lambda_init: float = 0.05
    grokfast_lambda_max: float = 0.25

    baking_iterations: int = 5
    convergence_threshold: float = 0.90
    baking_learning_rate: float = 1e-5
    baking_epochs_per_iteration: int = 3
    baking_strength: float = 0.15

    # Target layers for baking (early layers for tool patterns, later for personas)
    tool_baking_layers: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
    persona_baking_layers: list[int] = field(default_factory=lambda: [8, 9, 10, 11])

    # Evaluation configuration
    evaluation_tasks: list[str] = field(
        default_factory=lambda: [
            "Calculate the area of a circle with radius 5",
            "Search for information about quantum computing",
            "Write a Python function to sort a list",
            "Explain machine learning to a beginner",
            "Solve this logic puzzle: If all birds can fly...",
            "Debug this code: def factorial(n): return n * factorial(n+1)",
        ]
    )

    # A/B testing configuration
    ab_test_samples: int = 50
    ab_test_rounds: int = 3
    min_improvement_threshold: float = 0.05

    # Training configuration
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_sequence_length: int = 512
    warmup_ratio: float = 0.1

    # System configuration
    device: str = "auto"
    mixed_precision: bool = True
    seed: int = 42

    # DSPy optimization
    enable_dspy_optimization: bool = False
    dspy_optimized_prompt: str | None = None

    # Logging and monitoring
    log_interval: int = 20
    save_intermediate_models: bool = True
    wandb_project: str | None = "agent_forge"

    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


class GrokfastAdamW(torch.optim.Optimizer):
    """
    AdamW optimizer enhanced with Grokfast gradient filtering.
    Amplifies slow-changing gradients to accelerate tool/persona learning.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        ema_alpha: float = 0.98,
        grokfast_lambda: float = 0.05,
        grokfast_enabled: bool = True,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            ema_alpha=ema_alpha,
            grokfast_lambda=grokfast_lambda,
            grokfast_enabled=grokfast_enabled,
        )
        super().__init__(params, defaults)

        # Initialize Grokfast EMA buffers
        self._init_grokfast_buffers()

    def _init_grokfast_buffers(self):
        """Initialize EMA gradient buffers for Grokfast."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    state = self.state[p]
                    state["ema_grad"] = torch.zeros_like(p.data)
                    state["grokfast_initialized"] = False

    @torch.no_grad()
    def step(self, closure=None, grokfast_lambda_override: float | None = None):
        """Perform optimization step with Grokfast filtering."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            ema_alpha = group["ema_alpha"]
            grokfast_lambda = grokfast_lambda_override or group["grokfast_lambda"]
            grokfast_enabled = group["grokfast_enabled"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # Apply Grokfast filtering if enabled
                if grokfast_enabled and grokfast_lambda > 0:
                    grad = self._apply_grokfast_filter(grad, p, ema_alpha, grokfast_lambda)

                # Standard AdamW update
                state = self.state[p]

                # State initialization
                if len(state) == 0 or "step" not in state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1

                # Update moments
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Compute step size
                step_size = group["lr"] / bias_correction1
                denom = (exp_avg_sq.sqrt() / bias_correction2**0.5).add_(group["eps"])

                # Update parameters
                p.data.add_(exp_avg / denom, alpha=-step_size)

                # Weight decay (AdamW style)
                if group["weight_decay"] > 0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

        return loss

    def _apply_grokfast_filter(
        self,
        grad: torch.Tensor,
        param: torch.nn.Parameter,
        ema_alpha: float,
        grokfast_lambda: float,
    ) -> torch.Tensor:
        """Apply Grokfast gradient filtering."""
        state = self.state[param]

        # Update EMA of gradients
        if not state.get("grokfast_initialized", False):
            state["ema_grad"] = grad.clone()
            state["grokfast_initialized"] = True
        else:
            state["ema_grad"].mul_(ema_alpha).add_(grad, alpha=1 - ema_alpha)

        ema_grad = state["ema_grad"]

        # Avoid division by zero
        ema_norm_sq = torch.sum(ema_grad * ema_grad)
        if ema_norm_sq > 1e-10:
            # Project gradient onto EMA direction
            projection_scalar = torch.sum(grad * ema_grad) / ema_norm_sq
            # Amplify component in EMA direction
            filtered_grad = grad + grokfast_lambda * projection_scalar * ema_grad
        else:
            filtered_grad = grad

        return filtered_grad


class ToolIntegrationSystem:
    """System for integrating and testing tool usage patterns."""

    def __init__(self, config: ToolPersonaBakingConfig):
        self.config = config
        self.tool_implementations = self._create_tool_implementations()
        self.tool_usage_stats = defaultdict(lambda: {"calls": 0, "successes": 0})

    def _create_tool_implementations(self) -> dict[str, Any]:
        """Create mock tool implementations for testing."""
        return {
            "calculator": self._calculator_tool,
            "search": self._search_tool,
            "code_executor": self._code_executor_tool,
            "file_manager": self._file_manager_tool,
            "web_scraper": self._web_scraper_tool,
            "data_analyzer": self._data_analyzer_tool,
            "text_processor": self._text_processor_tool,
        }

    def _calculator_tool(self, expression: str) -> dict[str, Any]:
        """Mock calculator tool."""
        try:
            # Simple mathematical expressions
            if "circle" in expression.lower() and "radius" in expression.lower():
                # Extract radius if possible
                if "5" in expression:
                    result = f"Area = π × 5² = {3.14159 * 25:.2f} square units"
                else:
                    result = "Area = π × r² where r is the radius"
            elif any(op in expression for op in ["+", "-", "*", "/", "^"]):
                # Try to evaluate simple expressions safely
                try:
                    result = str(self._safe_eval(expression))
                except Exception:
                    result = f"Mathematical expression: {expression}"
            else:
                result = f"Calculator result for: {expression}"

            return {"success": True, "result": result, "tool": "calculator"}
        except Exception as e:
            return {"success": False, "error": str(e), "tool": "calculator"}

    def _safe_eval(self, expression: str) -> float:
        """Safely evaluate a mathematical expression."""
        if not re.fullmatch(r"[0-9+\-*/().\s^]*", expression):
            raise ValueError("Invalid characters in expression")
        expression = expression.replace("^", "**")
        node = ast.parse(expression, mode="eval")

        def _eval(node: ast.AST) -> float:
            if isinstance(node, ast.Expression):
                return _eval(node.body)
            if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_OPERATORS:
                return ALLOWED_OPERATORS[type(node.op)](_eval(node.left), _eval(node.right))
            if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_OPERATORS:
                return ALLOWED_OPERATORS[type(node.op)](_eval(node.operand))
            if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
                return node.value
            raise ValueError("Unsupported expression")

        return _eval(node)

    def _search_tool(self, query: str) -> dict[str, Any]:
        """Mock search tool."""
        # Simulate search results based on query
        search_results = {
            "quantum computing": "Quantum computing uses quantum mechanics to process information...",
            "machine learning": "Machine learning is a subset of AI that enables computers to learn...",
            "default": f"Search results for '{query}' would appear here...",
        }

        result = search_results.get(query.lower(), search_results["default"])
        return {"success": True, "result": result, "tool": "search"}

    def _code_executor_tool(self, code: str) -> dict[str, Any]:
        """Mock code executor tool."""
        try:
            if "def " in code and "sort" in code.lower():
                result = """def sort_list(lst):
    return sorted(lst)

# Example usage:
# sorted_list = sort_list([3, 1, 4, 1, 5])
# Output: [1, 1, 3, 4, 5]"""
            elif "factorial" in code.lower():
                result = """def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

# Fixed: Added base case to prevent infinite recursion"""
            else:
                result = f"Code execution result for: {code[:50]}..."

            return {"success": True, "result": result, "tool": "code_executor"}
        except Exception as e:
            return {"success": False, "error": str(e), "tool": "code_executor"}

    def _file_manager_tool(self, operation: str) -> dict[str, Any]:
        """Mock file manager tool."""
        return {"success": True, "result": f"File operation: {operation}", "tool": "file_manager"}

    def _web_scraper_tool(self, url: str) -> dict[str, Any]:
        """Mock web scraper tool."""
        return {"success": True, "result": f"Web content from: {url}", "tool": "web_scraper"}

    def _data_analyzer_tool(self, data: str) -> dict[str, Any]:
        """Mock data analyzer tool."""
        return {"success": True, "result": f"Data analysis: {data}", "tool": "data_analyzer"}

    def _text_processor_tool(self, text: str) -> dict[str, Any]:
        """Mock text processor tool."""
        return {"success": True, "result": f"Processed text: {text}", "tool": "text_processor"}

    def call_tool(self, tool_name: str, *args, **kwargs) -> dict[str, Any]:
        """Call a tool and track usage statistics."""
        self.tool_usage_stats[tool_name]["calls"] += 1

        if tool_name in self.tool_implementations:
            try:
                result = self.tool_implementations[tool_name](*args, **kwargs)
                if result.get("success", False):
                    self.tool_usage_stats[tool_name]["successes"] += 1
                return result
            except Exception as e:
                return {"success": False, "error": str(e), "tool": tool_name}
        else:
            return {"success": False, "error": f"Tool '{tool_name}' not available", "tool": tool_name}

    def get_tool_usage_stats(self) -> dict[str, dict[str, Any]]:
        """Get tool usage statistics."""
        stats = {}
        for tool_name, tool_stats in self.tool_usage_stats.items():
            calls = tool_stats["calls"]
            successes = tool_stats["successes"]
            success_rate = successes / calls if calls > 0 else 0.0
            stats[tool_name] = {"calls": calls, "successes": successes, "success_rate": success_rate}
        return stats


class PersonaOptimizer:
    """System for optimizing persona-based responses."""

    def __init__(self, config: ToolPersonaBakingConfig):
        self.config = config
        self.persona_performance = defaultdict(lambda: {"scores": [], "avg_score": 0.0})

    def generate_persona_examples(self, persona: str, task: str) -> list[str]:
        """Generate training examples for a specific persona."""
        template = self.config.persona_templates.get(persona, "I'll help with {task}.")
        base_prompt = template.format(task=task)

        if self.config.enable_dspy_optimization and DSPyAgentOptimizer is not None:
            try:
                optimizer = DSPyAgentOptimizer()
                optimized = optimizer.get_optimized_prompt("tool_persona_baking")
                if optimized:
                    base_prompt = optimized.format(task=task)
                    self.config.dspy_optimized_prompt = optimized
            except Exception as opt_err:  # pragma: no cover - best effort
                logger.warning(f"DSPy optimization skipped: {opt_err}")

        # Generate variations
        examples = [base_prompt]

        # Add persona-specific variations
        if persona == "helpful_assistant":
            examples.extend(
                [
                    f"I'm happy to help you with {task}.",
                    f"Let me assist you with {task}.",
                    f"I'd be glad to help with {task}.",
                ]
            )
        elif persona == "expert_researcher":
            examples.extend(
                [
                    f"Based on research, I can help with {task}.",
                    f"Let me provide an evidence-based approach to {task}.",
                    f"From an expert perspective, here's how to handle {task}.",
                ]
            )
        elif persona == "creative_writer":
            examples.extend(
                [
                    f"Let me creatively explore {task}.",
                    f"Here's an imaginative approach to {task}.",
                    f"I'll bring creativity to {task}.",
                ]
            )
        elif persona == "technical_analyst":
            examples.extend(
                [
                    f"Let me provide a technical analysis of {task}.",
                    f"From a technical standpoint, {task} involves...",
                    f"Here's the technical breakdown of {task}.",
                ]
            )
        elif persona == "problem_solver":
            examples.extend(
                [
                    f"Let me solve {task} step by step.",
                    f"I'll break down {task} systematically.",
                    f"Here's my problem-solving approach to {task}.",
                ]
            )
        elif persona == "teacher":
            examples.extend(
                [
                    f"Let me teach you about {task}.",
                    f"I'll explain {task} clearly for you.",
                    f"Here's how to understand {task}.",
                ]
            )

        return examples

    def evaluate_persona_response(self, persona: str, task: str, response: str) -> float:
        """Evaluate how well a response matches the expected persona."""
        # Simplified persona evaluation
        score = 0.5  # Base score

        response_lower = response.lower()

        if persona == "helpful_assistant":
            if any(word in response_lower for word in ["help", "assist", "glad", "happy"]):
                score += 0.3
            if "let me" in response_lower:
                score += 0.2

        elif persona == "expert_researcher":
            if any(word in response_lower for word in ["research", "evidence", "analysis", "based on"]):
                score += 0.3
            if any(word in response_lower for word in ["expert", "professional", "systematic"]):
                score += 0.2

        elif persona == "creative_writer":
            if any(word in response_lower for word in ["creative", "imaginative", "story", "artistic"]):
                score += 0.3
            if len(response.split()) > 20:  # More elaborate responses
                score += 0.2

        elif persona == "technical_analyst":
            if any(word in response_lower for word in ["technical", "analysis", "system", "process"]):
                score += 0.3
            if any(word in response_lower for word in ["algorithm", "method", "implementation"]):
                score += 0.2

        elif persona == "problem_solver":
            if any(word in response_lower for word in ["step", "solve", "solution", "approach"]):
                score += 0.3
            if any(word in response_lower for word in ["first", "then", "next", "finally"]):
                score += 0.2

        elif persona == "teacher":
            if any(word in response_lower for word in ["explain", "teach", "understand", "learn"]):
                score += 0.3
            if any(word in response_lower for word in ["simple", "easy", "clear", "example"]):
                score += 0.2

        # General quality indicators
        if len(response) > 50:  # Substantial response
            score += 0.1

        return min(score, 1.0)

    def update_persona_performance(self, persona: str, score: float):
        """Update performance tracking for a persona."""
        self.persona_performance[persona]["scores"].append(score)
        scores = self.persona_performance[persona]["scores"]
        self.persona_performance[persona]["avg_score"] = np.mean(scores)

    def get_best_persona_for_task(self, task: str) -> str:
        """Determine the best persona for a given task."""
        # Simple task-persona matching
        task_lower = task.lower()

        if any(word in task_lower for word in ["calculate", "math", "number", "formula"]):
            return "technical_analyst"
        elif any(word in task_lower for word in ["search", "find", "research", "information"]):
            return "expert_researcher"
        elif any(word in task_lower for word in ["write", "create", "story", "poem"]):
            return "creative_writer"
        elif any(word in task_lower for word in ["solve", "problem", "puzzle", "logic"]):
            return "problem_solver"
        elif any(word in task_lower for word in ["explain", "teach", "learn", "understand"]):
            return "teacher"
        else:
            return "helpful_assistant"


class ToolPersonaBakingDataset(Dataset):
    """Dataset for tool and persona baking training."""

    def __init__(
        self,
        config: ToolPersonaBakingConfig,
        tokenizer,
        tool_system: ToolIntegrationSystem,
        persona_optimizer: PersonaOptimizer,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.tool_system = tool_system
        self.persona_optimizer = persona_optimizer
        self.examples = []

        self._generate_examples()

    def _generate_examples(self):
        """Generate training examples for tools and personas."""
        # Generate tool usage examples
        for tool_name in self.config.available_tools:
            if tool_name in self.config.tool_patterns:
                patterns = self.config.tool_patterns[tool_name]
                for pattern in patterns:
                    # Generate example usage
                    if tool_name == "calculator":
                        example = pattern.format(expression="2 + 2 * 3")
                    elif tool_name == "search":
                        example = pattern.format(query="machine learning")
                    elif tool_name == "code_executor":
                        example = pattern.format(code="print('Hello World')")
                    else:
                        example = pattern.format(**{list(pattern.split("{"))[1].split("}")[0]: "example"})

                    self.examples.append({"text": example, "type": "tool", "tool_name": tool_name, "pattern": pattern})

        # Generate persona examples
        sample_tasks = [
            "Explain quantum physics",
            "Write a short story",
            "Solve this math problem",
            "Debug this code",
            "Research renewable energy",
            "Create a lesson plan",
        ]

        for persona in self.config.agent_personas:
            for task in sample_tasks:
                examples = self.persona_optimizer.generate_persona_examples(persona, task)
                for example in examples:
                    self.examples.append({"text": example, "type": "persona", "persona": persona, "task": task})

        logger.info(f"Generated {len(self.examples)} training examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Tokenize the text
        encoding = self.tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=self.config.max_sequence_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze(),
            "example_type": example["type"],
            "metadata": {k: v for k, v in example.items() if k != "text"},
        }


class ABTestHarness:
    """A/B testing system for tool and persona optimization."""

    def __init__(self, config: ToolPersonaBakingConfig):
        self.config = config
        self.test_results = []

    async def run_ab_test(
        self,
        model_before,
        model_after,
        tokenizer,
        tool_system: ToolIntegrationSystem,
        persona_optimizer: PersonaOptimizer,
    ) -> dict[str, Any]:
        """Run A/B test comparing models before and after baking."""
        logger.info("Running A/B test on tool and persona performance")

        results = {
            "tool_performance": {},
            "persona_performance": {},
            "overall_improvement": 0.0,
            "significant_improvement": False,
        }

        # Test tool performance
        tool_scores_before = []
        tool_scores_after = []

        for task in self.config.evaluation_tasks:
            if self._is_tool_task(task):
                score_before = await self._evaluate_tool_response(model_before, tokenizer, task, tool_system)
                score_after = await self._evaluate_tool_response(model_after, tokenizer, task, tool_system)

                tool_scores_before.append(score_before)
                tool_scores_after.append(score_after)

        # Test persona performance
        persona_scores_before = []
        persona_scores_after = []

        for task in self.config.evaluation_tasks:
            if not self._is_tool_task(task):
                score_before = await self._evaluate_persona_response(model_before, tokenizer, task, persona_optimizer)
                score_after = await self._evaluate_persona_response(model_after, tokenizer, task, persona_optimizer)

                persona_scores_before.append(score_before)
                persona_scores_after.append(score_after)

        # Calculate improvements
        tool_improvement = np.mean(tool_scores_after) - np.mean(tool_scores_before) if tool_scores_after else 0.0
        persona_improvement = (
            np.mean(persona_scores_after) - np.mean(persona_scores_before) if persona_scores_after else 0.0
        )
        overall_improvement = (tool_improvement + persona_improvement) / 2

        results["tool_performance"] = {
            "before": np.mean(tool_scores_before) if tool_scores_before else 0.0,
            "after": np.mean(tool_scores_after) if tool_scores_after else 0.0,
            "improvement": tool_improvement,
        }

        results["persona_performance"] = {
            "before": np.mean(persona_scores_before) if persona_scores_before else 0.0,
            "after": np.mean(persona_scores_after) if persona_scores_after else 0.0,
            "improvement": persona_improvement,
        }

        results["overall_improvement"] = overall_improvement
        results["significant_improvement"] = overall_improvement > self.config.min_improvement_threshold

        logger.info(
            f"A/B test results - Tool improvement: {tool_improvement:.3f}, Persona improvement: {persona_improvement:.3f}"
        )

        return results

    def _is_tool_task(self, task: str) -> bool:
        """Determine if a task requires tool usage."""
        task_lower = task.lower()
        return any(word in task_lower for word in ["calculate", "search", "code", "execute", "run", "compute"])

    async def _evaluate_tool_response(self, model, tokenizer, task: str, tool_system: ToolIntegrationSystem) -> float:
        """Evaluate how well a model handles tool-requiring tasks."""
        # Generate response
        inputs = tokenizer(task, return_tensors="pt").to(self.config.device)

        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        # Score tool usage appropriateness
        score = 0.5  # Base score
        response_lower = response.lower()

        if "calculate" in task.lower():
            if any(word in response_lower for word in ["calculate", "compute", "math"]):
                score += 0.3
            if any(char in response for char in "0123456789+*/="):
                score += 0.2

        elif "search" in task.lower():
            if any(word in response_lower for word in ["search", "find", "look"]):
                score += 0.3
            if "information" in response_lower:
                score += 0.2

        elif "code" in task.lower():
            if any(word in response_lower for word in ["code", "function", "def"]):
                score += 0.3
            if any(char in response for char in "(){}"):
                score += 0.2

        return min(score, 1.0)

    async def _evaluate_persona_response(
        self, model, tokenizer, task: str, persona_optimizer: PersonaOptimizer
    ) -> float:
        """Evaluate how well a model handles persona-based tasks."""
        # Determine best persona for task
        best_persona = persona_optimizer.get_best_persona_for_task(task)

        # Generate response
        inputs = tokenizer(task, return_tensors="pt").to(self.config.device)

        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        # Evaluate persona appropriateness
        return persona_optimizer.evaluate_persona_response(best_persona, task, response)


class ToolPersonaBakingPhase(PhaseController):
    """
    Phase 5: Tool & Persona Baking with Grokfast
    Comprehensive tool integration and persona optimization.
    """

    def __init__(self, config: ToolPersonaBakingConfig):
        super().__init__(config)
        self.config = config

        # Set random seeds for reproducibility
        torch.manual_seed(getattr(config, 'seed', 42))
        np.random.seed(getattr(config, 'seed', 42))

        # Initialize systems
        self.tool_system = ToolIntegrationSystem(config)
        self.persona_optimizer = PersonaOptimizer(config)
        self.ab_tester = ABTestHarness(config)

        # Training state
        self.current_iteration = 0
        self.convergence_scores = []

    async def run(self, model: nn.Module) -> PhaseResult:
        """
        Execute the Tool & Persona Baking phase processing.

        Args:
            model: Input model from previous phase

        Returns:
            PhaseResult with processed model and metrics
        """
        # Validate input model
        if not self.validate_input_model(model):
            return self.create_failure_result(model, "Input model validation failed")

        start_time = time.time()

        try:
            # Save model temporarily to pass to execute_phase method
            temp_model_path = Path(self.config.output_path) / "temp_input_model"
            temp_model_path.mkdir(parents=True, exist_ok=True)

            model.save_pretrained(str(temp_model_path))

            # Execute the phase using existing execute_phase method
            inputs = {"model_path": str(temp_model_path)}
            result = await self.execute_phase(inputs)

            duration = time.time() - start_time

            if result.success:
                return self.create_success_result(
                    model=result.model,
                    metrics=result.metrics or {},
                    artifacts=result.artifacts or {},
                    duration=duration
                )
            else:
                return self.create_failure_result(model, result.error or "Tool & Persona Baking failed", duration)

        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Tool & Persona Baking phase failed: {e}")
            return self.create_failure_result(model, str(e), duration)

    async def validate_inputs(self, inputs: dict[str, Any]) -> bool:
        """Validate phase inputs."""
        if "model_path" not in inputs:
            logger.error("Missing model_path in inputs")
            return False

        model_path = Path(inputs["model_path"])
        if not model_path.exists():
            logger.error(f"Model path does not exist: {model_path}")
            return False

        return True

    async def execute_phase(self, inputs: dict[str, Any]) -> PhaseResult:
        """Execute the tool and persona baking phase."""
        try:
            logger.info("Starting Phase 5: Tool & Persona Baking with Grokfast")

            # Load model and tokenizer
            model_path = inputs["model_path"]
            self.config.model_path = model_path

            model, tokenizer = await self._load_model(model_path)

            # Create baseline copy for A/B testing
            baseline_model = await self._create_model_copy(model)

            # Main baking loop
            baked_model = await self._run_baking_iterations(model, tokenizer)

            # Final A/B test
            ab_results = await self.ab_tester.run_ab_test(
                baseline_model, baked_model, tokenizer, self.tool_system, self.persona_optimizer
            )

            # Save baked model
            output_path = Path(self.config.output_path)
            output_path.mkdir(parents=True, exist_ok=True)

            baked_model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)

            # Compile results
            success = (
                ab_results["overall_improvement"] > self.config.min_improvement_threshold
                and ab_results["significant_improvement"]
            )

            outputs = {
                "baked_model_path": str(output_path),
                "ab_test_results": ab_results,
                "tool_usage_stats": self.tool_system.get_tool_usage_stats(),
                "persona_performance": dict(self.persona_optimizer.persona_performance),
                "baking_iterations": self.current_iteration,
                "convergence_scores": self.convergence_scores,
            }

            metrics = {
                "overall_improvement": ab_results["overall_improvement"],
                "tool_improvement": ab_results["tool_performance"]["improvement"],
                "persona_improvement": ab_results["persona_performance"]["improvement"],
                "baking_iterations_completed": self.current_iteration,
                "convergence_achieved": len(self.convergence_scores) > 0
                and self.convergence_scores[-1] > self.config.convergence_threshold,
                "final_convergence_score": self.convergence_scores[-1] if self.convergence_scores else 0.0,
            }

            return PhaseResult(success=success, phase_name="tool_persona_baking", outputs=outputs, metrics=metrics)

        except Exception as e:
            logger.error(f"Tool & Persona Baking phase failed: {e}")
            return PhaseResult(success=False, phase_name="tool_persona_baking", error=str(e), outputs={})

    async def _load_model(self, model_path: str) -> tuple[nn.Module, AutoTokenizer]:
        """Load model and tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path or model_path)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
            device_map=self.config.device,
        )

        return model, tokenizer

    async def _create_model_copy(self, model: nn.Module) -> nn.Module:
        """Create a copy of the model for baseline comparison."""
        # Simple deep copy for baseline
        import copy

        return copy.deepcopy(model)

    async def _run_baking_iterations(self, model: nn.Module, tokenizer) -> nn.Module:
        """Run iterative baking process."""
        logger.info(f"Starting {self.config.baking_iterations} baking iterations")

        # Create dataset
        dataset = ToolPersonaBakingDataset(self.config, tokenizer, self.tool_system, self.persona_optimizer)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=2)

        for iteration in range(self.config.baking_iterations):
            self.current_iteration = iteration + 1
            logger.info(f"Baking iteration {self.current_iteration}/{self.config.baking_iterations}")

            # Bake tool patterns
            model = await self._bake_tool_patterns(model, tokenizer, dataloader)

            # Bake persona patterns
            model = await self._bake_persona_patterns(model, tokenizer, dataloader)

            # Test convergence
            convergence_score = await self._test_convergence(model, tokenizer)
            self.convergence_scores.append(convergence_score)

            logger.info(f"Iteration {self.current_iteration} convergence: {convergence_score:.3f}")

            # Save intermediate model if requested
            if self.config.save_intermediate_models:
                await self._save_intermediate_model(model, tokenizer, iteration)

            # Check for early convergence
            if convergence_score >= self.config.convergence_threshold:
                logger.info(f"Convergence achieved at iteration {self.current_iteration}")
                break

        return model

    async def _bake_tool_patterns(self, model: nn.Module, tokenizer, dataloader) -> nn.Module:
        """Bake tool usage patterns into specific layers."""
        logger.info("Baking tool patterns")

        # Get parameters for tool baking layers
        baking_params = []
        for layer_idx in self.config.tool_baking_layers:
            try:
                if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
                    if layer_idx < len(model.transformer.h):
                        layer = model.transformer.h[layer_idx]
                        baking_params.extend(layer.parameters())
                elif hasattr(model, "model") and hasattr(model.model, "layers"):
                    if layer_idx < len(model.model.layers):
                        layer = model.model.layers[layer_idx]
                        baking_params.extend(layer.parameters())
            except (IndexError, AttributeError):
                logger.warning(f"Could not access layer {layer_idx} for tool baking")

        if not baking_params:
            logger.warning("No parameters found for tool baking, using all parameters")
            baking_params = list(model.parameters())

        # Create Grokfast optimizer
        optimizer = GrokfastAdamW(
            baking_params,
            lr=self.config.baking_learning_rate,
            grokfast_enabled=self.config.enable_grokfast,
            ema_alpha=self.config.grokfast_ema_alpha,
            grokfast_lambda=self.config.grokfast_lambda_init,
        )

        # Training loop
        model.train()
        total_loss = 0.0

        for epoch in range(self.config.baking_epochs_per_iteration):
            epoch_loss = 0.0
            num_batches = 0

            for batch in tqdm(dataloader, desc=f"Tool baking epoch {epoch+1}"):
                # Only train on tool examples
                if batch["example_type"][0] != "tool":
                    continue

                # Move to device
                input_ids = batch["input_ids"].to(self.config.device)
                attention_mask = batch["attention_mask"].to(self.config.device)
                labels = batch["labels"].to(self.config.device)

                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss * self.config.baking_strength

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            if num_batches > 0:
                avg_epoch_loss = epoch_loss / num_batches
                total_loss += avg_epoch_loss
                logger.info(f"Tool baking epoch {epoch+1} loss: {avg_epoch_loss:.4f}")

        model.eval()
        return model

    async def _bake_persona_patterns(self, model: nn.Module, tokenizer, dataloader) -> nn.Module:
        """Bake persona patterns into specific layers."""
        logger.info("Baking persona patterns")

        # Get parameters for persona baking layers
        baking_params = []
        for layer_idx in self.config.persona_baking_layers:
            try:
                if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
                    if layer_idx < len(model.transformer.h):
                        layer = model.transformer.h[layer_idx]
                        baking_params.extend(layer.parameters())
                elif hasattr(model, "model") and hasattr(model.model, "layers"):
                    if layer_idx < len(model.model.layers):
                        layer = model.model.layers[layer_idx]
                        baking_params.extend(layer.parameters())
            except (IndexError, AttributeError):
                logger.warning(f"Could not access layer {layer_idx} for persona baking")

        if not baking_params:
            logger.warning("No parameters found for persona baking, using all parameters")
            baking_params = list(model.parameters())

        # Create Grokfast optimizer
        optimizer = GrokfastAdamW(
            baking_params,
            lr=self.config.baking_learning_rate,
            grokfast_enabled=self.config.enable_grokfast,
            ema_alpha=self.config.grokfast_ema_alpha,
            grokfast_lambda=self.config.grokfast_lambda_max,  # Use max lambda for persona baking
        )

        # Training loop
        model.train()
        total_loss = 0.0

        for epoch in range(self.config.baking_epochs_per_iteration):
            epoch_loss = 0.0
            num_batches = 0

            for batch in tqdm(dataloader, desc=f"Persona baking epoch {epoch+1}"):
                # Only train on persona examples
                if batch["example_type"][0] != "persona":
                    continue

                # Move to device
                input_ids = batch["input_ids"].to(self.config.device)
                attention_mask = batch["attention_mask"].to(self.config.device)
                labels = batch["labels"].to(self.config.device)

                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss * self.config.baking_strength

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            if num_batches > 0:
                avg_epoch_loss = epoch_loss / num_batches
                total_loss += avg_epoch_loss
                logger.info(f"Persona baking epoch {epoch+1} loss: {avg_epoch_loss:.4f}")

        model.eval()
        return model

    async def _test_convergence(self, model: nn.Module, tokenizer) -> float:
        """Test if patterns have converged (stuck) in the model."""
        model.eval()

        # Test a sample of patterns
        test_prompts = [
            "Calculate 15 * 23",
            "Search for quantum computing information",
            "Write a Python function",
            "Explain this concept clearly",
            "Help me solve this problem",
            "Analyze this data systematically",
        ]

        correct_responses = 0
        total_responses = len(test_prompts)

        with torch.no_grad():
            for prompt in test_prompts:
                inputs = tokenizer(prompt, return_tensors="pt").to(self.config.device)

                outputs = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=50,
                    temperature=0.3,  # Lower temperature for consistency testing
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

                response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

                # Simple convergence check - look for appropriate patterns
                if self._response_shows_convergence(prompt, response):
                    correct_responses += 1

        convergence_score = correct_responses / total_responses
        return convergence_score

    def _response_shows_convergence(self, prompt: str, response: str) -> bool:
        """Check if response shows converged pattern usage."""
        prompt_lower = prompt.lower()
        response_lower = response.lower()

        # Tool pattern convergence
        if "calculate" in prompt_lower:
            return any(word in response_lower for word in ["calculate", "compute", "math", "="])
        elif "search" in prompt_lower:
            return any(word in response_lower for word in ["search", "find", "information"])
        elif "function" in prompt_lower:
            return any(word in response_lower for word in ["def", "function", "code"])

        # Persona pattern convergence
        elif "explain" in prompt_lower:
            return any(word in response_lower for word in ["explain", "understand", "clear"])
        elif "help" in prompt_lower:
            return any(word in response_lower for word in ["help", "assist", "glad"])
        elif "analyze" in prompt_lower:
            return any(word in response_lower for word in ["analysis", "systematic", "data"])

        return False

    async def _save_intermediate_model(self, model: nn.Module, tokenizer, iteration: int):
        """Save intermediate model checkpoint."""
        intermediate_path = Path(self.config.output_path).parent / f"intermediate_iter_{iteration}"
        intermediate_path.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(intermediate_path)
        tokenizer.save_pretrained(intermediate_path)

        # Save iteration metrics
        metrics = {
            "iteration": iteration,
            "convergence_scores": self.convergence_scores,
            "tool_usage_stats": self.tool_system.get_tool_usage_stats(),
        }

        with open(intermediate_path / "iteration_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        logger.info(f"Saved intermediate model at iteration {iteration}")


async def create_tool_persona_baking_phase(
    model_path: str,
    output_path: str,
    available_tools: list[str] | None = None,
    agent_personas: list[str] | None = None,
    enable_grokfast: bool = True,
    **kwargs,
) -> PhaseResult:
    """
    Factory function to create and execute Tool & Persona Baking phase.

    Args:
        model_path: Path to input model from training phase
        output_path: Path for baked model output
        available_tools: List of tools to integrate
        agent_personas: List of personas to optimize
        enable_grokfast: Enable Grokfast acceleration
        **kwargs: Additional configuration options

    Returns:
        PhaseResult with baking results
    """
    config = ToolPersonaBakingConfig(
        model_path=model_path, output_path=output_path, enable_grokfast=enable_grokfast, **kwargs
    )

    if available_tools:
        config.available_tools = available_tools
    if agent_personas:
        config.agent_personas = agent_personas

    phase = ToolPersonaBakingPhase(config)

    inputs = {"model_path": model_path}
    return await phase.execute_phase(inputs)


# Alias for backward compatibility
PersonaOptimizationSystem = PersonaOptimizer
