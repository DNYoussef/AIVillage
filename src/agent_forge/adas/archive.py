"""
ADAS Archive system for storing and ranking expert configurations.
Manages the evolutionary search history and results leaderboard.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ExpertSpec(BaseModel):
    """YAML schema for expert configuration specification."""

    layers: list[str] = Field(..., description="Target layers: attn_qkv, mlp, block_N")
    rank: int = Field(..., ge=1, le=64, description="SVD rank for low-rank approximation")
    svd_scope: str = Field(..., description="Scope: per-matrix or per-block")
    init: str = Field(..., description="Initialization: random, pca_activations, fisher")
    activation_rule: str = Field(..., description="Activation: always or gated")
    budget: dict[str, int] = Field(..., description="Resource constraints")

    class Config:
        schema_extra = {
            "example": {
                "layers": ["attn_qkv", "mlp", "block_12"],
                "rank": 4,
                "svd_scope": "per-matrix",
                "init": "pca_activations",
                "activation_rule": "gated",
                "budget": {"max_active": 3, "max_latency_ms": 50},
            }
        }


class DispatchSpec(BaseModel):
    """YAML schema for dispatch policy specification."""

    features: list[str] = Field(..., description="Input features for dispatch")
    mix_fn: str = Field(..., description="Mixing function: linear, softmax, energy")
    granularity: str = Field(..., description="Granularity: sequence, segment, token")

    class Config:
        schema_extra = {
            "example": {
                "features": ["prompt_stats", "logits_entropy", "activation_sketch"],
                "mix_fn": "softmax",
                "granularity": "sequence",
            }
        }


@dataclass
class ExperimentResult:
    """Results from evaluating an expert configuration."""

    # Configuration
    expert_spec: dict[str, Any]
    dispatch_spec: dict[str, Any]

    # Performance metrics
    score: float = 0.0
    latency_ms: float = 0.0
    vram_gb: float = 0.0

    # Evaluation metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    trial_id: str = ""
    task_suite: str = ""
    num_samples: int = 0

    # Error tracking
    success: bool = True
    error_msg: str = ""

    # Additional metrics
    metrics: dict[str, float] = field(default_factory=dict)


class ADASArchive:
    """
    Archives experimental results and provides ranking/analysis functionality.
    Stores results in JSONL format for easy streaming and analysis.
    """

    def __init__(self, archive_path: Path):
        self.archive_path = Path(archive_path)
        self.archive_path.parent.mkdir(parents=True, exist_ok=True)

        # In-memory cache for fast access
        self._results_cache: list[ExperimentResult] = []
        self._load_archive()

    def _load_archive(self):
        """Load existing archive into memory."""
        if not self.archive_path.exists():
            logger.info(f"Creating new archive at {self.archive_path}")
            return

        try:
            with open(self.archive_path) as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            data = json.loads(line)
                            result = ExperimentResult(**data)
                            self._results_cache.append(result)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping malformed line {line_num}: {e}")
                        except TypeError as e:
                            logger.warning(f"Skipping invalid result {line_num}: {e}")

            logger.info(f"Loaded {len(self._results_cache)} results from archive")

        except Exception as e:
            logger.error(f"Failed to load archive: {e}")
            self._results_cache = []

    def add_result(self, result: ExperimentResult) -> None:
        """Add a new experimental result to the archive."""
        # Add to cache
        self._results_cache.append(result)

        # Append to file
        try:
            with open(self.archive_path, "a") as f:
                json.dump(asdict(result), f)
                f.write("\n")

            logger.debug(f"Added result to archive: score={result.score:.4f}, " f"latency={result.latency_ms:.1f}ms")

        except Exception as e:
            logger.error(f"Failed to write result to archive: {e}")

    def get_leaderboard(
        self, top_k: int = 10, sort_by: str = "score", filter_successful: bool = True
    ) -> list[ExperimentResult]:
        """Get top-k results sorted by specified metric."""
        results = self._results_cache.copy()

        if filter_successful:
            results = [r for r in results if r.success]

        if not results:
            return []

        # Sort by metric (higher is better for score, lower for latency/vram)
        reverse = sort_by == "score"
        try:
            results.sort(key=lambda x: getattr(x, sort_by), reverse=reverse)
        except AttributeError:
            logger.warning(f"Unknown sort metric: {sort_by}, using score")
            results.sort(key=lambda x: x.score, reverse=True)

        return results[:top_k]

    def get_pareto_frontier(self, metrics: list[str] = ["score", "latency_ms"]) -> list[ExperimentResult]:
        """Get Pareto-optimal results across multiple objectives."""
        if len(metrics) != 2:
            raise ValueError("Pareto frontier requires exactly 2 metrics")

        results = [r for r in self._results_cache if r.success]
        if not results:
            return []

        # For score: higher is better, for latency/vram: lower is better
        def dominates(a: ExperimentResult, b: ExperimentResult) -> bool:
            """Check if result a dominates result b."""
            better_in_any = False

            for metric in metrics:
                a_val = getattr(a, metric)
                b_val = getattr(b, metric)

                # Determine if higher or lower is better
                if metric == "score":
                    if a_val < b_val:
                        return False
                    elif a_val > b_val:
                        better_in_any = True
                else:  # latency_ms, vram_gb - lower is better
                    if a_val > b_val:
                        return False
                    elif a_val < b_val:
                        better_in_any = True

            return better_in_any

        # Find Pareto frontier
        pareto_results = []
        for candidate in results:
            is_dominated = False
            for other in results:
                if other != candidate and dominates(other, candidate):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_results.append(candidate)

        return pareto_results

    def get_statistics(self) -> dict[str, Any]:
        """Get archive statistics."""
        if not self._results_cache:
            return {"total_results": 0}

        successful = [r for r in self._results_cache if r.success]

        stats = {
            "total_results": len(self._results_cache),
            "successful_results": len(successful),
            "success_rate": len(successful) / len(self._results_cache),
        }

        if successful:
            scores = [r.score for r in successful]
            latencies = [r.latency_ms for r in successful]
            vram_usage = [r.vram_gb for r in successful]

            stats.update(
                {
                    "score_stats": {
                        "mean": sum(scores) / len(scores),
                        "min": min(scores),
                        "max": max(scores),
                    },
                    "latency_stats": {
                        "mean": sum(latencies) / len(latencies),
                        "min": min(latencies),
                        "max": max(latencies),
                    },
                    "vram_stats": {
                        "mean": sum(vram_usage) / len(vram_usage),
                        "min": min(vram_usage),
                        "max": max(vram_usage),
                    },
                }
            )

        return stats

    def export_yaml_configs(self, output_dir: Path, top_k: int = 5) -> list[Path]:
        """Export top-k configurations as YAML files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        leaderboard = self.get_leaderboard(top_k=top_k)
        exported_files = []

        for i, result in enumerate(leaderboard):
            # Create combined config
            config = {
                "expert_spec": result.expert_spec,
                "dispatch_spec": result.dispatch_spec,
                "metadata": {
                    "score": result.score,
                    "latency_ms": result.latency_ms,
                    "vram_gb": result.vram_gb,
                    "rank": i + 1,
                    "timestamp": result.timestamp,
                },
            }

            # Export to YAML
            filename = f"rank_{i + 1:02d}_score_{result.score:.4f}.yaml"
            filepath = output_dir / filename

            with open(filepath, "w") as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)

            exported_files.append(filepath)
            logger.info(f"Exported config rank {i + 1} to {filepath}")

        return exported_files

    def search_similar(
        self, target_spec: dict[str, Any], metric: str = "score", top_k: int = 5
    ) -> list[tuple[ExperimentResult, float]]:
        """Find results with similar configurations."""

        # Simple similarity based on shared configuration keys
        def compute_similarity(spec_a: dict[str, Any], spec_b: dict[str, Any]) -> float:
            """Compute configuration similarity (0-1)."""
            if not spec_a or not spec_b:
                return 0.0

            shared_keys = set(spec_a.keys()) & set(spec_b.keys())
            if not shared_keys:
                return 0.0

            matches = sum(1 for k in shared_keys if spec_a[k] == spec_b[k])
            return matches / len(shared_keys)

        results_with_sim = []
        for result in self._results_cache:
            if not result.success:
                continue

            sim = compute_similarity(target_spec, result.expert_spec)
            if sim > 0.1:  # Minimum similarity threshold
                results_with_sim.append((result, sim))

        # Sort by similarity first, then by metric
        results_with_sim.sort(key=lambda x: (x[1], getattr(x[0], metric)), reverse=True)

        return results_with_sim[:top_k]


def validate_expert_spec(spec_dict: dict[str, Any]) -> ExpertSpec:
    """Validate and parse expert specification."""
    try:
        return ExpertSpec(**spec_dict)
    except Exception as e:
        raise ValueError(f"Invalid expert spec: {e}")


def validate_dispatch_spec(spec_dict: dict[str, Any]) -> DispatchSpec:
    """Validate and parse dispatch specification."""
    try:
        return DispatchSpec(**spec_dict)
    except Exception as e:
        raise ValueError(f"Invalid dispatch spec: {e}")


# Schema exports for external use
EXPERT_SPEC_SCHEMA = ExpertSpec.schema()
DISPATCH_SPEC_SCHEMA = DispatchSpec.schema()


def get_example_configs() -> list[dict[str, Any]]:
    """Get example configurations for testing and initialization."""
    return [
        {
            "expert": {
                "layers": ["attn_qkv"],
                "rank": 2,
                "svd_scope": "per-matrix",
                "init": "pca_activations",
                "activation_rule": "gated",
                "budget": {"max_active": 2, "max_latency_ms": 30},
            },
            "dispatch": {
                "features": ["prompt_stats", "activation_sketch"],
                "mix_fn": "softmax",
                "granularity": "sequence",
            },
            "motivation": "Lightweight attention specialization with PCA init",
        },
        {
            "expert": {
                "layers": ["mlp", "block_12"],
                "rank": 4,
                "svd_scope": "per-block",
                "init": "fisher",
                "activation_rule": "always",
                "budget": {"max_active": 3, "max_latency_ms": 50},
            },
            "dispatch": {
                "features": ["logits_entropy", "activation_sketch"],
                "mix_fn": "energy",
                "granularity": "token",
            },
            "motivation": "Deep MLP specialization with Fisher information",
        },
    ]
