"""
Temperature curriculum system for alternating self-modeling training.
Implements temperature bin scheduling, snippet datasets, and teacher consistency.
"""

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import torch.utils.data as data


class TempRound(Enum):
    """Temperature curriculum round types."""

    NONOVERLAP = "nonoverlap"  # Non-overlapping bins: [0-0.1], [0.2-0.3], etc.
    OVERLAP = "overlap"  # Overlapping bins: [0-0.2], [0.1-0.3], etc.


class TempBinType(Enum):
    """Temperature bin classification types."""

    LOW = "low"  # τ ∈ [0.0, 0.2)
    MID = "mid"  # τ ∈ [0.2, 0.8)
    HIGH = "high"  # τ ∈ [0.8, 1.5]
    OVERLAP_LOW = "overlap_low"  # τ ∈ [0.0, 0.3)
    OVERLAP_MID = "overlap_mid"  # τ ∈ [0.2, 0.9)
    OVERLAP_HIGH = "overlap_high"  # τ ∈ [0.7, 1.5]


class GrokStage(Enum):
    """Grokking stages for classification."""

    PRE = "pre"  # Before grokking onset
    ONSET = "onset"  # During grok onset (ID↓, S_slow↑)
    CONSOLIDATE = "consolidate"  # After grokking


@dataclass
class TempBin:
    """Temperature bin definition."""

    low: float
    high: float
    center: float
    bin_type: TempBinType

    def __post_init__(self):
        if self.center is None:
            self.center = (self.low + self.high) / 2

    def contains(self, temperature: float) -> bool:
        """Check if temperature falls within this bin."""
        return self.low <= temperature < self.high

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "low": self.low,
            "high": self.high,
            "center": self.center,
            "bin_type": self.bin_type.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TempBin":
        """Create from dictionary."""
        return cls(
            low=data["low"],
            high=data["high"],
            center=data["center"],
            bin_type=TempBinType(data["bin_type"]),
        )


@dataclass
class GeneratedSnippet:
    """A generated code/reasoning snippet with metadata."""

    id: str
    tau_bin: TempBin
    domain: str
    topic: str
    text: str
    rubric: str
    unit_tests: list[str] = field(default_factory=list)
    telemetry: dict[str, Any] | None = None
    stage_label: GrokStage | None = None
    temp_bin_label: TempBinType | None = None
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "tau_bin": self.tau_bin.to_dict(),
            "domain": self.domain,
            "topic": self.topic,
            "text": self.text,
            "rubric": self.rubric,
            "unit_tests": self.unit_tests,
            "telemetry": self.telemetry,
            "stage_label": self.stage_label.value if self.stage_label else None,
            "temp_bin_label": self.temp_bin_label.value if self.temp_bin_label else None,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GeneratedSnippet":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            tau_bin=TempBin.from_dict(data["tau_bin"]),
            domain=data["domain"],
            topic=data["topic"],
            text=data["text"],
            rubric=data["rubric"],
            unit_tests=data.get("unit_tests", []),
            telemetry=data.get("telemetry"),
            stage_label=GrokStage(data["stage_label"]) if data.get("stage_label") else None,
            temp_bin_label=TempBinType(data["temp_bin_label"]) if data.get("temp_bin_label") else None,
            confidence=data.get("confidence", 0.0),
        )


class TempBinScheduler:
    """
    Schedules temperature bins for curriculum learning.
    Supports both non-overlapping and overlapping rounds.
    """

    def __init__(
        self,
        round_type: TempRound = TempRound.NONOVERLAP,
        temp_range: tuple[float, float] = (0.0, 1.5),
        bin_width: float = 0.1,
        overlap_ratio: float = 0.5,
    ):
        self.round_type = round_type
        self.temp_range = temp_range
        self.bin_width = bin_width
        self.overlap_ratio = overlap_ratio

        self.bins = self._generate_bins()
        self.mastered_bins: list[str] = []
        self.current_round = 0

    def _generate_bins(self) -> list[TempBin]:
        """Generate temperature bins based on round type."""
        bins = []
        temp_min, temp_max = self.temp_range

        if self.round_type == TempRound.NONOVERLAP:
            # Non-overlapping bins: [0-0.1], [0.2-0.3], etc.
            current = temp_min
            while current < temp_max:
                high = min(current + self.bin_width, temp_max)
                center = (current + high) / 2

                # Classify bin type based on center temperature
                if center < 0.2:
                    bin_type = TempBinType.LOW
                elif center < 0.8:
                    bin_type = TempBinType.MID
                else:
                    bin_type = TempBinType.HIGH

                bins.append(TempBin(low=current, high=high, center=center, bin_type=bin_type))

                # Skip gap for non-overlapping
                current = high + self.bin_width

        else:  # OVERLAP
            # Overlapping bins: [0-0.2], [0.1-0.3], etc.
            overlap_width = self.bin_width * 2
            step = self.bin_width * (1 - self.overlap_ratio)

            current = temp_min
            while current < temp_max:
                high = min(current + overlap_width, temp_max)
                center = (current + high) / 2

                # Classify overlapping bin type
                if center < 0.3:
                    bin_type = TempBinType.OVERLAP_LOW
                elif center < 0.9:
                    bin_type = TempBinType.OVERLAP_MID
                else:
                    bin_type = TempBinType.OVERLAP_HIGH

                bins.append(TempBin(low=current, high=high, center=center, bin_type=bin_type))

                current += step

        return bins

    def get_bins(self) -> list[TempBin]:
        """Get current temperature bins."""
        return self.bins.copy()

    def get_alternate_bin(self, current_bin: TempBin) -> TempBin:
        """Get an alternate bin for training (different from current)."""
        available_bins = [b for b in self.bins if b != current_bin]
        if not available_bins:
            # Fallback: create a complementary bin
            if current_bin.center < 0.5:
                return TempBin(0.8, 1.2, 1.0, TempBinType.HIGH)
            else:
                return TempBin(0.0, 0.2, 0.1, TempBinType.LOW)

        return random.choice(available_bins)

    def mark_mastered(self, bin_id: str):
        """Mark a bin as mastered."""
        if bin_id not in self.mastered_bins:
            self.mastered_bins.append(bin_id)

    def get_mastery_stats(self) -> dict[str, int]:
        """Get mastery statistics."""
        total_bins = len(self.bins)
        mastered = len(self.mastered_bins)
        return {
            "total_bins": total_bins,
            "mastered": mastered,
            "learning": max(0, total_bins - mastered),
            "mastery_rate": mastered / total_bins if total_bins > 0 else 0,
        }

    def advance_round(self):
        """Advance to next curriculum round."""
        self.current_round += 1
        if self.round_type == TempRound.NONOVERLAP:
            # Switch to overlapping
            self.round_type = TempRound.OVERLAP
            self.bins = self._generate_bins()

        # Reset mastery tracking
        self.mastered_bins.clear()


class SnippetDataset(data.Dataset):
    """
    Dataset wrapper for generated code/reasoning snippets.
    Supports filtering by domain, temperature bins, and stages.
    """

    def __init__(
        self,
        snippets: list[GeneratedSnippet],
        tokenizer=None,
        max_length: int = 512,
        include_telemetry: bool = True,
    ):
        self.snippets = snippets
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_telemetry = include_telemetry

        # Build indices for efficient filtering
        self._build_indices()

    def _build_indices(self):
        """Build indices for efficient filtering."""
        self.domain_index = defaultdict(list)
        self.bin_index = defaultdict(list)
        self.stage_index = defaultdict(list)

        for idx, snippet in enumerate(self.snippets):
            self.domain_index[snippet.domain].append(idx)
            if snippet.temp_bin_label:
                self.bin_index[snippet.temp_bin_label].append(idx)
            if snippet.stage_label:
                self.stage_index[snippet.stage_label].append(idx)

    def __len__(self) -> int:
        return len(self.snippets)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        snippet = self.snippets[idx]

        # Tokenize text if tokenizer provided
        if self.tokenizer:
            encoding = self.tokenizer(
                snippet.text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)
        else:
            # Dummy tensors for testing
            input_ids = torch.zeros(self.max_length, dtype=torch.long)
            attention_mask = torch.ones(self.max_length, dtype=torch.long)

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "snippet_id": snippet.id,
            "text": snippet.text,
            "tau_bin": snippet.tau_bin.to_dict(),
            "domain": snippet.domain,
            "topic": snippet.topic,
            "confidence": snippet.confidence,
        }

        # Add classification labels
        if snippet.temp_bin_label:
            # Convert temp bin to class index
            temp_classes = list(TempBinType)
            item["temp_label"] = temp_classes.index(snippet.temp_bin_label)
        else:
            item["temp_label"] = -1

        if snippet.stage_label:
            # Convert stage to class index
            stage_classes = list(GrokStage)
            item["stage_label"] = stage_classes.index(snippet.stage_label)
        else:
            item["stage_label"] = -1

        # Add telemetry if available
        if self.include_telemetry and snippet.telemetry:
            item["telemetry"] = snippet.telemetry

        return item

    def filter_by_domain(self, domain: str) -> "SnippetDataset":
        """Create filtered dataset by domain."""
        indices = self.domain_index.get(domain, [])
        filtered_snippets = [self.snippets[i] for i in indices]
        return SnippetDataset(filtered_snippets, self.tokenizer, self.max_length, self.include_telemetry)

    def filter_by_bin_type(self, bin_type: TempBinType) -> "SnippetDataset":
        """Create filtered dataset by temperature bin type."""
        indices = self.bin_index.get(bin_type, [])
        filtered_snippets = [self.snippets[i] for i in indices]
        return SnippetDataset(filtered_snippets, self.tokenizer, self.max_length, self.include_telemetry)

    def filter_by_stage(self, stage: GrokStage) -> "SnippetDataset":
        """Create filtered dataset by grokking stage."""
        indices = self.stage_index.get(stage, [])
        filtered_snippets = [self.snippets[i] for i in indices]
        return SnippetDataset(filtered_snippets, self.tokenizer, self.max_length, self.include_telemetry)

    def get_domain_distribution(self) -> dict[str, int]:
        """Get distribution of samples by domain."""
        return {domain: len(indices) for domain, indices in self.domain_index.items()}

    def get_bin_distribution(self) -> dict[str, int]:
        """Get distribution of samples by temperature bin."""
        return {bin_type.value: len(indices) for bin_type, indices in self.bin_index.items()}

    def get_stage_distribution(self) -> dict[str, int]:
        """Get distribution of samples by grokking stage."""
        return {stage.value: len(indices) for stage, indices in self.stage_index.items()}


@dataclass
class TeacherReference:
    """Reference distribution from teacher model for consistency training."""

    tau_center: float
    canonical_answer: str
    ngram_stats: list[dict[str, Any]]
    logit_distribution: torch.Tensor | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "tau_center": self.tau_center,
            "canonical_answer": self.canonical_answer,
            "ngram_stats": self.ngram_stats,
            "has_logits": self.logit_distribution is not None,
        }


class TeacherConsistency:
    """
    Builds teacher reference distributions for KL consistency training.
    Provides canonical answers at specific temperature centers.
    """

    def __init__(self, model=None, tokenizer=None, max_length: int = 256, cache_size: int = 1000):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_size = cache_size

        # Reference cache
        self.reference_cache: dict[str, TeacherReference] = {}

    def get_cache_key(self, prompt: str, tau_center: float, domain: str) -> str:
        """Generate cache key for reference."""
        content = f"{prompt}|{tau_center:.3f}|{domain}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def build_reference(self, prompt: str, tau_center: float, domain: str, use_cache: bool = True) -> TeacherReference:
        """Build teacher reference distribution at tau_center."""

        cache_key = self.get_cache_key(prompt, tau_center, domain)

        # Check cache first
        if use_cache and cache_key in self.reference_cache:
            return self.reference_cache[cache_key]

        # Generate reference using model if available
        if self.model and self.tokenizer:
            reference = self._generate_model_reference(prompt, tau_center, domain)
        else:
            # Fallback to heuristic reference
            reference = self._generate_heuristic_reference(prompt, tau_center, domain)

        # Cache reference
        if len(self.reference_cache) < self.cache_size:
            self.reference_cache[cache_key] = reference

        return reference

    def _generate_model_reference(self, prompt: str, tau_center: float, domain: str) -> TeacherReference:
        """Generate reference using actual model."""

        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length)

        # Generate with specific temperature
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                temperature=tau_center,
                max_new_tokens=100,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Decode generated text
        generated_ids = outputs.sequences[0][inputs.input_ids.shape[1] :]
        canonical_answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Compute n-gram statistics
        tokens = canonical_answer.split()
        ngram_stats = []

        for n in [2, 3]:
            if len(tokens) >= n:
                ngrams = [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
                ngram_counts = {}
                for ngram in ngrams:
                    ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

                # Top n-grams
                top_ngrams = sorted(ngram_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                ngram_stats.append(
                    {
                        "n": n,
                        "top": [{"ngram": ng, "count": ct} for ng, ct in top_ngrams],
                    }
                )

        # Extract logit distribution from last token
        logit_distribution = None
        if outputs.scores:
            logit_distribution = torch.softmax(outputs.scores[-1][0], dim=-1)

        return TeacherReference(
            tau_center=tau_center,
            canonical_answer=canonical_answer,
            ngram_stats=ngram_stats,
            logit_distribution=logit_distribution,
        )

    def _generate_heuristic_reference(self, prompt: str, tau_center: float, domain: str) -> TeacherReference:
        """Generate heuristic reference when model unavailable."""

        # Domain-specific canonical answers
        if domain == "coding-python":
            if "list" in prompt.lower():
                canonical_answer = "[x for x in range(10) if x % 2 == 0]"
            elif "function" in prompt.lower():
                canonical_answer = "def solve(x): return x * 2"
            else:
                canonical_answer = "# Python solution here"
        elif domain == "math":
            canonical_answer = "The answer is 42."
        else:
            canonical_answer = f"Solution for temperature {tau_center:.2f}"

        # Simple n-gram stats
        tokens = canonical_answer.split()
        ngram_stats = [
            {
                "n": 2,
                "top": [{"ngram": " ".join(tokens[i : i + 2]), "count": 1} for i in range(min(3, len(tokens) - 1))],
            }
        ]

        return TeacherReference(
            tau_center=tau_center,
            canonical_answer=canonical_answer,
            ngram_stats=ngram_stats,
        )

    def compute_kl_target(self, student_logits: torch.Tensor, reference: TeacherReference) -> torch.Tensor:
        """Compute KL divergence target for consistency training."""

        if reference.logit_distribution is not None:
            # Use actual teacher logits
            teacher_probs = reference.logit_distribution
        else:
            # Use uniform distribution as fallback
            teacher_probs = torch.ones_like(student_logits)
            teacher_probs = torch.softmax(teacher_probs, dim=-1)

        # Ensure compatible shapes
        if teacher_probs.shape != student_logits.shape:
            # Truncate or pad to match
            min_size = min(teacher_probs.size(-1), student_logits.size(-1))
            teacher_probs = teacher_probs[..., :min_size]
            student_logits = student_logits[..., :min_size]

        # Compute KL divergence
        student_probs = torch.softmax(student_logits, dim=-1)
        kl_div = torch.kl_div(
            torch.log_softmax(student_logits, dim=-1),
            teacher_probs,
            reduction="batchmean",
        )

        return kl_div

    def clear_cache(self):
        """Clear reference cache."""
        self.reference_cache.clear()


# Factory functions for easy configuration


def create_nonoverlap_scheduler(
    temp_range: tuple[float, float] = (0.0, 1.5), bin_width: float = 0.1
) -> TempBinScheduler:
    """Create non-overlapping temperature bin scheduler."""
    return TempBinScheduler(round_type=TempRound.NONOVERLAP, temp_range=temp_range, bin_width=bin_width)


def create_overlap_scheduler(
    temp_range: tuple[float, float] = (0.0, 1.5),
    bin_width: float = 0.1,
    overlap_ratio: float = 0.5,
) -> TempBinScheduler:
    """Create overlapping temperature bin scheduler."""
    return TempBinScheduler(
        round_type=TempRound.OVERLAP,
        temp_range=temp_range,
        bin_width=bin_width,
        overlap_ratio=overlap_ratio,
    )


def create_snippet_dataset(
    snippets_path: str | Path,
    tokenizer=None,
    domain_filter: str | None = None,
    max_length: int = 512,
) -> SnippetDataset:
    """Create snippet dataset from stored snippets."""

    if isinstance(snippets_path, str):
        snippets_path = Path(snippets_path)

    # Load snippets from JSONL
    snippets = []
    if snippets_path.exists():
        with open(snippets_path) as f:
            for line in f:
                snippet_data = json.loads(line)
                snippet = GeneratedSnippet.from_dict(snippet_data)

                # Apply domain filter
                if domain_filter is None or snippet.domain == domain_filter:
                    snippets.append(snippet)

    return SnippetDataset(
        snippets=snippets,
        tokenizer=tokenizer,
        max_length=max_length,
        include_telemetry=True,
    )


if __name__ == "__main__":
    # Demo temperature curriculum system
    print("🌡️ Temperature Curriculum System Demo")
    print("=" * 50)

    # Test non-overlapping scheduler
    print("1. Non-overlapping Bin Scheduler:")
    scheduler = create_nonoverlap_scheduler()
    bins = scheduler.get_bins()

    print(f"   Created {len(bins)} non-overlapping bins:")
    for i, bin_def in enumerate(bins[:5]):  # Show first 5
        print(
            f"   {i + 1}. [{bin_def.low:.1f}, {bin_def.high:.1f}) center={bin_def.center:.1f} type={bin_def.bin_type.value}"
        )

    # Test alternation
    test_bin = bins[0]
    alt_bin = scheduler.get_alternate_bin(test_bin)
    print(f"   Alternate for {test_bin.center:.1f}: {alt_bin.center:.1f}")

    print()

    # Test overlapping scheduler
    print("2. Overlapping Bin Scheduler:")
    scheduler.advance_round()  # Switch to overlapping
    overlap_bins = scheduler.get_bins()

    print(f"   Created {len(overlap_bins)} overlapping bins:")
    for i, bin_def in enumerate(overlap_bins[:5]):  # Show first 5
        print(
            f"   {i + 1}. [{bin_def.low:.1f}, {bin_def.high:.1f}) center={bin_def.center:.1f} type={bin_def.bin_type.value}"
        )

    print()

    # Test snippet dataset
    print("3. Snippet Dataset:")

    # Create mock snippets
    mock_snippets = []
    domains = ["coding-python", "math", "logic"]
    topics = ["list-comprehensions", "functions", "edge-cases"]

    for i in range(10):
        snippet = GeneratedSnippet(
            id=f"snippet_{i:03d}",
            tau_bin=random.choice(bins),
            domain=random.choice(domains),
            topic=random.choice(topics),
            text=f"# Example code snippet {i}\nresult = process_data(input_{i})",
            rubric="Check for correctness and efficiency",
            unit_tests=[f"assert result_{i} == expected_{i}"],
            temp_bin_label=random.choice(list(TempBinType)),
            stage_label=random.choice(list(GrokStage)),
            confidence=random.uniform(0.5, 0.95),
        )
        mock_snippets.append(snippet)

    dataset = SnippetDataset(mock_snippets)

    print(f"   Dataset size: {len(dataset)}")
    print(f"   Domain distribution: {dataset.get_domain_distribution()}")
    print(f"   Bin distribution: {dataset.get_bin_distribution()}")
    print(f"   Stage distribution: {dataset.get_stage_distribution()}")

    # Test filtering
    coding_dataset = dataset.filter_by_domain("coding-python")
    print(f"   Coding subset size: {len(coding_dataset)}")

    print()

    # Test teacher consistency
    print("4. Teacher Consistency:")
    teacher = TeacherConsistency()

    test_prompts = [
        ("def fibonacci(n):", 0.1, "coding-python"),
        ("What is 2 + 2?", 0.5, "math"),
        ("Solve for x: 2x + 3 = 7", 0.9, "math"),
    ]

    for prompt, tau, domain in test_prompts:
        reference = teacher.build_reference(prompt, tau, domain)
        print(f"   τ={tau:.1f} {domain}: '{reference.canonical_answer[:30]}...'")
        print(f"       N-grams: {len(reference.ngram_stats)} sets")

    print()

    # Test mastery tracking
    print("5. Mastery Tracking:")
    initial_stats = scheduler.get_mastery_stats()
    print(f"   Initial: {initial_stats}")

    # Simulate mastering some bins
    for i in range(3):
        scheduler.mark_mastered(f"bin_{i}")

    final_stats = scheduler.get_mastery_stats()
    print(f"   After mastering 3: {final_stats}")

    print()
    print("✅ Temperature Curriculum System Demo Complete")
    print()
    print("Key Features Demonstrated:")
    print("  • Non-overlapping and overlapping bin scheduling")
    print("  • Temperature bin classification and alternation")
    print("  • Snippet dataset with filtering and indexing")
    print("  • Teacher reference generation for KL consistency")
    print("  • Mastery tracking and curriculum progression")
    print("  • Comprehensive data structures for training")
