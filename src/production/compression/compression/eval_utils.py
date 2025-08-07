from dataclasses import dataclass
import json
import os
import tempfile
import time

import torch


@dataclass
class EvaluationResult:
    """Results from compression evaluation."""

    accuracy: float
    original_size_mb: float
    compressed_size_mb: float
    compression_ratio: float
    inference_time_ms: float
    memory_usage_mb: float
    perplexity: float | None = None


class CompressionEvaluator:
    """Evaluation harness for compression pipeline."""

    def __init__(self, model_path: str, tokenizer_path: str | None = None) -> None:
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_hellaswag_sample(self, sample_path: str) -> list[dict]:
        """Load HellaSwag evaluation sample."""
        if not os.path.exists(sample_path):
            return self._generate_hellaswag_sample(sample_path)

        with open(sample_path) as f:
            return [json.loads(line) for line in f]

    def _generate_hellaswag_sample(self, sample_path: str) -> list[dict]:
        """Generate sample HellaSwag data for testing."""
        sample_data = [
            {
                "ctx": "A person is trying to cut a piece of paper with scissors. The paper is thick and the scissors are dull. The person should",
                "endings": [
                    "throw the scissors away and get new ones",
                    "use a knife instead of scissors",
                    "sharpen the scissors or use different ones",
                    "cut the paper into smaller pieces first",
                ],
                "label": 2,
            },
            {
                "ctx": "Someone is cooking pasta. The water is boiling and they add the pasta. To check if it's done, they should",
                "endings": [
                    "taste a piece to see if it's al dente",
                    "wait exactly 10 minutes",
                    "check the color of the water",
                    "smell the pasta cooking",
                ],
                "label": 0,
            },
            {
                "ctx": "A student is studying for an exam. They have limited time and many topics to cover. The best strategy would be to",
                "endings": [
                    "study all topics equally",
                    "focus on the most important topics first",
                    "skip the difficult topics",
                    "memorize everything word for word",
                ],
                "label": 1,
            },
        ]

        os.makedirs(os.path.dirname(sample_path), exist_ok=True)
        with open(sample_path, "w") as f:
            for item in sample_data:
                f.write(json.dumps(item) + "\n")

        return sample_data

    def evaluate_model_accuracy(self, model, tokenizer, eval_data: list[dict]) -> float:
        """Evaluate model accuracy on HellaSwag-style data."""
        model.eval()
        correct = 0
        total = len(eval_data)

        with torch.no_grad():
            for item in eval_data:
                ctx = item["ctx"]
                endings = item["endings"]
                correct_idx = item["label"]

                # Calculate likelihood for each ending
                likelihoods = []
                for ending in endings:
                    full_text = ctx + " " + ending
                    inputs = tokenizer(
                        full_text, return_tensors="pt", truncation=True, max_length=512
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    outputs = model(**inputs)
                    logits = outputs.logits

                    # Calculate likelihood of the ending tokens
                    ctx_len = len(tokenizer(ctx, return_tensors="pt")["input_ids"][0])
                    ending_logits = logits[0, ctx_len - 1 : -1, :]
                    ending_tokens = inputs["input_ids"][0, ctx_len:]

                    if len(ending_tokens) > 0:
                        log_probs = torch.log_softmax(ending_logits, dim=-1)
                        likelihood = (
                            torch.gather(log_probs, 1, ending_tokens.unsqueeze(1))
                            .sum()
                            .item()
                        )
                        likelihoods.append(likelihood)
                    else:
                        likelihoods.append(float("-inf"))

                # Check if highest likelihood matches correct answer
                if likelihoods.index(max(likelihoods)) == correct_idx:
                    correct += 1

        return correct / total

    def measure_model_size(self, model) -> float:
        """Measure model size in MB."""
        with tempfile.NamedTemporaryFile() as tmp:
            torch.save(model.state_dict(), tmp.name)
            size_bytes = os.path.getsize(tmp.name)
            return size_bytes / (1024 * 1024)

    def measure_inference_time(self, model, tokenizer, num_samples: int = 10) -> float:
        """Measure average inference time in milliseconds."""
        model.eval()
        test_input = "The quick brown fox jumps over the lazy dog."

        times = []
        with torch.no_grad():
            for _ in range(num_samples):
                inputs = tokenizer(test_input, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                start_time = time.time()
                model(**inputs)
                end_time = time.time()

                times.append((end_time - start_time) * 1000)

        return sum(times) / len(times)

    def measure_memory_usage(self, model) -> float:
        """Measure model memory usage in MB."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Move model to GPU and run inference
            model = model.to(self.device)
            test_input = torch.randint(0, 1000, (1, 50)).to(self.device)

            with torch.no_grad():
                model(test_input)

            memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            return memory_mb
        # Rough estimate for CPU
        param_count = sum(p.numel() for p in model.parameters())
        return param_count * 4 / (1024 * 1024)  # 4 bytes per float32

    def evaluate_compressed_model(
        self, original_model, compressed_model, tokenizer, eval_data: list[dict]
    ) -> EvaluationResult:
        """Comprehensive evaluation of compressed model."""
        # Accuracy evaluation
        accuracy = self.evaluate_model_accuracy(compressed_model, tokenizer, eval_data)

        # Size measurements
        original_size = self.measure_model_size(original_model)
        compressed_size = self.measure_model_size(compressed_model)
        compression_ratio = (
            original_size / compressed_size if compressed_size > 0 else 0
        )

        # Performance measurements
        inference_time = self.measure_inference_time(compressed_model, tokenizer)
        memory_usage = self.measure_memory_usage(compressed_model)

        return EvaluationResult(
            accuracy=accuracy,
            original_size_mb=original_size,
            compressed_size_mb=compressed_size,
            compression_ratio=compression_ratio,
            inference_time_ms=inference_time,
            memory_usage_mb=memory_usage,
        )

    def print_evaluation_report(self, result: EvaluationResult) -> None:
        """Print formatted evaluation report."""
        print("\n" + "=" * 50)
        print("COMPRESSION EVALUATION REPORT")
        print("=" * 50)
        print(f"Accuracy: {result.accuracy:.3f}")
        print(f"Original size: {result.original_size_mb:.2f} MB")
        print(f"Compressed size: {result.compressed_size_mb:.2f} MB")
        print(f"Compression ratio: {result.compression_ratio:.1f}x")
        print(f"Inference time: {result.inference_time_ms:.2f} ms")
        print(f"Memory usage: {result.memory_usage_mb:.2f} MB")
        print("=" * 50)

    def check_constraints(
        self,
        result: EvaluationResult,
        max_accuracy_drop: float = 0.05,
        min_compression_ratio: float = 10.0,
    ) -> bool:
        """Check if compression meets constraints."""
        # Note: Would need baseline accuracy for proper accuracy drop calculation
        # For now, assuming we want accuracy > (1 - max_accuracy_drop)
        accuracy_ok = result.accuracy > (1.0 - max_accuracy_drop)
        compression_ok = result.compression_ratio >= min_compression_ratio

        print("\nConstraint Check:")
        print(
            f"Accuracy constraint (>{1.0 - max_accuracy_drop:.3f}): {'PASS' if accuracy_ok else 'FAIL'}"
        )
        print(
            f"Compression constraint (>={min_compression_ratio}x): {'PASS' if compression_ok else 'FAIL'}"
        )

        return accuracy_ok and compression_ok
