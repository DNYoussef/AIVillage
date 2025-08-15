#!/usr/bin/env python3
"""Self-data generation for BitNet training using local models only.

Generates instruction-following datasets using the same base model that will be compressed.
All operations are offline-safe with no external dependencies.
"""

import argparse
import hashlib
import json
import os
import random
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class SelfDataGenerator:
    """Generate training data using the base model itself."""

    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = Path(model_path)
        self.device = (
            device
            if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Load model and tokenizer from local directory
        print(f"[SelfGen] Loading model from {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), local_files_only=True, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto" if self.device == "cuda" else None,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"[SelfGen] Model loaded on {self.device}")

    def generate_instruction_templates(self) -> list[dict[str, str]]:
        """Generate diverse instruction templates for different domains."""
        templates = [
            # Coding tasks
            {
                "domain": "coding",
                "instruction": "Write a Python function to {task}",
                "tasks": [
                    "calculate the factorial of a number",
                    "reverse a string",
                    "find the maximum element in a list",
                    "check if a number is prime",
                    "sort a list using bubble sort",
                    "implement binary search",
                    "count vowels in a string",
                    "find the GCD of two numbers",
                    "check if a string is a palindrome",
                    "calculate fibonacci sequence",
                ],
            },
            # Math tasks
            {
                "domain": "math",
                "instruction": "Solve this math problem: {task}",
                "tasks": [
                    "What is 15% of 240?",
                    "Find the area of a circle with radius 7",
                    "Solve for x: 2x + 5 = 13",
                    "What is the square root of 144?",
                    "Calculate 8! (8 factorial)",
                    "Find the slope between points (2,3) and (5,9)",
                    "Convert 3/4 to a decimal",
                    "What is 25 squared?",
                    "Find the hypotenuse of a right triangle with legs 3 and 4",
                    "Simplify: 2(x + 3) + 4x",
                ],
            },
            # Writing tasks
            {
                "domain": "writing",
                "instruction": "Write a {task}",
                "tasks": [
                    "short story about a magical forest",
                    "professional email requesting a meeting",
                    "product description for a smartphone",
                    "summary of the benefits of exercise",
                    "persuasive paragraph about renewable energy",
                    "haiku about winter",
                    "restaurant review",
                    "how-to guide for making coffee",
                    "character description for a detective",
                    "thank you note to a teacher",
                ],
            },
            # Reasoning tasks
            {
                "domain": "reasoning",
                "instruction": "Answer this question with step-by-step reasoning: {task}",
                "tasks": [
                    "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
                    "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost?",
                    "What comes next in this sequence: 2, 4, 8, 16, ?",
                    "If all roses are flowers and all flowers are plants, are all roses plants?",
                    "How many triangles are in a pentagram?",
                    "If you have 3 apples and give away 2, then buy 5 more, how many do you have?",
                    "What day of the week will it be 100 days from Tuesday?",
                    "If A is taller than B, and B is taller than C, who is the shortest?",
                    "How many seconds are in 2.5 hours?",
                    "If every student in a class is either tall or short, and 60% are tall, what percentage are short?",
                ],
            },
        ]
        return templates

    def generate_sample(self, instruction: str, max_new_tokens: int = 256) -> str:
        """Generate a response for the given instruction."""
        prompt = f"Instruction: {instruction}\nResponse:"

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        )
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the generated part
        if "Response:" in response:
            response = response.split("Response:", 1)[1].strip()

        return response

    def apply_quality_filters(self, instruction: str, response: str) -> bool:
        """Apply quality filters to instruction-response pairs."""
        # Length filters
        if len(response) < 32 or len(response) > 2048:
            return False

        # Remove AI disclaimers
        ai_patterns = [
            r"as an ai",
            r"i am an ai",
            r"i'm an ai",
            r"as a language model",
            r"i cannot",
            r"i can't",
            r"i don't have",
            r"i'm not able to",
        ]

        response_lower = response.lower()
        if any(re.search(pattern, response_lower) for pattern in ai_patterns):
            return False

        # Minimum word count
        word_count = len(response.split())
        if word_count < 8:
            return False

        # Check for reasonable instruction-response coherence
        if len(instruction.split()) > len(response.split()):
            return False

        return True

    def deduplicate_samples(self, samples: list[dict]) -> list[dict]:
        """Remove duplicate samples based on response content hash."""
        seen_hashes = set()
        unique_samples = []

        for sample in samples:
            response_hash = hashlib.md5(sample["output"].encode()).hexdigest()
            if response_hash not in seen_hashes:
                seen_hashes.add(response_hash)
                unique_samples.append(sample)

        removed = len(samples) - len(unique_samples)
        if removed > 0:
            print(f"[SelfGen] Removed {removed} duplicate samples")

        return unique_samples

    def generate_dataset(
        self, num_samples: int, max_new_tokens: int = 256, seed: int = 42
    ) -> list[dict]:
        """Generate a complete dataset with diverse instruction-response pairs."""
        random.seed(seed)
        torch.manual_seed(seed)

        templates = self.generate_instruction_templates()
        samples = []

        print(f"[SelfGen] Generating {num_samples} samples...")

        for i in range(num_samples):
            # Select random template and task
            template = random.choice(templates)
            task = random.choice(template["tasks"])
            instruction = template["instruction"].format(task=task)

            try:
                response = self.generate_sample(instruction, max_new_tokens)

                if self.apply_quality_filters(instruction, response):
                    sample = {
                        "instruction": instruction,
                        "input": "",  # No additional input for these simple tasks
                        "output": response,
                        "domain": template["domain"],
                    }
                    samples.append(sample)

                    if (i + 1) % 50 == 0:
                        print(
                            f"[SelfGen] Generated {len(samples)} valid samples ({i + 1}/{num_samples} attempted)"
                        )

            except Exception as e:
                print(f"[SelfGen] Error generating sample {i}: {e}")
                continue

        # Deduplicate
        samples = self.deduplicate_samples(samples)

        print(f"[SelfGen] Final dataset: {len(samples)} unique samples")
        return samples


def main():
    parser = argparse.ArgumentParser(
        description="Generate self-training data for BitNet"
    )
    parser.add_argument(
        "--model_path", required=True, help="Path to local model directory"
    )
    parser.add_argument("--out", required=True, help="Output JSONL file path")
    parser.add_argument(
        "--num", type=int, default=1000, help="Number of samples to generate"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=256, help="Max tokens per response"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="auto", help="Device: auto, cuda, cpu")

    args = parser.parse_args()

    # Set up environment defaults
    os.environ.setdefault("AIV_ROOT", r"D:\AIVillage")
    os.environ.setdefault("AIV_ARTIFACTS_DIR", r"D:\AIVillage\artifacts")
    os.environ.setdefault("WANDB_MODE", "offline")

    # Ensure output directory exists
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate dataset
    generator = SelfDataGenerator(args.model_path, args.device)
    samples = generator.generate_dataset(args.num, args.max_new_tokens, args.seed)

    # Save as JSONL
    with open(out_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"[SelfGen] Saved {len(samples)} samples to {out_path}")

    # Print sample statistics
    domains = {}
    for sample in samples:
        domain = sample.get("domain", "unknown")
        domains[domain] = domains.get(domain, 0) + 1

    print(f"[SelfGen] Domain distribution: {domains}")


if __name__ == "__main__":
    main()
