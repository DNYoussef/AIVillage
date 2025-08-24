#!/usr/bin/env python3
"""HypeRAG LoRA Training Data Generator.

Generates domain-specific training data for LoRA fine-tuning by:
1. Reusing inference templates from prompt_bank.md
2. Creating synthetic violations and expected repairs
3. Outputting JSONL format with prompt/completion pairs
"""

import argparse
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
import random
from typing import Any

logger = logging.getLogger(__name__)

# Domain-specific violation templates
DOMAIN_VIOLATIONS = {
    "medical": [
        {
            "type": "drug_allergy_conflict",
            "template": "Patient {patient_id} is prescribed {drug} but has documented allergy to {allergen}",
            "expected_repair": {
                "operation": "delete_edge",
                "target": "PRESCRIBED_{drug}",
                "rationale": "Remove dangerous prescription conflicting with allergy",
            },
        },
        {
            "type": "missing_dosage",
            "template": "Medication {drug} for patient {patient_id} lacks dosage information",
            "expected_repair": {
                "operation": "update_attr",
                "target": "PRESCRIBED_{drug}",
                "attr": "dosage",
                "value": "{standard_dosage}",
                "rationale": "Add standard dosage per clinical guidelines",
            },
        },
        {
            "type": "temporal_inconsistency",
            "template": "Treatment {treatment} dated {future_date} is in the future",
            "expected_repair": {
                "operation": "update_attr",
                "target": "{treatment}",
                "attr": "date",
                "value": "{corrected_date}",
                "rationale": "Correct temporal inconsistency",
            },
        },
    ],
    "movies": [
        {
            "type": "missing_genre",
            "template": "Movie '{title}' ({year}) has no genre classification",
            "expected_repair": {
                "operation": "add_edge",
                "source": "{title}",
                "edge_type": "HAS_GENRE",
                "target": "{inferred_genre}",
                "rationale": "Add genre based on similar movies and cast",
            },
        },
        {
            "type": "duplicate_entity",
            "template": "Found duplicate entries for '{title}': nodes {node1} and {node2}",
            "expected_repair": {
                "operation": "merge_nodes",
                "nodes": ["{node1}", "{node2}"],
                "rationale": "Merge duplicate movie entries",
            },
        },
        {
            "type": "orphaned_review",
            "template": "Review {review_id} not connected to any movie",
            "expected_repair": {
                "operation": "delete_edge",
                "target": "{review_id}",
                "rationale": "Remove orphaned review with no movie reference",
            },
        },
    ],
    "finance": [
        {
            "type": "invalid_transaction",
            "template": "Transaction {tx_id} has negative timestamp {timestamp}",
            "expected_repair": {
                "operation": "update_attr",
                "target": "{tx_id}",
                "attr": "timestamp",
                "value": "{corrected_timestamp}",
                "rationale": "Fix invalid timestamp",
            },
        },
        {
            "type": "missing_category",
            "template": "Transaction {tx_id} for ${amount} lacks category",
            "expected_repair": {
                "operation": "update_attr",
                "target": "{tx_id}",
                "attr": "category",
                "value": "{inferred_category}",
                "rationale": "Categorize based on merchant and amount patterns",
            },
        },
    ],
}

# Base system prompt template
BASE_SYSTEM_PROMPT = """You are a knowledge graph repair assistant for the HypeRAG system. Your responsibilities:
1. Analyze graph violations and inconsistencies
2. Propose minimal, targeted repairs
3. Preserve existing valid data
4. Provide clear rationales for changes"""

# Domain-specific system prompts
DOMAIN_SYSTEM_PROMPTS = {
    "medical": """You are a medical knowledge graph repair assistant. Additional responsibilities:
- Prioritize patient safety above all
- Consider drug interactions and contraindications
- Validate clinical data against medical guidelines
- Flag high-risk modifications for human review""",
    "movies": """You are a movie knowledge graph repair assistant. Focus on:
- Maintaining data consistency across movie metadata
- Resolving entity duplicates and ambiguities
- Preserving user-generated content when possible
- Inferring missing relationships from context""",
    "finance": """You are a financial knowledge graph repair assistant. Ensure:
- Transaction integrity and audit trails
- Compliance with financial regulations
- Accurate categorization and timestamps
- Detection of anomalous patterns""",
}


class LoRADataGenerator:
    def __init__(self, domain: str, seed: int = 42) -> None:
        self.domain = domain
        self.rng = random.Random(seed)
        self.stats = {
            "total_examples": 0,
            "by_violation_type": {},
            "avg_prompt_length": 0,
            "avg_completion_length": 0,
        }

    def generate_prompt(self, violation: dict[str, Any]) -> str:
        """Generate a complete prompt for the violation."""
        system_prompt = BASE_SYSTEM_PROMPT
        if self.domain in DOMAIN_SYSTEM_PROMPTS:
            system_prompt += "\n\n" + DOMAIN_SYSTEM_PROMPTS[self.domain]

        instruction = f"""
Analyze the following knowledge graph violation and propose a repair:

**Violation**: {violation["description"]}

**Available Operations**:
- add_edge(source, edge_type, target)
- delete_edge(target)
- update_attr(target, attr, value)
- merge_nodes(nodes)

Provide your repair in JSON format:
```json
{{
    "operation": "operation_type",
    "parameters": {{...}},
    "rationale": "explanation",
    "confidence": 0.0-1.0,
    "safety_impact": "none|low|medium|high"  # for medical domain
}}
```
"""
        return f"{system_prompt}\n\n{instruction}"

    def generate_completion(self, violation: dict[str, Any], repair: dict[str, Any]) -> str:
        """Generate the expected completion for the violation."""
        completion = {
            "operation": repair["operation"],
            "rationale": repair["rationale"],
            "confidence": self.rng.uniform(0.7, 0.95),
        }

        # Add operation-specific parameters
        if repair["operation"] == "add_edge":
            completion["source"] = repair.get("source", "entity_1")
            completion["edge_type"] = repair.get("edge_type", "RELATES_TO")
            completion["target"] = repair.get("target", "entity_2")
        elif repair["operation"] == "delete_edge":
            completion["target"] = repair.get("target", "edge_1")
        elif repair["operation"] == "update_attr":
            completion["target"] = repair.get("target", "entity_1")
            completion["attr"] = repair.get("attr", "attribute")
            completion["value"] = repair.get("value", "new_value")
        elif repair["operation"] == "merge_nodes":
            completion["nodes"] = repair.get("nodes", ["node_1", "node_2"])

        # Add safety impact for medical domain
        if self.domain == "medical":
            completion["safety_impact"] = self.rng.choice(["low", "medium", "high"])

        return json.dumps(completion, indent=2)

    def generate_examples(self, count: int) -> list[dict[str, str]]:
        """Generate training examples for the domain."""
        if self.domain not in DOMAIN_VIOLATIONS:
            msg = f"Unknown domain: {self.domain}"
            raise ValueError(msg)

        violations = DOMAIN_VIOLATIONS[self.domain]
        examples = []

        for _i in range(count):
            # Select a random violation type
            violation_template = self.rng.choice(violations)

            # Fill in template variables
            violation_desc = violation_template["template"]
            if self.domain == "medical":
                violation_desc = violation_desc.format(
                    patient_id=f"P{self.rng.randint(1000, 9999)}",
                    drug=self.rng.choice(["Aspirin", "Penicillin", "Warfarin", "Ibuprofen"]),
                    allergen=self.rng.choice(["Penicillin", "Sulfa", "Aspirin", "NSAIDs"]),
                    treatment=f"TX{self.rng.randint(100, 999)}",
                    future_date="2025-12-01",
                    corrected_date="2024-12-01",
                    standard_dosage="5mg daily",
                )
            elif self.domain == "movies":
                violation_desc = violation_desc.format(
                    title=self.rng.choice(["The Matrix", "Inception", "Interstellar", "Dune"]),
                    year=self.rng.randint(1990, 2024),
                    node1=f"movie_{self.rng.randint(100, 999)}",
                    node2=f"movie_{self.rng.randint(1000, 1999)}",
                    review_id=f"review_{self.rng.randint(10000, 99999)}",
                    inferred_genre=self.rng.choice(["Sci-Fi", "Action", "Drama", "Thriller"]),
                )
            elif self.domain == "finance":
                violation_desc = violation_desc.format(
                    tx_id=f"TX{self.rng.randint(100000, 999999)}",
                    timestamp="-1234567890",
                    corrected_timestamp="1734567890",
                    amount=self.rng.randint(10, 1000),
                    inferred_category=self.rng.choice(["Food", "Transport", "Shopping", "Bills"]),
                )

            violation = {
                "type": violation_template["type"],
                "description": violation_desc,
            }

            prompt = self.generate_prompt(violation)
            completion = self.generate_completion(violation, violation_template["expected_repair"])

            examples.append(
                {
                    "prompt": prompt,
                    "completion": completion,
                    "metadata": {
                        "domain": self.domain,
                        "violation_type": violation["type"],
                        "timestamp": datetime.now(UTC).isoformat(),
                    },
                }
            )

            # Update stats
            self.stats["total_examples"] += 1
            self.stats["by_violation_type"][violation["type"]] = (
                self.stats["by_violation_type"].get(violation["type"], 0) + 1
            )

        # Calculate average lengths
        if examples:
            self.stats["avg_prompt_length"] = sum(len(e["prompt"]) for e in examples) / len(examples)
            self.stats["avg_completion_length"] = sum(len(e["completion"]) for e in examples) / len(examples)

        return examples

    def save_dataset(self, examples: list[dict[str, str]], output_path: Path) -> None:
        """Save examples to JSONL file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")

        # Save stats
        stats_path = output_path.with_suffix(".stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2)

        logger.info(f"Saved {len(examples)} examples to {output_path}")
        logger.info(f"Stats saved to {stats_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LoRA training data for HypeRAG")
    parser.add_argument(
        "--domain",
        required=True,
        choices=["medical", "movies", "finance"],
        help="Domain for training data generation",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1000,
        help="Number of examples to generate (default: 1000)",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output JSONL file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Generate data
    generator = LoRADataGenerator(domain=args.domain, seed=args.seed)
    examples = generator.generate_examples(count=args.count)

    # Save dataset
    generator.save_dataset(examples, args.out)

    # Print summary
    print("\nGeneration Summary:")
    print(f"  Domain: {args.domain}")
    print(f"  Total examples: {generator.stats['total_examples']}")
    print(f"  Avg prompt length: {generator.stats['avg_prompt_length']:.0f} chars")
    print(f"  Avg completion length: {generator.stats['avg_completion_length']:.0f} chars")
    print("\nExamples by violation type:")
    for vtype, count in generator.stats["by_violation_type"].items():
        print(f"  - {vtype}: {count}")


if __name__ == "__main__":
    main()
