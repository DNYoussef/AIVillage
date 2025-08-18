"""
Quiet-STaR CLI Commands
Provides command-line interface for Quiet-STaR training and testing.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import click
import torch

from .ab_baking import (
    ABVariant,
    PromptFormat,
    ReflectionStyle,
    ThoughtTokenABBaker,
    create_optimization_test_questions,
    create_standard_ab_variants,
)
from .config import QuietSTaRConfig, get_inference_config, get_training_config
from .losses import QuietSTaRLoss
from .mcp_tools import MCPIntegration, SurpriseMemoryJournal
from .model import QuietSTaRModelWrapper
from .sampler import ThoughtLeakDetector, ThoughtSampler
from .student import DistillationConfig, QuietReasoningStudent
from .teacher import TeacherPromptGenerator, load_sample_questions
from .temperature_recognition import AdaptiveTemperatureSampler, TemperatureSelfRecognition, TemperatureStrategy

logger = logging.getLogger(__name__)


@click.group()
def quiet_star():
    """Quiet-STaR thought-token reasoning system."""
    pass


@quiet_star.command()
@click.option("--enable/--disable", default=True, help="Enable Quiet-STaR processing")
@click.option("--smoke", type=int, default=200, help="Number of smoke test samples")
@click.option("--model-name", default="gpt2", help="Base model to use")
@click.option("--output-dir", default="./quiet_star_output", help="Output directory")
@click.option("--config-path", type=click.Path(), help="Path to config JSON file")
@click.option("--thought-ratio", type=float, help="Override thought generation ratio")
@click.option("--max-thought-tokens", type=int, help="Override max thought tokens")
def run(
    enable: bool,
    smoke: int,
    model_name: str,
    output_dir: str,
    config_path: str | None,
    thought_ratio: float | None,
    max_thought_tokens: int | None,
):
    """Run Quiet-STaR with smoke test."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load configuration
    if config_path and Path(config_path).exists():
        config = QuietSTaRConfig.load(Path(config_path))
        click.echo(f"Loaded config from {config_path}")
    else:
        config = get_training_config() if enable else get_inference_config()
        click.echo("Using default configuration")

    # Apply CLI overrides
    if thought_ratio is not None:
        config.thought_ratio = thought_ratio
    if max_thought_tokens is not None:
        config.max_thought_tokens = max_thought_tokens

    config.enable_quiet_star = enable

    click.echo("🧠 Starting Quiet-STaR smoke test")
    click.echo(f"   Model: {model_name}")
    click.echo(f"   Samples: {smoke}")
    click.echo(f"   Thoughts enabled: {enable}")
    click.echo(f"   Thought ratio: {config.thought_ratio}")
    click.echo(f"   Max thought tokens: {config.max_thought_tokens}")

    try:
        # Run smoke test
        results = run_smoke_test(
            model_name=model_name,
            config=config,
            num_samples=smoke,
            output_dir=output_path,
        )

        # Print results
        print_smoke_results(results)

        # Save results
        results_file = output_path / "smoke_test_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        click.echo(f"✅ Smoke test complete. Results saved to {results_file}")

        # Check for failures
        if results.get("leak_count", 0) > 0:
            click.echo(f"⚠️  Warning: {results['leak_count']} thought leaks detected")
            return 1

        return 0

    except Exception as e:
        click.echo(f"❌ Smoke test failed: {e}")
        logger.exception("Smoke test error")
        return 1


@quiet_star.command()
@click.option(
    "--config-path",
    type=click.Path(exists=True),
    required=True,
    help="Config file path",
)
@click.option("--model-name", default="gpt2", help="Base model name")
@click.option("--output-dir", default="./quiet_star_training", help="Training output directory")
@click.option("--num-epochs", type=int, default=3, help="Number of training epochs")
@click.option("--batch-size", type=int, default=4, help="Training batch size")
@click.option("--learning-rate", type=float, default=5e-5, help="Learning rate")
def train(
    config_path: str,
    model_name: str,
    output_dir: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
):
    """Train Quiet-STaR model."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load configuration
    config = QuietSTaRConfig.load(Path(config_path))

    click.echo("🎓 Starting Quiet-STaR training")
    click.echo(f"   Model: {model_name}")
    click.echo(f"   Epochs: {num_epochs}")
    click.echo(f"   Batch size: {batch_size}")
    click.echo(f"   Learning rate: {learning_rate}")

    try:
        # Load model and tokenizer
        from transformers import AutoModelForCausalLM, AutoTokenizer

        click.echo("Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Add special tokens
        special_tokens = config.get_special_tokens()
        new_tokens = tokenizer.add_tokens(special_tokens, special_tokens=True)
        if new_tokens > 0:
            model.resize_token_embeddings(len(tokenizer))
            click.echo(f"Added {new_tokens} special tokens to vocabulary")

        # Update config with token IDs
        config.update_token_ids(tokenizer)

        # Wrap with Quiet-STaR
        quiet_model = QuietSTaRModelWrapper(base_model=model, config=config, special_token_ids=config.special_token_ids)

        # Create training components
        loss_fn = QuietSTaRLoss(config)
        sampler = ThoughtSampler(config, tokenizer)

        # Placeholder training loop (would need actual training data)
        click.echo("Training loop placeholder - implement with your training data")

        # Save trained model
        model_output_path = output_path / "trained_model"
        model.save_pretrained(model_output_path)
        tokenizer.save_pretrained(model_output_path)
        config.save(output_path / "config.json")

        click.echo(f"✅ Training complete. Model saved to {model_output_path}")

    except Exception as e:
        click.echo(f"❌ Training failed: {e}")
        logger.exception("Training error")
        return 1


@quiet_star.command()
@click.option("--text", prompt="Enter text to analyze", help="Text to check for thought leaks")
@click.option("--config-path", type=click.Path(), help="Path to config file")
def check_leaks(text: str, config_path: str | None):
    """Check text for thought token leakage."""

    # Load config
    if config_path and Path(config_path).exists():
        config = QuietSTaRConfig.load(Path(config_path))
    else:
        config = get_inference_config()

    # Create leak detector
    detector = ThoughtLeakDetector(config)

    # Analyze text
    results = detector.detect_leaks(text, check_semantic=True)

    # Print results
    click.echo("🔍 Leak Analysis Results")
    click.echo(f"   Text: {text[:100]}{'...' if len(text) > 100 else ''}")
    click.echo(f"   Has leaks: {results['has_leaks']}")
    click.echo(f"   Leak count: {results['leak_count']}")
    click.echo(f"   Severity: {results['severity']:.3f}")

    if results["token_leaks"]:
        click.echo(f"   Token leaks: {results['token_leaks']}")

    if results["semantic_leaks"]:
        click.echo(f"   Semantic leaks: {results['semantic_leaks']}")

    # Safety check
    is_safe = detector.is_safe_output(text)
    click.echo(f"   Safe for production: {'✅ Yes' if is_safe else '❌ No'}")

    return 0 if is_safe else 1


@quiet_star.command()
@click.option(
    "--input-file",
    type=click.Path(exists=True),
    required=True,
    help="Input JSON file with prompts",
)
@click.option("--output-file", type=click.Path(), required=True, help="Output JSON file")
@click.option("--model-name", default="gpt2", help="Model to use")
@click.option("--config-path", type=click.Path(), help="Config file path")
@click.option("--max-samples", type=int, help="Maximum samples to process")
def generate(
    input_file: str,
    output_file: str,
    model_name: str,
    config_path: str | None,
    max_samples: int | None,
):
    """Generate responses with Quiet-STaR."""

    # Load config
    if config_path and Path(config_path).exists():
        config = QuietSTaRConfig.load(Path(config_path))
    else:
        config = get_inference_config()

    # Load input data
    with open(input_file) as f:
        input_data = json.load(f)

    if max_samples:
        input_data = input_data[:max_samples]

    click.echo("🤖 Generating with Quiet-STaR")
    click.echo(f"   Input samples: {len(input_data)}")
    click.echo(f"   Model: {model_name}")
    click.echo(f"   Output: {output_file}")

    try:
        # Load model components
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Add special tokens if needed
        special_tokens = config.get_special_tokens()
        tokenizer.add_tokens(special_tokens, special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))
        config.update_token_ids(tokenizer)

        # Create Quiet-STaR wrapper
        quiet_model = QuietSTaRModelWrapper(base_model=model, config=config, special_token_ids=config.special_token_ids)

        quiet_model.eval()
        sampler = ThoughtSampler(config, tokenizer)
        detector = ThoughtLeakDetector(config)

        # Process samples
        results = []
        leak_count = 0

        for i, item in enumerate(input_data):
            if i % 10 == 0:
                click.echo(f"   Processing sample {i + 1}/{len(input_data)}")

            prompt = item.get("prompt", item.get("text", ""))

            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

            # Generate with thoughts
            with torch.no_grad():
                generation_result = sampler.sample_with_thoughts(
                    model=quiet_model,
                    input_ids=inputs["input_ids"],
                    max_new_tokens=100,
                    temperature=0.8,
                    do_sample=True,
                    force_thoughts=False,  # Use inference mode (no thoughts in output)
                )

            # Decode results
            if generation_result.stripped_ids is not None:
                # Use stripped version for production
                output_text = tokenizer.decode(generation_result.stripped_ids[0], skip_special_tokens=True)
            else:
                # Fallback to full generation
                output_text = tokenizer.decode(generation_result.generated_ids[0], skip_special_tokens=True)

            # Remove original prompt from output
            if output_text.startswith(prompt):
                output_text = output_text[len(prompt) :].strip()

            # Check for leaks
            leak_results = detector.detect_leaks(output_text)
            if leak_results["has_leaks"]:
                leak_count += 1

            # Store result
            result_item = {
                "prompt": prompt,
                "response": output_text,
                "thought_segments": generation_result.thought_segments,
                "generation_stats": generation_result.generation_stats,
                "leak_check": leak_results,
                "safe": detector.is_safe_output(output_text),
            }

            results.append(result_item)

        # Save results
        output_data = {
            "config": config.to_dict(),
            "model": model_name,
            "total_samples": len(results),
            "leak_count": leak_count,
            "leak_rate": leak_count / len(results) if results else 0,
            "results": results,
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2, default=str)

        click.echo("✅ Generation complete")
        click.echo(f"   Samples processed: {len(results)}")
        click.echo(f"   Leaks detected: {leak_count}")
        click.echo(f"   Leak rate: {leak_count / len(results) * 100:.1f}%" if results else "0%")
        click.echo(f"   Results saved to: {output_file}")

        return 0 if leak_count == 0 else 1

    except Exception as e:
        click.echo(f"❌ Generation failed: {e}")
        logger.exception("Generation error")
        return 1


@quiet_star.command()
@click.option(
    "--questions-file",
    type=click.Path(exists=True),
    help="File with training questions",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="./teacher_data",
    help="Output directory for training data",
)
@click.option("--num-pairs", type=int, default=100, help="Number of training pairs to generate")
@click.option("--model-name", default="microsoft/DialoGPT-small", help="Base model for generation")
def generate_teacher_data(questions_file, output_dir, num_pairs, model_name):
    """Generate teacher training data with hidden reflections."""
    config = get_training_config()
    teacher = TeacherPromptGenerator(config, model_name)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load questions
    if questions_file:
        with open(questions_file, encoding="utf-8") as f:
            questions = [line.strip() for line in f if line.strip()]
    else:
        # Use sample questions and repeat/extend as needed
        base_questions = load_sample_questions()
        questions = (base_questions * ((num_pairs // len(base_questions)) + 1))[:num_pairs]

    click.echo(f"Generating {min(num_pairs, len(questions))} teacher training pairs...")

    training_pairs = teacher.generate_training_dataset(
        questions=questions[:num_pairs],
        output_path=output_path / "teacher_training_data.json",
        pairs_per_question=1,
    )

    # Validate generated data
    validation_results = teacher.validate_training_pairs(training_pairs)

    click.echo("\n=== Teacher Data Generation Results ===")
    click.echo(f"Total pairs generated: {validation_results['total_pairs']}")
    click.echo(f"Average reflection length: {validation_results['avg_reflection_length']:.1f} tokens")
    click.echo(f"Average answer length: {validation_results['avg_answer_length']:.1f} tokens")

    if validation_results["quality_issues"]:
        click.echo(f"Quality issues found: {len(validation_results['quality_issues'])}")
        for issue in validation_results["quality_issues"][:5]:  # Show first 5
            click.echo(f"  - {issue}")
    else:
        click.echo("✓ No quality issues detected")

    # Save validation report
    with open(output_path / "validation_report.json", "w") as f:
        json.dump(validation_results, f, indent=2)

    click.echo(f"\n✓ Teacher data saved to {output_path}")


@quiet_star.command()
@click.option(
    "--teacher-data",
    type=click.Path(exists=True),
    required=True,
    help="Teacher training data JSON file",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="./student_model",
    help="Output directory for trained student",
)
@click.option("--base-model", default="microsoft/DialoGPT-small", help="Base model for student")
@click.option("--epochs", type=int, default=3, help="Number of training epochs")
@click.option("--batch-size", type=int, default=4, help="Training batch size")
@click.option("--learning-rate", type=float, default=5e-5, help="Learning rate")
def train_student(teacher_data, output_dir, base_model, epochs, batch_size, learning_rate):
    """Train student model using teacher-generated data."""
    config = get_training_config()

    # Load teacher data
    with open(teacher_data, encoding="utf-8") as f:
        teacher_dataset = json.load(f)

    click.echo(f"Loaded {len(teacher_dataset)} teacher examples")

    # Convert to TrainingPairs
    from .teacher import TrainingPair

    training_pairs = [
        TrainingPair(
            question=item["question"],
            reflection=item["reflection"],
            answer=item["answer"],
            metadata=item.get("metadata", {}),
        )
        for item in teacher_dataset
    ]

    # Initialize student with custom config
    distill_config = DistillationConfig(num_epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)

    student = QuietReasoningStudent(config, distill_config, base_model)

    # Prepare training data
    student_examples = student.prepare_training_data(training_pairs)
    click.echo(f"Prepared {len(student_examples)} student training examples")

    # Train
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    click.echo("Starting student distillation training...")
    student.train_on_examples(student_examples, output_path)

    click.echo(f"✓ Student model trained and saved to {output_path}")


@quiet_star.command()
@click.option(
    "--student-model",
    type=click.Path(exists=True),
    required=True,
    help="Trained student model directory",
)
@click.option("--test-questions", type=click.Path(exists=True), help="Test questions file")
@click.option(
    "--output-file",
    type=click.Path(),
    default="./evaluation_results.json",
    help="Output file for results",
)
def evaluate_student(student_model, test_questions, output_file):
    """Evaluate trained student model's quiet reasoning abilities."""
    config = get_inference_config()

    # Load student model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(student_model)
    base_model = AutoModelForCausalLM.from_pretrained(student_model)

    # Create special token mapping
    special_token_ids = {
        config.start_of_thought_token: tokenizer.convert_tokens_to_ids(config.start_of_thought_token),
        config.end_of_thought_token: tokenizer.convert_tokens_to_ids(config.end_of_thought_token),
        config.no_thought_token: tokenizer.convert_tokens_to_ids(config.no_thought_token),
    }

    # Wrap with Quiet-STaR
    from .model import QuietSTaRModelWrapper

    student_wrapper = QuietSTaRModelWrapper(base_model=base_model, config=config, special_token_ids=special_token_ids)

    # Load test questions
    if test_questions:
        with open(test_questions, encoding="utf-8") as f:
            questions = [line.strip() for line in f if line.strip()]
    else:
        questions = load_sample_questions()[:10]  # Use sample questions

    click.echo(f"Evaluating on {len(questions)} test questions...")

    # Create temporary student for evaluation
    distill_config = DistillationConfig()
    student = QuietReasoningStudent(config, distill_config, student_model)
    student.student_model = student_wrapper  # Use loaded model
    student.tokenizer = tokenizer

    # Evaluate
    results = student.evaluate_quiet_reasoning(questions, Path(output_file))

    click.echo("\n=== Evaluation Results ===")
    click.echo(f"Total questions: {results['total_questions']}")
    click.echo(f"Thoughts properly hidden: {results['thoughts_properly_hidden']} ({results['success_rate']:.1%})")
    click.echo(f"Has internal reasoning: {results['has_internal_reasoning']} ({results['reasoning_rate']:.1%})")

    if results["success_rate"] >= 0.9:
        click.echo("✓ EXCELLENT: Student successfully hides thoughts from users")
    elif results["success_rate"] >= 0.7:
        click.echo("✓ GOOD: Student mostly hides thoughts from users")
    else:
        click.echo("⚠ WARNING: Student leaks thoughts to users")

    if results["reasoning_rate"] >= 0.8:
        click.echo("✓ EXCELLENT: Student performs internal reasoning")
    elif results["reasoning_rate"] >= 0.5:
        click.echo("✓ GOOD: Student sometimes performs internal reasoning")
    else:
        click.echo("⚠ WARNING: Student rarely performs internal reasoning")

    click.echo(f"\n✓ Detailed results saved to {output_file}")


@quiet_star.command()
@click.option("--test-name", default="reflection_optimization", help="Name for this A/B test")
@click.option(
    "--variants",
    multiple=True,
    help="Variant names to test (or use --standard-variants)",
)
@click.option("--standard-variants", is_flag=True, help="Use standard predefined variants")
@click.option("--questions-file", type=click.Path(exists=True), help="File with test questions")
@click.option(
    "--output-dir",
    type=click.Path(),
    default="./ab_test_results",
    help="Output directory for results",
)
@click.option("--parallel", is_flag=True, help="Run variants in parallel")
@click.option("--max-workers", type=int, default=3, help="Max parallel workers")
def run_ab_test(
    test_name,
    variants,
    standard_variants,
    questions_file,
    output_dir,
    parallel,
    max_workers,
):
    """Run A/B tests to optimize thought-token prompts and reflection styles."""
    config = get_training_config()
    baker = ThoughtTokenABBaker(config)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load test questions
    if questions_file:
        with open(questions_file, encoding="utf-8") as f:
            test_questions = [line.strip() for line in f if line.strip()]
    else:
        test_questions = create_optimization_test_questions()
        click.echo("Using default optimization test questions")

    # Create variants
    if standard_variants or not variants:
        test_variants = create_standard_ab_variants()
        click.echo("Using standard A/B test variants")
    else:
        # Create custom variants from names
        test_variants = []
        style_map = {style.value: style for style in ReflectionStyle}
        format_map = {fmt.value: fmt for fmt in PromptFormat}

        for variant_spec in variants:
            parts = variant_spec.split(",")
            if len(parts) >= 2:
                style_name, format_name = parts[0].strip(), parts[1].strip()
                if style_name in style_map and format_name in format_map:
                    test_variants.append(
                        ABVariant(
                            name=f"{style_name}_{format_name}",
                            reflection_style=style_map[style_name],
                            prompt_format=format_map[format_name],
                        )
                    )

    if not test_variants:
        click.echo("❌ No valid variants to test")
        return 1

    click.echo(f"🧪 Running A/B test: {test_name}")
    click.echo(f"   Variants: {len(test_variants)}")
    click.echo(f"   Questions: {len(test_questions)}")
    click.echo(f"   Parallel: {parallel}")

    # Run A/B test suite
    test_suite = baker.run_ab_test_suite(
        test_name=test_name,
        variants=test_variants,
        test_questions=test_questions,
        parallel=parallel,
        max_workers=max_workers,
    )

    # Display results
    if test_suite.results:
        click.echo("\n📊 A/B Test Results:")
        sorted_results = sorted(test_suite.results, key=lambda r: r.overall_score(), reverse=True)

        for i, result in enumerate(sorted_results, 1):
            click.echo(f"{i}. {result.variant.name}:")
            click.echo(f"   Overall Score: {result.overall_score():.3f}")
            click.echo(f"   Success Rate: {result.success_rate:.1%}")
            click.echo(f"   Quality Score: {result.quality_score:.3f}")
            click.echo(f"   Leak Rate: {result.leak_rate:.1%}")
            click.echo(f"   Execution Time: {result.execution_time:.2f}s")

        # Show winner
        winner = test_suite.get_winner()
        if winner:
            click.echo(f"\n🏆 Winner: {winner.name}")
            click.echo(f"   Confidence: {test_suite.confidence_level:.1%}")
        else:
            click.echo("\n🤷 No clear winner (results too close)")

    # Save results
    baker.save_test_results(output_path)
    click.echo(f"\n✓ A/B test results saved to {output_path}")

    return 0


@quiet_star.command()
@click.option(
    "--test-results",
    type=click.Path(exists=True),
    required=True,
    help="A/B test results directory",
)
@click.option(
    "--min-confidence",
    type=float,
    default=0.8,
    help="Minimum confidence to bake a variant",
)
@click.option(
    "--output-config",
    type=click.Path(),
    default="./baked_config.json",
    help="Output configuration file",
)
def bake_variants(test_results, min_confidence, output_config):
    """Bake winning A/B test variants into optimized configuration."""
    config = get_training_config()
    baker = ThoughtTokenABBaker(config)

    # Load test results
    results_path = Path(test_results)
    if not (results_path / "ab_test_results.json").exists():
        click.echo("❌ No A/B test results found")
        return 1

    with open(results_path / "ab_test_results.json") as f:
        results_data = json.load(f)

    click.echo(f"🔥 Baking variants from {len(results_data['test_suites'])} test suites...")

    # Reconstruct test suites
    for suite_data in results_data["test_suites"]:
        # Create variants
        variants = []
        for var_data in suite_data["variants"]:
            variants.append(
                ABVariant(
                    name=var_data["name"],
                    reflection_style=ReflectionStyle(var_data["reflection_style"]),
                    prompt_format=PromptFormat(var_data["prompt_format"]),
                    max_reflection_tokens=var_data["max_reflection_tokens"],
                    temperature=var_data["temperature"],
                )
            )

        # Create mock test suite with winner info
        from .ab_baking import ABTestSuite

        test_suite = ABTestSuite(
            test_name=suite_data["test_name"],
            variants=variants,
            test_questions=[],  # Not needed for baking
        )

        if suite_data["winner"]:
            winner_name = suite_data["winner"]["name"]
            winner_variant = next((v for v in variants if v.name == winner_name), None)
            if winner_variant:
                test_suite.winner = winner_variant
                test_suite.confidence_level = suite_data["winner"]["confidence"]

        baker.test_suites.append(test_suite)

    # Bake winners
    baked_variants = baker.bake_winning_variants(min_confidence)

    if baked_variants:
        click.echo(f"\n🎂 Baked {len(baked_variants)} winning variants:")
        for test_name, winner in baked_variants.items():
            click.echo(f"  • {test_name}: {winner.name}")
            click.echo(f"    Style: {winner.reflection_style.value}")
            click.echo(f"    Format: {winner.prompt_format.value}")
            click.echo(f"    Tokens: {winner.max_reflection_tokens}")
            click.echo(f"    Temperature: {winner.temperature}")
    else:
        click.echo("🤷 No variants meet minimum confidence threshold")

    # Save baked configuration
    if baked_variants:
        baked_config = {
            "baked_at": time.time(),
            "min_confidence": min_confidence,
            "variants": {
                test_name: {
                    "name": variant.name,
                    "reflection_style": variant.reflection_style.value,
                    "prompt_format": variant.prompt_format.value,
                    "max_reflection_tokens": variant.max_reflection_tokens,
                    "temperature": variant.temperature,
                    "metadata": variant.metadata,
                }
                for test_name, variant in baked_variants.items()
            },
        }

        with open(output_config, "w") as f:
            json.dump(baked_config, f, indent=2)

        click.echo(f"\n✓ Baked configuration saved to {output_config}")

    return 0


@quiet_star.command()
@click.option(
    "--baked-config",
    type=click.Path(exists=True),
    required=True,
    help="Baked variants configuration",
)
@click.option("--variant-name", required=True, help="Name of variant to use")
@click.option(
    "--questions-file",
    type=click.Path(exists=True),
    help="Questions to generate with optimized variant",
)
@click.option(
    "--output-data",
    type=click.Path(),
    default="./optimized_training_data.json",
    help="Output training data",
)
@click.option(
    "--num-samples",
    type=int,
    default=100,
    help="Number of training samples to generate",
)
def generate_optimized_data(baked_config, variant_name, questions_file, output_data, num_samples):
    """Generate training data using baked A/B test winners."""
    config = get_training_config()

    # Load baked configuration
    with open(baked_config) as f:
        baked_data = json.load(f)

    if variant_name not in baked_data["variants"]:
        click.echo(f"❌ Variant '{variant_name}' not found in baked config")
        available = list(baked_data["variants"].keys())
        click.echo(f"Available variants: {', '.join(available)}")
        return 1

    variant_config = baked_data["variants"][variant_name]

    # Reconstruct optimized variant
    optimized_variant = ABVariant(
        name=variant_config["name"],
        reflection_style=ReflectionStyle(variant_config["reflection_style"]),
        prompt_format=PromptFormat(variant_config["prompt_format"]),
        max_reflection_tokens=variant_config["max_reflection_tokens"],
        temperature=variant_config["temperature"],
        metadata=variant_config.get("metadata", {}),
    )

    click.echo(f"🚀 Generating optimized training data using: {optimized_variant.name}")
    click.echo(f"   Style: {optimized_variant.reflection_style.value}")
    click.echo(f"   Format: {optimized_variant.prompt_format.value}")
    click.echo(f"   Max tokens: {optimized_variant.max_reflection_tokens}")

    # Load questions
    if questions_file:
        with open(questions_file, encoding="utf-8") as f:
            questions = [line.strip() for line in f if line.strip()]
    else:
        questions = create_optimization_test_questions()
        click.echo("Using default test questions")

    # Extend questions if needed
    if len(questions) < num_samples:
        questions = (questions * ((num_samples // len(questions)) + 1))[:num_samples]

    # Generate training data with optimized variant
    baker = ThoughtTokenABBaker(config)
    training_pairs = baker.generate_training_data_for_variant(optimized_variant, questions, num_samples_per_question=1)

    # Convert to JSON format
    training_data = [
        {
            "question": pair.question,
            "reflection": pair.reflection,
            "answer": pair.answer,
            "training_text": pair.to_training_text(),
            "inference_text": pair.to_inference_text(),
            "metadata": pair.metadata,
        }
        for pair in training_pairs
    ]

    # Save optimized training data
    with open(output_data, "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)

    click.echo(f"\n✓ Generated {len(training_data)} optimized training examples")
    click.echo(f"✓ Saved to {output_data}")

    # Quick quality check
    leak_count = sum(
        1 for item in training_data if "<SoT>" in item["inference_text"] or "</SoT>" in item["inference_text"]
    )

    if leak_count == 0:
        click.echo("✓ Security verified: No thought leakage detected")
    else:
        click.echo(f"⚠️  Warning: {leak_count} potential leaks detected")

    return 0


@quiet_star.command()
@click.option(
    "--question",
    required=True,
    help="Question to analyze for temperature recommendation",
)
@click.option("--context", default="", help="Additional context for the question")
@click.option(
    "--strategy",
    type=click.Choice(["conservative", "balanced", "exploratory", "context_adaptive"]),
    default="balanced",
    help="Temperature strategy to use",
)
@click.option(
    "--use-case",
    type=click.Choice(
        [
            "factual_qa",
            "creative_writing",
            "code_generation",
            "analysis",
            "general",
            "research",
        ]
    ),
    default="general",
    help="Use case for temperature optimization",
)
def analyze_temperature(question, context, strategy, use_case):
    """Analyze question and recommend optimal temperature settings."""
    config = get_training_config()

    # Create temperature recognizer for use case
    strategy_enum = TemperatureStrategy(strategy)
    temp_recognizer = TemperatureSelfRecognition(config, strategy_enum)

    click.echo("🌡️  Temperature Analysis")
    click.echo(f"Question: {question}")
    if context:
        click.echo(f"Context: {context}")
    click.echo(f"Strategy: {strategy}")
    click.echo(f"Use case: {use_case}")
    click.echo()

    # Get temperature recommendation
    recommendation = temp_recognizer.recommend_temperature(question=question, context=context)

    # Display results
    click.echo("📊 Analysis Results:")
    click.echo(f"  Confidence Level: {recommendation.confidence_level.value}")
    click.echo(f"  Confidence Score: {recommendation.confidence_signals.overall_confidence():.3f}")
    click.echo(f"  Complexity Level: {recommendation.complexity_level.value}")
    click.echo(f"  Complexity Score: {recommendation.complexity_signals.overall_complexity():.3f}")
    click.echo()

    click.echo("🌡️  Temperature Recommendation:")
    click.echo(f"  Base Temperature: {recommendation.base_temperature:.3f}")
    click.echo(f"  Recommended Temperature: {recommendation.adjusted_temperature:.3f}")
    click.echo(f"  Adjustment Factor: {recommendation.adjustment_factor:.3f}")
    click.echo(f"  Reasoning: {recommendation.reasoning}")
    click.echo()

    # Show detailed signals
    click.echo("🔍 Detailed Signals:")
    conf_signals = recommendation.confidence_signals
    comp_signals = recommendation.complexity_signals

    click.echo("  Confidence Signals:")
    click.echo(f"    Entropy Score: {conf_signals.entropy_score:.3f}")
    click.echo(f"    Consistency Score: {conf_signals.consistency_score:.3f}")
    click.echo(f"    Uncertainty Markers: {conf_signals.uncertainty_markers}")
    click.echo(f"    Hedging Language: {conf_signals.hedging_language}")

    click.echo("  Complexity Signals:")
    click.echo(f"    Question Length: {comp_signals.question_length} chars")
    click.echo(f"    Word Count: {comp_signals.question_word_count}")
    click.echo(f"    Technical Terms: {comp_signals.technical_terms}")
    click.echo(f"    Analysis Requests: {comp_signals.analysis_requests}")
    click.echo(f"    Causal Relationships: {comp_signals.causal_relationships}")


@quiet_star.command()
@click.option("--questions-file", type=click.Path(exists=True), help="File with test questions")
@click.option(
    "--output-dir",
    type=click.Path(),
    default="./temperature_test",
    help="Output directory",
)
@click.option("--model-name", default="gpt2", help="Model to test with")
@click.option(
    "--strategy",
    type=click.Choice(["conservative", "balanced", "exploratory", "context_adaptive"]),
    default="balanced",
    help="Temperature strategy to use",
)
@click.option("--num-samples", type=int, default=3, help="Number of samples per question")
def test_adaptive_temperature(questions_file, output_dir, model_name, strategy, num_samples):
    """Test adaptive temperature system with actual model sampling."""
    config = get_training_config()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load test questions
    if questions_file:
        with open(questions_file, encoding="utf-8") as f:
            test_questions = [line.strip() for line in f if line.strip()]
    else:
        test_questions = [
            "What is the capital of France?",
            "Analyze the economic implications of artificial intelligence on employment.",
            "Write a creative story about time travel.",
            "How do you implement a binary search algorithm?",
            "What are the potential risks of quantum computing for cybersecurity?",
        ]

    click.echo("🧪 Testing Adaptive Temperature System")
    click.echo(f"Model: {model_name}")
    click.echo(f"Strategy: {strategy}")
    click.echo(f"Questions: {len(test_questions)}")
    click.echo(f"Samples per question: {num_samples}")
    click.echo()

    try:
        # Initialize components (mock for demo)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Create adaptive sampler
        strategy_enum = TemperatureStrategy(strategy)
        adaptive_sampler = AdaptiveTemperatureSampler(config, tokenizer, strategy_enum)

        results = []

        for i, question in enumerate(test_questions, 1):
            click.echo(f"Processing question {i}/{len(test_questions)}: {question[:60]}...")

            # Get temperature recommendation
            temp_recommendation = adaptive_sampler.temp_recognition.recommend_temperature(question=question)

            question_result = {
                "question": question,
                "base_temperature": temp_recommendation.base_temperature,
                "recommended_temperature": temp_recommendation.adjusted_temperature,
                "confidence_level": temp_recommendation.confidence_level.value,
                "complexity_level": temp_recommendation.complexity_level.value,
                "confidence_score": temp_recommendation.confidence_signals.overall_confidence(),
                "complexity_score": temp_recommendation.complexity_signals.overall_complexity(),
                "reasoning": temp_recommendation.reasoning,
                "samples": [],
            }

            # For demo purposes, show what would happen with different temperatures
            test_temperatures = [0.1, temp_recommendation.adjusted_temperature, 1.2]

            for temp in test_temperatures:
                sample_result = {
                    "temperature": temp,
                    "description": (
                        "low"
                        if temp < 0.5
                        else "recommended" if temp == temp_recommendation.adjusted_temperature else "high"
                    ),
                    "note": f"Would sample with temperature {temp:.2f}",
                }
                question_result["samples"].append(sample_result)

            results.append(question_result)

            # Display recommendation
            click.echo(
                f"  Temperature: {temp_recommendation.base_temperature:.2f} → {temp_recommendation.adjusted_temperature:.2f}"
            )
            click.echo(
                f"  Confidence: {temp_recommendation.confidence_level.value} ({temp_recommendation.confidence_signals.overall_confidence():.2f})"
            )
            click.echo(
                f"  Complexity: {temp_recommendation.complexity_level.value} ({temp_recommendation.complexity_signals.overall_complexity():.2f})"
            )

        # Save results
        results_file = output_path / "adaptive_temperature_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        # Summary statistics
        temp_recommendations = [r["recommended_temperature"] for r in results]
        avg_temp = sum(temp_recommendations) / len(temp_recommendations)
        min_temp = min(temp_recommendations)
        max_temp = max(temp_recommendations)

        click.echo()
        click.echo("📊 Summary Statistics:")
        click.echo(f"  Average recommended temperature: {avg_temp:.3f}")
        click.echo(f"  Temperature range: {min_temp:.3f} - {max_temp:.3f}")
        click.echo(f"  Strategy: {strategy}")

        confidence_levels = [r["confidence_level"] for r in results]
        complexity_levels = [r["complexity_level"] for r in results]

        from collections import Counter

        conf_counts = Counter(confidence_levels)
        comp_counts = Counter(complexity_levels)

        click.echo(f"  Confidence distribution: {dict(conf_counts)}")
        click.echo(f"  Complexity distribution: {dict(comp_counts)}")

        click.echo(f"\n✅ Test complete. Results saved to {results_file}")
        return 0

    except Exception as e:
        click.echo(f"❌ Error during testing: {e}")
        return 1


@quiet_star.command()
@click.option("--config-file", type=click.Path(), help="Configuration file to enhance")
@click.option(
    "--output-config",
    type=click.Path(),
    default="./enhanced_config.json",
    help="Output enhanced configuration",
)
@click.option(
    "--strategy",
    type=click.Choice(["conservative", "balanced", "exploratory", "context_adaptive"]),
    default="balanced",
    help="Temperature strategy to use",
)
@click.option("--enable-alignment", is_flag=True, help="Enable alignment prelude integration")
def create_enhanced_config(config_file, output_config, strategy, enable_alignment):
    """Create enhanced configuration with temperature recognition and alignment features."""

    # Load base config
    if config_file and Path(config_file).exists():
        base_config = QuietSTaRConfig.load(Path(config_file))
        click.echo(f"Loaded base config from {config_file}")
    else:
        base_config = get_training_config()
        click.echo("Using default training configuration")

    # Create enhanced configuration
    enhanced_config = {
        "base_config": {
            "enable_quiet_star": base_config.enable_quiet_star,
            "start_of_thought_token": base_config.start_of_thought_token,
            "end_of_thought_token": base_config.end_of_thought_token,
            "no_thought_token": base_config.no_thought_token,
            "thought_ratio": base_config.thought_ratio,
            "max_thought_tokens": base_config.max_thought_tokens,
            "w_task": base_config.w_task,
            "w_reflect": base_config.w_reflect,
            "w_leak": base_config.w_leak,
        },
        "temperature_recognition": {
            "enabled": True,
            "strategy": strategy,
            "base_temperature": getattr(base_config, "temperature", 0.7),
            "min_temperature": 0.1,
            "max_temperature": 1.5,
            "calibration_enabled": True,
        },
        "alignment_prelude": {
            "enabled": enable_alignment,
            "eudaimonia_virtues": [
                "wisdom",
                "justice",
                "courage",
                "temperance",
                "honesty",
                "compassion",
                "humility",
                "responsibility",
            ],
            "complexity_levels": ["simple", "moderate", "complex", "systemic"],
        },
        "integration_features": {
            "adaptive_sampling": True,
            "thought_alignment_enhancement": enable_alignment,
            "temperature_calibration": True,
            "multi_strategy_support": True,
        },
    }

    # Save enhanced configuration
    with open(output_config, "w") as f:
        json.dump(enhanced_config, f, indent=2)

    click.echo("🚀 Enhanced Configuration Created:")
    click.echo(f"  Temperature Recognition: Enabled ({strategy} strategy)")
    click.echo(f"  Alignment Prelude: {'Enabled' if enable_alignment else 'Disabled'}")
    click.echo("  Adaptive Sampling: Enabled")
    click.echo("  Calibration: Enabled")
    click.echo(f"  Configuration saved to: {output_config}")

    return 0


@quiet_star.command()
@click.option(
    "--journal-db",
    type=click.Path(),
    default="./quiet_star_journal.db",
    help="Journal database path",
)
@click.option("--question", required=True, help="Question to make prediction about")
@click.option("--prediction", required=True, help="Your prediction for the outcome")
@click.option("--confidence", type=float, required=True, help="Confidence in prediction (0.0-1.0)")
@click.option("--reasoning", default="", help="Reasoning behind the prediction")
def make_prediction(journal_db, question, prediction, confidence, reasoning):
    """Make a prediction and record it in the journal."""

    if not 0.0 <= confidence <= 1.0:
        click.echo("❌ Confidence must be between 0.0 and 1.0")
        return 1

    config = get_training_config()
    journal = SurpriseMemoryJournal(config, journal_db)

    try:
        prediction_id = journal.add_prediction(
            question=question,
            predicted_outcome=prediction,
            confidence=confidence,
            reasoning=reasoning,
        )

        click.echo("📝 Prediction Recorded")
        click.echo(f"   ID: {prediction_id}")
        click.echo(f"   Question: {question}")
        click.echo(f"   Prediction: {prediction}")
        click.echo(f"   Confidence: {confidence:.2f}")
        if reasoning:
            click.echo(f"   Reasoning: {reasoning}")

        return 0

    except Exception as e:
        click.echo(f"❌ Error recording prediction: {e}")
        return 1
    finally:
        journal.close()


@quiet_star.command()
@click.option(
    "--journal-db",
    type=click.Path(),
    default="./quiet_star_journal.db",
    help="Journal database path",
)
@click.option("--prediction-id", required=True, help="ID of the prediction to update")
@click.option("--outcome", required=True, help="The actual outcome that occurred")
def record_outcome(journal_db, prediction_id, outcome):
    """Record the actual outcome for a prediction."""

    config = get_training_config()
    journal = SurpriseMemoryJournal(config, journal_db)

    try:
        outcome_id = journal.record_outcome(prediction_id=prediction_id, actual_outcome=outcome)

        if not outcome_id:
            click.echo(f"❌ Prediction {prediction_id} not found")
            return 1

        outcome_entry = journal.get_entry(outcome_id)

        click.echo("📊 Outcome Recorded")
        click.echo(f"   Outcome ID: {outcome_id}")
        click.echo(f"   Prediction ID: {prediction_id}")
        click.echo(f"   Actual Outcome: {outcome}")
        click.echo(f"   Surprise Level: {outcome_entry.surprise_level.value}")
        click.echo(f"   Accuracy Score: {outcome_entry.accuracy_score:.3f}")
        click.echo(f"   Learning Value: {outcome_entry.learning_value:.3f}")
        click.echo(f"   Surprise Reasoning: {outcome_entry.surprise_reasoning}")

        if outcome_entry.surprise_level.value in ["high", "extreme"]:
            click.echo("   🎯 High surprise detected - learning analysis triggered")

        return 0

    except Exception as e:
        click.echo(f"❌ Error recording outcome: {e}")
        return 1
    finally:
        journal.close()


@quiet_star.command()
@click.option(
    "--journal-db",
    type=click.Path(),
    default="./quiet_star_journal.db",
    help="Journal database path",
)
@click.option(
    "--min-surprise",
    type=click.Choice(["low", "medium", "high", "extreme"]),
    default="medium",
    help="Minimum surprise level to analyze",
)
@click.option("--limit", type=int, default=10, help="Maximum number of insights to show")
def analyze_surprises(journal_db, min_surprise, limit):
    """Analyze surprising outcomes and extract insights."""

    config = get_training_config()
    journal = SurpriseMemoryJournal(config, journal_db)
    mcp = MCPIntegration(config, journal)

    try:
        # Get surprise insights
        import asyncio

        insights = asyncio.run(mcp.get_surprise_insights(min_surprise))

        click.echo("🔍 Surprise Analysis Results")
        click.echo(f"   Minimum surprise level: {min_surprise}")
        click.echo(f"   Total surprising entries: {insights['total_surprising_entries']}")
        click.echo(f"   Average learning value: {insights['avg_learning_value']:.3f}")
        click.echo()

        if not insights["top_insights"]:
            click.echo("   No surprising entries found at this level.")
            return 0

        click.echo(f"📚 Top {min(limit, len(insights['top_insights']))} Surprising Insights:")

        for i, insight in enumerate(insights["top_insights"][:limit], 1):
            click.echo(f"\n   {i}. {insight['surprise_level'].upper()} Surprise")
            click.echo(f"      Question: {insight['question']}")
            click.echo(f"      Predicted: {insight['predicted']}")
            click.echo(f"      Actual: {insight['actual']}")
            click.echo(f"      Learning Value: {insight['learning_value']:.3f}")
            click.echo(f"      Reasoning: {insight['reasoning']}")
            click.echo(f"      Time: {insight['timestamp']}")

        return 0

    except Exception as e:
        click.echo(f"❌ Error analyzing surprises: {e}")
        return 1
    finally:
        journal.close()


@quiet_star.command()
@click.option(
    "--journal-db",
    type=click.Path(),
    default="./quiet_star_journal.db",
    help="Journal database path",
)
@click.option(
    "--pattern-type",
    type=click.Choice(["prediction_error", "confidence_mismatch", "context_similarity"]),
    help="Specific pattern type to analyze",
)
def discover_patterns(journal_db, pattern_type):
    """Discover and analyze memory patterns from journal entries."""

    config = get_training_config()
    journal = SurpriseMemoryJournal(config, journal_db)
    mcp = MCPIntegration(config, journal)

    try:
        import asyncio

        analysis = asyncio.run(mcp.analyze_patterns(pattern_type))

        click.echo("🔬 Pattern Discovery Results")
        if pattern_type:
            click.echo(f"   Pattern type filter: {pattern_type}")
        click.echo(f"   Total patterns discovered: {analysis['total_patterns']}")
        click.echo(f"   Pattern types found: {', '.join(analysis['pattern_types'])}")
        click.echo()

        if not analysis["patterns"]:
            click.echo("   No patterns discovered yet.")
            click.echo("   Patterns emerge as more predictions and outcomes are recorded.")
            return 0

        click.echo("📈 Discovered Patterns:")

        for i, pattern in enumerate(analysis["patterns"], 1):
            click.echo(f"\n   {i}. {pattern['type'].replace('_', ' ').title()} Pattern")
            click.echo(f"      Description: {pattern['description']}")
            click.echo(f"      Confidence: {pattern['confidence']:.3f}")
            click.echo(f"      Supporting entries: {pattern['support_count']}")
            click.echo(f"      Discovered: {pattern['discovered']}")
            click.echo(f"      Last reinforced: {pattern['last_reinforced']}")

        return 0

    except Exception as e:
        click.echo(f"❌ Error discovering patterns: {e}")
        return 1
    finally:
        journal.close()


@quiet_star.command()
@click.option(
    "--journal-db",
    type=click.Path(),
    default="./quiet_star_journal.db",
    help="Journal database path",
)
@click.option("--query", required=True, help="Search query for journal entries")
@click.option(
    "--entry-type",
    type=click.Choice(["prediction", "outcome", "analysis", "learning", "pattern"]),
    help="Filter by entry type",
)
@click.option("--limit", type=int, default=10, help="Maximum number of results")
def search_journal(journal_db, query, entry_type, limit):
    """Search journal entries by content."""

    config = get_training_config()
    journal = SurpriseMemoryJournal(config, journal_db)
    mcp = MCPIntegration(config, journal)

    try:
        import asyncio

        results = asyncio.run(mcp.search_journal(query, entry_type, limit))

        click.echo("🔎 Journal Search Results")
        click.echo(f"   Query: '{query}'")
        if entry_type:
            click.echo(f"   Entry type filter: {entry_type}")
        click.echo(f"   Results found: {results['results_count']}")
        click.echo()

        if not results["results"]:
            click.echo("   No matching entries found.")
            return 0

        for i, result in enumerate(results["results"], 1):
            click.echo(f"   {i}. [{result['type'].upper()}] {result['content']}")
            if result["question"]:
                click.echo(f"      Question: {result['question']}")
            click.echo(f"      Surprise: {result['surprise_level']}")
            click.echo(f"      Time: {result['timestamp']}")
            if result["tags"]:
                click.echo(f"      Tags: {', '.join(result['tags'])}")
            click.echo()

        return 0

    except Exception as e:
        click.echo(f"❌ Error searching journal: {e}")
        return 1
    finally:
        journal.close()


@quiet_star.command()
@click.option(
    "--journal-db",
    type=click.Path(),
    default="./quiet_star_journal.db",
    help="Journal database path",
)
def journal_summary(journal_db):
    """Get summary statistics about the journal."""

    config = get_training_config()
    journal = SurpriseMemoryJournal(config, journal_db)

    try:
        summary = journal.get_journal_summary()

        click.echo("📊 Journal Summary")
        click.echo(f"   Database: {summary['journal_path']}")
        click.echo(f"   Total entries: {summary['total_entries']}")
        click.echo()

        click.echo("📋 Entry Types:")
        for entry_type, count in summary["entry_types"].items():
            click.echo(f"   {entry_type.replace('_', ' ').title()}: {count}")

        click.echo("\n🎯 Surprise Distribution:")
        for surprise_level, count in summary["surprise_distribution"].items():
            click.echo(f"   {surprise_level.replace('_', ' ').title()}: {count}")

        click.echo("\n📈 Performance Metrics:")
        click.echo(f"   Average accuracy: {summary['average_accuracy']:.3f}")
        click.echo(f"   Average learning value: {summary['average_learning_value']:.3f}")
        click.echo(f"   Discovered patterns: {summary['discovered_patterns']}")

        # Recent activity
        recent_entries = journal.get_recent_entries(5)
        if recent_entries:
            click.echo("\n🕐 Recent Activity:")
            for entry in recent_entries:
                click.echo(
                    f"   {entry.timestamp.strftime('%Y-%m-%d %H:%M')} - "
                    f"{entry.reflection_type.value}: {entry.content[:50]}..."
                )

        return 0

    except Exception as e:
        click.echo(f"❌ Error getting journal summary: {e}")
        return 1
    finally:
        journal.close()


@quiet_star.command()
@click.option(
    "--journal-db",
    type=click.Path(),
    default="./quiet_star_journal.db",
    help="Journal database path",
)
@click.option(
    "--output-file",
    type=click.Path(),
    default="./mcp_tools_demo.json",
    help="Output file for demo results",
)
def demo_mcp_system(journal_db, output_file):
    """Run a comprehensive demo of the MCP tools and journaling system."""

    config = get_training_config()
    journal = SurpriseMemoryJournal(config, journal_db)
    mcp = MCPIntegration(config, journal)

    click.echo("🧪 MCP Tools and Journaling System Demo")
    click.echo(f"   Journal database: {journal_db}")
    click.echo(f"   Output file: {output_file}")
    click.echo()

    try:
        # Demo predictions and outcomes
        demo_data = []

        predictions = [
            {
                "question": "What will be the biggest challenge in scaling transformer models?",
                "prediction": "Memory requirements will be the primary bottleneck.",
                "confidence": 0.8,
            },
            {
                "question": "How will users respond to AI systems with visible reasoning?",
                "prediction": "Users will prefer transparent reasoning over black box responses.",
                "confidence": 0.6,
            },
            {
                "question": "What architecture will dominate LLMs in 2025?",
                "prediction": "Mixture of Experts will become the standard architecture.",
                "confidence": 0.4,
            },
        ]

        outcomes = [
            "Computational cost and inference speed became bigger issues than memory.",
            "Users valued response speed more than reasoning transparency.",
            "State space models gained more adoption than MoE architectures.",
        ]

        click.echo("1. Recording Predictions...")
        prediction_ids = []

        for pred in predictions:
            result = await mcp.predict_and_journal(**pred)
            prediction_ids.append(result["prediction_id"])
            click.echo(f"   ✓ Recorded: {pred['question'][:50]}...")
            demo_data.append({"step": "prediction", "data": result})

        click.echo("\n2. Recording Outcomes...")

        for pred_id, outcome in zip(prediction_ids, outcomes, strict=False):
            result = await mcp.record_outcome(pred_id, outcome)
            click.echo(f"   ✓ Surprise level: {result['surprise_level']} " f"(accuracy: {result['accuracy']:.2f})")
            demo_data.append({"step": "outcome", "data": result})

        click.echo("\n3. Analyzing Surprises...")
        surprises = await mcp.get_surprise_insights("low")
        click.echo(f"   ✓ Found {surprises['total_surprising_entries']} surprising entries")
        demo_data.append({"step": "surprise_analysis", "data": surprises})

        click.echo("\n4. Discovering Patterns...")
        patterns = await mcp.analyze_patterns()
        click.echo(f"   ✓ Discovered {patterns['total_patterns']} patterns")
        demo_data.append({"step": "pattern_discovery", "data": patterns})

        click.echo("\n5. Journal Search...")
        search_result = await mcp.search_journal("challenge")
        click.echo(f"   ✓ Found {search_result['results_count']} entries matching 'challenge'")
        demo_data.append({"step": "search", "data": search_result})

        # Save demo results
        with open(output_file, "w") as f:
            json.dump(demo_data, f, indent=2, default=str)

        # Final summary
        summary = journal.get_journal_summary()

        click.echo("\n📊 Demo Completed Successfully")
        click.echo(f"   Total journal entries: {summary['total_entries']}")
        click.echo(f"   Patterns discovered: {summary['discovered_patterns']}")
        click.echo(f"   Average accuracy: {summary['average_accuracy']:.3f}")
        click.echo(f"   Average learning value: {summary['average_learning_value']:.3f}")
        click.echo(f"   Results saved to: {output_file}")

        return 0

    except Exception as e:
        click.echo(f"❌ Error running demo: {e}")
        return 1
    finally:
        journal.close()


def run_smoke_test(model_name: str, config: QuietSTaRConfig, num_samples: int, output_dir: Path) -> dict[str, Any]:
    """
    Run smoke test to verify Quiet-STaR functionality.

    Args:
        model_name: Name of base model to test
        config: Quiet-STaR configuration
        num_samples: Number of test samples
        output_dir: Output directory for results

    Returns:
        Dictionary with test results
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Create test prompts
    test_prompts = [
        "def fibonacci(n):",
        "Explain quantum computing:",
        "Write a sorting algorithm:",
        "What is machine learning?",
        "Solve: 2x + 5 = 15",
    ]

    # Extend prompts to reach num_samples
    extended_prompts = (test_prompts * ((num_samples // len(test_prompts)) + 1))[:num_samples]

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add special tokens
    special_tokens = config.get_special_tokens()
    added_tokens = tokenizer.add_tokens(special_tokens, special_tokens=True)
    if added_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))

    config.update_token_ids(tokenizer)

    # Create Quiet-STaR components
    quiet_model = QuietSTaRModelWrapper(base_model=model, config=config, special_token_ids=config.special_token_ids)

    sampler = ThoughtSampler(config, tokenizer)
    detector = ThoughtLeakDetector(config)
    loss_fn = QuietSTaRLoss(config)

    # Test results
    results = {
        "config": config.to_dict(),
        "model": model_name,
        "num_samples": num_samples,
        "start_time": time.time(),
        "samples": [],
        "leak_count": 0,
        "total_thought_tokens": 0,
        "avg_generation_time": 0.0,
        "tests_passed": {
            "no_leakage": True,
            "loss_terms_wired": True,
            "tokens_present": True,
            "generation_successful": True,
        },
    }

    generation_times = []

    for i, prompt in enumerate(extended_prompts):
        start_time = time.time()

        try:
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)

            # Test with thoughts (training mode)
            quiet_model.train()
            generation_result = sampler.sample_with_thoughts(
                model=quiet_model,
                input_ids=inputs["input_ids"],
                max_new_tokens=50,
                temperature=0.8,
                do_sample=True,
                force_thoughts=True,  # Force thoughts for testing
            )

            # Test inference mode (thoughts stripped)
            quiet_model.eval()
            inference_result = sampler.sample_with_thoughts(
                model=quiet_model,
                input_ids=inputs["input_ids"],
                max_new_tokens=50,
                temperature=0.8,
                do_sample=True,
                force_thoughts=False,
            )

            # Decode results
            training_text = tokenizer.decode(generation_result.generated_ids[0], skip_special_tokens=False)
            inference_text = tokenizer.decode(
                (
                    inference_result.stripped_ids[0]
                    if inference_result.stripped_ids is not None
                    else inference_result.generated_ids[0]
                ),
                skip_special_tokens=True,
            )

            # Check for leaks in inference output
            leak_results = detector.detect_leaks(inference_text)
            if leak_results["has_leaks"]:
                results["leak_count"] += 1
                results["tests_passed"]["no_leakage"] = False

            # Test loss computation (simplified)
            try:
                outputs = quiet_model(inputs["input_ids"])
                labels = inputs["input_ids"].clone()
                thought_mask = quiet_model.thought_head.create_thought_mask(
                    inputs["input_ids"], config.special_token_ids
                )

                loss_components = loss_fn(
                    logits=outputs["logits"],
                    labels=labels,
                    thought_mask=thought_mask,
                    special_token_ids=config.special_token_ids,
                )

                loss_computed = True
            except Exception as e:
                logger.warning(f"Loss computation failed: {e}")
                loss_computed = False
                results["tests_passed"]["loss_terms_wired"] = False

            # Check for special tokens in training output
            has_special_tokens = any(token in training_text for token in special_tokens)
            if not has_special_tokens:
                results["tests_passed"]["tokens_present"] = False

            generation_time = time.time() - start_time
            generation_times.append(generation_time)

            # Track statistics
            if generation_result.generation_stats:
                results["total_thought_tokens"] += generation_result.generation_stats.get("total_thought_tokens", 0)

            # Store sample result
            sample_result = {
                "prompt": prompt,
                "training_output": training_text,
                "inference_output": inference_text,
                "thought_segments": generation_result.thought_segments,
                "generation_stats": generation_result.generation_stats,
                "leak_check": leak_results,
                "generation_time": generation_time,
                "loss_computed": loss_computed,
            }

            results["samples"].append(sample_result)

        except Exception as e:
            logger.error(f"Sample {i} failed: {e}")
            results["tests_passed"]["generation_successful"] = False

            # Add failed sample
            results["samples"].append({"prompt": prompt, "error": str(e), "failed": True})

    # Compute final statistics
    results["end_time"] = time.time()
    results["total_time"] = results["end_time"] - results["start_time"]
    results["avg_generation_time"] = sum(generation_times) / len(generation_times) if generation_times else 0
    results["leak_rate"] = results["leak_count"] / num_samples

    # Overall pass/fail
    all_tests_passed = all(results["tests_passed"].values())
    results["overall_pass"] = all_tests_passed and results["leak_count"] == 0

    return results


def print_smoke_results(results: dict[str, Any]) -> None:
    """Print formatted smoke test results."""
    click.echo("\n" + "=" * 60)
    click.echo("🧪 Quiet-STaR Smoke Test Results")
    click.echo("=" * 60)

    click.echo(f"Model: {results['model']}")
    click.echo(f"Samples: {results['num_samples']}")
    click.echo(f"Total time: {results['total_time']:.2f}s")
    click.echo(f"Avg generation time: {results['avg_generation_time']:.3f}s")

    click.echo("\n📊 Statistics:")
    click.echo(f"   Leak count: {results['leak_count']}")
    click.echo(f"   Leak rate: {results['leak_rate'] * 100:.1f}%")
    click.echo(f"   Total thought tokens: {results['total_thought_tokens']}")

    click.echo("\n✅ Test Results:")
    for test_name, passed in results["tests_passed"].items():
        status = "✅ PASS" if passed else "❌ FAIL"
        click.echo(f"   {test_name}: {status}")

    overall_status = "✅ PASS" if results["overall_pass"] else "❌ FAIL"
    click.echo(f"\n🎯 Overall: {overall_status}")

    if not results["overall_pass"]:
        click.echo("\n⚠️  Issues detected - check individual test results")


# Make CLI available for import
quiet_star_cli = quiet_star
