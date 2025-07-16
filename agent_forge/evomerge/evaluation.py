from concurrent.futures import ProcessPoolExecutor
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import EvoMergeException

logger = logging.getLogger(__name__)


def evaluate_model(model_path: str) -> dict[str, float | str]:
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        results = {}

        # Perplexity evaluation
        results["perplexity"] = evaluate_perplexity(model, tokenizer)

        # Task-specific evaluations
        results["coding"] = evaluate_coding(model, tokenizer)
        results["mathematics"] = evaluate_mathematics(model, tokenizer)
        results["writing"] = evaluate_writing(model, tokenizer)

        # Zero-shot task evaluation
        results["zero_shot_classification"] = evaluate_zero_shot_classification(
            model, tokenizer
        )
        results["zero_shot_qa"] = evaluate_zero_shot_qa(model, tokenizer)

        # Coherence and fluency
        results["coherence"] = evaluate_coherence(model, tokenizer)

        # Calculate overall score
        results["overall_score"] = calculate_overall_score(results)

        return results

    except Exception as e:
        logger.error(f"Error during model evaluation: {e!s}")
        raise EvoMergeException(f"Error evaluating model: {e!s}")


def evaluate_perplexity(
    model, tokenizer, test_text="The quick brown fox jumps over the lazy dog"
):
    inputs = tokenizer(test_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        return torch.exp(outputs.loss).item()


def evaluate_coding(model, tokenizer):
    coding_prompt = "Write a Python function to find the nth Fibonacci number."
    inputs = tokenizer(coding_prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=200)
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Basic evaluation of generated code
    score = 0
    if "def" in generated_code:
        score += 0.2
    if "fibonacci" in generated_code.lower():
        score += 0.2
    if "return" in generated_code:
        score += 0.2
    if "for" in generated_code or "while" in generated_code:
        score += 0.2
    if "if" in generated_code:
        score += 0.2

    return score


def evaluate_mathematics(model, tokenizer):
    math_problems = [
        "What is 15 + 27?",
        "If x + 5 = 12, what is x?",
        "What is the square root of 64?",
        "What is 20% of 80?",
        "If a triangle has a base of 6 and a height of 4, what is its area?",
    ]
    correct_answers = [42, 7, 8, 16, 12]

    score = 0
    for problem, correct_answer in zip(math_problems, correct_answers, strict=False):
        inputs = tokenizer(problem, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=50)
        generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        try:
            # Extract the numerical answer from the generated text
            numerical_answer = float("".join(filter(str.isdigit, generated_answer)))
            if (
                abs(numerical_answer - correct_answer) < 0.1
            ):  # Allow for small floating-point errors
                score += 1
        except ValueError:
            pass

    return score / len(math_problems)


def evaluate_writing(model, tokenizer):
    writing_prompt = "Write a short paragraph about the importance of renewable energy."
    inputs = tokenizer(writing_prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=200)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Basic evaluation of generated text
    score = 0
    if len(generated_text.split()) >= 50:  # Check if it's at least 50 words
        score += 0.3
    if "renewable energy" in generated_text.lower():
        score += 0.2
    if any(
        word in generated_text.lower()
        for word in ["solar", "wind", "hydro", "geothermal"]
    ):
        score += 0.2
    if "environment" in generated_text.lower() or "climate" in generated_text.lower():
        score += 0.2
    if "future" in generated_text.lower() or "sustainable" in generated_text.lower():
        score += 0.1

    return score


def evaluate_zero_shot_classification(model, tokenizer):
    texts = [
        "I love this product! It's amazing!",
        "This is the worst experience I've ever had.",
        "The weather is quite nice today.",
        "I'm not sure how I feel about this.",
    ]
    labels = ["positive", "negative", "neutral"]

    score = 0
    for text in texts:
        inputs = tokenizer(
            f"Classify the sentiment of this text: '{text}' Labels: {', '.join(labels)}",
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=50)
        generated_label = tokenizer.decode(outputs[0], skip_special_tokens=True).lower()

        if any(label in generated_label for label in labels):
            score += 1

    return score / len(texts)


def evaluate_zero_shot_qa(model, tokenizer):
    qa_pairs = [
        ("What is the capital of France?", "Paris"),
        ("Who wrote 'Romeo and Juliet'?", "Shakespeare"),
        ("What is the largest planet in our solar system?", "Jupiter"),
        ("What year did World War II end?", "1945"),
    ]

    score = 0
    for question, answer in qa_pairs:
        inputs = tokenizer(f"Question: {question} Answer:", return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=50)
        generated_answer = tokenizer.decode(
            outputs[0], skip_special_tokens=True
        ).lower()

        if answer.lower() in generated_answer:
            score += 1

    return score / len(qa_pairs)


def evaluate_coherence(model, tokenizer):
    prompt = "Write a short story about a time traveler."
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=200)
    generated_story = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Basic evaluation of coherence
    score = 0
    sentences = generated_story.split(".")
    if len(sentences) >= 3:
        score += 0.3
    if "time" in generated_story.lower() and "travel" in generated_story.lower():
        score += 0.2
    if (
        len(set(generated_story.split())) / len(generated_story.split()) > 0.7
    ):  # Vocabulary diversity
        score += 0.2
    if any(
        word in generated_story.lower() for word in ["past", "future", "history", "era"]
    ):
        score += 0.2
    if "the end" in generated_story.lower() or "conclusion" in generated_story.lower():
        score += 0.1

    return score


def calculate_overall_score(results: dict[str, float]) -> float:
    weights = {
        "perplexity": 0.2,
        "coding": 0.15,
        "mathematics": 0.15,
        "writing": 0.15,
        "zero_shot_classification": 0.1,
        "zero_shot_qa": 0.1,
        "coherence": 0.15,
    }

    overall_score = 0
    for metric, weight in weights.items():
        if metric == "perplexity":
            # Lower perplexity is better, so we invert it
            overall_score += weight * (1 / results[metric])
        else:
            overall_score += weight * results[metric]

    return overall_score


def parallel_evaluate_models(
    model_paths: list[str], max_workers: int = None
) -> list[dict[str, float | str]]:
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(evaluate_model, model_paths))
