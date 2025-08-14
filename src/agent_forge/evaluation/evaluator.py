import re
import logging
from concurrent.futures import ProcessPoolExecutor

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def evaluate_thought_quality(model, eval_data):
    thought_coherence = []
    thought_relevance = []

    for batch in eval_data:
        inputs, attention_mask, targets = batch
        thoughts = model.generate_thoughts(inputs, attention_mask)

        coherence = measure_coherence(thoughts)
        thought_coherence.append(coherence)

        relevance = measure_relevance(thoughts, targets)
        thought_relevance.append(relevance)

    return {
        "avg_coherence": sum(thought_coherence) / len(thought_coherence),
        "avg_relevance": sum(thought_relevance) / len(thought_relevance),
    }


def evaluate_model(model_or_path, eval_data=None):
    """Evaluate a model on a variety of metrics.

    The function accepts either a model instance or a path to a model. When a
    path is provided, the model and tokenizer are loaded automatically.  If
    ``eval_data`` is supplied, traditional language modelling metrics such as
    perplexity and accuracy are calculated over the dataset and thought quality
    is assessed.  Regardless of ``eval_data`` the EvoMerge benchmark metrics are
    also computed.
    """

    if isinstance(model_or_path, str):
        logger.info("Loading model from %s", model_or_path)
        model = AutoModelForCausalLM.from_pretrained(model_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_or_path)
    else:
        model = model_or_path
        tokenizer = getattr(model, "tokenizer", None) or getattr(model, "tok", None)
        if tokenizer is None and hasattr(model, "config"):
            try:
                tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
            except Exception:  # pragma: no cover - best effort
                logger.warning("Unable to infer tokenizer from model; some metrics may fail")

    metrics = {}
    if eval_data is not None:
        total_loss = 0.0
        total_accuracy = 0.0
        for batch in eval_data:
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs.loss
            total_loss += loss.item()
            predictions = outputs.logits.argmax(dim=-1)
            accuracy = (predictions == labels).float().mean()
            total_accuracy += accuracy.item()

        avg_loss = total_loss / len(eval_data)
        avg_accuracy = total_accuracy / len(eval_data)
        metrics.update(
            {
                "perplexity": torch.exp(torch.tensor(avg_loss)).item(),
                "accuracy": avg_accuracy,
                **evaluate_thought_quality(model, eval_data),
            }
        )
    else:
        # Fallback perplexity on a short generic text
        metrics["perplexity"] = evaluate_perplexity(model, tokenizer)

    # EvoMerge specialised metrics
    if tokenizer is not None:
        try:
            metrics.update(
                {
                    "coding": evaluate_coding(model, tokenizer),
                    "mathematics": evaluate_mathematics(model, tokenizer),
                    "writing": evaluate_writing(model, tokenizer),
                    "zero_shot_classification": evaluate_zero_shot_classification(model, tokenizer),
                    "zero_shot_qa": evaluate_zero_shot_qa(model, tokenizer),
                    "coherence": evaluate_story_coherence(model, tokenizer),
                }
            )
            metrics["overall_score"] = calculate_overall_score(metrics)
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("Specialised evaluation failed: %s", exc)

    return metrics


def measure_coherence(text: str) -> float:
    """Approximate coherence via cosine similarity between consecutive sentences."""
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    if len(sentences) < 2:
        return 1.0
    vectorizer = TfidfVectorizer().fit(sentences)
    vectors = vectorizer.transform(sentences)
    sims = cosine_similarity(vectors[:-1], vectors[1:])
    return float(sims.diagonal().mean())


def measure_relevance(text: str, query: str) -> float:
    """Relevance computed from cosine similarity of TF-IDF vectors."""
    docs = [query, text]
    vectorizer = TfidfVectorizer().fit(docs)
    vectors = vectorizer.transform(docs)
    sim = cosine_similarity(vectors[0:1], vectors[1:2])[0, 0]
    return float(sim)


# --- EvoMerge specialised metrics -------------------------------------------------


def evaluate_perplexity(model, tokenizer, test_text="The quick brown fox jumps over the lazy dog"):
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
            numerical_answer = float("".join(filter(str.isdigit, generated_answer)))
            if abs(numerical_answer - correct_answer) < 0.1:
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

    score = 0
    if len(generated_text.split()) >= 50:
        score += 0.3
    if "renewable energy" in generated_text.lower():
        score += 0.2
    if any(word in generated_text.lower() for word in ["solar", "wind", "hydro", "geothermal"]):
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
        generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True).lower()

        if answer.lower() in generated_answer:
            score += 1

    return score / len(qa_pairs)


def evaluate_story_coherence(model, tokenizer):
    prompt = "Write a short story about a time traveler."
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=200)
    generated_story = tokenizer.decode(outputs[0], skip_special_tokens=True)

    score = 0
    sentences = generated_story.split(".")
    if len(sentences) >= 3:
        score += 0.3
    if "time" in generated_story.lower() and "travel" in generated_story.lower():
        score += 0.2
    if len(set(generated_story.split())) / len(generated_story.split()) > 0.7:
        score += 0.2
    if any(word in generated_story.lower() for word in ["past", "future", "history", "era"]):
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

    overall_score = 0.0
    for metric, weight in weights.items():
        if metric in results:
            if metric == "perplexity":
                overall_score += weight * (1 / results[metric])
            else:
                overall_score += weight * results[metric]
    return overall_score


def parallel_evaluate_models(model_paths: list[str], max_workers: int | None = None) -> list[dict[str, float | str]]:
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(evaluate_model, model_paths))
