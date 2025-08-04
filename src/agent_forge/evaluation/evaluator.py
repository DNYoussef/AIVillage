import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch


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


def evaluate_model(model, eval_data):
    total_loss = 0
    total_accuracy = 0

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

    thought_metrics = evaluate_thought_quality(model, eval_data)

    return {
        "perplexity": torch.exp(torch.tensor(avg_loss)).item(),
        "accuracy": avg_accuracy,
        **thought_metrics,
    }


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
