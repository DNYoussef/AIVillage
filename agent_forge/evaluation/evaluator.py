import torch
import torch.nn.functional as F
import language_tool_python
import re
import math
import nltk

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
        "avg_relevance": sum(thought_relevance) / len(thought_relevance)
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
        **thought_metrics
    }

_lt = language_tool_python.LanguageTool("en-US")


def measure_coherence(text: str) -> float:
    """Coherence computed from grammar and flow."""
    matches = _lt.check(text)
    grammar_penalty = len(matches)
    edge_penalty = len(re.findall(r"\.\s+[a-z]", text))
    return 1.0 / (1.0 + grammar_penalty + edge_penalty)

def measure_relevance(text: str, query: str) -> float:
    """Relevance via unigram BLEU overlap."""
    ref = query.lower().split()
    hyp = text.lower().split()
    if not ref or not hyp:
        return 0.0
    bleu = nltk.translate.bleu_score.sentence_bleu([ref], hyp, weights=(1, 0, 0, 0))
    return float(bleu)
