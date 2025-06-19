import torch
from nlp.metrics import measure_coherence, measure_relevance

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
