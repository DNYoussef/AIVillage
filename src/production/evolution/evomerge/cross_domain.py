import types

import torch
from transformers import AutoTokenizer

from .config import MergeSettings, ModelDomain
from .evaluation import (
    evaluate_coding,
    evaluate_mathematics,
    evaluate_writing,
    evaluate_zero_shot_classification,
    evaluate_zero_shot_qa,
)


def get_model_domain(model: torch.nn.Module, tokenizer: AutoTokenizer) -> ModelDomain:
    """Detect the domain of a model based on its architecture and vocabulary."""
    architecture = model.__class__.__name__
    vocab = tokenizer.get_vocab()

    # Check for domain-specific keywords in the vocabulary
    coding_keywords = ["function", "class", "def", "return", "import"]
    math_keywords = ["sum", "product", "equation", "integral", "derivative"]
    writing_keywords = ["paragraph", "essay", "narrative", "character", "plot"]

    domain_scores = {
        "coding": sum(1 for word in coding_keywords if word in vocab),
        "mathematics": sum(1 for word in math_keywords if word in vocab),
        "writing": sum(1 for word in writing_keywords if word in vocab),
    }

    primary_domain = max(domain_scores, key=domain_scores.get)

    if "GPT" in architecture or "Transformer" in architecture:
        task_type = "language_modeling"
    elif "BERT" in architecture:
        task_type = "masked_language_modeling"
    elif "T5" in architecture:
        task_type = "text_to_text"
    else:
        task_type = "unknown"

    return ModelDomain(name=primary_domain, architecture=architecture, task_type=task_type)


def evaluate_cross_domain_model(
    model: torch.nn.Module, tokenizer: AutoTokenizer, target_domain: ModelDomain
) -> dict[str, float]:
    """Evaluate a cross-domain merged model on tasks specific to the target domain and general tasks."""
    results = {}

    # Evaluate on domain-specific tasks
    if target_domain.name == "coding":
        results["coding_score"] = evaluate_coding(model, tokenizer)
    elif target_domain.name == "mathematics":
        results["math_score"] = evaluate_mathematics(model, tokenizer)
    elif target_domain.name == "writing":
        results["writing_score"] = evaluate_writing(model, tokenizer)

    # Evaluate on general tasks
    results["zero_shot_classification"] = evaluate_zero_shot_classification(model, tokenizer)
    results["zero_shot_qa"] = evaluate_zero_shot_qa(model, tokenizer)

    # Calculate perplexity on a general text
    general_text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(general_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        results["perplexity"] = torch.exp(outputs.loss).item()

    # Calculate overall score
    results["overall_score"] = sum(results.values()) / len(results)

    return results


def merge_cross_domain_models(models: list[torch.nn.Module], merge_settings: MergeSettings) -> torch.nn.Module:
    """Merge models from different domains using a more advanced strategy."""
    base_model = models[0]
    num_models = len(models)

    # Initialize merged parameters
    merged_params = {}
    for name, param in base_model.named_parameters():
        merged_params[name] = torch.zeros_like(param)

    # Merge parameters using a weighted average based on layer depth
    with torch.no_grad():
        for name, param in base_model.named_parameters():
            layer_depth = int(name.split(".")[1]) if "." in name else 0
            for i, model in enumerate(models):
                # Calculate weight based on layer depth and model index
                weight = (1 + layer_depth) / (num_models * (1 + layer_depth) - i)
                merged_params[name] += weight * model.state_dict()[name]

    # Apply merged parameters to the base model
    base_model.load_state_dict(merged_params)

    return base_model


def adapt_model_to_domain(model: torch.nn.Module, target_domain: ModelDomain) -> torch.nn.Module:
    """Adapt a merged model to a specific target domain."""
    # Add domain-specific adapter layers
    adapter_config = {
        "coding": {"hidden_size": 64, "num_layers": 2},
        "mathematics": {"hidden_size": 32, "num_layers": 1},
        "writing": {"hidden_size": 128, "num_layers": 3},
    }

    config = adapter_config.get(target_domain.name, {"hidden_size": 64, "num_layers": 2})

    # Add adapter layers to the model
    for i in range(len(model.transformer.h)):
        adapter = torch.nn.Sequential(
            torch.nn.Linear(model.config.hidden_size, config["hidden_size"]),
            torch.nn.ReLU(),
            torch.nn.Linear(config["hidden_size"], model.config.hidden_size),
        )
        model.transformer.h[i].adapter = adapter

    # Modify the forward pass to include adapter layers
    def modified_forward(self, *args, **kwargs):
        outputs = self.original_forward(*args, **kwargs)
        hidden_states = outputs[0]
        for layer in self.transformer.h:
            hidden_states = hidden_states + layer.adapter(hidden_states)
        outputs = (hidden_states, *outputs[1:])
        return outputs

    model.original_forward = model.forward
    model.forward = types.MethodType(modified_forward, model)

    return model


def cross_domain_fine_tuning(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    target_domain: ModelDomain,
    train_data: list[str],
    num_epochs: int = 3,
):
    """Fine-tune a merged model on domain-specific data."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    for epoch in range(num_epochs):
        total_loss = 0
        for text in train_data:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_data)}")

    return model


# Example usage of the new functions
def cross_domain_merge_and_adapt(
    models: list[torch.nn.Module],
    tokenizers: list[AutoTokenizer],
    merge_settings: MergeSettings,
    target_domain: ModelDomain,
    train_data: list[str],
) -> torch.nn.Module:
    # Detect domains of input models
    domains = [get_model_domain(model, tokenizer) for model, tokenizer in zip(models, tokenizers, strict=False)]
    print(f"Input model domains: {[domain.name for domain in domains]}")

    # Merge models
    merged_model = merge_cross_domain_models(models, merge_settings)
    print("Models merged successfully")

    # Adapt merged model to target domain
    adapted_model = adapt_model_to_domain(merged_model, target_domain)
    print(f"Model adapted to target domain: {target_domain.name}")

    # Fine-tune the adapted model
    fine_tuned_model = cross_domain_fine_tuning(adapted_model, tokenizers[0], target_domain, train_data)
    print("Model fine-tuned on domain-specific data")

    # Evaluate the final model
    evaluation_results = evaluate_cross_domain_model(fine_tuned_model, tokenizers[0], target_domain)
    print(f"Final model evaluation results: {evaluation_results}")

    return fine_tuned_model
