import torch
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, TextDataset, DataCollatorForLanguageModeling, AutoConfig
from torch.utils.data import DataLoader
from .utils import EvoMergeException
from .config import MergeSettings
from .merge_techniques import MERGE_TECHNIQUES

def is_instruction_tuned_model(model: torch.nn.Module, tokenizer: AutoTokenizer) -> bool:
    """
    Check if a model is likely to be instruction-tuned based on its vocabulary and special tokens.
    """
    instruction_keywords = ["instruction", "task", "input", "output", "human", "assistant"]
    special_tokens = tokenizer.all_special_tokens
    vocab = tokenizer.get_vocab()
    
    # Check for instruction-related keywords in vocabulary
    keyword_presence = sum(1 for keyword in instruction_keywords if keyword in vocab)
    
    # Check for special tokens that might indicate instruction tuning
    special_token_presence = sum(1 for token in special_tokens if any(keyword in token.lower() for keyword in instruction_keywords))
    
    # If the model has a significant number of instruction-related keywords or special tokens, consider it instruction-tuned
    return (keyword_presence >= 3) or (special_token_presence >= 2)

def merge_instruction_tuned_models(models: List[torch.nn.Module], tokenizers: List[AutoTokenizer], merge_settings: MergeSettings) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    Merge instruction-tuned models while preserving their instruction-following capabilities.
    """
    try:
        if merge_settings.merge_method == "ps":
            merged_model = merge_ps(models, merge_settings)
        elif merge_settings.merge_method == "dfs":
            merged_model = merge_dfs(models, merge_settings)
        elif merge_settings.merge_method == "ps_dfs":
            merged_model = merge_ps_dfs(models, merge_settings)
        else:
            raise EvoMergeException(f"Unsupported merge method for instruction-tuned models: {merge_settings.merge_method}")
        
        # Merge tokenizers
        merged_tokenizer = merge_tokenizers(tokenizers)
        
        # Preserve instruction-tuning capabilities
        merged_model = preserve_instruction_tuning(merged_model, models, tokenizers)
        
        return merged_model, merged_tokenizer
    except Exception as e:
        raise EvoMergeException(f"Failed to merge instruction-tuned models: {str(e)}")

def merge_ps(models: List[torch.nn.Module], merge_settings: MergeSettings) -> torch.nn.Module:
    """Merge models using parameter space techniques."""
    merged_state_dict = {}
    for technique in merge_settings.ps_techniques:
        if technique not in MERGE_TECHNIQUES:
            raise ValueError(f"Unknown merge technique: {technique}")
        
        technique_params = merge_settings.parameters.get(technique, {})
        weights = technique_params.pop('weights', None)
        
        if weights is not None:
            merged_state_dict = MERGE_TECHNIQUES[technique](merged_state_dict, models, weights=weights, **technique_params)
        else:
            merged_state_dict = MERGE_TECHNIQUES[technique](merged_state_dict, models, **technique_params)
    
    config = AutoConfig.from_pretrained(models[0].config._name_or_path)
    merged_model = AutoModelForCausalLM.from_config(config)
    merged_model.load_state_dict(merged_state_dict)
    return merged_model

def merge_dfs(models: List[torch.nn.Module], merge_settings: MergeSettings) -> torch.nn.Module:
    """Merge models using deep fusion space techniques."""
    config = AutoConfig.from_pretrained(models[0].config._name_or_path)
    merged_model = AutoModelForCausalLM.from_config(config)
    
    for technique in merge_settings.dfs_techniques:
        if technique not in MERGE_TECHNIQUES:
            raise ValueError(f"Unknown merge technique: {technique}")
        merged_model = MERGE_TECHNIQUES[technique](merged_model, models, **merge_settings.parameters.get(technique, {}))
    return merged_model

def merge_ps_dfs(models: List[torch.nn.Module], merge_settings: MergeSettings) -> torch.nn.Module:
    """Merge models using both parameter space and deep fusion space techniques."""
    ps_model = merge_ps(models, merge_settings)
    return merge_dfs([ps_model] + models, merge_settings)

def merge_tokenizers(tokenizers: List[AutoTokenizer]) -> AutoTokenizer:
    """Merge tokenizers from multiple models."""
    base_tokenizer = tokenizers[0]
    for tokenizer in tokenizers[1:]:
        base_tokenizer.add_tokens(list(set(tokenizer.get_vocab().keys()) - set(base_tokenizer.get_vocab().keys())))
    return base_tokenizer

def preserve_instruction_tuning(merged_model: torch.nn.Module, original_models: List[torch.nn.Module], tokenizers: List[AutoTokenizer]) -> torch.nn.Module:
    """Preserve instruction-tuning capabilities in the merged model."""
    # Identify instruction-related tokens and embeddings
    instruction_embeddings = []
    for model, tokenizer in zip(original_models, tokenizers):
        instruction_tokens = [token for token in tokenizer.get_vocab().keys() if "instruction" in token.lower() or "task" in token.lower()]
        instruction_ids = tokenizer.convert_tokens_to_ids(instruction_tokens)
        instruction_embeddings.append(model.get_input_embeddings().weight[instruction_ids])
    
    # Average the instruction-related embeddings
    avg_instruction_embedding = torch.mean(torch.stack(instruction_embeddings), dim=0)
    
    # Update the merged model's embeddings for instruction-related tokens
    merged_tokenizer = merge_tokenizers(tokenizers)
    merged_instruction_tokens = [token for token in merged_tokenizer.get_vocab().keys() if "instruction" in token.lower() or "task" in token.lower()]
    merged_instruction_ids = merged_tokenizer.convert_tokens_to_ids(merged_instruction_tokens)
    
    with torch.no_grad():
        merged_model.get_input_embeddings().weight[merged_instruction_ids] = avg_instruction_embedding
    
    return merged_model

def fine_tune_on_instructions(model: torch.nn.Module, tokenizer: AutoTokenizer, instruction_dataset: List[str], num_epochs: int = 3):
    """Fine-tune the merged model on instruction-following tasks."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Prepare the dataset
    encoded_dataset = [tokenizer.encode(text, truncation=True, max_length=512) for text in instruction_dataset]
    dataset = TextDataset(encoded_dataset, tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=data_collator)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            inputs = batch["input_ids"].to(model.device)
            labels = inputs.clone()
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")
    
    return model

def evaluate_instruction_following(model: torch.nn.Module, tokenizer: AutoTokenizer, test_instructions: List[str]) -> float:
    """Evaluate the instruction-following capabilities of the merged model."""
    model.eval()
    correct_responses = 0
    
    for instruction in test_instructions:
        inputs = tokenizer.encode(instruction, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Simple heuristic: check if the response contains key instruction-related words
        if any(keyword in response.lower() for keyword in ["answer", "result", "output", "response"]):
            correct_responses += 1
    
    accuracy = correct_responses / len(test_instructions)
    return accuracy

# Example usage
def merge_and_evaluate_instruction_models(models: List[torch.nn.Module], tokenizers: List[AutoTokenizer], merge_settings: MergeSettings, instruction_dataset: List[str], test_instructions: List[str]) -> Tuple[torch.nn.Module, float]:
    merged_model, merged_tokenizer = merge_instruction_tuned_models(models, tokenizers, merge_settings)
    fine_tuned_model = fine_tune_on_instructions(merged_model, merged_tokenizer, instruction_dataset)
    accuracy = evaluate_instruction_following(fine_tuned_model, merged_tokenizer, test_instructions)
    print(f"Instruction-following accuracy: {accuracy:.2f}")
    return fine_tuned_model, accuracy
