import torch
import logging
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from .config import MergeSettings
from .utils import EvoMergeException

logger = logging.getLogger(__name__)

def is_instruction_tuned_model(model: torch.nn.Module) -> bool:
    """
    Detect if a model is instruction-tuned based on its architecture and special tokens.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
        
        # Check for instruction-related special tokens
        instruction_tokens = ['<instruction>', '<system>', '<human>', '<assistant>', '<|im_start|>', '<|im_end|>']
        if any(token in tokenizer.special_tokens_map.values() for token in instruction_tokens):
            return True

        # Check for instruction-specific layers in the model architecture
        if any('instruction' in name.lower() for name, _ in model.named_modules()):
            return True

        return False
    except Exception as e:
        logger.warning(f"Error checking if model is instruction-tuned: {str(e)}")
        return False

def merge_instruction_tuned_models(models: List[torch.nn.Module], merge_settings: MergeSettings) -> torch.nn.Module:
    """
    Merge instruction-tuned models while preserving their instruction-following capabilities.
    """
    logger.info("Merging instruction-tuned models")
    
    if len(models) < 2:
        raise EvoMergeException("At least two models are required for merging")

    base_model = models[0]
    merged_state_dict = base_model.state_dict()

    try:
        # Merge model parameters
        for key in merged_state_dict.keys():
            if "lm_head" in key or "embed_tokens" in key:
                # Preserve instruction-tuning in output layer and embeddings
                continue
            
            merged_weights = torch.stack([model.state_dict()[key] for model in models])
            if merge_settings.merge_method == "mean":
                merged_state_dict[key] = torch.mean(merged_weights, dim=0)
            elif merge_settings.merge_method == "max":
                merged_state_dict[key] = torch.max(merged_weights, dim=0)[0]
            else:
                raise EvoMergeException(f"Unsupported merge method for instruction-tuned models: {merge_settings.merge_method}")

        # Special handling for instruction-specific layers
        instruction_layer_keys = [key for key in merged_state_dict.keys() if 'instruction' in key.lower()]
        for key in instruction_layer_keys:
            instruction_weights = torch.stack([model.state_dict()[key] for model in models])
            merged_state_dict[key] = torch.max(instruction_weights, dim=0)[0]

        # Create a new model with the merged parameters
        merged_model = type(base_model)(base_model.config)
        merged_model.load_state_dict(merged_state_dict)

        # Merge tokenizers
        merged_tokenizer = merge_instruction_tokenizers([AutoTokenizer.from_pretrained(model.config._name_or_path) for model in models])

        # Save the merged tokenizer
        merged_tokenizer.save_pretrained(merge_settings.custom_dir)

        logger.info("Successfully merged instruction-tuned models")
        return merged_model

    except Exception as e:
        logger.error(f"Error merging instruction-tuned models: {str(e)}")
        raise EvoMergeException(f"Failed to merge instruction-tuned models: {str(e)}")

def merge_instruction_tokenizers(tokenizers: List[AutoTokenizer]) -> AutoTokenizer:
    """
    Merge tokenizers from instruction-tuned models.
    """
    base_tokenizer = tokenizers[0]
    merged_vocab = base_tokenizer.get_vocab()
    merged_special_tokens = base_tokenizer.special_tokens_map

    for tokenizer in tokenizers[1:]:
        # Merge vocabularies
        for token, index in tokenizer.get_vocab().items():
            if token not in merged_vocab:
                merged_vocab[token] = len(merged_vocab)

        # Merge special tokens
        for key, value in tokenizer.special_tokens_map.items():
            if key not in merged_special_tokens:
                merged_special_tokens[key] = value

    # Create a new tokenizer with the merged vocabulary and special tokens
    merged_tokenizer = type(base_tokenizer)(
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
    )
    merged_tokenizer.vocab = merged_vocab
    merged_tokenizer.add_special_tokens(merged_special_tokens)

    return merged_tokenizer

def generate_text_with_instruction_preservation(model: torch.nn.Module, tokenizer: AutoTokenizer, prompt: str, max_length: int = 100) -> str:
    """
    Generate text while preserving instruction-related special tokens.
    """
    instruction_tokens = ['<instruction>', '<system>', '<human>', '<assistant>', '<|im_start|>', '<|im_end|>']
    for token in instruction_tokens:
        if token in tokenizer.special_tokens_map.values():
            prompt = prompt.replace(token, tokenizer.special_tokens_map[token])

    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length)
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return generated_text

# Add any other necessary functions for instruction-tuned model handling here
