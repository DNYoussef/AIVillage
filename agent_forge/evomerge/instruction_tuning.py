import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def is_instruction_tuned_model(model: torch.nn.Module) -> bool:
    """
    Check if a model is likely to be instruction-tuned based on its architecture
    or special tokens in its tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
    
    # Check for instruction-related special tokens
    instruction_tokens = ['<instruction>', '<system>', '<human>', '<assistant>']
    if any(token in tokenizer.special_tokens_map.values() for token in instruction_tokens):
        return True

    # Check for instruction-specific layers in the model architecture
    if any('instruction' in name for name, _ in model.named_modules()):
        return True

    return False

def preserve_instruction_prompt(tokenizer: AutoTokenizer, text: str) -> str:
    """
    Preserve instruction prompts during tokenization and generation.
    """
    instruction_tokens = ['<instruction>', '<system>', '<human>', '<assistant>']
    for token in instruction_tokens:
        if token in tokenizer.special_tokens_map.values():
            text = text.replace(token, tokenizer.special_tokens_map[token])
    return text

def generate_text_with_instruction_preservation(model: torch.nn.Module, tokenizer: AutoTokenizer, prompt: str, max_length: int = 100) -> str:
    preserved_prompt = preserve_instruction_prompt(tokenizer, prompt)
    inputs = tokenizer(preserved_prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length)
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return generated_text
