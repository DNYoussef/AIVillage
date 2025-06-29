from __future__ import annotations
import torch


def inject_morality(model, tokenizer, prompt: str):
    """Prepend baked morality anchor embeddings to prompt."""
    anchor = model.prompt_bank.get("morality_v1_emb")
    if anchor is None:
        return tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    tok_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    base = model.get_input_embeddings()(tok_ids)
    embeds = torch.cat([anchor.unsqueeze(0), base], dim=1)
    return embeds
