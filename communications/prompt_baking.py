import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer

class PromptBank(nn.Module):
    """Frozen embeddings for baked-in system prompts and manifests."""

    def __init__(self, manifest: dict, tokenizer_name: str, embed_dim: int):
        super().__init__()
        prompt_text = json.dumps(manifest, sort_keys=True, indent=None)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        token_ids = self.tokenizer(prompt_text)["input_ids"]
        self.register_buffer("prompt_ids", torch.tensor(token_ids))
        self.embed = nn.Embedding(len(token_ids), embed_dim)
        nn.init.normal_(self.embed.weight, std=0.02)
        for p in self.parameters():
            p.requires_grad_(False)

    def prepend(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        """Prepend baked prompt tokens to a batch of input ids."""
        batch_size = input_ids.size(0)
        prompt_batch = self.prompt_ids.unsqueeze(0).repeat(batch_size, 1)
        return torch.cat([prompt_batch, input_ids], dim=1)
