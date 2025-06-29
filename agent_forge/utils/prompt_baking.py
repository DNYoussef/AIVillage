import torch.nn as nn


class PromptBank(nn.Module):
    def __init__(self, num_prompts: int, embed_dim: int):
        super().__init__()
        self.prompts = nn.Embedding(num_prompts, embed_dim)

    def forward(self, prompt_ids):
        return self.prompts(prompt_ids)


def bake_prompts(model, prompt_bank: PromptBank):
    model.prompt_bank = prompt_bank
    for p in prompt_bank.parameters():
        p.requires_grad_(False)
