import torch
from torch import nn
from transformers import AutoModelForCausalLM


class QuietSTaRModel(nn.Module):
    """Simplified Quiet-STaR architecture with thought token generation."""

    def __init__(self, base_model: AutoModelForCausalLM) -> None:
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size
        self.start_thought = nn.Parameter(torch.randn(1, hidden_size))
        self.end_thought = nn.Parameter(torch.randn(1, hidden_size))
        self.mixing_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        generate_thoughts: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return logits and optional thought logits."""
        base_out = self.base_model(input_ids, attention_mask=attention_mask)
        if not generate_thoughts:
            return base_out.logits, None

        thought_logits = []
        for idx in range(input_ids.size(1)):
            ctx = input_ids[:, : idx + 1]
            thought_out = self.base_model(ctx)
            thought_logits.append(thought_out.logits[:, -1, :].unsqueeze(1))
        thought_logits = torch.cat(thought_logits, dim=1)
        mixed = base_out.logits + self.mixing_head(thought_logits)
        return mixed, thought_logits
