import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class QuietStarModel(nn.Module):
    def __init__(self, model_path, thought_config):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.thought_config = thought_config
        config = self.model.config
        self.thought_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.mixing_head = nn.Linear(config.hidden_size * 2, 1)
        self.layer_norm = nn.LayerNorm(config.hidden_size)

    def generate_thoughts(self, hidden_states, attention_mask, return_probs=False):
        thoughts = []
        log_probs = []

        for i in range(self.thought_config["num_thoughts"]):
            thought_input = torch.cat([hidden_states, self.thought_embeddings(thoughts[-1] if thoughts else torch.zeros_like(hidden_states[:, :1]))], dim=1)
            thought_output = self.model(inputs_embeds=thought_input, attention_mask=attention_mask, return_dict=True)
            thought = thought_output.last_hidden_state[:, -1:]
            thoughts.append(thought)

            if return_probs:
                log_probs.append(F.log_softmax(self.model.lm_head(thought), dim=-1))

        thoughts = torch.cat(thoughts, dim=1)
        if return_probs:
            log_probs = torch.cat(log_probs, dim=1)
            return thoughts, log_probs
        return thoughts

    def mix_predictions(self, base_hidden, thought_hidden):
        combined = torch.cat([base_hidden, thought_hidden], dim=-1)
        mixing_weight = torch.sigmoid(self.mixing_head(combined))
        mixed = self.layer_norm(mixing_weight * thought_hidden + (1 - mixing_weight) * base_hidden)
        return mixed

    def forward(self, input_ids, attention_mask=None, labels=None):
        base_output = self.model(input_ids, attention_mask=attention_mask, return_dict=True)
        thoughts = self.generate_thoughts(base_output.last_hidden_state, attention_mask)
        mixed_hidden = self.mix_predictions(base_output.last_hidden_state, thoughts)
        logits = self.model.lm_head(mixed_hidden)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return QuietStarOutput(loss=loss, logits=logits, hidden_states=mixed_hidden, thoughts=thoughts)

class QuietStarOutput:
    def __init__(self, loss, logits, hidden_states, thoughts):
        self.loss = loss
        self.logits = logits
        self.hidden_states = hidden_states
        self.thoughts = thoughts
