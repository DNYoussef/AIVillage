import torch
import torch.nn.functional as F

class REINFORCEOptimizer:
    def __init__(self, model, learning_rate=1e-4):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def optimize_step(self, input_ids, attention_mask, labels):
        self.optimizer.zero_grad()

        thoughts, log_probs = self.model.generate_thoughts(input_ids, attention_mask, return_probs=True)
        outputs = self.model(input_ids, attention_mask)

        loss = F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
        reward = -loss.item()

        reinforce_loss = -log_probs.mean() * reward

        reinforce_loss.backward()
        self.optimizer.step()

        return loss.item(), reward