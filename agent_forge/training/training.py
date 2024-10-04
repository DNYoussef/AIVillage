import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

from agent_forge.training.grokfast import GrokFast

class EnhancedQuietSTaR(nn.Module):
    def __init__(self, model_path="deep_baked_model", device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.cognitive_strategies = [
            "systems_thinking", "first_principles", "cross_domain",
            "probabilistic_thinking", "rapid_iteration", "paradox_resolution"
        ]

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def generate_thoughts(self, input_ids, attention_mask, return_probs=False):
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=input_ids.size(1) + 200,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
            output_scores=return_probs,
            return_dict_in_generate=True,
        )

        thoughts = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=False)
        
        if return_probs:
            log_probs = outputs.scores[0].log_softmax(-1)
            return thoughts, log_probs
        else:
            return thoughts

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

        return loss.item(), reward, thoughts

class CognitiveTrainingPipeline:
    def __init__(self, model, train_data, val_data, num_epochs, device='cuda'):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.num_epochs = num_epochs
        self.device = device
        self.optimizer = REINFORCEOptimizer(model)
        self.grokfast = None

    def extract_thoughts(self, text):
        thoughts = re.findall(r'<start of thought>(.*?)<end of thought>', text, re.DOTALL)
        return thoughts

    def analyze_reasoning(self, thought):
        reasoning_score = 0
        for strategy in self.model.cognitive_strategies:
            if f"<{strategy}>" in thought and f"</{strategy}>" in thought:
                reasoning_score += 1
        return reasoning_score / len(self.model.cognitive_strategies)

    def score_output(self, output, correct_answer):
        if correct_answer.lower() in output.lower():
            return 1.0
        return 0.0

    def grade_insight_creativity(self, thought):
        insight_score = len(set(self.model.cognitive_strategies) & set(re.findall(r'<(\w+)>', thought)))
        creativity_score = len(re.findall(r'<alternative perspective>', thought))
        return (insight_score + creativity_score) / (len(self.model.cognitive_strategies) + 1)

    def compute_reward(self, output, correct_answer):
        thoughts = self.extract_thoughts(output)
        
        reasoning_reward = 0
        output_reward = 0
        insight_creativity_reward = 0
        
        for thought in thoughts:
            reasoning_reward += self.analyze_reasoning(thought)
            output_reward += self.score_output(thought, correct_answer)
            insight_creativity_reward += self.grade_insight_creativity(thought)
        
        num_thoughts = len(thoughts) if thoughts else 1
        
        reasoning_reward /= num_thoughts
        output_reward /= num_thoughts
        insight_creativity_reward /= num_thoughts
        
        total_reward = reasoning_reward + output_reward + insight_creativity_reward
        return total_reward

    def train(self):
        best_val_loss = float('inf')
        overfitting_detected = False
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0
            epoch_reward = 0
            for input_ids, attention_mask, labels, correct_answer in self.train_data:
                loss, reward, thoughts = self.optimizer.optimize_step(input_ids, attention_mask, labels)
                additional_reward = self.compute_reward(thoughts, correct_answer)
                epoch_loss += loss
                epoch_reward += reward + additional_reward

                if self.grokfast:
                    self.grokfast.filter_gradients()

            avg_loss = epoch_loss / len(self.train_data)
            avg_reward = epoch_reward / len(self.train_data)
            print(f"Epoch {epoch+1}/{self.num_epochs}, Avg Loss: {avg_loss:.4f}, Avg Reward: {avg_reward:.4f}")

            val_loss = self.evaluate()
            print(f"Validation Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), f"best_model_epoch_{epoch+1}.pth")
            elif val_loss > best_val_loss * 1.05:  # Early stopping condition
                if not overfitting_detected:
                    print("Overfitting detected. Activating GrokFast.")
                    self.grokfast = GrokFast(self.model, method='EMA', lamb=2.0, alpha=0.98)
                    overfitting_detected = True
                else:
                    print("Overfitting persists. Stopping training.")
                    break

        return self.model

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for input_ids, attention_mask, labels, _ in self.val_data:
                outputs = self.model(input_ids, attention_mask)
                loss = F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
                total_loss += loss.item()
        return total_loss / len(self.val_data)

model_path = "deep_baked_model"
train_data = [...]  # Your training data (input_ids, attention_mask, labels, correct_answer)
val_data = [...]  # Your validation data (input_ids, attention_mask, labels, correct_answer)
model = EnhancedQuietSTaR(model_path)
pipeline = CognitiveTrainingPipeline(model, train_data, val_data, num_epochs=50)
trained_model = pipeline.train()