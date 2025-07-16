import random

from langroid import ChatAgent, ChatAgentConfig, Task
from langroid.language_models.openai_gpt import OpenAIGPTConfig
import torch
from torch import nn, optim
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

# shared training state from geometry feedback loop
state = dict(G={"ID_nl": 0.0}, pre_grok=False)

try:
    from grokfast import GrokFastTask
except ImportError:
    # Stub implementation for missing grokfast dependency
    class GrokFastTask:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            return {"status": "stub", "message": "GrokFast not available"}


try:
    from sleep_and_dream import DreamNet, SleepNet
except ImportError:
    # Stub implementations for missing sleep_and_dream dependency
    class SleepNet:
        def __init__(self, *args, **kwargs):
            pass

    class DreamNet:
        def __init__(self, *args, **kwargs):
            pass


from agent_forge.model_compression.bitlinearization import quantize_weights


class CodingTask:
    def __init__(self, description: str, difficulty: int):
        self.description = description
        self.difficulty = difficulty


class CodingTaskGenerationTask(Task):
    async def generate_coding_tasks(
        self, num_tasks: int, avg_difficulty: int
    ) -> list[CodingTask]:
        tasks = []
        for _ in range(num_tasks):
            prompt = f"Create a coding task with difficulty level {avg_difficulty}/100. Include a clear problem description."
            response = await self.agent.llm_response(prompt)
            task_description = response.content
            tasks.append(
                CodingTask(description=task_description, difficulty=avg_difficulty)
            )
        return tasks


class SelfModelingTask(Task):
    def __init__(
        self,
        agent: ChatAgent,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(agent)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        self.sleep_net = SleepNet(
            input_size=self.model.config.hidden_size,
            output_size=self.model.config.hidden_size,
            num_sleep_blocks=3,
        )
        self.dream_net = DreamNet(
            input_size=self.model.config.hidden_size,
            output_size=self.model.config.hidden_size,
            num_dream_blocks=3,
        )
        self.grokfast_task = GrokFastTask(agent, self.model)
        self.avg_difficulty = 50  # Start with an average difficulty of 50

        # Quantize the initial model weights
        self.quantize_model()

        # predictor for self-modeling loss
        h = self.model.config.hidden_size
        self.hidden_pred = nn.Sequential(
            nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h)
        ).to(self.device)
        self.beta = 0.1

    def quantize_model(self):
        with torch.no_grad():
            for param in self.model.parameters():
                param.data = quantize_weights(param.data)

    async def generate_text(self, prompt: str, temperature: float) -> str:
        meta = f"<geom idnl={state['G']['ID_nl']:.2f} temp={temperature:.2f}/>"
        prompt = meta + prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=100,
                num_return_sequences=1,
                do_sample=True,
                temperature=temperature,
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def mask_and_fill(
        self, text: str, num_masks: int
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        input_ids = inputs.input_ids.to(self.device)
        labels = input_ids.clone()

        mask_candidates = [
            i
            for i in range(len(input_ids[0]))
            if input_ids[0][i]
            not in [
                self.tokenizer.cls_token_id,
                self.tokenizer.sep_token_id,
                self.tokenizer.pad_token_id,
            ]
        ]

        mask_indices = random.sample(
            mask_candidates, min(num_masks, len(mask_candidates))
        )

        for idx in mask_indices:
            input_ids[0][idx] = self.tokenizer.mask_token_id

        return input_ids, labels, mask_indices

    async def train_step(
        self, input_ids: torch.Tensor, labels: torch.Tensor, mask_indices: list[int]
    ) -> tuple[float, float]:
        outputs = self.model(
            input_ids=input_ids, labels=labels, output_hidden_states=True
        )
        pred_hidden = self.hidden_pred(outputs.hidden_states[-1].detach())
        L_self = torch.nn.functional.mse_loss(pred_hidden, outputs.hidden_states[-1])
        loss = outputs.loss + self.beta * L_self
        loss.backward()

        # Quantize gradients
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.data = quantize_weights(param.grad.data)

        self.optimizer.step(amplify=state["pre_grok"])

        # Quantize updated weights
        with torch.no_grad():
            for param in self.model.parameters():
                param.data = quantize_weights(param.data)

        self.optimizer.zero_grad()

        # Calculate accuracy of masked token predictions
        predictions = torch.argmax(outputs.logits, dim=-1)
        correct_predictions = sum(
            predictions[0][idx] == labels[0][idx] for idx in mask_indices
        )
        accuracy = correct_predictions / len(mask_indices)

        return loss.item(), accuracy

    async def self_modeling_cycle(self, curriculum_level: int, num_cycles: int = 100):
        task_generation = CodingTaskGenerationTask(self.agent)
        tasks = await task_generation.generate_coding_tasks(1000, self.avg_difficulty)

        temperature_ranges = [
            (0.0, 0.05),
            (0.2, 0.3),
            (0.45, 0.55),
            (0.7, 0.8),
            (0.95, 1.0),
        ]

        delta = 0.1 * curriculum_level
        temperature_ranges = [
            (max(0, r[0] - delta), min(1, r[1] + delta)) for r in temperature_ranges
        ]

        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-5)

        base_masks = 1
        masks_per_level = 1
        num_masks = base_masks + (curriculum_level - 1) * masks_per_level

        for cycle in tqdm(
            range(num_cycles), desc=f"Self-modeling cycle (Level {curriculum_level})"
        ):
            for task in tasks:
                for temp_range in temperature_ranges:
                    temperature = random.uniform(*temp_range)
                    original_prompt = f"You are an AI model solving a coding task. The task is: {task.description}"
                    generated_text = await self.generate_text(
                        original_prompt, temperature
                    )

                    self_modeling_prompt = f"""I am an AI model engaging in self-modeling.
                    In the past, I generated the following text based on this coding task: "{task.description}"
                    Generated text: "{generated_text}"
                    Now, I will try to predict my own masked tokens to understand my thought process better."""

                    input_ids, labels, mask_indices = self.mask_and_fill(
                        generated_text, num_masks
                    )

                    self_modeling_input = self.tokenizer.encode(
                        self_modeling_prompt, return_tensors="pt"
                    ).to(self.device)
                    self.model(
                        input_ids=self_modeling_input
                    )  # Inform the model about the self-modeling task

                    loss, accuracy = await self.train_step(
                        input_ids, labels, mask_indices
                    )

                    print(
                        f"Cycle {cycle}, Task Difficulty {task.difficulty}, Temperature {temperature:.2f}, Loss: {loss:.4f}, Accuracy: {accuracy:.2f}"
                    )

                    if cycle > 50 and loss < 0.01:  # Simple overfitting detection
                        await self.grokfast_task.filter_gradients()

                    if cycle % 10 == 0:
                        with torch.no_grad():
                            hidden_states = self.model.base_model.encoder(
                                input_ids
                            ).last_hidden_state
                            sleep_output = self.sleep_net(hidden_states)
                            dream_output = self.dream_net(sleep_output)
                            update = 0.01 * dream_output.mean(dim=1)
                            self.model.base_model.encoder.embed_tokens.weight.data += (
                                quantize_weights(update)
                            )

    async def evolve_across_curriculum(self, num_levels: int = 10):
        for level in range(1, num_levels + 1):
            print(f"Starting curriculum level {level}")
            await self.self_modeling_cycle(curriculum_level=level)

            eval_loader = [
                (torch.randint(0, 100, (4,)), torch.ones(4)) for _ in range(2)
            ]
            eval_score = await self.evaluate_model(eval_loader)
            print(f"Evaluation score after level {level}: {eval_score:.4f}")

            self.avg_difficulty = max(
                1, min(100, int(self.avg_difficulty + (eval_score - 0.5) * 10))
            )

            torch.save(
                self.model.state_dict(), f"self_model_checkpoint_level_{level}.pth"
            )

    async def evaluate_model(self, val_loader) -> float:
        """Return the validation perplexity for the current model."""
        from agent_forge.evaluation import evaluator

        eval_data = []
        for batch in val_loader:
            txt = batch["txt"].to(self.device)
            attn = torch.ones_like(txt)
            eval_data.append((txt, attn, txt))

        class EvalWrapper(nn.Module):
            def __init__(self, model, tokenizer):
                super().__init__()
                self.model = model
                self.tokenizer = tokenizer

            def forward(self, input_ids, attention_mask=None, labels=None):
                return self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

            def generate_thoughts(self, inputs, attention_mask):
                with torch.no_grad():
                    out = self.model.generate(
                        inputs,
                        attention_mask=attention_mask,
                        max_length=inputs.size(1) + 1,
                    )
                return self.tokenizer.decode(out[0], skip_special_tokens=True)

        wrapped = EvalWrapper(self.model, self.tokenizer)
        metrics = evaluator.evaluate_model(wrapped, eval_data)
        return float(metrics["perplexity"])

    async def run(self):
        await self.evolve_across_curriculum()
        return "Self-modeling completed successfully"


# Usage example
if __name__ == "__main__":
    import asyncio

    async def main():
        config = ChatAgentConfig(
            name="SelfModelingAgent",
            llm=OpenAIGPTConfig(chat_model="gpt-3.5-turbo"),
        )
        agent = ChatAgent(config)
        model_name = "bert-base-uncased"  # You can change this to any suitable model
        task = SelfModelingTask(agent, model_name)
        result = await task.run()
        print(result)

    asyncio.run(main())
