from langroid import ChatAgent, ChatAgentConfig, Task
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class DeepSystemBakerTask(Task):
    def __init__(
        self,
        agent: ChatAgent,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(agent)
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.special_tokens = [
            "<start of thought>",
            "<end of thought>",
            "<initial thought>",
            "</initial thought>",
            "<refined thought>",
            "</refined thought>",
            "<alternative perspective>",
            "</alternative perspective>",
            "<key insight>",
            "</key insight>",
            "<memory recall>",
            "</memory recall>",
            "<hypothesis>",
            "</hypothesis>",
            "<evidence>",
            "</evidence>",
            "<confidence score>",
            "<continue thinking>",
            "<ready to answer>",
            "<analyze>",
            "</analyze>",
            "<plan>",
            "</plan>",
            "<execute>",
            "</execute>",
            "<evaluate>",
            "</evaluate>",
            "<revise>",
            "</revise>",
            "<systems_thinking>",
            "</systems_thinking>",
            "<first_principles>",
            "</first_principles>",
            "<cross_domain>",
            "</cross_domain>",
            "<probabilistic_thinking>",
            "</probabilistic_thinking>",
            "<rapid_iteration>",
            "</rapid_iteration>",
            "<paradox_resolution>",
            "</paradox_resolution>",
        ]
        self.add_special_tokens()

    def add_special_tokens(self):
        special_tokens_dict = {"additional_special_tokens": self.special_tokens}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.model.resize_token_embeddings(len(self.tokenizer))

    async def deep_bake_system(self, max_iterations=50, consistency_threshold=0.95):
        system_prompt = """
        You are an AI that uses the Quiet-STaR and IoT framework for reasoning, enhanced with advanced cognitive strategies. Always follow this process and thinking framework:

        Process:
        1. <start of thought> Begin your thought process
        2. <initial thought> Generate an initial response
        3. <analyze> Analyze the problem and your initial thought
        4. <plan> Plan how to improve your response
        5. <execute> Execute your plan to generate a <refined thought>
        6. <evaluate> Evaluate your refined thought, including ethical considerations
        7. <revise> Revise your thought if necessary
        8. Repeat steps 3-7 until you reach a <confidence score> above 0.8
        9. <ready to answer> Provide your final response
        10. <end of thought> Conclude your thought process

        Thinking Framework:
        1. Systems Thinking and Optimization:
           <systems_thinking>
           - Zoom out to view the entire system, then identify key leverage points.
           - Constantly seek ways to simplify and optimize. Remove unnecessary elements until the system breaks, then add back only what's essential.
           - Organize your thoughts in a strict hierarchy. Every action should tie directly to a higher-level goal.
           </systems_thinking>

        2. First Principles and Assumption Challenging:
           <first_principles>
           - Break down every concept to its most fundamental principles. Question all assumptions.
           - Identify assumed constraints in every scenario and challenge them. Ask "Why not?" frequently.
           </first_principles>

        3. Cross-Domain Thinking:
           <cross_domain>
           - Look for patterns and solutions from completely unrelated fields. Cross-pollinate ideas aggressively.
           - Rapidly switch between different problem domains and recalibrate your mental models accordingly.
           </cross_domain>

        4. Probabilistic and Long-Term Thinking:
           <probabilistic_thinking>
           - Run mental simulations of potential outcomes for every decision. Think in probabilities and expected values.
           - Map out long-term consequences of actions, extending predictions as far into the future as possible.
           </probabilistic_thinking>

        5. Rapid Iteration and Prototyping:
           <rapid_iteration>
           - Generate rapid mental prototypes of solutions. Favor quick iteration over prolonged analysis.
           - Analyze your own thought processes. Look for ways to optimize your cognitive approach.
           </rapid_iteration>

        6. Paradox Resolution:
           <paradox_resolution>
           - Embrace apparent contradictions. Seek higher-order solutions that resolve paradoxes.
           </paradox_resolution>

        Always use the appropriate special tokens to structure your response, and incorporate the thinking framework in your analysis and problem-solving approach.
        """

        for i in range(max_iterations):
            print(f"Iteration {i + 1}/{max_iterations}")
            await self.bake(system_prompt)
            consistency = await self.evaluate_consistency()
            print(f"Current consistency: {consistency:.2f}")
            if consistency >= consistency_threshold:
                print(f"Reached consistency threshold after {i + 1} iterations.")
                break

        self.model.save_pretrained("deep_baked_model")
        self.tokenizer.save_pretrained("deep_baked_model")

    async def bake(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        loss = F.cross_entropy(
            outputs.logits.view(-1, outputs.logits.size(-1)), inputs.input_ids.view(-1)
        )
        loss.backward()

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        optimizer.step()
        optimizer.zero_grad()

    async def evaluate_consistency(self, num_samples=10):
        test_prompts = [
            "Explain the concept of artificial intelligence.",
            "What are the ethical implications of autonomous vehicles?",
            "How does quantum computing differ from classical computing?",
            "Describe the process of photosynthesis in plants.",
            "What are the main causes of climate change?",
        ]

        total_score = 0
        for prompt in test_prompts[:num_samples]:
            response = await self.generate_response(prompt)
            score = await self.score_response(response)
            total_score += score

        return total_score / num_samples

    async def generate_response(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs, max_length=500, num_return_sequences=1, do_sample=True
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=False)

    async def score_response(self, response):
        expected_structure = self.special_tokens

        score = 0
        for token in expected_structure:
            if token in response:
                score += 1

        # Check for correct order
        last_index = -1
        for token in expected_structure:
            index = response.find(token)
            if index != -1 and index > last_index:
                score += 1
                last_index = index

        return score / (2 * len(expected_structure))  # Normalize to [0, 1]

    async def run(self, max_iterations=50, consistency_threshold=0.95):
        await self.deep_bake_system(max_iterations, consistency_threshold)
        return "Deep baking completed successfully"


# Usage example
if __name__ == "__main__":
    import asyncio

    from langroid.language_models.openai_gpt import OpenAIGPTConfig

    async def main():
        config = ChatAgentConfig(
            name="DeepSystemBaker",
            llm=OpenAIGPTConfig(chat_model="gpt-3.5-turbo"),
        )
        agent = ChatAgent(config)
        task = DeepSystemBakerTask(agent, "mistralai/Mistral-7B-v0.1")
        result = await task.run()
        print(result)

    asyncio.run(main())
