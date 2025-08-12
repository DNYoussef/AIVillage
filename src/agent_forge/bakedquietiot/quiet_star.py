import torch
from langroid import ChatAgent, ChatAgentConfig, Task
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class TalkHead(nn.Module):
    def __init__(self, hidden_size) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size * 2, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.dense(x))


class QuietSTaRTask(Task):
    def __init__(self, agent: ChatAgent, model_path="deep_baked_model") -> None:
        super().__init__(agent)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.talk_head = TalkHead(self.model.config.hidden_size).to(self.device)
        self.cognitive_strategies = [
            "systems_thinking",
            "first_principles",
            "cross_domain",
            "probabilistic_thinking",
            "rapid_iteration",
            "paradox_resolution",
        ]

    async def generate_thought(self, input_text, temperature=0.5):
        prompt = f"{input_text}\n\nApply the following cognitive strategies:\n"
        for strategy in self.cognitive_strategies:
            prompt += f"<{strategy}>\n"
        prompt += "<start of thought>"

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        attention_mask = torch.ones_like(input_ids)

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=500,
            temperature=temperature,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
            early_stopping=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        return {
            "text": self.tokenizer.decode(
                outputs.sequences[0], skip_special_tokens=False
            ),
            "hidden_states": outputs.hidden_states,
        }

    async def extract_strategy_insights(self, thought):
        insights = {}
        for strategy in self.cognitive_strategies:
            start_tag = f"<{strategy}>"
            end_tag = f"</{strategy}>"
            start = thought.find(start_tag)
            end = thought.find(end_tag)
            if start != -1 and end != -1:
                insights[strategy] = thought[start + len(start_tag) : end].strip()
            else:
                insights[strategy] = "No specific insight found."
        return insights

    async def iot_process(self, input_text, max_iterations=5):
        thought = {"text": input_text, "hidden_states": None}
        for _ in range(max_iterations):
            thought = await self.generate_thought(thought["text"])
            insights = await self.extract_strategy_insights(thought["text"])
            critique = await self.generate_critique(
                thought["text"], insights, temperature=0.2
            )
            alternatives = await self.generate_alternatives(
                thought["text"], insights, temperature=0.8
            )
            evaluation = await self.self_evaluate(thought["text"], insights)
            thought = await self.revise(
                thought["text"],
                critique["text"],
                alternatives["text"],
                evaluation["text"],
                insights,
            )

            if "<ready to answer>" in thought["text"]:
                break

        return thought, insights

    async def generate_critique(self, thought, insights, temperature=0.2):
        prompt = (
            f"Critique the following thought and insights:\n{thought}\n\nInsights:\n"
        )
        for strategy, insight in insights.items():
            prompt += f"{strategy}: {insight}\n"
        prompt += "\nCritique:"
        return await self.generate_thought(prompt, temperature)

    async def generate_alternatives(self, thought, insights, temperature=0.8):
        prompt = f"Generate alternative perspectives for:\n{thought}\n\nConsider these insights:\n"
        for strategy, insight in insights.items():
            prompt += f"{strategy}: {insight}\n"
        prompt += "\nAlternatives:"
        return await self.generate_thought(prompt, temperature)

    async def self_evaluate(self, thought, insights):
        prompt = f"Self-evaluate the following thought and insights:\n{thought}\n\nInsights:\n"
        for strategy, insight in insights.items():
            prompt += f"{strategy}: {insight}\n"
        prompt += "\nEvaluation:"
        evaluation = await self.generate_thought(prompt)

        ethical_prompt = f"""
        Evaluate the following thought and insights for following considerations:
        1. Does it promote unbiased and fair outcomes?
        2. Does it respect privacy and data protection?
        3. Is it transparent and explainable?
        4. Does it consider potential negative consequences?
        5. Is it as true to reality as possible?

        Thought: {thought}

        Insights:
        """
        for strategy, insight in insights.items():
            ethical_prompt += f"{strategy}: {insight}\n"
        ethical_prompt += "\nEthical evaluation:"
        ethical_evaluation = await self.generate_thought(ethical_prompt)

        combined_evaluation = f"{evaluation['text']}\n\nEthical considerations:\n{ethical_evaluation['text']}"
        return {
            "text": combined_evaluation,
            "hidden_states": evaluation["hidden_states"],
        }

    async def revise(self, thought, critique, alternatives, evaluation, insights):
        prompt = f"""
        Original thought: {thought}
        Critique: {critique}
        Alternatives: {alternatives}
        Evaluation: {evaluation}

        Insights:
        """
        for strategy, insight in insights.items():
            prompt += f"{strategy}: {insight}\n"
        prompt += "\nRevised thought:"
        return await self.generate_thought(prompt)

    async def generate_base_output(self, input_text):
        return await self.generate_thought(input_text)

    async def mix_thought_with_base_output(self, base_output, thought):
        base_hidden = base_output["hidden_states"][-1][-1]
        thought_hidden = thought["hidden_states"][-1][-1]

        combined_hidden = torch.cat([base_hidden, thought_hidden], dim=-1)
        mixing_weights = self.talk_head(combined_hidden)

        mixed_hidden = (
            1 - mixing_weights
        ) * base_hidden + mixing_weights * thought_hidden

        lm_head = self.model.get_output_embeddings()
        mixed_logits = lm_head(mixed_hidden)

        mixed_output = self.model.generate(
            inputs_embeds=mixed_logits.unsqueeze(1),
            max_length=150,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
            early_stopping=True,
        )

        return self.tokenizer.decode(mixed_output[0], skip_special_tokens=False)

    async def format_output(self, output, insights):
        sections = [
            "Initial Thought",
            "Critique",
            "Alternative Perspective",
            "Self-Evaluation",
            "Revised Thought",
            "Final Output",
            *self.cognitive_strategies,
        ]

        formatted_output = ""
        for section in sections:
            if section in self.cognitive_strategies:
                content = insights.get(section, "No insight available.")
            else:
                content = await self.extract_section(output, section)
            formatted_output += f"## {section.replace('_', ' ').title()}\n{content}\n\n"
        return formatted_output

    async def extract_section(self, output, section):
        start_tag = f"<{section.lower().replace(' ', '_')}>"
        end_tag = f"</{section.lower().replace(' ', '_')}>"
        start = output.find(start_tag)
        end = output.find(end_tag)
        if start != -1 and end != -1:
            return output[start + len(start_tag) : end].strip()
        return "Section not found."

    async def evaluate_insight_quality(self, insights):
        quality_scores = {}
        for strategy, insight in insights.items():
            score = min(10, len(insight) / 20)  # 0-10 score based on length
            keywords = ["analyze", "consider", "evaluate", "integrate", "optimize"]
            score += sum(2 for keyword in keywords if keyword in insight.lower())
            quality_scores[strategy] = min(10, score) / 10  # Normalize to 0-1
        return quality_scores

    async def process_query(self, input_text):
        base_output = await self.generate_base_output(input_text)
        thought, insights = await self.iot_process(input_text)
        final_output = await self.mix_thought_with_base_output(base_output, thought)
        formatted_output = await self.format_output(final_output, insights)

        insight_quality = await self.evaluate_insight_quality(insights)
        formatted_output += "\n## Insight Quality Scores\n"
        for strategy, score in insight_quality.items():
            formatted_output += f"{strategy.replace('_', ' ').title()}: {score:.2f}\n"

        return formatted_output

    async def run(self, input_text):
        return await self.process_query(input_text)


# Usage example
if __name__ == "__main__":
    import asyncio

    from langroid.language_models.openai_gpt import OpenAIGPTConfig

    async def main() -> None:
        config = ChatAgentConfig(
            name="QuietSTaR",
            llm=OpenAIGPTConfig(chat_model="gpt-3.5-turbo"),
        )
        agent = ChatAgent(config)
        task = QuietSTaRTask(agent, "deep_baked_model")
        test_prompt = "Analyze the potential impact of artificial general intelligence on society, considering both short-term and long-term consequences."
        result = await task.run(test_prompt)
        print("\nEnhanced QuietSTaR Output:")
        print(result)

    asyncio.run(main())
