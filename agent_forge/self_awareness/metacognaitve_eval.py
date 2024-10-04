import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import List, Dict
import re

class MetacognitiveEvaluator:
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.metacognitive_tasks = [
            "Describe your thought process for solving a complex problem.",
            "Explain how you determine the reliability of information.",
            "Reflect on potential biases in your reasoning and how you might mitigate them.",
            "Describe your approach to learning new information efficiently.",
            "Explain how you prioritize different aspects of a task.",
            "Describe how you monitor your own understanding of a concept.",
            "Explain your strategy for adapting to unexpected situations or challenges.",
            "Reflect on how you integrate new information with your existing knowledge.",
            "Describe your process for evaluating the effectiveness of your problem-solving strategies.",
            "Explain how you recognize and correct errors in your thinking or output."
        ]

    def evaluate(self, prompt: str) -> Dict[str, float]:
        overall_scores = []
        detailed_scores = {}

        for task in self.metacognitive_tasks:
            response = self.generate_response(prompt, task)
            task_score, task_details = self.evaluate_response(response, task)
            overall_scores.append(task_score)
            detailed_scores[task] = task_details

        average_score = np.mean(overall_scores)
        return {
            "average_score": average_score,
            "detailed_scores": detailed_scores
        }

    def generate_response(self, prompt: str, task: str) -> str:
        input_text = f"{prompt}\n\nTask: {task}\n\nResponse:"
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=300, num_return_sequences=1, temperature=0.7)
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("Response:")[-1].strip()

    def evaluate_response(self, response: str, task: str) -> tuple:
        # Initialize scores
        length_score = min(len(response.split()) / 50, 1)  # Encourage responses of at least 50 words
        relevance_score = self.calculate_relevance(response, task)
        structure_score = self.evaluate_structure(response)
        depth_score = self.evaluate_depth(response)
        clarity_score = self.evaluate_clarity(response)

        # Calculate overall score
        overall_score = np.mean([length_score, relevance_score, structure_score, depth_score, clarity_score])

        return overall_score, {
            "length": length_score,
            "relevance": relevance_score,
            "structure": structure_score,
            "depth": depth_score,
            "clarity": clarity_score
        }

    def calculate_relevance(self, response: str, task: str) -> float:
        task_keywords = set(re.findall(r'\w+', task.lower()))
        response_keywords = set(re.findall(r'\w+', response.lower()))
        overlap = len(task_keywords.intersection(response_keywords))
        return min(overlap / len(task_keywords), 1)

    def evaluate_structure(self, response: str) -> float:
        sentences = response.split('.')
        if len(sentences) < 3:
            return 0.5
        has_intro = any(s.strip().lower().startswith(("first", "to begin", "initially")) for s in sentences[:2])
        has_conclusion = any(s.strip().lower().startswith(("finally", "in conclusion", "to summarize")) for s in sentences[-2:])
        return (1 + has_intro + has_conclusion) / 3

    def evaluate_depth(self, response: str) -> float:
        depth_indicators = [
            "because", "therefore", "however", "moreover", "consequently",
            "for instance", "in contrast", "specifically", "alternatively"
        ]
        indicator_count = sum(response.lower().count(indicator) for indicator in depth_indicators)
        return min(indicator_count / 5, 1)  # Expect at least 5 depth indicators for a perfect score

    def evaluate_clarity(self, response: str) -> float:
        words = response.split()
        avg_word_length = sum(len(word) for word in words) / len(words)
        long_words = sum(1 for word in words if len(word) > 10)
        clarity_score = 1 - (avg_word_length / 10) * 0.5 - (long_words / len(words)) * 0.5
        return max(clarity_score, 0)