from dataclasses import dataclass, field
from typing import List

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


@dataclass
class Question:
    text: str
    answer: str
    difficulty: int
    domain: str


@dataclass
class CurriculumLevel:
    level: int
    difficulty: int
    organic_data: List[str] = field(default_factory=list)
    synthetic_data: List[str] = field(default_factory=list)
    rag_data: List[str] = field(default_factory=list)
    interaction_data: List[str] = field(default_factory=list)
    self_awareness_complexity: int = 1


class CurriculumGenerator:
    """Generate curriculum levels by assessing model competence."""

    def __init__(self, frontier_model: str, domain: str):
        self.domain = domain
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.frontier_model = AutoModelForCausalLM.from_pretrained(
            frontier_model
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(frontier_model)

    def _generate(self, prompt: str, max_length: int = 200) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            output = self.frontier_model.generate(
                input_ids, max_length=max_length
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def create_assessment_questions(
        self, num_questions: int = 1000
    ) -> List[Question]:
        questions = []
        for level in range(1, num_questions + 1):
            prompt = (
                f"Generate a {self.domain} question at difficulty "
                f"level {level}/1000 "
                "(1=preschool, 1000=Nobel Prize). "
                "Provide the answer after 'Answer:'"
            )
            text = self._generate(prompt)
            if "Answer:" in text:
                q, a = text.split("Answer:", 1)
            else:
                q, a = text, ""
            questions.append(
                Question(q.strip(), a.strip(), level, self.domain)
            )
        return questions

    def find_model_baseline(self, model, questions: List[Question]) -> int:
        failures = []
        for q in questions:
            input_ids = self.tokenizer.encode(q.text, return_tensors="pt").to(
                self.device
            )
            with torch.no_grad():
                out = model.generate(input_ids, max_length=200)
            ans = self.tokenizer.decode(out[0], skip_special_tokens=True)
            if ans.strip() != q.answer.strip():
                failures.append(q.difficulty)
        if not failures:
            return 1
        failure_point = min(failures)
        return max(1, failure_point - 50)

    def create_curriculum_levels(self, baseline: int) -> List[CurriculumLevel]:
        world_class = min(1000, baseline + 450)
        levels = []
        for i in range(10):
            diff = int(baseline + (i * (world_class - baseline) / 9))
            levels.append(
                CurriculumLevel(
                    level=i + 1,
                    difficulty=diff,
                    self_awareness_complexity=i + 1,
                )
            )
        return levels
