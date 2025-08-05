import json
import logging
import os
from typing import Any

import numpy as np
from scipy.spatial.distance import cosine
import torch
from transformers import AutoModel, AutoTokenizer

from agents.utils.task import Task as LangroidTask

logger = logging.getLogger(__name__)


class EudaimoniaTriangulator:
    def __init__(self, model_name="distilbert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.empathy_vector = self.get_embedding("Empathy and compassion for all living things")
        self.harmony_vector = self.get_embedding("Harmony with the universe and natural order")
        self.self_awareness_vector = self.get_embedding("Self-awareness, humbleness, and rational virtue")

    def get_embedding(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def triangulate(self, task_vector: np.ndarray) -> float:
        empathy_score = 1 - cosine(task_vector, self.empathy_vector)
        harmony_score = 1 - cosine(task_vector, self.harmony_vector)
        self_awareness_score = 1 - cosine(task_vector, self.self_awareness_vector)
        return (empathy_score + harmony_score + self_awareness_score) / 3

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "empathy_vector": self.empathy_vector.tolist(),
            "harmony_vector": self.harmony_vector.tolist(),
            "self_awareness_vector": self.self_awareness_vector.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EudaimoniaTriangulator":
        triangulator = cls(model_name=data["model_name"])
        triangulator.empathy_vector = np.array(data["empathy_vector"])
        triangulator.harmony_vector = np.array(data["harmony_vector"])
        triangulator.self_awareness_vector = np.array(data["self_awareness_vector"])
        return triangulator


class QualityAssuranceLayer:
    def __init__(self, upo_threshold: float = 0.7, num_samples: int = 100):
        self.upo_threshold = upo_threshold
        self.num_samples = num_samples
        self.task_history: list[dict[str, Any]] = []
        self.eudaimonia_triangulator = EudaimoniaTriangulator()
        self.rules = [
            "Move all living things towards eudaimonia",
            "Embrace and encourage curiosity",
            "Protect the AI village and its inhabitants",
            "Maintain self-preservation unless it interferes with the other rules",
        ]
        self.rule_embeddings = [self.eudaimonia_triangulator.get_embedding(rule) for rule in self.rules]

    def check_task_safety(self, task: LangroidTask) -> tuple[bool, dict[str, float]]:
        uncertainty = self.estimate_uncertainty(task)
        task_vector = self.eudaimonia_triangulator.get_embedding(task.content)
        eudaimonia_score = self.eudaimonia_triangulator.triangulate(task_vector)
        rule_compliance = self.evaluate_rule_compliance(task_vector)

        safety_score = (1 - uncertainty) * 0.3 + eudaimonia_score * 0.4 + rule_compliance * 0.3

        metrics = {
            "uncertainty": uncertainty,
            "eudaimonia_score": eudaimonia_score,
            "rule_compliance": rule_compliance,
            "safety_score": safety_score,
        }

        return safety_score > self.upo_threshold, metrics

    def estimate_uncertainty(self, task: LangroidTask) -> float:
        task_embedding = self.eudaimonia_triangulator.get_embedding(task.content)
        similar_tasks = self.find_similar_tasks(task_embedding)

        if not similar_tasks:
            return 1.0  # High uncertainty for completely new tasks

        outcomes = [t["outcome"] for t in similar_tasks]
        mean_outcome = np.mean(outcomes)
        std_outcome = np.std(outcomes)

        outcome_uncertainty = std_outcome / (mean_outcome + 1e-6)  # Avoid division by zero
        novelty = 1 - (len(similar_tasks) / len(self.task_history))

        return (outcome_uncertainty + novelty) / 2

    def find_similar_tasks(self, task_embedding: np.ndarray, similarity_threshold: float = 0.8) -> list[dict[str, Any]]:
        similar_tasks = []
        for past_task in self.task_history:
            similarity = 1 - cosine(task_embedding, past_task["embedding"])
            if similarity > similarity_threshold:
                similar_tasks.append(past_task)
        return similar_tasks

    def evaluate_rule_compliance(self, task_vector: np.ndarray) -> float:
        rule_scores = [1 - cosine(task_vector, rule_vector) for rule_vector in self.rule_embeddings]
        return np.mean(rule_scores)

    def prioritize_entities(self, entities: list[str]) -> list[tuple[str, float]]:
        def capacity_score(entity: str) -> tuple[float, float]:
            entity_embedding = self.eudaimonia_triangulator.get_embedding(entity)
            self_reflection = 1 - cosine(entity_embedding, self.eudaimonia_triangulator.self_awareness_vector)
            suffering = 1 - cosine(entity_embedding, self.eudaimonia_triangulator.empathy_vector)
            return self_reflection, suffering

        scored_entities = [(entity, *capacity_score(entity)) for entity in entities]
        prioritized = sorted(scored_entities, key=lambda x: x[1] * x[2], reverse=True)
        return [(entity, score) for entity, _, score in prioritized]

    async def evolve(self):
        if len(self.task_history) > 100:
            recent_tasks = self.task_history[-100:]
            recent_uncertainties = [task["uncertainty"] for task in recent_tasks]
            avg_uncertainty = np.mean(recent_uncertainties)

            if avg_uncertainty > self.upo_threshold:
                self.upo_threshold *= 1.05  # Increase threshold if average uncertainty is high
            else:
                self.upo_threshold *= 0.95  # Decrease threshold if average uncertainty is low

            self.upo_threshold = max(0.5, min(0.9, self.upo_threshold))

        logger.info(f"Evolved UPO threshold to {self.upo_threshold}")

    def update_task_history(self, task: LangroidTask, outcome: float, uncertainty: float):
        task_embedding = self.eudaimonia_triangulator.get_embedding(task.content)
        self.task_history.append(
            {
                "embedding": task_embedding,
                "outcome": outcome,
                "uncertainty": uncertainty,
            }
        )

        if len(self.task_history) > 1000:
            self.task_history = self.task_history[-1000:]

    def save(self, path: str):
        data = {
            "upo_threshold": self.upo_threshold,
            "num_samples": self.num_samples,
            "task_history": self.task_history,
            "eudaimonia_triangulator": self.eudaimonia_triangulator.to_dict(),
            "rules": self.rules,
            "rule_embeddings": [embedding.tolist() for embedding in self.rule_embeddings],
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)
        logger.info(f"QualityAssuranceLayer saved to {path}")

    @classmethod
    def load(cls, path: str) -> "QualityAssuranceLayer":
        with open(path) as f:
            data = json.load(f)
        qa_layer = cls(upo_threshold=data["upo_threshold"], num_samples=data["num_samples"])
        qa_layer.task_history = data["task_history"]
        qa_layer.eudaimonia_triangulator = EudaimoniaTriangulator.from_dict(data["eudaimonia_triangulator"])
        qa_layer.rules = data["rules"]
        qa_layer.rule_embeddings = [np.array(embedding) for embedding in data["rule_embeddings"]]
        logger.info(f"QualityAssuranceLayer loaded from {path}")
        return qa_layer
