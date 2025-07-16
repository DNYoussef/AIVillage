from functools import lru_cache
import logging

import torch
from torch import nn, optim
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)

class AgentRouter(nn.Module):
    def __init__(self, model_name="bert-base-uncased", cache_size=1000, confidence_threshold=0.7):
        super(AgentRouter, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.cache_size = cache_size
        self.confidence_threshold = confidence_threshold
        self.agent_mapping = {}  # Will be populated dynamically
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.classifier = None


    def initialize_classifier(self, num_agents):
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_agents)
        self.optimizer = optim.AdamW(self.parameters(), lr=2e-5)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.classifier(pooled_output)

    @lru_cache(maxsize=1000)
    def _cached_tokenize(self, task_description: str) -> dict[str, torch.Tensor]:
        return self.tokenizer(task_description, return_tensors="pt", padding=True, truncation=True)

    async def route(self, task_descriptions: list[str]) -> list[tuple[str, float]]:
        self.eval()
        inputs = self.tokenizer(task_descriptions, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            logits = self(inputs["input_ids"], inputs["attention_mask"])
        probabilities = torch.softmax(logits, dim=1)
        max_probs, predictions = torch.max(probabilities, dim=1)

        results = []
        for pred, prob in zip(predictions, max_probs, strict=False):
            agent = self.agent_mapping.get(pred.item(), "unknown")
            confidence = prob.item()
            if confidence < self.confidence_threshold:
                agent = "undecided"
            results.append((agent, confidence))

        return results

    async def train_model(self, preference_data: list[dict], num_epochs=3, batch_size=32):
        self.train()
        train_dataloader = self._create_dataloader(preference_data, batch_size)
        total_steps = len(train_dataloader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        for epoch in range(num_epochs):
            total_loss = 0
            for batch in train_dataloader:
                inputs = self.tokenizer(batch["tasks"], return_tensors="pt", padding=True, truncation=True)
                labels = torch.tensor(batch["labels"])

                self.optimizer.zero_grad()
                outputs = self(inputs["input_ids"], inputs["attention_mask"])
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    def _create_dataloader(self, preference_data: list[dict], batch_size: int):
        tasks = [data["task"] for data in preference_data]
        labels = [self._get_label(data["assigned_agent"]) for data in preference_data]
        return [{"tasks": tasks[i:i+batch_size], "labels": labels[i:i+batch_size]}
                for i in range(0, len(tasks), batch_size)]

    def _get_label(self, agent: str) -> int:
        if agent not in self.agent_mapping.values():
            new_label = len(self.agent_mapping)
            self.agent_mapping[new_label] = agent
            if self.classifier is None or self.classifier.out_features != len(self.agent_mapping):
                self.initialize_classifier(len(self.agent_mapping))
        return list(self.agent_mapping.keys())[list(self.agent_mapping.values()).index(agent)]

    def save(self, path: str):
        torch.save({
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "agent_mapping": self.agent_mapping,
            "confidence_threshold": self.confidence_threshold,
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.agent_mapping = checkpoint["agent_mapping"]
        self.confidence_threshold = checkpoint["confidence_threshold"]
        self.initialize_classifier(len(self.agent_mapping))
        if checkpoint["optimizer_state_dict"]:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"Model loaded from {path}")

    async def update_agent_list(self, agent_list: list[str]):
        for agent in agent_list:
            if agent not in self.agent_mapping.values():
                new_label = len(self.agent_mapping)
                self.agent_mapping[new_label] = agent
        self.initialize_classifier(len(self.agent_mapping))
        logger.info(f"Updated agent list. Total agents: {len(self.agent_mapping)}")

import logging
from typing import Any

from ..utils.exceptions import AIVillageException
from .route_llm import RouteLLM

logger = logging.getLogger(__name__)

class Router:
    def __init__(self):
        self.route_llm = RouteLLM()

    async def route_task(self, task: dict[str, Any]) -> dict[str, Any]:
        try:
            logger.info(f"Routing task: {task}")
            routed_task = await self.route_llm.route_task(task)
            return routed_task
        except Exception as e:
            logger.error(f"Error routing task: {e!s}", exc_info=True)
            raise AIVillageException(f"Error routing task: {e!s}")

    async def update_model(self, task: dict[str, Any], result: Any):
        try:
            logger.info(f"Updating router model with task result: {result}")
            await self.route_llm.update_model(task, result)
        except Exception as e:
            logger.error(f"Error updating router model: {e!s}", exc_info=True)
            raise AIVillageException(f"Error updating router model: {e!s}")

    async def save_models(self, path: str):
        try:
            logger.info(f"Saving router models to {path}")
            await self.route_llm.save_model(path)
        except Exception as e:
            logger.error(f"Error saving router models: {e!s}", exc_info=True)
            raise AIVillageException(f"Error saving router models: {e!s}")

    async def load_models(self, path: str):
        try:
            logger.info(f"Loading router models from {path}")
            await self.route_llm.load_model(path)
        except Exception as e:
            logger.error(f"Error loading router models: {e!s}", exc_info=True)
            raise AIVillageException(f"Error loading router models: {e!s}")

    async def introspect(self) -> dict[str, Any]:
        return {
            "type": "Router",
            "description": "Routes tasks to appropriate agents or components",
            "route_llm_info": str(self.route_llm)
        }
