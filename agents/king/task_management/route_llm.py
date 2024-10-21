import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from typing import List, Dict, Tuple
import logging
from functools import lru_cache
import networkx as nx
from agents.king.planning.unified_planning_and_decision import GraphManager
from agents.king.task_management.incentive_model import IncentiveModel  # Assuming the path
import asyncio

logger = logging.getLogger(__name__)

class AgentRouter(nn.Module):
    def __init__(self, model_name='bert-base-uncased', cache_size=1000, confidence_threshold=0.7):
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
    def _cached_tokenize(self, task_description: str) -> Dict[str, torch.Tensor]:
        return self.tokenizer(task_description, return_tensors='pt', padding=True, truncation=True)

    async def route(self, task_descriptions: List[str]) -> List[Tuple[str, float]]:
        self.eval()
        inputs = self.tokenizer(task_descriptions, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            logits = self(inputs['input_ids'], inputs['attention_mask'])
        probabilities = torch.softmax(logits, dim=1)
        max_probs, predictions = torch.max(probabilities, dim=1)
        
        results = []
        for pred, prob in zip(predictions, max_probs):
            agent = self.agent_mapping.get(pred.item(), 'unknown')
            confidence = prob.item()
            if confidence < self.confidence_threshold:
                agent = 'undecided'
            results.append((agent, confidence))
        
        return results

    async def train_model(self, preference_data: List[Dict], num_epochs=3, batch_size=32):
        self.train()
        train_dataloader = self._create_dataloader(preference_data, batch_size)
        total_steps = len(train_dataloader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        for epoch in range(num_epochs):
            total_loss = 0
            for batch in train_dataloader:
                inputs = self.tokenizer(batch['tasks'], return_tensors='pt', padding=True, truncation=True)
                labels = torch.tensor(batch['labels'])

                self.optimizer.zero_grad()
                outputs = self(inputs['input_ids'], inputs['attention_mask'])
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    def _create_dataloader(self, preference_data: List[Dict], batch_size: int):
        tasks = [data['task'] for data in preference_data]
        labels = [self._get_label(data['assigned_agent']) for data in preference_data]
        return [{'tasks': tasks[i:i+batch_size], 'labels': labels[i:i+batch_size]} 
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
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'agent_mapping': self.agent_mapping,
            'confidence_threshold': self.confidence_threshold,
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.agent_mapping = checkpoint['agent_mapping']
        self.confidence_threshold = checkpoint['confidence_threshold']
        self.initialize_classifier(len(self.agent_mapping))
        if checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Model loaded from {path}")

    async def update_agent_list(self, agent_list: List[str]):
        for agent in agent_list:
            if agent not in self.agent_mapping.values():
                new_label = len(self.agent_mapping)
                self.agent_mapping[new_label] = agent
        self.initialize_classifier(len(self.agent_mapping))
        logger.info(f"Updated agent list. Total agents: {len(self.agent_mapping)}")


class Router:
    def __init__(self, graph_manager: GraphManager, incentive_model: IncentiveModel):
        self.route_llm = RouteLLM()
        self.graph_manager = graph_manager
        self.incentive_model = incentive_model

    async def route_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        try:
            logger.info(f"Routing task: {task}")
            task_id = task.get('id', f"task_{random.randint(1000,9999)}")
            agent_nodes = [n for n, d in self.graph_manager.G.nodes(data=True) if d['type'] == 'agent']
            task_node = task_id
            self.graph_manager.add_task_node(task_node, {"description": task.get('description', '')})
            
            # Calculate edge weights based on IncentiveModel
            for agent in agent_nodes:
                incentive = self.incentive_model.calculate_incentive(agent, task_node)
                self.graph_manager.G.add_edge(agent, task_node, weight=incentive)
                logger.info(f"Added edge from {agent} to {task_node} with weight {incentive}")
            
            # Find optimal path for task routing
            best_agent, total_weight = self.find_optimal_path(task_node)
            confidence = 1 / (1 + math.exp(-total_weight))  # Example confidence calculation
            
            if confidence < self.graph_manager.G[best_agent][task_node]['weight']:
                assigned_agent = 'undecided'
            else:
                assigned_agent = best_agent
            
            result = {
                "task_id": task_id,
                "assigned_agent": assigned_agent,
                "confidence": confidence
            }
            
            # Update graph with task assignment
            self.graph_manager.G.add_edge(best_agent, task_node, status='assigned')
            logger.info(f"Task {task_id} assigned to {assigned_agent} with confidence {confidence}")
            
            return result
        except Exception as e:
            logger.error(f"Error routing task: {str(e)}", exc_info=True)
            raise AIVillageException(f"Error routing task: {str(e)}")

    def find_optimal_path(self, task_node: str) -> Tuple[str, float]:
        try:
            # Find the agent with the highest edge weight to the task_node
            agent_weights = self.graph_manager.G.edges(data=True)
            agent_weights_filtered = [(u, d['weight']) for u, v, d in agent_weights if v == task_node]
            if not agent_weights_filtered:
                return ('undecided', 0.0)
            best_agent, best_weight = max(agent_weights_filtered, key=lambda x: x[1])
            return (best_agent, best_weight)
        except Exception as e:
            logger.error(f"Error finding optimal path: {str(e)}", exc_info=True)
            return ('undecided', 0.0)

    async def calculate_edge_weight(self, agent_id: str, task_id: str) -> float:
        try:
            incentive = self.incentive_model.calculate_incentive(agent_id, task_id)
            return incentive
        except Exception as e:
            logger.error(f"Error calculating edge weight: {str(e)}", exc_info=True)
            return 1.0  # Default weight

    def save_models(self, path: str):
        try:
            logger.info(f"Saving router models to {path}")
            asyncio.create_task(self.route_llm.save_model(path))
        except Exception as e:
            logger.error(f"Error saving router models: {str(e)}", exc_info=True)
            raise AIVillageException(f"Error saving router models: {str(e)}")

    async def load_models(self, path: str):
        try:
            logger.info(f"Loading router models from {path}")
            await self.route_llm.load_model(path)
        except Exception as e:
            logger.error(f"Error loading router models: {str(e)}", exc_info=True)
            raise AIVillageException(f"Error loading router models: {str(e)}")

    async def introspect(self) -> Dict[str, Any]:
        return {
            "type": "Router",
            "description": "Routes tasks to appropriate agents or components",
            "route_llm_info": str(self.route_llm),
            "graph_manager_nodes": self.graph_manager.G.number_of_nodes(),
            "graph_manager_edges": self.graph_manager.G.number_of_edges()
        }
