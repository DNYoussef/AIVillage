"""Tree of Thoughts reasoning technique implementation."""

from typing import Dict, Any, List, Optional, Set, Tuple
import json
from dataclasses import dataclass
import asyncio
from uuid import uuid4
import networkx as nx
from langroid.agent.chat_agent import ChatAgent

from .base import AgentTechnique, TechniqueResult

@dataclass
class ThoughtNode:
    """A node in the tree of thoughts."""
    id: str
    content: str
    reasoning: str
    evaluation: float
    parent_id: Optional[str]
    children_ids: Set[str]
    depth: int
    is_terminal: bool = False

class TreeOfThoughtsTechnique(AgentTechnique):
    """
    Implementation of Tree of Thoughts reasoning technique.
    
    This technique creates a tree-like search of multiple reasoning paths,
    evaluating different branches and selecting the most promising paths.
    It uses beam search to efficiently explore the solution space.
    """
    
    def __init__(self):
        super().__init__(
            thought="Tree-of-Thoughts creates a tree-like search of multiple reasoning paths, "
                   "improving search and planning capabilities for complex problem-solving.",
            name="Tree-of-Thoughts",
            code=self.__class__.__module__
        )
        self.max_depth = 3
        self.beam_width = 3
        self.thought_tree: Dict[str, ThoughtNode] = {}
        self.root_id: Optional[str] = None
    
    def get_prompt(self, task: str, context: Optional[str] = None) -> str:
        """Create the Tree of Thoughts prompt."""
        if context:
            return f"""
            Consider the following task and context:

            Task: {task}
            Current thought: {context}

            Generate {self.beam_width} different ways to proceed with this reasoning.
            For each approach:
            1. Explain your reasoning
            2. Provide a specific next step or conclusion
            3. Rate your confidence (0-1) in this direction

            Format each approach as:
            Approach X:
            Reasoning: [your reasoning]
            Next Step: [specific next step or conclusion]
            Confidence: [0-1]
            """
        else:
            return f"""
            Consider the following task:

            Task: {task}

            Generate {self.beam_width} different initial approaches to solve this task.
            For each approach:
            1. Explain your reasoning
            2. Provide a specific first step
            3. Rate your confidence (0-1) in this direction

            Format each approach as:
            Approach X:
            Reasoning: [your reasoning]
            First Step: [specific first step]
            Confidence: [0-1]
            """

    async def apply(self, agent: ChatAgent, task: str) -> TechniqueResult:
        """
        Apply Tree of Thoughts reasoning to a task.
        
        Args:
            agent: ChatAgent instance to use for reasoning
            task: Task description or problem to solve
            
        Returns:
            TechniqueResult containing the reasoning process and result
        """
        # Initialize tree
        self.thought_tree = {}
        self.root_id = str(uuid4())
        
        # Create root node
        root_node = ThoughtNode(
            id=self.root_id,
            content=task,
            reasoning="Initial problem statement",
            evaluation=1.0,
            parent_id=None,
            children_ids=set(),
            depth=0
        )
        self.thought_tree[self.root_id] = root_node
        
        # Perform beam search through thought space
        current_nodes = [self.root_id]
        for depth in range(self.max_depth):
            next_nodes = await self._expand_nodes(agent, task, current_nodes)
            if not next_nodes:
                break
            current_nodes = await self._select_best_nodes(agent, next_nodes)
        
        # Find best path
        best_path = await self._find_best_path(agent)
        result = await self._generate_final_answer(agent, task, best_path)
        
        return result

    async def _expand_nodes(
        self,
        agent: ChatAgent,
        task: str,
        node_ids: List[str]
    ) -> List[str]:
        """Expand the given nodes to the next level of the tree."""
        new_node_ids = []
        
        for node_id in node_ids:
            node = self.thought_tree[node_id]
            
            # Get next thoughts
            prompt = self.get_prompt(task, node.content)
            response = await agent.llm_response(prompt)
            thoughts = self._parse_thoughts(response.content)
            
            # Create child nodes
            for thought in thoughts:
                child_id = str(uuid4())
                child_node = ThoughtNode(
                    id=child_id,
                    content=thought['next_step'],
                    reasoning=thought['reasoning'],
                    evaluation=thought['confidence'],
                    parent_id=node_id,
                    children_ids=set(),
                    depth=node.depth + 1
                )
                self.thought_tree[child_id] = child_node
                node.children_ids.add(child_id)
                new_node_ids.append(child_id)
        
        return new_node_ids

    async def _select_best_nodes(
        self,
        agent: ChatAgent,
        node_ids: List[str]
    ) -> List[str]:
        """Select the best nodes for further expansion using beam search."""
        if len(node_ids) <= self.beam_width:
            return node_ids
        
        # Sort nodes by evaluation score
        sorted_nodes = sorted(
            node_ids,
            key=lambda x: self.thought_tree[x].evaluation,
            reverse=True
        )
        
        return sorted_nodes[:self.beam_width]

    async def _find_best_path(self, agent: ChatAgent) -> List[str]:
        """Find the best path from root to leaf."""
        # Start from leaf nodes
        leaf_nodes = [
            node_id for node_id, node in self.thought_tree.items()
            if not node.children_ids
        ]
        
        # Find path with highest cumulative evaluation
        best_path = []
        best_score = float('-inf')
        
        for leaf_id in leaf_nodes:
            path = []
            current_id = leaf_id
            score = 0
            
            while current_id is not None:
                node = self.thought_tree[current_id]
                path.append(current_id)
                score += node.evaluation
                current_id = node.parent_id
            
            path.reverse()
            path_score = score / len(path)  # Average score along path
            
            if path_score > best_score:
                best_score = path_score
                best_path = path
        
        return best_path

    async def _generate_final_answer(
        self,
        agent: ChatAgent,
        task: str,
        path: List[str]
    ) -> TechniqueResult:
        """Generate final answer based on the best reasoning path."""
        # Construct reasoning chain
        reasoning_chain = []
        for node_id in path:
            node = self.thought_tree[node_id]
            reasoning_chain.append({
                'content': node.content,
                'reasoning': node.reasoning,
                'evaluation': node.evaluation
            })
        
        # Generate final answer
        synthesis_prompt = f"""
        Based on this chain of reasoning:
        {json.dumps(reasoning_chain, indent=2)}

        Provide a final answer to the original task:
        {task}

        Format your response as:
        Reasoning: [synthesis of the reasoning chain]
        Answer: [final answer]
        Confidence: [0-1]
        """
        
        response = await agent.llm_response(synthesis_prompt)
        final_result = self._parse_final_answer(response.content)
        
        return TechniqueResult(
            thought=final_result['reasoning'],
            result=final_result['answer'],
            confidence=final_result['confidence'],
            metadata={
                'reasoning_chain': reasoning_chain,
                'tree_structure': self._get_tree_structure()
            }
        )

    def _parse_thoughts(self, response: str) -> List[Dict[str, Any]]:
        """Parse thought generations from model response."""
        thoughts = []
        current_thought = {}
        
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('Approach '):
                if current_thought:
                    thoughts.append(current_thought)
                current_thought = {}
            elif line.startswith('Reasoning: '):
                current_thought['reasoning'] = line[len('Reasoning: '):].strip()
            elif line.startswith(('Next Step: ', 'First Step: ')):
                current_thought['next_step'] = line.split(': ', 1)[1].strip()
            elif line.startswith('Confidence: '):
                try:
                    current_thought['confidence'] = float(line[len('Confidence: '):].strip())
                except ValueError:
                    current_thought['confidence'] = 0.5
        
        if current_thought:
            thoughts.append(current_thought)
        
        return thoughts

    def _parse_final_answer(self, response: str) -> Dict[str, Any]:
        """Parse final answer from model response."""
        result = {
            'reasoning': '',
            'answer': '',
            'confidence': 0.0
        }
        
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('Reasoning: '):
                result['reasoning'] = line[len('Reasoning: '):].strip()
            elif line.startswith('Answer: '):
                result['answer'] = line[len('Answer: '):].strip()
            elif line.startswith('Confidence: '):
                try:
                    result['confidence'] = float(line[len('Confidence: '):].strip())
                except ValueError:
                    result['confidence'] = 0.5
        
        return result

    def _get_tree_structure(self) -> Dict[str, Any]:
        """Get the complete tree structure for visualization."""
        return {
            'nodes': [
                {
                    'id': node_id,
                    'content': node.content,
                    'reasoning': node.reasoning,
                    'evaluation': node.evaluation,
                    'depth': node.depth
                }
                for node_id, node in self.thought_tree.items()
            ],
            'edges': [
                {
                    'source': node_id,
                    'target': child_id
                }
                for node_id, node in self.thought_tree.items()
                for child_id in node.children_ids
            ]
        }

# Register the technique
from .registry import TechniqueRegistry
TechniqueRegistry.register(TreeOfThoughtsTechnique())
