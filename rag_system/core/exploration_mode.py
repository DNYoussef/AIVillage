"""Exploration mode for RAG system."""

import logging
from typing import Dict, Any, List, Tuple, Optional
import random
import networkx as nx
from datetime import datetime
from ..core.base_component import BaseComponent
from ..retrieval.graph_store import GraphStore
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from ..processing.advanced_nlp import AdvancedNLP
from ..utils.error_handling import log_and_handle_errors, ErrorContext

logger = logging.getLogger(__name__)

class ExplorationMode(BaseComponent):
    """
    Exploration mode for enhanced knowledge discovery.
    Implements strategies for exploring and discovering new knowledge paths.
    """
    
    def __init__(self, graph_store: GraphStore, llm_config: OpenAIGPTConfig, advanced_nlp: AdvancedNLP):
        """
        Initialize exploration mode.
        
        Args:
            graph_store: Graph store instance
            llm_config: Language model configuration
            advanced_nlp: Advanced NLP processor
        """
        self.graph_store = graph_store
        self.llm = llm_config.create()
        self.advanced_nlp = advanced_nlp
        self.initialized = False
        self.exploration_stats = {
            "total_explorations": 0,
            "successful_explorations": 0,
            "failed_explorations": 0,
            "avg_exploration_time": 0.0
        }
    
    @log_and_handle_errors()
    async def initialize(self) -> None:
        """Initialize exploration mode."""
        if not self.initialized:
            await self.graph_store.initialize()
            self.initialized = True
    
    @log_and_handle_errors()
    async def shutdown(self) -> None:
        """Shutdown exploration mode."""
        if self.initialized:
            await self.graph_store.shutdown()
            self.initialized = False
    
    @log_and_handle_errors()
    async def get_status(self) -> Dict[str, Any]:
        """Get component status."""
        return {
            "initialized": self.initialized,
            "graph_store": await self.graph_store.get_status(),
            "exploration_stats": self.exploration_stats
        }
    
    @log_and_handle_errors()
    async def update_config(self, config: Dict[str, Any]) -> None:
        """Update component configuration."""
        if 'graph_store' in config:
            await self.graph_store.update_config(config['graph_store'])

    async def explore_knowledge_graph(self, start_node: str, depth: int = 3) -> Dict[str, Any]:
        """
        Explore the knowledge graph starting from a given node.

        Args:
            start_node: The node to start exploration from
            depth: The depth of exploration
            
        Returns:
            Dictionary containing exploration results
        """
        async with ErrorContext("ExplorationMode.explore"):
            explored_nodes = set()
            exploration_results = []
            
            await self._explore_recursive(start_node, depth, explored_nodes, exploration_results)
            
            return {
                "start_node": start_node,
                "depth": depth,
                "explored_nodes": list(explored_nodes),
                "exploration_results": exploration_results
            }

    async def _explore_recursive(self, node: str, depth: int, explored_nodes: set, exploration_results: List[Dict[str, Any]]):
        """Recursively explore nodes in the graph."""
        if depth == 0 or node in explored_nodes:
            return

        explored_nodes.add(node)
        
        # Get node information and connections
        node_info = await self.graph_store.get_node_info(node)
        connections = node_info.get("connections", [])
        
        for connection in connections:
            if connection["target"] not in explored_nodes:
                relation = await self._analyze_relation(
                    node,
                    connection["target"],
                    connection["type"]
                )
                exploration_results.append(relation)
                
                # Recursively explore connected nodes
                await self._explore_recursive(
                    connection["target"],
                    depth - 1,
                    explored_nodes,
                    exploration_results
                )

    async def _analyze_relation(self, source: str, target: str, relation_type: str) -> Dict[str, Any]:
        """
        Analyze the relation between two nodes.
        
        Args:
            source: Source node
            target: Target node
            relation_type: Type of relation
            
        Returns:
            Dictionary containing relation analysis
        """
        prompt = f"""
        Analyze the relation between the following concepts:
        Source: {source}
        Target: {target}
        Relation Type: {relation_type}

        Please provide:
        1. A brief description of how these concepts are related.
        2. Potential implications of this relationship.
        3. Any interesting insights or questions that arise from this connection.
        4. Possible new relations that could be inferred from this connection.

        Format your response as a JSON object with keys: "description", "implications", "insights", and "inferred_relations".
        """
        
        response = await self.llm.complete(prompt)
        analysis = self._parse_json_response(response.text)
        
        return {
            "source": source,
            "target": target,
            "relation_type": relation_type,
            "analysis": analysis
        }

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from language model."""
        import json
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON response: {response}")
            return {"error": "Failed to parse response"}

    async def discover_new_relations(self, num_attempts: int = 10) -> List[Dict[str, Any]]:
        """
        Attempt to discover new relations in the knowledge graph.
        
        Args:
            num_attempts: Number of attempts to discover relations
            
        Returns:
            List of newly discovered relations
        """
        new_relations = []
        
        for _ in range(num_attempts):
            source, target = await self._select_random_nodes()
            if source != target:
                new_relation = await self._propose_relation(source, target)
                if new_relation:
                    new_relations.append(new_relation)
        
        return new_relations

    async def _select_random_nodes(self) -> Tuple[str, str]:
        """Select two random nodes from the graph."""
        nodes = await self.graph_store.get_all_nodes()
        return random.sample(nodes, 2)

    async def _propose_relation(self, source: str, target: str) -> Optional[Dict[str, Any]]:
        """
        Propose a potential new relation between nodes.
        
        Args:
            source: Source node
            target: Target node
            
        Returns:
            Optional dictionary containing proposed relation
        """
        source_info = await self.graph_store.get_node_info(source)
        target_info = await self.graph_store.get_node_info(target)
        
        prompt = f"""
        Consider the following two concepts from our knowledge graph:
        Source: {source}
        Source Info: {source_info}
        Target: {target}
        Target Info: {target_info}

        Propose a potential relation between these concepts. If you believe there's no meaningful relation, respond with "No relation".
        Otherwise, provide:
        1. The type of relation.
        2. A brief description of the relation.
        3. The confidence level in this proposed relation (0-1).
        4. A list of potential implications or insights from this relation.

        Format your response as a JSON object with keys: "relation_type", "description", "confidence", and "implications".
        """
        
        response = await self.llm.complete(prompt)
        proposal = self._parse_json_response(response.text)
        
        if proposal == "No relation" or proposal.get("confidence", 0) < 0.5:
            return None
        
        return {
            "source": source,
            "target": target,
            "relation_type": proposal["relation_type"],
            "description": proposal["description"],
            "confidence": proposal["confidence"],
            "implications": proposal["implications"]
        }

    async def validate_proposed_relation(self, relation: Dict[str, Any]) -> bool:
        """
        Validate a proposed relation.
        
        Args:
            relation: Proposed relation to validate
            
        Returns:
            Boolean indicating if relation is valid
        """
        prompt = f"""
        Validate the following proposed relation:
        Source: {relation['source']}
        Target: {relation['target']}
        Relation Type: {relation['relation_type']}
        Description: {relation['description']}
        Implications: {relation['implications']}

        Please assess whether this relation is valid, meaningful, and consistent with existing knowledge.
        Provide your assessment as a JSON object with keys:
        - "is_valid": boolean
        - "confidence": float (0-1)
        - "reasoning": string (brief explanation of your assessment)
        """
        
        response = await self.llm.complete(prompt)
        validation = self._parse_json_response(response.text)
        
        # Use advanced NLP to analyze semantic similarity
        source_embedding = self.advanced_nlp.get_embeddings([relation['source']])[0]
        target_embedding = self.advanced_nlp.get_embeddings([relation['target']])[0]
        semantic_similarity = self.advanced_nlp.calculate_similarity(
            source_embedding,
            target_embedding
        )
        
        # Combine LLM validation with semantic similarity
        is_valid = validation['is_valid'] and semantic_similarity > 0.5
        logger.info(
            f"Relation validation result: {is_valid} "
            f"(LLM: {validation['is_valid']}, "
            f"Similarity: {semantic_similarity:.2f})"
        )
        
        return is_valid

    async def update_knowledge_graph(self, new_relations: List[Dict[str, Any]]) -> None:
        """
        Update graph with new relations.
        
        Args:
            new_relations: List of new relations to add
        """
        for relation in new_relations:
            if await self.validate_proposed_relation(relation):
                await self.graph_store.add_edge(
                    relation["source"],
                    relation["target"],
                    relation["relation_type"],
                    {
                        "description": relation["description"],
                        "confidence": relation["confidence"],
                        "implications": relation["implications"]
                    }
                )
                logger.info(f"Added new relation to knowledge graph: {relation}")
            else:
                logger.info(f"Rejected invalid relation: {relation}")

    async def generate_exploration_report(self, exploration_results: Dict[str, Any]) -> str:
        """
        Generate human-readable exploration report.
        
        Args:
            exploration_results: Results from exploration
            
        Returns:
            Formatted report string
        """
        report = f"Exploration Report\n\n"
        report += f"Start Node: {exploration_results['start_node']}\n"
        report += f"Exploration Depth: {exploration_results['depth']}\n"
        report += f"Nodes Explored: {len(exploration_results['explored_nodes'])}\n\n"
        
        report += "Key Findings:\n"
        for result in exploration_results['exploration_results']:
            report += f"- Relation: {result['source']} -> {result['target']} ({result['relation_type']})\n"
            report += f"  Description: {result['analysis']['description']}\n"
            report += f"  Implications: {result['analysis']['implications']}\n"
            report += f"  Insights: {result['analysis']['insights']}\n"
            if 'inferred_relations' in result['analysis']:
                report += f"  Inferred Relations: {result['analysis']['inferred_relations']}\n"
            report += "\n"
        
        return report

    async def find_causal_paths(self,
                              start_node: str,
                              end_node: str,
                              max_depth: int = 5) -> List[List[str]]:
        """
        Find direct causal paths between nodes.
        
        Args:
            start_node: Starting node
            end_node: Ending node
            max_depth: Maximum path depth
            
        Returns:
            List of causal paths
        """
        graph = await self.graph_store.get_graph()
        paths = list(nx.all_simple_paths(graph, start_node, end_node, cutoff=max_depth))
        
        if not paths:
            return []
        
        causal_paths = []
        for path in paths:
            if await self._is_causal_path(path):
                causal_paths.append(path)
        
        return sorted(causal_paths, key=len)

    async def _is_causal_path(self, path: List[str]) -> bool:
        """Check if path is causal."""
        for i in range(len(path) - 1):
            edge_data = await self.graph_store.get_edge_data(path[i], path[i+1])
            if not edge_data.get('is_causal', False):
                return False
        return True

    async def find_creative_connections(self,
                                     start_node: str,
                                     end_node: str,
                                     excluded_nodes: List[str],
                                     max_depth: int = 7) -> List[List[str]]:
        """
        Find creative connections between nodes.
        
        Args:
            start_node: Starting node
            end_node: Ending node
            excluded_nodes: Nodes to exclude
            max_depth: Maximum path depth
            
        Returns:
            List of creative paths
        """
        graph = await self.graph_store.get_graph()
        excluded_graph = graph.copy()
        excluded_graph.remove_nodes_from(excluded_nodes)
        
        paths = list(nx.all_simple_paths(
            excluded_graph,
            start_node,
            end_node,
            cutoff=max_depth
        ))
        
        if not paths:
            return []
        
        creative_paths = []
        for path in paths:
            creativity_score = await self._calculate_creativity_score(path)
            creative_paths.append((path, creativity_score))
        
        return [path for path, score in sorted(
            creative_paths,
            key=lambda x: x[1],
            reverse=True
        )]

    async def _calculate_creativity_score(self, path: List[str]) -> float:
        """Calculate creativity score for path."""
        score = 0
        for i in range(len(path) - 1):
            edge_data = await self.graph_store.get_edge_data(path[i], path[i+1])
            score += edge_data.get('novelty', 0) * edge_data.get('relevance', 0)
        return score / (len(path) - 1)

    async def generate_new_ideas(self,
                               start_node: str,
                               end_node: str) -> List[Dict[str, Any]]:
        """
        Generate new ideas between nodes.
        
        Args:
            start_node: Starting node
            end_node: Ending node
            
        Returns:
            List of new ideas
        """
        causal_paths = await self.find_causal_paths(start_node, end_node)
        
        if not causal_paths:
            return []
        
        main_causal_path = causal_paths[0]
        excluded_nodes = set(
            node for path in causal_paths for node in path[1:-1]
        )
        
        creative_paths = await self.find_creative_connections(
            start_node,
            end_node,
            list(excluded_nodes)
        )
        
        new_ideas = []
        for path in creative_paths:
            new_idea = await self._generate_idea_from_path(path, main_causal_path)
            new_ideas.append(new_idea)
        
        return new_ideas

    async def _generate_idea_from_path(self,
                                     creative_path: List[str],
                                     causal_path: List[str]) -> Dict[str, Any]:
        """
        Generate idea from creative and causal paths.
        
        Args:
            creative_path: Creative path between nodes
            causal_path: Main causal path
            
        Returns:
            Dictionary containing new idea
        """
        prompt = f"""
        Given the following information:
        Main causal path: {' -> '.join(causal_path)}
        Creative alternative path: {' -> '.join(creative_path)}

        Generate a new idea or connection that:
        1. Leverages the insights from both paths
        2. Proposes a novel way to connect the start and end concepts
        3. Explains how this new connection might lead to innovative solutions or perspectives

        Provide your response as a JSON object with the following structure:
        {{
            "new_idea": "Brief description of the new idea",
            "explanation": "Detailed explanation of the idea and its potential impact",
            "nodes_to_add": ["List of new nodes to add to the knowledge graph"],
            "edges_to_add": [
                {{
                    "source": "Source node",
                    "target": "Target node",
                    "relation": "Type of relation",
                    "description": "Description of the relation"
                }}
            ]
        }}
        """
        
        response = await self.llm.complete(prompt)
        return self._parse_json_response(response.text)

    async def update_graph_with_new_ideas(self, new_ideas: List[Dict[str, Any]]) -> None:
        """
        Update graph with new ideas.
        
        Args:
            new_ideas: List of new ideas to add
        """
        for idea in new_ideas:
            # Add new nodes
            for node in idea['nodes_to_add']:
                await self.graph_store.add_node(
                    node,
                    {'type': 'concept', 'description': ''}
                )
            
            # Add new edges
            for edge in idea['edges_to_add']:
                await self.graph_store.add_edge(
                    edge['source'],
                    edge['target'],
                    edge['relation'],
                    {
                        'description': edge['description'],
                        'is_causal': False,
                        'novelty': 0.8,
                        'relevance': 0.7
                    }
                )
        
        logger.info(f"Added {len(new_ideas)} new ideas to the knowledge graph")

    async def creative_exploration(self,
                                 start_node: str,
                                 end_node: str) -> Dict[str, Any]:
        """
        Perform creative exploration between nodes.
        
        Args:
            start_node: Starting node
            end_node: Ending node
            
        Returns:
            Dictionary containing exploration results
        """
        causal_paths = await self.find_causal_paths(start_node, end_node)
        new_ideas = await self.generate_new_ideas(start_node, end_node)
        await self.update_graph_with_new_ideas(new_ideas)
        
        return {
            "start_node": start_node,
            "end_node": end_node,
            "causal_paths": causal_paths,
            "new_ideas": new_ideas
        }
