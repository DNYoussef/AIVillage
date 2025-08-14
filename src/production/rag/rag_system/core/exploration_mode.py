import logging
import random
from typing import Any

import networkx as nx
from rag_system.processing.advanced_nlp import AdvancedNLP
from rag_system.retrieval.graph_store import GraphStore

from agents.language_models.openai_gpt import OpenAIGPTConfig

logger = logging.getLogger(__name__)


class ExplorationMode:
    def __init__(
        self,
        graph_store: GraphStore,
        llm_config: OpenAIGPTConfig,
        advanced_nlp: AdvancedNLP,
    ) -> None:
        self.graph_store = graph_store
        self.llm = llm_config.create()
        self.advanced_nlp = advanced_nlp

    async def explore_knowledge_graph(self, start_node: str, depth: int = 3) -> dict[str, Any]:
        """Explore the knowledge graph starting from a given node.

        Args:
            start_node (str): The node to start exploration from.
            depth (int): The depth of exploration.

        Returns:
            Dict[str, Any]: A dictionary containing the exploration results.
        """
        explored_nodes = set()
        exploration_results = []

        await self._explore_recursive(start_node, depth, explored_nodes, exploration_results)

        return {
            "start_node": start_node,
            "depth": depth,
            "explored_nodes": list(explored_nodes),
            "exploration_results": exploration_results,
        }

    async def _explore_recursive(
        self,
        node: str,
        depth: int,
        explored_nodes: set,
        exploration_results: list[dict[str, Any]],
    ) -> None:
        if depth == 0 or node in explored_nodes:
            return

        explored_nodes.add(node)

        # Get node information and connections
        node_info = await self.graph_store.get_node_info(node)
        connections = node_info.get("connections", [])

        for connection in connections:
            if connection["target"] not in explored_nodes:
                relation = await self._analyze_relation(node, connection["target"], connection["type"])
                exploration_results.append(relation)

                # Recursively explore connected nodes
                await self._explore_recursive(connection["target"], depth - 1, explored_nodes, exploration_results)

    async def _analyze_relation(self, source: str, target: str, relation_type: str) -> dict[str, Any]:
        """Analyze the relation between two nodes.

        Args:
            source (str): The source node.
            target (str): The target node.
            relation_type (str): The type of relation between the nodes.

        Returns:
            Dict[str, Any]: A dictionary containing the analysis of the relation.
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
            "analysis": analysis,
        }

    def _parse_json_response(self, response: str) -> dict[str, Any]:
        import json

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.exception(f"Failed to parse JSON response: {response}")
            return {"error": "Failed to parse response"}

    async def discover_new_relations(self, num_attempts: int = 10) -> list[dict[str, Any]]:
        """Attempt to discover new relations in the knowledge graph.

        Args:
            num_attempts (int): The number of attempts to discover new relations.

        Returns:
            List[Dict[str, Any]]: A list of newly discovered relations.
        """
        new_relations = []

        for _ in range(num_attempts):
            source, target = await self._select_random_nodes()
            if source != target:
                new_relation = await self._propose_relation(source, target)
                if new_relation:
                    new_relations.append(new_relation)

        return new_relations

    async def _select_random_nodes(self) -> tuple[str, str]:
        """Select two random nodes from the knowledge graph."""
        nodes = await self.graph_store.get_all_nodes()
        return random.sample(nodes, 2)

    async def _propose_relation(self, source: str, target: str) -> dict[str, Any]:
        """Propose a potential new relation between two nodes."""
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
            "implications": proposal["implications"],
        }

    async def validate_proposed_relation(self, relation: dict[str, Any]) -> bool:
        """Validate a proposed relation using the language model and advanced NLP techniques.

        Args:
            relation (Dict[str, Any]): The proposed relation to validate.

        Returns:
            bool: True if the relation is valid, False otherwise.
        """
        prompt = f"""
        Validate the following proposed relation:
        Source: {relation["source"]}
        Target: {relation["target"]}
        Relation Type: {relation["relation_type"]}
        Description: {relation["description"]}
        Implications: {relation["implications"]}

        Please assess whether this relation is valid, meaningful, and consistent with existing knowledge.
        Provide your assessment as a JSON object with keys:
        - "is_valid": boolean
        - "confidence": float (0-1)
        - "reasoning": string (brief explanation of your assessment)
        """

        response = await self.llm.complete(prompt)
        validation = self._parse_json_response(response.text)

        # Use advanced NLP to analyze the semantic similarity between the source and target
        source_embedding = self.advanced_nlp.get_embeddings([relation["source"]])[0]
        target_embedding = self.advanced_nlp.get_embeddings([relation["target"]])[0]
        semantic_similarity = self.advanced_nlp.calculate_similarity(source_embedding, target_embedding)

        # Combine LLM validation with semantic similarity
        is_valid = validation["is_valid"] and semantic_similarity > 0.5
        logger.info(
            f"Relation validation result: {is_valid} (LLM: {validation['is_valid']}, Similarity: {semantic_similarity:.2f})"
        )

        return is_valid

    async def update_knowledge_graph(self, new_relations: list[dict[str, Any]]) -> None:
        """Update the knowledge graph with newly discovered and validated relations.

        Args:
            new_relations (List[Dict[str, Any]]): List of new relations to add to the graph.
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
                        "implications": relation["implications"],
                    },
                )
                logger.info(f"Added new relation to knowledge graph: {relation}")
            else:
                logger.info(f"Rejected invalid relation: {relation}")

    async def generate_exploration_report(self, exploration_results: dict[str, Any]) -> str:
        """Generate a human-readable report of the exploration results.

        Args:
            exploration_results (Dict[str, Any]): The results from the explore_knowledge_graph method.

        Returns:
            str: A formatted report of the exploration results.
        """
        report = "Exploration Report\n\n"
        report += f"Start Node: {exploration_results['start_node']}\n"
        report += f"Exploration Depth: {exploration_results['depth']}\n"
        report += f"Nodes Explored: {len(exploration_results['explored_nodes'])}\n\n"

        report += "Key Findings:\n"
        for result in exploration_results["exploration_results"]:
            report += f"- Relation: {result['source']} -> {result['target']} ({result['relation_type']})\n"
            report += f"  Description: {result['analysis']['description']}\n"
            report += f"  Implications: {result['analysis']['implications']}\n"
            report += f"  Insights: {result['analysis']['insights']}\n"
            if "inferred_relations" in result["analysis"]:
                report += f"  Inferred Relations: {result['analysis']['inferred_relations']}\n"
            report += "\n"

        return report

    async def find_causal_paths(self, start_node: str, end_node: str, max_depth: int = 5) -> list[list[str]]:
        """Find the most direct causal path between two nodes in the knowledge graph.

        Args:
            start_node (str): The starting node.
            end_node (str): The ending node.
            max_depth (int): Maximum depth to search for paths.

        Returns:
            List[List[str]]: A list of causal paths, where each path is a list of node names.
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

    async def _is_causal_path(self, path: list[str]) -> bool:
        """Check if a given path is causal.

        Args:
            path (List[str]): A list of node names representing a path.

        Returns:
            bool: True if the path is causal, False otherwise.
        """
        for i in range(len(path) - 1):
            edge_data = await self.graph_store.get_edge_data(path[i], path[i + 1])
            if not edge_data.get("is_causal", False):
                return False
        return True

    async def find_creative_connections(
        self,
        start_node: str,
        end_node: str,
        excluded_nodes: list[str],
        max_depth: int = 7,
    ) -> list[list[str]]:
        """Find creative connections between two nodes, excluding certain nodes and paths.

        Args:
            start_node (str): The starting node.
            end_node (str): The ending node.
            excluded_nodes (List[str]): Nodes to exclude from the path.
            max_depth (int): Maximum depth to search for paths.

        Returns:
            List[List[str]]: A list of creative paths, where each path is a list of node names.
        """
        graph = await self.graph_store.get_graph()
        excluded_graph = graph.copy()
        excluded_graph.remove_nodes_from(excluded_nodes)

        paths = list(nx.all_simple_paths(excluded_graph, start_node, end_node, cutoff=max_depth))

        if not paths:
            return []

        creative_paths = []
        for path in paths:
            creativity_score = await self._calculate_creativity_score(path)
            creative_paths.append((path, creativity_score))

        return [path for path, score in sorted(creative_paths, key=lambda x: x[1], reverse=True)]

    async def _calculate_creativity_score(self, path: list[str]) -> float:
        """Calculate a creativity score for a given path.

        Args:
            path (List[str]): A list of node names representing a path.

        Returns:
            float: A creativity score for the path.
        """
        total = 0.0
        for i in range(len(path) - 1):
            edge = await self.graph_store.get_edge_data(path[i], path[i + 1])
            novelty = float(edge.get("novelty", 0.0))
            relevance = float(edge.get("relevance", 0.0))
            total += novelty * 0.7 + relevance * 0.3
        if len(path) > 1:
            return total / (len(path) - 1)
        return 0.0

    async def generate_new_ideas(self, start_node: str, end_node: str) -> list[dict[str, Any]]:
        """Generate new ideas by finding causal paths and creative connections.

        Args:
            start_node (str): The starting node.
            end_node (str): The ending node.

        Returns:
            List[Dict[str, Any]]: A list of new ideas and connections.
        """
        causal_paths = await self.find_causal_paths(start_node, end_node)

        if not causal_paths:
            return []

        main_causal_path = causal_paths[0]
        excluded_nodes = {node for path in causal_paths for node in path[1:-1]}

        creative_paths = await self.find_creative_connections(start_node, end_node, list(excluded_nodes))

        new_ideas = []
        for path in creative_paths:
            new_idea = await self._generate_idea_from_path(path, main_causal_path)
            new_ideas.append(new_idea)

        return new_ideas

    async def _generate_idea_from_path(self, creative_path: list[str], causal_path: list[str]) -> dict[str, Any]:
        """Generate a new idea based on a creative path and the main causal path.

        Args:
            creative_path (List[str]): A creative path between two nodes.
            causal_path (List[str]): The main causal path between the same two nodes.

        Returns:
            Dict[str, Any]: A new idea or connection.
        """
        prompt = f"""
        Given the following information:
        Main causal path: {" -> ".join(causal_path)}
        Creative alternative path: {" -> ".join(creative_path)}

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

    async def update_graph_with_new_ideas(self, new_ideas: list[dict[str, Any]]) -> None:
        """Update the knowledge graph with new ideas and connections.

        Args:
            new_ideas (List[Dict[str, Any]]): A list of new ideas to add to the graph.
        """
        for idea in new_ideas:
            # Add new nodes
            for node in idea["nodes_to_add"]:
                await self.graph_store.add_node(node, {"type": "concept", "description": ""})

            # Add new edges
            for edge in idea["edges_to_add"]:
                await self.graph_store.add_edge(
                    edge["source"],
                    edge["target"],
                    edge["relation"],
                    {
                        "description": edge["description"],
                        "is_causal": False,
                        "novelty": 0.8,
                        "relevance": 0.7,
                    },
                )

        logger.info(f"Added {len(new_ideas)} new ideas to the knowledge graph")

    async def creative_exploration(self, start_node: str, end_node: str) -> dict[str, Any]:
        """Perform a creative exploration between two nodes in the knowledge graph.

        Args:
            start_node (str): The starting node.
            end_node (str): The ending node.

        Returns:
            Dict[str, Any]: A report of the creative exploration process and results.
        """
        causal_paths = await self.find_causal_paths(start_node, end_node)
        new_ideas = await self.generate_new_ideas(start_node, end_node)
        await self.update_graph_with_new_ideas(new_ideas)

        return {
            "start_node": start_node,
            "end_node": end_node,
            "causal_paths": causal_paths,
            "new_ideas": new_ideas,
        }
