import logging
from typing import Dict, Any, List, Tuple
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from rag_system.error_handling.error_handler import error_handler, safe_execute, AIVillageException
import networkx as nx
import matplotlib.pyplot as plt
import io
import json

logger = logging.getLogger(__name__)

class KnowledgeGraphAgent:
    def __init__(self, llm_config: OpenAIGPTConfig):
        self.llm = llm_config.create()
        self.graph = nx.Graph()

    @error_handler.handle_error
    async def query_graph(self, query: str) -> Dict[str, Any]:
        """
        Query the knowledge graph based on the given query.

        Args:
            query (str): The query to execute on the knowledge graph.

        Returns:
            Dict[str, Any]: The query results.
        """
        prompt = self._create_graph_query_prompt(query)
        response = await self.llm.complete(prompt)
        return self._parse_graph_query_response(response.text)

    def _create_graph_query_prompt(self, query: str) -> str:
        return f"""
        Given the following knowledge graph structure:

        {self._get_graph_structure()}

        Execute the following query on the knowledge graph:

        Query: {query}

        Please provide the query results in a structured JSON format, including:
        1. Matched nodes
        2. Relevant relationships
        3. Any inferred information based on the graph structure

        If the query involves graph traversal or complex operations, please explain the steps taken to arrive at the result.
        """

    def _parse_graph_query_response(self, response: str) -> Dict[str, Any]:
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse graph query response: {response}")
            raise AIVillageException("Failed to parse graph query response")

    @error_handler.handle_error
    async def update_graph(self, new_information: Dict[str, Any]) -> bool:
        """
        Update the knowledge graph with new information.

        Args:
            new_information (Dict[str, Any]): The new information to add to the graph.

        Returns:
            bool: True if the update was successful, False otherwise.
        """
        prompt = self._create_graph_update_prompt(new_information)
        response = await self.llm.complete(prompt)
        update_instructions = self._parse_graph_update_response(response.text)
        return self._apply_graph_updates(update_instructions)

    def _create_graph_update_prompt(self, new_information: Dict[str, Any]) -> str:
        return f"""
        Given the following knowledge graph structure:

        {self._get_graph_structure()}

        And the following new information:

        {json.dumps(new_information, indent=2)}

        Please provide instructions on how to update the knowledge graph. Include:
        1. New nodes to add
        2. New edges to add
        3. Existing nodes or edges to modify
        4. Any nodes or edges to remove (if applicable)

        Provide your instructions in a structured JSON format.
        """

    def _parse_graph_update_response(self, response: str) -> Dict[str, Any]:
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse graph update response: {response}")
            raise AIVillageException("Failed to parse graph update response")

    def _apply_graph_updates(self, update_instructions: Dict[str, Any]) -> bool:
        try:
            for node in update_instructions.get('add_nodes', []):
                self.graph.add_node(node['id'], **node.get('attributes', {}))
            
            for edge in update_instructions.get('add_edges', []):
                self.graph.add_edge(edge['source'], edge['target'], **edge.get('attributes', {}))
            
            for node in update_instructions.get('modify_nodes', []):
                self.graph.nodes[node['id']].update(node.get('attributes', {}))
            
            for edge in update_instructions.get('modify_edges', []):
                self.graph[edge['source']][edge['target']].update(edge.get('attributes', {}))
            
            for node in update_instructions.get('remove_nodes', []):
                self.graph.remove_node(node)
            
            for edge in update_instructions.get('remove_edges', []):
                self.graph.remove_edge(edge['source'], edge['target'])
            
            return True
        except Exception as e:
            logger.error(f"Error applying graph updates: {str(e)}")
            return False

    @error_handler.handle_error
    async def perform_reasoning(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform graph-based reasoning to infer new relationships or information.

        Args:
            context (Dict[str, Any]): The context for the reasoning task.

        Returns:
            Dict[str, Any]: The results of the reasoning process.
        """
        prompt = self._create_reasoning_prompt(context)
        response = await self.llm.complete(prompt)
        return self._parse_reasoning_response(response.text)

    def _create_reasoning_prompt(self, context: Dict[str, Any]) -> str:
        return f"""
        Given the following knowledge graph structure:

        {self._get_graph_structure()}

        And the following context:

        {json.dumps(context, indent=2)}

        Please perform graph-based reasoning to infer new relationships or information. Consider:
        1. Path analysis between relevant nodes
        2. Identification of common patterns or motifs
        3. Application of transitive relationships
        4. Detection of potential inconsistencies or conflicts

        Provide your reasoning and conclusions in a structured JSON format, including any newly inferred relationships or information.
        """

    def _parse_reasoning_response(self, response: str) -> Dict[str, Any]:
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse reasoning response: {response}")
            raise AIVillageException("Failed to parse reasoning response")

    def _get_graph_structure(self) -> str:
        return nx.to_dict_of_dicts(self.graph)

    @safe_execute
    async def process_input(self, query: str, new_information: Dict[str, Any], reasoning_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input by querying the graph, updating it with new information, and performing reasoning.

        Args:
            query (str): The query to execute on the knowledge graph.
            new_information (Dict[str, Any]): New information to add to the graph.
            reasoning_context (Dict[str, Any]): Context for the reasoning task.

        Returns:
            Dict[str, Any]: A dictionary containing the results of all operations.
        """
        query_result = await self.query_graph(query)
        update_success = await self.update_graph(new_information)
        reasoning_result = await self.perform_reasoning(reasoning_context)
        
        return {
            "query_result": query_result,
            "update_success": update_success,
            "reasoning_result": reasoning_result,
            "graph_structure": self._get_graph_structure()
        }

    def visualize_graph(self, highlight_nodes: List[str] = None, highlight_edges: List[Tuple[str, str]] = None) -> bytes:
        """
        Visualize the knowledge graph and return the image as bytes.

        Args:
            highlight_nodes (List[str], optional): List of node IDs to highlight.
            highlight_edges (List[Tuple[str, str]], optional): List of edge tuples to highlight.

        Returns:
            bytes: The PNG image of the graph visualization as bytes.
        """
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph)
        
        # Draw all nodes and edges
        nx.draw_networkx_nodes(self.graph, pos, node_size=1000, node_color='lightblue')
        nx.draw_networkx_edges(self.graph, pos, edge_color='gray')
        nx.draw_networkx_labels(self.graph, pos)
        
        # Highlight specific nodes and edges if provided
        if highlight_nodes:
            nx.draw_networkx_nodes(self.graph, pos, nodelist=highlight_nodes, node_color='red', node_size=1200)
        
        if highlight_edges:
            nx.draw_networkx_edges(self.graph, pos, edgelist=highlight_edges, edge_color='red', width=2)
        
        # Add edge labels
        edge_labels = nx.get_edge_attributes(self.graph, 'relationship')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        
        # Save the plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf.getvalue()

# Example usage
if __name__ == "__main__":
    import asyncio

    async def main():
        llm_config = OpenAIGPTConfig(chat_model="gpt-4")
        kg_agent = KnowledgeGraphAgent(llm_config)
        
        # Initialize the graph with some sample data
        kg_agent.graph.add_node("Alice", type="Person")
        kg_agent.graph.add_node("Bob", type="Person")
        kg_agent.graph.add_node("CompanyX", type="Company")
        kg_agent.graph.add_edge("Alice", "Bob", relationship="friend")
        kg_agent.graph.add_edge("Alice", "CompanyX", relationship="works_for")
        
        query = "Find all people who work for CompanyX"
        new_information = {
            "nodes": [{"id": "Charlie", "type": "Person"}],
            "edges": [{"source": "Charlie", "target": "CompanyX", "relationship": "works_for"}]
        }
        reasoning_context = {"focus": "employee relationships"}
        
        result = await kg_agent.process_input(query, new_information, reasoning_context)
        
        print("Query Result:")
        print(result["query_result"])
        print("\nUpdate Success:", result["update_success"])
        print("\nReasoning Result:")
        print(result["reasoning_result"])
        
        # Visualize the updated graph
        graph_image = kg_agent.visualize_graph(highlight_nodes=["Charlie"], highlight_edges=[("Charlie", "CompanyX")])
        with open("knowledge_graph.png", "wb") as f:
            f.write(graph_image)
        print("\nKnowledge graph visualization saved as 'knowledge_graph.png'")

    asyncio.run(main())
