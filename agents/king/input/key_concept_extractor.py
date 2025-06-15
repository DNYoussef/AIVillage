import logging
from typing import Dict, Any, List
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from utils.error_handler import error_handler, safe_execute, AIVillageException
import networkx as nx
import matplotlib.pyplot as plt
import io

logger = logging.getLogger(__name__)

class KeyConceptExtractor:
    def __init__(self, llm_config: OpenAIGPTConfig):
        self.llm = llm_config.create()

    @error_handler
    async def extract_key_concepts(self, user_input: str, interpreted_intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract key concepts from the user input and interpreted intent.

        Args:
            user_input (str): The raw input from the user.
            interpreted_intent (Dict[str, Any]): The interpreted intent from the UserIntentInterpreter.

        Returns:
            Dict[str, Any]: A dictionary containing the extracted key concepts and their relationships.
        """
        prompt = self._create_concept_extraction_prompt(user_input, interpreted_intent)
        response = await self.llm.complete(prompt)
        extracted_concepts = self._parse_concept_response(response.text)
        concept_graph = self._build_concept_graph(extracted_concepts)
        return {
            "concepts": extracted_concepts,
            "graph": concept_graph
        }

    def _create_concept_extraction_prompt(self, user_input: str, interpreted_intent: Dict[str, Any]) -> str:
        return f"""
        Analyze the following user input and interpreted intent to extract key concepts:

        User Input: "{user_input}"

        Interpreted Intent: {interpreted_intent}

        Please provide a detailed extraction of key concepts, including:
        1. Main Concepts: The primary ideas or topics mentioned.
        2. Related Concepts: Secondary or related ideas that are relevant.
        3. Attributes: Important characteristics or properties of the main concepts.
        4. Relationships: How the concepts are related to each other.
        5. Hierarchy: Any hierarchical structure among the concepts.

        Provide your analysis in a structured JSON format, where each concept is an object with properties for its attributes and relationships.
        """

    def _parse_concept_response(self, response: str) -> Dict[str, Any]:
        import json
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse concept response: {response}")
            raise AIVillageException("Failed to parse concept response")

    def _build_concept_graph(self, concepts: Dict[str, Any]) -> nx.Graph:
        G = nx.Graph()
        for concept, data in concepts.items():
            G.add_node(concept, **data.get('attributes', {}))
            for related in data.get('relationships', []):
                G.add_edge(concept, related['concept'], type=related['type'])
        return G

    @error_handler
    async def analyze_concept_importance(self, concepts: Dict[str, Any]) -> Dict[str, float]:
        """
        Analyze the importance of each extracted concept.

        Args:
            concepts (Dict[str, Any]): The extracted concepts and their relationships.

        Returns:
            Dict[str, float]: A dictionary mapping each concept to its importance score.
        """
        prompt = self._create_importance_analysis_prompt(concepts)
        response = await self.llm.complete(prompt)
        return self._parse_importance_response(response.text)

    def _create_importance_analysis_prompt(self, concepts: Dict[str, Any]) -> str:
        return f"""
        Analyze the importance of the following extracted concepts:

        Concepts: {concepts}

        For each concept, assign an importance score between 0 and 1, where:
        0 = Not important at all
        1 = Extremely important

        Consider the following factors when assigning scores:
        1. Relevance to the main topic or intent
        2. Number of relationships with other concepts
        3. Depth of attributes or properties
        4. Position in the concept hierarchy

        Provide your analysis as a JSON object where keys are concept names and values are their importance scores.
        """

    def _parse_importance_response(self, response: str) -> Dict[str, float]:
        import json
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse importance response: {response}")
            raise AIVillageException("Failed to parse importance response")

    @safe_execute
    async def process_input(self, user_input: str, interpreted_intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the user input and interpreted intent to extract and analyze key concepts.

        Args:
            user_input (str): The raw input from the user.
            interpreted_intent (Dict[str, Any]): The interpreted intent from the UserIntentInterpreter.

        Returns:
            Dict[str, Any]: A dictionary containing the extracted concepts, their relationships, and importance scores.
        """
        extraction_result = await self.extract_key_concepts(user_input, interpreted_intent)
        importance_scores = await self.analyze_concept_importance(extraction_result['concepts'])
        
        # Add importance scores to the concept graph
        for node, score in importance_scores.items():
            extraction_result['graph'].nodes[node]['importance'] = score

        return {
            "concepts": extraction_result['concepts'],
            "concept_graph": extraction_result['graph'],
            "importance_scores": importance_scores
        }

    def visualize_concept_graph(self, graph: nx.Graph) -> bytes:
        """
        Visualize the concept graph and return the image as bytes.

        Args:
            graph (nx.Graph): The NetworkX graph of concepts.

        Returns:
            bytes: The PNG image of the graph visualization as bytes.
        """
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(graph)
        
        # Draw nodes
        nx.draw_networkx_nodes(graph, pos, node_size=1000, node_color='lightblue')
        
        # Draw edges
        nx.draw_networkx_edges(graph, pos)
        
        # Draw labels
        nx.draw_networkx_labels(graph, pos)
        
        # Draw edge labels
        edge_labels = nx.get_edge_attributes(graph, 'type')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
        
        # Save the plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        return buf.getvalue()

# Example usage
if __name__ == "__main__":
    import asyncio

    async def main():
        llm_config = OpenAIGPTConfig(chat_model="gpt-4")
        extractor = KeyConceptExtractor(llm_config)
        
        user_input = "I need to improve my team's productivity and communication skills."
        interpreted_intent = {
            "primary_intent": "Improve team performance",
            "secondary_intents": ["Enhance productivity", "Develop communication skills"],
            "key_entities": ["team", "productivity", "communication skills"],
            "sentiment": "Determined",
            "urgency": "Medium",
            "context": "Workplace improvement"
        }
        
        result = await extractor.process_input(user_input, interpreted_intent)
        
        print("Extracted Concepts:")
        print(result["concepts"])
        print("\nImportance Scores:")
        print(result["importance_scores"])
        
        # Visualize the concept graph
        graph_image = extractor.visualize_concept_graph(result["concept_graph"])
        with open("concept_graph.png", "wb") as f:
            f.write(graph_image)
        print("\nConcept graph visualization saved as 'concept_graph.png'")

    asyncio.run(main())
