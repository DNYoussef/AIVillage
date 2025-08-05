import io
import json
import logging
from typing import Any

from langroid.language_models.openai_gpt import OpenAIGPTConfig
import matplotlib.pyplot as plt
import networkx as nx

from core.error_handling import AIVillageException, error_handler, safe_execute

logger = logging.getLogger(__name__)


class UnifiedInputProcessor:
    def __init__(self, llm_config: OpenAIGPTConfig):
        self.llm = llm_config.create()

    @error_handler.handle_error
    async def process_input(self, user_input: str) -> dict[str, Any]:
        """Process the user input by interpreting intent and extracting key concepts.

        Args:
            user_input (str): The raw input from the user.

        Returns:
            Dict[str, Any]: A dictionary containing the interpreted intent, key concepts, and concept graph.
        """
        interpreted_intent = await self._interpret_intent(user_input)
        key_concepts = await self._extract_key_concepts(user_input, interpreted_intent)
        concept_graph = self._build_concept_graph(key_concepts)

        return {
            "interpreted_intent": interpreted_intent,
            "key_concepts": key_concepts,
            "concept_graph": concept_graph,
        }

    async def _interpret_intent(self, user_input: str) -> dict[str, Any]:
        prompt = f"""
        Analyze the following user input and determine the user's intent:

        User Input: "{user_input}"

        Please provide a detailed interpretation of the user's intent, including:
        1. Primary Intent: The main goal or purpose of the user's input.
        2. Secondary Intents: Any additional or implied intentions.
        3. Key Entities: Important entities (e.g., names, places, concepts) mentioned in the input.
        4. Sentiment: The overall sentiment or emotion expressed in the input.
        5. Urgency: The level of urgency or importance of the user's request.
        6. Context: Any relevant context that might be important for understanding the intent.

        Provide your analysis in a structured JSON format.
        """
        response = await self.llm.complete(prompt)
        return self._parse_json_response(response.text)

    async def _extract_key_concepts(self, user_input: str, interpreted_intent: dict[str, Any]) -> dict[str, Any]:
        prompt = f"""
        Based on the following user input and interpreted intent, extract the key concepts:

        User Input: "{user_input}"
        Interpreted Intent: {json.dumps(interpreted_intent, indent=2)}

        Please provide a detailed extraction of key concepts, including:
        1. Main Concepts: The primary ideas or topics mentioned.
        2. Related Concepts: Secondary or related ideas that are relevant.
        3. Attributes: Important characteristics or properties of the main concepts.
        4. Relationships: How the concepts are related to each other.
        5. Hierarchy: Any hierarchical structure among the concepts.

        Provide your analysis in a structured JSON format, where each concept is an object with properties for its attributes and relationships.
        """
        response = await self.llm.complete(prompt)
        return self._parse_json_response(response.text)

    def _build_concept_graph(self, concepts: dict[str, Any]) -> nx.Graph:
        G = nx.Graph()
        for concept, data in concepts.items():
            G.add_node(concept, **data.get("attributes", {}))
            for related in data.get("relationships", []):
                G.add_edge(concept, related["concept"], type=related["type"])
        return G

    def _parse_json_response(self, response: str) -> dict[str, Any]:
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON response: {response}")
            raise AIVillageException("Failed to parse JSON response")

    @safe_execute
    async def analyze_input_importance(self, processed_input: dict[str, Any]) -> dict[str, float]:
        """Analyze the importance of each extracted concept and intent.

        Args:
            processed_input (Dict[str, Any]): The processed input containing interpreted intent and key concepts.

        Returns:
            Dict[str, float]: A dictionary mapping each concept and intent to its importance score.
        """
        prompt = f"""
        Analyze the importance of the following interpreted intent and key concepts:

        Interpreted Intent: {json.dumps(processed_input["interpreted_intent"], indent=2)}
        Key Concepts: {json.dumps(processed_input["key_concepts"], indent=2)}

        For each intent and concept, assign an importance score between 0 and 1, where:
        0 = Not important at all
        1 = Extremely important

        Consider the following factors when assigning scores:
        1. Relevance to the main topic or goal
        2. Number of relationships with other concepts
        3. Depth of attributes or properties
        4. Position in the concept hierarchy
        5. Urgency and sentiment expressed

        Provide your analysis as a JSON object where keys are intent/concept names and values are their importance scores.
        """
        response = await self.llm.complete(prompt)
        return self._parse_json_response(response.text)

    def visualize_concept_graph(self, graph: nx.Graph) -> bytes:
        """Visualize the concept graph and return the image as bytes.

        Args:
            graph (nx.Graph): The NetworkX graph of concepts.

        Returns:
            bytes: The PNG image of the graph visualization as bytes.
        """
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(graph)

        nx.draw_networkx_nodes(graph, pos, node_size=1000, node_color="lightblue")
        nx.draw_networkx_edges(graph, pos)
        nx.draw_networkx_labels(graph, pos)

        edge_labels = nx.get_edge_attributes(graph, "type")
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()

        return buf.getvalue()


# Example usage
if __name__ == "__main__":
    import asyncio

    async def main():
        llm_config = OpenAIGPTConfig(chat_model="gpt-4")
        processor = UnifiedInputProcessor(llm_config)

        user_input = "I need to improve my team's productivity and communication skills within the next quarter."

        result = await processor.process_input(user_input)

        print("Interpreted Intent:")
        print(json.dumps(result["interpreted_intent"], indent=2))
        print("\nKey Concepts:")
        print(json.dumps(result["key_concepts"], indent=2))

        importance_scores = await processor.analyze_input_importance(result)
        print("\nImportance Scores:")
        print(json.dumps(importance_scores, indent=2))

        # Visualize the concept graph
        graph_image = processor.visualize_concept_graph(result["concept_graph"])
        with open("concept_graph.png", "wb") as f:
            f.write(graph_image)
        print("\nConcept graph visualization saved as 'concept_graph.png'")

    asyncio.run(main())
