# rag_system/processing/batch_processor.py


from .confidence_estimator import estimate_confidence
from .prompt_constructor import construct_extrapolation_prompt


class BatchProcessor:
    def __init__(self, knowledge_graph, llm) -> None:
        self.knowledge_graph = knowledge_graph
        self.llm = llm

    async def batch_extrapolate(
        self, entity_pairs: list[tuple[str, str, str]]
    ) -> list[tuple[str, str, str, str, float]]:
        """Perform batch extrapolation for multiple entity pairs and relations.

        :param entity_pairs: A list of tuples containing (entity1, relation, entity2).
        :return: A list of tuples containing (entity1, relation, entity2, extrapolation, confidence).
        """
        results = []
        for entity1, relation, entity2 in entity_pairs:
            known_facts = self.knowledge_graph.get_related_facts(entity1, entity2)
            prompt = construct_extrapolation_prompt(entity1, relation, entity2, known_facts)
            extrapolation = await self.llm.generate(prompt)
            confidence = estimate_confidence(extrapolation)
            results.append((entity1, relation, entity2, extrapolation, confidence))
        return results

    async def extrapolate_group_connections(
        self,
        entity_group1: list[str],
        entity_group2: list[str],
        confidence_threshold: float,
    ) -> list[tuple[str, str, str, float]]:
        """Extrapolate potential relations between two groups of entities.

        :param entity_group1: The first group of entities.
        :param entity_group2: The second group of entities.
        :param confidence_threshold: The minimum confidence level to include a connection.
        :return: A list of tuples containing (entity1, relation, entity2, confidence).
        """
        potential_relations = self.knowledge_graph.get_potential_relations(entity_group1, entity_group2)

        entity_pairs = [
            (entity1, relation, entity2)
            for entity1 in entity_group1
            for entity2 in entity_group2
            for relation in potential_relations
        ]

        batch_results = await self.batch_extrapolate(entity_pairs)

        extrapolated_connections = [
            (entity1, relation, entity2, confidence)
            for entity1, relation, entity2, _, confidence in batch_results
            if confidence > confidence_threshold
        ]

        return extrapolated_connections

    async def iterative_extrapolation(self, initial_entities: list[str], max_iterations: int = 3) -> dict:
        """Perform iterative extrapolation to discover new connections.

        :param initial_entities: A list of initial entities to start the extrapolation from.
        :param max_iterations: Maximum number of iterations to perform.
        :return: A dictionary of newly discovered connections.
        """
        discovered_connections = {}
        current_entities = initial_entities

        for _ in range(max_iterations):
            new_connections = await self.extrapolate_group_connections(current_entities, current_entities, 0.5)

            for entity1, relation, entity2, confidence in new_connections:
                key = (entity1, relation, entity2)
                if key not in discovered_connections:
                    discovered_connections[key] = confidence

            # Update current_entities with newly discovered entities
            current_entities = list(
                set(current_entities + [conn[0] for conn in new_connections] + [conn[2] for conn in new_connections])
            )

        return discovered_connections
