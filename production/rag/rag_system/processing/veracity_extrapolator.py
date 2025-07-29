# rag_system/processing/veracity_extrapolator.py


from .batch_processor import BatchProcessor
from .confidence_estimator import estimate_confidence
from .prompt_constructor import construct_extrapolation_prompt


class VeracityExtrapolator:
    def __init__(self, knowledge_graph, llm, config):
        """Initialize the VeracityExtrapolator.

        :param knowledge_graph: An instance of the knowledge graph.
        :param llm: An instance of the language model to use for extrapolation.
        :param config: Configuration object containing settings like CONFIDENCE_THRESHOLD.
        """
        self.knowledge_graph = knowledge_graph
        self.llm = llm
        self.config = config
        self.batch_processor = BatchProcessor(knowledge_graph, llm)

    async def extrapolate(self, entity1: str, relation: str, entity2: str) -> tuple[str, float]:
        """Extrapolate the veracity of a relation between two entities.

        :param entity1: The first entity.
        :param relation: The relation to assess.
        :param entity2: The second entity.
        :return: A tuple containing the extrapolation result and confidence score.
        """
        known_facts = self.knowledge_graph.get_related_facts(entity1, entity2)
        prompt = construct_extrapolation_prompt(entity1, relation, entity2, known_facts)
        extrapolation = await self.llm.generate(prompt)
        confidence = estimate_confidence(extrapolation)
        return extrapolation, confidence

    async def extrapolate_group_connections(self, entity_group1: list[str], entity_group2: list[str]) -> list[tuple[str, str, str, float]]:
        """Extrapolate potential relations between two groups of entities.

        :param entity_group1: The first group of entities.
        :param entity_group2: The second group of entities.
        :return: A list of tuples containing (entity1, relation, entity2, confidence).
        """
        return await self.batch_processor.extrapolate_group_connections(
            entity_group1, entity_group2, self.config.CONFIDENCE_THRESHOLD
        )

    def update_knowledge_graph(self, extrapolated_connections: list[tuple[str, str, str, float]]):
        """Update the knowledge graph with extrapolated connections.

        :param extrapolated_connections: A list of tuples containing (entity1, relation, entity2, confidence).
        """
        for entity1, relation, entity2, confidence in extrapolated_connections:
            self.knowledge_graph.add_relation(entity1, relation, entity2, confidence)

    async def iterative_extrapolation(self, initial_entities: list[str], max_iterations: int = 3):
        """Perform iterative extrapolation to discover new connections.

        :param initial_entities: A list of initial entities to start the extrapolation from.
        :param max_iterations: Maximum number of iterations to perform.
        :return: A dictionary of newly discovered connections.
        """
        return await self.batch_processor.iterative_extrapolation(initial_entities, max_iterations)
