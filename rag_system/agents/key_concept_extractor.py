from typing import List, Dict
from rag_system.core.agent_interface import AgentInterface
from utils.embedding import BERTEmbeddingModel
from nlp.named_entity_recognition import NamedEntityRecognizer

class KeyConceptExtractorAgent(AgentInterface):
    """
    Agent responsible for extracting key concepts from text using advanced NLP techniques.
    """

    def __init__(self):
        super().__init__()
        self.embedding_model = BERTEmbeddingModel()
        self.named_entity_recognizer = NamedEntityRecognizer()

    def extract_key_concepts(self, text: str) -> Dict[str, List[str]]:
        """
        Extract key concepts from the given text.

        Args:
            text (str): Input text from which to extract key concepts.

        Returns:
            Dict[str, List[str]]: A dictionary containing extracted entities and keywords.
        """
        entities = self.named_entity_recognizer.recognize(text)
        embeddings = self.embedding_model.encode(text)
        keywords = self._extract_keywords_from_embeddings(embeddings)
        return {
            'entities': entities,
            'keywords': keywords
        }

    def _extract_keywords_from_embeddings(self, embeddings) -> List[str]:
        # TODO: Implement keyword extraction logic using embeddings
        return []
