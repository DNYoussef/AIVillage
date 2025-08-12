from rag_system.core.agent_interface import AgentInterface
from rag_system.utils.embedding import BERTEmbeddingModel

from nlp.named_entity_recognition import NamedEntityRecognizer


class KeyConceptExtractorAgent(AgentInterface):
    """Agent responsible for extracting key concepts from text using advanced NLP techniques."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding_model = BERTEmbeddingModel()
        self.named_entity_recognizer = NamedEntityRecognizer()

    def extract_key_concepts(self, text: str) -> dict[str, list[str]]:
        """Extract key concepts from the given text.

        Args:
            text (str): Input text from which to extract key concepts.

        Returns:
            Dict[str, List[str]]: A dictionary containing extracted entities and keywords.
        """
        entities = self.named_entity_recognizer.recognize(text)
        embeddings = self.embedding_model.encode(text)
        keywords = self._extract_keywords_from_embeddings(embeddings)
        return {"entities": entities, "keywords": keywords}

    def _extract_keywords_from_embeddings(self, embeddings) -> list[str]:
        """Derive simple keywords from token embeddings.

        The ``BERTEmbeddingModel.encode`` method returns a tuple of tokens and
        the corresponding embedding tensor.  This helper identifies tokens that
        are most similar to the mean sentence embedding and returns them as
        keywords.  Special tokens are ignored.
        """
        tokens, token_embeddings = embeddings

        if len(tokens) == 0:
            return []

        import torch.nn.functional as F

        sentence_emb = token_embeddings.mean(dim=0)
        similarities = F.cosine_similarity(
            token_embeddings, sentence_emb.unsqueeze(0), dim=1
        )
        topk = similarities.topk(min(5, len(tokens))).indices.tolist()

        keywords = []
        for idx in topk:
            token = tokens[idx]
            if token.startswith("[") and token.endswith("]"):
                continue
            if token.startswith("##"):
                token = token[2:]
            keywords.append(token)
            if len(keywords) == 5:
                break
        return keywords
