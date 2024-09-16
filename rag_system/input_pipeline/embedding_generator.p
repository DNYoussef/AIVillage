# rag_system/input_pipeline/embedding_generator.py

from langchain.schema import Document
from typing import List
from ..embeddings.base_embedding import EmbeddingModel

class EmbeddingGenerator:
    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model

    def generate_embeddings(self, documents: List[Document]) -> List[Document]:
        for doc in documents:
            doc.metadata['embedding'] = self.embedding_model.embed_query(doc.page_content)
        return documents

