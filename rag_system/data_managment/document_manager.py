# rag_system/data_management/document_manager.py

from typing import Dict, Any
from ..core.config import RAGConfig
from ..retrieval.vector_store import VectorStore
from ..retrieval.graph_store import GraphStore
from ..data_manager.data_manager import DataManager

class DocumentManager:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.vector_store = VectorStore(config)
        self.graph_store = GraphStore(config)
        self.data_manager = DataManager(config)

    async def add_document(self, content: bytes, filename: str) -> Dict[str, str]:
        try:
            doc = await self.data_manager.process_document(content, filename)
            await self.vector_store.add_document(doc)
            await self.graph_store.add_document(doc)
            return {"message": "Document added successfully"}
        except Exception as e:
            raise Exception(f"Error adding document: {str(e)}")

    async def update_document(self, doc_id: str, content: str, metadata: dict) -> Dict[str, str]:
        try:
            updated_doc = await self.data_manager.update_document(doc_id, content, metadata)
            await self.vector_store.update_document(updated_doc)
            await self.graph_store.update_document(updated_doc)
            return {"message": "Document updated successfully"}
        except Exception as e:
            raise Exception(f"Error updating document: {str(e)}")

    async def delete_document(self, doc_id: str) -> Dict[str, str]:
        try:
            await self.data_manager.delete_document(doc_id)
            await self.vector_store.delete_document(doc_id)
            await self.graph_store.delete_document(doc_id)
            return {"message": "Document deleted successfully"}
        except Exception as e:
            raise Exception(f"Error deleting document: {str(e)}")
