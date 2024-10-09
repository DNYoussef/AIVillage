import asyncio
from typing import Dict, Any
from core.config import RAGConfig
from core.pipeline import RAGPipeline

class RAGAgent:
    def __init__(self, agent_id: str, config: RAGConfig = None):
        self.agent_id = agent_id
        self.config = config or RAGConfig()
        self.pipeline = RAGPipeline(self.config)

    async def query_rag(self, query: str) -> Dict[str, Any]:
        """
        Submit a query to the RAG system and receive a structured response.
        """
        result = await self.pipeline.process_query(query)
        return result

    async def add_document(self, content: str, filename: str):
        """
        Add a new document to the RAG system.
        """
        from rag_system.data_managment.document_manager import DocumentManager
        manager = DocumentManager(self.config)
        await manager.add_document(content, filename)
