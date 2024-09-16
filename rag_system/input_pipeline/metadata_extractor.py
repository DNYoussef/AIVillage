# rag_system/input_pipeline/metadata_extractor.py

from langchain.schema import Document
from typing import List
from ..core.agent_interface import AgentInterface

class MetadataExtractor:
    def __init__(self, agent: AgentInterface):
        self.agent = agent

    async def extract_metadata(self, documents: List[Document]) -> List[Document]:
        for doc in documents:
            metadata = await self.agent.extract_metadata(doc.page_content)
            doc.metadata.update(metadata)
        return documents
