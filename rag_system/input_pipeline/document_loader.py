# rag_system/input_pipeline/document_loader.py

from langchain.document_loaders import UnstructuredFileLoader
from typing import List, Union
from langchain.schema import Document

class DocumentLoader:
    def load(self, file_paths: Union[str, List[str]]) -> List[Document]:
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        documents = []
        for file_path in file_paths:
            loader = UnstructuredFileLoader(file_path)
            documents.extend(loader.load())
        return documents
