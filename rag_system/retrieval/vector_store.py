from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime
import faiss
import pickle
import os
from ..core.config import UnifiedConfig
from ..core.structures import RetrievalResult

class VectorStore:
    def __init__(self, config: UnifiedConfig, dimension: int):
        self.config = config
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents: List[Dict[str, Any]] = []

    def add_documents(self, documents: List[Dict[str, Any]]):
        vectors = [doc['embedding'] for doc in documents]
        self.index.add(np.array(vectors).astype('float32'))
        self.documents.extend(documents)

    def update_document(self, doc_id: str, new_doc: Dict[str, Any]):
        for i, doc in enumerate(self.documents):
            if doc['id'] == doc_id:
                old_vector = np.array([doc['embedding']]).astype('float32')
                new_vector = np.array([new_doc['embedding']]).astype('float32')
                self.index.remove_ids(np.array([i]))
                self.index.add(new_vector)
                self.documents[i] = new_doc
                break

    def delete_document(self, doc_id: str):
        for i, doc in enumerate(self.documents):
            if doc['id'] == doc_id:
                self.index.remove_ids(np.array([i]))
                del self.documents[i]
                break

    async def retrieve(self, query_vector: List[float], k: int, timestamp: Optional[datetime] = None, metadata_filter: Optional[Dict[str, Any]] = None) -> List[RetrievalResult]:
        query_vector_np = np.array([query_vector]).astype('float32')
        distances, indices = self.index.search(query_vector_np, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            doc = self.documents[idx]
            if (timestamp is None or doc['timestamp'] <= timestamp) and \
               (metadata_filter is None or all(doc.get(key) == value for key, value in metadata_filter.items())):
                result = RetrievalResult(
                    id=doc['id'],
                    content=doc['content'],
                    score=1 / (1 + distances[0][i]),  # Convert distance to similarity score
                    timestamp=doc['timestamp']
                )
                results.append(result)
        
        return results

    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        for doc in self.documents:
            if doc['id'] == doc_id:
                return doc
        return None

    def get_size(self) -> int:
        return len(self.documents)

    def save(self, file_path: str):
        with open(file_path, 'wb') as f:
            pickle.dump({
                'index': faiss.serialize_index(self.index),
                'documents': self.documents,
                'dimension': self.dimension
            }, f)

    @classmethod
    def load(cls, file_path: str, config: UnifiedConfig) -> 'VectorStore':
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        vector_store = cls(config, data['dimension'])
        vector_store.index = faiss.deserialize_index(data['index'])
        vector_store.documents = data['documents']
        
        return vector_store
