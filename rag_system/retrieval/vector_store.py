from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime
import pickle
import os

try:
    import faiss
    _FAISS_AVAILABLE = True
except Exception:  # pragma: no cover - faiss optional
    faiss = None  # type: ignore
    _FAISS_AVAILABLE = False
from ..core.config import UnifiedConfig
from ..core.structures import RetrievalResult

DEFAULT_DIMENSION = 768

class VectorStore:
    def __init__(self, config: Optional[UnifiedConfig] = None, dimension: int = DEFAULT_DIMENSION):
        """Create a VectorStore.

        The previous version of :class:`VectorStore` required ``config`` and
        ``dimension`` arguments.  Many parts of the codebase still instantiate
        this class without any parameters which resulted in ``TypeError`` being
        raised at runtime.  To maintain backwards compatibility we allow both
        arguments to be optional and provide sensible defaults.
        """

        self.config = config or UnifiedConfig()
        self.dimension = dimension
        if _FAISS_AVAILABLE:
            self.index = faiss.IndexFlatL2(dimension)
        else:
            self.index = None
            self.vectors: List[np.ndarray] = []
        self.documents: List[Dict[str, Any]] = []

    def add_documents(self, documents: List[Dict[str, Any]]):
        vectors = [doc['embedding'] for doc in documents]
        if _FAISS_AVAILABLE:
            self.index.add(np.array(vectors).astype('float32'))
        else:
            self.vectors.extend(np.array(vectors).astype('float32'))
        self.documents.extend(documents)

    def update_document(self, doc_id: str, new_doc: Dict[str, Any]):
        for i, doc in enumerate(self.documents):
            if doc['id'] == doc_id:
                new_vector = np.array([new_doc['embedding']]).astype('float32')
                if _FAISS_AVAILABLE:
                    self.index.remove_ids(np.array([i]))
                    self.index.add(new_vector)
                else:
                    self.vectors[i] = new_vector[0]
                self.documents[i] = new_doc
                break

    def delete_document(self, doc_id: str):
        for i, doc in enumerate(self.documents):
            if doc['id'] == doc_id:
                if _FAISS_AVAILABLE:
                    self.index.remove_ids(np.array([i]))
                else:
                    del self.vectors[i]
                del self.documents[i]
                break

    async def retrieve(self, query_vector: List[float], k: int, timestamp: Optional[datetime] = None, metadata_filter: Optional[Dict[str, Any]] = None) -> List[RetrievalResult]:
        query_vector_np = np.array([query_vector]).astype('float32')
        if _FAISS_AVAILABLE:
            distances, indices = self.index.search(query_vector_np, k)
            distances = distances[0]
            indices = indices[0]
        else:
            if not self.vectors:
                return []
            vectors_np = np.array(self.vectors)
            distances = np.linalg.norm(vectors_np - query_vector_np, axis=1)
            indices = np.argsort(distances)[:k]
            distances = distances[indices]
        
        results = []
        for i, idx in enumerate(indices):
            doc = self.documents[idx]
            if (timestamp is None or doc['timestamp'] <= timestamp) and \
               (metadata_filter is None or all(doc.get(key) == value for key, value in metadata_filter.items())):
                result = RetrievalResult(
                    id=doc['id'],
                    content=doc['content'],
                    score=1 / (1 + distances[i]) if distances[i] is not None else 1.0,
                    uncertainty=doc.get('uncertainty', 0.0),
                    timestamp=doc.get('timestamp'),
                    version=doc.get('version', 1)
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
            data = {
                'documents': self.documents,
                'dimension': self.dimension,
            }
            if _FAISS_AVAILABLE:
                data['index'] = faiss.serialize_index(self.index)
            else:
                data['vectors'] = self.vectors
            pickle.dump(data, f)

    @classmethod
    def load(cls, file_path: str, config: UnifiedConfig) -> 'VectorStore':
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        vector_store = cls(config, data['dimension'])
        if _FAISS_AVAILABLE and 'index' in data:
            vector_store.index = faiss.deserialize_index(data['index'])
        elif not _FAISS_AVAILABLE and 'vectors' in data:
            vector_store.vectors = data['vectors']
        vector_store.documents = data['documents']
        
        return vector_store
