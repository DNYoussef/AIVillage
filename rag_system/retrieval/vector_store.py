"""Vector store implementation for RAG system."""

from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime
import faiss
import pickle
import os
from ..core.base_component import BaseComponent
from ..core.config import UnifiedConfig
from ..core.structures import RetrievalResult
from ..utils.error_handling import log_and_handle_errors, ErrorContext

class VectorStore(BaseComponent):
    """Vector store for efficient similarity search."""
    
    def __init__(self, config: Optional[UnifiedConfig] = None, dimension: int = 768):  # Default BERT dimension
        """
        Initialize vector store.
        
        Args:
            config: Optional configuration instance
            dimension: Embedding dimension (default: 768 for BERT)
        """
        self.config = config or UnifiedConfig()
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents: List[Dict[str, Any]] = []
        self.initialized = False
    
    @log_and_handle_errors()
    async def initialize(self) -> None:
        """Initialize vector store."""
        if not self.initialized:
            # Load any saved index and documents
            if hasattr(self.config, 'vector_store_path') and os.path.exists(self.config.vector_store_path):
                self.load(self.config.vector_store_path, self.config)
            # Add a dummy document if store is empty to prevent index errors
            if not self.documents:
                self.add_texts(["Initialization document"])
            self.initialized = True
    
    @log_and_handle_errors()
    async def shutdown(self) -> None:
        """Shutdown vector store."""
        if self.initialized:
            # Save current index and documents
            if hasattr(self.config, 'vector_store_path'):
                self.save(self.config.vector_store_path)
            self.initialized = False
    
    @log_and_handle_errors()
    async def get_status(self) -> Dict[str, Any]:
        """Get component status."""
        return {
            "initialized": self.initialized,
            "index_size": self.index.ntotal,
            "document_count": len(self.documents),
            "dimension": self.dimension
        }
    
    @log_and_handle_errors()
    async def update_config(self, config: Dict[str, Any]) -> None:
        """Update component configuration."""
        self.config = config

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the vector store."""
        if not documents:
            return
        vectors = [doc['embedding'] for doc in documents]
        self.index.add(np.array(vectors).astype('float32'))
        self.documents.extend(documents)

    def update_document(self, doc_id: str, new_doc: Dict[str, Any]) -> None:
        """Update an existing document."""
        for i, doc in enumerate(self.documents):
            if doc['id'] == doc_id:
                old_vector = np.array([doc['embedding']]).astype('float32')
                new_vector = np.array([new_doc['embedding']]).astype('float32')
                self.index.remove_ids(np.array([i]))
                self.index.add(new_vector)
                self.documents[i] = new_doc
                break

    def delete_document(self, doc_id: str) -> None:
        """Delete a document from the vector store."""
        for i, doc in enumerate(self.documents):
            if doc['id'] == doc_id:
                self.index.remove_ids(np.array([i]))
                del self.documents[i]
                break

    async def retrieve(self,
                      query_vector: List[float],
                      k: int,
                      timestamp: Optional[datetime] = None,
                      metadata_filter: Optional[Dict[str, Any]] = None) -> List[RetrievalResult]:
        """
        Retrieve similar documents.
        
        Args:
            query_vector: Query embedding
            k: Number of results to retrieve
            timestamp: Optional timestamp filter
            metadata_filter: Optional metadata filter
            
        Returns:
            List of retrieval results
        """
        async with ErrorContext("VectorStore.retrieve"):
            if not self.initialized:
                await self.initialize()

            if not self.documents:
                return []

            # Ensure k is not larger than the number of documents
            k = min(k, len(self.documents))
            
            query_vector_np = np.array([query_vector]).astype('float32')
            distances, indices = self.index.search(query_vector_np, k)
            
            results = []
            if len(indices) > 0 and len(indices[0]) > 0:
                for i, idx in enumerate(indices[0]):
                    if idx >= 0 and idx < len(self.documents):  # Check index bounds
                        doc = self.documents[idx]
                        if (timestamp is None or doc.get('timestamp') <= timestamp) and \
                           (metadata_filter is None or all(doc.get(key) == value for key, value in metadata_filter.items())):
                            result = RetrievalResult(
                                id=doc['id'],
                                content=doc['content'],
                                score=1 / (1 + distances[0][i]),  # Convert distance to similarity score
                                timestamp=doc.get('timestamp')
                            )
                            results.append(result)
            
            return results

    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID."""
        for doc in self.documents:
            if doc['id'] == doc_id:
                return doc
        return None

    def get_size(self) -> int:
        """Get number of documents in store."""
        return len(self.documents)

    def save(self, file_path: str) -> None:
        """Save vector store to file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump({
                'index': faiss.serialize_index(self.index),
                'documents': self.documents,
                'dimension': self.dimension
            }, f)

    @classmethod
    def load(cls, file_path: str, config: UnifiedConfig) -> 'VectorStore':
        """Load vector store from file."""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        vector_store = cls(config, data['dimension'])
        vector_store.index = faiss.deserialize_index(data['index'])
        vector_store.documents = data['documents']
        
        return vector_store

    def add_texts(self, texts: List[str]) -> None:
        """Add texts to vector store."""
        if not texts:
            return
            
        documents = []
        for i, text in enumerate(texts):
            doc = {
                'id': f'doc_{len(self.documents) + i}',
                'content': text,
                'embedding': np.random.randn(self.dimension).astype('float32'),  # Placeholder
                'timestamp': datetime.now()
            }
            documents.append(doc)
        self.add_documents(documents)
