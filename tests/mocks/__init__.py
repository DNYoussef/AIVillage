"""
Consolidated test mocks for AIVillage testing infrastructure.

This module provides comprehensive mock implementations for complex dependencies 
that are expensive to initialize or require external resources during testing.
"""

import asyncio
import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock
from typing import Dict, Any, List, Optional

# Add project paths for comprehensive import resolution
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))  
sys.path.insert(0, str(project_root / "packages"))

def install_mocks() -> None:
    """Install comprehensive mock modules into sys.modules when imports fail.

    This function provides fallback implementations for all major dependencies,
    ensuring tests can run without requiring heavy external dependencies.
    """
    
    # Core ML/AI Libraries
    ml_modules = [
        'torch', 'torch.nn', 'torch.optim', 'torch.utils.data',
        'transformers', 'sentence_transformers', 'huggingface_hub',
        'numpy', 'scipy', 'pandas', 'scikit_learn',
        'chromadb', 'faiss', 'qdrant_client', 'pinecone'
    ]
    
    # API and Service Libraries  
    api_modules = [
        'openai', 'anthropic', 'langchain', 'langchain_community',
        'fastapi', 'uvicorn', 'httpx', 'aiohttp', 'websockets',
        'redis', 'sqlalchemy', 'asyncpg', 'psycopg2', 'neo4j'
    ]
    
    # Monitoring and Analytics
    monitoring_modules = [
        'wandb', 'prometheus_client', 'sentry_sdk', 'structlog',
        'rich', 'plotly', 'matplotlib', 'streamlit', 'gradio'
    ]
    
    # Infrastructure and Deployment
    infra_modules = [
        'kubernetes', 'docker', 'celery', 'pydantic',
        'click', 'typer', 'asyncio_mqtt', 'scapy'
    ]
    
    # Networking and Security
    network_modules = [
        'bluetooth', 'pybluez', 'aioquic', 'cryptography', 'nacl'
    ]
    
    all_mock_modules = ml_modules + api_modules + monitoring_modules + infra_modules + network_modules
    
    for module_name in all_mock_modules:
        try:
            importlib.import_module(module_name)
        except Exception:
            if module_name not in sys.modules:
                # Create smart mock with common attributes
                mock = MagicMock()
                mock.__version__ = "mock-version"
                mock.__name__ = module_name
                sys.modules[module_name] = mock

    # Specialized RAG system mocks
    rag_modules = [
        "rag", "rag.core", "rag.core.hyper_rag", "rag.base",
        "rag.edge_device_bridge", "rag.p2p_network_bridge", 
        "rag.fog_compute_bridge", "rag.integration"
    ]
    
    for mod in rag_modules:
        try:
            importlib.import_module(mod)
        except Exception:
            sys.modules.setdefault(mod, MagicMock())

    # Core system modules  
    system_modules = [
        "rag_system", "rag_system.core.config", "rag_system.core.pipeline",
        "rag_system.retrieval.vector_store", "rag_system.tracking.unified_knowledge_tracker",
        "rag_system.utils.error_handling", "services", "services.gateway", "services.twin",
        "agent_forge.memory_manager", "agent_forge.wandb_manager"
    ]
    
    for mod in system_modules:
        try:
            importlib.import_module(mod)
        except Exception:
            sys.modules.setdefault(mod, MagicMock())

def create_mock_rag_config():
    """Create a comprehensive mock RAG configuration for testing."""
    return {
        'enable_hippo_rag': True,
        'enable_graph_rag': True, 
        'enable_context_rag': True,
        'vector_db_type': 'memory',
        'embedding_model': 'mock-sentence-transformers/all-MiniLM-L6-v2',
        'database_path': ':memory:',
        'max_context_length': 4000,
        'chunk_size': 512,
        'chunk_overlap': 50,
        'retrieval_top_k': 5,
        'similarity_threshold': 0.7,
        'enable_reranking': True,
        'cache_embeddings': True,
        'batch_size': 32,
        'max_retries': 3,
        'timeout_seconds': 30.0
    }

class MockRAGSystem:
    """Comprehensive mock RAG system for testing without heavy dependencies."""
    
    def __init__(self, config=None):
        self.config = config or create_mock_rag_config()
        self.vector_store = {}
        self.documents = []
        self.is_initialized = False
        self.query_history = []
    
    async def initialize(self):
        """Mock initialization with realistic delay."""
        await asyncio.sleep(0.01)  # Simulate startup time
        self.is_initialized = True
        return True
    
    async def add_documents(self, documents: List[Dict[str, Any]]):
        """Mock document ingestion."""
        self.documents.extend(documents)
        return len(documents)
    
    async def query(self, query: str, mode: str = "hybrid") -> Dict[str, Any]:
        """Mock query processing with comprehensive response."""
        self.query_history.append({'query': query, 'mode': mode})
        
        return {
            'answer': f'Mock comprehensive answer for: {query}',
            'sources': [
                {'content': 'Mock source 1 with relevant information', 'score': 0.95, 'id': 'doc1'},
                {'content': 'Mock source 2 with supporting details', 'score': 0.87, 'id': 'doc2'},
                {'content': 'Mock source 3 with additional context', 'score': 0.82, 'id': 'doc3'}
            ],
            'metadata': {
                'mode': mode, 
                'processing_time': 0.15,
                'total_documents': len(self.documents),
                'retrieval_method': 'mock_hybrid_search',
                'confidence': 0.88
            },
            'retrieved_chunks': 3,
            'query_complexity': 'medium'
        }
    
    def get_stats(self):
        """Get system statistics."""
        return {
            'total_documents': len(self.documents),
            'total_queries': len(self.query_history),
            'system_health': 'healthy',
            'uptime': '100%'
        }

class MockP2PNetwork:
    """Mock P2P network for testing distributed features."""
    
    def __init__(self, node_id: str = "test_node"):
        self.node_id = node_id
        self.peers = {}
        self.messages = []
        self.is_running = False
        self.network_health = 1.0
    
    async def start(self):
        """Mock network start."""
        await asyncio.sleep(0.05)
        self.is_running = True
        return True
    
    async def stop(self):
        """Mock network stop."""
        self.is_running = False
        return True
    
    async def send_message(self, peer_id: str, message: Dict[str, Any]):
        """Mock message sending with network simulation."""
        self.messages.append({
            'to': peer_id,
            'from': self.node_id,
            'message': message,
            'timestamp': asyncio.get_event_loop().time(),
            'status': 'delivered'
        })
        return True
    
    def get_peers(self) -> List[str]:
        """Mock peer discovery."""
        return list(self.peers.keys())
    
    def get_network_status(self):
        """Get network health status."""
        return {
            'connected_peers': len(self.peers),
            'messages_sent': len(self.messages),
            'network_health': self.network_health,
            'is_running': self.is_running
        }

# Global mock instances for easy access
mock_rag = MockRAGSystem()
mock_p2p = MockP2PNetwork()

# Auto-install mocks when imported
install_mocks()
