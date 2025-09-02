"""
Specialized RAG system mocks for comprehensive testing.

These mocks provide realistic behavior patterns for testing RAG workflows
without requiring heavy dependencies or external services.
"""

import asyncio
import hashlib
import json
import random
import time
from typing import Dict, List, Any, Optional, Union
from unittest.mock import AsyncMock, MagicMock
from dataclasses import dataclass, field
from enum import Enum


class MockQueryMode(Enum):
    """Mock query modes matching real RAG system."""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    GRAPH = "graph"
    CONTEXTUAL = "contextual"


class MockMemoryType(Enum):
    """Mock memory types for RAG testing."""
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    WORKING = "working"


@dataclass
class MockRAGConfig:
    """Mock RAG configuration with realistic defaults."""
    enable_hippo_rag: bool = True
    enable_graph_rag: bool = True
    enable_context_rag: bool = True
    vector_db_type: str = "memory"
    embedding_model: str = "mock-sentence-transformers/all-MiniLM-L6-v2"
    database_path: str = ":memory:"
    max_context_length: int = 4000
    chunk_size: int = 512
    chunk_overlap: int = 50
    retrieval_top_k: int = 5
    similarity_threshold: float = 0.7
    enable_reranking: bool = True
    cache_embeddings: bool = True
    batch_size: int = 32
    max_retries: int = 3
    timeout_seconds: float = 30.0


@dataclass 
class MockDocument:
    """Mock document structure for RAG testing."""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    chunk_id: Optional[str] = None
    source: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class MockQueryResult:
    """Mock query result structure."""
    answer: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.85
    processing_time: float = 0.1
    mode_used: str = "hybrid"
    retrieved_chunks: int = 5


class MockHyperRAG:
    """Comprehensive mock HyperRAG system with realistic behavior."""
    
    def __init__(self, config: Optional[MockRAGConfig] = None):
        self.config = config or MockRAGConfig()
        self.documents: List[MockDocument] = []
        self.vector_store: Dict[str, List[float]] = {}
        self.graph_store: Dict[str, List[str]] = {}  # Simple adjacency list
        self.is_initialized = False
        self.query_history: List[Dict[str, Any]] = []
        self._embedding_cache: Dict[str, List[float]] = {}
    
    async def initialize(self) -> bool:
        """Mock initialization with realistic startup time."""
        await asyncio.sleep(0.1)  # Simulate startup time
        self.is_initialized = True
        return True
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """Mock document ingestion with chunking simulation."""
        added_count = 0
        
        for doc_data in documents:
            # Create mock document
            doc_id = doc_data.get('id', hashlib.md5(doc_data.get('content', '').encode()).hexdigest())
            doc = MockDocument(
                id=doc_id,
                content=doc_data.get('content', ''),
                metadata=doc_data.get('metadata', {}),
                source=doc_data.get('source'),
                embedding=await self._mock_embed(doc_data.get('content', ''))
            )
            
            # Simulate chunking
            content = doc.content
            chunk_size = self.config.chunk_size
            overlap = self.config.chunk_overlap
            
            for i in range(0, len(content), chunk_size - overlap):
                chunk_content = content[i:i + chunk_size]
                if chunk_content.strip():
                    chunk_id = f"{doc_id}_chunk_{i // (chunk_size - overlap)}"
                    chunk_doc = MockDocument(
                        id=chunk_id,
                        content=chunk_content,
                        metadata={**doc.metadata, 'chunk_index': i // (chunk_size - overlap)},
                        chunk_id=chunk_id,
                        source=doc.source,
                        embedding=await self._mock_embed(chunk_content)
                    )
                    self.documents.append(chunk_doc)
                    self.vector_store[chunk_id] = chunk_doc.embedding
                    added_count += 1
        
        return added_count
    
    async def query(self, query: str, mode: str = "hybrid", **kwargs) -> MockQueryResult:
        """Mock query processing with realistic result generation."""
        start_time = time.time()
        
        # Record query
        self.query_history.append({
            'query': query,
            'mode': mode,
            'timestamp': start_time,
            'kwargs': kwargs
        })
        
        # Mock query processing delay
        await asyncio.sleep(random.uniform(0.05, 0.15))
        
        # Generate mock results based on query mode
        if mode == MockQueryMode.SEMANTIC.value:
            sources = await self._mock_semantic_search(query)
        elif mode == MockQueryMode.KEYWORD.value:
            sources = await self._mock_keyword_search(query)
        elif mode == MockQueryMode.GRAPH.value:
            sources = await self._mock_graph_search(query)
        else:  # hybrid
            sources = await self._mock_hybrid_search(query)
        
        # Generate mock answer
        answer = self._generate_mock_answer(query, sources)
        
        processing_time = time.time() - start_time
        
        return MockQueryResult(
            answer=answer,
            sources=sources,
            metadata={
                'query_mode': mode,
                'total_documents': len(self.documents),
                'embedding_model': self.config.embedding_model,
                'retrieval_method': mode
            },
            confidence=random.uniform(0.75, 0.95),
            processing_time=processing_time,
            mode_used=mode,
            retrieved_chunks=len(sources)
        )
    
    async def _mock_embed(self, text: str) -> List[float]:
        """Mock text embedding generation."""
        if self.config.cache_embeddings and text in self._embedding_cache:
            return self._embedding_cache[text]
        
        # Generate deterministic but realistic-looking embeddings
        hash_val = hashlib.md5(text.encode()).hexdigest()
        embedding = []
        for i in range(0, 384, 8):  # 384-dim embedding
            chunk = hash_val[i % len(hash_val):(i % len(hash_val)) + 8]
            val = int(chunk or '0', 16) / (16**8)  # Normalize to 0-1
            embedding.append((val - 0.5) * 2)  # Scale to -1 to 1
        
        # Pad to 384 dimensions
        while len(embedding) < 384:
            embedding.append(random.uniform(-0.1, 0.1))
        
        if self.config.cache_embeddings:
            self._embedding_cache[text] = embedding
        
        return embedding
    
    async def _mock_semantic_search(self, query: str) -> List[Dict[str, Any]]:
        """Mock semantic similarity search."""
        query_embedding = await self._mock_embed(query)
        results = []
        
        for doc in self.documents[:self.config.retrieval_top_k]:
            # Mock cosine similarity
            similarity = random.uniform(0.6, 0.95) 
            results.append({
                'content': doc.content[:200] + '...' if len(doc.content) > 200 else doc.content,
                'score': similarity,
                'source': doc.source or 'mock_document',
                'metadata': doc.metadata,
                'id': doc.id
            })
        
        return sorted(results, key=lambda x: x['score'], reverse=True)
    
    async def _mock_keyword_search(self, query: str) -> List[Dict[str, Any]]:
        """Mock keyword-based search."""
        query_terms = query.lower().split()
        results = []
        
        for doc in self.documents:
            content_lower = doc.content.lower()
            matches = sum(1 for term in query_terms if term in content_lower)
            if matches > 0:
                score = matches / len(query_terms)
                results.append({
                    'content': doc.content[:200] + '...' if len(doc.content) > 200 else doc.content,
                    'score': score,
                    'source': doc.source or 'mock_document',
                    'metadata': doc.metadata,
                    'id': doc.id,
                    'matches': matches
                })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:self.config.retrieval_top_k]
    
    async def _mock_graph_search(self, query: str) -> List[Dict[str, Any]]:
        """Mock graph-based search."""
        # Simulate graph traversal
        results = []
        
        # Mock connected documents
        for i, doc in enumerate(self.documents[:self.config.retrieval_top_k]):
            # Simulate graph connectivity score
            connectivity_score = random.uniform(0.5, 0.9)
            results.append({
                'content': doc.content[:200] + '...' if len(doc.content) > 200 else doc.content,
                'score': connectivity_score,
                'source': doc.source or 'mock_document',
                'metadata': {**doc.metadata, 'graph_hops': random.randint(1, 3)},
                'id': doc.id,
                'connections': random.randint(2, 8)
            })
        
        return results
    
    async def _mock_hybrid_search(self, query: str) -> List[Dict[str, Any]]:
        """Mock hybrid search combining multiple methods."""
        semantic_results = await self._mock_semantic_search(query)
        keyword_results = await self._mock_keyword_search(query)
        
        # Combine and rerank
        combined_results = {}
        
        for result in semantic_results:
            doc_id = result['id']
            combined_results[doc_id] = result
            combined_results[doc_id]['semantic_score'] = result['score']
            combined_results[doc_id]['hybrid_score'] = result['score'] * 0.6
        
        for result in keyword_results:
            doc_id = result['id']
            if doc_id in combined_results:
                combined_results[doc_id]['keyword_score'] = result['score']
                combined_results[doc_id]['hybrid_score'] += result['score'] * 0.4
            else:
                combined_results[doc_id] = result
                combined_results[doc_id]['keyword_score'] = result['score']
                combined_results[doc_id]['hybrid_score'] = result['score'] * 0.4
        
        # Sort by hybrid score
        final_results = list(combined_results.values())
        final_results.sort(key=lambda x: x.get('hybrid_score', x['score']), reverse=True)
        
        return final_results[:self.config.retrieval_top_k]
    
    def _generate_mock_answer(self, query: str, sources: List[Dict[str, Any]]) -> str:
        """Generate a realistic mock answer based on query and sources."""
        if not sources:
            return f"I don't have enough information to answer '{query}' based on the available documents."
        
        # Generate contextual answer
        query_lower = query.lower()
        if 'what' in query_lower:
            return f"Based on the available information, {query} refers to concepts found in {len(sources)} relevant sources. The main points include information from the retrieved documents that best match your query."
        elif 'how' in query_lower:
            return f"To address '{query}', the process involves several steps as described in the relevant sources. The methodology is outlined in the retrieved documentation."
        elif 'why' in query_lower:
            return f"The reason for '{query}' can be understood through the analysis of {len(sources)} relevant sources, which provide context and explanation for this topic."
        elif 'when' in query_lower:
            return f"Regarding the timing of '{query}', the available information suggests specific timeframes as documented in the retrieved sources."
        elif 'where' in query_lower:
            return f"The location or context for '{query}' is described in the available documentation, with details found across {len(sources)} relevant sources."
        else:
            return f"In response to '{query}', the available information from {len(sources)} sources provides comprehensive coverage of this topic with relevant details and context."


class MockEdgeDeviceRAGBridge:
    """Mock Edge Device RAG Bridge for mobile/IoT testing."""
    
    def __init__(self, device_profile: Optional[Dict[str, Any]] = None):
        self.device_profile = device_profile or {
            'device_type': 'smartphone',
            'cpu_cores': 8,
            'ram_mb': 6144,
            'storage_mb': 64000,
            'battery_level': 85,
            'network_type': '5G',
            'constraints': ['low_power', 'limited_storage']
        }
        self.local_cache = {}
        self.optimization_enabled = True
    
    async def process_query(self, query: str, constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Mock edge-optimized query processing."""
        constraints = constraints or {}
        
        # Simulate edge constraints
        processing_time = random.uniform(0.05, 0.2)  # Faster on-device
        await asyncio.sleep(processing_time)
        
        return {
            'answer': f'Edge-optimized response for: {query}',
            'device_id': self.device_profile.get('device_id', 'test_edge_device'),
            'processing_time': processing_time,
            'local_cache_hit': query in self.local_cache,
            'optimization_applied': self.optimization_enabled,
            'constraints_applied': list(constraints.keys()) if constraints else [],
            'device_metrics': {
                'cpu_usage': random.uniform(15, 45),
                'memory_usage': random.uniform(256, 1024),
                'battery_drain': random.uniform(0.5, 2.0)
            }
        }


class MockP2PNetworkRAGBridge:
    """Mock P2P Network RAG Bridge for decentralized testing."""
    
    def __init__(self, network_config: Optional[Dict[str, Any]] = None):
        self.network_config = network_config or {
            'node_id': 'test_p2p_node',
            'max_peers': 10,
            'consensus_threshold': 0.7,
            'routing_protocol': 'DHT',
            'encryption_enabled': True
        }
        self.connected_peers = []
        self.knowledge_cache = {}
    
    async def distributed_query(self, query: str, peer_count: int = 3) -> Dict[str, Any]:
        """Mock distributed P2P query processing."""
        # Simulate network coordination time
        await asyncio.sleep(random.uniform(0.2, 0.5))
        
        # Mock peer contributions
        peer_contributions = []
        for i in range(min(peer_count, 5)):
            peer_id = f"peer_{i+1}"
            contribution = {
                'peer_id': peer_id,
                'answer_fragment': f'P2P contribution {i+1} for: {query}',
                'confidence': random.uniform(0.6, 0.9),
                'latency': random.uniform(50, 200),
                'reputation_score': random.uniform(0.7, 1.0)
            }
            peer_contributions.append(contribution)
        
        # Mock consensus
        consensus_score = sum(c['confidence'] for c in peer_contributions) / len(peer_contributions)
        
        return {
            'answer': f'Consensus answer from {peer_count} peers for: {query}',
            'peer_contributions': peer_contributions,
            'consensus_score': consensus_score,
            'network_latency': random.uniform(100, 400),
            'participating_peers': peer_count,
            'network_health': random.uniform(0.8, 1.0)
        }


class MockFogComputeBridge:
    """Mock Fog Computing RAG Bridge for distributed processing."""
    
    def __init__(self, fog_nodes: Optional[List[Dict[str, Any]]] = None):
        self.fog_nodes = fog_nodes or [
            {'node_id': 'fog_node_1', 'region': 'us-west', 'capacity': 100, 'load': 0.3},
            {'node_id': 'fog_node_2', 'region': 'us-east', 'capacity': 80, 'load': 0.5},
            {'node_id': 'fog_node_3', 'region': 'eu-central', 'capacity': 120, 'load': 0.2}
        ]
        self.active_workloads = {}
    
    async def process_workload(self, workload: Dict[str, Any]) -> Dict[str, Any]:
        """Mock fog computing workload processing."""
        workload_type = workload.get('type', 'rag_query')
        complexity = workload.get('complexity', 'medium')
        
        # Select optimal fog node
        best_node = min(self.fog_nodes, key=lambda n: n['load'])
        
        # Simulate processing time based on complexity
        processing_times = {'low': (0.1, 0.3), 'medium': (0.3, 0.8), 'high': (0.8, 2.0)}
        min_time, max_time = processing_times.get(complexity, (0.3, 0.8))
        processing_time = random.uniform(min_time, max_time)
        
        await asyncio.sleep(processing_time)
        
        workload_id = f"workload_{len(self.active_workloads)}"
        result = {
            'workload_id': workload_id,
            'result': {
                'output': f'Fog-processed result for {workload_type}',
                'quality_score': random.uniform(0.8, 0.95),
                'optimizations_applied': ['load_balancing', 'edge_caching']
            },
            'processing_node': best_node['node_id'],
            'processing_time': processing_time,
            'resource_usage': {
                'cpu_time': processing_time * random.uniform(0.5, 1.5),
                'memory_mb': random.uniform(128, 512),
                'network_kb': random.uniform(50, 200)
            }
        }
        
        self.active_workloads[workload_id] = result
        return result


# Factory functions for easy mock creation
def create_mock_rag(config: Optional[MockRAGConfig] = None) -> MockHyperRAG:
    """Factory function to create mock RAG system."""
    return MockHyperRAG(config)

def create_mock_edge_bridge(device_profile: Optional[Dict[str, Any]] = None) -> MockEdgeDeviceRAGBridge:
    """Factory function to create mock edge device bridge."""
    return MockEdgeDeviceRAGBridge(device_profile)

def create_mock_p2p_bridge(network_config: Optional[Dict[str, Any]] = None) -> MockP2PNetworkRAGBridge:
    """Factory function to create mock P2P network bridge."""
    return MockP2PNetworkRAGBridge(network_config)

def create_mock_fog_bridge(fog_nodes: Optional[List[Dict[str, Any]]] = None) -> MockFogComputeBridge:
    """Factory function to create mock fog compute bridge."""
    return MockFogComputeBridge(fog_nodes)