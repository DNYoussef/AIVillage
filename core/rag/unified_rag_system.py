"""
AIVillage Unified RAG System
Combines the best of all RAG implementations with MCP integration

Key Features:
1. Advanced Ingestion (PDF/DOCX/HTML processing)
2. HippoRAG (Hippocampus-inspired episodic memory)
3. Vector RAG with Dual Context Tagging
4. Bayesian Knowledge Graph RAG
5. Cognitive Nexus Search Integration
6. Creative Graph Search for Brainstorming
7. Missing Node Detection for Knowledge Gaps
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import logging
from pathlib import Path
from typing import Any

import networkx as nx

# Core dependencies
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class QueryMode(Enum):
    """Unified query processing modes"""
    FAST = "fast"                    # Quick retrieval with basic processing
    BALANCED = "balanced"            # Standard multi-source retrieval
    COMPREHENSIVE = "comprehensive"  # Full multi-modal analysis
    CREATIVE = "creative"           # Brainstorming and creative connections
    ANALYTICAL = "analytical"       # Deep reasoning with Bayesian inference
    DISTRIBUTED = "distributed"     # P2P distributed processing
    HIPPO = "hippo"                 # Hippocampus-inspired memory retrieval
    MISSING_NODES = "missing_nodes" # Focus on knowledge gap discovery

class ContextTag(Enum):
    """Dual context tagging system"""
    CONTENT_CONTEXT = "content"      # Direct content context
    SEMANTIC_CONTEXT = "semantic"    # Semantic relationship context
    TEMPORAL_CONTEXT = "temporal"    # Time-based context
    SPATIAL_CONTEXT = "spatial"      # Location-based context
    CAUSAL_CONTEXT = "causal"        # Cause-effect relationships
    ANALOGICAL_CONTEXT = "analogical" # Analogical reasoning context

@dataclass
class RAGQuery:
    """Unified RAG query structure"""
    text: str
    mode: QueryMode = QueryMode.BALANCED
    context_tags: list[ContextTag] = field(default_factory=list)
    max_results: int = 10
    confidence_threshold: float = 0.7
    creative_depth: int = 3
    missing_node_analysis: bool = False
    temporal_window: tuple[datetime, datetime] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class RAGResult:
    """Unified RAG result structure"""
    content: str
    source: str
    confidence: float
    context_tags: list[ContextTag]
    reasoning_chain: list[str] = field(default_factory=list)
    missing_nodes: list[str] = field(default_factory=list)
    creative_connections: list[dict[str, Any]] = field(default_factory=list)
    bayesian_probability: float | None = None
    hippo_memory_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

class AdvancedIngestionEngine:
    """Multi-format document ingestion with semantic chunking"""
    
    def __init__(self, markitdown_mcp=None):
        self.markitdown = markitdown_mcp
        self.supported_formats = ['.pdf', '.docx', '.html', '.md', '.txt']
        
    async def ingest_document(self, file_path: Path, metadata: dict[str, Any] = None) -> list[dict[str, Any]]:
        """Ingest document with semantic chunking and dual context tagging"""
        if file_path.suffix not in self.supported_formats:
            raise ValueError(f"Unsupported format: {file_path.suffix}")
            
        # Extract text content
        if self.markitdown and file_path.suffix in ['.pdf', '.docx', '.html']:
            content = await self.markitdown.convert(str(file_path))
        else:
            content = file_path.read_text(encoding='utf-8')
        
        # Semantic chunking with dual context
        chunks = self._semantic_chunk(content, file_path)
        
        # Add metadata and context tags
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            processed_chunk = {
                'content': chunk,
                'source': str(file_path),
                'chunk_id': f"{file_path.stem}_{i}",
                'context_tags': self._generate_context_tags(chunk),
                'metadata': metadata or {},
                'ingestion_timestamp': datetime.utcnow().isoformat()
            }
            processed_chunks.append(processed_chunk)
            
        return processed_chunks
    
    def _semantic_chunk(self, content: str, file_path: Path) -> list[str]:
        """Semantic chunking based on content structure"""
        # Basic semantic chunking (can be enhanced with NLP)
        paragraphs = content.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) > 1000:  # Chunk size limit
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
                
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def _generate_context_tags(self, content: str) -> list[ContextTag]:
        """Generate context tags based on content analysis"""
        tags = [ContextTag.CONTENT_CONTEXT]  # Always include content context
        
        # Simple heuristics for context detection
        if any(word in content.lower() for word in ['when', 'time', 'date', 'year']):
            tags.append(ContextTag.TEMPORAL_CONTEXT)
        if any(word in content.lower() for word in ['where', 'location', 'place']):
            tags.append(ContextTag.SPATIAL_CONTEXT)
        if any(word in content.lower() for word in ['because', 'cause', 'reason', 'due to']):
            tags.append(ContextTag.CAUSAL_CONTEXT)
        if any(word in content.lower() for word in ['like', 'similar', 'analogous', 'compared']):
            tags.append(ContextTag.ANALOGICAL_CONTEXT)
            
        tags.append(ContextTag.SEMANTIC_CONTEXT)  # Always include semantic context
        return tags

class HippoRAGMemory:
    """Hippocampus-inspired episodic memory system"""
    
    def __init__(self, memory_mcp=None, huggingface_mcp=None):
        self.memory_mcp = memory_mcp
        self.huggingface = huggingface_mcp
        self.episodic_memory = {}
        self.entity_graph = nx.Graph()
        self.memory_consolidation_threshold = 5  # Consolidate after 5 retrievals
        
    async def store_episode(self, content: str, context: dict[str, Any]) -> str:
        """Store episodic memory with entity linking"""
        episode_id = hashlib.md5(f"{content}_{datetime.utcnow()}".encode()).hexdigest()[:12]
        
        # Extract named entities (simplified)
        entities = self._extract_entities(content)
        
        episode = {
            'id': episode_id,
            'content': content,
            'entities': entities,
            'context': context,
            'timestamp': datetime.utcnow().isoformat(),
            'retrieval_count': 0,
            'consolidation_score': 1.0
        }
        
        self.episodic_memory[episode_id] = episode
        
        # Update entity graph
        for entity in entities:
            if not self.entity_graph.has_node(entity):
                self.entity_graph.add_node(entity)
            
            # Add edges between co-occurring entities
            for other_entity in entities:
                if entity != other_entity:
                    if self.entity_graph.has_edge(entity, other_entity):
                        self.entity_graph[entity][other_entity]['weight'] += 1
                    else:
                        self.entity_graph.add_edge(entity, other_entity, weight=1)
        
        # Store in Memory MCP if available
        if self.memory_mcp:
            await self.memory_mcp.store(f"episode_{episode_id}", episode)
            
        return episode_id
    
    async def retrieve_episodes(self, query: str, max_episodes: int = 5) -> list[dict[str, Any]]:
        """Retrieve relevant episodes with consolidation"""
        query_entities = self._extract_entities(query)
        
        # Calculate relevance scores
        episode_scores = []
        for episode_id, episode in self.episodic_memory.items():
            score = self._calculate_episode_relevance(query, query_entities, episode)
            episode_scores.append((episode_id, score))
            
            # Update retrieval count
            episode['retrieval_count'] += 1
            
            # Memory consolidation
            if episode['retrieval_count'] >= self.memory_consolidation_threshold:
                episode['consolidation_score'] *= 1.2
        
        # Sort by relevance and return top episodes
        episode_scores.sort(key=lambda x: x[1], reverse=True)
        top_episodes = []
        
        for episode_id, score in episode_scores[:max_episodes]:
            episode = self.episodic_memory[episode_id].copy()
            episode['relevance_score'] = score
            top_episodes.append(episode)
            
        return top_episodes
    
    def _extract_entities(self, text: str) -> list[str]:
        """Simple entity extraction (can be enhanced with NER)"""
        # Basic entity extraction using capitalization patterns
        words = text.split()
        entities = []
        
        for word in words:
            if word[0].isupper() and len(word) > 2:
                entities.append(word.strip('.,!?'))
                
        return list(set(entities))
    
    def _calculate_episode_relevance(self, query: str, query_entities: list[str], episode: dict[str, Any]) -> float:
        """Calculate episode relevance score"""
        # Entity overlap score
        episode_entities = set(episode['entities'])
        query_entities_set = set(query_entities)
        entity_overlap = len(episode_entities.intersection(query_entities_set)) / max(len(query_entities_set), 1)
        
        # Content similarity (simplified)
        content_similarity = len(set(query.lower().split()).intersection(
            set(episode['content'].lower().split())
        )) / max(len(query.split()), 1)
        
        # Consolidation boost
        consolidation_boost = episode['consolidation_score']
        
        return (entity_overlap * 0.5 + content_similarity * 0.3) * consolidation_boost

class DualContextVectorRAG:
    """Vector RAG with dual context tagging"""
    
    def __init__(self, huggingface_mcp=None):
        self.huggingface = huggingface_mcp
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.content_embeddings = {}
        self.semantic_embeddings = {}
        self.documents = {}
        
    async def add_document(self, doc_id: str, content: str, context_tags: list[ContextTag], metadata: dict[str, Any] = None):
        """Add document with dual context embeddings"""
        # Content context embedding
        content_embedding = self.embedder.encode([content])[0]
        
        # Semantic context embedding (enhanced with context tags)
        semantic_content = self._enhance_with_context_tags(content, context_tags)
        semantic_embedding = self.embedder.encode([semantic_content])[0]
        
        self.content_embeddings[doc_id] = content_embedding
        self.semantic_embeddings[doc_id] = semantic_embedding
        self.documents[doc_id] = {
            'content': content,
            'context_tags': context_tags,
            'metadata': metadata or {}
        }
        
    async def search(self, query: str, context_tags: list[ContextTag] = None, max_results: int = 10) -> list[tuple[str, float]]:
        """Search with dual context consideration"""
        if not self.content_embeddings:
            return []
            
        query_embedding = self.embedder.encode([query])[0]
        
        # Enhanced query with context tags
        if context_tags:
            enhanced_query = self._enhance_with_context_tags(query, context_tags)
            semantic_query_embedding = self.embedder.encode([enhanced_query])[0]
        else:
            semantic_query_embedding = query_embedding
            
        results = []
        
        for doc_id in self.documents:
            # Content similarity
            content_sim = cosine_similarity([query_embedding], [self.content_embeddings[doc_id]])[0][0]
            
            # Semantic similarity
            semantic_sim = cosine_similarity([semantic_query_embedding], [self.semantic_embeddings[doc_id]])[0][0]
            
            # Context tag matching bonus
            doc_tags = set(self.documents[doc_id]['context_tags'])
            query_tags = set(context_tags or [])
            tag_overlap = len(doc_tags.intersection(query_tags)) / max(len(query_tags), 1) if query_tags else 0
            
            # Combined score
            combined_score = (content_sim * 0.4 + semantic_sim * 0.4 + tag_overlap * 0.2)
            results.append((doc_id, combined_score))
            
        # Sort and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_results]
    
    def _enhance_with_context_tags(self, content: str, context_tags: list[ContextTag]) -> str:
        """Enhance content with context tag information"""
        tag_descriptions = {
            ContextTag.CONTENT_CONTEXT: "direct content",
            ContextTag.SEMANTIC_CONTEXT: "semantic meaning relationship",
            ContextTag.TEMPORAL_CONTEXT: "time-related temporal information",
            ContextTag.SPATIAL_CONTEXT: "location spatial information",
            ContextTag.CAUSAL_CONTEXT: "cause and effect relationship",
            ContextTag.ANALOGICAL_CONTEXT: "analogical similarity comparison"
        }
        
        enhanced_content = content
        for tag in context_tags:
            if tag in tag_descriptions:
                enhanced_content += f" [{tag_descriptions[tag]}]"
                
        return enhanced_content

class BayesianKnowledgeGraph:
    """Bayesian knowledge graph with probabilistic reasoning"""
    
    def __init__(self, deepwiki_mcp=None, sequential_thinking_mcp=None):
        self.deepwiki = deepwiki_mcp
        self.sequential = sequential_thinking_mcp
        self.knowledge_graph = nx.DiGraph()
        self.trust_scores = {}
        
    async def add_knowledge(self, subject: str, predicate: str, object_: str, confidence: float, source: str):
        """Add knowledge with Bayesian probability"""
        
        # Add to graph
        if not self.knowledge_graph.has_node(subject):
            self.knowledge_graph.add_node(subject, type='entity')
        if not self.knowledge_graph.has_node(object_):
            self.knowledge_graph.add_node(object_, type='entity')
            
        self.knowledge_graph.add_edge(subject, object_, predicate=predicate, confidence=confidence, source=source)
        
        # Update trust scores with Bayesian inference
        if source in self.trust_scores:
            # Update existing source trust
            old_trust = self.trust_scores[source]
            new_trust = self._bayesian_update(old_trust, confidence)
            self.trust_scores[source] = new_trust
        else:
            # Initial trust score
            self.trust_scores[source] = confidence
            
    async def query_knowledge(self, subject: str, predicate: str = None, reasoning_depth: int = 3) -> list[dict[str, Any]]:
        """Query knowledge with multi-hop reasoning"""
        results = []
        
        if not self.knowledge_graph.has_node(subject):
            return results
            
        # Direct connections
        for neighbor in self.knowledge_graph.successors(subject):
            edge_data = self.knowledge_graph[subject][neighbor]
            if predicate is None or edge_data.get('predicate') == predicate:
                result = {
                    'subject': subject,
                    'predicate': edge_data['predicate'],
                    'object': neighbor,
                    'confidence': edge_data['confidence'],
                    'source_trust': self.trust_scores.get(edge_data['source'], 0.5),
                    'reasoning_path': [f"{subject} -> {neighbor}"],
                    'reasoning_depth': 1
                }
                results.append(result)
        
        # Multi-hop reasoning
        if reasoning_depth > 1:
            multi_hop_results = await self._multi_hop_reasoning(subject, predicate, reasoning_depth)
            results.extend(multi_hop_results)
            
        return results
    
    async def _multi_hop_reasoning(self, start_node: str, target_predicate: str, max_depth: int) -> list[dict[str, Any]]:
        """Multi-hop reasoning with path confidence calculation"""
        results = []
        visited_paths = set()
        
        def dfs_reasoning(current_node, path, depth, accumulated_confidence):
            if depth >= max_depth:
                return
                
            for neighbor in self.knowledge_graph.successors(current_node):
                edge_data = self.knowledge_graph[current_node][neighbor]
                new_path = path + [f"{current_node} -> {neighbor}"]
                path_key = "->".join(new_path)
                
                if path_key in visited_paths:
                    continue
                    
                visited_paths.add(path_key)
                
                # Calculate path confidence
                edge_confidence = edge_data['confidence']
                source_trust = self.trust_scores.get(edge_data['source'], 0.5)
                path_confidence = accumulated_confidence * edge_confidence * source_trust
                
                if target_predicate is None or edge_data.get('predicate') == target_predicate:
                    result = {
                        'subject': start_node,
                        'predicate': edge_data['predicate'],
                        'object': neighbor,
                        'confidence': path_confidence,
                        'source_trust': source_trust,
                        'reasoning_path': new_path,
                        'reasoning_depth': depth + 1
                    }
                    results.append(result)
                
                # Continue reasoning
                dfs_reasoning(neighbor, new_path, depth + 1, path_confidence)
        
        dfs_reasoning(start_node, [], 0, 1.0)
        return results
    
    def _bayesian_update(self, prior: float, evidence: float) -> float:
        """Bayesian update of trust scores"""
        # Simplified Bayesian update
        likelihood = evidence
        posterior = (likelihood * prior) / ((likelihood * prior) + ((1 - likelihood) * (1 - prior)))
        return posterior

class CreativeGraphSearch:
    """Creative graph search for brainstorming and novel connections"""
    
    def __init__(self, sequential_thinking_mcp=None):
        self.sequential = sequential_thinking_mcp
        self.concept_graph = nx.Graph()
        self.creativity_weight = 0.3  # Balance between relevance and creativity
        
    async def add_concept(self, concept: str, related_concepts: list[str], weights: list[float] = None):
        """Add concept with creative connections"""
        if not self.concept_graph.has_node(concept):
            self.concept_graph.add_node(concept)
            
        for i, related_concept in enumerate(related_concepts):
            if not self.concept_graph.has_node(related_concept):
                self.concept_graph.add_node(related_concept)
                
            weight = weights[i] if weights and i < len(weights) else 1.0
            self.concept_graph.add_edge(concept, related_concept, weight=weight)
    
    async def creative_search(self, seed_concepts: list[str], depth: int = 3, creativity_bias: float = 0.5) -> list[dict[str, Any]]:
        """Creative search with novelty bias"""
        if not seed_concepts:
            return []
            
        creative_results = []
        explored_paths = set()
        
        for seed in seed_concepts:
            if not self.concept_graph.has_node(seed):
                continue
                
            # Find creative paths
            paths = self._find_creative_paths(seed, depth, creativity_bias)
            
            for path in paths:
                if tuple(path) in explored_paths:
                    continue
                    
                explored_paths.add(tuple(path))
                
                # Calculate creativity score
                creativity_score = self._calculate_creativity_score(path)
                relevance_score = self._calculate_relevance_score(path, seed)
                
                combined_score = (creativity_score * creativity_bias + 
                                relevance_score * (1 - creativity_bias))
                
                creative_result = {
                    'path': path,
                    'seed_concept': seed,
                    'creativity_score': creativity_score,
                    'relevance_score': relevance_score,
                    'combined_score': combined_score,
                    'novel_connections': self._identify_novel_connections(path)
                }
                creative_results.append(creative_result)
        
        # Sort by combined score
        creative_results.sort(key=lambda x: x['combined_score'], reverse=True)
        return creative_results
    
    def _find_creative_paths(self, start_node: str, max_depth: int, creativity_bias: float) -> list[list[str]]:
        """Find creative paths with novelty preference"""
        paths = []
        
        def dfs_creative(current_node, path, depth):
            if depth >= max_depth:
                if len(path) > 1:
                    paths.append(path.copy())
                return
                
            neighbors = list(self.concept_graph.neighbors(current_node))
            
            # Sort neighbors by creativity potential (inverse of edge weight for novelty)
            neighbors.sort(key=lambda n: self.concept_graph[current_node][n].get('weight', 1.0))
            
            for neighbor in neighbors:
                if neighbor not in path:  # Avoid cycles
                    path.append(neighbor)
                    dfs_creative(neighbor, path, depth + 1)
                    path.pop()
        
        dfs_creative(start_node, [start_node], 0)
        return paths
    
    def _calculate_creativity_score(self, path: list[str]) -> float:
        """Calculate creativity score based on path novelty"""
        if len(path) < 2:
            return 0.0
            
        total_novelty = 0.0
        path_length = len(path)
        
        for i in range(len(path) - 1):
            current_node = path[i]
            next_node = path[i + 1]
            
            # Novelty is inverse of edge weight
            edge_weight = self.concept_graph[current_node][next_node].get('weight', 1.0)
            novelty = 1.0 / max(edge_weight, 0.1)
            total_novelty += novelty
            
        return total_novelty / (path_length - 1)
    
    def _calculate_relevance_score(self, path: list[str], seed_concept: str) -> float:
        """Calculate relevance to original concept"""
        if not path or path[0] != seed_concept:
            return 0.0
            
        # Relevance decreases with path length but considers connection strength
        relevance = 1.0
        for i in range(len(path) - 1):
            current_node = path[i]
            next_node = path[i + 1]
            edge_weight = self.concept_graph[current_node][next_node].get('weight', 1.0)
            relevance *= edge_weight * 0.8  # Decay factor
            
        return relevance
    
    def _identify_novel_connections(self, path: list[str]) -> list[dict[str, Any]]:
        """Identify potentially novel connections in the path"""
        novel_connections = []
        
        for i in range(len(path) - 2):  # Skip direct connections
            start_concept = path[i]
            end_concept = path[i + 2]
            
            # Check if there's a direct connection
            if not self.concept_graph.has_edge(start_concept, end_concept):
                novel_connections.append({
                    'from_concept': start_concept,
                    'to_concept': end_concept,
                    'via_concept': path[i + 1],
                    'novelty_score': 1.0 / (i + 1)  # Higher score for shorter bridges
                })
                
        return novel_connections

class MissingNodeDetector:
    """Missing node detection for knowledge gap analysis"""
    
    def __init__(self, memory_mcp=None, deepwiki_mcp=None):
        self.memory = memory_mcp
        self.deepwiki = deepwiki_mcp
        self.knowledge_patterns = {}
        
    async def analyze_missing_nodes(self, knowledge_graph: nx.Graph, query_context: str = None) -> list[dict[str, Any]]:
        """Analyze knowledge graph for missing nodes"""
        missing_nodes = []
        
        # Pattern-based missing node detection
        pattern_missing = await self._pattern_based_detection(knowledge_graph)
        missing_nodes.extend(pattern_missing)
        
        # Structural analysis
        structural_missing = await self._structural_analysis(knowledge_graph)
        missing_nodes.extend(structural_missing)
        
        # Query-driven analysis
        if query_context:
            query_missing = await self._query_driven_analysis(knowledge_graph, query_context)
            missing_nodes.extend(query_missing)
            
        # Remove duplicates and rank
        unique_missing = self._deduplicate_and_rank(missing_nodes)
        
        return unique_missing
    
    async def _pattern_based_detection(self, graph: nx.Graph) -> list[dict[str, Any]]:
        """Detect missing nodes based on learned patterns"""
        missing_patterns = []
        
        # Analyze common patterns
        for node in graph.nodes():
            neighbors = list(graph.neighbors(node))
            
            if len(neighbors) >= 2:
                # Look for missing intermediate nodes
                for i, neighbor1 in enumerate(neighbors):
                    for j, neighbor2 in enumerate(neighbors[i+1:], i+1):
                        # Check if neighbor1 and neighbor2 should have an intermediate connection
                        if not graph.has_edge(neighbor1, neighbor2):
                            potential_missing = await self._predict_intermediate_node(node, neighbor1, neighbor2)
                            if potential_missing:
                                missing_patterns.append({
                                    'type': 'intermediate_node',
                                    'suggested_node': potential_missing,
                                    'connecting_nodes': [neighbor1, neighbor2],
                                    'context_node': node,
                                    'confidence': 0.6
                                })
        
        return missing_patterns
    
    async def _structural_analysis(self, graph: nx.Graph) -> list[dict[str, Any]]:
        """Structural analysis for missing nodes"""
        structural_missing = []
        
        # Analyze degree distribution
        degrees = dict(graph.degree())
        avg_degree = sum(degrees.values()) / len(degrees) if degrees else 0
        
        # Find nodes with unusually low connectivity
        for node, degree in degrees.items():
            if degree < avg_degree * 0.5 and degree > 0:  # Significantly under-connected
                # Suggest potential connections based on neighbors of neighbors
                second_degree_nodes = set()
                for neighbor in graph.neighbors(node):
                    for second_neighbor in graph.neighbors(neighbor):
                        if second_neighbor != node and not graph.has_edge(node, second_neighbor):
                            second_degree_nodes.add(second_neighbor)
                
                if second_degree_nodes:
                    structural_missing.append({
                        'type': 'under_connected',
                        'node': node,
                        'current_degree': degree,
                        'average_degree': avg_degree,
                        'suggested_connections': list(second_degree_nodes)[:5],  # Top 5
                        'confidence': 0.4
                    })
        
        return structural_missing
    
    async def _query_driven_analysis(self, graph: nx.Graph, query_context: str) -> list[dict[str, Any]]:
        """Query-driven missing node analysis"""
        query_missing = []
        
        # Extract key concepts from query
        query_concepts = query_context.lower().split()
        
        # Find concepts that exist in graph
        existing_concepts = []
        for concept in query_concepts:
            for node in graph.nodes():
                if concept in node.lower() or node.lower() in concept:
                    existing_concepts.append(node)
        
        # Identify gaps between existing concepts
        if len(existing_concepts) >= 2:
            for i, concept1 in enumerate(existing_concepts):
                for concept2 in existing_concepts[i+1:]:
                    if not graph.has_edge(concept1, concept2):
                        # Try to find shortest path
                        try:
                            path = nx.shortest_path(graph, concept1, concept2)
                            if len(path) > 3:  # Path is too long, might need intermediate
                                middle_idx = len(path) // 2
                                missing_concept = await self._generate_bridging_concept(
                                    path[middle_idx-1], path[middle_idx+1], query_context
                                )
                                if missing_concept:
                                    query_missing.append({
                                        'type': 'query_bridging',
                                        'suggested_node': missing_concept,
                                        'bridges': [concept1, concept2],
                                        'current_path_length': len(path),
                                        'query_context': query_context,
                                        'confidence': 0.7
                                    })
                        except nx.NetworkXNoPath:
                            # No path exists, suggest direct connection concept
                            bridging_concept = await self._generate_bridging_concept(
                                concept1, concept2, query_context
                            )
                            if bridging_concept:
                                query_missing.append({
                                    'type': 'query_connection',
                                    'suggested_node': bridging_concept,
                                    'connects': [concept1, concept2],
                                    'query_context': query_context,
                                    'confidence': 0.8
                                })
        
        return query_missing
    
    async def _predict_intermediate_node(self, context: str, node1: str, node2: str) -> str | None:
        """Predict intermediate node between two concepts"""
        # Simple prediction based on common patterns
        
        # Generate potential intermediate concepts
        concepts = [node1.split()[-1], node2.split()[-1], context.split()[-1]]
        potential_intermediate = f"{concepts[0]}-{concepts[1]}"
        
        return potential_intermediate if len(potential_intermediate) < 50 else None
    
    async def _generate_bridging_concept(self, concept1: str, concept2: str, context: str) -> str | None:
        """Generate a concept that bridges two existing concepts"""
        # Simple bridging concept generation
        words1 = set(concept1.lower().split())
        words2 = set(concept2.lower().split())
        context_words = set(context.lower().split())
        
        # Find common ground
        common_words = words1.intersection(words2)
        if common_words:
            return f"{concept1}-{list(common_words)[0]}-{concept2}"
        
        # Use context to bridge
        relevant_context = context_words.intersection(words1.union(words2))
        if relevant_context:
            return f"{concept1}-{list(relevant_context)[0]}-{concept2}"
        
        return f"{concept1}-bridge-{concept2}"
    
    def _deduplicate_and_rank(self, missing_nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove duplicates and rank by confidence"""
        unique_nodes = {}
        
        for missing_node in missing_nodes:
            key = str(missing_node.get('suggested_node', ''))
            if key not in unique_nodes or missing_node.get('confidence', 0) > unique_nodes[key].get('confidence', 0):
                unique_nodes[key] = missing_node
        
        # Sort by confidence
        ranked_nodes = sorted(unique_nodes.values(), key=lambda x: x.get('confidence', 0), reverse=True)
        return ranked_nodes

class UnifiedRAGSystem:
    """Main unified RAG system orchestrating all components"""
    
    def __init__(self, config: dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize MCP connections (would be actual MCP clients in production)
        self.huggingface_mcp = None  # HuggingFaceMCP()
        self.sequential_mcp = None   # SequentialThinkingMCP()
        self.memory_mcp = None       # MemoryMCP()
        self.markitdown_mcp = None   # MarkitdownMCP()
        self.deepwiki_mcp = None     # DeepWikiMCP()
        
        # Initialize components
        self.ingestion = AdvancedIngestionEngine(self.markitdown_mcp)
        self.hippo_memory = HippoRAGMemory(self.memory_mcp, self.huggingface_mcp)
        self.vector_rag = DualContextVectorRAG(self.huggingface_mcp)
        self.bayesian_kg = BayesianKnowledgeGraph(self.deepwiki_mcp, self.sequential_mcp)
        self.creative_search = CreativeGraphSearch(self.sequential_mcp)
        self.missing_detector = MissingNodeDetector(self.memory_mcp, self.deepwiki_mcp)
        
        logger.info("Unified RAG System initialized with all components")
    
    async def query(self, rag_query: RAGQuery) -> RAGResult:
        """Main query interface routing to appropriate processing modes"""
        logger.info(f"Processing query in {rag_query.mode.value} mode: {rag_query.text[:100]}...")
        
        if rag_query.mode == QueryMode.FAST:
            return await self._fast_query(rag_query)
        elif rag_query.mode == QueryMode.HIPPO:
            return await self._hippo_query(rag_query)
        elif rag_query.mode == QueryMode.CREATIVE:
            return await self._creative_query(rag_query)
        elif rag_query.mode == QueryMode.ANALYTICAL:
            return await self._analytical_query(rag_query)
        elif rag_query.mode == QueryMode.MISSING_NODES:
            return await self._missing_nodes_query(rag_query)
        elif rag_query.mode == QueryMode.COMPREHENSIVE:
            return await self._comprehensive_query(rag_query)
        else:
            return await self._balanced_query(rag_query)
    
    async def _fast_query(self, query: RAGQuery) -> RAGResult:
        """Fast query using vector search only"""
        results = await self.vector_rag.search(query.text, query.context_tags, query.max_results)
        
        if results:
            doc_id, confidence = results[0]
            doc = self.vector_rag.documents.get(doc_id, {})
            
            return RAGResult(
                content=doc.get('content', ''),
                source=doc_id,
                confidence=confidence,
                context_tags=doc.get('context_tags', []),
                metadata={'mode': 'fast', 'processing_time': 'minimal'}
            )
        
        return RAGResult(
            content="No results found",
            source="system",
            confidence=0.0,
            context_tags=[]
        )
    
    async def _hippo_query(self, query: RAGQuery) -> RAGResult:
        """HippoRAG episodic memory query"""
        episodes = await self.hippo_memory.retrieve_episodes(query.text, query.max_results)
        
        if episodes:
            best_episode = episodes[0]
            
            return RAGResult(
                content=best_episode['content'],
                source=f"episode_{best_episode['id']}",
                confidence=best_episode['relevance_score'],
                context_tags=[ContextTag.SEMANTIC_CONTEXT, ContextTag.TEMPORAL_CONTEXT],
                hippo_memory_score=best_episode['consolidation_score'],
                metadata={
                    'mode': 'hippo',
                    'episode_id': best_episode['id'],
                    'entities': best_episode['entities'],
                    'retrieval_count': best_episode['retrieval_count']
                }
            )
        
        return RAGResult(
            content="No episodic memories found",
            source="hippo_memory",
            confidence=0.0,
            context_tags=[]
        )
    
    async def _creative_query(self, query: RAGQuery) -> RAGResult:
        """Creative brainstorming query"""
        # Extract concepts from query
        query_concepts = query.text.split()[:3]  # Use first 3 words as seed concepts
        
        creative_results = await self.creative_search.creative_search(
            query_concepts, 
            depth=query.creative_depth,
            creativity_bias=0.7
        )
        
        if creative_results:
            best_result = creative_results[0]
            
            # Generate creative content from path
            creative_content = self._generate_creative_content(best_result)
            
            return RAGResult(
                content=creative_content,
                source="creative_search",
                confidence=best_result['combined_score'],
                context_tags=[ContextTag.ANALOGICAL_CONTEXT, ContextTag.SEMANTIC_CONTEXT],
                creative_connections=creative_results[:5],  # Include top 5 creative connections
                metadata={
                    'mode': 'creative',
                    'path': best_result['path'],
                    'creativity_score': best_result['creativity_score'],
                    'novel_connections': best_result['novel_connections']
                }
            )
        
        return RAGResult(
            content="No creative connections found",
            source="creative_search",
            confidence=0.0,
            context_tags=[]
        )
    
    async def _analytical_query(self, query: RAGQuery) -> RAGResult:
        """Deep analytical query with Bayesian reasoning"""
        # Extract subject and predicate from query (simplified)
        words = query.text.split()
        subject = words[0] if words else ""
        
        knowledge_results = await self.bayesian_kg.query_knowledge(subject, reasoning_depth=3)
        
        if knowledge_results:
            # Combine results with reasoning chains
            reasoning_chain = []
            combined_confidence = 1.0
            
            for result in knowledge_results[:3]:  # Top 3 results
                reasoning_chain.extend(result['reasoning_path'])
                combined_confidence *= result['confidence']
            
            analytical_content = self._generate_analytical_content(knowledge_results)
            
            return RAGResult(
                content=analytical_content,
                source="bayesian_kg",
                confidence=combined_confidence,
                context_tags=[ContextTag.CAUSAL_CONTEXT, ContextTag.SEMANTIC_CONTEXT],
                reasoning_chain=reasoning_chain,
                bayesian_probability=combined_confidence,
                metadata={
                    'mode': 'analytical',
                    'knowledge_results': len(knowledge_results),
                    'reasoning_depth': max(r['reasoning_depth'] for r in knowledge_results)
                }
            )
        
        return RAGResult(
            content="No analytical results found",
            source="bayesian_kg",
            confidence=0.0,
            context_tags=[]
        )
    
    async def _missing_nodes_query(self, query: RAGQuery) -> RAGResult:
        """Missing node analysis query"""
        # Use the knowledge graph from Bayesian KG
        missing_nodes = await self.missing_detector.analyze_missing_nodes(
            self.bayesian_kg.knowledge_graph,
            query.text
        )
        
        if missing_nodes:
            missing_content = self._generate_missing_nodes_content(missing_nodes)
            
            return RAGResult(
                content=missing_content,
                source="missing_node_detector",
                confidence=0.8,
                context_tags=[ContextTag.SEMANTIC_CONTEXT],
                missing_nodes=[node['suggested_node'] for node in missing_nodes[:5]],
                metadata={
                    'mode': 'missing_nodes',
                    'detected_gaps': len(missing_nodes),
                    'gap_types': list(set(node['type'] for node in missing_nodes))
                }
            )
        
        return RAGResult(
            content="No knowledge gaps detected",
            source="missing_node_detector",
            confidence=0.5,
            context_tags=[]
        )
    
    async def _comprehensive_query(self, query: RAGQuery) -> RAGResult:
        """Comprehensive query using all systems"""
        # Run all query types in parallel
        tasks = [
            self._fast_query(query),
            self._hippo_query(query),
            self._creative_query(query),
            self._analytical_query(query),
            self._missing_nodes_query(query)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        valid_results = [r for r in results if isinstance(r, RAGResult) and r.confidence > 0.1]
        
        if valid_results:
            # Combine results intelligently
            combined_content = self._combine_comprehensive_results(valid_results)
            
            # Calculate weighted confidence
            total_weight = sum(r.confidence for r in valid_results)
            weighted_confidence = total_weight / len(valid_results)
            
            # Combine all context tags
            all_context_tags = list(set(tag for result in valid_results for tag in result.context_tags))
            
            # Combine reasoning chains
            combined_reasoning = []
            for result in valid_results:
                combined_reasoning.extend(result.reasoning_chain)
            
            return RAGResult(
                content=combined_content,
                source="comprehensive_analysis",
                confidence=weighted_confidence,
                context_tags=all_context_tags,
                reasoning_chain=combined_reasoning,
                creative_connections=[cc for r in valid_results for cc in r.creative_connections],
                missing_nodes=[mn for r in valid_results for mn in r.missing_nodes],
                metadata={
                    'mode': 'comprehensive',
                    'component_results': len(valid_results),
                    'components_used': [r.source for r in valid_results]
                }
            )
        
        return RAGResult(
            content="No comprehensive results available",
            source="comprehensive_analysis",
            confidence=0.0,
            context_tags=[]
        )
    
    async def _balanced_query(self, query: RAGQuery) -> RAGResult:
        """Balanced query using vector + Bayesian systems"""
        # Combine vector search with knowledge graph
        vector_results = await self.vector_rag.search(query.text, query.context_tags, query.max_results)
        
        if vector_results:
            doc_id, vector_confidence = vector_results[0]
            doc = self.vector_rag.documents.get(doc_id, {})
            
            # Enhance with knowledge graph if available
            words = query.text.split()
            subject = words[0] if words else ""
            kg_results = await self.bayesian_kg.query_knowledge(subject, reasoning_depth=2)
            
            combined_content = doc.get('content', '')
            if kg_results:
                kg_content = self._generate_analytical_content(kg_results[:2])
                combined_content = f"{combined_content}\n\nKnowledge Context:\n{kg_content}"
            
            return RAGResult(
                content=combined_content,
                source=f"balanced_{doc_id}",
                confidence=(vector_confidence + (kg_results[0]['confidence'] if kg_results else 0.5)) / 2,
                context_tags=doc.get('context_tags', []),
                reasoning_chain=[f"Vector similarity: {vector_confidence:.3f}"] + 
                              ([kg_results[0]['reasoning_path'][0]] if kg_results else []),
                metadata={
                    'mode': 'balanced',
                    'vector_confidence': vector_confidence,
                    'kg_results': len(kg_results)
                }
            )
        
        return RAGResult(
            content="No balanced results found",
            source="balanced_search",
            confidence=0.0,
            context_tags=[]
        )
    
    def _generate_creative_content(self, creative_result: dict[str, Any]) -> str:
        """Generate content from creative search results"""
        path = creative_result['path']
        connections = creative_result.get('novel_connections', [])
        
        content = f"Creative exploration starting from '{path[0]}':\n\n"
        content += f"Connection path: {' -> '.join(path)}\n\n"
        
        if connections:
            content += "Novel connections discovered:\n"
            for conn in connections[:3]:
                content += f"- {conn['from_concept']} connects to {conn['to_concept']} via {conn['via_concept']}\n"
        
        content += f"\nCreativity score: {creative_result['creativity_score']:.3f}"
        return content
    
    def _generate_analytical_content(self, knowledge_results: list[dict[str, Any]]) -> str:
        """Generate content from Bayesian knowledge graph results"""
        if not knowledge_results:
            return "No analytical insights available"
        
        content = "Analytical insights:\n\n"
        
        for result in knowledge_results[:3]:
            content += f"• {result['subject']} {result['predicate']} {result['object']} "
            content += f"(confidence: {result['confidence']:.3f})\n"
            if result['reasoning_path']:
                content += f"  Reasoning: {result['reasoning_path'][-1]}\n"
        
        return content
    
    def _generate_missing_nodes_content(self, missing_nodes: list[dict[str, Any]]) -> str:
        """Generate content describing missing nodes"""
        if not missing_nodes:
            return "No knowledge gaps detected"
        
        content = "Identified knowledge gaps:\n\n"
        
        for gap in missing_nodes[:5]:
            content += f"• Missing: {gap.get('suggested_node', 'Unknown')}\n"
            content += f"  Type: {gap.get('type', 'unknown')}\n"
            content += f"  Confidence: {gap.get('confidence', 0):.3f}\n"
            
            if 'bridges' in gap:
                content += f"  Would connect: {gap['bridges']}\n"
            elif 'connecting_nodes' in gap:
                content += f"  Between: {gap['connecting_nodes']}\n"
            
            content += "\n"
        
        return content
    
    def _combine_comprehensive_results(self, results: list[RAGResult]) -> str:
        """Combine results from comprehensive query"""
        content = "Comprehensive Analysis Results:\n\n"
        
        for i, result in enumerate(results[:5], 1):
            content += f"{i}. {result.source.upper()} (confidence: {result.confidence:.3f}):\n"
            content += f"{result.content}\n\n"
        
        return content
    
    async def ingest_document(self, file_path: Path, metadata: dict[str, Any] = None) -> bool:
        """Ingest document into all appropriate systems"""
        try:
            # Process document
            chunks = await self.ingestion.ingest_document(file_path, metadata)
            
            for chunk in chunks:
                doc_id = chunk['chunk_id']
                content = chunk['content']
                context_tags = chunk['context_tags']
                
                # Add to vector RAG
                await self.vector_rag.add_document(doc_id, content, context_tags, chunk['metadata'])
                
                # Store as episodic memory
                await self.hippo_memory.store_episode(content, {
                    'source': str(file_path),
                    'context_tags': [tag.value for tag in context_tags],
                    'metadata': chunk['metadata']
                })
                
                # Add concepts to creative search (simplified)
                concepts = content.split()[:10]  # First 10 words as concepts
                if len(concepts) >= 2:
                    await self.creative_search.add_concept(concepts[0], concepts[1:])
            
            logger.info(f"Successfully ingested {len(chunks)} chunks from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to ingest document {file_path}: {e}")
            return False
    
    async def add_knowledge_triple(self, subject: str, predicate: str, object_: str, 
                                 confidence: float, source: str):
        """Add knowledge to Bayesian knowledge graph"""
        await self.bayesian_kg.add_knowledge(subject, predicate, object_, confidence, source)
        logger.info(f"Added knowledge: {subject} {predicate} {object_} (confidence: {confidence})")

# Factory function for easy instantiation
def create_unified_rag_system(config: dict[str, Any] = None) -> UnifiedRAGSystem:
    """Create a unified RAG system with default configuration"""
    default_config = {
        'embedder_model': 'all-MiniLM-L6-v2',
        'chunk_size': 1000,
        'max_results': 10,
        'confidence_threshold': 0.7,
        'creativity_bias': 0.5
    }
    
    if config:
        default_config.update(config)
    
    return UnifiedRAGSystem(default_config)

# Example usage
async def main():
    """Example usage of the Unified RAG System"""
    # Create system
    rag_system = create_unified_rag_system()
    
    # Example document ingestion
    # await rag_system.ingest_document(Path("example.pdf"))
    
    # Example queries
    queries = [
        RAGQuery(text="What are the key concepts in AI?", mode=QueryMode.FAST),
        RAGQuery(text="Creative applications of machine learning", mode=QueryMode.CREATIVE, creative_depth=4),
        RAGQuery(text="Analyze deep learning architectures", mode=QueryMode.ANALYTICAL),
        RAGQuery(text="Find missing connections in neural networks", mode=QueryMode.MISSING_NODES),
        RAGQuery(text="Comprehensive AI overview", mode=QueryMode.COMPREHENSIVE)
    ]
    
    for query in queries:
        result = await rag_system.query(query)
        print(f"\nQuery: {query.text}")
        print(f"Mode: {query.mode.value}")
        print(f"Result: {result.content[:200]}...")
        print(f"Confidence: {result.confidence:.3f}")

if __name__ == "__main__":
    asyncio.run(main())