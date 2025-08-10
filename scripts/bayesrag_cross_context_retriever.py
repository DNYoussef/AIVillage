#!/usr/bin/env python3
"""
Cross-context retrieval system for BayesRAG.
Handles queries spanning global and local contexts with trust-weighted ranking.
"""

import asyncio
import json
import logging
import sqlite3
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import heapq

from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Result from cross-context retrieval."""
    chunk_id: str
    content: str
    local_summary: str
    global_summary: str  # From parent article
    parent_title: str
    section_title: Optional[str]
    relevance_score: float
    trust_score: float
    combined_score: float
    context_type: str  # 'global', 'local', 'cross-reference'
    matching_contexts: List[str]  # Which contexts matched the query
    graph_path_trust: Optional[float]  # Trust score from graph path
    evidence_chain: List[str]  # Chain of evidence supporting this result

@dataclass
class QueryContext:
    """Context information extracted from query."""
    temporal_hints: List[str]
    geographic_hints: List[str]
    topic_hints: List[str]
    specificity_level: str  # 'global', 'local', 'specific'
    cross_reference_hints: List[str]

class CrossContextRetriever:
    """Advanced retrieval system with cross-context understanding."""
    
    def __init__(self, data_dir: Path = Path("data")):
        self.data_dir = data_dir
        
        # Models
        self.embedder = SentenceTransformer('paraphrase-MiniLM-L3-v2')
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-2-v2')
        
        # Database paths
        self.local_db_path = self.data_dir / "wikipedia_local_context.db"
        self.global_db_path = self.data_dir / "wikipedia_global_context.db"
        self.graph_db_path = self.data_dir / "wikipedia_graph.db"
        
        # Caches
        self.embedding_cache = {}
        self.graph_cache = None
        
    def analyze_query_context(self, query: str) -> QueryContext:
        """Analyze query to determine context requirements."""
        
        query_lower = query.lower()
        
        # Temporal hints
        temporal_hints = []
        temporal_patterns = [
            r'\b(1[0-9]{3}|20[0-2][0-9])\b',  # Years
            r'\b(medieval|renaissance|industrial|modern)\b',  # Periods
            r'\b(century|decade|era|period)\b',  # Time references
            r'\b(before|after|during|between)\b'  # Temporal relations
        ]
        
        import re
        for pattern in temporal_patterns:
            matches = re.findall(pattern, query_lower)
            temporal_hints.extend(matches)
            
        # Geographic hints
        geographic_hints = []
        geo_patterns = ['germany', 'france', 'england', 'europe', 'asia', 'africa', 
                       'america', 'city', 'country', 'region', 'continent']
        
        for geo in geo_patterns:
            if geo in query_lower:
                geographic_hints.append(geo)
                
        # Topic hints (domain-specific terms)
        topic_hints = []
        topic_domains = {
            'history': ['war', 'battle', 'empire', 'king', 'revolution', 'historical'],
            'science': ['theory', 'research', 'discovery', 'experiment', 'scientific'],
            'culture': ['art', 'literature', 'music', 'cultural', 'artistic'],
            'politics': ['government', 'political', 'parliament', 'democracy', 'policy'],
            'economics': ['trade', 'economy', 'economic', 'market', 'commerce']
        }
        
        for domain, terms in topic_domains.items():
            if any(term in query_lower for term in terms):
                topic_hints.append(domain)
                
        # Determine specificity level
        if any(word in query_lower for word in ['overview', 'general', 'about', 'what is']):
            specificity_level = 'global'
        elif any(word in query_lower for word in ['specific', 'detail', 'exactly', 'precisely']):
            specificity_level = 'specific'
        else:
            specificity_level = 'local'
            
        # Cross-reference hints
        cross_ref_hints = []
        if any(word in query_lower for word in ['related', 'connection', 'link', 'similar']):
            cross_ref_hints.append('related_concepts')
        if any(word in query_lower for word in ['caused', 'led to', 'resulted']):
            cross_ref_hints.append('causal_relationships')
            
        return QueryContext(
            temporal_hints=temporal_hints,
            geographic_hints=geographic_hints,
            topic_hints=topic_hints,
            specificity_level=specificity_level,
            cross_reference_hints=cross_ref_hints
        )
        
    async def retrieve_cross_context(
        self, 
        query: str, 
        k: int = 10,
        trust_threshold: float = 0.3,
        use_graph_paths: bool = True
    ) -> List[RetrievalResult]:
        """Perform cross-context retrieval with trust weighting."""
        
        logger.info(f"Cross-context retrieval for query: '{query}'")
        
        # Analyze query context
        query_context = self.analyze_query_context(query)
        logger.info(f"Query context: {query_context.specificity_level} level, "
                   f"topics: {query_context.topic_hints}")
        
        # Generate query embedding
        query_embedding = self.embedder.encode([query])[0]
        
        # Get candidates from different context levels
        candidates = []
        
        # 1. Local context retrieval
        local_candidates = await self._retrieve_local_context(
            query_embedding, query_context, k * 2
        )
        candidates.extend(local_candidates)
        
        # 2. Global context retrieval (if query is global or general)
        if query_context.specificity_level in ['global', 'local']:
            global_candidates = await self._retrieve_global_context(
                query_embedding, query_context, k // 2
            )
            candidates.extend(global_candidates)
            
        # 3. Cross-reference retrieval
        if query_context.cross_reference_hints:
            cross_ref_candidates = await self._retrieve_cross_references(
                query_embedding, query_context, k // 2
            )
            candidates.extend(cross_ref_candidates)
            
        # 4. Graph-based retrieval (if enabled)
        if use_graph_paths and query_context.cross_reference_hints:
            graph_candidates = await self._retrieve_via_graph_paths(
                query, candidates[:5], k // 2
            )
            candidates.extend(graph_candidates)
            
        # Remove duplicates
        unique_candidates = {}
        for candidate in candidates:
            if candidate.chunk_id not in unique_candidates:
                unique_candidates[candidate.chunk_id] = candidate
            else:
                # Keep the one with higher score
                if candidate.combined_score > unique_candidates[candidate.chunk_id].combined_score:
                    unique_candidates[candidate.chunk_id] = candidate
                    
        candidates = list(unique_candidates.values())
        
        # Re-rank with cross-encoder
        candidates = await self._rerank_with_cross_encoder(query, candidates)
        
        # Apply trust threshold
        candidates = [c for c in candidates if c.trust_score >= trust_threshold]
        
        # Final ranking by combined score
        candidates.sort(key=lambda x: x.combined_score, reverse=True)
        
        return candidates[:k]
        
    async def _retrieve_local_context(
        self, 
        query_embedding: np.ndarray, 
        query_context: QueryContext, 
        k: int
    ) -> List[RetrievalResult]:
        """Retrieve from local chunk contexts."""
        
        candidates = []
        
        with sqlite3.connect(self.local_db_path) as conn:
            cursor = conn.execute("""
                SELECT lc.chunk_id, lc.parent_title, lc.section_title, lc.content,
                       lc.local_summary, lc.local_tags, lc.temporal_context,
                       lc.geographic_context, lc.cross_references, lc.embedding,
                       gc.summary as global_summary, gc.trust_score
                FROM local_contexts lc
                JOIN global_contexts gc ON lc.parent_title = gc.title
            """)
            
            for row in cursor.fetchall():
                chunk_id, parent_title, section_title, content = row[0:4]
                local_summary, local_tags, temporal_context, geographic_context = row[4:8]
                cross_references, embedding_bytes, global_summary, trust_score = row[8:12]
                
                # Reconstruct embedding
                chunk_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                
                # Calculate semantic similarity
                semantic_sim = cosine_similarity([query_embedding], [chunk_embedding])[0][0]
                
                # Calculate context relevance
                context_relevance = self._calculate_context_relevance(
                    query_context, 
                    json.loads(local_tags) if local_tags else [],
                    temporal_context,
                    geographic_context
                )
                
                # Combined relevance score
                relevance_score = semantic_sim * 0.7 + context_relevance * 0.3
                
                # Combined score with trust
                combined_score = relevance_score * (0.7 + trust_score * 0.3)
                
                # Determine matching contexts
                matching_contexts = self._identify_matching_contexts(
                    query_context, local_tags, temporal_context, geographic_context
                )
                
                result = RetrievalResult(
                    chunk_id=chunk_id,
                    content=content,
                    local_summary=local_summary,
                    global_summary=global_summary,
                    parent_title=parent_title,
                    section_title=section_title,
                    relevance_score=relevance_score,
                    trust_score=trust_score,
                    combined_score=combined_score,
                    context_type='local',
                    matching_contexts=matching_contexts,
                    graph_path_trust=None,
                    evidence_chain=[chunk_id]
                )
                
                candidates.append(result)
                
        # Sort and return top k
        candidates.sort(key=lambda x: x.combined_score, reverse=True)
        return candidates[:k]
        
    async def _retrieve_global_context(
        self, 
        query_embedding: np.ndarray, 
        query_context: QueryContext, 
        k: int
    ) -> List[RetrievalResult]:
        """Retrieve based on global article context."""
        
        candidates = []
        
        with sqlite3.connect(self.global_db_path) as conn:
            cursor = conn.execute("""
                SELECT title, summary, global_tags, trust_score, categories
                FROM global_contexts
                ORDER BY trust_score DESC
            """)
            
            for row in cursor.fetchall():
                title, summary, global_tags_json, trust_score, categories_json = row
                
                global_tags = json.loads(global_tags_json) if global_tags_json else []
                categories = json.loads(categories_json) if categories_json else []
                
                # Generate embedding for summary
                summary_embedding = self.embedder.encode([summary])[0]
                
                # Calculate similarity
                semantic_sim = cosine_similarity([query_embedding], [summary_embedding])[0][0]
                
                # Calculate context relevance
                context_relevance = self._calculate_global_context_relevance(
                    query_context, global_tags, categories
                )
                
                relevance_score = semantic_sim * 0.6 + context_relevance * 0.4
                combined_score = relevance_score * (0.6 + trust_score * 0.4)
                
                # For global results, we need to find a representative chunk
                representative_chunk = await self._get_representative_chunk(title)
                
                if representative_chunk:
                    result = RetrievalResult(
                        chunk_id=representative_chunk['chunk_id'],
                        content=representative_chunk['content'][:500] + "...",  # Truncate
                        local_summary=representative_chunk['local_summary'],
                        global_summary=summary,
                        parent_title=title,
                        section_title="Overview",
                        relevance_score=relevance_score,
                        trust_score=trust_score,
                        combined_score=combined_score,
                        context_type='global',
                        matching_contexts=['global_summary'],
                        graph_path_trust=None,
                        evidence_chain=[representative_chunk['chunk_id']]
                    )
                    
                    candidates.append(result)
                    
        candidates.sort(key=lambda x: x.combined_score, reverse=True)
        return candidates[:k]
        
    async def _get_representative_chunk(self, title: str) -> Optional[Dict[str, Any]]:
        """Get the most representative chunk for an article."""
        
        with sqlite3.connect(self.local_db_path) as conn:
            cursor = conn.execute("""
                SELECT chunk_id, content, local_summary
                FROM local_contexts
                WHERE parent_title = ?
                ORDER BY LENGTH(content) DESC
                LIMIT 1
            """, (title,))
            
            result = cursor.fetchone()
            
            if result:
                return {
                    'chunk_id': result[0],
                    'content': result[1],
                    'local_summary': result[2]
                }
                
        return None
        
    async def _retrieve_cross_references(
        self, 
        query_embedding: np.ndarray, 
        query_context: QueryContext, 
        k: int
    ) -> List[RetrievalResult]:
        """Retrieve based on cross-reference relationships."""
        
        # This would involve finding chunks that reference each other
        # and exploring those connections for relevant content
        candidates = []
        
        # Implementation would involve:
        # 1. Find chunks with cross-references matching query topics
        # 2. Follow reference chains to find related content
        # 3. Score based on reference strength and semantic similarity
        
        # For now, return empty list - this is a complex feature
        return candidates
        
    async def _retrieve_via_graph_paths(
        self, 
        query: str, 
        seed_candidates: List[RetrievalResult], 
        k: int
    ) -> List[RetrievalResult]:
        """Retrieve additional candidates via graph paths."""
        
        if not seed_candidates:
            return []
            
        # Load graph if not cached
        if self.graph_cache is None:
            await self._load_graph_cache()
            
        candidates = []
        
        for seed in seed_candidates[:3]:  # Use top 3 seeds
            # Find neighbors in knowledge graph
            neighbors = await self._find_trusted_neighbors(seed.chunk_id, max_depth=2)
            
            for neighbor_id, path_trust in neighbors:
                # Get neighbor details
                neighbor_data = await self._get_chunk_data(neighbor_id)
                
                if neighbor_data:
                    # Calculate relevance using cross-encoder
                    relevance_score = self._calculate_graph_relevance(
                        query, neighbor_data['content']
                    )
                    
                    combined_score = relevance_score * path_trust * 0.8  # Discount for indirect
                    
                    result = RetrievalResult(
                        chunk_id=neighbor_id,
                        content=neighbor_data['content'],
                        local_summary=neighbor_data['local_summary'],
                        global_summary=neighbor_data.get('global_summary', ''),
                        parent_title=neighbor_data['parent_title'],
                        section_title=neighbor_data.get('section_title'),
                        relevance_score=relevance_score,
                        trust_score=neighbor_data.get('trust_score', 0.5),
                        combined_score=combined_score,
                        context_type='cross-reference',
                        matching_contexts=['graph_path'],
                        graph_path_trust=path_trust,
                        evidence_chain=[seed.chunk_id, neighbor_id]
                    )
                    
                    candidates.append(result)
                    
        candidates.sort(key=lambda x: x.combined_score, reverse=True)
        return candidates[:k]
        
    async def _load_graph_cache(self):
        """Load graph relationships into memory cache."""
        
        self.graph_cache = {}
        
        with sqlite3.connect(self.graph_db_path) as conn:
            cursor = conn.execute("""
                SELECT source_node, target_node, trust_weight, relationship_type
                FROM graph_edges
                WHERE trust_weight > 0.2
            """)
            
            for row in cursor.fetchall():
                source, target, trust, rel_type = row
                
                if source not in self.graph_cache:
                    self.graph_cache[source] = []
                    
                self.graph_cache[source].append({
                    'target': target,
                    'trust': trust,
                    'type': rel_type
                })
                
        logger.info(f"Loaded graph cache with {len(self.graph_cache)} nodes")
        
    async def _find_trusted_neighbors(
        self, 
        chunk_id: str, 
        max_depth: int = 2, 
        min_trust: float = 0.3
    ) -> List[Tuple[str, float]]:
        """Find trusted neighbors via graph traversal."""
        
        if not self.graph_cache or chunk_id not in self.graph_cache:
            return []
            
        visited = set()
        queue = [(chunk_id, 1.0, 0)]  # (node, trust, depth)
        neighbors = []
        
        while queue:
            current_node, current_trust, depth = queue.pop(0)
            
            if current_node in visited or depth >= max_depth:
                continue
                
            visited.add(current_node)
            
            if depth > 0 and current_trust >= min_trust:
                neighbors.append((current_node, current_trust))
                
            # Add neighbors to queue
            if current_node in self.graph_cache and depth < max_depth - 1:
                for neighbor in self.graph_cache[current_node]:
                    if neighbor['target'] not in visited:
                        new_trust = current_trust * neighbor['trust']
                        queue.append((neighbor['target'], new_trust, depth + 1))
                        
        # Sort by trust and return
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors[:10]
        
    async def _get_chunk_data(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get chunk data from database."""
        
        with sqlite3.connect(self.local_db_path) as conn:
            cursor = conn.execute("""
                SELECT lc.content, lc.local_summary, lc.parent_title, lc.section_title,
                       gc.summary as global_summary, gc.trust_score
                FROM local_contexts lc
                JOIN global_contexts gc ON lc.parent_title = gc.title
                WHERE lc.chunk_id = ?
            """, (chunk_id,))
            
            result = cursor.fetchone()
            
            if result:
                return {
                    'content': result[0],
                    'local_summary': result[1],
                    'parent_title': result[2],
                    'section_title': result[3],
                    'global_summary': result[4],
                    'trust_score': result[5]
                }
                
        return None
        
    def _calculate_context_relevance(
        self, 
        query_context: QueryContext, 
        chunk_tags: List[str],
        temporal_context: Optional[str],
        geographic_context: Optional[str]
    ) -> float:
        """Calculate how well chunk context matches query context."""
        
        relevance = 0.0
        factors = 0
        
        # Topic relevance
        if query_context.topic_hints and chunk_tags:
            topic_overlap = len(set(query_context.topic_hints) & set(chunk_tags))
            if topic_overlap > 0:
                relevance += topic_overlap / len(query_context.topic_hints)
                factors += 1
                
        # Temporal relevance
        if query_context.temporal_hints and temporal_context:
            if any(hint in temporal_context.lower() for hint in query_context.temporal_hints):
                relevance += 1.0
                factors += 1
                
        # Geographic relevance  
        if query_context.geographic_hints and geographic_context:
            if any(hint in geographic_context.lower() for hint in query_context.geographic_hints):
                relevance += 1.0
                factors += 1
                
        return relevance / max(factors, 1)
        
    def _calculate_global_context_relevance(
        self, 
        query_context: QueryContext, 
        global_tags: List[str], 
        categories: List[str]
    ) -> float:
        """Calculate relevance for global context."""
        
        relevance = 0.0
        
        # Topic matching with global tags
        if query_context.topic_hints:
            tag_overlap = len(set(query_context.topic_hints) & set(global_tags))
            if tag_overlap > 0:
                relevance += tag_overlap / len(query_context.topic_hints)
                
        # Category matching
        if categories:
            category_text = ' '.join(categories).lower()
            for hint in query_context.topic_hints + query_context.geographic_hints:
                if hint in category_text:
                    relevance += 0.2
                    
        return min(1.0, relevance)
        
    def _identify_matching_contexts(
        self, 
        query_context: QueryContext, 
        local_tags: str, 
        temporal_context: Optional[str],
        geographic_context: Optional[str]
    ) -> List[str]:
        """Identify which context aspects matched the query."""
        
        matching = []
        
        if local_tags:
            chunk_tags = json.loads(local_tags)
            if any(tag in query_context.topic_hints for tag in chunk_tags):
                matching.append('topic_tags')
                
        if temporal_context and any(hint in temporal_context.lower() 
                                   for hint in query_context.temporal_hints):
            matching.append('temporal_context')
            
        if geographic_context and any(hint in geographic_context.lower() 
                                     for hint in query_context.geographic_hints):
            matching.append('geographic_context')
            
        return matching
        
    def _calculate_graph_relevance(self, query: str, content: str) -> float:
        """Calculate relevance using cross-encoder for graph-retrieved content."""
        
        try:
            scores = self.cross_encoder.predict([(query, content[:512])])
            return float(scores[0]) if hasattr(scores, '__iter__') else float(scores)
        except:
            # Fallback to simple keyword matching
            query_words = set(query.lower().split())
            content_words = set(content.lower().split())
            overlap = len(query_words & content_words)
            return overlap / len(query_words) if query_words else 0.0
            
    async def _rerank_with_cross_encoder(
        self, 
        query: str, 
        candidates: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Re-rank candidates using cross-encoder."""
        
        if not candidates:
            return candidates
            
        try:
            # Prepare pairs for cross-encoder
            pairs = []
            for candidate in candidates:
                text = f"{candidate.local_summary} {candidate.content[:300]}"
                pairs.append((query, text))
                
            # Get cross-encoder scores
            scores = self.cross_encoder.predict(pairs)
            
            # Update combined scores
            for i, candidate in enumerate(candidates):
                cross_encoder_score = float(scores[i]) if hasattr(scores, '__iter__') else float(scores)
                
                # Weighted combination of original score and cross-encoder score
                candidate.combined_score = (
                    candidate.combined_score * 0.7 + 
                    cross_encoder_score * 0.3
                )
                
        except Exception as e:
            logger.warning(f"Cross-encoder reranking failed: {e}")
            
        return candidates

async def main():
    """Test the cross-context retriever."""
    
    retriever = CrossContextRetriever()
    
    # Test queries of different types
    test_queries = [
        "What caused World War I in Europe?",  # Global + temporal + geographic
        "German unification process in 19th century",  # Local + temporal + geographic  
        "Industrial revolution impact on society",  # Cross-reference + temporal
        "Renaissance art and culture overview"  # Global + topic
    ]
    
    for query in test_queries:
        print(f"\n=== Query: {query} ===")
        
        try:
            results = await retriever.retrieve_cross_context(
                query, k=5, use_graph_paths=True
            )
            
            for i, result in enumerate(results):
                print(f"\nResult {i+1}:")
                print(f"  Article: {result.parent_title}")
                print(f"  Section: {result.section_title}")
                print(f"  Context Type: {result.context_type}")
                print(f"  Relevance: {result.relevance_score:.3f}")
                print(f"  Trust: {result.trust_score:.3f}")
                print(f"  Combined: {result.combined_score:.3f}")
                print(f"  Matching: {result.matching_contexts}")
                if result.graph_path_trust:
                    print(f"  Graph Trust: {result.graph_path_trust:.3f}")
                print(f"  Summary: {result.local_summary}")
                
        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            
    logger.info("Cross-context retrieval testing completed!")

if __name__ == "__main__":
    asyncio.run(main())