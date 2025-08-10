"""
Enhanced BayesRAG-CODEX Integration with Trust-Weighted Retrieval and Hierarchical Context.

This module extends the CODEX RAG pipeline with BayesRAG's innovative features:
- Trust-weighted result ranking based on Bayesian scoring
- Hierarchical context metadata (global summaries + local details)
- Cross-reference discovery via knowledge graph
- Context-aware query routing (temporal, geographic, topical)
"""

import asyncio
import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from codex_rag_integration import (
    CODEXRAGPipeline,
    Document,
    RetrievalResult,
    CODEXCompliantCache
)

logger = logging.getLogger(__name__)


@dataclass
class TrustMetrics:
    """Trust metrics for Bayesian ranking."""
    base_score: float
    citation_count: int
    source_quality: float
    temporal_relevance: float = 1.0
    geographic_relevance: float = 1.0
    cross_reference_weight: float = 0.0
    
    @property
    def trust_score(self) -> float:
        """Calculate combined trust score using Bayesian approach."""
        # Bayesian trust calculation
        prior = self.source_quality
        likelihood = min(1.0, self.citation_count / 100)  # Normalize citations
        
        # Apply temporal and geographic factors
        context_factor = (self.temporal_relevance + self.geographic_relevance) / 2
        
        # Combine with cross-reference bonus
        trust = (prior * likelihood * context_factor) + (self.cross_reference_weight * 0.2)
        
        return min(1.0, trust)


@dataclass
class HierarchicalContext:
    """Hierarchical context combining global and local information."""
    global_summary: str
    local_details: List[str]
    section_hierarchy: List[str]
    parent_topics: List[str]
    child_topics: List[str]
    temporal_context: Optional[str] = None
    geographic_context: Optional[str] = None
    cross_references: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class EnhancedRetrievalResult(RetrievalResult):
    """Extended retrieval result with trust metrics and hierarchical context."""
    trust_metrics: Optional[TrustMetrics] = None
    hierarchical_context: Optional[HierarchicalContext] = None
    bayesian_score: float = 0.0
    context_type: str = "standard"  # standard, temporal, geographic, cross-reference


class BayesRAGEnhancedPipeline(CODEXRAGPipeline):
    """
    Enhanced CODEX pipeline with BayesRAG trust-weighted retrieval
    and hierarchical context metadata.
    """
    
    def __init__(self, bayesrag_data_dir: Path = Path("data")):
        super().__init__()
        
        self.bayesrag_data_dir = bayesrag_data_dir
        
        # BayesRAG databases
        self.global_context_db = bayesrag_data_dir / "wikipedia_global_context.db"
        self.local_context_db = bayesrag_data_dir / "wikipedia_local_context.db"
        self.graph_db = bayesrag_data_dir / "wikipedia_graph.db"
        
        # Trust score cache
        self.trust_cache = {}
        
        # Hierarchical context index
        self.hierarchy_index = {}
        
        # Cross-reference graph
        self.cross_reference_graph = {}
        
        # Load BayesRAG enhancements if available
        self._load_bayesrag_enhancements()
        
    def _load_bayesrag_enhancements(self):
        """Load BayesRAG trust scores and hierarchical context."""
        
        # Load global contexts with trust scores
        if self.global_context_db.exists():
            try:
                with sqlite3.connect(self.global_context_db) as conn:
                    cursor = conn.execute("""
                        SELECT title, trust_score, citation_count, source_quality
                        FROM global_contexts
                    """)
                    
                    for row in cursor.fetchall():
                        title = row[0]
                        self.trust_cache[title] = TrustMetrics(
                            base_score=row[1] if row[1] else 0.5,
                            citation_count=row[2] if row[2] else 0,
                            source_quality=row[3] if row[3] else 0.5
                        )
                        
                logger.info(f"Loaded trust scores for {len(self.trust_cache)} documents")
            except Exception as e:
                logger.warning(f"Failed to load trust scores: {e}")
                
        # Load hierarchical context
        if self.local_context_db.exists():
            try:
                with sqlite3.connect(self.local_context_db) as conn:
                    cursor = conn.execute("""
                        SELECT parent_title, section_title, local_summary,
                               temporal_context, geographic_context, cross_references
                        FROM local_contexts
                    """)
                    
                    for row in cursor.fetchall():
                        parent = row[0]
                        if parent not in self.hierarchy_index:
                            self.hierarchy_index[parent] = []
                            
                        self.hierarchy_index[parent].append({
                            'section': row[1],
                            'summary': row[2],
                            'temporal': row[3],
                            'geographic': row[4],
                            'cross_refs': json.loads(row[5]) if row[5] else []
                        })
                        
                logger.info(f"Loaded hierarchical context for {len(self.hierarchy_index)} documents")
            except Exception as e:
                logger.warning(f"Failed to load hierarchical context: {e}")
                
        # Load cross-reference graph
        if self.graph_db.exists():
            try:
                with sqlite3.connect(self.graph_db) as conn:
                    cursor = conn.execute("""
                        SELECT source_title, target_title, relationship_type, weight
                        FROM relationships
                    """)
                    
                    for row in cursor.fetchall():
                        source = row[0]
                        if source not in self.cross_reference_graph:
                            self.cross_reference_graph[source] = []
                            
                        self.cross_reference_graph[source].append({
                            'target': row[1],
                            'type': row[2],
                            'weight': row[3] if row[3] else 1.0
                        })
                        
                logger.info(f"Loaded cross-reference graph with {len(self.cross_reference_graph)} nodes")
            except Exception as e:
                logger.warning(f"Failed to load cross-reference graph: {e}")
                
    def _calculate_bayesian_score(
        self,
        base_score: float,
        trust_metrics: TrustMetrics,
        query_context: Dict[str, Any]
    ) -> float:
        """
        Calculate Bayesian-weighted retrieval score.
        
        Combines:
        - Base retrieval score (vector/keyword similarity)
        - Trust score (citations, source quality)
        - Context relevance (temporal, geographic)
        - Cross-reference bonus
        """
        
        # Prior: trust score
        prior = trust_metrics.trust_score
        
        # Likelihood: base retrieval score
        likelihood = base_score
        
        # Context adjustments
        context_multiplier = 1.0
        
        # Temporal relevance
        if query_context.get('temporal_filter'):
            context_multiplier *= trust_metrics.temporal_relevance
            
        # Geographic relevance
        if query_context.get('geographic_filter'):
            context_multiplier *= trust_metrics.geographic_relevance
            
        # Cross-reference bonus
        if trust_metrics.cross_reference_weight > 0:
            context_multiplier += (trust_metrics.cross_reference_weight * 0.1)
            
        # Bayesian combination
        bayesian_score = prior * likelihood * context_multiplier
        
        # Normalize to [0, 1]
        return min(1.0, bayesian_score)
        
    def _extract_query_context(self, query: str) -> Dict[str, Any]:
        """Extract temporal, geographic, and topical context from query."""
        
        context = {
            'temporal_filter': None,
            'geographic_filter': None,
            'topic_focus': None
        }
        
        # Temporal patterns
        temporal_patterns = [
            r'\b(19\d{2}|20\d{2})\b',  # Years
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
            r'\b(ancient|medieval|modern|contemporary)\b',
            r'\b(century|decade|year)\b'
        ]
        
        # Geographic patterns
        geographic_patterns = [
            r'\b(europe|asia|africa|america|australia|antarctica)\b',
            r'\b(country|city|region|continent)\b',
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'  # Proper nouns (potential places)
        ]
        
        query_lower = query.lower()
        
        # Check for temporal context
        import re
        for pattern in temporal_patterns:
            if re.search(pattern, query_lower):
                context['temporal_filter'] = True
                break
                
        # Check for geographic context
        for pattern in geographic_patterns:
            if re.search(pattern, query_lower):
                context['geographic_filter'] = True
                break
                
        # Extract main topic (simplified - would use NER in production)
        # Focus on capitalized multi-word phrases
        topic_match = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', query)
        if topic_match:
            context['topic_focus'] = topic_match[0]
            
        return context
        
    def _build_hierarchical_context(
        self,
        document_id: str,
        chunk_metadata: Dict[str, Any]
    ) -> HierarchicalContext:
        """Build hierarchical context for a retrieval result."""
        
        parent_title = chunk_metadata.get('parent_title', '')
        
        # Get hierarchy information
        hierarchy_info = self.hierarchy_index.get(parent_title, [])
        
        # Build context
        local_details = []
        cross_refs = []
        
        for info in hierarchy_info:
            if info['summary']:
                local_details.append(info['summary'])
            if info['cross_refs']:
                cross_refs.extend(info['cross_refs'])
                
        # Get parent and child topics from graph
        parent_topics = []
        child_topics = []
        
        if parent_title in self.cross_reference_graph:
            for ref in self.cross_reference_graph[parent_title]:
                if ref['type'] == 'parent':
                    parent_topics.append(ref['target'])
                elif ref['type'] == 'child':
                    child_topics.append(ref['target'])
                    
        return HierarchicalContext(
            global_summary=chunk_metadata.get('global_summary', ''),
            local_details=local_details[:5],  # Limit to top 5
            section_hierarchy=chunk_metadata.get('section_hierarchy', []),
            parent_topics=parent_topics[:3],  # Limit to top 3
            child_topics=child_topics[:5],  # Limit to top 5
            temporal_context=chunk_metadata.get('temporal_context'),
            geographic_context=chunk_metadata.get('geographic_context'),
            cross_references=cross_refs[:10]  # Limit to top 10
        )
        
    async def retrieve_with_trust(
        self,
        query: str,
        k: int = 10,
        use_cache: bool = True,
        trust_weight: float = 0.3
    ) -> Tuple[List[EnhancedRetrievalResult], Dict[str, Any]]:
        """
        Retrieve with trust-weighted ranking and hierarchical context.
        
        Args:
            query: Search query
            k: Number of results to return
            use_cache: Whether to use caching
            trust_weight: Weight for trust score (0-1)
            
        Returns:
            Enhanced retrieval results with trust metrics and hierarchical context
        """
        
        start_time = time.perf_counter()
        
        # Extract query context
        query_context = self._extract_query_context(query)
        
        # Get base retrieval results
        base_results, base_metrics = await super().retrieve(query, k=k*2, use_cache=use_cache)
        
        # Enhance results with trust metrics and hierarchical context
        enhanced_results = []
        
        for result in base_results:
            # Get trust metrics
            doc_title = result.metadata.get('parent_title', '') if result.metadata else ''
            trust_metrics = self.trust_cache.get(doc_title, TrustMetrics(
                base_score=0.5,
                citation_count=0,
                source_quality=0.5
            ))
            
            # Update trust metrics based on query context
            if query_context['temporal_filter'] and result.metadata:
                if result.metadata.get('temporal_context'):
                    trust_metrics.temporal_relevance = 1.2  # Boost
                else:
                    trust_metrics.temporal_relevance = 0.8  # Penalty
                    
            if query_context['geographic_filter'] and result.metadata:
                if result.metadata.get('geographic_context'):
                    trust_metrics.geographic_relevance = 1.2  # Boost
                else:
                    trust_metrics.geographic_relevance = 0.8  # Penalty
                    
            # Check for cross-references
            if doc_title in self.cross_reference_graph:
                trust_metrics.cross_reference_weight = len(self.cross_reference_graph[doc_title]) / 10
                
            # Calculate Bayesian score
            bayesian_score = self._calculate_bayesian_score(
                result.score,
                trust_metrics,
                query_context
            )
            
            # Build hierarchical context
            hierarchical_context = None
            if result.metadata:
                hierarchical_context = self._build_hierarchical_context(
                    result.document_id,
                    result.metadata
                )
                
            # Determine context type
            context_type = "standard"
            if query_context['temporal_filter'] and trust_metrics.temporal_relevance > 1.0:
                context_type = "temporal"
            elif query_context['geographic_filter'] and trust_metrics.geographic_relevance > 1.0:
                context_type = "geographic"
            elif trust_metrics.cross_reference_weight > 0.5:
                context_type = "cross-reference"
                
            # Create enhanced result
            enhanced_result = EnhancedRetrievalResult(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                text=result.text,
                score=result.score,
                retrieval_method=result.retrieval_method,
                metadata=result.metadata,
                trust_metrics=trust_metrics,
                hierarchical_context=hierarchical_context,
                bayesian_score=bayesian_score,
                context_type=context_type
            )
            
            enhanced_results.append(enhanced_result)
            
        # Re-rank by Bayesian score
        enhanced_results.sort(key=lambda x: x.bayesian_score, reverse=True)
        
        # Take top k results
        enhanced_results = enhanced_results[:k]
        
        # Calculate enhanced metrics
        latency = (time.perf_counter() - start_time) * 1000
        
        enhanced_metrics = {
            **base_metrics,
            'latency_ms': latency,
            'trust_weighted': True,
            'query_context': query_context,
            'avg_trust_score': np.mean([r.trust_metrics.trust_score for r in enhanced_results if r.trust_metrics]),
            'context_types': list(set(r.context_type for r in enhanced_results))
        }
        
        return enhanced_results, enhanced_metrics
        
    def format_hierarchical_response(
        self,
        results: List[EnhancedRetrievalResult],
        max_context_length: int = 2000
    ) -> str:
        """
        Format retrieval results with hierarchical context for response generation.
        
        Returns a structured context string optimized for LLM consumption.
        """
        
        if not results:
            return ""
            
        response_parts = []
        
        # Add global context summary
        top_result = results[0]
        if top_result.hierarchical_context:
            response_parts.append("## Global Context")
            response_parts.append(top_result.hierarchical_context.global_summary[:500])
            response_parts.append("")
            
        # Add top results with trust indicators
        response_parts.append("## Relevant Information (Trust-Weighted)")
        
        for i, result in enumerate(results[:5], 1):
            trust_indicator = "⭐" * min(5, int(result.trust_metrics.trust_score * 5)) if result.trust_metrics else ""
            
            response_parts.append(f"\n### Result {i} {trust_indicator}")
            response_parts.append(f"**Source**: {result.metadata.get('parent_title', 'Unknown')}")
            
            if result.trust_metrics:
                response_parts.append(f"**Trust Score**: {result.trust_metrics.trust_score:.2f}")
                response_parts.append(f"**Citations**: {result.trust_metrics.citation_count}")
                
            if result.context_type != "standard":
                response_parts.append(f"**Context Type**: {result.context_type}")
                
            response_parts.append(f"\n{result.text[:400]}")
            
            # Add hierarchical context details
            if result.hierarchical_context:
                if result.hierarchical_context.temporal_context:
                    response_parts.append(f"**Time Period**: {result.hierarchical_context.temporal_context}")
                    
                if result.hierarchical_context.geographic_context:
                    response_parts.append(f"**Location**: {result.hierarchical_context.geographic_context}")
                    
                if result.hierarchical_context.cross_references:
                    refs = ", ".join([ref.get('title', '') for ref in result.hierarchical_context.cross_references[:3]])
                    response_parts.append(f"**Related Topics**: {refs}")
                    
        # Add cross-reference network if available
        if any(r.hierarchical_context and r.hierarchical_context.cross_references for r in results):
            response_parts.append("\n## Related Topics Network")
            
            all_refs = set()
            for result in results:
                if result.hierarchical_context and result.hierarchical_context.cross_references:
                    for ref in result.hierarchical_context.cross_references:
                        all_refs.add(ref.get('title', ''))
                        
            if all_refs:
                response_parts.append(", ".join(list(all_refs)[:10]))
                
        # Combine and truncate
        response = "\n".join(response_parts)
        
        if len(response) > max_context_length:
            response = response[:max_context_length] + "\n\n[Context truncated]"
            
        return response


class SemanticCache(CODEXCompliantCache):
    """
    Enhanced cache with semantic matching and trust-based prioritization.
    """
    
    def __init__(self, embedding_model: SentenceTransformer = None):
        super().__init__()
        
        # Use same embedding model as pipeline
        self.embedder = embedding_model or SentenceTransformer("paraphrase-MiniLM-L3-v2")
        
        # Semantic cache for embedding-based matching
        self.semantic_cache = {}
        self.cache_embeddings = []
        self.cache_keys = []
        
        # Trust-weighted priority queue
        self.priority_scores = {}
        
    async def get_semantic(
        self,
        query: str,
        similarity_threshold: float = 0.85
    ) -> Optional[List[RetrievalResult]]:
        """
        Get cached results using semantic similarity matching.
        
        Args:
            query: Search query
            similarity_threshold: Minimum similarity for cache hit
            
        Returns:
            Cached results if semantically similar query found
        """
        
        if not self.enabled or not self.cache_embeddings:
            return None
            
        start_time = time.perf_counter()
        
        # Encode query
        query_embedding = self.embedder.encode(query, convert_to_numpy=True)
        
        # Calculate similarities
        similarities = []
        for cached_embedding in self.cache_embeddings:
            similarity = np.dot(query_embedding, cached_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(cached_embedding)
            )
            similarities.append(similarity)
            
        # Find best match
        max_similarity = max(similarities) if similarities else 0
        
        if max_similarity >= similarity_threshold:
            best_idx = similarities.index(max_similarity)
            cache_key = self.cache_keys[best_idx]
            
            # Get from regular cache
            result = await super().get(cache_key)
            
            if result:
                latency = (time.perf_counter() - start_time) * 1000
                logger.debug(f"Semantic cache hit (similarity: {max_similarity:.3f}, latency: {latency:.2f}ms)")
                return result
                
        return None
        
    async def set_with_priority(
        self,
        query: str,
        results: List[RetrievalResult],
        trust_score: float = 0.5
    ) -> None:
        """
        Store results with trust-based priority for cache eviction.
        
        Higher trust content stays in cache longer.
        """
        
        if not self.enabled:
            return
            
        # Store in regular cache
        await super().set(query, results)
        
        # Add to semantic cache
        query_embedding = self.embedder.encode(query, convert_to_numpy=True)
        cache_key = self._make_key(query)
        
        self.semantic_cache[cache_key] = query_embedding
        self.cache_embeddings.append(query_embedding)
        self.cache_keys.append(cache_key)
        
        # Store priority score
        self.priority_scores[cache_key] = trust_score
        
        # Limit semantic cache size
        max_semantic_cache = 1000
        if len(self.cache_embeddings) > max_semantic_cache:
            # Evict lowest priority items
            sorted_keys = sorted(
                self.priority_scores.items(),
                key=lambda x: x[1]
            )
            
            # Remove lowest priority 10%
            to_remove = sorted_keys[:max_semantic_cache // 10]
            
            for key, _ in to_remove:
                if key in self.semantic_cache:
                    idx = self.cache_keys.index(key)
                    del self.cache_embeddings[idx]
                    del self.cache_keys[idx]
                    del self.semantic_cache[key]
                    del self.priority_scores[key]


async def test_enhanced_pipeline():
    """Test the enhanced BayesRAG-CODEX integration."""
    
    logger.info("Testing Enhanced BayesRAG-CODEX Pipeline...")
    
    # Initialize enhanced pipeline
    pipeline = BayesRAGEnhancedPipeline()
    
    # Test queries with different context types
    test_queries = [
        "What were the causes of World War I in Europe?",  # Temporal + Geographic
        "Explain quantum computing applications",  # Technical
        "Industrial Revolution impact on British society",  # Temporal + Geographic + Topic
        "Machine learning algorithms for classification",  # Technical + Specific
        "Ancient Egyptian pyramid construction methods"  # Temporal + Geographic + Technical
    ]
    
    print("\n=== Enhanced Retrieval Tests ===\n")
    
    for query in test_queries:
        print(f"Query: {query}")
        print("-" * 50)
        
        # Retrieve with trust weighting
        results, metrics = await pipeline.retrieve_with_trust(
            query=query,
            k=3,
            trust_weight=0.4
        )
        
        print(f"Results: {len(results)}")
        print(f"Latency: {metrics['latency_ms']:.2f}ms")
        print(f"Query Context: {metrics['query_context']}")
        print(f"Avg Trust Score: {metrics.get('avg_trust_score', 0):.3f}")
        print(f"Context Types: {metrics.get('context_types', [])}")
        
        # Show top result with trust metrics
        if results:
            top = results[0]
            print(f"\nTop Result:")
            print(f"  Text: {top.text[:150]}...")
            print(f"  Bayesian Score: {top.bayesian_score:.3f}")
            
            if top.trust_metrics:
                print(f"  Trust Score: {top.trust_metrics.trust_score:.3f}")
                print(f"  Citations: {top.trust_metrics.citation_count}")
                
            if top.hierarchical_context:
                print(f"  Global Summary: {top.hierarchical_context.global_summary[:100]}...")
                
                if top.hierarchical_context.cross_references:
                    refs = [r.get('title', '') for r in top.hierarchical_context.cross_references[:3]]
                    print(f"  Cross References: {', '.join(refs)}")
                    
        print("\n")
        
    # Test hierarchical response formatting
    print("=== Hierarchical Response Format ===\n")
    
    if results:
        formatted = pipeline.format_hierarchical_response(results)
        print(formatted[:1000])
        print("\n[Response truncated for display]")
        
    return True


if __name__ == "__main__":
    asyncio.run(test_enhanced_pipeline())