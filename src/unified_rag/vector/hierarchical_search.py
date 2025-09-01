"""
Hierarchical Search Engine for Vector RAG
Implements multi-level search with context hierarchy and scope management
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from .contextual_embeddings import ContextualEmbeddingEngine, ContextualEmbedding

logger = logging.getLogger(__name__)

class SearchScope(Enum):
    """Search scope levels for hierarchical search"""
    GLOBAL = "global"           # Search entire knowledge base
    DOMAIN = "domain"           # Search within specific domain
    CONTEXT = "context"         # Search within specific context
    LOCAL = "local"             # Search within local neighborhood

@dataclass
class SearchNode:
    """Node in hierarchical search structure"""
    id: str
    level: SearchScope
    content: str
    embedding: ContextualEmbedding
    children: List['SearchNode'] = field(default_factory=list)
    parent: Optional['SearchNode'] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_child(self, child: 'SearchNode'):
        """Add child node and set parent reference"""
        child.parent = self
        self.children.append(child)

@dataclass
class SearchResult:
    """Result from hierarchical search"""
    node: SearchNode
    score: float
    level: SearchScope
    path: List[str]  # Path from root to this node
    context_relevance: float = 0.0
    semantic_relevance: float = 0.0

class HierarchicalSearchEngine:
    """
    Multi-level hierarchical search engine with context-aware ranking
    """
    
    def __init__(self, embedding_engine: ContextualEmbeddingEngine):
        self.embedding_engine = embedding_engine
        self.search_tree: Dict[str, SearchNode] = {}
        self.root_nodes: List[SearchNode] = []
        self.level_indices: Dict[SearchScope, List[SearchNode]] = {
            level: [] for level in SearchScope
        }
    
    async def add_node(self, 
                      content: str,
                      level: SearchScope,
                      parent_id: Optional[str] = None,
                      metadata: Optional[Dict[str, Any]] = None,
                      context: Optional[str] = None) -> str:
        """
        Add node to hierarchical search structure
        
        Args:
            content: Text content of the node
            level: Search scope level
            parent_id: ID of parent node (if any)
            metadata: Additional metadata
            context: Semantic context for embedding
            
        Returns:
            Generated node ID
        """
        # Generate contextual embedding
        embedding = await self.embedding_engine.embed_with_context(
            content, context, metadata=metadata
        )
        
        # Create node
        node_id = f"{level.value}_{len(self.search_tree)}"
        node = SearchNode(
            id=node_id,
            level=level,
            content=content,
            embedding=embedding,
            metadata=metadata or {}
        )
        
        # Add to tree structure
        if parent_id and parent_id in self.search_tree:
            parent = self.search_tree[parent_id]
            parent.add_child(node)
        else:
            self.root_nodes.append(node)
        
        # Add to indices
        self.search_tree[node_id] = node
        self.level_indices[level].append(node)
        
        logger.debug(f"Added node {node_id} at level {level.value}")
        return node_id
    
    async def hierarchical_search(self,
                                query: str,
                                max_results: int = 10,
                                search_levels: Optional[List[SearchScope]] = None,
                                context: Optional[str] = None,
                                min_score: float = 0.3) -> List[SearchResult]:
        """
        Perform hierarchical search across multiple levels
        
        Args:
            query: Search query
            max_results: Maximum number of results
            search_levels: Specific levels to search (default: all)
            context: Additional context for query embedding
            min_score: Minimum similarity score
            
        Returns:
            List of search results ranked by relevance
        """
        # Generate query embedding
        query_embedding = await self.embedding_engine.embed_with_context(
            query, context
        )
        
        # Determine search levels
        if search_levels is None:
            search_levels = list(SearchScope)
        
        # Collect results from all levels
        all_results = []
        
        for level in search_levels:
            level_results = await self._search_level(
                query_embedding, level, min_score
            )
            all_results.extend(level_results)
        
        # Rank and filter results
        ranked_results = self._rank_results(all_results, query_embedding)
        
        return ranked_results[:max_results]
    
    async def _search_level(self,
                           query_embedding: ContextualEmbedding,
                           level: SearchScope,
                           min_score: float) -> List[SearchResult]:
        """Search within a specific level"""
        results = []
        
        for node in self.level_indices[level]:
            # Compute similarity
            score = self.embedding_engine.compute_similarity(
                query_embedding, node.embedding
            )
            
            if score >= min_score:
                # Build path from root
                path = self._build_path(node)
                
                result = SearchResult(
                    node=node,
                    score=score,
                    level=level,
                    path=path,
                    context_relevance=score,  # Can be enhanced
                    semantic_relevance=score  # Can be enhanced
                )
                results.append(result)
        
        return results
    
    def _build_path(self, node: SearchNode) -> List[str]:
        """Build path from root to node"""
        path = []
        current = node
        
        while current:
            path.append(current.id)
            current = current.parent
        
        return list(reversed(path))
    
    def _rank_results(self, 
                     results: List[SearchResult],
                     query_embedding: ContextualEmbedding) -> List[SearchResult]:
        """Rank results using hierarchical scoring"""
        
        # Enhanced ranking considering:
        # 1. Base similarity score
        # 2. Level importance
        # 3. Path context
        # 4. Node metadata
        
        level_weights = {
            SearchScope.GLOBAL: 0.6,
            SearchScope.DOMAIN: 0.8,
            SearchScope.CONTEXT: 1.0,
            SearchScope.LOCAL: 0.9
        }
        
        for result in results:
            # Apply level weight
            level_weight = level_weights.get(result.level, 1.0)
            
            # Path depth penalty (prefer more specific results)
            path_factor = 1.0 + (len(result.path) * 0.1)
            
            # Final ranking score
            result.score = result.score * level_weight * path_factor
        
        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        
        return results
    
    async def expand_search(self,
                           base_results: List[SearchResult],
                           expansion_depth: int = 1) -> List[SearchResult]:
        """
        Expand search results by including related nodes
        
        Args:
            base_results: Initial search results
            expansion_depth: Depth of expansion (children/siblings)
            
        Returns:
            Expanded results list
        """
        expanded_results = base_results.copy()
        
        for result in base_results:
            node = result.node
            
            # Add children if within depth
            if expansion_depth > 0:
                for child in node.children:
                    child_result = SearchResult(
                        node=child,
                        score=result.score * 0.8,  # Reduce score for children
                        level=child.level,
                        path=self._build_path(child),
                        context_relevance=result.context_relevance * 0.8,
                        semantic_relevance=result.semantic_relevance * 0.8
                    )
                    expanded_results.append(child_result)
            
            # Add siblings (same parent)
            if node.parent:
                for sibling in node.parent.children:
                    if sibling.id != node.id:
                        sibling_result = SearchResult(
                            node=sibling,
                            score=result.score * 0.9,  # Slight reduction for siblings
                            level=sibling.level,
                            path=self._build_path(sibling),
                            context_relevance=result.context_relevance * 0.9,
                            semantic_relevance=result.semantic_relevance * 0.9
                        )
                        expanded_results.append(sibling_result)
        
        # Remove duplicates and re-rank
        seen_ids = set()
        unique_results = []
        for result in expanded_results:
            if result.node.id not in seen_ids:
                unique_results.append(result)
                seen_ids.add(result.node.id)
        
        unique_results.sort(key=lambda r: r.score, reverse=True)
        return unique_results
    
    def get_tree_stats(self) -> Dict[str, Any]:
        """Get statistics about the search tree"""
        return {
            'total_nodes': len(self.search_tree),
            'root_nodes': len(self.root_nodes),
            'nodes_by_level': {
                level.value: len(nodes) 
                for level, nodes in self.level_indices.items()
            },
            'max_depth': self._get_max_depth(),
            'avg_children_per_node': self._get_avg_children()
        }
    
    def _get_max_depth(self) -> int:
        """Calculate maximum depth of tree"""
        max_depth = 0
        
        for root in self.root_nodes:
            depth = self._calculate_depth(root, 0)
            max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _calculate_depth(self, node: SearchNode, current_depth: int) -> int:
        """Recursively calculate depth of a subtree"""
        if not node.children:
            return current_depth
        
        max_child_depth = current_depth
        for child in node.children:
            child_depth = self._calculate_depth(child, current_depth + 1)
            max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth
    
    def _get_avg_children(self) -> float:
        """Calculate average number of children per node"""
        if not self.search_tree:
            return 0.0
        
        total_children = sum(len(node.children) for node in self.search_tree.values())
        return total_children / len(self.search_tree)