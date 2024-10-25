from typing import Dict, Any, List, Optional, Tuple
import re
from dataclasses import dataclass
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

@dataclass
class QueryContext:
    """Context information for query processing."""
    original_query: str
    timestamp: Optional[float] = None
    user_context: Optional[Dict[str, Any]] = None
    previous_queries: Optional[List[str]] = None
    session_data: Optional[Dict[str, Any]] = None

class SelfReferentialQueryProcessor:
    """
    Implements self-referential query processing capabilities.
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.query_history: List[str] = []
        self.context_history: List[Dict[str, Any]] = []
        
    async def process_query(self, query: str, context: Dict[str, Any]) -> str:
        """
        Process query with self-referential capabilities.

        :param query: The original query string.
        :param context: Additional context information.
        :return: Processed query string.
        """
        # Create query context
        query_context = QueryContext(
            original_query=query,
            timestamp=context.get('timestamp'),
            user_context=context.get('user_context'),
            previous_queries=self.query_history[-5:],  # Last 5 queries
            session_data=context.get('session_data')
        )
        
        # Extract self-references
        self_refs = self._extract_self_references(query)
        
        # Resolve references using context
        resolved_refs = await self._resolve_references(self_refs, query_context)
        
        # Reconstruct query
        processed_query = self._reconstruct_query(query, resolved_refs)
        
        # Update history
        self._update_history(query, context)
        
        return processed_query
        
    def _extract_self_references(self, query: str) -> List[Dict[str, Any]]:
        """
        Extract self-referential components from query.

        :param query: The query string.
        :return: List of extracted references.
        """
        references = []
        
        # Pattern for "previous" references
        previous_pattern = r"(previous|last|earlier) (query|result|answer|response)"
        previous_matches = re.finditer(previous_pattern, query, re.IGNORECASE)
        
        for match in previous_matches:
            references.append({
                'type': 'previous',
                'span': match.span(),
                'text': match.group(),
                'target': match.group(2)
            })
            
        # Pattern for "that" references
        that_pattern = r"that (query|result|answer|response)"
        that_matches = re.finditer(that_pattern, query, re.IGNORECASE)
        
        for match in that_matches:
            references.append({
                'type': 'that',
                'span': match.span(),
                'text': match.group(),
                'target': match.group(1)
            })
            
        return references
        
    async def _resolve_references(self, references: List[Dict[str, Any]], context: QueryContext) -> Dict[str, str]:
        """
        Resolve extracted references using context.

        :param references: List of extracted references.
        :param context: Query context.
        :return: Dictionary of resolved references.
        """
        resolved = {}
        
        for ref in references:
            if ref['type'] == 'previous':
                resolved[ref['text']] = self._resolve_previous_reference(ref, context)
            elif ref['type'] == 'that':
                resolved[ref['text']] = self._resolve_that_reference(ref, context)
                
        return resolved
        
    def _resolve_previous_reference(self, reference: Dict[str, Any], context: QueryContext) -> str:
        """
        Resolve reference to previous query or result.

        :param reference: Reference information.
        :param context: Query context.
        :return: Resolved reference text.
        """
        if not context.previous_queries:
            return ""
            
        if reference['target'] in ['query', 'question']:
            return context.previous_queries[-1]
            
        if reference['target'] in ['result', 'answer', 'response']:
            if self.context_history:
                last_context = self.context_history[-1]
                return last_context.get('result', "")
                
        return ""
        
    def _resolve_that_reference(self, reference: Dict[str, Any], context: QueryContext) -> str:
        """
        Resolve "that" reference using context.

        :param reference: Reference information.
        :param context: Query context.
        :return: Resolved reference text.
        """
        # Try to find the most relevant previous content
        if self.context_history:
            last_context = self.context_history[-1]
            
            if reference['target'] in ['result', 'answer', 'response']:
                return last_context.get('result', "")
                
            if reference['target'] == 'query':
                return last_context.get('query', "")
                
        return ""
        
    def _reconstruct_query(self, original_query: str, resolved_refs: Dict[str, str]) -> str:
        """
        Reconstruct query by replacing references with resolved content.

        :param original_query: Original query string.
        :param resolved_refs: Dictionary of resolved references.
        :return: Reconstructed query string.
        """
        reconstructed = original_query
        
        # Sort references by span position (reverse order to avoid position shifts)
        sorted_refs = sorted(
            [(ref, resolved) for ref, resolved in resolved_refs.items()],
            key=lambda x: len(x[0]),
            reverse=True
        )
        
        # Replace references with resolved content
        for ref, resolved in sorted_refs:
            if resolved:
                reconstructed = reconstructed.replace(ref, resolved)
                
        return reconstructed
        
    def _update_history(self, query: str, context: Dict[str, Any]):
        """
        Update query and context history.

        :param query: Current query string.
        :param context: Current context.
        """
        self.query_history.append(query)
