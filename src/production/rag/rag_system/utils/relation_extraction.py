import re
import logging
from typing import List, Dict, Tuple, Any

logger = logging.getLogger(__name__)


class RelationExtractor:
    """Extract relationships between entities in text using pattern matching and NLP techniques."""
    
    def __init__(self):
        # Common relationship patterns
        self.relation_patterns = {
            'is_a': [
                r'(\w+)\s+is\s+a\s+(\w+)',
                r'(\w+)\s+are\s+(\w+)',
                r'(\w+)\s+is\s+an?\s+(\w+)',
            ],
            'has': [
                r'(\w+)\s+has\s+(\w+)',
                r'(\w+)\s+have\s+(\w+)',
                r'(\w+)\s+contains?\s+(\w+)',
                r'(\w+)\s+includes?\s+(\w+)',
            ],
            'located_in': [
                r'(\w+)\s+is\s+located\s+in\s+(\w+)',
                r'(\w+)\s+is\s+in\s+(\w+)',
                r'(\w+)\s+located\s+in\s+(\w+)',
            ],
            'works_for': [
                r'(\w+)\s+works?\s+for\s+(\w+)',
                r'(\w+)\s+is\s+employed\s+by\s+(\w+)',
                r'(\w+)\s+employee\s+of\s+(\w+)',
            ],
            'created_by': [
                r'(\w+)\s+created\s+by\s+(\w+)',
                r'(\w+)\s+developed\s+by\s+(\w+)',
                r'(\w+)\s+built\s+by\s+(\w+)',
                r'(\w+)\s+made\s+by\s+(\w+)',
            ],
            'related_to': [
                r'(\w+)\s+related\s+to\s+(\w+)',
                r'(\w+)\s+associated\s+with\s+(\w+)',
                r'(\w+)\s+connected\s+to\s+(\w+)',
            ]
        }
    
    def extract(self, text: str) -> List[Dict[str, Any]]:
        """Extract relationships from text using pattern matching.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of relationship dictionaries with 'subject', 'predicate', 'object', 'confidence'
        """
        if not text or not isinstance(text, str):
            logger.warning("Invalid text input for relation extraction")
            return []
        
        relations = []
        text_lower = text.lower()
        
        try:
            # Extract relationships using pattern matching
            for relation_type, patterns in self.relation_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                    
                    for match in matches:
                        if len(match.groups()) >= 2:
                            subject = match.group(1).strip()
                            obj = match.group(2).strip()
                            
                            # Basic filtering for meaningful entities
                            if len(subject) > 1 and len(obj) > 1 and subject != obj:
                                relations.append({
                                    'subject': subject,
                                    'predicate': relation_type,
                                    'object': obj,
                                    'confidence': self._calculate_confidence(match, text),
                                    'source_text': match.group(0),
                                    'position': match.span()
                                })
            
            # Remove duplicates and sort by confidence
            relations = self._deduplicate_relations(relations)
            relations.sort(key=lambda x: x['confidence'], reverse=True)
            
            logger.info(f"Extracted {len(relations)} relationships from text")
            return relations
            
        except Exception as e:
            logger.error(f"Error in relation extraction: {e}")
            return []
    
    def _calculate_confidence(self, match: re.Match, text: str) -> float:
        """Calculate confidence score for a relation match."""
        # Simple confidence based on match quality
        base_confidence = 0.7
        
        # Boost confidence for longer entities
        subject_len = len(match.group(1).strip())
        object_len = len(match.group(2).strip())
        length_bonus = min((subject_len + object_len) / 20, 0.2)
        
        # Reduce confidence for very short entities
        if subject_len < 3 or object_len < 3:
            length_bonus -= 0.3
        
        return max(0.1, min(1.0, base_confidence + length_bonus))
    
    def _deduplicate_relations(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate relations based on subject-predicate-object triples."""
        seen = set()
        deduplicated = []
        
        for relation in relations:
            triple = (
                relation['subject'].lower(),
                relation['predicate'],
                relation['object'].lower()
            )
            
            if triple not in seen:
                seen.add(triple)
                deduplicated.append(relation)
        
        return deduplicated
