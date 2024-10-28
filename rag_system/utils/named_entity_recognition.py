"""Named entity recognition utilities for RAG system."""

from typing import List, Dict, Any, Optional, Tuple
import spacy
from ..core.config import UnifiedConfig

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # Fallback to small model if main model not available
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # Download if not available
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")

def extract_entities(
    text: str,
    config: Optional[UnifiedConfig] = None,
    entity_types: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Extract named entities from text.
    
    Args:
        text: Input text
        config: Optional configuration
        entity_types: Optional list of entity types to extract
        
    Returns:
        List of extracted entities with metadata
    """
    # Process text
    doc = nlp(text)
    
    # Filter entity types if specified
    if entity_types:
        entities = [ent for ent in doc.ents if ent.label_ in entity_types]
    else:
        entities = doc.ents
    
    # Convert to structured format
    return [
        {
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
            "description": spacy.explain(ent.label_)
        }
        for ent in entities
    ]

def extract_relations(
    text: str,
    config: Optional[UnifiedConfig] = None
) -> List[Dict[str, Any]]:
    """
    Extract relations between entities.
    
    Args:
        text: Input text
        config: Optional configuration
        
    Returns:
        List of extracted relations
    """
    doc = nlp(text)
    relations = []
    
    for token in doc:
        if token.dep_ in ["nsubj", "nsubjpass"]:  # Subject-verb relations
            subject = token
            verb = token.head
            objects = [child for child in verb.children if child.dep_ in ["dobj", "pobj"]]
            
            for obj in objects:
                relations.append({
                    "subject": subject.text,
                    "predicate": verb.text,
                    "object": obj.text,
                    "relation_type": "subject-verb-object",
                    "confidence": 0.8  # Placeholder confidence score
                })
    
    return relations

def extract_entity_context(
    text: str,
    entity: str,
    window_size: int = 50,
    config: Optional[UnifiedConfig] = None
) -> List[Dict[str, Any]]:
    """
    Extract context around entity mentions.
    
    Args:
        text: Input text
        entity: Entity to find context for
        window_size: Size of context window
        config: Optional configuration
        
    Returns:
        List of context windows
    """
    doc = nlp(text)
    contexts = []
    
    for sent in doc.sents:
        sent_text = sent.text.lower()
        if entity.lower() in sent_text:
            # Find entity position
            start_idx = sent_text.index(entity.lower())
            end_idx = start_idx + len(entity)
            
            # Get context window
            context_start = max(0, start_idx - window_size)
            context_end = min(len(sent_text), end_idx + window_size)
            
            contexts.append({
                "entity": entity,
                "context": sent_text[context_start:context_end],
                "sentence": sent.text,
                "position": (start_idx, end_idx)
            })
    
    return contexts

def extract_entity_attributes(
    text: str,
    entity: str,
    config: Optional[UnifiedConfig] = None
) -> Dict[str, Any]:
    """
    Extract attributes associated with an entity.
    
    Args:
        text: Input text
        entity: Entity to analyze
        config: Optional configuration
        
    Returns:
        Dictionary of entity attributes
    """
    doc = nlp(text)
    attributes = {
        "entity": entity,
        "mentions": [],
        "modifiers": [],
        "actions": [],
        "relations": []
    }
    
    for sent in doc.sents:
        if entity.lower() in sent.text.lower():
            # Find entity token
            entity_tokens = [token for token in sent if token.text.lower() == entity.lower()]
            
            for token in entity_tokens:
                # Record mention
                attributes["mentions"].append({
                    "text": token.text,
                    "sentence": sent.text,
                    "position": (token.idx, token.idx + len(token.text))
                })
                
                # Find modifiers (adjectives, numbers)
                modifiers = [child.text for child in token.children 
                           if child.pos_ in ["ADJ", "NUM"]]
                attributes["modifiers"].extend(modifiers)
                
                # Find actions (verbs)
                if token.head.pos_ == "VERB":
                    attributes["actions"].append({
                        "verb": token.head.text,
                        "sentence": sent.text
                    })
                
                # Find relations
                for child in token.children:
                    if child.dep_ in ["prep", "agent"]:
                        for grandchild in child.children:
                            if grandchild.pos_ in ["NOUN", "PROPN"]:
                                attributes["relations"].append({
                                    "type": child.text,
                                    "target": grandchild.text,
                                    "sentence": sent.text
                                })
    
    # Remove duplicates
    attributes["modifiers"] = list(set(attributes["modifiers"]))
    
    return attributes

def batch_extract_entities(
    texts: List[str],
    config: Optional[UnifiedConfig] = None,
    entity_types: Optional[List[str]] = None,
    batch_size: int = 32
) -> List[List[Dict[str, Any]]]:
    """
    Extract entities from multiple texts.
    
    Args:
        texts: List of input texts
        config: Optional configuration
        entity_types: Optional list of entity types to extract
        batch_size: Size of batches to process
        
    Returns:
        List of entity lists for each text
    """
    all_entities = []
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        docs = list(nlp.pipe(batch))
        
        batch_entities = []
        for doc in docs:
            # Filter entity types if specified
            if entity_types:
                entities = [ent for ent in doc.ents if ent.label_ in entity_types]
            else:
                entities = doc.ents
            
            # Convert to structured format
            batch_entities.append([
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "description": spacy.explain(ent.label_)
                }
                for ent in entities
            ])
        
        all_entities.extend(batch_entities)
    
    return all_entities

def get_entity_statistics(
    texts: List[str],
    config: Optional[UnifiedConfig] = None
) -> Dict[str, Any]:
    """
    Get statistics about entities in texts.
    
    Args:
        texts: List of input texts
        config: Optional configuration
        
    Returns:
        Dictionary of entity statistics
    """
    all_entities = []
    entity_types = {}
    entity_counts = {}
    
    # Process all texts
    docs = nlp.pipe(texts)
    
    for doc in docs:
        for ent in doc.ents:
            all_entities.append(ent)
            
            # Count entity types
            entity_types[ent.label_] = entity_types.get(ent.label_, 0) + 1
            
            # Count unique entities
            entity_counts[ent.text.lower()] = entity_counts.get(ent.text.lower(), 0) + 1
    
    return {
        "total_entities": len(all_entities),
        "unique_entities": len(entity_counts),
        "entity_types": entity_types,
        "most_common": sorted(
            entity_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10],
        "type_distribution": {
            label: count / len(all_entities)
            for label, count in entity_types.items()
        }
    }
