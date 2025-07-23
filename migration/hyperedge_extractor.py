#!/usr/bin/env python3
"""
Hyperedge Extractor

Extracts multi-entity relationships (hyperedges) from document corpus.
Creates rich semantic relationships for HypeRAG knowledge graph.
"""

import json
import logging
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set
from datetime import datetime, timezone
import numpy as np
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter

# Import HypeRAG components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from mcp_servers.hyperag.memory.hypergraph_kg import HypergraphKG

logger = logging.getLogger(__name__)

@dataclass
class Hyperedge:
    """Multi-entity relationship (hyperedge)"""
    hyperedge_id: str
    entities: List[str]
    relationship_type: str
    confidence: float
    context: str
    source_document: str
    extraction_method: str
    properties: Dict[str, Any]

@dataclass
class ExtractionMetrics:
    """Metrics for hyperedge extraction process"""
    total_documents_processed: int
    total_hyperedges_extracted: int
    hyperedges_by_type: Dict[str, int]
    hyperedges_by_method: Dict[str, int]
    avg_entities_per_hyperedge: float
    avg_confidence: float
    extraction_time: float

class PatternBasedExtractor:
    """Extracts hyperedges using linguistic patterns"""

    def __init__(self):
        self.patterns = self._load_extraction_patterns()
        self.entity_cache = {}

    def _load_extraction_patterns(self) -> Dict[str, List[Dict]]:
        """Load linguistic patterns for relationship extraction"""
        return {
            "temporal_relations": [
                {
                    "pattern": r"(\w+)\s+(?:happened|occurred|took place)\s+(?:before|after|during)\s+(\w+)",
                    "relationship": "TEMPORAL_RELATION",
                    "confidence": 0.8
                },
                {
                    "pattern": r"(\w+)\s+(?:was|were)\s+(?:founded|created|established)\s+(?:in|on|during)\s+(\d{4}|\w+)",
                    "relationship": "FOUNDED_IN",
                    "confidence": 0.9
                }
            ],
            "causal_relations": [
                {
                    "pattern": r"(\w+)\s+(?:caused|led to|resulted in)\s+(\w+)",
                    "relationship": "CAUSES",
                    "confidence": 0.85
                },
                {
                    "pattern": r"(\w+)\s+(?:because of|due to|as a result of)\s+(\w+)",
                    "relationship": "CAUSED_BY",
                    "confidence": 0.8
                }
            ],
            "compositional_relations": [
                {
                    "pattern": r"(\w+)\s+(?:consists of|contains|includes)\s+(\w+)(?:,\s*(\w+))*",
                    "relationship": "COMPOSED_OF",
                    "confidence": 0.9
                },
                {
                    "pattern": r"(\w+)(?:,\s*(\w+))*\s+(?:are|is)\s+(?:part of|components of)\s+(\w+)",
                    "relationship": "PART_OF",
                    "confidence": 0.85
                }
            ],
            "associative_relations": [
                {
                    "pattern": r"(\w+)\s+(?:and|with|alongside)\s+(\w+)\s+(?:collaborated|worked together|partnered)",
                    "relationship": "COLLABORATED_WITH",
                    "confidence": 0.8
                },
                {
                    "pattern": r"(\w+)(?:,\s*(\w+))*\s+(?:are|were)\s+(?:related to|associated with|connected to)\s+(\w+)",
                    "relationship": "ASSOCIATED_WITH",
                    "confidence": 0.7
                }
            ],
            "hierarchical_relations": [
                {
                    "pattern": r"(\w+)\s+(?:is a|are)\s+(?:type of|kind of|subclass of)\s+(\w+)",
                    "relationship": "IS_A",
                    "confidence": 0.9
                },
                {
                    "pattern": r"(\w+)\s+(?:includes|encompasses|comprises)\s+(\w+)(?:,\s*(\w+))*",
                    "relationship": "INCLUDES",
                    "confidence": 0.85
                }
            ]
        }

    def extract_entities_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using various methods"""
        entities = []

        # Named Entity Recognition using simple patterns
        # In production, would use SpaCy or similar NLP library

        # Capitalized words (potential proper nouns)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        for noun in proper_nouns:
            entities.append({
                "text": noun,
                "type": "PROPER_NOUN",
                "start": text.find(noun),
                "end": text.find(noun) + len(noun),
                "confidence": 0.7
            })

        # Numbers and dates
        numbers = re.findall(r'\b\d{4}\b|\b\d{1,3}(?:,\d{3})*\b', text)
        for number in numbers:
            entities.append({
                "text": number,
                "type": "NUMBER",
                "start": text.find(number),
                "end": text.find(number) + len(number),
                "confidence": 0.9
            })

        # Technical terms (words with specific patterns)
        technical_terms = re.findall(r'\b[a-z]+(?:-[a-z]+)*(?:ing|tion|ism|ology|graphy)\b', text.lower())
        for term in technical_terms:
            entities.append({
                "text": term,
                "type": "TECHNICAL_TERM",
                "start": text.lower().find(term),
                "end": text.lower().find(term) + len(term),
                "confidence": 0.6
            })

        return entities

    def extract_hyperedges(self, text: str, document_id: str) -> List[Hyperedge]:
        """Extract hyperedges from text using patterns"""
        hyperedges = []
        hyperedge_counter = 0

        # Extract entities first
        entities = self.extract_entities_from_text(text)
        entity_texts = [e["text"] for e in entities]

        # Apply each pattern category
        for category, patterns in self.patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info["pattern"]
                relationship = pattern_info["relationship"]
                confidence = pattern_info["confidence"]

                # Find all matches
                matches = re.finditer(pattern, text, re.IGNORECASE)

                for match in matches:
                    # Extract entities from match groups
                    groups = [g for g in match.groups() if g is not None]

                    if len(groups) >= 2:
                        # Create hyperedge
                        hyperedge_id = f"{document_id}_hyperedge_{hyperedge_counter}"
                        hyperedge_counter += 1

                        # Get context around the match
                        start = max(0, match.start() - 50)
                        end = min(len(text), match.end() + 50)
                        context = text[start:end].strip()

                        hyperedge = Hyperedge(
                            hyperedge_id=hyperedge_id,
                            entities=groups,
                            relationship_type=relationship,
                            confidence=confidence,
                            context=context,
                            source_document=document_id,
                            extraction_method="pattern_based",
                            properties={
                                "pattern_category": category,
                                "match_position": match.start(),
                                "match_length": match.end() - match.start()
                            }
                        )

                        hyperedges.append(hyperedge)

        return hyperedges

class CooccurrenceExtractor:
    """Extracts hyperedges based on entity co-occurrence"""

    def __init__(self, window_size: int = 100, min_cooccurrence: int = 2):
        self.window_size = window_size
        self.min_cooccurrence = min_cooccurrence
        self.cooccurrence_matrix = defaultdict(lambda: defaultdict(int))

    def extract_cooccurrences(self, text: str, entities: List[Dict[str, Any]]) -> List[Tuple[str, str, float]]:
        """Extract entity co-occurrences within sliding windows"""
        cooccurrences = []

        # Sort entities by position
        entities_sorted = sorted(entities, key=lambda e: e["start"])

        # Find co-occurrences within windows
        for i, entity1 in enumerate(entities_sorted):
            for j, entity2 in enumerate(entities_sorted[i+1:], i+1):
                # Check if entities are within window
                distance = entity2["start"] - entity1["end"]

                if distance <= self.window_size:
                    # Calculate co-occurrence strength
                    strength = 1.0 / (1.0 + distance / self.window_size)

                    cooccurrences.append((entity1["text"], entity2["text"], strength))

                    # Update global co-occurrence matrix
                    self.cooccurrence_matrix[entity1["text"]][entity2["text"]] += 1
                    self.cooccurrence_matrix[entity2["text"]][entity1["text"]] += 1

        return cooccurrences

    def extract_hyperedges(self, text: str, document_id: str, entities: List[Dict[str, Any]]) -> List[Hyperedge]:
        """Extract hyperedges based on co-occurrence patterns"""
        hyperedges = []

        # Get co-occurrences for this document
        cooccurrences = self.extract_cooccurrences(text, entities)

        # Group co-occurrences into potential hyperedges
        entity_groups = defaultdict(set)

        for entity1, entity2, strength in cooccurrences:
            if strength > 0.5:  # Threshold for strong co-occurrence
                entity_groups[entity1].add(entity2)
                entity_groups[entity2].add(entity1)

        # Create hyperedges for entity groups
        processed_groups = set()
        hyperedge_counter = 0

        for entity, related_entities in entity_groups.items():
            if len(related_entities) >= 1:  # At least one relationship
                # Create entity group signature for deduplication
                group_signature = tuple(sorted([entity] + list(related_entities)))

                if group_signature not in processed_groups:
                    processed_groups.add(group_signature)

                    # Calculate average co-occurrence strength
                    strengths = [strength for e1, e2, strength in cooccurrences
                               if (e1 == entity and e2 in related_entities) or
                                  (e2 == entity and e1 in related_entities)]
                    avg_strength = sum(strengths) / len(strengths) if strengths else 0.5

                    # Create hyperedge
                    hyperedge_id = f"{document_id}_cooccur_{hyperedge_counter}"
                    hyperedge_counter += 1

                    hyperedge = Hyperedge(
                        hyperedge_id=hyperedge_id,
                        entities=list(group_signature),
                        relationship_type="CO_OCCURS",
                        confidence=min(avg_strength, 0.9),
                        context=f"Entities co-occur in document {document_id}",
                        source_document=document_id,
                        extraction_method="cooccurrence",
                        properties={
                            "avg_cooccurrence_strength": avg_strength,
                            "window_size": self.window_size,
                            "num_cooccurrences": len(strengths)
                        }
                    )

                    hyperedges.append(hyperedge)

        return hyperedges

class SemanticExtractor:
    """Extracts hyperedges using semantic similarity"""

    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold

        # Load pre-trained embeddings (simplified - would use actual embeddings in production)
        self.embeddings = {}

    def get_entity_embedding(self, entity: str) -> np.ndarray:
        """Get embedding for entity (mock implementation)"""
        if entity not in self.embeddings:
            # Generate random embedding (in production, use real embeddings)
            self.embeddings[entity] = np.random.rand(300)
        return self.embeddings[entity]

    def calculate_semantic_similarity(self, entity1: str, entity2: str) -> float:
        """Calculate semantic similarity between entities"""
        emb1 = self.get_entity_embedding(entity1)
        emb2 = self.get_entity_embedding(entity2)

        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norms = np.linalg.norm(emb1) * np.linalg.norm(emb2)

        if norms == 0:
            return 0.0

        return dot_product / norms

    def extract_hyperedges(self, text: str, document_id: str, entities: List[Dict[str, Any]]) -> List[Hyperedge]:
        """Extract hyperedges based on semantic similarity"""
        hyperedges = []

        if len(entities) < 2:
            return hyperedges

        # Calculate semantic similarities between all entity pairs
        semantic_groups = []

        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], i+1):
                similarity = self.calculate_semantic_similarity(entity1["text"], entity2["text"])

                if similarity >= self.similarity_threshold:
                    # Find if either entity is already in a group
                    group_found = False
                    for group in semantic_groups:
                        if entity1["text"] in group or entity2["text"] in group:
                            group.add(entity1["text"])
                            group.add(entity2["text"])
                            group_found = True
                            break

                    if not group_found:
                        semantic_groups.append({entity1["text"], entity2["text"]})

        # Create hyperedges for semantic groups
        for group_idx, group in enumerate(semantic_groups):
            if len(group) >= 2:
                hyperedge_id = f"{document_id}_semantic_{group_idx}"

                # Calculate average similarity within group
                similarities = []
                group_list = list(group)
                for i, entity1 in enumerate(group_list):
                    for entity2 in group_list[i+1:]:
                        sim = self.calculate_semantic_similarity(entity1, entity2)
                        similarities.append(sim)

                avg_similarity = sum(similarities) / len(similarities) if similarities else 0.7

                hyperedge = Hyperedge(
                    hyperedge_id=hyperedge_id,
                    entities=group_list,
                    relationship_type="SEMANTICALLY_SIMILAR",
                    confidence=min(avg_similarity, 0.95),
                    context=f"Semantically similar entities in document {document_id}",
                    source_document=document_id,
                    extraction_method="semantic_similarity",
                    properties={
                        "avg_similarity": avg_similarity,
                        "similarity_threshold": self.similarity_threshold,
                        "group_size": len(group)
                    }
                )

                hyperedges.append(hyperedge)

        return hyperedges

class HyperedgeExtractor:
    """Main hyperedge extraction system"""

    def __init__(self, output_dir: Path = Path("./hyperedges")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize extractors
        self.pattern_extractor = PatternBasedExtractor()
        self.cooccurrence_extractor = CooccurrenceExtractor()
        self.semantic_extractor = SemanticExtractor()

        # Metrics
        self.metrics = ExtractionMetrics(0, 0, {}, {}, 0.0, 0.0, 0.0)

    def load_documents(self, documents_path: Path) -> List[Dict[str, Any]]:
        """Load documents from various formats"""
        documents = []

        if documents_path.is_file():
            # Single file
            if documents_path.suffix == '.json':
                with open(documents_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        documents = data
                    else:
                        documents = [data]
            elif documents_path.suffix == '.jsonl':
                with open(documents_path, 'r') as f:
                    for line in f:
                        documents.append(json.loads(line))
            else:
                # Plain text file
                with open(documents_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents = [{
                        "id": documents_path.stem,
                        "content": content,
                        "source": str(documents_path)
                    }]

        elif documents_path.is_dir():
            # Directory of files
            for file_path in documents_path.rglob("*.txt"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append({
                        "id": file_path.stem,
                        "content": content,
                        "source": str(file_path)
                    })

        logger.info(f"Loaded {len(documents)} documents")
        return documents

    def extract_from_document(self, document: Dict[str, Any]) -> List[Hyperedge]:
        """Extract all hyperedges from a single document"""
        doc_id = document.get("id", "unknown")
        content = document.get("content", "")

        if not content.strip():
            return []

        all_hyperedges = []

        # Extract entities first (shared across extractors)
        entities = self.pattern_extractor.extract_entities_from_text(content)

        # Pattern-based extraction
        pattern_hyperedges = self.pattern_extractor.extract_hyperedges(content, doc_id)
        all_hyperedges.extend(pattern_hyperedges)

        # Co-occurrence extraction
        cooccurrence_hyperedges = self.cooccurrence_extractor.extract_hyperedges(content, doc_id, entities)
        all_hyperedges.extend(cooccurrence_hyperedges)

        # Semantic similarity extraction
        semantic_hyperedges = self.semantic_extractor.extract_hyperedges(content, doc_id, entities)
        all_hyperedges.extend(semantic_hyperedges)

        return all_hyperedges

    def extract_from_corpus(self, documents_path: Path) -> List[Hyperedge]:
        """Extract hyperedges from entire document corpus"""
        start_time = datetime.now()

        # Load documents
        documents = self.load_documents(documents_path)
        self.metrics.total_documents_processed = len(documents)

        all_hyperedges = []

        # Process each document
        for i, document in enumerate(documents):
            logger.info(f"Processing document {i+1}/{len(documents)}: {document.get('id', 'unknown')}")

            try:
                document_hyperedges = self.extract_from_document(document)
                all_hyperedges.extend(document_hyperedges)

                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i+1} documents, extracted {len(all_hyperedges)} hyperedges")

            except Exception as e:
                logger.error(f"Error processing document {document.get('id')}: {e}")
                continue

        # Calculate metrics
        self.metrics.total_hyperedges_extracted = len(all_hyperedges)
        self.metrics.extraction_time = (datetime.now() - start_time).total_seconds()

        # Analyze hyperedges
        self._analyze_hyperedges(all_hyperedges)

        logger.info(f"Extraction completed: {len(all_hyperedges)} hyperedges from {len(documents)} documents")
        return all_hyperedges

    def _analyze_hyperedges(self, hyperedges: List[Hyperedge]):
        """Analyze extracted hyperedges for metrics"""
        if not hyperedges:
            return

        # Count by type
        type_counts = Counter(h.relationship_type for h in hyperedges)
        self.metrics.hyperedges_by_type = dict(type_counts)

        # Count by method
        method_counts = Counter(h.extraction_method for h in hyperedges)
        self.metrics.hyperedges_by_method = dict(method_counts)

        # Average entities per hyperedge
        total_entities = sum(len(h.entities) for h in hyperedges)
        self.metrics.avg_entities_per_hyperedge = total_entities / len(hyperedges)

        # Average confidence
        self.metrics.avg_confidence = sum(h.confidence for h in hyperedges) / len(hyperedges)

    def save_hyperedges(self, hyperedges: List[Hyperedge], output_path: Path):
        """Save hyperedges to file"""
        # Convert to serializable format
        hyperedges_data = [asdict(h) for h in hyperedges]

        with open(output_path, 'w') as f:
            json.dump(hyperedges_data, f, indent=2, default=str)

        logger.info(f"Saved {len(hyperedges)} hyperedges to {output_path}")

    def save_metrics_report(self, output_path: Path):
        """Save extraction metrics report"""
        report = {
            "extraction_metadata": {
                "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
                "extraction_time_seconds": self.metrics.extraction_time,
                "documents_per_second": self.metrics.total_documents_processed / self.metrics.extraction_time if self.metrics.extraction_time > 0 else 0
            },
            "extraction_metrics": asdict(self.metrics),
            "quality_analysis": {
                "high_confidence_hyperedges": sum(1 for method in self.metrics.hyperedges_by_method.keys() if "pattern" in method),
                "medium_confidence_hyperedges": self.metrics.hyperedges_by_method.get("cooccurrence", 0),
                "exploratory_hyperedges": self.metrics.hyperedges_by_method.get("semantic_similarity", 0)
            }
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Metrics report saved to {output_path}")

    def export_for_hyperag(self, hyperedges: List[Hyperedge], kg_path: Path):
        """Export hyperedges directly to HypeRAG knowledge graph"""
        kg = HypergraphKG()

        # Create hyperedge entities and relationships
        for hyperedge in hyperedges:
            # Create hyperedge node
            kg.add_node(hyperedge.hyperedge_id, {
                "type": "hyperedge",
                "relationship_type": hyperedge.relationship_type,
                "confidence": hyperedge.confidence,
                "context": hyperedge.context,
                "source_document": hyperedge.source_document,
                "extraction_method": hyperedge.extraction_method,
                "properties": hyperedge.properties,
                "created_at": datetime.now(timezone.utc).isoformat()
            })

            # Create entity nodes and connect to hyperedge
            for entity in hyperedge.entities:
                entity_id = f"entity_{entity.replace(' ', '_').lower()}"

                # Add entity node if it doesn't exist
                if not hasattr(kg, 'has_node') or not kg.has_node(entity_id):
                    kg.add_node(entity_id, {
                        "type": "entity",
                        "name": entity,
                        "confidence": 0.8
                    })

                # Connect entity to hyperedge
                kg.add_edge(entity_id, hyperedge.hyperedge_id, "PARTICIPATES_IN", {
                    "confidence": hyperedge.confidence,
                    "role": "participant"
                })

        # Save knowledge graph
        kg.save(kg_path)
        logger.info(f"Exported {len(hyperedges)} hyperedges to HypeRAG knowledge graph: {kg_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract hyperedges from document corpus")
    parser.add_argument("--input", required=True, help="Path to document(s) - file or directory")
    parser.add_argument("--output-dir", default="./hyperedges", help="Output directory for results")
    parser.add_argument("--export-kg", help="Export directly to HypeRAG knowledge graph")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        # Create extractor
        output_dir = Path(args.output_dir)
        extractor = HyperedgeExtractor(output_dir)

        # Extract hyperedges
        logger.info("Starting hyperedge extraction...")
        hyperedges = extractor.extract_from_corpus(Path(args.input))

        if not hyperedges:
            logger.warning("No hyperedges extracted!")
            return

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save hyperedges
        hyperedges_file = output_dir / f"hyperedges_{timestamp}.json"
        extractor.save_hyperedges(hyperedges, hyperedges_file)

        # Save metrics
        metrics_file = output_dir / f"extraction_metrics_{timestamp}.json"
        extractor.save_metrics_report(metrics_file)

        # Export to knowledge graph if requested
        if args.export_kg:
            extractor.export_for_hyperag(hyperedges, Path(args.export_kg))

        # Print summary
        print(f"\nExtraction Summary:")
        print(f"  Documents processed: {extractor.metrics.total_documents_processed}")
        print(f"  Hyperedges extracted: {extractor.metrics.total_hyperedges_extracted}")
        print(f"  Average entities per hyperedge: {extractor.metrics.avg_entities_per_hyperedge:.1f}")
        print(f"  Average confidence: {extractor.metrics.avg_confidence:.3f}")
        print(f"  Extraction time: {extractor.metrics.extraction_time:.2f} seconds")

        print(f"\nHyperedges by type:")
        for rel_type, count in extractor.metrics.hyperedges_by_type.items():
            print(f"  - {rel_type}: {count}")

        print(f"\nHyperedges by method:")
        for method, count in extractor.metrics.hyperedges_by_method.items():
            print(f"  - {method}: {count}")

    except Exception as e:
        logger.error(f"Hyperedge extraction failed: {e}")
        raise

if __name__ == "__main__":
    main()
