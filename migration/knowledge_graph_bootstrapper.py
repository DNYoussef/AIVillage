#!/usr/bin/env python3
"""Knowledge Graph Bootstrapper

Constructs initial HypeRAG knowledge graph from multiple data sources.
Handles entity resolution, relationship extraction, and confidence initialization.
"""

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path

# Import HypeRAG components and migration tools
import sys
from typing import Any

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from mcp_servers.hyperag.memory.hypergraph_kg import HypergraphKG
from migration.hyperedge_extractor import Hyperedge
from migration.vector_converter import EntityExtractor

logger = logging.getLogger(__name__)


@dataclass
class BootstrapConfig:
    """Configuration for knowledge graph bootstrapping"""

    entity_similarity_threshold: float = 0.8
    relationship_confidence_threshold: float = 0.5
    max_entities_per_document: int = 50
    enable_entity_linking: bool = True
    confidence_decay_factor: float = 0.95
    temporal_weight: float = 0.1


@dataclass
class BootstrapMetrics:
    """Metrics for bootstrapping process"""

    total_sources: int
    entities_extracted: int
    entities_after_resolution: int
    relationships_extracted: int
    confidence_scores_initialized: int
    duplicate_entities_merged: int
    bootstrap_time: float


class EntityResolver:
    """Resolves entity references across documents"""

    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
        self.entity_signatures = {}
        self.entity_clusters = defaultdict(list)

    def _calculate_entity_signature(self, entity_text: str) -> str:
        """Calculate signature for entity deduplication"""
        # Normalize text
        normalized = entity_text.lower().strip()

        # Remove common noise words
        noise_words = {"the", "a", "an", "of", "in", "on", "at", "to", "for", "by"}
        words = [w for w in normalized.split() if w not in noise_words]

        # Create signature
        signature = " ".join(sorted(words))
        return signature

    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using Jaccard similarity"""
        set1 = set(str1.lower().split())
        set2 = set(str2.lower().split())

        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        return intersection / union if union > 0 else 0.0

    def resolve_entities(
        self, entities: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Resolve entity references and merge duplicates"""
        resolved_entities = {}
        entity_id_counter = 0

        # Group entities by similarity
        for entity in entities:
            entity_text = entity.get("text", "").strip()
            if not entity_text:
                continue

            signature = self._calculate_entity_signature(entity_text)

            # Find similar existing entities
            matched_cluster = None
            for existing_signature, cluster_entities in self.entity_clusters.items():
                similarity = self._calculate_string_similarity(
                    signature, existing_signature
                )

                if similarity >= self.similarity_threshold:
                    matched_cluster = existing_signature
                    break

            if matched_cluster:
                # Add to existing cluster
                self.entity_clusters[matched_cluster].append(entity)
            else:
                # Create new cluster
                self.entity_clusters[signature] = [entity]

        # Create resolved entities from clusters
        for signature, cluster_entities in self.entity_clusters.items():
            entity_id = f"entity_{entity_id_counter:06d}"
            entity_id_counter += 1

            # Merge entity information
            primary_entity = cluster_entities[0]

            # Collect all names/aliases
            names = set()
            types = set()
            sources = set()
            confidences = []

            for entity in cluster_entities:
                names.add(entity.get("text", ""))
                if entity.get("type"):
                    types.add(entity["type"])
                if entity.get("source"):
                    sources.add(entity["source"])
                if entity.get("confidence"):
                    confidences.append(entity["confidence"])

            # Calculate merged confidence
            merged_confidence = np.mean(confidences) if confidences else 0.5

            # Create resolved entity
            resolved_entities[entity_id] = {
                "id": entity_id,
                "primary_name": primary_entity.get("text", ""),
                "aliases": list(names),
                "types": list(types),
                "sources": list(sources),
                "confidence": merged_confidence,
                "cluster_size": len(cluster_entities),
                "signature": signature,
                "resolved_at": datetime.now(timezone.utc).isoformat(),
            }

        logger.info(
            f"Resolved {len(entities)} entities into {len(resolved_entities)} unique entities"
        )
        return resolved_entities


class RelationshipExtractor:
    """Extracts relationships from various data sources"""

    def __init__(self):
        self.relationship_patterns = self._load_relationship_patterns()

    def _load_relationship_patterns(self) -> dict[str, list[str]]:
        """Load patterns for relationship extraction"""
        return {
            "hierarchical": [
                "is a",
                "is an",
                "are a",
                "are an",
                "type of",
                "kind of",
                "subclass of",
                "instance of",
                "example of",
                "category of",
            ],
            "compositional": [
                "contains",
                "includes",
                "comprises",
                "consists of",
                "made of",
                "part of",
                "component of",
                "member of",
                "belongs to",
            ],
            "temporal": [
                "before",
                "after",
                "during",
                "while",
                "when",
                "since",
                "until",
                "preceded by",
                "followed by",
                "concurrent with",
            ],
            "causal": [
                "causes",
                "caused by",
                "leads to",
                "results in",
                "due to",
                "because of",
                "triggers",
                "influences",
                "affects",
            ],
            "associative": [
                "related to",
                "associated with",
                "connected to",
                "linked to",
                "similar to",
                "compared to",
                "contrasted with",
            ],
        }

    def extract_from_text(
        self, text: str, entities: dict[str, dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Extract relationships from text using entities"""
        relationships = []

        # Get entity mentions in text
        entity_mentions = []
        for entity_id, entity_data in entities.items():
            for name in entity_data["aliases"]:
                if name.lower() in text.lower():
                    start_pos = text.lower().find(name.lower())
                    entity_mentions.append(
                        {
                            "entity_id": entity_id,
                            "name": name,
                            "start": start_pos,
                            "end": start_pos + len(name),
                        }
                    )

        # Sort by position
        entity_mentions.sort(key=lambda x: x["start"])

        # Extract relationships between nearby entities
        for i, mention1 in enumerate(entity_mentions):
            for mention2 in entity_mentions[i + 1 :]:
                # Check if entities are close enough
                distance = mention2["start"] - mention1["end"]
                if distance > 200:  # Too far apart
                    break

                # Extract text between entities
                between_text = text[mention1["end"] : mention2["start"]].lower().strip()

                # Classify relationship
                relationship_type = self._classify_relationship(between_text)

                if relationship_type:
                    confidence = self._calculate_relationship_confidence(
                        between_text, relationship_type
                    )

                    relationships.append(
                        {
                            "source": mention1["entity_id"],
                            "target": mention2["entity_id"],
                            "type": relationship_type,
                            "confidence": confidence,
                            "context": text[mention1["start"] : mention2["end"]],
                            "extracted_from": "text_analysis",
                        }
                    )

        return relationships

    def _classify_relationship(self, text: str) -> str | None:
        """Classify the relationship type based on text"""
        text_lower = text.lower()

        for rel_type, patterns in self.relationship_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return rel_type.upper()

        # Default relationship if no pattern matches
        if len(text) < 50 and any(word in text_lower for word in ["and", "with", "&"]):
            return "ASSOCIATED"

        return None

    def _calculate_relationship_confidence(self, text: str, rel_type: str) -> float:
        """Calculate confidence score for relationship"""
        base_confidence = 0.6

        # Boost confidence for clear patterns
        patterns = self.relationship_patterns.get(rel_type.lower(), [])
        for pattern in patterns:
            if pattern in text.lower():
                base_confidence += 0.2
                break

        # Reduce confidence for long, complex text
        if len(text) > 100:
            base_confidence *= 0.8

        # Boost confidence for short, clear connections
        if 5 < len(text) < 30:
            base_confidence *= 1.2

        return min(max(base_confidence, 0.1), 0.95)

    def extract_from_hyperedges(
        self, hyperedges: list[Hyperedge]
    ) -> list[dict[str, Any]]:
        """Extract relationships from hyperedge data"""
        relationships = []

        for hyperedge in hyperedges:
            if len(hyperedge.entities) >= 2:
                # Create pairwise relationships from hyperedge
                for i, entity1 in enumerate(hyperedge.entities):
                    for entity2 in hyperedge.entities[i + 1 :]:
                        relationships.append(
                            {
                                "source": f"entity_{entity1.replace(' ', '_').lower()}",
                                "target": f"entity_{entity2.replace(' ', '_').lower()}",
                                "type": hyperedge.relationship_type,
                                "confidence": hyperedge.confidence,
                                "context": hyperedge.context,
                                "extracted_from": f"hyperedge_{hyperedge.hyperedge_id}",
                            }
                        )

        return relationships


class ConfidenceBootstrapper:
    """Initializes confidence scores for entities and relationships"""

    def __init__(self, config: BootstrapConfig):
        self.config = config
        self.source_reliability = {
            "academic_paper": 0.9,
            "news_article": 0.8,
            "encyclopedia": 0.95,
            "social_media": 0.6,
            "blog": 0.7,
            "government": 0.9,
            "corporate": 0.8,
            "unknown": 0.5,
        }

    def calculate_source_confidence(self, source: str) -> float:
        """Calculate confidence based on source reliability"""
        source_lower = source.lower()

        for source_type, reliability in self.source_reliability.items():
            if source_type in source_lower:
                return reliability

        return self.source_reliability["unknown"]

    def calculate_frequency_confidence(
        self, entity_id: str, entity_data: dict[str, Any]
    ) -> float:
        """Calculate confidence based on entity frequency"""
        cluster_size = entity_data.get("cluster_size", 1)
        sources_count = len(entity_data.get("sources", []))

        # Frequency boost
        frequency_factor = min(1.0, cluster_size / 10.0)  # Normalize to max of 1.0

        # Source diversity boost
        diversity_factor = min(1.0, sources_count / 5.0)  # Normalize to max of 1.0

        return 0.5 + (frequency_factor * 0.3) + (diversity_factor * 0.2)

    def calculate_temporal_confidence(self, timestamp: str | None) -> float:
        """Calculate confidence based on temporal factors"""
        if not timestamp:
            return 1.0

        try:
            entity_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            current_time = datetime.now(timezone.utc)
            age_days = (current_time - entity_time).days

            # Decay confidence for older entities
            decay_factor = self.config.confidence_decay_factor ** (age_days / 365)
            return max(0.3, decay_factor)

        except:
            return 1.0

    def bootstrap_entity_confidence(
        self, entities: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Initialize confidence scores for entities"""
        for entity_id, entity_data in entities.items():
            # Base confidence from entity resolution
            base_confidence = entity_data.get("confidence", 0.5)

            # Source-based confidence
            sources = entity_data.get("sources", [])
            source_confidences = [self.calculate_source_confidence(s) for s in sources]
            avg_source_confidence = (
                np.mean(source_confidences) if source_confidences else 0.5
            )

            # Frequency-based confidence
            frequency_confidence = self.calculate_frequency_confidence(
                entity_id, entity_data
            )

            # Temporal confidence
            temporal_confidence = self.calculate_temporal_confidence(
                entity_data.get("resolved_at")
            )

            # Combined confidence
            final_confidence = (
                base_confidence * 0.4
                + avg_source_confidence * 0.3
                + frequency_confidence * 0.2
                + temporal_confidence * 0.1
            )

            entity_data["confidence"] = min(max(final_confidence, 0.1), 0.99)
            entity_data["confidence_components"] = {
                "base": base_confidence,
                "source": avg_source_confidence,
                "frequency": frequency_confidence,
                "temporal": temporal_confidence,
            }

        return entities

    def bootstrap_relationship_confidence(
        self, relationships: list[dict[str, Any]], entities: dict[str, dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Initialize confidence scores for relationships"""
        for relationship in relationships:
            # Base confidence from extraction
            base_confidence = relationship.get("confidence", 0.5)

            # Entity confidence influence
            source_entity = entities.get(relationship["source"])
            target_entity = entities.get(relationship["target"])

            entity_confidences = []
            if source_entity:
                entity_confidences.append(source_entity["confidence"])
            if target_entity:
                entity_confidences.append(target_entity["confidence"])

            avg_entity_confidence = (
                np.mean(entity_confidences) if entity_confidences else 0.5
            )

            # Extraction method confidence
            extraction_method = relationship.get("extracted_from", "")
            method_confidence = 0.8 if "hyperedge" in extraction_method else 0.6

            # Context quality confidence
            context = relationship.get("context", "")
            context_confidence = min(1.0, len(context) / 100.0) if context else 0.5

            # Combined confidence
            final_confidence = (
                base_confidence * 0.5
                + avg_entity_confidence * 0.2
                + method_confidence * 0.2
                + context_confidence * 0.1
            )

            relationship["confidence"] = min(max(final_confidence, 0.1), 0.95)
            relationship["confidence_components"] = {
                "base": base_confidence,
                "entities": avg_entity_confidence,
                "method": method_confidence,
                "context": context_confidence,
            }

        return relationships


class KnowledgeGraphBootstrapper:
    """Main bootstrapping system"""

    def __init__(self, config: BootstrapConfig = None):
        self.config = config or BootstrapConfig()
        self.entity_resolver = EntityResolver(self.config.entity_similarity_threshold)
        self.relationship_extractor = RelationshipExtractor()
        self.confidence_bootstrapper = ConfidenceBootstrapper(self.config)
        self.metrics = BootstrapMetrics(0, 0, 0, 0, 0, 0, 0.0)

    def load_data_sources(
        self, source_configs: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[Hyperedge]]:
        """Load data from multiple sources"""
        all_entities = []
        all_hyperedges = []

        self.metrics.total_sources = len(source_configs)

        for source_config in source_configs:
            source_type = source_config["type"]
            source_path = Path(source_config["path"])

            logger.info(f"Loading data from {source_type}: {source_path}")

            if source_type == "documents":
                # Load document entities
                entity_extractor = EntityExtractor()

                if source_path.suffix == ".json":
                    with open(source_path) as f:
                        documents = json.load(f)
                else:
                    documents = [
                        {"content": source_path.read_text(), "source": str(source_path)}
                    ]

                for doc in documents:
                    doc_entities = entity_extractor.extract_entities(
                        doc.get("content", "")
                    )
                    for entity in doc_entities:
                        entity["source"] = doc.get("source", str(source_path))
                    all_entities.extend(doc_entities)

            elif source_type == "hyperedges":
                # Load hyperedges
                with open(source_path) as f:
                    hyperedge_data = json.load(f)

                for he_data in hyperedge_data:
                    hyperedge = Hyperedge(**he_data)
                    all_hyperedges.append(hyperedge)

            elif source_type == "structured":
                # Load structured data (CSV, JSON, etc.)
                if source_path.suffix == ".json":
                    with open(source_path) as f:
                        data = json.load(f)

                    # Extract entities from structured data
                    for item in data:
                        for key, value in item.items():
                            if isinstance(value, str) and len(value.strip()) > 0:
                                all_entities.append(
                                    {
                                        "text": value,
                                        "type": key.upper(),
                                        "source": str(source_path),
                                        "confidence": 0.8,
                                    }
                                )

        logger.info(
            f"Loaded {len(all_entities)} entities and {len(all_hyperedges)} hyperedges from {len(source_configs)} sources"
        )
        return all_entities, all_hyperedges

    def bootstrap_knowledge_graph(
        self, source_configs: list[dict[str, Any]], output_path: Path
    ) -> BootstrapMetrics:
        """Bootstrap complete knowledge graph from sources"""
        start_time = datetime.now()

        logger.info("Starting knowledge graph bootstrapping...")

        # Step 1: Load data from sources
        all_entities_raw, all_hyperedges = self.load_data_sources(source_configs)
        self.metrics.entities_extracted = len(all_entities_raw)

        # Step 2: Entity resolution
        logger.info("Resolving entity references...")
        resolved_entities = self.entity_resolver.resolve_entities(all_entities_raw)
        self.metrics.entities_after_resolution = len(resolved_entities)
        self.metrics.duplicate_entities_merged = (
            self.metrics.entities_extracted - self.metrics.entities_after_resolution
        )

        # Step 3: Extract relationships
        logger.info("Extracting relationships...")
        relationships = []

        # From hyperedges
        hyperedge_relationships = self.relationship_extractor.extract_from_hyperedges(
            all_hyperedges
        )
        relationships.extend(hyperedge_relationships)

        # From text (if available)
        for source_config in source_configs:
            if source_config["type"] == "documents":
                source_path = Path(source_config["path"])
                if source_path.exists():
                    text_content = (
                        source_path.read_text() if source_path.is_file() else ""
                    )
                    if text_content:
                        text_relationships = (
                            self.relationship_extractor.extract_from_text(
                                text_content, resolved_entities
                            )
                        )
                        relationships.extend(text_relationships)

        self.metrics.relationships_extracted = len(relationships)

        # Step 4: Bootstrap confidence scores
        logger.info("Initializing confidence scores...")
        resolved_entities = self.confidence_bootstrapper.bootstrap_entity_confidence(
            resolved_entities
        )
        relationships = self.confidence_bootstrapper.bootstrap_relationship_confidence(
            relationships, resolved_entities
        )
        self.metrics.confidence_scores_initialized = len(resolved_entities) + len(
            relationships
        )

        # Step 5: Create knowledge graph
        logger.info("Building knowledge graph...")
        kg = HypergraphKG()

        # Add entities
        for entity_id, entity_data in resolved_entities.items():
            kg.add_node(
                entity_id,
                {
                    "type": "entity",
                    "name": entity_data["primary_name"],
                    "aliases": entity_data["aliases"],
                    "entity_types": entity_data["types"],
                    "confidence": entity_data["confidence"],
                    "sources": entity_data["sources"],
                    "cluster_size": entity_data["cluster_size"],
                    "bootstrapped_at": datetime.now(timezone.utc).isoformat(),
                    "confidence_components": entity_data.get(
                        "confidence_components", {}
                    ),
                },
            )

        # Add relationships
        for relationship in relationships:
            # Filter out relationships below confidence threshold
            if (
                relationship["confidence"]
                >= self.config.relationship_confidence_threshold
            ):
                kg.add_edge(
                    relationship["source"],
                    relationship["target"],
                    relationship["type"],
                    {
                        "confidence": relationship["confidence"],
                        "context": relationship.get("context", ""),
                        "extracted_from": relationship.get("extracted_from", ""),
                        "confidence_components": relationship.get(
                            "confidence_components", {}
                        ),
                    },
                )

        # Save knowledge graph
        kg.save(output_path)

        # Calculate final metrics
        self.metrics.bootstrap_time = (datetime.now() - start_time).total_seconds()

        logger.info("Knowledge graph bootstrapping completed!")
        logger.info(
            f"Created knowledge graph with {len(resolved_entities)} entities and {len(relationships)} relationships"
        )

        return self.metrics

    def save_bootstrap_report(self, output_path: Path):
        """Save detailed bootstrapping report"""
        report = {
            "bootstrap_metadata": {
                "bootstrap_timestamp": datetime.now(timezone.utc).isoformat(),
                "config": asdict(self.config),
                "bootstrap_time_seconds": self.metrics.bootstrap_time,
            },
            "metrics": asdict(self.metrics),
            "entity_resolution_stats": {
                "original_entities": self.metrics.entities_extracted,
                "resolved_entities": self.metrics.entities_after_resolution,
                "merge_ratio": self.metrics.duplicate_entities_merged
                / self.metrics.entities_extracted
                if self.metrics.entities_extracted > 0
                else 0,
                "entities_per_cluster": self.metrics.entities_extracted
                / self.metrics.entities_after_resolution
                if self.metrics.entities_after_resolution > 0
                else 1,
            },
            "knowledge_graph_stats": {
                "entity_count": self.metrics.entities_after_resolution,
                "relationship_count": self.metrics.relationships_extracted,
                "graph_density": (2 * self.metrics.relationships_extracted)
                / (
                    self.metrics.entities_after_resolution
                    * (self.metrics.entities_after_resolution - 1)
                )
                if self.metrics.entities_after_resolution > 1
                else 0,
            },
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Bootstrap report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap HypeRAG knowledge graph from multiple sources"
    )
    parser.add_argument(
        "--config", required=True, help="Configuration file with data sources"
    )
    parser.add_argument(
        "--output", required=True, help="Output path for knowledge graph"
    )
    parser.add_argument("--report", help="Path to save bootstrap report")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        # Load configuration
        with open(args.config) as f:
            config_data = json.load(f)

        source_configs = config_data["data_sources"]
        bootstrap_config = BootstrapConfig(**config_data.get("bootstrap_config", {}))

        # Create bootstrapper
        bootstrapper = KnowledgeGraphBootstrapper(bootstrap_config)

        # Bootstrap knowledge graph
        metrics = bootstrapper.bootstrap_knowledge_graph(
            source_configs, Path(args.output)
        )

        # Save report if requested
        if args.report:
            bootstrapper.save_bootstrap_report(Path(args.report))

        # Print summary
        print("\nBootstrap Summary:")
        print(f"  Data sources processed: {metrics.total_sources}")
        print(f"  Entities extracted: {metrics.entities_extracted}")
        print(f"  Entities after resolution: {metrics.entities_after_resolution}")
        print(f"  Duplicate entities merged: {metrics.duplicate_entities_merged}")
        print(f"  Relationships extracted: {metrics.relationships_extracted}")
        print(
            f"  Confidence scores initialized: {metrics.confidence_scores_initialized}"
        )
        print(f"  Bootstrap time: {metrics.bootstrap_time:.2f} seconds")

        reduction_ratio = (
            metrics.duplicate_entities_merged / metrics.entities_extracted
            if metrics.entities_extracted > 0
            else 0
        )
        print(f"  Entity reduction ratio: {reduction_ratio:.1%}")

    except Exception as e:
        logger.error(f"Knowledge graph bootstrapping failed: {e}")
        raise


if __name__ == "__main__":
    main()
