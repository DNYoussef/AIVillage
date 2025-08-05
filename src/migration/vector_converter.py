#!/usr/bin/env python3
"""Vector to Hypergraph Converter.

Converts existing vector store embeddings (FAISS/Qdrant) to HypeRAG hypergraph entities.
Preserves semantic information while enabling graph-based reasoning.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import pickle

# Import HypeRAG components
import sys
from typing import Any

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from mcp_servers.hyperag.memory.hypergraph_kg import HypergraphKG

logger = logging.getLogger(__name__)


@dataclass
class VectorDocument:
    """Document with vector embedding and metadata."""

    doc_id: str
    embedding: np.ndarray
    metadata: dict[str, Any]
    content: str
    source: str
    timestamp: str | None = None


@dataclass
class ConversionMetrics:
    """Metrics tracking conversion progress."""

    total_documents: int
    converted_documents: int
    failed_conversions: int
    total_entities_created: int
    total_relationships_created: int
    conversion_time: float


class VectorStoreLoader:
    """Loads vectors from different vector store formats."""

    def __init__(self, vector_store_path: str, store_type: str = "faiss") -> None:
        self.vector_store_path = Path(vector_store_path)
        self.store_type = store_type.lower()

    def load_faiss_store(self) -> list[VectorDocument]:
        """Load documents from FAISS vector store."""
        try:
            import faiss

            # Load FAISS index
            index_path = self.vector_store_path / "index.faiss"
            if not index_path.exists():
                msg = f"FAISS index not found: {index_path}"
                raise FileNotFoundError(msg)

            index = faiss.read_index(str(index_path))

            # Load metadata
            metadata_path = self.vector_store_path / "metadata.json"
            if not metadata_path.exists():
                msg = f"Metadata file not found: {metadata_path}"
                raise FileNotFoundError(msg)

            with open(metadata_path) as f:
                metadata = json.load(f)

            # Load embeddings
            documents = []
            for i in range(index.ntotal):
                # Get embedding vector
                embedding = index.reconstruct(i)

                # Get document metadata
                doc_metadata = metadata.get(str(i), {})
                doc_id = doc_metadata.get("id", f"doc_{i}")

                document = VectorDocument(
                    doc_id=doc_id,
                    embedding=embedding,
                    metadata=doc_metadata,
                    content=doc_metadata.get("content", ""),
                    source=doc_metadata.get("source", "unknown"),
                    timestamp=doc_metadata.get("timestamp"),
                )
                documents.append(document)

            logger.info(f"Loaded {len(documents)} documents from FAISS store")
            return documents

        except ImportError:
            logger.exception("FAISS not installed. Please install with: pip install faiss-cpu")
            raise
        except Exception as e:
            logger.exception(f"Error loading FAISS store: {e}")
            raise

    def load_qdrant_store(self) -> list[VectorDocument]:
        """Load documents from Qdrant vector store."""
        try:
            from qdrant_client import QdrantClient

            # Initialize Qdrant client
            client = QdrantClient(path=str(self.vector_store_path))

            # Get collection info
            collections = client.get_collections().collections
            if not collections:
                msg = "No collections found in Qdrant store"
                raise ValueError(msg)

            collection_name = collections[0].name
            logger.info(f"Loading from Qdrant collection: {collection_name}")

            # Scroll through all points
            documents = []
            offset = None

            while True:
                result = client.scroll(
                    collection_name=collection_name,
                    offset=offset,
                    limit=100,
                    with_payload=True,
                    with_vectors=True,
                )

                points, next_offset = result

                for point in points:
                    document = VectorDocument(
                        doc_id=str(point.id),
                        embedding=np.array(point.vector),
                        metadata=point.payload or {},
                        content=point.payload.get("content", ""),
                        source=point.payload.get("source", "unknown"),
                        timestamp=point.payload.get("timestamp"),
                    )
                    documents.append(document)

                if next_offset is None:
                    break
                offset = next_offset

            logger.info(f"Loaded {len(documents)} documents from Qdrant store")
            return documents

        except ImportError:
            logger.exception("Qdrant client not installed. Please install with: pip install qdrant-client")
            raise
        except Exception as e:
            logger.exception(f"Error loading Qdrant store: {e}")
            raise

    def load_custom_store(self) -> list[VectorDocument]:
        """Load documents from custom pickle format."""
        try:
            store_file = self.vector_store_path / "vector_store.pkl"

            with open(store_file, "rb") as f:
                data = pickle.load(f)

            documents = []
            for doc_data in data:
                document = VectorDocument(
                    doc_id=doc_data["id"],
                    embedding=np.array(doc_data["embedding"]),
                    metadata=doc_data.get("metadata", {}),
                    content=doc_data.get("content", ""),
                    source=doc_data.get("source", "unknown"),
                    timestamp=doc_data.get("timestamp"),
                )
                documents.append(document)

            logger.info(f"Loaded {len(documents)} documents from custom store")
            return documents

        except Exception as e:
            logger.exception(f"Error loading custom store: {e}")
            raise

    def load_documents(self) -> list[VectorDocument]:
        """Load documents based on store type."""
        if self.store_type == "faiss":
            return self.load_faiss_store()
        if self.store_type == "qdrant":
            return self.load_qdrant_store()
        if self.store_type == "custom":
            return self.load_custom_store()
        msg = f"Unsupported store type: {self.store_type}"
        raise ValueError(msg)


class EntityExtractor:
    """Extracts entities from document content for knowledge graph."""

    def __init__(self) -> None:
        try:
            import spacy

            self.nlp = spacy.load("en_core_web_sm")
        except ImportError:
            logger.warning("SpaCy not available. Using simple entity extraction.")
            self.nlp = None
        except OSError:
            logger.warning("SpaCy model not found. Using simple entity extraction.")
            self.nlp = None

    def extract_entities(self, text: str) -> list[dict[str, Any]]:
        """Extract named entities from text."""
        entities = []

        if self.nlp:
            # Use SpaCy for entity extraction
            doc = self.nlp(text)

            for ent in doc.ents:
                entities.append(
                    {
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "confidence": 0.8,  # SpaCy doesn't provide confidence scores
                    }
                )
        else:
            # Simple entity extraction using capitalization patterns
            import re

            # Find capitalized words (potential entities)
            capitalized_words = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)

            for _i, word in enumerate(capitalized_words):
                entities.append(
                    {
                        "text": word,
                        "label": "UNKNOWN",
                        "start": text.find(word),
                        "end": text.find(word) + len(word),
                        "confidence": 0.5,
                    }
                )

        return entities

    def extract_relationships(self, text: str, entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Extract relationships between entities."""
        relationships = []

        if len(entities) < 2:
            return relationships

        # Simple relationship extraction based on proximity and patterns
        for i, entity1 in enumerate(entities):
            for _j, entity2 in enumerate(entities[i + 1 :], i + 1):
                # Check if entities are in the same sentence
                if abs(entity1["start"] - entity2["start"]) < 200:  # Within 200 characters
                    # Extract text between entities
                    start = min(entity1["end"], entity2["end"])
                    end = max(entity1["start"], entity2["start"])
                    between_text = text[start:end].lower()

                    # Simple relationship classification
                    relationship_type = "RELATED_TO"  # Default
                    confidence = 0.5

                    if any(word in between_text for word in ["is", "are", "was", "were"]):
                        relationship_type = "IS_A"
                        confidence = 0.7
                    elif any(word in between_text for word in ["has", "have", "contains"]):
                        relationship_type = "HAS"
                        confidence = 0.7
                    elif any(word in between_text for word in ["in", "at", "on", "located"]):
                        relationship_type = "LOCATED_IN"
                        confidence = 0.6
                    elif any(word in between_text for word in ["created", "made", "developed"]):
                        relationship_type = "CREATED_BY"
                        confidence = 0.6

                    relationships.append(
                        {
                            "source": entity1["text"],
                            "target": entity2["text"],
                            "type": relationship_type,
                            "confidence": confidence,
                            "context": between_text.strip(),
                        }
                    )

        return relationships


class VectorToHypergraphConverter:
    """Main converter class."""

    def __init__(
        self,
        vector_store_path: str,
        output_kg_path: str,
        store_type: str = "faiss",
        batch_size: int = 100,
    ) -> None:
        self.vector_store_path = vector_store_path
        self.output_kg_path = output_kg_path
        self.store_type = store_type
        self.batch_size = batch_size

        # Initialize components
        self.vector_loader = VectorStoreLoader(vector_store_path, store_type)
        self.entity_extractor = EntityExtractor()
        self.kg = HypergraphKG()

        # Metrics
        self.metrics = ConversionMetrics(0, 0, 0, 0, 0, 0.0)

    def convert_document(self, document: VectorDocument) -> bool:
        """Convert a single document to knowledge graph entities."""
        try:
            # Create main document entity
            doc_entity_id = f"doc_{document.doc_id}"

            self.kg.add_node(
                doc_entity_id,
                {
                    "type": "document",
                    "title": document.metadata.get("title", f"Document {document.doc_id}"),
                    "source": document.source,
                    "content_preview": (
                        document.content[:200] + "..." if len(document.content) > 200 else document.content
                    ),
                    "embedding": document.embedding.tolist() if document.embedding is not None else None,
                    "original_doc_id": document.doc_id,
                    "migrated_at": datetime.now(timezone.utc).isoformat(),
                    "confidence": 0.9,
                },
            )

            self.metrics.total_entities_created += 1

            # Extract entities from content
            if document.content:
                entities = self.entity_extractor.extract_entities(document.content)

                # Create entity nodes and relationships
                for entity in entities:
                    entity_id = f"entity_{entity['text'].replace(' ', '_').lower()}"

                    # Add entity node if it doesn't exist
                    if not self.kg.has_node(entity_id):
                        self.kg.add_node(
                            entity_id,
                            {
                                "type": "entity",
                                "name": entity["text"],
                                "label": entity["label"],
                                "confidence": entity["confidence"],
                                "created_from": document.doc_id,
                            },
                        )
                        self.metrics.total_entities_created += 1

                    # Create relationship between document and entity
                    self.kg.add_edge(
                        doc_entity_id,
                        entity_id,
                        "CONTAINS",
                        {
                            "position": entity["start"],
                            "confidence": entity["confidence"],
                            "context": document.content[max(0, entity["start"] - 50) : entity["end"] + 50],
                        },
                    )
                    self.metrics.total_relationships_created += 1

                # Extract relationships between entities
                relationships = self.entity_extractor.extract_relationships(document.content, entities)

                for relationship in relationships:
                    source_id = f"entity_{relationship['source'].replace(' ', '_').lower()}"
                    target_id = f"entity_{relationship['target'].replace(' ', '_').lower()}"

                    if self.kg.has_node(source_id) and self.kg.has_node(target_id):
                        self.kg.add_edge(
                            source_id,
                            target_id,
                            relationship["type"],
                            {
                                "confidence": relationship["confidence"],
                                "context": relationship["context"],
                                "extracted_from": document.doc_id,
                            },
                        )
                        self.metrics.total_relationships_created += 1

            # Add metadata entities
            for key, value in document.metadata.items():
                if key in ["author", "category", "domain", "tags"]:
                    if isinstance(value, str) and value.strip():
                        metadata_entity_id = f"metadata_{key}_{value.replace(' ', '_').lower()}"

                        if not self.kg.has_node(metadata_entity_id):
                            self.kg.add_node(
                                metadata_entity_id,
                                {
                                    "type": "metadata",
                                    "category": key,
                                    "value": value,
                                    "confidence": 0.8,
                                },
                            )
                            self.metrics.total_entities_created += 1

                        self.kg.add_edge(
                            doc_entity_id,
                            metadata_entity_id,
                            f"HAS_{key.upper()}",
                            {"confidence": 0.9},
                        )
                        self.metrics.total_relationships_created += 1

            self.metrics.converted_documents += 1
            return True

        except Exception as e:
            logger.exception(f"Error converting document {document.doc_id}: {e}")
            self.metrics.failed_conversions += 1
            return False

    def convert_all_documents(self) -> ConversionMetrics:
        """Convert all documents from vector store to knowledge graph."""
        start_time = datetime.now()

        logger.info("Loading documents from vector store...")
        documents = self.vector_loader.load_documents()
        self.metrics.total_documents = len(documents)

        logger.info(f"Converting {len(documents)} documents to knowledge graph...")

        # Process documents in batches
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i : i + self.batch_size]

            logger.info(f"Processing batch {i // self.batch_size + 1}/{(len(documents) - 1) // self.batch_size + 1}")

            for document in batch:
                self.convert_document(document)

                if self.metrics.converted_documents % 100 == 0:
                    logger.info(f"Converted {self.metrics.converted_documents} documents...")

        # Save knowledge graph
        logger.info("Saving knowledge graph...")
        self.kg.save(self.output_kg_path)

        # Calculate conversion time
        self.metrics.conversion_time = (datetime.now() - start_time).total_seconds()

        # Log final metrics
        logger.info("Conversion completed!")
        logger.info(f"Total documents: {self.metrics.total_documents}")
        logger.info(f"Converted documents: {self.metrics.converted_documents}")
        logger.info(f"Failed conversions: {self.metrics.failed_conversions}")
        logger.info(f"Entities created: {self.metrics.total_entities_created}")
        logger.info(f"Relationships created: {self.metrics.total_relationships_created}")
        logger.info(f"Conversion time: {self.metrics.conversion_time:.2f} seconds")

        return self.metrics

    def save_conversion_report(self, output_path: Path) -> None:
        """Save detailed conversion report."""
        report = {
            "conversion_metadata": {
                "source_vector_store": str(self.vector_store_path),
                "store_type": self.store_type,
                "output_kg_path": str(self.output_kg_path),
                "conversion_timestamp": datetime.now(timezone.utc).isoformat(),
                "batch_size": self.batch_size,
            },
            "metrics": {
                "total_documents": self.metrics.total_documents,
                "converted_documents": self.metrics.converted_documents,
                "failed_conversions": self.metrics.failed_conversions,
                "success_rate": (
                    self.metrics.converted_documents / self.metrics.total_documents
                    if self.metrics.total_documents > 0
                    else 0
                ),
                "total_entities_created": self.metrics.total_entities_created,
                "total_relationships_created": self.metrics.total_relationships_created,
                "conversion_time_seconds": self.metrics.conversion_time,
                "documents_per_second": (
                    self.metrics.converted_documents / self.metrics.conversion_time
                    if self.metrics.conversion_time > 0
                    else 0
                ),
            },
            "knowledge_graph_stats": {
                "total_nodes": len(self.kg.get_all_nodes()) if hasattr(self.kg, "get_all_nodes") else 0,
                "total_edges": len(self.kg.get_all_edges()) if hasattr(self.kg, "get_all_edges") else 0,
                "avg_entities_per_document": (
                    self.metrics.total_entities_created / self.metrics.converted_documents
                    if self.metrics.converted_documents > 0
                    else 0
                ),
                "avg_relationships_per_document": (
                    self.metrics.total_relationships_created / self.metrics.converted_documents
                    if self.metrics.converted_documents > 0
                    else 0
                ),
            },
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Conversion report saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert vector store to HypeRAG knowledge graph")
    parser.add_argument("--input", required=True, help="Path to vector store directory")
    parser.add_argument("--output", required=True, help="Output path for knowledge graph")
    parser.add_argument(
        "--store-type",
        choices=["faiss", "qdrant", "custom"],
        default="faiss",
        help="Type of vector store",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing documents",
    )
    parser.add_argument("--report", help="Path to save conversion report")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        # Create converter
        converter = VectorToHypergraphConverter(
            vector_store_path=args.input,
            output_kg_path=args.output,
            store_type=args.store_type,
            batch_size=args.batch_size,
        )

        # Perform conversion
        metrics = converter.convert_all_documents()

        # Save report if requested
        if args.report:
            converter.save_conversion_report(Path(args.report))

        # Print summary
        print("\nConversion Summary:")
        print(f"  Documents processed: {metrics.converted_documents}/{metrics.total_documents}")
        print(f"  Success rate: {metrics.converted_documents / metrics.total_documents:.1%}")
        print(f"  Entities created: {metrics.total_entities_created}")
        print(f"  Relationships created: {metrics.total_relationships_created}")
        print(f"  Conversion time: {metrics.conversion_time:.2f} seconds")

        if metrics.failed_conversions > 0:
            print(f"  Failed conversions: {metrics.failed_conversions}")

    except Exception as e:
        logger.exception(f"Conversion failed: {e}")
        raise


if __name__ == "__main__":
    main()
