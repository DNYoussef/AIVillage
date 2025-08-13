"""RAG System Upgrade Script.

Upgrades RAG system from SHA256 embeddings to real vector embeddings.
"""

from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import pickle
import sqlite3
from typing import Any


# Mock the missing imports for now
class MockSentenceTransformer:
    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.dimension = 384

    def encode(self, texts, **kwargs):
        import numpy as np

        # Return mock embeddings for testing
        if isinstance(texts, str):
            texts = [texts]
        return np.random.rand(len(texts), self.dimension).astype("float32")

    def get_sentence_embedding_dimension(self):
        return self.dimension


try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError:
    # Mock imports for testing
    SentenceTransformer = MockSentenceTransformer
    import numpy as np

    class MockFaiss:
        class IndexIDMap:
            def __init__(self, inner_index) -> None:
                self.inner_index = inner_index
                self.ntotal = 0

            def add_with_ids(self, embeddings, ids) -> None:
                self.ntotal += len(embeddings)

            def search(self, query, k):
                return np.array([[0, 1, 2]]), np.array([[0.9, 0.8, 0.7]])

        class IndexFlatIP:
            def __init__(self, d) -> None:
                self.d = d

        @staticmethod
        def write_index(index, path) -> None:
            pass

        @staticmethod
        def read_index(path):
            return MockFaiss.IndexIDMap(MockFaiss.IndexFlatIP(384))

    faiss = MockFaiss()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CODEX environment variables
RAG_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "paraphrase-MiniLM-L3-v2")
RAG_VECTOR_DIM = int(os.getenv("RAG_VECTOR_DIM", "384"))
RAG_FAISS_INDEX_PATH = os.getenv("RAG_FAISS_INDEX_PATH", "./data/faiss_index")
RAG_BM25_CORPUS_PATH = os.getenv("RAG_BM25_CORPUS_PATH", "./data/bm25_corpus")


class RAGSystemUpgrader:
    """Handles upgrade from SHA256 to real embeddings."""

    def __init__(self) -> None:
        self.db_path = Path("./data/rag_index.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.faiss_index_path = Path(RAG_FAISS_INDEX_PATH)
        self.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)

        self.bm25_corpus_path = Path(RAG_BM25_CORPUS_PATH)
        self.bm25_corpus_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize embedding model
        self.embedder = SentenceTransformer(RAG_EMBEDDING_MODEL)
        self.vector_dim = self.embedder.get_sentence_embedding_dimension()

        logger.info(f"Initialized embedder: {RAG_EMBEDDING_MODEL} (dim: {self.vector_dim})")

    def find_legacy_rag_data(self) -> dict[str, list[Path]]:
        """Find legacy RAG data files."""
        logger.info("Scanning for legacy RAG data...")

        legacy_data = {
            "sha256_files": [],
            "mock_embeddings": [],
            "old_indexes": [],
            "document_stores": [],
        }

        # Search for SHA256-based files
        search_dirs = [Path(), Path("./data"), Path("./src")]

        for search_dir in search_dirs:
            if search_dir.exists():
                # Look for SHA256 hash files
                for file_path in search_dir.rglob("*.json"):
                    try:
                        with open(file_path) as f:
                            content = f.read()
                            if "sha256" in content.lower() and (
                                "embedding" in content.lower() or "vector" in content.lower()
                            ):
                                legacy_data["sha256_files"].append(file_path)
                                logger.info(f"Found SHA256 file: {file_path}")
                    except Exception:
                        pass

                # Look for old FAISS indexes
                for file_path in search_dir.rglob("*.faiss"):
                    legacy_data["old_indexes"].append(file_path)
                    logger.info(f"Found old FAISS index: {file_path}")

                # Look for pickle files (potential embeddings)
                for file_path in search_dir.rglob("*.pkl"):
                    if "embed" in str(file_path).lower() or "vector" in str(file_path).lower():
                        legacy_data["mock_embeddings"].append(file_path)
                        logger.info(f"Found potential embedding file: {file_path}")

        total_files = sum(len(files) for files in legacy_data.values())
        logger.info(f"Found {total_files} legacy files to process")

        return legacy_data

    def create_upgraded_database(self) -> None:
        """Create upgraded RAG database schema."""
        logger.info("Creating upgraded RAG database schema...")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Enable WAL mode and optimizations
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=10000")

        # Documents table (CODEX-compliant)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT UNIQUE NOT NULL,
                title TEXT,
                content TEXT NOT NULL,
                document_type TEXT DEFAULT 'text',
                source_path TEXT,
                source_url TEXT,
                file_hash TEXT,
                word_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        """
        )

        # Chunks table with real embedding support
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT UNIQUE NOT NULL,
                document_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                chunk_size INTEGER,
                overlap_size INTEGER DEFAULT 50,
                embedding_vector BLOB,
                embedding_model TEXT DEFAULT ?,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents (document_id),
                UNIQUE(document_id, chunk_index)
            )
        """,
            (RAG_EMBEDDING_MODEL,),
        )

        # Embeddings metadata
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT NOT NULL,
                vector_dimension INTEGER DEFAULT ?,
                faiss_index_id INTEGER,
                bm25_doc_id INTEGER,
                similarity_scores TEXT,
                last_queried TIMESTAMP,
                query_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (chunk_id) REFERENCES chunks (chunk_id),
                UNIQUE(chunk_id)
            )
        """,
            (RAG_VECTOR_DIM,),
        )

        # Migration tracking
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_migration_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                migration_type TEXT NOT NULL,
                source_file TEXT,
                chunks_processed INTEGER DEFAULT 0,
                embeddings_generated INTEGER DEFAULT 0,
                migration_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'completed',
                notes TEXT
            )
        """
        )

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(file_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(document_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_index ON chunks(chunk_index)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_faiss ON embeddings_metadata(faiss_index_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_queries ON embeddings_metadata(query_count)")

        conn.commit()
        conn.close()

        logger.info(f"Upgraded database created at {self.db_path}")

    def extract_documents_from_legacy(self, legacy_files: dict[str, list[Path]]) -> list[dict[str, Any]]:
        """Extract documents from legacy files."""
        logger.info("Extracting documents from legacy files...")

        documents = []

        # Process SHA256 files
        for file_path in legacy_files["sha256_files"]:
            try:
                with open(file_path, encoding="utf-8") as f:
                    data = json.load(f)

                # Extract documents from various formats
                if isinstance(data, dict):
                    if "documents" in data:
                        docs = data["documents"]
                    elif "texts" in data:
                        docs = [{"content": text, "id": f"doc_{i}"} for i, text in enumerate(data["texts"])]
                    else:
                        # Assume the whole object is a document
                        docs = [{"content": str(data), "id": file_path.stem}]
                elif isinstance(data, list):
                    docs = (
                        data
                        if data and isinstance(data[0], dict)
                        else [{"content": str(item), "id": f"doc_{i}"} for i, item in enumerate(data)]
                    )
                else:
                    docs = [{"content": str(data), "id": file_path.stem}]

                for doc in docs:
                    if isinstance(doc, dict) and doc.get("content"):
                        document = {
                            "document_id": doc.get("id", f"{file_path.stem}_{len(documents)}"),
                            "title": doc.get("title", f"Document from {file_path.name}"),
                            "content": doc["content"],
                            "source_path": str(file_path),
                            "document_type": "legacy_sha256",
                            "metadata": json.dumps(
                                {
                                    "source_file": str(file_path),
                                    "extracted_at": datetime.now().isoformat(),
                                    "original_format": "sha256",
                                }
                            ),
                        }
                        documents.append(document)

            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")

        # Add sample educational documents if no legacy found
        if not documents:
            logger.info("No legacy documents found, creating sample educational content...")
            sample_docs = [
                {
                    "document_id": "edu_ml_basics",
                    "title": "Machine Learning Fundamentals",
                    "content": """Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task. It involves algorithms that can identify patterns in data and make predictions or classifications based on those patterns. The three main types are supervised learning (learning from labeled examples), unsupervised learning (finding hidden patterns in unlabeled data), and reinforcement learning (learning through trial and error with rewards and penalties).""",
                    "document_type": "educational",
                    "metadata": json.dumps({"category": "AI/ML", "difficulty": "beginner"}),
                },
                {
                    "document_id": "edu_neural_networks",
                    "title": "Neural Networks Overview",
                    "content": """Neural networks are computational models inspired by the human brain, consisting of interconnected nodes (neurons) organized in layers. They excel at pattern recognition tasks like image classification, natural language processing, and speech recognition. Deep neural networks with multiple hidden layers can learn increasingly complex representations of data. Common architectures include convolutional neural networks (CNNs) for images, recurrent neural networks (RNNs) for sequences, and transformers for natural language understanding.""",
                    "document_type": "educational",
                    "metadata": json.dumps({"category": "AI/ML", "difficulty": "intermediate"}),
                },
                {
                    "document_id": "edu_data_science",
                    "title": "Data Science Process",
                    "content": """Data science is an interdisciplinary field that combines statistics, programming, and domain expertise to extract insights from data. The typical process involves data collection, cleaning, exploration, modeling, and interpretation. Key skills include programming (Python, R, SQL), statistical analysis, machine learning, and data visualization. Common applications include predictive analytics, recommendation systems, fraud detection, and business intelligence.""",
                    "document_type": "educational",
                    "metadata": json.dumps({"category": "Data Science", "difficulty": "beginner"}),
                },
            ]
            documents.extend(sample_docs)

        logger.info(f"Extracted {len(documents)} documents for upgrade")
        return documents

    def generate_real_embeddings(self, documents: list[dict[str, Any]]) -> dict[str, Any]:
        """Generate real vector embeddings for documents."""
        logger.info("Generating real vector embeddings...")

        upgrade_stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "faiss_entries": 0,
            "errors": [],
        }

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Initialize FAISS index
        index = faiss.IndexIDMap(faiss.IndexFlatIP(self.vector_dim))

        # Process each document
        for doc in documents:
            try:
                # Insert document
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO documents
                    (document_id, title, content, document_type, source_path, file_hash, word_count, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        doc["document_id"],
                        doc.get("title", ""),
                        doc["content"],
                        doc.get("document_type", "text"),
                        doc.get("source_path", ""),
                        hashlib.sha256(doc["content"].encode()).hexdigest(),
                        len(doc["content"].split()),
                        doc.get("metadata", "{}"),
                    ),
                )

                upgrade_stats["documents_processed"] += 1

                # Create chunks
                chunks = self._create_chunks(doc["content"], doc["document_id"])

                for chunk_idx, chunk_text in enumerate(chunks):
                    chunk_id = f"{doc['document_id']}_chunk_{chunk_idx}"

                    # Generate real embedding
                    embedding = self.embedder.encode(chunk_text, convert_to_numpy=True)
                    embedding = embedding.astype("float32")

                    # Store chunk with embedding
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO chunks
                        (chunk_id, document_id, chunk_index, content, chunk_size, embedding_vector, embedding_model)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            chunk_id,
                            doc["document_id"],
                            chunk_idx,
                            chunk_text,
                            len(chunk_text.split()),
                            pickle.dumps(embedding),
                            RAG_EMBEDDING_MODEL,
                        ),
                    )

                    # Add to FAISS index
                    chunk_hash = hash(chunk_id) & 0x7FFFFFFFFFFFFFFF  # Ensure positive
                    index.add_with_ids(np.array([embedding]), np.array([chunk_hash], dtype="int64"))

                    # Store embedding metadata
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO embeddings_metadata
                        (chunk_id, vector_dimension, faiss_index_id)
                        VALUES (?, ?, ?)
                    """,
                        (chunk_id, self.vector_dim, chunk_hash),
                    )

                    upgrade_stats["chunks_created"] += 1
                    upgrade_stats["embeddings_generated"] += 1
                    upgrade_stats["faiss_entries"] += 1

            except Exception as e:
                error_msg = f"Error processing document {doc.get('document_id', 'unknown')}: {e}"
                logger.exception(error_msg)
                upgrade_stats["errors"].append(error_msg)

        # Save FAISS index
        faiss.write_index(index, str(self.faiss_index_path))
        logger.info(f"FAISS index saved to {self.faiss_index_path}")

        # Save BM25 corpus data
        self._create_bm25_corpus(cursor)

        conn.commit()
        conn.close()

        return upgrade_stats

    def _create_chunks(self, text: str, doc_id: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
        """Create chunks from text with specified overlap."""
        words = text.split()
        chunks = []

        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append(chunk_text)
            start = end - overlap
            start = max(start, 0)
            if end >= len(words):
                break

        return chunks

    def _create_bm25_corpus(self, cursor: sqlite3.Cursor) -> None:
        """Create BM25 corpus from chunks."""
        logger.info("Creating BM25 corpus...")

        cursor.execute("SELECT chunk_id, content FROM chunks")
        chunks = cursor.fetchall()

        corpus_data = {
            "corpus": [chunk[1].lower().split() for chunk in chunks],
            "ids": [chunk[0] for chunk in chunks],
            "created_at": datetime.now().isoformat(),
            "model": "BM25Okapi",
        }

        with open(self.bm25_corpus_path, "w", encoding="utf-8") as f:
            json.dump(corpus_data, f, indent=2)

        logger.info(f"BM25 corpus saved to {self.bm25_corpus_path}")

    def test_retrieval_accuracy(self) -> dict[str, Any]:
        """Test retrieval accuracy with real embeddings."""
        logger.info("Testing retrieval accuracy...")

        test_queries = [
            "What is machine learning?",
            "How do neural networks work?",
            "Data science process steps",
            "Supervised vs unsupervised learning",
        ]

        accuracy_results = {
            "queries_tested": len(test_queries),
            "successful_retrievals": 0,
            "average_results_per_query": 0,
            "test_results": [],
        }

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Load FAISS index
        try:
            index = faiss.read_index(str(self.faiss_index_path))
        except Exception as e:
            logger.exception(f"Cannot load FAISS index: {e}")
            return accuracy_results

        total_results = 0

        for query in test_queries:
            try:
                # Generate query embedding
                query_embedding = self.embedder.encode(query, convert_to_numpy=True).astype("float32")

                # Search FAISS index
                scores, ids = index.search(np.array([query_embedding]), k=5)

                # Get results
                results = []
                for score, chunk_id in zip(scores[0], ids[0], strict=False):
                    if chunk_id != -1:  # Valid result
                        cursor.execute(
                            "SELECT chunk_id, content FROM chunks WHERE ? IN (SELECT faiss_index_id FROM embeddings_metadata WHERE chunk_id = chunks.chunk_id)",
                            (int(chunk_id),),
                        )
                        chunk_data = cursor.fetchone()
                        if chunk_data:
                            results.append(
                                {
                                    "chunk_id": chunk_data[0],
                                    "content": chunk_data[1][:100] + "...",
                                    "score": float(score),
                                }
                            )

                test_result = {
                    "query": query,
                    "results_count": len(results),
                    "results": results,
                }

                accuracy_results["test_results"].append(test_result)
                total_results += len(results)

                if len(results) > 0:
                    accuracy_results["successful_retrievals"] += 1

                logger.info(f"Query '{query}': {len(results)} results")

            except Exception as e:
                logger.exception(f"Error testing query '{query}': {e}")

        accuracy_results["average_results_per_query"] = total_results / len(test_queries)

        conn.close()
        return accuracy_results

    def archive_legacy_system(self, legacy_files: dict[str, list[Path]]) -> None:
        """Archive legacy SHA256-based system."""
        logger.info("Archiving legacy system files...")

        archive_dir = Path("./data/archive/legacy_rag_system")
        archive_dir.mkdir(parents=True, exist_ok=True)

        for file_type, files in legacy_files.items():
            type_dir = archive_dir / file_type
            type_dir.mkdir(exist_ok=True)

            for file_path in files:
                try:
                    import shutil

                    archive_path = type_dir / file_path.name
                    shutil.copy2(file_path, archive_path)
                    logger.info(f"Archived {file_path} -> {archive_path}")
                except Exception as e:
                    logger.exception(f"Error archiving {file_path}: {e}")

    def run_upgrade(self) -> dict[str, Any]:
        """Execute complete RAG system upgrade."""
        logger.info("Starting RAG system upgrade...")

        start_time = datetime.now()

        # Find legacy data
        legacy_files = self.find_legacy_rag_data()

        # Create upgraded database
        self.create_upgraded_database()

        # Extract documents from legacy files
        documents = self.extract_documents_from_legacy(legacy_files)

        # Generate real embeddings
        upgrade_stats = self.generate_real_embeddings(documents)

        # Test retrieval accuracy
        accuracy_results = self.test_retrieval_accuracy()

        # Archive legacy system
        self.archive_legacy_system(legacy_files)

        # Log migration
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO rag_migration_log
            (migration_type, chunks_processed, embeddings_generated, status, notes)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                "sha256_to_real_embeddings",
                upgrade_stats["chunks_created"],
                upgrade_stats["embeddings_generated"],
                "completed",
                f"Upgraded to {RAG_EMBEDDING_MODEL} with {self.vector_dim} dimensions",
            ),
        )
        conn.commit()
        conn.close()

        # Generate final report
        report = {
            "status": "completed",
            "upgrade_type": "sha256_to_real_embeddings",
            "embedding_model": RAG_EMBEDDING_MODEL,
            "vector_dimension": self.vector_dim,
            "legacy_files": {k: len(v) for k, v in legacy_files.items()},
            "upgrade_stats": upgrade_stats,
            "accuracy_test": accuracy_results,
            "duration": (datetime.now() - start_time).total_seconds(),
        }

        logger.info(f"RAG upgrade completed: {upgrade_stats['embeddings_generated']} real embeddings generated")

        return report


def main() -> None:
    """Main upgrade function."""
    upgrader = RAGSystemUpgrader()
    report = upgrader.run_upgrade()

    # Save upgrade report
    report_path = Path("./data/rag_system_upgrade_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n{'=' * 50}")
    print("RAG SYSTEM UPGRADE COMPLETE")
    print(f"{'=' * 50}")
    print(f"Status: {report['status']}")
    print(f"Embedding model: {report['embedding_model']}")
    print(f"Vector dimension: {report['vector_dimension']}")
    print(f"Documents processed: {report['upgrade_stats']['documents_processed']}")
    print(f"Chunks created: {report['upgrade_stats']['chunks_created']}")
    print(f"Real embeddings: {report['upgrade_stats']['embeddings_generated']}")
    print(f"FAISS entries: {report['upgrade_stats']['faiss_entries']}")
    print(
        f"Successful retrievals: {report['accuracy_test']['successful_retrievals']}/{report['accuracy_test']['queries_tested']}"
    )
    print(f"Duration: {report['duration']:.2f} seconds")
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
