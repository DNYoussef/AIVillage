"""Wikipedia Data Loader for CODEX RAG Integration.

Loads and processes Wikipedia articles for educational content retrieval.
"""

import hashlib
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup
from rag_system.core.codex_rag_integration import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Wikipedia API configuration
WIKIPEDIA_API_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"
WIKIPEDIA_CONTENT_URL = "https://en.wikipedia.org/api/rest_v1/page/html/"

# Educational topics for initial corpus
EDUCATIONAL_TOPICS = [
    # Computer Science
    "Machine_learning",
    "Artificial_intelligence",
    "Neural_network",
    "Deep_learning",
    "Natural_language_processing",
    "Computer_vision",
    "Reinforcement_learning",
    "Algorithm",
    "Data_structure",
    "Python_(programming_language)",
    # Mathematics
    "Linear_algebra",
    "Calculus",
    "Statistics",
    "Probability_theory",
    "Graph_theory",
    # Science
    "Physics",
    "Chemistry",
    "Biology",
    "Quantum_mechanics",
    "Evolution",
    # History & Literature
    "World_War_II",
    "Renaissance",
    "Shakespeare",
    "Ancient_Rome",
    "Industrial_Revolution",
]


class WikipediaDataLoader:
    """Loads and processes Wikipedia articles for RAG system."""

    def __init__(self, data_dir: str = "./data") -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.corpus_path = self.data_dir / "wikipedia_corpus.json"
        self.db_path = self.data_dir / "rag_index.db"

        # Initialize database
        self._init_database()

    def _init_database(self) -> None:
        """Initialize the RAG index database according to CODEX specs."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create documents table
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

        # Create chunks table
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
                embedding_model TEXT DEFAULT 'paraphrase-MiniLM-L3-v2',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents (document_id),
                UNIQUE(document_id, chunk_index)
            )
        """
        )

        # Create embeddings metadata table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT NOT NULL,
                vector_dimension INTEGER DEFAULT 384,
                faiss_index_id INTEGER,
                bm25_doc_id INTEGER,
                similarity_scores TEXT,
                last_queried TIMESTAMP,
                query_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (chunk_id) REFERENCES chunks (chunk_id),
                UNIQUE(chunk_id)
            )
        """
        )

        # Create indices for performance
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(file_hash)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(document_type)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_index ON chunks(chunk_index)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_embeddings_faiss ON embeddings_metadata(faiss_index_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_embeddings_queries ON embeddings_metadata(query_count)"
        )

        conn.commit()
        conn.close()

        logger.info(f"Database initialized at {self.db_path}")

    def fetch_wikipedia_article(self, title: str) -> dict[str, Any] | None:
        """Fetch a Wikipedia article by title."""
        try:
            # Get article summary
            summary_url = f"{WIKIPEDIA_API_URL}{title}"
            summary_response = requests.get(summary_url, timeout=10)

            if summary_response.status_code != 200:
                logger.warning(
                    f"Failed to fetch {title}: {summary_response.status_code}"
                )
                return None

            summary_data = summary_response.json()

            # Get full content
            content_url = f"{WIKIPEDIA_CONTENT_URL}{title}"
            content_response = requests.get(content_url, timeout=10)

            if content_response.status_code == 200:
                # Parse HTML content
                soup = BeautifulSoup(content_response.text, "html.parser")

                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()

                # Get text content
                text = soup.get_text()

                # Clean up text
                lines = (line.strip() for line in text.splitlines())
                chunks = (
                    phrase.strip() for line in lines for phrase in line.split("  ")
                )
                text = " ".join(chunk for chunk in chunks if chunk)

                # Limit to reasonable length
                text = text[:50000]  # ~10k words
            else:
                # Fallback to extract from summary
                text = summary_data.get("extract", "")

            return {
                "title": summary_data.get("title", title),
                "content": text,
                "description": summary_data.get("description", ""),
                "extract": summary_data.get("extract", ""),
                "url": summary_data.get("content_urls", {})
                .get("desktop", {})
                .get("page", ""),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.exception(f"Error fetching {title}: {e}")
            return None

    def load_wikipedia_corpus(self, topics: list[str] | None = None) -> list[Document]:
        """Load Wikipedia articles for specified topics."""
        topics = topics or EDUCATIONAL_TOPICS
        documents = []

        logger.info(f"Loading {len(topics)} Wikipedia articles...")

        for topic in topics:
            logger.info(f"Fetching: {topic}")
            article = self.fetch_wikipedia_article(topic)

            if article:
                # Create document ID
                doc_id = f"wiki_{hashlib.md5(topic.encode()).hexdigest()[:8]}"

                # Create Document object
                doc = Document(
                    id=doc_id,
                    title=article["title"],
                    content=article["content"] or article["extract"],
                    source_type="wikipedia",
                    metadata={
                        "description": article["description"],
                        "url": article["url"],
                        "fetched_at": article["timestamp"],
                        "topic": topic,
                        "category": self._categorize_topic(topic),
                    },
                )
                documents.append(doc)

                # Store in database
                self._store_document(doc)

            # Rate limiting
            import time

            time.sleep(0.5)

        logger.info(f"Loaded {len(documents)} documents")

        # Save corpus to file
        self._save_corpus(documents)

        return documents

    def _categorize_topic(self, topic: str) -> str:
        """Categorize topic for metadata."""
        topic_lower = topic.lower()

        if any(
            term in topic_lower
            for term in [
                "learning",
                "intelligence",
                "neural",
                "algorithm",
                "computer",
                "python",
            ]
        ):
            return "Computer Science"
        if any(
            term in topic_lower
            for term in ["algebra", "calculus", "statistics", "probability", "graph"]
        ):
            return "Mathematics"
        if any(
            term in topic_lower
            for term in ["physics", "chemistry", "biology", "quantum", "evolution"]
        ):
            return "Science"
        if any(
            term in topic_lower
            for term in ["war", "renaissance", "shakespeare", "rome", "revolution"]
        ):
            return "History & Literature"
        return "General"

    def _store_document(self, doc: Document) -> None:
        """Store document in SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Calculate file hash
            file_hash = hashlib.sha256(doc.content.encode()).hexdigest()
            word_count = len(doc.content.split())

            # Insert or update document
            cursor.execute(
                """
                INSERT OR REPLACE INTO documents
                (document_id, title, content, document_type, source_url, file_hash, word_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    doc.id,
                    doc.title,
                    doc.content,
                    doc.source_type,
                    doc.metadata.get("url", ""),
                    file_hash,
                    word_count,
                    json.dumps(doc.metadata),
                ),
            )

            conn.commit()

        except Exception as e:
            logger.exception(f"Error storing document {doc.id}: {e}")
            conn.rollback()
        finally:
            conn.close()

    def _save_corpus(self, documents: list[Document]) -> None:
        """Save corpus to JSON file."""
        corpus_data = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "num_documents": len(documents),
            "documents": [
                {
                    "id": doc.id,
                    "title": doc.title,
                    "content": doc.content,
                    "source_type": doc.source_type,
                    "metadata": doc.metadata,
                }
                for doc in documents
            ],
        }

        with open(self.corpus_path, "w", encoding="utf-8") as f:
            json.dump(corpus_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Corpus saved to {self.corpus_path}")

    def load_corpus_from_file(self) -> list[Document]:
        """Load previously saved corpus."""
        if not self.corpus_path.exists():
            logger.warning(f"No corpus found at {self.corpus_path}")
            return []

        with open(self.corpus_path, encoding="utf-8") as f:
            data = json.load(f)

        documents = [Document(**doc_data) for doc_data in data["documents"]]

        logger.info(f"Loaded {len(documents)} documents from corpus")
        return documents

    def get_document_stats(self) -> dict[str, Any]:
        """Get statistics about stored documents."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get document count
        cursor.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]

        # Get total word count
        cursor.execute("SELECT SUM(word_count) FROM documents")
        total_words = cursor.fetchone()[0] or 0

        # Get category distribution
        cursor.execute(
            "SELECT document_type, COUNT(*) FROM documents GROUP BY document_type"
        )
        categories = dict(cursor.fetchall())

        # Get chunk count
        cursor.execute("SELECT COUNT(*) FROM chunks")
        chunk_count = cursor.fetchone()[0]

        conn.close()

        return {
            "document_count": doc_count,
            "total_words": total_words,
            "avg_words_per_doc": total_words / doc_count if doc_count > 0 else 0,
            "categories": categories,
            "chunk_count": chunk_count,
        }


def create_sample_educational_content() -> list[Document]:
    """Create sample educational content for testing."""
    samples = [
        Document(
            id="edu_1",
            title="Introduction to Machine Learning",
            content="""Machine learning is a revolutionary field of artificial intelligence that enables computers to learn from data without being explicitly programmed.
            At its core, machine learning involves algorithms that can identify patterns in data and make decisions based on those patterns.
            There are three main types of machine learning: supervised learning, where the algorithm learns from labeled examples;
            unsupervised learning, where the algorithm discovers hidden patterns in unlabeled data; and reinforcement learning,
            where the algorithm learns through trial and error by receiving rewards or penalties for its actions.

            Common applications of machine learning include image recognition, natural language processing, recommendation systems,
            fraud detection, and autonomous vehicles. The field has seen tremendous growth due to advances in computing power,
            the availability of large datasets, and improvements in algorithms. Deep learning, a subset of machine learning based on
            artificial neural networks, has been particularly successful in tasks like computer vision and speech recognition.""",
            source_type="educational",
            metadata={"difficulty": "beginner", "subject": "AI", "reading_time": 5},
        ),
        Document(
            id="edu_2",
            title="Understanding Neural Networks",
            content="""Neural networks are computational models inspired by the human brain's structure and function.
            They consist of interconnected nodes called neurons, organized in layers: an input layer that receives data,
            one or more hidden layers that process the information, and an output layer that produces the final result.

            Each connection between neurons has a weight that determines its strength, and these weights are adjusted during
            the training process through a technique called backpropagation. The network learns by minimizing the difference
            between its predictions and the actual outcomes using optimization algorithms like gradient descent.

            Deep neural networks, which have multiple hidden layers, can learn increasingly complex representations of data.
            Convolutional neural networks (CNNs) excel at image processing tasks by using specialized layers that can detect
            features like edges and shapes. Recurrent neural networks (RNNs) and their variants like LSTMs are designed to
            handle sequential data such as text or time series. Transformer architectures have revolutionized natural language
            processing with models like BERT and GPT.""",
            source_type="educational",
            metadata={"difficulty": "intermediate", "subject": "AI", "reading_time": 6},
        ),
        Document(
            id="edu_3",
            title="Python Programming Fundamentals",
            content="""Python is a high-level, interpreted programming language known for its simplicity and readability.
            Created by Guido van Rossum and first released in 1991, Python emphasizes code readability with its use of
            significant whitespace and clear, expressive syntax.

            Key features of Python include dynamic typing, automatic memory management, and support for multiple programming
            paradigms including procedural, object-oriented, and functional programming. Python's extensive standard library
            provides modules for everything from file I/O to web development, earning it the nickname "batteries included."

            Python is widely used in data science, machine learning, web development, automation, and scientific computing.
            Popular libraries like NumPy and Pandas enable efficient data manipulation, while frameworks like TensorFlow and
            PyTorch power machine learning applications. Web frameworks such as Django and Flask make it easy to build scalable
            web applications. Python's versatility and gentle learning curve make it an ideal first programming language.""",
            source_type="educational",
            metadata={
                "difficulty": "beginner",
                "subject": "Programming",
                "reading_time": 5,
            },
        ),
    ]

    return samples


if __name__ == "__main__":
    # Initialize loader
    loader = WikipediaDataLoader()

    # Load Wikipedia corpus
    documents = loader.load_wikipedia_corpus(
        EDUCATIONAL_TOPICS[:5]
    )  # Start with 5 topics

    # Add sample educational content
    samples = create_sample_educational_content()
    documents.extend(samples)

    # Save complete corpus
    loader._save_corpus(documents)

    # Print statistics
    stats = loader.get_document_stats()
    print("\n=== Corpus Statistics ===")
    print(f"Documents: {stats['document_count']}")
    print(f"Total words: {stats['total_words']:,}")
    print(f"Average words per document: {stats['avg_words_per_doc']:.0f}")
    print(f"Categories: {stats['categories']}")
    print(f"Chunks: {stats['chunk_count']}")
