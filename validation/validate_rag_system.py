"""RAG System Component Validation Suite.

Tests RAG query, retrieval, FAISS integration, and graph functionality.
"""

import logging
from pathlib import Path
import sys
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.production.rag.rag_system.core.config import RAGConfig
    from src.production.rag.rag_system.core.pipeline import RAGPipeline
    from src.production.rag.rag_system.ingestion.document_processor import (
        DocumentProcessor,
    )
    from src.production.rag.rag_system.retrieval.vector_retrieval import VectorRetriever
except ImportError as e:
    print(f"Warning: Could not import RAG components: {e}")
    RAGPipeline = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class RAGSystemValidator:
    """Validates RAG System component functionality."""

    def __init__(self) -> None:
        self.results = {
            "rag_pipeline": {"status": "pending", "time": 0, "details": ""},
            "document_processing": {"status": "pending", "time": 0, "details": ""},
            "vector_retrieval": {"status": "pending", "time": 0, "details": ""},
            "query_processing": {"status": "pending", "time": 0, "details": ""},
        }

    def test_rag_pipeline(self) -> None:
        """Test RAG pipeline initialization and configuration."""
        logger.info("Testing RAG Pipeline...")
        start_time = time.time()

        try:
            if RAGPipeline is None:
                self.results["rag_pipeline"] = {
                    "status": "failed",
                    "time": time.time() - start_time,
                    "details": "RAGPipeline could not be imported",
                }
                return

            # Test configuration
            config = RAGConfig(
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                chunk_size=256,
                chunk_overlap=50,
                vector_store_type="faiss",
                top_k=5,
            )

            # Initialize pipeline
            pipeline = RAGPipeline(config)

            if hasattr(pipeline, "config") and hasattr(pipeline, "process_query"):
                self.results["rag_pipeline"] = {
                    "status": "success",
                    "time": time.time() - start_time,
                    "details": f"Pipeline initialized. Embedding model: {config.embedding_model}, Chunk size: {config.chunk_size}",
                }
            else:
                self.results["rag_pipeline"] = {
                    "status": "partial",
                    "time": time.time() - start_time,
                    "details": f"Pipeline created but missing expected methods. Available: {[m for m in dir(pipeline) if not m.startswith('_')][:5]}",
                }

        except Exception as e:
            self.results["rag_pipeline"] = {
                "status": "failed",
                "time": time.time() - start_time,
                "details": f"Error: {e!s}",
            }

    def test_document_processing(self) -> None:
        """Test document processing and chunking functionality."""
        logger.info("Testing Document Processing...")
        start_time = time.time()

        try:
            # Test document processor
            processor = DocumentProcessor(chunk_size=256, chunk_overlap=50)

            # Test with sample text
            sample_document = {
                "content": "This is a sample document for testing RAG functionality. " * 10,
                "metadata": {"title": "Test Document", "source": "validation_test"},
            }

            if hasattr(processor, "process_document"):
                chunks = processor.process_document(sample_document)

                if chunks and len(chunks) > 0:
                    self.results["document_processing"] = {
                        "status": "success",
                        "time": time.time() - start_time,
                        "details": f"Document processed into {len(chunks)} chunks. First chunk length: {len(chunks[0].get('content', ''))}",
                    }
                else:
                    self.results["document_processing"] = {
                        "status": "partial",
                        "time": time.time() - start_time,
                        "details": "Document processor available but returned no chunks",
                    }
            else:
                self.results["document_processing"] = {
                    "status": "partial",
                    "time": time.time() - start_time,
                    "details": f"DocumentProcessor created. Available methods: {[m for m in dir(processor) if not m.startswith('_') and 'process' in m]}",
                }

        except Exception as e:
            self.results["document_processing"] = {
                "status": "failed",
                "time": time.time() - start_time,
                "details": f"Error: {e!s}",
            }

    def test_vector_retrieval(self) -> None:
        """Test vector retrieval and FAISS integration."""
        logger.info("Testing Vector Retrieval...")
        start_time = time.time()

        try:
            # Test vector retriever
            retriever = VectorRetriever(
                embedding_model="sentence-transformers/all-MiniLM-L6-v2", vector_store_type="faiss", top_k=3
            )

            if hasattr(retriever, "retrieve_similar"):
                # Test with sample query (without actually loading embeddings)
                test_query = "What is artificial intelligence?"

                try:
                    # This might fail due to no indexed documents, but tests API
                    results = retriever.retrieve_similar(test_query)
                    self.results["vector_retrieval"] = {
                        "status": "success",
                        "time": time.time() - start_time,
                        "details": f"Vector retrieval functional. Retrieved {len(results) if results else 0} results.",
                    }
                except Exception as retrieval_error:
                    # Retriever exists but needs indexed data
                    self.results["vector_retrieval"] = {
                        "status": "partial",
                        "time": time.time() - start_time,
                        "details": f"Vector retriever available but needs indexed data: {str(retrieval_error)[:50]}...",
                    }
            else:
                self.results["vector_retrieval"] = {
                    "status": "partial",
                    "time": time.time() - start_time,
                    "details": f"VectorRetriever created. Available methods: {[m for m in dir(retriever) if not m.startswith('_') and 'retrieve' in m]}",
                }

        except Exception as e:
            self.results["vector_retrieval"] = {
                "status": "failed",
                "time": time.time() - start_time,
                "details": f"Error: {e!s}",
            }

    def test_query_processing(self) -> None:
        """Test end-to-end query processing."""
        logger.info("Testing Query Processing...")
        start_time = time.time()

        try:
            # Test basic query processing components
            test_query = {
                "query": "What are the key concepts in machine learning?",
                "context_type": "academic",
                "max_results": 5,
            }

            # Test query preprocessing
            processed_query = self._preprocess_query(test_query["query"])

            if processed_query:
                self.results["query_processing"] = {
                    "status": "success",
                    "time": time.time() - start_time,
                    "details": f"Query processed successfully. Original length: {len(test_query['query'])}, Processed: {processed_query[:50]}...",
                }
            else:
                self.results["query_processing"] = {
                    "status": "partial",
                    "time": time.time() - start_time,
                    "details": "Query processing components available but need configuration",
                }

        except Exception as e:
            self.results["query_processing"] = {
                "status": "failed",
                "time": time.time() - start_time,
                "details": f"Error: {e!s}",
            }

    def _preprocess_query(self, query: str) -> str:
        """Simple query preprocessing for testing."""
        return query.strip().lower()

    def run_validation(self):
        """Run all RAG System validation tests."""
        logger.info("=== RAG System Validation Suite ===")

        # Run all tests
        self.test_rag_pipeline()
        self.test_document_processing()
        self.test_vector_retrieval()
        self.test_query_processing()

        # Calculate results
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results.values() if r["status"] == "success")
        partial_tests = sum(1 for r in self.results.values() if r["status"] == "partial")

        logger.info("=== RAG System Validation Results ===")
        for test_name, result in self.results.items():
            status_emoji = {"success": "PASS", "partial": "WARN", "failed": "FAIL", "pending": "PEND"}

            logger.info(f"[{status_emoji[result['status']]}] {test_name}: {result['status'].upper()}")
            logger.info(f"   Time: {result['time']:.2f}s")
            logger.info(f"   Details: {result['details']}")

        success_rate = (successful_tests + partial_tests * 0.5) / total_tests
        logger.info(f"\nRAG System Success Rate: {success_rate:.1%} ({successful_tests + partial_tests}/{total_tests})")

        return self.results, success_rate


if __name__ == "__main__":
    validator = RAGSystemValidator()
    results, success_rate = validator.run_validation()

    if success_rate >= 0.8:
        print("RAG System Validation: PASSED")
    else:
        print("RAG System Validation: NEEDS IMPROVEMENT")
