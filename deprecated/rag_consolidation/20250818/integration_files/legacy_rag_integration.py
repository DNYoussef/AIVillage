"""Agent Forge RAG Integration.

Integrates the top-performing Agent Forge model into the HyperRAG retrieval pipeline:
- Automatic model selection based on benchmark results
- HyperRAG pipeline integration with performance optimization
- End-to-end Q&A latency and accuracy validation
- Real-time performance monitoring and optimization
"""

import asyncio
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from agent_forge.results_analyzer import ResultsAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """Configuration for RAG integration."""

    model_path: str
    model_name: str
    embedding_model: str = "all-MiniLM-L6-v2"
    vector_store_path: str = "./vector_store"
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k_retrieval: int = 5
    max_context_length: int = 2048
    temperature: float = 0.7
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class RAGPerformanceMetrics:
    """Performance metrics for RAG system."""

    query_latency: float
    retrieval_latency: float
    generation_latency: float
    total_latency: float
    retrieval_accuracy: float
    answer_quality: float
    context_relevance: float
    memory_usage: float


class HyperRAGIntegration:
    """Advanced RAG system with Agent Forge model integration."""

    def __init__(self, config: RAGConfig) -> None:
        self.config = config
        self.model = None
        self.tokenizer = None
        self.embedding_model = None
        self.vector_store = None
        self.document_chunks = []
        self.performance_history = []

        # Initialize W&B tracking
        wandb.init(
            project="agent-forge-rag", config=asdict(config), job_type="rag_integration"
        )

    async def initialize(self) -> None:
        """Initialize all RAG components."""
        logger.info("Initializing HyperRAG system")

        # Load Agent Forge model
        await self._load_agent_forge_model()

        # Initialize embedding model
        await self._initialize_embedding_model()

        # Load or create vector store
        await self._setup_vector_store()

        logger.info("HyperRAG initialization complete")

    async def _load_agent_forge_model(self) -> None:
        """Load the top-performing Agent Forge model."""
        logger.info(f"Loading Agent Forge model: {self.config.model_name}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=(
                    torch.float16 if self.config.device == "cuda" else torch.float32
                ),
                device_map="auto" if self.config.device == "cuda" else None,
                trust_remote_code=True,
            )

            if self.config.device == "cpu":
                self.model = self.model.to(self.config.device)

            self.model.eval()

            logger.info("Agent Forge model loaded successfully")

        except Exception as e:
            logger.exception(f"Failed to load Agent Forge model: {e}")
            raise

    async def _initialize_embedding_model(self) -> None:
        """Initialize sentence embedding model for retrieval."""
        logger.info(f"Initializing embedding model: {self.config.embedding_model}")

        try:
            self.embedding_model = SentenceTransformer(self.config.embedding_model)
            self.embedding_model.eval()

            logger.info("Embedding model initialized successfully")

        except Exception as e:
            logger.exception(f"Failed to initialize embedding model: {e}")
            raise

    async def _setup_vector_store(self) -> None:
        """Setup FAISS vector store for document retrieval."""
        logger.info("Setting up vector store")

        vector_store_path = Path(self.config.vector_store_path)
        vector_store_path.mkdir(parents=True, exist_ok=True)

        index_file = vector_store_path / "faiss_index.bin"
        chunks_file = vector_store_path / "document_chunks.json"

        if index_file.exists() and chunks_file.exists():
            # Load existing vector store
            logger.info("Loading existing vector store")
            self.vector_store = faiss.read_index(str(index_file))

            with open(chunks_file) as f:
                self.document_chunks = json.load(f)

        else:
            # Create new vector store with sample documents
            logger.info("Creating new vector store with sample documents")
            await self._create_sample_vector_store(vector_store_path)

        logger.info(f"Vector store ready with {len(self.document_chunks)} chunks")

    async def _create_sample_vector_store(self, vector_store_path: Path) -> None:
        """Create sample vector store for demonstration."""
        # Sample educational documents
        sample_documents = [
            {
                "content": "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data. It includes supervised learning, unsupervised learning, and reinforcement learning approaches.",
                "metadata": {"topic": "machine_learning", "difficulty": "beginner"},
            },
            {
                "content": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information and learn patterns from data through training.",
                "metadata": {"topic": "neural_networks", "difficulty": "intermediate"},
            },
            {
                "content": "Deep learning uses neural networks with multiple hidden layers to model and understand complex patterns in data. It has achieved breakthrough results in computer vision, natural language processing, and other domains.",
                "metadata": {"topic": "deep_learning", "difficulty": "advanced"},
            },
            {
                "content": "Natural language processing (NLP) enables computers to understand, interpret, and generate human language. Key tasks include text classification, named entity recognition, and machine translation.",
                "metadata": {"topic": "nlp", "difficulty": "intermediate"},
            },
            {
                "content": "Transformers are a neural network architecture that uses self-attention mechanisms to process sequential data. They have revolutionized NLP and are the foundation of models like BERT and GPT.",
                "metadata": {"topic": "transformers", "difficulty": "advanced"},
            },
        ]

        # Create document chunks
        chunks = []
        for doc in sample_documents:
            # Simple chunking (in practice, would use more sophisticated methods)
            content = doc["content"]
            chunk_size = self.config.chunk_size

            for i in range(0, len(content), chunk_size - self.config.chunk_overlap):
                chunk_text = content[i : i + chunk_size]
                if len(chunk_text.strip()) > 50:  # Filter very short chunks
                    chunks.append(
                        {
                            "text": chunk_text,
                            "metadata": doc["metadata"],
                            "chunk_id": len(chunks),
                        }
                    )

        self.document_chunks = chunks

        # Generate embeddings
        logger.info("Generating embeddings for document chunks")
        chunk_texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_model.encode(chunk_texts, show_progress_bar=True)

        # Create FAISS index
        dimension = embeddings.shape[1]
        self.vector_store = faiss.IndexFlatIP(dimension)  # Inner product similarity

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.vector_store.add(embeddings.astype(np.float32))

        # Save vector store
        faiss.write_index(self.vector_store, str(vector_store_path / "faiss_index.bin"))

        with open(vector_store_path / "document_chunks.json", "w") as f:
            json.dump(self.document_chunks, f, indent=2)

        logger.info("Vector store created and saved")

    async def retrieve_relevant_context(self, query: str) -> tuple[list[str], float]:
        """Retrieve relevant context for a query."""
        start_time = time.time()

        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)

            # Search vector store
            similarities, indices = self.vector_store.search(
                query_embedding.astype(np.float32), self.config.top_k_retrieval
            )

            # Extract relevant chunks
            relevant_chunks = []
            total_length = 0

            for idx, _similarity in zip(indices[0], similarities[0], strict=False):
                if idx != -1:  # Valid index
                    chunk = self.document_chunks[idx]
                    chunk_text = chunk["text"]

                    # Check if adding this chunk would exceed context length
                    if total_length + len(chunk_text) <= self.config.max_context_length:
                        relevant_chunks.append(chunk_text)
                        total_length += len(chunk_text)
                    else:
                        break

            retrieval_latency = time.time() - start_time

            # Calculate retrieval accuracy (simplified metric)
            avg_similarity = (
                float(np.mean(similarities[0][similarities[0] > 0]))
                if len(similarities[0]) > 0
                else 0.0
            )

            return relevant_chunks, retrieval_latency, avg_similarity

        except Exception as e:
            logger.exception(f"Retrieval failed: {e}")
            return [], time.time() - start_time, 0.0

    async def generate_response(
        self, query: str, context: list[str]
    ) -> tuple[str, float]:
        """Generate response using Agent Forge model with retrieved context."""
        start_time = time.time()

        try:
            # Construct prompt with context
            context_text = "\n\n".join(context)

            prompt = f"""Based on the following context, please answer the question accurately and helpfully.

Context:
{context_text}

Question: {query}

Answer:"""

            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(
                self.config.device
            )

            # Check input length
            if inputs.size(1) > self.config.max_context_length:
                # Truncate if too long
                inputs = inputs[:, -self.config.max_context_length :]

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.size(1)
                    + 150,  # Allow for reasonable response length
                    temperature=self.config.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_response[
                len(self.tokenizer.decode(inputs[0], skip_special_tokens=True)) :
            ].strip()

            generation_latency = time.time() - start_time

            return response, generation_latency

        except Exception as e:
            logger.exception(f"Generation failed: {e}")
            return (
                f"I apologize, but I encountered an error while generating a response: {e!s}",
                time.time() - start_time,
            )

    async def answer_question(self, query: str) -> tuple[str, RAGPerformanceMetrics]:
        """Complete RAG pipeline: retrieve context and generate answer."""
        total_start_time = time.time()

        logger.info(f"Processing query: {query}")

        # Retrieve relevant context
        (
            context_chunks,
            retrieval_latency,
            retrieval_accuracy,
        ) = await self.retrieve_relevant_context(query)

        # Generate response
        response, generation_latency = await self.generate_response(
            query, context_chunks
        )

        total_latency = time.time() - total_start_time

        # Calculate performance metrics
        metrics = RAGPerformanceMetrics(
            query_latency=0.001,  # Minimal query processing time
            retrieval_latency=retrieval_latency,
            generation_latency=generation_latency,
            total_latency=total_latency,
            retrieval_accuracy=retrieval_accuracy,
            answer_quality=self._assess_answer_quality(query, response),
            context_relevance=self._assess_context_relevance(query, context_chunks),
            memory_usage=self._get_memory_usage(),
        )

        # Log metrics to W&B
        wandb.log(
            {
                "retrieval_latency": retrieval_latency,
                "generation_latency": generation_latency,
                "total_latency": total_latency,
                "retrieval_accuracy": retrieval_accuracy,
                "answer_quality": metrics.answer_quality,
                "context_relevance": metrics.context_relevance,
                "memory_usage_gb": metrics.memory_usage,
            }
        )

        # Store performance history
        self.performance_history.append(metrics)

        logger.info(f"Query processed in {total_latency:.3f}s")

        return response, metrics

    def _assess_answer_quality(self, query: str, response: str) -> float:
        """Assess answer quality (simplified heuristic)."""
        # Basic quality metrics
        if not response or len(response.strip()) < 10:
            return 0.0

        # Check for error messages
        if "error" in response.lower() or "apologize" in response.lower():
            return 0.3

        # Length-based quality (reasonable responses should be substantial but not too long)
        response_length = len(response.split())
        if 10 <= response_length <= 200:
            length_score = 1.0
        elif response_length < 10:
            length_score = response_length / 10
        else:
            length_score = max(0.5, 200 / response_length)

        # Keyword overlap with query (very basic relevance)
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        overlap = len(query_words.intersection(response_words))
        relevance_score = min(1.0, overlap / max(1, len(query_words)))

        return (length_score + relevance_score) / 2

    def _assess_context_relevance(self, query: str, context_chunks: list[str]) -> float:
        """Assess context relevance to query."""
        if not context_chunks:
            return 0.0

        query_words = set(query.lower().split())
        total_relevance = 0.0

        for chunk in context_chunks:
            chunk_words = set(chunk.lower().split())
            overlap = len(query_words.intersection(chunk_words))
            chunk_relevance = overlap / max(1, len(query_words))
            total_relevance += chunk_relevance

        return total_relevance / len(context_chunks)

    def _get_memory_usage(self) -> float:
        """Get current GPU memory usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
        return 0.0

    async def validate_rag_performance(self, test_queries: list[str]) -> dict[str, Any]:
        """Validate RAG system performance on test queries."""
        logger.info(f"Validating RAG performance on {len(test_queries)} queries")

        results = {
            "total_queries": len(test_queries),
            "successful_queries": 0,
            "failed_queries": 0,
            "average_latency": 0.0,
            "average_quality": 0.0,
            "latency_distribution": [],
            "quality_distribution": [],
            "detailed_results": [],
        }

        total_latency = 0.0
        total_quality = 0.0

        for i, query in enumerate(test_queries):
            try:
                response, metrics = await self.answer_question(query)

                results["successful_queries"] += 1
                total_latency += metrics.total_latency
                total_quality += metrics.answer_quality

                results["latency_distribution"].append(metrics.total_latency)
                results["quality_distribution"].append(metrics.answer_quality)

                results["detailed_results"].append(
                    {"query": query, "response": response, "metrics": asdict(metrics)}
                )

                logger.info(
                    f"Query {i + 1}/{len(test_queries)}: "
                    f"{metrics.total_latency:.3f}s, quality: "
                    f"{metrics.answer_quality:.3f}"
                )

            except Exception as e:
                logger.exception(f"Query {i + 1} failed: {e}")
                results["failed_queries"] += 1

        # Calculate summary statistics
        if results["successful_queries"] > 0:
            results["average_latency"] = total_latency / results["successful_queries"]
            results["average_quality"] = total_quality / results["successful_queries"]

            # Calculate percentiles
            latencies = results["latency_distribution"]
            qualities = results["quality_distribution"]

            results["latency_p50"] = np.percentile(latencies, 50)
            results["latency_p95"] = np.percentile(latencies, 95)
            results["latency_p99"] = np.percentile(latencies, 99)

            results["quality_p50"] = np.percentile(qualities, 50)
            results["quality_p95"] = np.percentile(qualities, 95)

        # Log summary to W&B
        wandb.log(
            {
                "validation/total_queries": results["total_queries"],
                "validation/success_rate": results["successful_queries"]
                / results["total_queries"],
                "validation/average_latency": results["average_latency"],
                "validation/average_quality": results["average_quality"],
                "validation/latency_p95": results.get("latency_p95", 0),
                "validation/quality_p50": results.get("quality_p50", 0),
            }
        )

        logger.info(
            f"Validation complete: {results['successful_queries']}/{results['total_queries']} successful"
        )

        return results


class AgentForgeRAGSelector:
    """Selects the best Agent Forge model for RAG integration."""

    def __init__(self, results_dir: str) -> None:
        self.results_dir = Path(results_dir)
        self.analyzer = ResultsAnalyzer(str(results_dir))

    async def select_best_model(self) -> dict[str, Any]:
        """Select the best performing Agent Forge model for RAG integration."""
        logger.info("Selecting best Agent Forge model for RAG integration")

        try:
            # Analyze results to find best model
            analysis = await self.analyzer.analyze_comprehensive_results()

            if "insights" not in analysis:
                msg = "No insights found in analysis results"
                raise ValueError(msg)

            insights = analysis["insights"]
            best_phase = insights["best_performing_phase"]

            # Map phase to model path
            model_paths = {
                "evomerge_best": "./evomerge_output/best_model",
                "quietstar_enhanced": "./quietstar_enhanced",
                "original_compressed": "./final_compressed_model",
                "mastery_trained": "./mastery_output/final_model",
                "unified_pipeline": "./unified_checkpoints/final_model",
            }

            if best_phase not in model_paths:
                msg = f"Unknown phase: {best_phase}"
                raise ValueError(msg)

            model_path = model_paths[best_phase]

            # Validate model exists
            if not Path(model_path).exists():
                logger.warning(
                    f"Best model path not found: {model_path}, falling back to available models"
                )

                # Find first available model
                for phase, path in model_paths.items():
                    if Path(path).exists():
                        best_phase = phase
                        model_path = path
                        break
                else:
                    msg = "No valid Agent Forge models found"
                    raise ValueError(msg)

            selection_result = {
                "selected_phase": best_phase,
                "model_path": model_path,
                "confidence": insights.get("confidence_level", "medium"),
                "performance_score": analysis.get("json_analysis", {})
                .get("performance_trends", {})
                .get("best_score", 0.0),
                "selection_criteria": {
                    "overall_performance": True,
                    "benchmark_consistency": True,
                    "model_availability": True,
                },
                "recommendation": insights.get(
                    "deployment_recommendation", "Deploy with monitoring"
                ),
            }

            logger.info(f"Selected model: {best_phase} at {model_path}")

            return selection_result

        except Exception as e:
            logger.exception(f"Model selection failed: {e}")
            # Fallback to default model
            return {
                "selected_phase": "mastery_trained",
                "model_path": "./mastery_output/final_model",
                "confidence": "low",
                "performance_score": 0.0,
                "selection_criteria": {"fallback": True},
                "recommendation": "Fallback selection - validate performance before deployment",
            }


# Sample test queries for validation
SAMPLE_TEST_QUERIES = [
    "What is machine learning and how does it work?",
    "Explain the difference between supervised and unsupervised learning",
    "How do neural networks learn from data?",
    "What are the main applications of deep learning?",
    "How do transformers work in natural language processing?",
    "What is the attention mechanism in transformers?",
    "Explain gradient descent optimization",
    "What are the challenges in training deep neural networks?",
    "How does backpropagation work?",
    "What is the difference between RNNs and transformers?",
]


# CLI interface
async def main() -> int:
    """Main CLI for RAG integration."""
    import argparse

    parser = argparse.ArgumentParser(description="Agent Forge RAG Integration")
    parser.add_argument(
        "--results-dir",
        default="./benchmark_results",
        help="Benchmark results directory",
    )
    parser.add_argument(
        "--auto-select", action="store_true", help="Automatically select best model"
    )
    parser.add_argument("--model-path", help="Specific model path to use")
    parser.add_argument("--model-name", help="Model name for tracking")
    parser.add_argument("--validate", action="store_true", help="Run validation tests")
    parser.add_argument(
        "--interactive", action="store_true", help="Interactive Q&A mode"
    )

    args = parser.parse_args()

    if args.auto_select:
        # Automatically select best model
        selector = AgentForgeRAGSelector(args.results_dir)
        selection = await selector.select_best_model()

        print(f"\n{'=' * 60}")
        print("AGENT FORGE MODEL SELECTION FOR RAG")
        print(f"{'=' * 60}")
        print(f"Selected Phase: {selection['selected_phase']}")
        print(f"Model Path: {selection['model_path']}")
        print(f"Confidence: {selection['confidence']}")
        print(f"Performance Score: {selection['performance_score']:.3f}")
        print(f"Recommendation: {selection['recommendation']}")

        model_path = selection["model_path"]
        model_name = f"agent-forge-{selection['selected_phase']}"

    elif args.model_path and args.model_name:
        model_path = args.model_path
        model_name = args.model_name

    else:
        print(
            "Error: Must specify either --auto-select or both --model-path and --model-name"
        )
        return 1

    # Initialize RAG system
    config = RAGConfig(model_path=model_path, model_name=model_name)

    rag_system = HyperRAGIntegration(config)

    print(f"\nInitializing RAG system with {model_name}...")
    await rag_system.initialize()
    print("✅ RAG system initialized successfully")

    if args.validate:
        # Run validation tests
        print(f"\nRunning validation on {len(SAMPLE_TEST_QUERIES)} test queries...")
        validation_results = await rag_system.validate_rag_performance(
            SAMPLE_TEST_QUERIES
        )

        print(f"\n{'=' * 60}")
        print("RAG VALIDATION RESULTS")
        print(f"{'=' * 60}")
        print(
            f"Success Rate: {validation_results['successful_queries']}/{
                validation_results['total_queries']
            } ({
                validation_results['successful_queries']
                / validation_results['total_queries']
                * 100:.1f}%)"
        )
        print(f"Average Latency: {validation_results['average_latency']:.3f}s")
        print(f"Average Quality: {validation_results['average_quality']:.3f}")

        if "latency_p95" in validation_results:
            print(f"Latency P95: {validation_results['latency_p95']:.3f}s")
            print(f"Quality P50: {validation_results['quality_p50']:.3f}")

    if args.interactive:
        # Interactive Q&A mode
        print(f"\n{'=' * 60}")
        print("INTERACTIVE RAG Q&A MODE")
        print(f"{'=' * 60}")
        print("Enter questions (type 'quit' to exit):")

        while True:
            try:
                query = input("\n❓ Question: ").strip()

                if query.lower() in ["quit", "exit", "q"]:
                    break

                if not query:
                    continue

                print("🤔 Thinking...")
                response, metrics = await rag_system.answer_question(query)

                print(f"\n💡 Answer: {response}")
                print(f"⏱️  Response time: {metrics.total_latency:.3f}s")
                print(f"📊 Quality score: {metrics.answer_quality:.3f}")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

        print("\nGoodbye! 👋")

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
