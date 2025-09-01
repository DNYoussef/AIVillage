"""
Performance Benchmarking Suite for Unified RAG System

Comprehensive performance testing and benchmarking with detailed metrics,
comparative analysis, and optimization recommendations.
"""

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil

# Add source directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from unified_rag import UnifiedRAGSystem
from unified_rag.core.unified_rag_system import QueryType, RetrievalMode, QueryContext

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a performance benchmark."""
    
    benchmark_name: str
    test_description: str
    
    # Performance metrics
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_dev_ms: float
    
    # Throughput metrics
    operations_per_second: float
    queries_per_second: float = 0.0
    documents_per_second: float = 0.0
    
    # Quality metrics
    avg_confidence: float = 0.0
    avg_relevance: float = 0.0
    success_rate: float = 1.0
    
    # Resource usage
    peak_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0
    
    # Component-specific metrics
    component_times: Dict[str, float] = field(default_factory=dict)
    component_usage: Dict[str, int] = field(default_factory=dict)
    
    # Test parameters
    test_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    system_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results."""
    
    suite_name: str
    total_duration_ms: float
    benchmark_results: List[BenchmarkResult] = field(default_factory=list)
    
    # Summary metrics
    total_operations: int = 0
    overall_success_rate: float = 1.0
    
    # System information
    system_specs: Dict[str, Any] = field(default_factory=dict)
    test_environment: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamp
    executed_at: str = field(default_factory=lambda: datetime.now().isoformat())


class PerformanceBenchmarks:
    """
    Comprehensive Performance Benchmarking System
    
    Tests all aspects of the unified RAG system performance including:
    - Query processing speed and accuracy
    - Document ingestion throughput
    - Memory usage and optimization
    - Component performance analysis
    - Scalability testing
    - Concurrent load testing
    """
    
    def __init__(self, output_dir: str = "benchmarks"):
        self.output_dir = output_dir
        self.system: Optional[UnifiedRAGSystem] = None
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # System monitoring
        self.process = psutil.Process()
        
        # Test data
        self.test_queries = self._generate_test_queries()
        self.test_documents = self._generate_test_documents()
    
    def _generate_test_queries(self) -> List[Tuple[str, QueryType]]:
        """Generate diverse test queries for benchmarking."""
        return [
            # Factual queries
            ("What is machine learning?", QueryType.FACTUAL),
            ("Define artificial neural networks", QueryType.FACTUAL),
            ("What is deep learning?", QueryType.FACTUAL),
            ("Explain natural language processing", QueryType.FACTUAL),
            ("What is reinforcement learning?", QueryType.FACTUAL),
            
            # Analytical queries
            ("How does supervised learning differ from unsupervised learning?", QueryType.ANALYTICAL),
            ("Why are neural networks effective for pattern recognition?", QueryType.ANALYTICAL),
            ("How do convolutional neural networks process images?", QueryType.ANALYTICAL),
            ("What makes transformer models effective for NLP?", QueryType.ANALYTICAL),
            ("Why is feature engineering important in machine learning?", QueryType.ANALYTICAL),
            
            # Comparative queries
            ("Compare gradient descent and genetic algorithms", QueryType.COMPARATIVE),
            ("Contrast decision trees with neural networks", QueryType.COMPARATIVE),
            ("Compare supervised and reinforcement learning approaches", QueryType.COMPARATIVE),
            ("Contrast CNN and RNN architectures", QueryType.COMPARATIVE),
            ("Compare batch and online learning methods", QueryType.COMPARATIVE),
            
            # Creative queries
            ("What are innovative applications of machine learning in healthcare?", QueryType.CREATIVE),
            ("How might quantum computing enhance machine learning?", QueryType.CREATIVE),
            ("What creative solutions exist for the explainability problem in AI?", QueryType.CREATIVE),
            ("Brainstorm novel uses of generative AI in education", QueryType.CREATIVE),
            ("What are unconventional approaches to neural architecture search?", QueryType.CREATIVE),
            
            # Exploratory queries
            ("Explore the connections between neuroscience and artificial intelligence", QueryType.EXPLORATORY),
            ("Investigate the relationship between statistics and machine learning", QueryType.EXPLORATORY),
            ("Examine the evolution of deep learning architectures", QueryType.EXPLORATORY),
            ("Discover emerging trends in federated learning", QueryType.EXPLORATORY),
            ("Explore the intersection of machine learning and robotics", QueryType.EXPLORATORY),
            
            # Synthesis queries
            ("Synthesize the key principles of effective machine learning systems", QueryType.SYNTHESIS),
            ("Integrate knowledge about optimization techniques in deep learning", QueryType.SYNTHESIS),
            ("Combine insights from computer vision and natural language processing", QueryType.SYNTHESIS),
            ("Synthesize best practices for machine learning model deployment", QueryType.SYNTHESIS),
            ("Integrate ethical considerations with technical machine learning concepts", QueryType.SYNTHESIS)
        ]
    
    def _generate_test_documents(self, count: int = 50) -> List[Dict[str, Any]]:
        """Generate test documents of varying sizes and complexity."""
        documents = []
        
        # Base content templates
        templates = [
            "Machine learning concepts and applications",
            "Deep learning neural network architectures", 
            "Natural language processing techniques and methods",
            "Computer vision and image recognition systems",
            "Reinforcement learning algorithms and strategies",
            "Data science and statistical analysis methods",
            "Artificial intelligence research and development",
            "Pattern recognition and feature extraction",
            "Optimization techniques and mathematical foundations",
            "Knowledge representation and reasoning systems"
        ]
        
        for i in range(count):
            template = templates[i % len(templates)]
            
            # Vary document lengths
            if i < count // 3:
                # Short documents (100-300 words)
                word_count = 100 + (i % 200)
                size_category = "short"
            elif i < 2 * count // 3:
                # Medium documents (300-800 words)
                word_count = 300 + (i % 500)
                size_category = "medium"
            else:
                # Long documents (800-2000 words)
                word_count = 800 + (i % 1200)
                size_category = "long"
            
            # Generate content
            content = self._generate_document_content(template, word_count)
            
            doc = {
                "doc_id": f"benchmark_doc_{i:04d}",
                "content": content,
                "metadata": {
                    "topic": template.split()[0].lower(),
                    "word_count": word_count,
                    "size_category": size_category,
                    "complexity": ["simple", "moderate", "complex"][i % 3],
                    "benchmark_doc": True,
                    "index": i
                }
            }
            
            documents.append(doc)
        
        return documents
    
    def _generate_document_content(self, template: str, word_count: int) -> str:
        """Generate document content based on template and target word count."""
        base_text = f"This document discusses {template}. "
        
        # Generate paragraphs
        paragraphs = []
        words_remaining = word_count
        paragraph_count = max(3, word_count // 100)  # Roughly 100 words per paragraph
        
        for i in range(paragraph_count):
            if words_remaining <= 0:
                break
            
            paragraph_words = min(words_remaining, 50 + (i * 10))  # Varying paragraph lengths
            
            # Generate paragraph content
            paragraph = f"Paragraph {i+1} about {template}. " + \
                       f"This section covers important aspects of the topic. " * (paragraph_words // 20)
            
            paragraphs.append(paragraph.strip())
            words_remaining -= paragraph_words
        
        return base_text + "\n\n".join(paragraphs)
    
    async def run_full_benchmark_suite(self) -> BenchmarkSuite:
        """Run the complete benchmark suite."""
        logger.info("🚀 Starting Unified RAG Performance Benchmark Suite")
        suite_start_time = time.time()
        
        # Initialize system
        self.system = UnifiedRAGSystem(
            enable_all_components=True,
            mcp_integration=True,
            performance_mode="balanced"
        )
        
        await self.system.initialize()
        
        try:
            suite = BenchmarkSuite(
                suite_name="unified_rag_performance_suite",
                total_duration_ms=0.0,
                system_specs=self._get_system_specs(),
                test_environment=self._get_test_environment()
            )
            
            # Run individual benchmarks
            benchmarks = [
                self._benchmark_document_ingestion,
                self._benchmark_query_performance,
                self._benchmark_retrieval_modes,
                self._benchmark_component_performance,
                self._benchmark_concurrent_load,
                self._benchmark_memory_usage,
                self._benchmark_scalability
            ]
            
            for benchmark_func in benchmarks:
                try:
                    logger.info(f"Running benchmark: {benchmark_func.__name__}")
                    result = await benchmark_func()
                    suite.benchmark_results.append(result)
                    suite.total_operations += result.test_parameters.get("operations", 0)
                    
                except Exception as e:
                    logger.error(f"Benchmark {benchmark_func.__name__} failed: {e}")
                    
                    # Create error result
                    error_result = BenchmarkResult(
                        benchmark_name=benchmark_func.__name__,
                        test_description=f"Failed with error: {str(e)}",
                        total_time_ms=0.0,
                        avg_time_ms=0.0,
                        min_time_ms=0.0,
                        max_time_ms=0.0,
                        std_dev_ms=0.0,
                        operations_per_second=0.0,
                        success_rate=0.0,
                        test_parameters={"error": str(e)}
                    )
                    suite.benchmark_results.append(error_result)
            
            suite.total_duration_ms = (time.time() - suite_start_time) * 1000
            suite.overall_success_rate = sum(r.success_rate for r in suite.benchmark_results) / len(suite.benchmark_results)
            
            # Save results
            await self._save_benchmark_results(suite)
            
            # Print summary
            self._print_benchmark_summary(suite)
            
            return suite
            
        finally:
            if self.system:
                await self.system.shutdown()
    
    async def _benchmark_document_ingestion(self) -> BenchmarkResult:
        """Benchmark document ingestion performance."""
        logger.info("📄 Benchmarking document ingestion...")
        
        documents = self.test_documents[:25]  # Use subset for timing
        
        times = []
        success_count = 0
        memory_usage = []
        
        start_time = time.time()
        
        for doc in documents:
            doc_start = time.time()
            memory_before = self.process.memory_info().rss / 1024 / 1024  # MB
            
            success = await self.system.ingest_document(
                content=doc["content"],
                doc_id=doc["doc_id"],
                metadata=doc["metadata"]
            )
            
            doc_time = (time.time() - doc_start) * 1000
            memory_after = self.process.memory_info().rss / 1024 / 1024
            
            times.append(doc_time)
            memory_usage.append(memory_after - memory_before)
            
            if success:
                success_count += 1
        
        total_time = (time.time() - start_time) * 1000
        
        return BenchmarkResult(
            benchmark_name="document_ingestion",
            test_description=f"Ingested {len(documents)} documents of varying sizes",
            total_time_ms=total_time,
            avg_time_ms=np.mean(times),
            min_time_ms=np.min(times),
            max_time_ms=np.max(times),
            std_dev_ms=np.std(times),
            operations_per_second=len(documents) / (total_time / 1000),
            documents_per_second=success_count / (total_time / 1000),
            success_rate=success_count / len(documents),
            peak_memory_mb=max(memory_usage) if memory_usage else 0,
            test_parameters={
                "documents": len(documents),
                "operations": len(documents),
                "avg_document_size": np.mean([len(d["content"]) for d in documents])
            }
        )
    
    async def _benchmark_query_performance(self) -> BenchmarkResult:
        """Benchmark query processing performance."""
        logger.info("❓ Benchmarking query performance...")
        
        # Ingest some documents first
        for doc in self.test_documents[:10]:
            await self.system.ingest_document(
                content=doc["content"],
                doc_id=doc["doc_id"],
                metadata=doc["metadata"]
            )
        
        # Test queries
        queries = self.test_queries[:20]  # Use subset
        
        times = []
        confidences = []
        relevances = []
        success_count = 0
        component_times = {}
        
        start_time = time.time()
        
        for query, query_type in queries:
            query_start = time.time()
            
            try:
                response = await self.system.query(
                    question=query,
                    context=QueryContext(max_results=5),
                    query_type=query_type
                )
                
                query_time = (time.time() - query_start) * 1000
                times.append(query_time)
                
                if response:
                    confidences.append(response.confidence)
                    relevances.append(getattr(response, 'avg_relevance', 0.0))
                    success_count += 1
                    
                    # Accumulate component times
                    for comp, comp_time in response.component_times.items():
                        if comp not in component_times:
                            component_times[comp] = []
                        component_times[comp].append(comp_time)
                
            except Exception as e:
                logger.warning(f"Query failed: {e}")
                times.append(0.0)
        
        total_time = (time.time() - start_time) * 1000
        
        # Calculate component averages
        avg_component_times = {
            comp: np.mean(times_list) 
            for comp, times_list in component_times.items()
        }
        
        return BenchmarkResult(
            benchmark_name="query_performance",
            test_description=f"Processed {len(queries)} queries of various types",
            total_time_ms=total_time,
            avg_time_ms=np.mean(times),
            min_time_ms=np.min(times),
            max_time_ms=np.max(times),
            std_dev_ms=np.std(times),
            operations_per_second=len(queries) / (total_time / 1000),
            queries_per_second=success_count / (total_time / 1000),
            avg_confidence=np.mean(confidences) if confidences else 0,
            avg_relevance=np.mean(relevances) if relevances else 0,
            success_rate=success_count / len(queries),
            component_times=avg_component_times,
            test_parameters={
                "queries": len(queries),
                "operations": len(queries),
                "query_types": len(set(qt.value for _, qt in queries))
            }
        )
    
    async def _benchmark_retrieval_modes(self) -> BenchmarkResult:
        """Benchmark different retrieval modes."""
        logger.info("🔄 Benchmarking retrieval modes...")
        
        # Ingest documents
        for doc in self.test_documents[:15]:
            await self.system.ingest_document(
                content=doc["content"],
                doc_id=doc["doc_id"],
                metadata=doc["metadata"]
            )
        
        query = "Explain machine learning and neural networks"
        modes = list(RetrievalMode)
        
        times = []
        mode_performance = {}
        
        start_time = time.time()
        
        for mode in modes:
            mode_times = []
            
            # Test each mode multiple times
            for _ in range(3):
                mode_start = time.time()
                
                response = await self.system.query(
                    question=query,
                    context=QueryContext(max_results=5),
                    retrieval_mode=mode
                )
                
                mode_time = (time.time() - mode_start) * 1000
                mode_times.append(mode_time)
            
            avg_mode_time = np.mean(mode_times)
            times.extend(mode_times)
            mode_performance[mode.value] = avg_mode_time
        
        total_time = (time.time() - start_time) * 1000
        
        return BenchmarkResult(
            benchmark_name="retrieval_modes",
            test_description=f"Tested {len(modes)} retrieval modes with {3} iterations each",
            total_time_ms=total_time,
            avg_time_ms=np.mean(times),
            min_time_ms=np.min(times),
            max_time_ms=np.max(times),
            std_dev_ms=np.std(times),
            operations_per_second=len(times) / (total_time / 1000),
            success_rate=1.0,  # All modes should work
            component_times=mode_performance,
            test_parameters={
                "retrieval_modes": len(modes),
                "iterations_per_mode": 3,
                "operations": len(times)
            }
        )
    
    async def _benchmark_component_performance(self) -> BenchmarkResult:
        """Benchmark individual component performance."""
        logger.info("⚙️ Benchmarking component performance...")
        
        # Ingest documents
        for doc in self.test_documents[:10]:
            await self.system.ingest_document(
                content=doc["content"],
                doc_id=doc["doc_id"],
                metadata=doc["metadata"]
            )
        
        query = "What are the key concepts in artificial intelligence?"
        
        # Test comprehensive retrieval to engage all components
        start_time = time.time()
        
        response = await self.system.query(
            question=query,
            context=QueryContext(
                max_results=10,
                enable_creative_search=True,
                include_reasoning=True
            ),
            retrieval_mode=RetrievalMode.COMPREHENSIVE
        )
        
        total_time = (time.time() - start_time) * 1000
        
        # Analyze component usage
        component_usage = {}
        if hasattr(response, 'vector_results'):
            component_usage["vector"] = len(response.vector_results)
        if hasattr(response, 'graph_results'):
            component_usage["graph"] = len(response.graph_results)
        if hasattr(response, 'memory_results'):
            component_usage["memory"] = len(response.memory_results)
        if hasattr(response, 'creative_results'):
            component_usage["creative"] = len(response.creative_results)
        
        return BenchmarkResult(
            benchmark_name="component_performance",
            test_description="Tested comprehensive retrieval using all components",
            total_time_ms=total_time,
            avg_time_ms=total_time,
            min_time_ms=total_time,
            max_time_ms=total_time,
            std_dev_ms=0.0,
            operations_per_second=1.0 / (total_time / 1000),
            avg_confidence=response.confidence if response else 0,
            success_rate=1.0 if response else 0.0,
            component_times=response.component_times if response else {},
            component_usage=component_usage,
            test_parameters={
                "comprehensive_retrieval": True,
                "operations": 1
            }
        )
    
    async def _benchmark_concurrent_load(self) -> BenchmarkResult:
        """Benchmark performance under concurrent load."""
        logger.info("🚀 Benchmarking concurrent load performance...")
        
        # Ingest documents
        for doc in self.test_documents[:10]:
            await self.system.ingest_document(
                content=doc["content"],
                doc_id=doc["doc_id"],
                metadata=doc["metadata"]
            )
        
        # Prepare concurrent queries
        concurrent_queries = [
            "What is machine learning?",
            "Explain deep learning concepts", 
            "How do neural networks work?",
            "What is natural language processing?",
            "Define reinforcement learning",
            "Compare supervised and unsupervised learning",
            "What are the applications of AI?",
            "How does gradient descent work?",
            "Explain convolutional neural networks",
            "What is transfer learning?"
        ]
        
        start_time = time.time()
        cpu_before = psutil.cpu_percent()
        memory_before = self.process.memory_info().rss / 1024 / 1024
        
        # Execute queries concurrently
        tasks = []
        for query in concurrent_queries:
            task = self.system.query(
                question=query,
                context=QueryContext(max_results=5)
            )
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = (time.time() - start_time) * 1000
        cpu_after = psutil.cpu_percent()
        memory_after = self.process.memory_info().rss / 1024 / 1024
        
        # Analyze results
        successful_responses = [r for r in responses if not isinstance(r, Exception)]
        success_count = len(successful_responses)
        
        if successful_responses:
            avg_confidence = np.mean([r.confidence for r in successful_responses])
            response_times = [r.processing_time_ms for r in successful_responses]
        else:
            avg_confidence = 0.0
            response_times = [0.0]
        
        return BenchmarkResult(
            benchmark_name="concurrent_load",
            test_description=f"Executed {len(concurrent_queries)} queries concurrently",
            total_time_ms=total_time,
            avg_time_ms=np.mean(response_times),
            min_time_ms=np.min(response_times),
            max_time_ms=np.max(response_times),
            std_dev_ms=np.std(response_times),
            operations_per_second=len(concurrent_queries) / (total_time / 1000),
            queries_per_second=success_count / (total_time / 1000),
            avg_confidence=avg_confidence,
            success_rate=success_count / len(concurrent_queries),
            peak_memory_mb=memory_after - memory_before,
            avg_cpu_percent=(cpu_before + cpu_after) / 2,
            test_parameters={
                "concurrent_queries": len(concurrent_queries),
                "operations": len(concurrent_queries),
                "parallelism": len(concurrent_queries)
            }
        )
    
    async def _benchmark_memory_usage(self) -> BenchmarkResult:
        """Benchmark memory usage patterns."""
        logger.info("💾 Benchmarking memory usage...")
        
        memory_measurements = []
        operation_times = []
        
        start_time = time.time()
        initial_memory = self.process.memory_info().rss / 1024 / 1024
        
        # Ingest documents and measure memory
        for i, doc in enumerate(self.test_documents[:20]):
            op_start = time.time()
            memory_before = self.process.memory_info().rss / 1024 / 1024
            
            await self.system.ingest_document(
                content=doc["content"],
                doc_id=doc["doc_id"],
                metadata=doc["metadata"]
            )
            
            memory_after = self.process.memory_info().rss / 1024 / 1024
            op_time = (time.time() - op_start) * 1000
            
            memory_measurements.append(memory_after)
            operation_times.append(op_time)
        
        # Perform queries and measure memory
        for query, _ in self.test_queries[:10]:
            op_start = time.time()
            
            await self.system.query(
                question=query,
                context=QueryContext(max_results=5)
            )
            
            memory_after = self.process.memory_info().rss / 1024 / 1024
            op_time = (time.time() - op_start) * 1000
            
            memory_measurements.append(memory_after)
            operation_times.append(op_time)
        
        total_time = (time.time() - start_time) * 1000
        final_memory = self.process.memory_info().rss / 1024 / 1024
        
        return BenchmarkResult(
            benchmark_name="memory_usage",
            test_description="Measured memory usage during ingestion and querying",
            total_time_ms=total_time,
            avg_time_ms=np.mean(operation_times),
            min_time_ms=np.min(operation_times),
            max_time_ms=np.max(operation_times),
            std_dev_ms=np.std(operation_times),
            operations_per_second=len(operation_times) / (total_time / 1000),
            success_rate=1.0,
            peak_memory_mb=max(memory_measurements) - initial_memory,
            test_parameters={
                "initial_memory_mb": initial_memory,
                "final_memory_mb": final_memory,
                "memory_increase_mb": final_memory - initial_memory,
                "peak_memory_mb": max(memory_measurements),
                "operations": len(operation_times)
            }
        )
    
    async def _benchmark_scalability(self) -> BenchmarkResult:
        """Benchmark system scalability."""
        logger.info("📈 Benchmarking system scalability...")
        
        # Test with increasing loads
        document_counts = [5, 10, 20, 30]
        scalability_results = {}
        
        total_start_time = time.time()
        
        for doc_count in document_counts:
            # Clear system (reinitialize)
            await self.system.shutdown()
            self.system = UnifiedRAGSystem(enable_all_components=True)
            await self.system.initialize()
            
            # Ingest documents
            docs_to_ingest = self.test_documents[:doc_count]
            
            ingest_start = time.time()
            for doc in docs_to_ingest:
                await self.system.ingest_document(
                    content=doc["content"],
                    doc_id=doc["doc_id"],
                    metadata=doc["metadata"]
                )
            ingest_time = (time.time() - ingest_start) * 1000
            
            # Test queries
            test_queries = self.test_queries[:5]  # Fixed number of queries
            
            query_start = time.time()
            for query, query_type in test_queries:
                await self.system.query(
                    question=query,
                    context=QueryContext(max_results=5),
                    query_type=query_type
                )
            query_time = (time.time() - query_start) * 1000
            
            scalability_results[f"docs_{doc_count}"] = {
                "documents": doc_count,
                "ingest_time_ms": ingest_time,
                "query_time_ms": query_time,
                "avg_ingest_per_doc": ingest_time / doc_count,
                "avg_query_time": query_time / len(test_queries)
            }
        
        total_time = (time.time() - total_start_time) * 1000
        
        # Calculate scalability metrics
        all_ingest_times = [r["ingest_time_ms"] for r in scalability_results.values()]
        all_query_times = [r["query_time_ms"] for r in scalability_results.values()]
        
        return BenchmarkResult(
            benchmark_name="scalability",
            test_description=f"Tested scalability with document counts: {document_counts}",
            total_time_ms=total_time,
            avg_time_ms=np.mean(all_ingest_times + all_query_times),
            min_time_ms=np.min(all_ingest_times + all_query_times),
            max_time_ms=np.max(all_ingest_times + all_query_times),
            std_dev_ms=np.std(all_ingest_times + all_query_times),
            operations_per_second=sum(document_counts) / (total_time / 1000),
            success_rate=1.0,
            component_times=scalability_results,
            test_parameters={
                "document_counts_tested": document_counts,
                "queries_per_test": 5,
                "operations": sum(document_counts) + len(document_counts) * 5
            }
        )
    
    def _get_system_specs(self) -> Dict[str, Any]:
        """Get system specifications."""
        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
            "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
            "platform": sys.platform,
            "python_version": sys.version.split()[0]
        }
    
    def _get_test_environment(self) -> Dict[str, Any]:
        """Get test environment information."""
        return {
            "unified_rag_version": "1.0.0",
            "test_suite_version": "1.0.0",
            "output_directory": self.output_dir,
            "test_data_size": {
                "queries": len(self.test_queries),
                "documents": len(self.test_documents)
            }
        }
    
    async def _save_benchmark_results(self, suite: BenchmarkSuite):
        """Save benchmark results to files."""
        # Save JSON results
        json_path = os.path.join(self.output_dir, f"benchmark_results_{int(time.time())}.json")
        
        # Convert to serializable format
        results_dict = {
            "suite_name": suite.suite_name,
            "total_duration_ms": suite.total_duration_ms,
            "total_operations": suite.total_operations,
            "overall_success_rate": suite.overall_success_rate,
            "executed_at": suite.executed_at,
            "system_specs": suite.system_specs,
            "test_environment": suite.test_environment,
            "benchmarks": []
        }
        
        for result in suite.benchmark_results:
            result_dict = {
                "benchmark_name": result.benchmark_name,
                "test_description": result.test_description,
                "performance_metrics": {
                    "total_time_ms": result.total_time_ms,
                    "avg_time_ms": result.avg_time_ms,
                    "min_time_ms": result.min_time_ms,
                    "max_time_ms": result.max_time_ms,
                    "std_dev_ms": result.std_dev_ms,
                    "operations_per_second": result.operations_per_second,
                    "queries_per_second": result.queries_per_second,
                    "documents_per_second": result.documents_per_second
                },
                "quality_metrics": {
                    "avg_confidence": result.avg_confidence,
                    "avg_relevance": result.avg_relevance,
                    "success_rate": result.success_rate
                },
                "resource_usage": {
                    "peak_memory_mb": result.peak_memory_mb,
                    "avg_cpu_percent": result.avg_cpu_percent
                },
                "component_metrics": {
                    "component_times": result.component_times,
                    "component_usage": result.component_usage
                },
                "test_parameters": result.test_parameters,
                "timestamp": result.timestamp,
                "system_info": result.system_info
            }
            
            results_dict["benchmarks"].append(result_dict)
        
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        logger.info(f"📊 Benchmark results saved to: {json_path}")
    
    def _print_benchmark_summary(self, suite: BenchmarkSuite):
        """Print comprehensive benchmark summary."""
        print("\n" + "="*80)
        print("🏆 UNIFIED RAG SYSTEM PERFORMANCE BENCHMARK RESULTS")
        print("="*80)
        
        print(f"\n📋 Test Suite: {suite.suite_name}")
        print(f"⏱️  Total Duration: {suite.total_duration_ms:.1f}ms")
        print(f"🔢 Total Operations: {suite.total_operations}")
        print(f"✅ Overall Success Rate: {suite.overall_success_rate:.1%}")
        print(f"📅 Executed At: {suite.executed_at}")
        
        print(f"\n💻 System Specifications:")
        for key, value in suite.system_specs.items():
            print(f"   {key}: {value}")
        
        print(f"\n📊 Individual Benchmark Results:")
        print("-" * 80)
        
        for result in suite.benchmark_results:
            print(f"\n🔍 {result.benchmark_name.upper()}")
            print(f"   Description: {result.test_description}")
            print(f"   Total Time: {result.total_time_ms:.1f}ms")
            print(f"   Average Time: {result.avg_time_ms:.1f}ms")
            print(f"   Operations/sec: {result.operations_per_second:.2f}")
            print(f"   Success Rate: {result.success_rate:.1%}")
            
            if result.avg_confidence > 0:
                print(f"   Avg Confidence: {result.avg_confidence:.3f}")
            
            if result.peak_memory_mb > 0:
                print(f"   Peak Memory: {result.peak_memory_mb:.1f}MB")
            
            if result.component_times:
                print(f"   Component Times: {result.component_times}")
        
        print("\n" + "="*80)
        print("🎯 PERFORMANCE SUMMARY")
        print("="*80)
        
        # Calculate overall metrics
        all_ops_per_sec = [r.operations_per_second for r in suite.benchmark_results if r.operations_per_second > 0]
        all_success_rates = [r.success_rate for r in suite.benchmark_results]
        all_confidences = [r.avg_confidence for r in suite.benchmark_results if r.avg_confidence > 0]
        
        if all_ops_per_sec:
            print(f"📈 Average Operations/Second: {np.mean(all_ops_per_sec):.2f}")
            print(f"🚀 Peak Operations/Second: {np.max(all_ops_per_sec):.2f}")
        
        print(f"✅ Average Success Rate: {np.mean(all_success_rates):.1%}")
        
        if all_confidences:
            print(f"🎯 Average Confidence: {np.mean(all_confidences):.3f}")
        
        print(f"\n💾 Results saved in: {self.output_dir}")
        print("="*80)


async def main():
    """Run the performance benchmark suite."""
    benchmarks = PerformanceBenchmarks()
    suite = await benchmarks.run_full_benchmark_suite()
    return suite


if __name__ == "__main__":
    asyncio.run(main())