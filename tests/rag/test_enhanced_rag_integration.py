"""
Comprehensive Integration Test for Enhanced RAG System.

Tests the complete integration of:
- Intelligent chunking with sliding window similarity analysis
- Enhanced CODEX RAG pipeline with content analysis
- Performance improvements and success rate validation

Critical Test: Validate improved question success rate from 57% to >80%
"""

import asyncio
import json
import logging
import time
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Import our enhanced RAG components
import sys

sys.path.append("src/production/rag/rag_system/core")

from codex_rag_integration import Document
from enhanced_codex_rag import EnhancedCODEXRAGPipeline


class RAGIntegrationTester:
    """Comprehensive tester for enhanced RAG system."""

    def __init__(self):
        self.pipeline = None
        self.test_documents = []
        self.test_questions = []
        self.results = {}

    def setup_test_documents(self):
        """Create comprehensive test documents covering multiple topics."""

        self.test_documents = [
            Document(
                id="ai_overview",
                title="Artificial Intelligence Overview",
                content="""
                Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving.
                
                Machine learning is a subset of AI that focuses on the use of data and algorithms to imitate the way that humans learn, gradually improving accuracy. Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers to model and understand complex patterns in data.
                
                Natural language processing (NLP) is a branch of AI that helps computers understand, interpret, and manipulate human language. NLP combines computational linguistics with statistical, machine learning, and deep learning models to enable computers to process human language in the form of text or voice data.
                
                Computer vision is another important field within AI that enables computers and systems to derive meaningful information from digital images, videos, and other visual inputs. It uses deep learning models, particularly convolutional neural networks, to analyze and understand visual content.
                
                The applications of AI are vast and growing, including healthcare diagnostics, autonomous vehicles, recommendation systems, financial fraud detection, virtual assistants, and scientific research acceleration.
                """,
                source_type="educational",
                metadata={
                    "domain": "technology",
                    "complexity": "intermediate",
                    "topics": ["AI", "ML", "NLP", "computer vision"],
                },
            ),
            Document(
                id="climate_science",
                title="Climate Change and Environmental Impact",
                content="""
                Climate change refers to long-term shifts in global temperatures and weather patterns. While climate variations are natural, scientific evidence shows that human activities have been the dominant driver of climate change since the 1950s.
                
                The primary cause of recent climate change is the increase in greenhouse gas concentrations in Earth's atmosphere. Carbon dioxide (CO2) is the most significant greenhouse gas, primarily released through burning fossil fuels like coal, oil, and natural gas. Other important greenhouse gases include methane (CH4), nitrous oxide (N2O), and fluorinated gases.
                
                The effects of climate change are already visible and include rising global temperatures, melting ice sheets and glaciers, rising sea levels, changing precipitation patterns, and more frequent extreme weather events such as hurricanes, droughts, and heatwaves.
                
                Climate change impacts ecosystems, biodiversity, agriculture, water resources, and human health. Many species are being forced to migrate or face extinction as their habitats change. Agricultural productivity is affected by changing temperature and precipitation patterns.
                
                Mitigation strategies include transitioning to renewable energy sources like solar and wind power, improving energy efficiency, protecting and restoring forests, developing carbon capture technologies, and implementing policies to reduce greenhouse gas emissions.
                """,
                source_type="educational",
                metadata={
                    "domain": "environment",
                    "complexity": "intermediate",
                    "topics": ["climate", "environment", "sustainability"],
                },
            ),
            Document(
                id="quantum_computing",
                title="Quantum Computing Fundamentals",
                content="""
                Quantum computing is a revolutionary computing paradigm that leverages the principles of quantum mechanics to process information in fundamentally different ways than classical computers. While classical computers use bits that exist in either 0 or 1 states, quantum computers use quantum bits or qubits that can exist in superposition of both states simultaneously.
                
                The key principles of quantum computing include superposition, entanglement, and quantum interference. Superposition allows qubits to be in multiple states at once, enabling quantum computers to explore many possible solutions simultaneously. Entanglement creates correlations between qubits that don't exist in classical systems.
                
                Quantum algorithms like Shor's algorithm for integer factorization and Grover's algorithm for database searching demonstrate potential exponential speedups over classical algorithms for specific problems. These algorithms could revolutionize cryptography, optimization, and search problems.
                
                Current quantum computers are still in early stages and face significant challenges including quantum decoherence, error rates, and the need for extremely low temperatures. However, companies like IBM, Google, and startups are making rapid progress in building more stable and powerful quantum systems.
                
                Potential applications include drug discovery through molecular simulation, financial portfolio optimization, artificial intelligence acceleration, cryptography, and solving complex optimization problems in logistics and supply chain management.
                """,
                source_type="educational",
                metadata={
                    "domain": "technology",
                    "complexity": "advanced",
                    "topics": ["quantum", "computing", "algorithms"],
                },
            ),
            Document(
                id="renewable_energy",
                title="Renewable Energy Technologies",
                content="""
                Renewable energy comes from natural resources that are replenished constantly, including sunlight, wind, water, and geothermal heat. These energy sources are essential for reducing greenhouse gas emissions and combating climate change.
                
                Solar energy harnesses sunlight through photovoltaic cells or solar thermal systems. Photovoltaic cells convert sunlight directly into electricity, while solar thermal systems use sunlight to heat water or air. Solar energy has become increasingly cost-effective and is now competitive with fossil fuels in many regions.
                
                Wind energy uses wind turbines to convert kinetic energy from moving air into electrical energy. Modern wind turbines are highly efficient and can generate electricity at very low costs. Both onshore and offshore wind farms are being developed worldwide.
                
                Hydroelectric power generates electricity by harnessing the energy of flowing water. Large-scale hydroelectric dams and smaller run-of-river systems provide clean electricity while also offering water storage and flood control benefits.
                
                Energy storage technologies like batteries, pumped hydro storage, and compressed air energy storage are crucial for managing the intermittent nature of renewable energy sources and ensuring grid stability.
                """,
                source_type="educational",
                metadata={
                    "domain": "energy",
                    "complexity": "intermediate",
                    "topics": ["renewable", "solar", "wind", "storage"],
                },
            ),
        ]

    def setup_test_questions(self):
        """Create comprehensive test questions with expected answer topics."""

        self.test_questions = [
            # AI and Technology Questions
            {
                "question": "What is artificial intelligence and how does it work?",
                "expected_topics": [
                    "AI",
                    "machine learning",
                    "algorithms",
                    "human intelligence",
                ],
                "expected_document": "ai_overview",
            },
            {
                "question": "What are the main applications of machine learning?",
                "expected_topics": [
                    "machine learning",
                    "applications",
                    "data",
                    "algorithms",
                ],
                "expected_document": "ai_overview",
            },
            {
                "question": "How does natural language processing help computers understand text?",
                "expected_topics": [
                    "NLP",
                    "natural language processing",
                    "text",
                    "computers",
                ],
                "expected_document": "ai_overview",
            },
            {
                "question": "What is computer vision and how does it analyze images?",
                "expected_topics": [
                    "computer vision",
                    "images",
                    "visual",
                    "neural networks",
                ],
                "expected_document": "ai_overview",
            },
            # Climate and Environment Questions
            {
                "question": "What causes climate change and global warming?",
                "expected_topics": [
                    "climate change",
                    "greenhouse gas",
                    "CO2",
                    "fossil fuels",
                ],
                "expected_document": "climate_science",
            },
            {
                "question": "What are the main effects of climate change on the environment?",
                "expected_topics": [
                    "temperature",
                    "sea levels",
                    "weather",
                    "ecosystems",
                ],
                "expected_document": "climate_science",
            },
            {
                "question": "How can we reduce greenhouse gas emissions?",
                "expected_topics": [
                    "renewable energy",
                    "mitigation",
                    "emissions",
                    "solar",
                    "wind",
                ],
                "expected_document": "climate_science",
            },
            # Quantum Computing Questions
            {
                "question": "How do quantum computers differ from classical computers?",
                "expected_topics": [
                    "quantum",
                    "qubits",
                    "superposition",
                    "classical",
                    "bits",
                ],
                "expected_document": "quantum_computing",
            },
            {
                "question": "What are the key principles of quantum mechanics used in quantum computing?",
                "expected_topics": [
                    "superposition",
                    "entanglement",
                    "quantum interference",
                ],
                "expected_document": "quantum_computing",
            },
            {
                "question": "What are some potential applications of quantum computing?",
                "expected_topics": [
                    "drug discovery",
                    "optimization",
                    "cryptography",
                    "AI",
                ],
                "expected_document": "quantum_computing",
            },
            # Renewable Energy Questions
            {
                "question": "What are the main types of renewable energy sources?",
                "expected_topics": ["solar", "wind", "hydroelectric", "renewable"],
                "expected_document": "renewable_energy",
            },
            {
                "question": "How does solar energy work and why is it cost-effective?",
                "expected_topics": ["solar", "photovoltaic", "sunlight", "electricity"],
                "expected_document": "renewable_energy",
            },
            {
                "question": "Why is energy storage important for renewable energy?",
                "expected_topics": [
                    "storage",
                    "batteries",
                    "intermittent",
                    "grid stability",
                ],
                "expected_document": "renewable_energy",
            },
            # Cross-domain Questions
            {
                "question": "How can AI help address climate change?",
                "expected_topics": [
                    "AI",
                    "climate",
                    "optimization",
                    "renewable energy",
                ],
                "expected_document": ["ai_overview", "climate_science"],
            },
            {
                "question": "What role could quantum computing play in renewable energy optimization?",
                "expected_topics": ["quantum", "optimization", "energy", "algorithms"],
                "expected_document": ["quantum_computing", "renewable_energy"],
            },
        ]

    async def run_comprehensive_test(self):
        """Run comprehensive integration test."""

        print("Enhanced RAG System Integration Test")
        print("=" * 60)

        # Initialize enhanced pipeline
        print("\n[INIT] Initializing Enhanced RAG Pipeline...")
        self.pipeline = EnhancedCODEXRAGPipeline(
            enable_intelligent_chunking=True,
            chunking_window_size=3,
            chunking_min_sentences=2,
            chunking_max_sentences=15,
            chunking_context_overlap=1,
        )

        # Setup test data
        self.setup_test_documents()
        self.setup_test_questions()

        # Analyze document structure
        print("\n[ANALYZE] Analyzing Document Structure...")
        for doc in self.test_documents:
            analysis = self.pipeline.analyze_document_structure(doc)
            print(f"DOC {doc.title}:")
            print(f"  - Document Type: {analysis.get('document_type', 'unknown')}")
            print(f"  - Sentences: {analysis.get('total_sentences', 0)}")
            print(f"  - Estimated Chunks: {analysis.get('estimated_chunks', 0)}")
            print(f"  - Content Types: {analysis.get('content_type_distribution', {})}")

        # Index documents
        print(f"\n[INDEX] Indexing {len(self.test_documents)} documents...")
        start_time = time.perf_counter()
        stats = self.pipeline.index_documents(self.test_documents)
        indexing_time = time.perf_counter() - start_time

        print(f"[SUCCESS] Indexing Complete in {indexing_time:.2f}s:")
        print(f"  - Documents: {stats['documents_processed']}")
        print(f"  - Chunks: {stats['chunks_created']}")
        print(f"  - Intelligent Chunks: {stats.get('intelligent_chunks_created', 0)}")
        print(f"  - Average Coherence: {stats.get('avg_chunk_coherence', 0):.3f}")
        print(f"  - Entities Extracted: {stats.get('total_entities_extracted', 0)}")
        print(f"  - Content Types: {stats.get('content_type_distribution', {})}")

        # Test retrieval quality
        print(f"\n[TEST] Testing Retrieval Quality with {len(self.test_questions)} questions...")
        print("-" * 60)

        correct_answers = 0
        total_questions = len(self.test_questions)
        retrieval_times = []
        detailed_results = []

        for i, test_case in enumerate(self.test_questions, 1):
            question = test_case["question"]
            expected_topics = test_case["expected_topics"]
            expected_doc = test_case["expected_document"]

            print(f"\nQ{i}: {question}")

            # Enhanced retrieval with content analysis
            start_time = time.perf_counter()
            results, metrics = await self.pipeline.retrieve_with_content_analysis(
                query=question, k=5, include_entities=True, min_coherence=0.0
            )
            retrieval_time = (time.perf_counter() - start_time) * 1000
            retrieval_times.append(retrieval_time)

            # Evaluate answer quality
            answer_quality = self._evaluate_answer_quality(question, results, expected_topics, expected_doc)

            if answer_quality["is_correct"]:
                correct_answers += 1
                status = "[CORRECT]"
            else:
                status = "[INCORRECT]"

            print(f"  {status} - Score: {answer_quality['confidence_score']:.3f}")
            print(f"  Latency: {retrieval_time:.1f}ms")
            print(f"  Results: {len(results)}")

            if results:
                best_result = results[0]
                print(f"  Best Match: {best_result.document_id}")
                print(f"  Content Type: {best_result.metadata.get('chunk_type', 'unknown')}")
                print(f"  Coherence: {best_result.metadata.get('topic_coherence', 0):.3f}")
                print(f"  Entities: {best_result.metadata.get('entities', [])}")
                print(f"  Text: {best_result.text[:150]}...")

            detailed_results.append(
                {
                    "question": question,
                    "expected_topics": expected_topics,
                    "expected_document": expected_doc,
                    "answer_quality": answer_quality,
                    "retrieval_time_ms": retrieval_time,
                    "num_results": len(results),
                    "metrics": metrics,
                }
            )

        # Calculate final metrics
        success_rate = (correct_answers / total_questions) * 100
        avg_latency = sum(retrieval_times) / len(retrieval_times)

        print("\n[RESULTS] Final Results Summary")
        print("=" * 60)
        print(f"Success Rate: {success_rate:.1f}% ({correct_answers}/{total_questions})")
        print(f"Average Latency: {avg_latency:.1f}ms")
        print(f"Performance Target (<100ms): {'PASS' if avg_latency < 100 else 'FAIL'}")

        # Performance metrics
        perf_metrics = self.pipeline.get_enhanced_performance_metrics()
        print("\n[METRICS] Enhanced Performance Metrics:")
        print(f"  - Chunking Method: {perf_metrics['chunking_method']}")
        print(f"  - Cache Hit Rate: {perf_metrics['cache_metrics']['hit_rate']:.2%}")
        print(f"  - Index Size: {perf_metrics['index_size']} vectors")
        print(f"  - Average Chunk Coherence: {perf_metrics['chunking_quality']['avg_coherence']:.3f}")
        print(f"  - Total Entities: {perf_metrics['chunking_quality']['total_entities']}")
        print(f"  - Content Diversity: {perf_metrics['chunking_quality']['content_diversity']} types")

        # Overall assessment
        print("\n[ASSESSMENT] Overall Assessment:")
        if success_rate >= 80 and avg_latency < 100 and perf_metrics["chunking_quality"]["avg_coherence"] > 0.6:
            print("EXCELLENT: Enhanced RAG system performing above targets!")
            print("  - Success rate >=80% target achieved")
            print("  - Latency <100ms target met")
            print("  - Intelligent chunking creating high-quality boundaries")
            print("  - Content analysis providing rich metadata")
        elif success_rate >= 70:
            print("GOOD: Enhanced RAG system functional but needs optimization")
            print(f"  - Success rate {success_rate:.1f}% (target: >=80%)")
            if avg_latency >= 100:
                print(f"  - Latency {avg_latency:.1f}ms exceeds 100ms target")
        else:
            print("NEEDS IMPROVEMENT: Enhanced RAG system requires fixes")
            print(f"  - Success rate {success_rate:.1f}% below 70% threshold")

        # Save detailed results
        results_file = "test_results_enhanced_rag.json"
        with open(results_file, "w") as f:
            json.dump(
                {
                    "summary": {
                        "success_rate": success_rate,
                        "correct_answers": correct_answers,
                        "total_questions": total_questions,
                        "avg_latency_ms": avg_latency,
                        "performance_target_met": avg_latency < 100,
                        "indexing_stats": stats,
                        "performance_metrics": perf_metrics,
                    },
                    "detailed_results": detailed_results,
                    "test_timestamp": time.time(),
                },
                f,
                indent=2,
            )

        print(f"\n[SAVE] Detailed results saved to: {results_file}")

        return {
            "success_rate": success_rate,
            "avg_latency_ms": avg_latency,
            "meets_targets": success_rate >= 80 and avg_latency < 100,
            "performance_metrics": perf_metrics,
        }

    def _evaluate_answer_quality(
        self, question: str, results: list, expected_topics: list[str], expected_doc
    ) -> dict[str, Any]:
        """Evaluate the quality of retrieval results."""

        if not results:
            return {
                "is_correct": False,
                "confidence_score": 0.0,
                "topic_matches": 0,
                "document_match": False,
                "explanation": "No results returned",
            }

        best_result = results[0]
        result_text = best_result.text.lower()

        # Check topic coverage
        topic_matches = 0
        for topic in expected_topics:
            if topic.lower() in result_text:
                topic_matches += 1

        topic_coverage = topic_matches / len(expected_topics)

        # Check document relevance
        document_match = False
        if isinstance(expected_doc, list):
            document_match = best_result.document_id in expected_doc
        else:
            document_match = best_result.document_id == expected_doc

        # Calculate confidence score
        confidence_score = (topic_coverage * 0.7) + (float(document_match) * 0.3)

        # Determine correctness (threshold: 0.5)
        is_correct = confidence_score >= 0.5

        return {
            "is_correct": is_correct,
            "confidence_score": confidence_score,
            "topic_matches": topic_matches,
            "topic_coverage": topic_coverage,
            "document_match": document_match,
            "explanation": f"Topics: {topic_matches}/{len(expected_topics)}, Doc match: {document_match}",
        }


async def main():
    """Run the comprehensive integration test."""

    tester = RAGIntegrationTester()
    results = await tester.run_comprehensive_test()

    # Exit with appropriate code
    if results["meets_targets"]:
        print("\nSUCCESS: Enhanced RAG system meets all targets!")
        exit(0)
    else:
        print("\nPARTIAL SUCCESS: Some targets not met, see details above")
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
