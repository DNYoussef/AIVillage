#!/usr/bin/env python3
"""RAG Answer Generation System with Local LLM Integration."""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(
    0,
    str(
        Path(__file__).parent.parent
        / "src"
        / "production"
        / "rag"
        / "rag_system"
        / "core"
    ),
)

from codex_rag_integration import CODEXRAGPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGAnswerGenerator:
    """RAG-based answer generation system."""

    def __init__(self):
        self.rag_pipeline = None

        # Simple prompt templates for answer generation
        self.answer_templates = {
            "informative": (
                "Based on the following context, provide a comprehensive answer to the question.\n\n"
                "Question: {question}\n\n"
                "Context:\n{context}\n\n"
                "Answer:"
            ),
            "concise": (
                "Answer this question concisely based on the provided context:\n\n"
                "Question: {question}\n"
                "Context: {context}\n"
                "Answer:"
            ),
            "educational": (
                "As an educational assistant, explain the following question using the context provided. "
                "Make it clear and suitable for learning.\n\n"
                "Question: {question}\n\n"
                "Relevant Information:\n{context}\n\n"
                "Educational Explanation:"
            ),
        }

        # Test questions for demonstration
        self.demo_questions = [
            "What is machine learning?",
            "How does artificial intelligence work?",
            "What are the applications of quantum physics?",
            "Explain renewable energy sources",
            "What is genetic engineering?",
            "How do computers process information?",
            "What causes climate change?",
            "Explain photosynthesis in plants",
            "What are the principles of chemistry?",
            "How does space exploration work?",
        ]

    async def initialize_rag_pipeline(self) -> bool:
        """Initialize the RAG pipeline."""
        try:
            self.rag_pipeline = CODEXRAGPipeline()
            logger.info("RAG pipeline initialized for answer generation")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
            return False

    def extract_context_text(
        self, retrieval_results: List[Any], max_context_length: int = 2000
    ) -> str:
        """Extract context text from retrieval results."""
        if not retrieval_results:
            return "No relevant information found."

        context_parts = []
        total_length = 0

        for i, result in enumerate(retrieval_results):
            text = result.text.strip()
            score = result.score

            # Add source information
            source_info = f"[Source {i + 1}] (relevance: {score:.3f})"
            context_part = f"{source_info}\n{text}"

            # Check if adding this would exceed max length
            if total_length + len(context_part) > max_context_length and context_parts:
                break

            context_parts.append(context_part)
            total_length += len(context_part) + 2  # +2 for newlines

        return "\n\n".join(context_parts)

    def generate_rule_based_answer(
        self, question: str, context: str, template_type: str = "informative"
    ) -> Dict[str, Any]:
        """Generate answer using rule-based approach without LLM."""
        if not context or context.strip() == "No relevant information found.":
            return {
                "answer": f"I don't have enough information in my knowledge base to answer '{question}'. "
                f"Consider asking about topics related to science, technology, or other educational subjects.",
                "confidence": 0.1,
                "method": "no_context_fallback",
            }

        # Simple rule-based answer generation
        question_lower = question.lower()
        context_sentences = [s.strip() for s in context.split(".") if s.strip()]

        # Find most relevant sentences based on question keywords
        question_words = set(question_lower.split())
        relevant_sentences = []

        for sentence in context_sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(question_words.intersection(sentence_words))

            if overlap > 0:
                relevant_sentences.append((sentence, overlap))

        # Sort by relevance and take top sentences
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in relevant_sentences[:3]]

        if not top_sentences:
            # Fallback to first few sentences of context
            top_sentences = context_sentences[:2]

        # Generate answer based on template type
        if template_type == "concise":
            answer = ". ".join(top_sentences[:2])
        elif template_type == "educational":
            answer = (
                f"Based on the available information: {'. '.join(top_sentences)}. "
                f"This relates to your question about {question.lower()} by providing relevant context and details."
            )
        else:  # informative
            answer = f"According to the available sources, {'. '.join(top_sentences)}."

        # Clean up the answer
        answer = answer.replace("[Source ", "").replace("] (relevance:", " (relevance:")
        answer = answer.strip()

        # Calculate confidence based on context quality
        confidence = min(0.9, len(relevant_sentences) * 0.2 + 0.3)

        return {
            "answer": answer,
            "confidence": confidence,
            "method": "rule_based_extraction",
            "sources_used": len(top_sentences),
        }

    async def answer_question(
        self,
        question: str,
        template_type: str = "informative",
        max_retrieval_results: int = 5,
        max_context_length: int = 2000,
    ) -> Dict[str, Any]:
        """Answer a question using RAG pipeline."""
        if not self.rag_pipeline:
            raise RuntimeError("RAG pipeline not initialized")

        start_time = time.perf_counter()

        try:
            # Retrieve relevant documents
            retrieval_results, retrieval_metrics = await self.rag_pipeline.retrieve(
                question, k=max_retrieval_results, use_cache=True
            )

            # Extract context
            context = self.extract_context_text(retrieval_results, max_context_length)

            # Generate answer using rule-based approach
            answer_result = self.generate_rule_based_answer(
                question, context, template_type
            )

            end_time = time.perf_counter()
            total_latency = (end_time - start_time) * 1000

            return {
                "question": question,
                "answer": answer_result["answer"],
                "confidence": answer_result["confidence"],
                "generation_method": answer_result["method"],
                "template_type": template_type,
                "sources_count": len(retrieval_results),
                "context_length": len(context),
                "retrieval_metrics": retrieval_metrics,
                "total_latency_ms": total_latency,
                "retrieval_latency_ms": retrieval_metrics.get("latency_ms", 0),
                "sources_used": answer_result.get("sources_used", 0),
            }

        except Exception as e:
            logger.error(f"Failed to answer question '{question}': {e}")
            end_time = time.perf_counter()

            return {
                "question": question,
                "answer": f"I apologize, but I encountered an error while trying to answer your question: {str(e)}",
                "confidence": 0.0,
                "generation_method": "error_fallback",
                "template_type": template_type,
                "sources_count": 0,
                "total_latency_ms": (end_time - start_time) * 1000,
                "error": str(e),
            }

    async def run_demonstration(
        self, questions: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Run demonstration of answer generation."""
        questions = questions or self.demo_questions[:5]  # Use first 5 demo questions

        logger.info(
            f"Running answer generation demonstration with {len(questions)} questions"
        )

        results = []

        for i, question in enumerate(questions):
            logger.info(f"Processing question {i + 1}/{len(questions)}: {question}")

            # Try different template types for variety
            template_types = ["informative", "concise", "educational"]
            template_type = template_types[i % len(template_types)]

            result = await self.answer_question(question, template_type=template_type)
            results.append(result)

            # Log result
            confidence_pct = result["confidence"] * 100
            logger.info(
                f"Answer generated (confidence: {confidence_pct:.1f}%, "
                f"latency: {result['total_latency_ms']:.2f}ms)"
            )

            # Small delay to avoid overwhelming the system
            await asyncio.sleep(0.5)

        return results

    def print_result(self, result: Dict[str, Any]) -> None:
        """Print a formatted result."""
        print(f"\n{'=' * 60}")
        print(f"Question: {result['question']}")
        print(f"{'=' * 60}")
        print(f"Answer: {result['answer']}")
        print(f"\nMetrics:")
        print(f"  Confidence: {result['confidence'] * 100:.1f}%")
        print(f"  Sources Used: {result['sources_count']}")
        print(f"  Total Latency: {result['total_latency_ms']:.2f}ms")
        print(f"  Method: {result['generation_method']}")
        print(f"  Template: {result['template_type']}")


async def main():
    """Main function to run answer generation demonstration."""
    try:
        generator = RAGAnswerGenerator()

        # Initialize RAG pipeline
        if not await generator.initialize_rag_pipeline():
            raise RuntimeError("Failed to initialize RAG pipeline")

        # Get pipeline state
        performance_metrics = generator.rag_pipeline.get_performance_metrics()
        index_size = performance_metrics.get("index_size", 0)
        print(f"RAG Pipeline initialized with {index_size} indexed documents")

        # Run demonstration
        results = await generator.run_demonstration()

        # Display results
        print(f"\n{'=' * 80}")
        print(f" RAG ANSWER GENERATION DEMONSTRATION RESULTS")
        print(f"{'=' * 80}")

        for result in results:
            generator.print_result(result)

        # Summary statistics
        avg_confidence = sum(r["confidence"] for r in results) / len(results)
        avg_latency = sum(r["total_latency_ms"] for r in results) / len(results)
        successful_answers = sum(1 for r in results if r["confidence"] > 0.3)

        print(f"\n{'=' * 80}")
        print(f" SUMMARY STATISTICS")
        print(f"{'=' * 80}")
        print(f"Total Questions: {len(results)}")
        print(
            f"Successful Answers: {successful_answers} ({successful_answers / len(results) * 100:.1f}%)"
        )
        print(f"Average Confidence: {avg_confidence * 100:.1f}%")
        print(f"Average Latency: {avg_latency:.2f}ms")
        print(f"Index Size: {index_size} documents")

        # Save results
        results_file = Path("data/answer_generation_results.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "results": results,
                    "summary": {
                        "total_questions": len(results),
                        "successful_answers": successful_answers,
                        "success_rate": successful_answers / len(results),
                        "avg_confidence": avg_confidence,
                        "avg_latency_ms": avg_latency,
                        "index_size": index_size,
                    },
                },
                f,
                indent=2,
                default=str,
            )

        print(f"\nResults saved to: {results_file}")

        return results

    except Exception as e:
        logger.error(f"Answer generation demonstration failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
