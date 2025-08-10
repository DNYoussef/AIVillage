#!/usr/bin/env python3
"""
Simple test of RAG system with real questions (no unicode).
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path('src/production/rag/rag_system/core')))

async def test_questions():
    """Test RAG with simple questions."""
    
    from bayesrag_codex_enhanced import BayesRAGEnhancedPipeline
    from codex_rag_integration import Document
    
    pipeline = BayesRAGEnhancedPipeline()
    
    # Add one test document
    doc = Document(
        id="ai_test",
        title="Artificial Intelligence Basics",
        content="Artificial intelligence (AI) is a branch of computer science that creates intelligent machines. Machine learning is a subset of AI that enables systems to learn from data. Deep learning uses neural networks with multiple layers. AI applications include computer vision, natural language processing, and robotics.",
        source_type="test"
    )
    
    pipeline.index_documents([doc])
    
    # Test questions
    questions = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What are AI applications?", 
        "What is deep learning?",
        "How do neural networks function?"
    ]
    
    results_summary = []
    
    for i, question in enumerate(questions, 1):
        print(f"Question {i}: {question}")
        
        start = time.perf_counter()
        results, metrics = await pipeline.retrieve_with_trust(question, k=2)
        latency = (time.perf_counter() - start) * 1000
        
        if results:
            best = results[0]
            answer = best.text[:200] + "..." if len(best.text) > 200 else best.text
            
            # Check relevance
            q_words = set(question.lower().split())
            a_words = set(answer.lower().split())
            common = {'what', 'is', 'how', 'do', 'does', 'the', 'a', 'an'}
            
            relevant_q = q_words - common
            relevant_a = a_words - common
            overlap = len(relevant_q & relevant_a)
            
            print(f"  Latency: {latency:.1f}ms")
            print(f"  Results: {len(results)}")
            print(f"  Answer: {answer}")
            print(f"  Relevance: {overlap}/{len(relevant_q)} words match")
            
            if hasattr(best, 'trust_metrics') and best.trust_metrics:
                print(f"  Trust: {best.trust_metrics.trust_score:.3f}")
            if hasattr(best, 'bayesian_score'):
                print(f"  Bayesian: {best.bayesian_score:.3f}")
                
            results_summary.append({
                'question': question,
                'got_results': True,
                'latency': latency,
                'relevance': overlap / len(relevant_q) if relevant_q else 0
            })
        else:
            print("  NO RESULTS")
            results_summary.append({
                'question': question, 
                'got_results': False,
                'latency': latency,
                'relevance': 0
            })
        print()
        
    # Summary
    print("SUMMARY:")
    total = len(results_summary)
    with_results = sum(1 for r in results_summary if r['got_results'])
    avg_latency = sum(r['latency'] for r in results_summary) / total
    avg_relevance = sum(r['relevance'] for r in results_summary if r['got_results'])
    avg_relevance = avg_relevance / with_results if with_results > 0 else 0
    
    print(f"Questions answered: {with_results}/{total}")
    print(f"Average latency: {avg_latency:.1f}ms")
    print(f"Average relevance: {avg_relevance:.2f}")
    
    success = with_results >= total * 0.8  # 80% success rate
    
    if success:
        print("VERDICT: RAG system can answer questions!")
    else:
        print("VERDICT: RAG system struggles with questions")
        
    return success

if __name__ == "__main__":
    success = asyncio.run(test_questions())
    print(f"Test result: {'PASS' if success else 'FAIL'}")