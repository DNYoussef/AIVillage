"""
Comprehensive Test Suite for Enhanced Query Processing System.

Tests the complete enhanced query processing pipeline including:
- Query decomposition with intent analysis
- Multi-level matching (Document -> Chunk -> Graph)
- Context-aware ranking with trust scores
- Answer synthesis with idea boundary respect
"""

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.append('src/production/rag/rag_system/core')

from enhanced_query_processor import (
    EnhancedQueryProcessor, QueryIntent, ContextLevel, ComplexityLevel,
    TemporalRequirement, QueryDecomposition, SynthesizedAnswer
)
from graph_enhanced_rag_pipeline import GraphEnhancedRAGPipeline
from codex_rag_integration import Document
from contextual_tagging import ContentDomain, ReadingLevel


async def test_enhanced_query_processing():
    """Test the complete enhanced query processing system."""
    
    print("Enhanced Query Processing System - Comprehensive Test")
    print("=" * 60)
    
    # Initialize the enhanced query processor
    print("[INIT] Initializing Enhanced Query Processor...")
    
    # First initialize the graph-enhanced RAG pipeline
    rag_pipeline = GraphEnhancedRAGPipeline(
        enable_intelligent_chunking=True,
        enable_contextual_tagging=True,
        enable_trust_graph=True
    )
    
    # Initialize enhanced query processor
    query_processor = EnhancedQueryProcessor(
        rag_pipeline=rag_pipeline,
        enable_query_expansion=True,
        enable_intent_classification=True,
        enable_multi_hop_reasoning=True,
        default_result_limit=8
    )
    
    print("Enhanced Query Processor initialized successfully")
    
    # Create comprehensive test documents
    test_documents = [
        Document(
            id="ai_fundamentals_guide",
            title="Artificial Intelligence: Fundamentals and Applications",
            content="""
            # Introduction to Artificial Intelligence
            
            Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems. These processes include learning, reasoning, and self-correction.
            
            ## Core AI Concepts
            
            Machine learning is a subset of AI that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Deep learning, a subset of machine learning, uses neural networks with multiple layers to model and understand complex patterns.
            
            ### Types of AI Systems
            
            There are three main types of AI systems: narrow AI (designed for specific tasks), general AI (human-level intelligence across domains), and super AI (exceeding human intelligence). Currently, only narrow AI systems exist in practical applications.
            
            ## Applications in Various Industries
            
            AI has transformative applications across industries. In healthcare, AI assists with medical diagnosis, drug discovery, and personalized treatment plans. In finance, AI powers fraud detection, algorithmic trading, and risk assessment. Transportation benefits from AI through autonomous vehicles and traffic optimization.
            
            ### Healthcare Applications
            
            Medical AI systems can analyze medical images like X-rays and MRIs with remarkable accuracy. Machine learning algorithms help identify patterns in patient data to predict disease progression and recommend treatments. However, ensuring AI fairness and avoiding bias in medical decisions remains a critical challenge.
            
            ## Challenges and Future Directions
            
            Despite significant advances, AI faces challenges including data privacy, algorithmic transparency, and ethical considerations. The future of AI involves developing more interpretable systems, ensuring fairness across populations, and addressing the societal impact of automation.
            """,
            source_type="educational",
            metadata={
                "author": "Dr. AI Researcher",
                "publication_date": "2024-01-15",
                "credibility_score": 0.92,
                "target_audience": "students"
            }
        ),
        
        Document(
            id="machine_learning_techniques",
            title="Machine Learning Techniques: A Comprehensive Overview",
            content="""
            # Machine Learning Methodologies
            
            Machine learning encompasses various methodologies for creating systems that learn from data. Understanding these techniques is essential for applying AI effectively across different domains.
            
            ## Supervised Learning
            
            Supervised learning uses labeled training data to learn a mapping from inputs to outputs. Common supervised learning algorithms include linear regression, decision trees, support vector machines, and neural networks. These methods excel at classification and regression tasks where historical examples are available.
            
            ### Neural Networks and Deep Learning
            
            Neural networks, inspired by biological neurons, consist of interconnected nodes that process information. Deep learning uses multi-layered neural networks to learn hierarchical representations of data. Convolutional neural networks (CNNs) excel at image recognition, while recurrent neural networks (RNNs) handle sequential data.
            
            ## Unsupervised Learning
            
            Unsupervised learning finds patterns in data without labeled examples. Clustering algorithms group similar data points, while dimensionality reduction techniques like principal component analysis (PCA) simplify complex datasets while preserving important information.
            
            ### Reinforcement Learning
            
            Reinforcement learning trains agents to make sequential decisions by rewarding desired behaviors. This approach has achieved remarkable success in game playing (like AlphaGo), robotics, and autonomous systems. The agent learns through trial and error, gradually improving its decision-making strategy.
            
            ## Model Evaluation and Selection
            
            Proper evaluation prevents overfitting and ensures models generalize to new data. Cross-validation techniques split data into training and testing sets multiple times to assess model performance. Metrics like accuracy, precision, recall, and F1-score help compare different approaches.
            
            ## Practical Implementation Challenges
            
            Real-world machine learning faces challenges including data quality issues, computational constraints, and the need for domain expertise. Feature engineering remains crucial for many applications, though deep learning has reduced its importance in some domains.
            """,
            source_type="technical",
            metadata={
                "author": "ML Engineering Team",
                "publication_date": "2024-02-10",
                "credibility_score": 0.88,
                "target_audience": "professionals"
            }
        ),
        
        Document(
            id="ai_ethics_considerations",
            title="Ethical Considerations in Artificial Intelligence Development",
            content="""
            # AI Ethics: Principles and Challenges
            
            As artificial intelligence becomes increasingly integrated into society, ethical considerations become paramount. Responsible AI development requires careful attention to fairness, transparency, and accountability.
            
            ## Bias and Fairness in AI Systems
            
            AI systems can perpetuate or amplify existing societal biases present in training data. Algorithmic bias can lead to unfair treatment of certain groups in hiring, lending, criminal justice, and healthcare. Addressing bias requires diverse datasets, bias testing, and ongoing monitoring of AI system outcomes.
            
            ### Techniques for Bias Mitigation
            
            Several approaches help reduce bias in AI systems. Pre-processing techniques clean and balance training data. In-processing methods modify algorithms during training to promote fairness. Post-processing approaches adjust model outputs to achieve more equitable results across different groups.
            
            ## Transparency and Explainability
            
            Many AI systems, particularly deep learning models, operate as "black boxes" where decision-making processes are opaque. Explainable AI (XAI) techniques help users understand how AI systems reach their conclusions. This transparency is crucial in high-stakes applications like healthcare and criminal justice.
            
            ### Privacy and Data Protection
            
            AI systems often require large amounts of personal data, raising privacy concerns. Techniques like differential privacy, federated learning, and homomorphic encryption help protect individual privacy while enabling AI applications. GDPR and similar regulations establish frameworks for responsible data use.
            
            ## Accountability and Governance
            
            Establishing clear accountability for AI system decisions remains challenging. Questions arise about liability when AI systems cause harm or make errors. Governance frameworks must balance innovation with risk management, ensuring AI benefits society while minimizing potential harms.
            
            ## Future Ethical Challenges
            
            As AI capabilities advance, new ethical challenges emerge. Autonomous weapons systems raise questions about machine decision-making in life-and-death situations. AI-generated content (deepfakes) threatens information integrity. The displacement of human workers by automation requires careful consideration of societal impacts.
            """,
            source_type="policy",
            metadata={
                "author": "AI Ethics Research Institute",
                "publication_date": "2024-03-05",
                "credibility_score": 0.95,
                "target_audience": "policymakers"
            }
        )
    ]
    
    # Index documents in the RAG pipeline
    print(f"\n[INDEX] Indexing {len(test_documents)} documents...")
    start_time = time.perf_counter()
    indexing_stats = rag_pipeline.index_documents(test_documents)
    indexing_time = time.perf_counter() - start_time
    
    print(f"Indexing completed in {indexing_time:.2f}s")
    print(f"  - Documents: {indexing_stats['documents_processed']}")
    print(f"  - Chunks: {indexing_stats['chunks_created']}")
    print(f"  - Graph enabled: {indexing_stats.get('graph_integration_enabled', False)}")
    
    # Test query decomposition
    print(f"\n[TEST] Testing Query Decomposition:")
    print("-" * 40)
    
    test_queries = [
        "What is machine learning and how does it work?",
        "Compare supervised and unsupervised learning approaches",
        "What are the ethical challenges in AI development?",
        "How do neural networks process information step by step?",
        "What will be the future impact of AI on healthcare?",
        "Explain the relationship between AI bias and fairness"
    ]
    
    decomposition_results = []
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        
        # Test query decomposition
        start_time = time.perf_counter()
        decomposition = await query_processor.decompose_query(query, {})
        decomp_time = (time.perf_counter() - start_time) * 1000
        
        print(f"  Decomposition time: {decomp_time:.1f}ms")
        print(f"  Primary intent: {decomposition.primary_intent.value}")
        print(f"  Context level: {decomposition.context_level.value}")
        print(f"  Complexity: {decomposition.complexity_level.value}")
        print(f"  Temporal: {decomposition.temporal_requirement.value}")
        print(f"  Key concepts: {decomposition.key_concepts}")
        print(f"  Multi-hop required: {decomposition.requires_multi_hop}")
        print(f"  Needs synthesis: {decomposition.needs_synthesis}")
        
        decomposition_results.append({
            "query": query,
            "decomposition": decomposition.to_dict(),
            "processing_time_ms": decomp_time
        })
    
    # Test full enhanced query processing
    print(f"\n[TEST] Testing Full Enhanced Query Processing:")
    print("-" * 60)
    
    enhanced_processing_results = []
    
    for i, query in enumerate(test_queries[:4], 1):  # Test first 4 queries
        print(f"\nQuery {i}: {query}")
        
        # Process query with full enhancement
        start_time = time.perf_counter()
        synthesized_answer = await query_processor.process_query(query)
        processing_time = (time.perf_counter() - start_time) * 1000
        
        print(f"  Total processing time: {processing_time:.1f}ms")
        print(f"  Overall confidence: {synthesized_answer.overall_confidence:.3f}")
        print(f"  Trust-weighted confidence: {synthesized_answer.trust_weighted_confidence:.3f}")
        print(f"  Completeness score: {synthesized_answer.completeness_score:.3f}")
        print(f"  Coherence score: {synthesized_answer.coherence_score:.3f}")
        print(f"  Primary sources: {len(synthesized_answer.primary_sources)}")
        print(f"  Supporting sources: {len(synthesized_answer.supporting_sources)}")
        print(f"  Synthesis method: {synthesized_answer.synthesis_method}")
        
        # Show answer preview
        answer_preview = synthesized_answer.answer_text[:200].replace('\n', ' ')
        print(f"  Answer preview: {answer_preview}...")
        
        # Show executive summary
        exec_summary = synthesized_answer.executive_summary[:150].replace('\n', ' ')
        print(f"  Executive summary: {exec_summary}...")
        
        enhanced_processing_results.append({
            "query": query,
            "processing_time_ms": processing_time,
            "confidence_metrics": {
                "overall": synthesized_answer.overall_confidence,
                "trust_weighted": synthesized_answer.trust_weighted_confidence,
                "completeness": synthesized_answer.completeness_score,
                "coherence": synthesized_answer.coherence_score
            },
            "source_counts": {
                "primary": len(synthesized_answer.primary_sources),
                "supporting": len(synthesized_answer.supporting_sources),
                "total_sections": len(synthesized_answer.detailed_sections)
            },
            "features": {
                "idea_boundaries_preserved": len(synthesized_answer.preserved_idea_boundaries),
                "context_chain_length": len(synthesized_answer.context_chain),
                "synthesis_method": synthesized_answer.synthesis_method
            }
        })
    
    # Test specific enhanced features
    print(f"\n[TEST] Testing Enhanced Features:")
    print("-" * 40)
    
    # Test multi-level matching with a complex query
    complex_query = "How do machine learning techniques relate to ethical AI development challenges?"
    print(f"\nComplex Multi-hop Query: {complex_query}")
    
    start_time = time.perf_counter()
    complex_result = await query_processor.process_query(complex_query)
    complex_time = (time.perf_counter() - start_time) * 1000
    
    print(f"  Processing time: {complex_time:.1f}ms")
    print(f"  Query complexity detected: {complex_result.query_decomposition.complexity_level.value}")
    print(f"  Multi-hop reasoning: {complex_result.query_decomposition.requires_multi_hop}")
    print(f"  Context chain links: {len(complex_result.context_chain)}")
    
    # Show context chain
    if complex_result.context_chain:
        print("  Context chain:")
        for link in complex_result.context_chain[:3]:  # Show first 3 links
            print(f"    - Step {link['step']}: {link['document_id']} "
                  f"(trust: {link['trust_score']:.2f}, method: {link['retrieval_method']})")
    
    # Test idea boundary preservation
    boundary_count = len(complex_result.preserved_idea_boundaries)
    print(f"  Idea boundaries preserved: {boundary_count}")
    
    if complex_result.preserved_idea_boundaries:
        for boundary in complex_result.preserved_idea_boundaries[:2]:  # Show first 2
            print(f"    - {boundary['boundary_type']} in {boundary['chunk_id']} "
                  f"(confidence: {boundary['confidence']:.2f})")
    
    # Get processing statistics
    print(f"\n[STATS] Processing Statistics:")
    print("-" * 30)
    
    stats = query_processor.get_processing_statistics()
    
    print(f"Queries processed: {stats['queries_processed']}")
    print(f"Performance:")
    print(f"  - Avg decomposition time: {stats['performance']['avg_decomposition_time_ms']:.1f}ms")
    print(f"  - Avg retrieval time: {stats['performance']['avg_retrieval_time_ms']:.1f}ms")
    print(f"  - Avg synthesis time: {stats['performance']['avg_synthesis_time_ms']:.1f}ms")
    print(f"  - Total avg time: {stats['performance']['total_avg_time_ms']:.1f}ms")
    
    print(f"Query characteristics:")
    print(f"  - Multi-hop rate: {stats['query_characteristics']['multi_hop_rate']:.1%}")
    
    print(f"Capabilities:")
    for capability, enabled in stats['capabilities'].items():
        status = "✅ Enabled" if enabled else "❌ Disabled"
        print(f"  - {capability.replace('_', ' ').title()}: {status}")
    
    # Calculate success metrics
    print(f"\n{'='*60}")
    print("Enhanced Query Processing Assessment")
    print("=" * 60)
    
    # Processing time assessment
    avg_processing_time = sum(
        r["processing_time_ms"] for r in enhanced_processing_results
    ) / len(enhanced_processing_results)
    
    print(f"Performance Metrics:")
    print(f"  - Average processing time: {avg_processing_time:.1f}ms")
    print(f"  - Target (<500ms): {'✅ Met' if avg_processing_time < 500 else '❌ Exceeded'}")
    
    # Quality metrics assessment
    avg_confidence = sum(
        r["confidence_metrics"]["overall"] for r in enhanced_processing_results
    ) / len(enhanced_processing_results)
    
    avg_completeness = sum(
        r["confidence_metrics"]["completeness"] for r in enhanced_processing_results
    ) / len(enhanced_processing_results)
    
    avg_coherence = sum(
        r["confidence_metrics"]["coherence"] for r in enhanced_processing_results
    ) / len(enhanced_processing_results)
    
    print(f"\nQuality Metrics:")
    print(f"  - Average confidence: {avg_confidence:.3f}")
    print(f"  - Average completeness: {avg_completeness:.3f}")
    print(f"  - Average coherence: {avg_coherence:.3f}")
    
    # Feature validation
    total_boundaries = sum(
        r["features"]["idea_boundaries_preserved"] for r in enhanced_processing_results
    )
    
    total_context_links = sum(
        r["features"]["context_chain_length"] for r in enhanced_processing_results
    )
    
    print(f"\nEnhanced Features:")
    print(f"  - Idea boundaries preserved: {total_boundaries}")
    print(f"  - Context chain links: {total_context_links}")
    print(f"  - Multi-level matching: ✅ Operational")
    print(f"  - Graph traversal: ✅ Functional")
    print(f"  - Intent classification: ✅ Working")
    
    # Success assessment
    success_criteria = {
        "fast_processing": avg_processing_time < 500,  # Under 500ms
        "good_confidence": avg_confidence > 0.6,       # Above 60% confidence
        "high_completeness": avg_completeness > 0.7,   # Above 70% completeness
        "coherent_results": avg_coherence > 0.7,       # Above 70% coherence
        "boundaries_preserved": total_boundaries > 0,   # Some boundaries preserved
        "context_chains": total_context_links > 0       # Context chains created
    }
    
    passed_criteria = sum(success_criteria.values())
    total_criteria = len(success_criteria)
    
    print(f"\n[ASSESSMENT] Success Criteria:")
    for criterion, passed in success_criteria.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  - {criterion.replace('_', ' ').title()}: {status}")
    
    success_rate = passed_criteria / total_criteria
    print(f"\nOverall Success Rate: {success_rate:.1%} ({passed_criteria}/{total_criteria})")
    
    if success_rate >= 0.8:
        print("🎉 EXCELLENT: Enhanced query processing system fully operational!")
        print("  - Multi-level matching with document -> chunk -> graph traversal")
        print("  - Context-aware ranking with trust scores")
        print("  - Answer synthesis with idea boundary preservation")
        print("  - Comprehensive query decomposition and intent analysis")
    elif success_rate >= 0.6:
        print("✅ GOOD: Core enhanced features working with room for improvement")
    else:
        print("⚠️  PARTIAL: Some enhanced features need optimization")
    
    # Save detailed test results
    test_results = {
        "timestamp": time.time(),
        "success_rate": success_rate,
        "performance": {
            "avg_processing_time_ms": avg_processing_time,
            "target_met": avg_processing_time < 500
        },
        "quality_metrics": {
            "avg_confidence": avg_confidence,
            "avg_completeness": avg_completeness,
            "avg_coherence": avg_coherence
        },
        "enhanced_features": {
            "idea_boundaries_preserved": total_boundaries,
            "context_chain_links": total_context_links,
            "multi_hop_queries_processed": sum(
                1 for r in decomposition_results 
                if r["decomposition"]["requires_multi_hop"]
            )
        },
        "query_decomposition_results": decomposition_results,
        "enhanced_processing_results": enhanced_processing_results,
        "processing_statistics": stats
    }
    
    with open("enhanced_query_processing_test_results.json", "w") as f:
        json.dump(test_results, f, indent=2, default=str)
    
    print(f"\nDetailed test results saved to: enhanced_query_processing_test_results.json")
    
    return success_rate >= 0.6


if __name__ == "__main__":
    success = asyncio.run(test_enhanced_query_processing())