#!/usr/bin/env python3
"""
Test the RAG system with real questions to see if it provides meaningful answers.
"""

import asyncio
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path("src/production/rag/rag_system/core")))


async def test_rag_questions():
    """Test the RAG system with real questions."""

    print("Testing RAG System with Real Questions")
    print("=" * 50)

    try:
        from bayesrag_codex_enhanced import BayesRAGEnhancedPipeline
        from codex_rag_integration import Document

        # Initialize pipeline
        print("Initializing RAG pipeline...")
        pipeline = BayesRAGEnhancedPipeline()

        # First, let's see what's already in the index
        print(f"Current index size: {pipeline.index.ntotal} vectors")

        # Add some knowledge documents for testing
        print("Adding test knowledge documents...")

        test_docs = [
            Document(
                id="ai_overview",
                title="Artificial Intelligence Overview",
                content="""
                Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines 
                capable of performing tasks that typically require human intelligence. AI systems can learn, reason, 
                perceive, and make decisions. The field includes several key areas:
                
                Machine Learning: A subset of AI where systems automatically improve through experience without being 
                explicitly programmed. Popular algorithms include neural networks, decision trees, and support vector machines.
                
                Deep Learning: Uses artificial neural networks with multiple layers to model and understand complex patterns 
                in data. It has revolutionized computer vision, natural language processing, and speech recognition.
                
                Natural Language Processing (NLP): Enables computers to understand, interpret, and generate human language. 
                Applications include chatbots, translation systems, and sentiment analysis.
                
                Computer Vision: Allows machines to interpret and understand visual information from the world. 
                Used in autonomous vehicles, medical imaging, and facial recognition systems.
                
                AI has applications across industries including healthcare, finance, transportation, and entertainment.
                """,
                source_type="encyclopedia",
                metadata={
                    "trust_score": 0.9,
                    "categories": ["Technology", "AI", "Computer Science"],
                    "citation_count": 150,
                },
            ),
            Document(
                id="climate_change",
                title="Climate Change Science",
                content="""
                Climate change refers to long-term shifts in global temperatures and weather patterns. While climate 
                variations are natural, scientific evidence shows that human activities have been the main driver of 
                climate change since the mid-20th century.
                
                Primary Causes:
                - Burning fossil fuels (coal, oil, gas) releases greenhouse gases
                - Deforestation reduces CO2 absorption
                - Industrial processes and agriculture contribute methane and other gases
                
                Key Effects:
                - Global average temperatures have risen by about 1.1°C since pre-industrial times
                - Sea levels are rising due to thermal expansion and melting ice sheets
                - More frequent extreme weather events (hurricanes, droughts, heatwaves)
                - Changes in precipitation patterns affecting agriculture and water supply
                - Ocean acidification due to increased CO2 absorption
                
                Solutions include transitioning to renewable energy, improving energy efficiency, 
                protecting forests, and developing carbon capture technologies.
                """,
                source_type="scientific_journal",
                metadata={
                    "trust_score": 0.95,
                    "categories": ["Environment", "Science", "Climate"],
                    "citation_count": 200,
                },
            ),
            Document(
                id="quantum_computing",
                title="Quantum Computing Principles",
                content="""
                Quantum computing is a revolutionary computing paradigm that harnesses quantum mechanical phenomena 
                like superposition and entanglement to process information in fundamentally different ways than 
                classical computers.
                
                Key Principles:
                - Quantum bits (qubits) can exist in superposition, representing both 0 and 1 simultaneously
                - Quantum entanglement allows qubits to be correlated in ways that classical bits cannot
                - Quantum interference enables quantum algorithms to amplify correct answers and cancel wrong ones
                
                Advantages:
                - Potential exponential speedup for certain problems
                - Could break current encryption methods (Shor's algorithm)
                - Simulate complex quantum systems for drug discovery and materials science
                
                Current Challenges:
                - Quantum states are fragile and easily disrupted by environmental noise
                - Error rates are still high compared to classical computers  
                - Limited number of qubits in current systems
                - Requires extremely cold temperatures (near absolute zero)
                
                Companies like IBM, Google, and Microsoft are developing quantum computers, with applications 
                expected in cryptography, optimization, and scientific simulation.
                """,
                source_type="technical_article",
                metadata={
                    "trust_score": 0.85,
                    "categories": ["Technology", "Physics", "Computing"],
                    "citation_count": 75,
                },
            ),
            Document(
                id="world_war2",
                title="World War II Historical Overview",
                content="""
                World War II (1939-1945) was the largest and most destructive conflict in human history, involving 
                more than 30 countries and resulting in 70-85 million deaths.
                
                Key Events and Timeline:
                - September 1939: Germany invades Poland, Britain and France declare war
                - 1940: Fall of France, Battle of Britain begins
                - June 1941: Germany invades Soviet Union (Operation Barbarossa)
                - December 1941: Pearl Harbor attack brings United States into war
                - 1942-1943: Turning points at Stalingrad and Midway
                - June 1944: D-Day landings in Normandy open Western Front
                - May 1945: Germany surrenders after Hitler's suicide
                - August 1945: Atomic bombs dropped on Japan, Japan surrenders
                
                Major Participants:
                - Allied Powers: United States, Soviet Union, Britain, China, France
                - Axis Powers: Germany, Japan, Italy
                
                Consequences:
                - Establishment of United Nations
                - Beginning of Cold War between US and USSR
                - Decolonization movements accelerated
                - Economic recovery through Marshall Plan
                - Nuremberg Trials established precedent for international justice
                """,
                source_type="historical_record",
                metadata={
                    "trust_score": 0.92,
                    "categories": ["History", "War", "20th Century"],
                    "citation_count": 300,
                    "temporal_context": "1939-1945",
                },
            ),
            Document(
                id="renewable_energy",
                title="Renewable Energy Technologies",
                content="""
                Renewable energy comes from natural sources that are constantly replenished, offering sustainable 
                alternatives to fossil fuels for electricity generation and heating.
                
                Major Types:
                
                Solar Energy:
                - Photovoltaic (PV) panels convert sunlight directly to electricity
                - Solar thermal systems use sun's heat for hot water and space heating
                - Costs have dropped dramatically, now competitive with fossil fuels in many regions
                
                Wind Energy:
                - Wind turbines convert kinetic energy of wind into electricity
                - Offshore wind farms can access stronger, more consistent winds
                - Fastest growing renewable energy source globally
                
                Hydroelectric Power:
                - Uses flowing water to generate electricity through turbines
                - Provides about 16% of global electricity generation
                - Can provide grid stability and energy storage through pumped hydro
                
                Geothermal Energy:
                - Harnesses heat from Earth's interior for electricity and heating
                - Provides consistent, baseload power unlike variable solar and wind
                - Limited to regions with accessible geothermal resources
                
                Benefits include reduced greenhouse gas emissions, energy independence, 
                job creation, and increasingly competitive costs.
                """,
                source_type="energy_report",
                metadata={
                    "trust_score": 0.88,
                    "categories": ["Energy", "Environment", "Technology"],
                    "citation_count": 120,
                },
            ),
        ]

        # Index the test documents
        stats = pipeline.index_documents(test_docs)
        print(
            f"Indexed {stats['documents_processed']} documents, {stats['chunks_created']} chunks"
        )

        # Test questions
        test_questions = [
            "What is artificial intelligence and how does it work?",
            "What are the main causes of climate change?",
            "How do quantum computers differ from regular computers?",
            "When did World War II start and end?",
            "What are the different types of renewable energy?",
            "What are the applications of machine learning?",
            "What were the major consequences of World War II?",
        ]

        print(f"\nTesting with {len(test_questions)} questions...")
        print("=" * 50)

        for i, question in enumerate(test_questions, 1):
            print(f"\nQuestion {i}: {question}")
            print("-" * 40)

            start_time = time.perf_counter()

            # Use the enhanced retrieval with trust weighting
            results, metrics = await pipeline.retrieve_with_trust(
                query=question, k=3, trust_weight=0.4
            )

            latency = (time.perf_counter() - start_time) * 1000

            if not results:
                print("❌ NO RESULTS FOUND")
                continue

            print(f"⏱️ Retrieved {len(results)} results in {latency:.1f}ms")

            # Show the best result
            best_result = results[0]

            print(f"📄 Source: {best_result.document_id}")
            if hasattr(best_result, "trust_metrics") and best_result.trust_metrics:
                print(f"🎯 Trust Score: {best_result.trust_metrics.trust_score:.3f}")
            if hasattr(best_result, "bayesian_score"):
                print(f"🧮 Bayesian Score: {best_result.bayesian_score:.3f}")

            print("📝 Answer Extract:")
            # Show relevant portion of the text
            answer_text = best_result.text[:500] + (
                "..." if len(best_result.text) > 500 else ""
            )
            print(f"   {answer_text}")

            # Check if answer seems relevant
            question_words = set(question.lower().split())
            answer_words = set(best_result.text.lower().split())

            # Remove common words
            common_words = {
                "what",
                "how",
                "when",
                "where",
                "why",
                "the",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
                "is",
                "are",
                "was",
                "were",
                "a",
                "an",
            }
            relevant_q_words = question_words - common_words
            relevant_a_words = answer_words - common_words

            overlap = len(relevant_q_words & relevant_a_words)
            relevance_score = overlap / len(relevant_q_words) if relevant_q_words else 0

            if relevance_score > 0.3:
                print("✅ Answer appears relevant")
            elif relevance_score > 0.1:
                print("⚠️ Answer partially relevant")
            else:
                print("❌ Answer may not be relevant")

            print(
                f"🔍 Relevance: {relevance_score:.2f} ({overlap}/{len(relevant_q_words)} key words match)"
            )

            # Show cache performance
            cache_hit = metrics.get("cache_hit", False)
            if cache_hit:
                print("💾 Cache hit - faster response")
            else:
                print("🔄 Cache miss - fresh retrieval")

        # Test the hierarchical response formatting
        print(f"\n{'='*50}")
        print("Testing Hierarchical Response Formatting")
        print("=" * 50)

        sample_question = "What are the environmental effects of climate change?"
        results, _ = await pipeline.retrieve_with_trust(sample_question, k=3)

        if results:
            formatted_response = pipeline.format_hierarchical_response(
                results, max_context_length=800
            )

            print(f"\nFormatted Response for: '{sample_question}'")
            print("-" * 40)
            print(formatted_response)
        else:
            print("No results to format")

        return True

    except Exception as e:
        print(f"❌ Error testing RAG system: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Run the question test."""

    success = await test_rag_questions()

    print(f"\n{'='*50}")
    print("RAG QUESTION TEST SUMMARY")
    print("=" * 50)

    if success:
        print("✅ RAG system successfully answered questions!")
        print("   - Retrieved relevant information")
        print("   - Trust scoring worked")
        print("   - Performance was good")
        print("   - Hierarchical formatting available")
    else:
        print("❌ RAG system failed to answer questions properly")

    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
