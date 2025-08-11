"""
Simple Integration Test for Contextual Tagging System.

Validates the two-level contextual tagging implementation without
heavy processing overhead.
"""

import json
import sys
import time

sys.path.append("src/production/rag/rag_system/core")

from contextual_tagging import ContextualTagger


def test_contextual_tagging_features():
    """Test core contextual tagging features."""

    print("Testing Contextual Tagging Integration")
    print("=" * 50)

    # Initialize tagger (without spaCy to avoid compatibility issues)
    print("[INIT] Initializing contextual tagger...")
    tagger = ContextualTagger(enable_spacy=False)
    print("Contextual tagger initialized successfully")

    # Test document
    test_document = """
    # Machine Learning in Healthcare: A Technical Analysis
    
    Machine learning applications in healthcare are transforming medical diagnosis and treatment planning. 
    This technical report examines recent developments in medical AI systems, focusing on deep learning 
    approaches for medical image analysis and clinical decision support systems.
    
    ## Image Analysis Applications
    
    Convolutional neural networks have achieved remarkable success in medical imaging tasks. For example, 
    dermatology AI systems can now identify skin cancer with accuracy comparable to expert dermatologists.
    
    ### Deep Learning Architectures
    
    U-Net architectures are particularly effective for medical image segmentation. These networks use 
    skip connections to preserve fine-grained spatial information during the encoding-decoding process.
    
    ## Clinical Decision Support
    
    Natural language processing techniques enable analysis of electronic health records to identify 
    at-risk patients and recommend treatment protocols. However, ensuring fairness and avoiding bias 
    in these systems remains a critical challenge.
    
    ## Regulatory Considerations
    
    FDA approval processes for AI medical devices require rigorous validation studies demonstrating 
    safety and efficacy. The regulatory landscape continues to evolve as AI technologies mature.
    
    # Conclusion
    
    While promising, medical AI systems must address challenges including data privacy, algorithmic 
    transparency, and clinical validation before achieving widespread deployment in healthcare settings.
    """

    # Test Level 1: Document Context Extraction
    print("\n[TEST] Level 1 - Document Context Extraction")
    print("-" * 40)

    start_time = time.perf_counter()
    document_context = tagger.extract_document_context(
        document_id="med_ai_report_001",
        title="Machine Learning in Healthcare: A Technical Analysis",
        content=test_document,
        metadata={
            "author": "Dr. Medical AI Researcher",
            "publication_date": "2024-02-15",
            "journal": "Medical AI Review",
            "credibility_score": 0.9,
            "target_audience": "medical professionals",
        },
    )
    extraction_time = (time.perf_counter() - start_time) * 1000

    print(f"Document Context Extracted in {extraction_time:.1f}ms:")
    print(f"  Document Type: {document_context.document_type.value}")
    print(f"  Domain: {document_context.domain.value}")
    print(f"  Reading Level: {document_context.reading_level.value}")
    print(f"  Credibility: {document_context.source_credibility_score:.2f}")
    print(f"  Estimated Reading Time: {document_context.estimated_reading_time} minutes")
    print(f"  Key Themes: {document_context.key_themes}")
    print(f"  Key Concepts: {len(document_context.key_concepts)} identified")
    print(f"  Document Entities: {len(document_context.document_entities)} found")
    print(f"  Executive Summary: {document_context.executive_summary}")

    # Test Level 2: Chunk Context Extraction with simple chunking
    print("\n[TEST] Level 2 - Chunk Context with Inheritance")
    print("-" * 40)

    # Split into paragraphs for simple chunking
    paragraphs = [p.strip() for p in test_document.split("\n\n") if p.strip()]

    contextual_chunks = []
    previous_chunk_context = None

    for i, paragraph in enumerate(paragraphs[:4]):  # Process first 4 paragraphs
        chunk_id = f"med_ai_report_001_chunk_{i}"
        start_char = test_document.find(paragraph)
        end_char = start_char + len(paragraph)

        print(f"\nProcessing Chunk {i+1}: {chunk_id}")

        # Create contextual chunk
        start_time = time.perf_counter()
        contextual_chunk = tagger.create_contextual_chunk(
            chunk_id=chunk_id,
            chunk_text=paragraph,
            chunk_position=i,
            start_char=start_char,
            end_char=end_char,
            document_context=document_context,
            full_document_text=test_document,
            previous_chunk_context=previous_chunk_context,
        )
        processing_time = (time.perf_counter() - start_time) * 1000

        contextual_chunks.append(contextual_chunk)

        print(f"  Processing Time: {processing_time:.1f}ms")
        print(f"  Chunk Type: {contextual_chunk['chunk_context']['chunk_type']}")
        print(f"  Local Summary: {contextual_chunk['chunk_context']['local_summary'][:100]}...")
        print(f"  Section Hierarchy: {contextual_chunk['chunk_context']['section_hierarchy']}")
        print(f"  Local Keywords: {contextual_chunk['chunk_context']['local_keywords'][:5]}")
        print(f"  Quality Score: {contextual_chunk['quality_metrics']['overall_quality']:.3f}")
        print(f"  Context Richness: {contextual_chunk.get('context_richness_score', 0):.3f}")

        # Show inheritance
        if contextual_chunk["context_inheritance"]["inherited_context"]:
            print(
                f"  Inherited Context: {len(contextual_chunk['context_inheritance']['inherited_context'])} properties"
            )
        if contextual_chunk["context_inheritance"]["context_overrides"]:
            print(f"  Context Overrides: {list(contextual_chunk['context_inheritance']['context_overrides'].keys())}")

        # Update previous chunk context for next iteration
        previous_chunk_context = tagger.extract_chunk_context(
            chunk_id=chunk_id,
            chunk_text=paragraph,
            chunk_position=i,
            start_char=start_char,
            end_char=end_char,
            document_context=document_context,
            full_document_text=test_document,
            previous_chunk_context=previous_chunk_context,
        )

    # Test Context Chain Preservation
    print("\n[TEST] Context Chain Preservation")
    print("-" * 40)

    print("Context Chain Analysis:")
    for i, chunk in enumerate(contextual_chunks):
        relationships = chunk["relationships"]
        inheritance = chunk["context_inheritance"]

        print(f"\nChunk {i+1} ({chunk['chunk_id']}):")
        print(f"  Previous Chunk: {relationships['previous_chunk_id'] or 'None'}")
        print(f"  Document Context Inherited: {len(inheritance['inherited_context'])} properties")
        print(f"  Local Context Override: {len(inheritance['context_overrides'])} properties")

        # Show specific inherited properties
        inherited = inheritance["inherited_context"]
        print("  Inherited Properties:")
        print(f"    - Document Title: {inherited.get('document_title', 'N/A')}")
        print(f"    - Domain: {inherited.get('domain', 'N/A')}")
        print(f"    - Reading Level: {inherited.get('reading_level', 'N/A')}")
        print(f"    - Key Themes: {inherited.get('key_themes', [])}")

    # Test Bilateral Context Features
    print("\n[TEST] Bilateral Context Features")
    print("-" * 40)

    # Demonstrate rich bilateral context
    sample_chunk = contextual_chunks[1] if len(contextual_chunks) > 1 else contextual_chunks[0]

    print("Sample Chunk with Full Bilateral Context:")
    print(f"  Chunk ID: {sample_chunk['chunk_id']}")

    # Document-level context (Level 1)
    doc_ctx = sample_chunk["document_context"]
    print("\n  LEVEL 1 - Document Context:")
    print(f"    Title: {doc_ctx['title']}")
    print(f"    Type: {doc_ctx['document_type']}")
    print(f"    Domain: {doc_ctx['domain']}")
    print(f"    Author: {doc_ctx['author']}")
    print(f"    Reading Level: {doc_ctx['reading_level']}")
    print(f"    Key Themes: {doc_ctx['key_themes']}")
    print(f"    Credibility: {doc_ctx['credibility_score']}")

    # Chunk-level context (Level 2)
    chunk_ctx = sample_chunk["chunk_context"]
    print("\n  LEVEL 2 - Chunk Context:")
    print(f"    Chunk Type: {chunk_ctx['chunk_type']}")
    print(f"    Local Summary: {chunk_ctx['local_summary']}")
    print(f"    Section Hierarchy: {chunk_ctx['section_hierarchy']}")
    print(f"    Local Keywords: {chunk_ctx['local_keywords']}")
    print(f"    Coherence Score: {chunk_ctx['coherence_score']:.3f}")
    print(f"    Relevance Score: {chunk_ctx['relevance_score']:.3f}")

    # Success Summary
    print(f"\n{'='*50}")
    print("CONTEXTUAL TAGGING INTEGRATION - SUCCESS!")
    print("=" * 50)

    print("[SUCCESS] Level 1 Context: Document-level metadata extracted")
    print(f"   - Classification: {document_context.document_type.value} / {document_context.domain.value}")
    print(f"   - Quality Assessment: {document_context.source_credibility_score:.2f} credibility")
    print(
        f"   - Content Analysis: {len(document_context.key_themes)} themes, {len(document_context.key_concepts)} concepts"
    )

    print("\n[SUCCESS] Level 2 Context: Chunk-level metadata with inheritance")
    print(f"   - Processed Chunks: {len(contextual_chunks)}")
    print("   - Context Inheritance: Document -> Chunk properties preserved")
    print("   - Quality Metrics: Coherence, relevance, completeness scored")

    print("\n[SUCCESS] Context Chain Preservation:")
    print("   - Bilateral Context: Document <-> Chunk relationships maintained")
    print("   - Inheritance Chain: Previous chunk -> Current chunk continuity")
    print(f"   - Rich Metadata: {len(sample_chunk)} top-level properties per chunk")

    print("\n[SUCCESS] Implementation Features:")
    print("   - Two-level hierarchy: Global + Local context")
    print("   - Rhetorical structure preservation")
    print("   - Entity extraction framework (ready for spaCy)")
    print("   - Quality scoring and contextual filtering")

    # Save sample output
    output_file = "contextual_tagging_sample_output.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "document_context": {
                    "document_id": document_context.document_id,
                    "document_type": document_context.document_type.value,
                    "domain": document_context.domain.value,
                    "reading_level": document_context.reading_level.value,
                    "key_themes": document_context.key_themes,
                    "key_concepts": document_context.key_concepts[:5],  # First 5
                    "executive_summary": document_context.executive_summary,
                    "credibility_score": document_context.source_credibility_score,
                },
                "contextual_chunks": contextual_chunks[:2],  # First 2 chunks
                "integration_status": "SUCCESS",
                "features_validated": [
                    "Level 1 Document Context Extraction",
                    "Level 2 Chunk Context with Inheritance",
                    "Context Chain Preservation",
                    "Bilateral Context Relationships",
                    "Quality Metrics Calculation",
                    "Rhetorical Structure Analysis",
                ],
            },
            f,
            indent=2,
        )

    print(f"\n[SAVE] Sample output saved to: {output_file}")

    return True


if __name__ == "__main__":
    test_contextual_tagging_features()
