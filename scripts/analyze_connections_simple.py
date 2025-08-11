#!/usr/bin/env python3
"""Simple Knowledge Connections Analysis.

Analyzes semantic connections between Grossman papers and AI research.
"""

from collections import Counter, defaultdict
import json
from pathlib import Path
import re


def extract_concepts(text):
    """Extract key concepts from text."""
    concepts = set()
    text_lower = text.lower()

    # Math concepts
    math_terms = [
        "calculus",
        "geometry",
        "topology",
        "manifold",
        "tensor",
        "differential",
        "integral",
        "nonlinear",
        "geometric",
        "arithmetic",
    ]

    # AI concepts
    ai_terms = [
        "neural",
        "network",
        "transformer",
        "attention",
        "embedding",
        "learning",
        "intelligence",
        "algorithm",
        "model",
        "quantization",
    ]

    # Bridge concepts
    bridge_terms = [
        "optimization",
        "gradient",
        "convergence",
        "approximation",
        "numerical",
        "computational",
        "probability",
        "bayesian",
    ]

    all_terms = math_terms + ai_terms + bridge_terms

    # Find terms
    words = re.findall(r"\b\w+\b", text_lower)
    for term in all_terms:
        if term in words:
            concepts.add(term)

    # Find compound terms
    compound_patterns = [
        "non newtonian",
        "meta calculus",
        "neural network",
        "deep learning",
        "machine learning",
        "language model",
        "attention mechanism",
        "retrieval augmented generation",
        "knowledge graph",
    ]

    for pattern in compound_patterns:
        if pattern in text_lower:
            concepts.add(pattern.replace(" ", "_"))

    return list(concepts)


def analyze_documents() -> None:
    """Analyze all documents."""
    papers_dir = Path("data/ingested_papers")

    if not papers_dir.exists():
        print("[ERROR] Ingested papers directory not found!")
        return

    print("KNOWLEDGE CONNECTIONS ANALYSIS")
    print("=" * 50)

    documents = {}

    print("[STEP 1] Processing documents...")

    for file_path in papers_dir.glob("*_full.txt"):
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            filename = file_path.stem.replace("_full", "")

            if "=== CONTENT ===" in content:
                text_content = content.split("=== CONTENT ===")[1]
            else:
                text_content = content

            # Classify document
            if "grossman" in filename.lower():
                doc_type = "mathematics"
                domain = "Mathematical Foundations"
            elif any(term in filename.lower() for term in ["rag", "retrieval"]):
                doc_type = "rag_research"
                domain = "Information Retrieval"
            elif any(term in filename.lower() for term in ["agent", "multi"]):
                doc_type = "multiagent"
                domain = "Multi-Agent Systems"
            elif any(term in filename.lower() for term in ["compress", "quant"]):
                doc_type = "compression"
                domain = "Model Optimization"
            else:
                doc_type = "ai_research"
                domain = "AI Research"

            # Extract concepts
            sample_text = text_content[:2000]
            concepts = extract_concepts(sample_text)

            documents[filename] = {
                "type": doc_type,
                "domain": domain,
                "concepts": concepts,
                "title": filename.replace("_", " ").title(),
            }

        except Exception:
            continue

    print(f"[SUCCESS] Processed {len(documents)} documents")

    # Find Grossman papers
    grossman_papers = [doc for doc in documents if "grossman" in doc.lower()]
    print(f"\n[GROSSMAN] Found {len(grossman_papers)} Grossman papers:")

    for paper in grossman_papers:
        print(f"  * {documents[paper]['title']}")
        print(f"    Concepts: {', '.join(documents[paper]['concepts'][:5])}")

    # Find connections
    print("\n[STEP 2] Finding connections...")

    concept_docs = defaultdict(list)
    for doc_id, doc_data in documents.items():
        for concept in doc_data["concepts"]:
            concept_docs[concept].append(doc_id)

    # Analyze Grossman connections
    print("\n[CONNECTIONS] Grossman paper connections to AI research:")

    total_connections = 0
    for grossman_paper in grossman_papers:
        grossman_data = documents[grossman_paper]
        connected_papers = set()
        shared_concepts = defaultdict(list)

        for concept in grossman_data["concepts"]:
            related_docs = concept_docs[concept]
            for doc in related_docs:
                if doc != grossman_paper and documents[doc]["type"] != "mathematics":
                    connected_papers.add(doc)
                    shared_concepts[doc].append(concept)

        # Filter for meaningful connections (2+ shared concepts)
        strong_connections = {doc: concepts for doc, concepts in shared_concepts.items() if len(concepts) >= 2}

        if strong_connections:
            total_connections += len(strong_connections)
            print(f"\n  [{grossman_data['title'][:30]}...]")
            print(f"    Connected to {len(strong_connections)} AI papers:")

            for connected_doc, shared in list(strong_connections.items())[:3]:
                connected_data = documents[connected_doc]
                print(f"      -> {connected_data['title'][:40]}...")
                print(f"         Domain: {connected_data['domain']}")
                print(f"         Shared: {', '.join(shared)}")

    # Cross-domain concept analysis
    print("\n[STEP 3] Cross-domain concept analysis...")

    cross_domain_concepts = {}
    for concept, docs in concept_docs.items():
        domains = {documents[doc]["domain"] for doc in docs}
        if len(domains) > 1 and len(docs) >= 3:
            cross_domain_concepts[concept] = {"domains": domains, "papers": len(docs)}

    print(f"[ANALYSIS] Found {len(cross_domain_concepts)} cross-domain concepts")
    print("\nTop bridging concepts:")

    sorted_concepts = sorted(cross_domain_concepts.items(), key=lambda x: x[1]["papers"], reverse=True)

    for concept, data in sorted_concepts[:8]:
        domains_list = ", ".join(sorted(data["domains"]))
        print(f"  * '{concept}': {data['papers']} papers")
        print(f"    Bridges: {domains_list}")

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    type_counts = Counter(doc["type"] for doc in documents.values())
    domain_counts = Counter(doc["domain"] for doc in documents.values())

    print("Document types:")
    for doc_type, count in type_counts.items():
        print(f"  {doc_type}: {count}")

    print("\nKnowledge domains:")
    for domain, count in domain_counts.items():
        print(f"  {domain}: {count}")

    print("\nConnection statistics:")
    print(f"  Total documents: {len(documents)}")
    print(f"  Grossman papers: {len(grossman_papers)}")
    print(f"  Grossman->AI connections: {total_connections}")
    print(f"  Cross-domain concepts: {len(cross_domain_concepts)}")

    # Export results
    results = {
        "total_documents": len(documents),
        "grossman_papers": len(grossman_papers),
        "ai_connections": total_connections,
        "cross_domain_concepts": len(cross_domain_concepts),
        "document_types": dict(type_counts),
        "bridging_concepts": {
            concept: {"papers": data["papers"], "domains": list(data["domains"])}
            for concept, data in sorted_concepts[:10]
        },
    }

    with open("connections_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n[SUCCESS] Analysis complete!")
    print("[OUTPUT] Results saved to connections_analysis.json")

    if total_connections > 0:
        print(f"\n[RESULT] CONFIRMED: Knowledge graph shows {total_connections} connections")
        print("         between Grossman mathematics and AI research!")
    else:
        print("\n[RESULT] Limited direct connections found - may need deeper analysis")


if __name__ == "__main__":
    analyze_documents()
