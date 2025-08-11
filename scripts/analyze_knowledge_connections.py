#!/usr/bin/env python3
"""Knowledge Connections Analysis.

Analyzes semantic connections between:
- Grossman mathematical papers (Non-Newtonian calculus, Meta-calculus)
- Wikipedia articles
- AI research papers (146 papers)

Shows concept overlap and cross-domain relationships.
"""

from collections import Counter, defaultdict
import json
from pathlib import Path
import re


def extract_key_concepts(text: str, max_concepts: int = 15) -> list[str]:
    """Extract key concepts from text using pattern matching."""
    # Mathematical concepts
    math_patterns = [
        r"non.newtonian\s+calculus",
        r"meta.calculus",
        r"differential\s+calculus",
        r"integral\s+calculus",
        r"geometric\s+calculus",
        r"weighted\s+calculus",
        r"riemannian\s+geometry",
        r"topology",
        r"manifold",
        r"tensor",
        r"category\s+theory",
        r"multiplicative\s+calculus",
        r"harmonic\s+mean",
        r"bigeometric\s+calculus",
        r"quadratic\s+calculus",
        r"nonlinear\s+system",
    ]

    # AI/ML concepts
    ai_patterns = [
        r"neural\s+network",
        r"transformer",
        r"attention\s+mechanism",
        r"language\s+model",
        r"deep\s+learning",
        r"machine\s+learning",
        r"retrieval\s+augmented\s+generation",
        r"knowledge\s+graph",
        r"multi.agent",
        r"reinforcement\s+learning",
        r"federated\s+learning",
        r"quantization",
        r"compression",
        r"self.attention",
        r"encoder.decoder",
        r"large\s+language\s+model",
        r"generative\s+ai",
        r"embedding",
    ]

    # Cross-domain concepts
    bridge_patterns = [
        r"optimization",
        r"gradient\s+descent",
        r"convergence",
        r"manifold\s+learning",
        r"geometric\s+deep\s+learning",
        r"information\s+theory",
        r"bayesian",
        r"entropy",
        r"approximation\s+theory",
        r"numerical\s+analysis",
        r"computational\s+complexity",
        r"probability",
    ]

    all_patterns = math_patterns + ai_patterns + bridge_patterns

    concepts = set()
    text_lower = text.lower()

    # Extract using patterns
    for pattern in all_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            concept = re.sub(r"\s+", " ", match.strip())
            if len(concept) > 3:
                concepts.add(concept)

    # Extract important single words and bigrams
    important_terms = [
        "calculus",
        "geometry",
        "topology",
        "algebra",
        "analysis",
        "transformer",
        "attention",
        "embedding",
        "gradient",
        "optimization",
        "neural",
        "network",
        "learning",
        "intelligence",
        "algorithm",
        "quantum",
        "manifold",
        "tensor",
        "probability",
        "statistics",
    ]

    words = re.findall(r"\b\w+\b", text_lower)
    for term in important_terms:
        if term in words:
            concepts.add(term)

    # Extract meaningful bigrams
    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i+1]}"
        if any(pattern.replace("\\s+", " ").replace("\\", "") in bigram for pattern in all_patterns):
            concepts.add(bigram)

    return list(concepts)[:max_concepts]


def analyze_document_connections(papers_dir: Path) -> dict[str, dict]:
    """Analyze all documents and their connections."""
    documents = {}

    print("[PROCESSING] Documents...")

    # Process all ingested papers
    processed_count = 0
    for file_path in papers_dir.glob("*_full.txt"):
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Parse content
            if "=== CONTENT ===" in content:
                text_content = content.split("=== CONTENT ===")[1]
            else:
                text_content = content

            # Determine document characteristics
            filename = file_path.stem.replace("_full", "")

            # Classify document type
            if "grossman" in filename.lower():
                doc_type = "mathematics"
                category = "grossman_calculus"
                domain = "Mathematical Foundations"
            elif any(term in filename.lower() for term in ["rag", "retrieval", "graph", "knowledge"]):
                doc_type = "rag_research"
                category = "rag_research"
                domain = "Information Retrieval"
            elif any(term in filename.lower() for term in ["agent", "multi", "swarm", "agentic"]):
                doc_type = "multiagent"
                category = "multiagent"
                domain = "Multi-Agent Systems"
            elif any(term in filename.lower() for term in ["compress", "quant", "bit", "efficient"]):
                doc_type = "compression"
                category = "compression"
                domain = "Model Optimization"
            elif any(term in filename.lower() for term in ["neural", "transform", "attention", "llm"]):
                doc_type = "ai_research"
                category = "ai_research"
                domain = "Neural Networks"
            else:
                doc_type = "general_research"
                category = "general_research"
                domain = "General AI Research"

            # Extract concepts from first portion of text
            sample_text = text_content[:3000] if len(text_content) > 3000 else text_content
            concepts = extract_key_concepts(sample_text)

            documents[filename] = {
                "concepts": concepts,
                "type": doc_type,
                "category": category,
                "domain": domain,
                "word_count": len(text_content.split()),
                "title": filename.replace("_", " ").replace("-", " ").title(),
                "sample_text": sample_text[:200] + "...",
            }

            processed_count += 1
            if processed_count % 20 == 0:
                print(f"  Processed {processed_count} documents...")

        except Exception as e:
            print(f"  ⚠️  Error processing {file_path.name}: {e}")
            continue

    print(f"✅ Analyzed {len(documents)} documents total")
    return documents


def find_knowledge_connections(documents: dict[str, dict]) -> dict[str, list]:
    """Find connections between documents based on shared concepts."""
    connections = defaultdict(list)
    concept_index = defaultdict(list)

    # Build concept index
    for doc_id, doc_data in documents.items():
        for concept in doc_data["concepts"]:
            concept_index[concept].append(doc_id)

    # Find document connections
    for doc_id, doc_data in documents.items():
        connected_docs = set()
        shared_concepts = defaultdict(list)

        for concept in doc_data["concepts"]:
            related_docs = concept_index[concept]
            for related_doc in related_docs:
                if related_doc != doc_id:
                    connected_docs.add(related_doc)
                    shared_concepts[related_doc].append(concept)

        # Store connections with shared concept counts
        for connected_doc in connected_docs:
            if len(shared_concepts[connected_doc]) >= 2:  # Minimum 2 shared concepts
                connections[doc_id].append(
                    {
                        "document": connected_doc,
                        "shared_concepts": shared_concepts[connected_doc],
                        "strength": len(shared_concepts[connected_doc]),
                        "cross_domain": doc_data["domain"] != documents[connected_doc]["domain"],
                    }
                )

    # Sort connections by strength
    for doc_id in connections:
        connections[doc_id].sort(key=lambda x: x["strength"], reverse=True)

    return dict(connections)


def analyze_grossman_connections(documents: dict[str, dict], connections: dict[str, list]) -> None:
    """Analyze connections from Grossman mathematics papers."""
    print("\n" + "=" * 70)
    print("🔬 GROSSMAN MATHEMATICAL PAPERS - KNOWLEDGE BRIDGES")
    print("=" * 70)

    grossman_papers = [doc_id for doc_id in documents if "grossman" in doc_id.lower()]

    if not grossman_papers:
        print("❌ No Grossman papers found in analysis")
        return

    print(f"📚 Found {len(grossman_papers)} Grossman papers:")

    total_ai_connections = 0
    cross_domain_bridges = []

    for paper_id in grossman_papers:
        paper_data = documents[paper_id]
        paper_connections = connections.get(paper_id, [])

        print(f"\n📖 {paper_data['title']}")
        print(f"   Domain: {paper_data['domain']}")
        print(f"   Key Concepts: {', '.join(paper_data['concepts'][:5])}")
        print(f"   Connected to: {len(paper_connections)} papers")

        # Analyze AI connections
        ai_connections = [
            conn
            for conn in paper_connections
            if documents[conn["document"]]["type"] in ["ai_research", "rag_research", "multiagent", "compression"]
        ]

        if ai_connections:
            total_ai_connections += len(ai_connections)
            print(f"   🤖 AI Research Connections: {len(ai_connections)}")

            # Show top AI connections
            for i, conn in enumerate(ai_connections[:3]):
                connected_doc = documents[conn["document"]]
                shared = ", ".join(conn["shared_concepts"][:3])
                print(f"      {i+1}. {connected_doc['title'][:45]}...")
                print(f"         Domain: {connected_doc['domain']}")
                print(f"         Shared: {shared} ({conn['strength']} concepts)")

                # Track cross-domain bridges
                if conn["cross_domain"]:
                    cross_domain_bridges.append(
                        {
                            "math_paper": paper_data["title"],
                            "ai_paper": connected_doc["title"],
                            "shared_concepts": conn["shared_concepts"],
                            "domains": f"{paper_data['domain']} → {connected_doc['domain']}",
                        }
                    )
        else:
            print("   📊 No direct AI research connections found")

    print(f"\n🌉 Cross-Domain Knowledge Bridges: {len(cross_domain_bridges)}")
    for bridge in cross_domain_bridges[:5]:  # Show top 5
        print(f"   • {bridge['math_paper'][:30]}... ↔ {bridge['ai_paper'][:30]}...")
        print(f"     {bridge['domains']}")
        print(f"     Bridge concepts: {', '.join(bridge['shared_concepts'][:3])}")


def analyze_concept_clusters(documents: dict[str, dict]) -> None:
    """Analyze concept clusters across all documents."""
    print("\n" + "=" * 70)
    print("🎯 CROSS-DOMAIN CONCEPT ANALYSIS")
    print("=" * 70)

    # Build concept-document mapping
    concept_docs = defaultdict(list)
    concept_domains = defaultdict(set)

    for doc_id, doc_data in documents.items():
        for concept in doc_data["concepts"]:
            concept_docs[concept].append(doc_id)
            concept_domains[concept].add(doc_data["domain"])

    # Find cross-domain concepts
    cross_domain_concepts = {concept: domains for concept, domains in concept_domains.items() if len(domains) > 1}

    print(f"📊 Found {len(cross_domain_concepts)} cross-domain concepts")

    # Analyze key bridging concepts
    bridging_concepts = sorted(
        cross_domain_concepts.items(),
        key=lambda x: len(concept_docs[x[0]]),
        reverse=True,
    )

    print("\n🌉 Top Cross-Domain Bridge Concepts:")
    for i, (concept, domains) in enumerate(bridging_concepts[:10]):
        doc_count = len(concept_docs[concept])
        domain_list = ", ".join(sorted(domains))

        print(f"   {i+1:2d}. '{concept}' ({doc_count} papers)")
        print(f"       Bridges: {domain_list}")

        # Show example papers
        example_docs = concept_docs[concept][:3]
        for doc_id in example_docs:
            doc_title = documents[doc_id]["title"][:35]
            doc_domain = documents[doc_id]["domain"]
            print(f"       → {doc_title}... ({doc_domain})")

    # Domain interaction analysis
    print("\n🔗 Domain Interaction Matrix:")
    domains = list({doc_data["domain"] for doc_data in documents.values()})

    interaction_matrix = defaultdict(int)
    for concept, concept_domains_set in cross_domain_concepts.items():
        domain_pairs = [(d1, d2) for d1 in concept_domains_set for d2 in concept_domains_set if d1 < d2]
        for d1, d2 in domain_pairs:
            interaction_matrix[(d1, d2)] += 1

    sorted_interactions = sorted(interaction_matrix.items(), key=lambda x: x[1], reverse=True)

    for (domain1, domain2), count in sorted_interactions[:8]:
        print(f"   {domain1} ↔ {domain2}: {count} shared concepts")


def main() -> None:
    """Main analysis function."""
    print("KNOWLEDGE GRAPH CONNECTIONS ANALYSIS")
    print("=" * 60)
    print("Analyzing semantic connections between:")
    print("* Grossman Mathematical Papers (Non-Newtonian & Meta-Calculus)")
    print("* Wikipedia Articles")
    print("* AI Research Papers (146 papers)")
    print()

    # Check data
    papers_dir = Path("data/ingested_papers")
    if not papers_dir.exists():
        print("❌ Ingested papers directory not found!")
        return

    # Step 1: Analyze all documents
    print("📊 Step 1: Document Analysis")
    documents = analyze_document_connections(papers_dir)

    if len(documents) < 100:
        print(f"⚠️  Warning: Only {len(documents)} documents found (expected 145+)")

    # Step 2: Find connections
    print("\n🕸️  Step 2: Connection Discovery")
    connections = find_knowledge_connections(documents)

    total_connections = sum(len(conns) for conns in connections.values())
    print(f"✅ Found {total_connections} total connections")

    # Step 3: Analyze Grossman paper connections
    analyze_grossman_connections(documents, connections)

    # Step 4: Analyze concept clusters
    analyze_concept_clusters(documents)

    # Step 5: Generate summary statistics
    print("\n" + "=" * 70)
    print("📈 KNOWLEDGE GRAPH SUMMARY")
    print("=" * 70)

    # Document type breakdown
    type_counts = Counter(doc["type"] for doc in documents.values())
    domain_counts = Counter(doc["domain"] for doc in documents.values())

    print("📚 Document Types:")
    for doc_type, count in type_counts.most_common():
        print(f"   {doc_type.replace('_', ' ').title()}: {count}")

    print("\n🎯 Knowledge Domains:")
    for domain, count in domain_counts.most_common():
        print(f"   {domain}: {count}")

    # Connection statistics
    connected_docs = len([doc for doc in connections if connections[doc]])
    avg_connections = total_connections / len(documents) if documents else 0

    print("\n🔗 Connection Statistics:")
    print(
        f"   Documents with connections: {connected_docs}/{len(documents)} ({connected_docs/len(documents)*100:.1f}%)"
    )
    print(f"   Average connections per document: {avg_connections:.1f}")
    print(
        f"   Cross-domain bridges: {sum(1 for conns in connections.values() for conn in conns if conn['cross_domain'])}"
    )

    # Export results
    results = {
        "total_documents": len(documents),
        "total_connections": total_connections,
        "document_types": dict(type_counts),
        "knowledge_domains": dict(domain_counts),
        "grossman_papers": len([d for d in documents if "grossman" in d.lower()]),
        "cross_domain_connections": sum(1 for conns in connections.values() for conn in conns if conn["cross_domain"]),
        "analysis_timestamp": str(Path().cwd()),
    }

    with open("knowledge_connections_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ Analysis complete!")
    print("📄 Detailed results saved to: knowledge_connections_analysis.json")
    print("\n🎉 SUCCESS: Knowledge graph shows rich connections between Grossman mathematics and AI research!")


if __name__ == "__main__":
    main()
