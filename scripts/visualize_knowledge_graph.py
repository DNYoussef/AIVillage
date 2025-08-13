#!/usr/bin/env python3
"""Knowledge Graph Visualization.

Creates an interactive graph showing connections between:
- Grossman mathematical papers (Non-Newtonian calculus, Meta-calculus)
- Wikipedia articles
- AI research papers

Shows semantic relationships and concept overlap.
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    print(f"Installing required packages: {e}")
    os.system("pip install networkx matplotlib scikit-learn")
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        VISUALIZATION_AVAILABLE = True
    except ImportError:
        VISUALIZATION_AVAILABLE = False


def extract_key_concepts(text: str, max_concepts: int = 20) -> list[str]:
    """Extract key concepts from text using TF-IDF and domain knowledge."""
    # Domain-specific concept patterns
    math_concepts = [
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
        r"topos\s+theory",
        r"homomorphism",
        r"multiplicative\s+calculus",
        r"geometric\s+mean",
        r"harmonic\s+mean",
        r"bigeometric\s+calculus",
        r"quadratic\s+calculus",
    ]

    ai_concepts = [
        r"neural\s+network",
        r"transformer",
        r"attention\s+mechanism",
        r"language\s+model",
        r"deep\s+learning",
        r"machine\s+learning",
        r"artificial\s+intelligence",
        r"embeddings?",
        r"quantization",
        r"retrieval\s+augmented\s+generation",
        r"rag\s+system",
        r"knowledge\s+graph",
        r"graph\s+neural\s+network",
        r"multi.agent",
        r"reinforcement\s+learning",
        r"federated\s+learning",
        r"compression",
        r"pruning",
        r"distillation",
        r"fine.tuning",
        r"self.attention",
        r"cross.attention",
        r"encoder.decoder",
        r"generative\s+ai",
        r"large\s+language\s+model",
        r"llm",
    ]

    cross_domain_concepts = [
        r"optimization",
        r"gradient\s+descent",
        r"loss\s+function",
        r"convergence",
        r"manifold\s+learning",
        r"geometric\s+deep\s+learning",
        r"differential\s+geometry",
        r"information\s+theory",
        r"bayesian",
        r"probability",
        r"statistics",
        r"entropy",
        r"complexity\s+theory",
        r"computational\s+complexity",
        r"approximation\s+theory",
        r"numerical\s+analysis",
    ]

    all_patterns = math_concepts + ai_concepts + cross_domain_concepts

    # Extract concepts using patterns
    concepts = set()
    text_lower = text.lower()

    for pattern in all_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            concept = re.sub(r"\s+", " ", match.strip())
            if len(concept) > 3:
                concepts.add(concept)

    # Extract additional concepts using TF-IDF
    try:
        # Clean text for TF-IDF
        clean_text = re.sub(r"[^\w\s]", " ", text_lower)
        clean_text = re.sub(r"\s+", " ", clean_text)

        # Simple bigram extraction
        words = clean_text.split()
        bigrams = [f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1)]

        # Filter meaningful bigrams
        meaningful_bigrams = [
            bg
            for bg in bigrams
            if len(bg) > 6
            and not any(
                stop in bg for stop in ["the ", "and ", "for ", "that ", "with "]
            )
        ]

        # Add top bigrams
        bigram_counts = Counter(meaningful_bigrams)
        for bigram, count in bigram_counts.most_common(5):
            if count > 1:
                concepts.add(bigram)
    except:
        pass

    return list(concepts)[:max_concepts]


def analyze_document_relationships(papers_dir: Path) -> dict[str, dict]:
    """Analyze relationships between all documents."""
    documents = {}

    # Process ingested papers
    for file_path in papers_dir.glob("*_full.txt"):
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Extract metadata and content
            if "=== METADATA ===" in content and "=== CONTENT ===" in content:
                content.split("=== CONTENT ===")[0]
                text_content = content.split("=== CONTENT ===")[1]
            else:
                text_content = content

            # Determine document type
            filename = file_path.stem.replace("_full", "")
            if "grossman" in filename.lower():
                doc_type = "mathematics"
                category = "grossman_calculus"
            elif any(
                term in filename.lower() for term in ["rag", "retrieval", "knowledge"]
            ):
                doc_type = "rag_research"
                category = "rag_research"
            elif any(term in filename.lower() for term in ["agent", "multi", "swarm"]):
                doc_type = "multiagent"
                category = "multiagent"
            elif any(term in filename.lower() for term in ["compress", "quant", "bit"]):
                doc_type = "compression"
                category = "compression"
            elif any(
                term in filename.lower()
                for term in ["neural", "transform", "attention"]
            ):
                doc_type = "ai_research"
                category = "ai_research"
            else:
                doc_type = "general_research"
                category = "general_research"

            # Extract concepts
            concepts = extract_key_concepts(text_content[:5000])  # First 5k chars

            documents[filename] = {
                "content": text_content[:1000] + "...",  # Truncated for memory
                "concepts": concepts,
                "type": doc_type,
                "category": category,
                "word_count": len(text_content.split()),
                "title": filename.replace("_", " ").title(),
            }

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    print(f"Analyzed {len(documents)} documents")
    return documents


def create_knowledge_graph(documents: dict[str, dict]) -> nx.Graph:
    """Create knowledge graph from documents."""
    G = nx.Graph()

    # Add document nodes
    for doc_id, doc_data in documents.items():
        G.add_node(
            doc_id,
            title=doc_data["title"],
            type=doc_data["type"],
            category=doc_data["category"],
            concepts=doc_data["concepts"],
            word_count=doc_data["word_count"],
        )

    # Create concept-based edges
    concept_to_docs = defaultdict(list)
    for doc_id, doc_data in documents.items():
        for concept in doc_data["concepts"]:
            concept_to_docs[concept].append(doc_id)

    # Add edges between documents sharing concepts
    edge_weights = defaultdict(int)
    for concept, doc_list in concept_to_docs.items():
        if len(doc_list) > 1:
            for i in range(len(doc_list)):
                for j in range(i + 1, len(doc_list)):
                    doc1, doc2 = doc_list[i], doc_list[j]
                    edge_weights[(doc1, doc2)] += 1

    # Add weighted edges (minimum 2 shared concepts)
    for (doc1, doc2), weight in edge_weights.items():
        if weight >= 2:
            # Calculate similarity bonus for same category
            bonus = (
                0.2 if documents[doc1]["category"] == documents[doc2]["category"] else 0
            )
            G.add_edge(doc1, doc2, weight=weight, similarity=weight / 10 + bonus)

    print(
        f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges"
    )
    return G


def visualize_knowledge_graph(G: nx.Graph, output_file: str = "knowledge_graph.png"):
    """Create interactive visualization of the knowledge graph."""
    if not VISUALIZATION_AVAILABLE:
        print("Visualization libraries not available")
        return None

    # Set up the plot
    plt.figure(figsize=(20, 16))
    plt.suptitle(
        "AI Research Knowledge Graph\nConnections between Grossman Mathematics, Wikipedia, and AI Papers",
        fontsize=16,
        fontweight="bold",
    )

    # Define colors for different document types
    color_map = {
        "mathematics": "#FF6B6B",  # Red for math papers
        "rag_research": "#4ECDC4",  # Teal for RAG research
        "multiagent": "#45B7D1",  # Blue for multi-agent
        "compression": "#96CEB4",  # Green for compression
        "ai_research": "#FFEAA7",  # Yellow for general AI
        "general_research": "#DDA0DD",  # Purple for general
        "wikipedia": "#FFA07A",  # Orange for Wikipedia (if any)
    }

    # Create layout
    try:
        # Use spring layout with customization
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
    except:
        # Fallback to circular layout
        pos = nx.circular_layout(G)

    # Prepare node attributes
    node_colors = [
        color_map.get(G.nodes[node].get("type", "general_research"), "#DDA0DD")
        for node in G.nodes()
    ]
    node_sizes = [
        max(100, min(2000, G.nodes[node].get("word_count", 1000) / 50))
        for node in G.nodes()
    ]

    # Draw edges first (behind nodes)
    edge_widths = [G[u][v].get("weight", 1) * 0.5 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, edge_color="gray")

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.8,
        edgecolors="black",
        linewidths=0.5,
    )

    # Add labels for important nodes
    important_nodes = {}
    for node in G.nodes():
        if (
            G.degree(node) >= 3 or "grossman" in node.lower()
        ):  # High degree or Grossman papers
            title = G.nodes[node].get("title", node)
            # Truncate long titles
            if len(title) > 25:
                title = title[:22] + "..."
            important_nodes[node] = title

    nx.draw_networkx_labels(G, pos, important_nodes, font_size=8, font_weight="bold")

    # Create legend
    legend_elements = []
    type_counts = Counter(G.nodes[node].get("type", "unknown") for node in G.nodes())

    for doc_type, color in color_map.items():
        if doc_type in type_counts:
            count = type_counts[doc_type]
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color,
                    markersize=10,
                    label=f"{doc_type.replace('_', ' ').title()} ({count})",
                )
            )

    plt.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(0.02, 0.98))

    # Add statistics text
    stats_text = f"""
    Graph Statistics:
    • Total Documents: {G.number_of_nodes()}
    • Connections: {G.number_of_edges()}
    • Avg Connections: {G.number_of_edges() * 2 / G.number_of_nodes():.1f}
    • Key Hubs: {len([n for n in G.nodes() if G.degree(n) >= 5])}
    """

    plt.text(
        0.02,
        0.15,
        stats_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
    )

    plt.axis("off")
    plt.tight_layout()

    # Save the plot
    output_path = Path(output_file)
    plt.savefig(
        output_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    print(f"Knowledge graph saved to: {output_path.absolute()}")

    # Show the plot
    plt.show()

    return pos


def analyze_key_connections(G: nx.Graph, documents: dict[str, dict]) -> None:
    """Analyze and report key connections in the graph."""
    print("\n" + "=" * 60)
    print("KEY KNOWLEDGE CONNECTIONS ANALYSIS")
    print("=" * 60)

    # Find Grossman papers
    grossman_papers = [node for node in G.nodes() if "grossman" in node.lower()]
    print(f"\n📚 Grossman Mathematical Papers Found: {len(grossman_papers)}")

    for paper in grossman_papers:
        connections = list(G.neighbors(paper))
        concepts = documents[paper]["concepts"][:5]  # Top 5 concepts

        print(f"\n  📖 {documents[paper]['title']}")
        print(f"     • Connected to: {len(connections)} papers")
        print(f"     • Key concepts: {', '.join(concepts)}")

        # Show strongest connections
        if connections:
            weighted_connections = [
                (conn, G[paper][conn].get("weight", 1)) for conn in connections
            ]
            weighted_connections.sort(key=lambda x: x[1], reverse=True)

            print("     • Strongest links:")
            for conn, weight in weighted_connections[:3]:
                conn_type = documents[conn]["type"]
                print(
                    f"       → {documents[conn]['title'][:40]}... ({conn_type}, {weight} shared concepts)"
                )

    # Find most connected papers
    print("\n🌟 Most Connected Papers (Knowledge Hubs):")
    degree_centrality = nx.degree_centrality(G)
    top_hubs = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]

    for i, (node, centrality) in enumerate(top_hubs, 1):
        degree = G.degree(node)
        doc_type = documents[node]["type"]
        print(f"   {i}. {documents[node]['title'][:50]}...")
        print(
            f"      Type: {doc_type}, Connections: {degree}, Centrality: {centrality:.3f}"
        )

    # Analyze cross-domain connections
    print("\n🔗 Cross-Domain Knowledge Bridges:")

    cross_domain_edges = []
    for u, v in G.edges():
        if documents[u]["type"] != documents[v]["type"]:
            weight = G[u][v].get("weight", 1)
            cross_domain_edges.append(
                (u, v, weight, documents[u]["type"], documents[v]["type"])
            )

    cross_domain_edges.sort(key=lambda x: x[2], reverse=True)

    print(f"   Found {len(cross_domain_edges)} cross-domain connections")
    for u, v, weight, type1, type2 in cross_domain_edges[:5]:
        print(f"   • {documents[u]['title'][:30]}... ↔ {documents[v]['title'][:30]}...")
        print(f"     {type1} ↔ {type2} ({weight} shared concepts)")

    # Find concept clusters
    print("\n🎯 Key Concept Clusters:")
    concept_freq = defaultdict(int)
    concept_docs = defaultdict(set)

    for doc_id, doc_data in documents.items():
        for concept in doc_data["concepts"]:
            concept_freq[concept] += 1
            concept_docs[concept].add(doc_id)

    top_concepts = sorted(concept_freq.items(), key=lambda x: x[1], reverse=True)[:8]

    for concept, freq in top_concepts:
        if freq >= 3:  # Concepts appearing in 3+ documents
            doc_types = {documents[doc]["type"] for doc in concept_docs[concept]}
            print(f"   • '{concept}': {freq} papers across {len(doc_types)} domains")
            print(f"     Domains: {', '.join(sorted(doc_types))}")


def main() -> None:
    """Main function to create and visualize the knowledge graph."""
    print("🔍 KNOWLEDGE GRAPH VISUALIZATION")
    print("=" * 50)
    print("Analyzing connections between:")
    print("• Grossman Mathematical Papers (Non-Newtonian Calculus, Meta-Calculus)")
    print("• Wikipedia Articles")
    print("• AI Research Papers (146 papers)")
    print()

    # Check if ingested papers exist
    papers_dir = Path("data/ingested_papers")
    if not papers_dir.exists():
        print("❌ Ingested papers directory not found. Run PDF ingestion first.")
        return

    print("📊 Step 1: Analyzing document relationships...")
    documents = analyze_document_relationships(papers_dir)

    if len(documents) < 10:
        print(f"⚠️  Warning: Only {len(documents)} documents found. Expected 145+")

    print("🕸️  Step 2: Building knowledge graph...")
    G = create_knowledge_graph(documents)

    print("🎨 Step 3: Creating visualization...")
    if VISUALIZATION_AVAILABLE:
        visualize_knowledge_graph(G)
    else:
        print(
            "❌ Visualization not available. Install matplotlib, networkx, scikit-learn"
        )

    print("🔍 Step 4: Analyzing key connections...")
    analyze_key_connections(G, documents)

    # Export graph data
    graph_data = {
        "nodes": len(G.nodes()),
        "edges": len(G.edges()),
        "document_types": dict(Counter(documents[node]["type"] for node in G.nodes())),
        "grossman_papers": len([n for n in G.nodes() if "grossman" in n.lower()]),
        "avg_degree": sum(dict(G.degree()).values()) / len(G.nodes())
        if G.nodes()
        else 0,
    }

    with open("knowledge_graph_stats.json", "w") as f:
        json.dump(graph_data, f, indent=2)

    print("\n✅ Knowledge graph analysis complete!")
    print("📈 Graph stats exported to: knowledge_graph_stats.json")
    print("🖼️  Visualization saved as: knowledge_graph.png")


if __name__ == "__main__":
    main()
