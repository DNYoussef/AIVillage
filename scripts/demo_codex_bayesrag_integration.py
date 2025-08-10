#!/usr/bin/env python3
"""
Demonstration of BayesRAG and CODEX integration concept.
Shows how BayesRAG enhancements can be integrated with existing CODEX infrastructure.
"""

import json
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Any

import requests

def test_current_codex_api():
    """Test current CODEX RAG API status."""
    
    print("=== Testing Current CODEX RAG API ===")
    
    try:
        # Test health endpoint
        response = requests.get("http://localhost:8082/health/rag", timeout=5)
        health_data = response.json()
        
        print(f"Health Status: {health_data}")
        print(f"API Accessible: {health_data.get('status') == 'healthy'}")
        print(f"Current Index Size: {health_data.get('index_size', 0)}")
        print(f"Average Latency: {health_data.get('avg_latency_ms', 0)}ms")
        
        return True
        
    except Exception as e:
        print(f"❌ CODEX API Error: {e}")
        return False

def analyze_bayesrag_data():
    """Analyze BayesRAG data for integration potential."""
    
    print("\n=== Analyzing BayesRAG Data for Integration ===")
    
    data_dir = Path("../data")
    
    # Check BayesRAG databases
    databases = {
        'Global Contexts': data_dir / "wikipedia_global_context.db",
        'Local Contexts': data_dir / "wikipedia_local_context.db", 
        'Knowledge Graph': data_dir / "wikipedia_graph.db"
    }
    
    integration_data = {}
    
    for db_name, db_path in databases.items():
        if db_path.exists():
            print(f"Found {db_name}: {db_path}")
            
            try:
                with sqlite3.connect(db_path) as conn:
                    if 'global' in db_name.lower():
                        cursor = conn.execute("SELECT COUNT(*) FROM global_contexts")
                        count = cursor.fetchone()[0]
                        print(f"   📚 Articles: {count}")
                        
                        # Get trust score distribution
                        cursor = conn.execute("SELECT AVG(trust_score), MIN(trust_score), MAX(trust_score) FROM global_contexts")
                        avg_trust, min_trust, max_trust = cursor.fetchone()
                        print(f"   🔒 Trust Scores: avg={avg_trust:.3f}, range={min_trust:.3f}-{max_trust:.3f}")
                        
                        integration_data['articles'] = count
                        integration_data['avg_trust'] = avg_trust
                        
                    elif 'local' in db_name.lower():
                        cursor = conn.execute("SELECT COUNT(*) FROM local_contexts")
                        count = cursor.fetchone()[0]
                        print(f"   📝 Chunks: {count}")
                        
                        # Get context coverage
                        cursor = conn.execute("SELECT COUNT(*) FROM local_contexts WHERE temporal_context IS NOT NULL")
                        temporal_count = cursor.fetchone()[0]
                        cursor = conn.execute("SELECT COUNT(*) FROM local_contexts WHERE geographic_context IS NOT NULL") 
                        geo_count = cursor.fetchone()[0]
                        
                        temporal_coverage = (temporal_count / count) * 100 if count > 0 else 0
                        geo_coverage = (geo_count / count) * 100 if count > 0 else 0
                        
                        print(f"   🏷️  Temporal Coverage: {temporal_coverage:.1f}%")
                        print(f"   🌍 Geographic Coverage: {geo_coverage:.1f}%")
                        
                        integration_data['chunks'] = count
                        integration_data['temporal_coverage'] = temporal_coverage
                        integration_data['geo_coverage'] = geo_coverage
                        
                    elif 'graph' in db_name.lower():
                        cursor = conn.execute("SELECT COUNT(*) FROM graph_edges")
                        count = cursor.fetchone()[0]
                        print(f"   🕸️  Relationships: {count}")
                        
                        # Get average trust weight
                        cursor = conn.execute("SELECT AVG(trust_weight) FROM graph_edges")
                        avg_trust_weight = cursor.fetchone()[0]
                        print(f"   ⭐ Avg Relationship Trust: {avg_trust_weight:.3f}")
                        
                        integration_data['relationships'] = count
                        integration_data['relationship_trust'] = avg_trust_weight
                        
            except Exception as e:
                print(f"   ❌ Error reading {db_name}: {e}")
        else:
            print(f"❌ {db_name}: Not found at {db_path}")
            
    return integration_data

def demonstrate_integration_potential(integration_data: Dict[str, Any]):
    """Demonstrate how BayesRAG data enhances CODEX capabilities."""
    
    print("\n=== Integration Enhancement Potential ===")
    
    articles = integration_data.get('articles', 0)
    chunks = integration_data.get('chunks', 0)
    relationships = integration_data.get('relationships', 0)
    avg_trust = integration_data.get('avg_trust', 0)
    
    print(f"📊 Current BayesRAG Data Scale:")
    print(f"   • {articles} Wikipedia articles with hierarchical context")
    print(f"   • {chunks} contextual chunks with embeddings")
    print(f"   • {relationships:,} semantic relationships")
    print(f"   • {avg_trust:.3f} average trust score")
    
    print(f"\n🚀 Integration Benefits for CODEX:")
    print(f"   • Hierarchical Context: Global summaries + local details")
    print(f"   • Trust-Weighted Results: Bayesian reliability scoring") 
    print(f"   • Cross-Reference Discovery: {relationships:,} knowledge graph connections")
    print(f"   • Enhanced Metadata: Temporal/geographic/topical context")
    print(f"   • Query Intelligence: Multi-level query understanding")
    
    print(f"\n📈 Scale Enhancement Needed:")
    target_articles = 1000
    scale_factor = target_articles / max(articles, 1)
    print(f"   • Target: {target_articles} articles (scale factor: {scale_factor:.1f}x)")
    print(f"   • Projected chunks: ~{int(chunks * scale_factor):,}")
    print(f"   • Projected relationships: ~{int(relationships * scale_factor):,}")
    
    print(f"\n⚡ Performance Optimization Opportunities:")
    print(f"   • Context-aware caching: Cache by trust score and context hierarchy")
    print(f"   • Semantic cache matching: Use embeddings for cache similarity")
    print(f"   • Trust-based prefetching: Pre-load high-trust content")
    print(f"   • Graph-accelerated retrieval: Use relationships for expansion")

def test_sample_integration_query():
    """Test how a sample query would work with integrated system."""
    
    print("\n=== Sample Integrated Query Demonstration ===")
    
    # Sample query that would benefit from BayesRAG enhancements
    query = "What caused World War I in Europe?"
    print(f"Sample Query: '{query}'")
    
    print(f"\nTraditional CODEX Response:")
    print(f"   • Standard embedding similarity search")
    print(f"   • FAISS + BM25 hybrid retrieval") 
    print(f"   • Basic relevance scoring")
    
    print(f"\nBayesRAG-Enhanced CODEX Response:")
    print(f"   • Query context analysis: temporal='1914-1918', geographic='Europe', topic='history'")
    print(f"   • Hierarchical retrieval: Global overview + local details")
    print(f"   • Trust-weighted ranking: Results scored by source reliability")
    print(f"   • Cross-reference expansion: Related causes, consequences, key figures")
    print(f"   • Context-aware caching: Cache hits for similar historical queries")
    
    print(f"\nExpected Performance:")
    print(f"   • Standard CODEX: ~50-100ms retrieval")
    print(f"   • BayesRAG-Enhanced: ~10-50ms (due to context-aware caching)")
    print(f"   • Improved relevance: Trust scores prioritize authoritative sources")
    print(f"   • Richer results: Global context + detailed local information")

def main():
    """Run the complete integration demonstration."""
    
    print("BayesRAG + CODEX Integration Demonstration")
    print("=" * 60)
    
    # Test current CODEX API
    codex_accessible = test_current_codex_api()
    
    # Analyze BayesRAG data
    integration_data = analyze_bayesrag_data()
    
    # Demonstrate integration potential
    demonstrate_integration_potential(integration_data)
    
    # Show sample query enhancement
    test_sample_integration_query()
    
    print("\n" + "=" * 60)
    print("🎉 Integration Demonstration Complete")
    
    if codex_accessible and integration_data.get('chunks', 0) > 0:
        print("✅ Ready for Phase 1 integration: BayesRAG data → CODEX pipeline")
        print("📋 Next Steps:")
        print("   1. Run migration script to transfer BayesRAG data to CODEX")
        print("   2. Enhance CODEX caching with semantic matching")
        print("   3. Scale BayesRAG ingestion to 1000+ articles")
        print("   4. Optimize performance for <100ms latency target")
    else:
        print("⚠️ Prerequisites missing:")
        if not codex_accessible:
            print("   • CODEX API not accessible on port 8082")
        if integration_data.get('chunks', 0) == 0:
            print("   • BayesRAG data not found (run BayesRAG pipeline first)")
            
    print("\n📚 See CODEX_RAG_INTEGRATION_PLAN.md for detailed implementation roadmap")

if __name__ == "__main__":
    main()