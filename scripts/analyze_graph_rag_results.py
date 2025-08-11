#!/usr/bin/env python3
"""Analyze Graph RAG Results and Identify Knowledge Gaps.

This script analyzes the chunked documents and identifies gaps in the knowledge graph
that need to be filled, providing the analysis the user requested.
"""

import json
import logging
import re
import time
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("graph_rag_analysis.log"),
    ],
)
logger = logging.getLogger(__name__)


class GraphRagAnalyzer:
    """Analyzer for Graph RAG results and knowledge gaps."""
    
    def __init__(self, ingested_papers_dir: str = "data/ingested_papers"):
        self.ingested_papers_dir = Path(ingested_papers_dir)
        self.documents = {}
        self.chunks = {}
        self.topics = defaultdict(list)
        self.entities = defaultdict(list)
        self.knowledge_gaps = []
        
    def load_and_analyze_documents(self):
        """Load and analyze all ingested documents."""
        logger.info("Loading and analyzing documents...")
        
        if not self.ingested_papers_dir.exists():
            logger.error(f"Directory {self.ingested_papers_dir} does not exist")
            return
            
        # Process all full text files
        full_text_files = list(self.ingested_papers_dir.glob("*_full.txt"))
        chunk_files = list(self.ingested_papers_dir.glob("*_chunks.txt"))
        
        logger.info(f"Found {len(full_text_files)} full text files and {len(chunk_files)} chunk files")
        
        for full_file in full_text_files:
            doc_id = full_file.stem.replace("_full", "")
            
            try:
                # Load full text
                with open(full_file, 'r', encoding='utf-8') as f:
                    full_text = f.read()
                
                # Load chunks if available
                chunk_file = self.ingested_papers_dir / f"{doc_id}_chunks.txt"
                chunks = []
                if chunk_file.exists():
                    with open(chunk_file, 'r', encoding='utf-8') as f:
                        chunk_text = f.read()
                        # Split by common chunk separators
                        chunks = [c.strip() for c in re.split(r'\n\s*\n\s*---+\s*\n\s*\n', chunk_text) if c.strip()]
                
                # Analyze document
                self.documents[doc_id] = {
                    'title': self._extract_title(full_text, doc_id),
                    'full_text': full_text,
                    'chunks': chunks,
                    'word_count': len(full_text.split()),
                    'chunk_count': len(chunks),
                    'topics': self._extract_topics(full_text),
                    'entities': self._extract_entities(full_text),
                    'quality_score': self._calculate_quality_score(full_text),
                    'credibility_score': self._calculate_credibility_score(doc_id, full_text),
                }
                
                # Store chunks for analysis
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{doc_id}_chunk_{i}"
                    self.chunks[chunk_id] = {
                        'document_id': doc_id,
                        'text': chunk,
                        'position': i,
                        'word_count': len(chunk.split()),
                        'topics': self._extract_topics(chunk),
                        'entities': self._extract_entities(chunk),
                    }
                    
            except Exception as e:
                logger.error(f"Failed to process {full_file}: {e}")
                continue
                
        logger.info(f"Loaded {len(self.documents)} documents with {len(self.chunks)} total chunks")
    
    def _extract_title(self, text: str, doc_id: str) -> str:
        """Extract document title from text or filename."""
        # Try to find title in first few lines
        lines = text.split('\n')[:5]
        for line in lines:
            line = line.strip()
            if len(line) > 10 and len(line) < 200 and not line.startswith('Abstract'):
                if any(word in line.upper() for word in ['ABSTRACT', 'INTRODUCTION', 'TITLE']):
                    continue
                return line
        
        # Fall back to document ID
        return doc_id.replace('_', ' ').replace('-', ' ').title()
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from text using keyword analysis."""
        # Common AI/ML/research topics
        topic_keywords = {
            'machine_learning': ['machine learning', 'ml', 'neural network', 'deep learning', 'artificial intelligence'],
            'natural_language_processing': ['nlp', 'natural language', 'text processing', 'language model', 'transformer'],
            'computer_vision': ['computer vision', 'image', 'visual', 'cnn', 'convolutional'],
            'reinforcement_learning': ['reinforcement learning', 'rl', 'reward', 'policy', 'agent'],
            'optimization': ['optimization', 'gradient', 'loss', 'minimize', 'objective'],
            'data_science': ['data science', 'analytics', 'statistics', 'data mining'],
            'robotics': ['robot', 'robotics', 'autonomous', 'control'],
            'quantum': ['quantum', 'qubit', 'quantum computing'],
            'compression': ['compression', 'quantization', 'pruning', 'distillation'],
            'retrieval': ['retrieval', 'search', 'rag', 'retrieval augmented'],
            'graph': ['graph', 'network', 'node', 'edge', 'knowledge graph'],
            'attention': ['attention', 'transformer', 'self-attention'],
            'training': ['training', 'fine-tuning', 'learning', 'optimization'],
        }
        
        text_lower = text.lower()
        found_topics = []
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                found_topics.append(topic)
        
        return found_topics
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text using simple pattern matching."""
        # Simple entity extraction using common patterns
        entities = []
        
        # Model names
        model_patterns = [
            r'\b(?:GPT|BERT|T5|RoBERTa|ELECTRA|DeBERTa|ALBERT|DistilBERT)-?\w*\b',
            r'\b(?:ResNet|VGG|AlexNet|Inception|MobileNet|EfficientNet)\b',
            r'\b(?:LLaMA|LLaMa|Claude|PaLM|Gemini|Bard)\b',
        ]
        
        for pattern in model_patterns:
            entities.extend(re.findall(pattern, text, re.IGNORECASE))
        
        # Metrics and methods
        method_patterns = [
            r'\b(?:BLEU|ROUGE|F1|accuracy|precision|recall|AUC|ROC)\b',
            r'\b(?:Adam|SGD|RMSprop|AdaGrad|momentum)\b',
            r'\b(?:cross-entropy|MSE|MAE|cosine similarity)\b',
        ]
        
        for pattern in method_patterns:
            entities.extend(re.findall(pattern, text, re.IGNORECASE))
        
        return list(set(entities))
    
    def _calculate_quality_score(self, text: str) -> float:
        """Calculate document quality score based on various factors."""
        score = 0.5  # Base score
        
        # Length bonus (longer documents often more comprehensive)
        word_count = len(text.split())
        if word_count > 1000:
            score += 0.1
        if word_count > 5000:
            score += 0.1
        
        # Structure bonus (has abstract, introduction, conclusion)
        text_lower = text.lower()
        if 'abstract' in text_lower:
            score += 0.1
        if 'introduction' in text_lower:
            score += 0.1
        if any(word in text_lower for word in ['conclusion', 'conclusions', 'summary']):
            score += 0.1
        
        # Citation bonus (has references)
        if any(word in text_lower for word in ['references', 'bibliography', 'citation']):
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_credibility_score(self, doc_id: str, text: str) -> float:
        """Calculate document credibility score."""
        score = 0.6  # Base score
        
        # ArXiv paper bonus
        if re.match(r'\d{4}\.\d{5}', doc_id):
            score += 0.2
        
        # Academic indicators
        text_lower = text.lower()
        if any(word in text_lower for word in ['university', 'research', 'institute']):
            score += 0.1
        if any(word in text_lower for word in ['peer review', 'journal', 'conference']):
            score += 0.1
        
        return min(score, 1.0)
    
    def analyze_knowledge_gaps(self):
        """Analyze the knowledge graph for gaps and holes."""
        logger.info("Analyzing knowledge gaps...")
        
        # Topic coverage analysis
        topic_coverage = defaultdict(int)
        topic_quality = defaultdict(list)
        
        for doc_id, doc in self.documents.items():
            for topic in doc['topics']:
                topic_coverage[topic] += 1
                topic_quality[topic].append(doc['quality_score'])
        
        # Identify under-covered topics
        total_docs = len(self.documents)
        for topic, count in topic_coverage.items():
            coverage_ratio = count / total_docs
            avg_quality = sum(topic_quality[topic]) / len(topic_quality[topic])
            
            if coverage_ratio < 0.1:  # Less than 10% coverage
                self.knowledge_gaps.append({
                    'type': 'low_coverage_topic',
                    'topic': topic,
                    'coverage_ratio': coverage_ratio,
                    'document_count': count,
                    'avg_quality': avg_quality,
                    'severity': 'high' if coverage_ratio < 0.05 else 'medium'
                })
        
        # Entity co-occurrence analysis
        entity_cooccurrence = defaultdict(lambda: defaultdict(int))
        for doc_id, doc in self.documents.items():
            entities = doc['entities']
            for i, entity1 in enumerate(entities):
                for entity2 in entities[i+1:]:
                    entity_cooccurrence[entity1][entity2] += 1
                    entity_cooccurrence[entity2][entity1] += 1
        
        # Find isolated entities (mentioned but not connected)
        all_entities = set()
        for doc in self.documents.values():
            all_entities.update(doc['entities'])
        
        isolated_entities = []
        for entity in all_entities:
            if len(entity_cooccurrence[entity]) < 2:  # Connected to fewer than 2 other entities
                isolated_entities.append(entity)
        
        if isolated_entities:
            self.knowledge_gaps.append({
                'type': 'isolated_entities',
                'entities': isolated_entities[:20],  # Limit to top 20
                'count': len(isolated_entities),
                'severity': 'medium'
            })
        
        # Chunk connectivity analysis
        chunk_topics = defaultdict(list)
        for chunk_id, chunk in self.chunks.items():
            for topic in chunk['topics']:
                chunk_topics[topic].append(chunk_id)
        
        # Find topics with fragmented chunks
        fragmented_topics = []
        for topic, chunk_ids in chunk_topics.items():
            if len(chunk_ids) > 5:  # Only analyze topics with sufficient chunks
                # Analyze if chunks are from diverse documents (good) or isolated (bad)
                doc_distribution = defaultdict(int)
                for chunk_id in chunk_ids:
                    doc_id = chunk_id.split('_chunk_')[0]
                    doc_distribution[doc_id] += 1
                
                # If topic appears in only one document but has many chunks, it might be over-segmented
                if len(doc_distribution) == 1 and len(chunk_ids) > 10:
                    fragmented_topics.append({
                        'topic': topic,
                        'chunk_count': len(chunk_ids),
                        'document_count': len(doc_distribution),
                        'severity': 'low'
                    })
        
        if fragmented_topics:
            self.knowledge_gaps.append({
                'type': 'over_segmented_topics',
                'topics': fragmented_topics,
                'severity': 'low'
            })
        
        # Cross-document connectivity analysis
        cross_doc_connections = defaultdict(set)
        for chunk_id, chunk in self.chunks.items():
            doc_id = chunk_id.split('_chunk_')[0]
            for topic in chunk['topics']:
                for other_chunk_id, other_chunk in self.chunks.items():
                    other_doc_id = other_chunk_id.split('_chunk_')[0]
                    if doc_id != other_doc_id and topic in other_chunk['topics']:
                        cross_doc_connections[topic].add((doc_id, other_doc_id))
        
        # Find topics with poor cross-document connectivity
        poorly_connected_topics = []
        for topic, connections in cross_doc_connections.items():
            if topic in topic_coverage and topic_coverage[topic] > 3:  # Topic appears in multiple docs
                connection_ratio = len(connections) / (topic_coverage[topic] * (topic_coverage[topic] - 1) / 2)
                if connection_ratio < 0.3:  # Less than 30% of potential connections
                    poorly_connected_topics.append({
                        'topic': topic,
                        'connection_ratio': connection_ratio,
                        'potential_connections': topic_coverage[topic] * (topic_coverage[topic] - 1) // 2,
                        'actual_connections': len(connections),
                        'severity': 'medium'
                    })
        
        if poorly_connected_topics:
            self.knowledge_gaps.append({
                'type': 'poorly_connected_topics',
                'topics': poorly_connected_topics,
                'severity': 'medium'
            })
        
        logger.info(f"Identified {len(self.knowledge_gaps)} knowledge gap patterns")
    
    def generate_repair_recommendations(self):
        """Generate recommendations for repairing knowledge gaps."""
        logger.info("Generating repair recommendations...")
        
        recommendations = []
        
        for gap in self.knowledge_gaps:
            if gap['type'] == 'low_coverage_topic':
                recommendations.append({
                    'gap_id': f"low_coverage_{gap['topic']}",
                    'priority': 'high' if gap['severity'] == 'high' else 'medium',
                    'action': 'ADD_CONTENT',
                    'description': f"Add more documents covering {gap['topic']} (currently only {gap['document_count']} documents)",
                    'suggested_sources': self._suggest_sources_for_topic(gap['topic']),
                    'estimated_effort': 'high'
                })
            
            elif gap['type'] == 'isolated_entities':
                recommendations.append({
                    'gap_id': 'isolated_entities',
                    'priority': 'medium',
                    'action': 'ADD_RELATIONSHIPS',
                    'description': f"Create relationships for {gap['count']} isolated entities",
                    'entities_sample': gap['entities'][:10],
                    'estimated_effort': 'medium'
                })
            
            elif gap['type'] == 'poorly_connected_topics':
                for topic_info in gap['topics']:
                    recommendations.append({
                        'gap_id': f"poor_connection_{topic_info['topic']}",
                        'priority': 'medium',
                        'action': 'STRENGTHEN_CONNECTIONS',
                        'description': f"Improve cross-document connections for {topic_info['topic']} topic",
                        'current_ratio': topic_info['connection_ratio'],
                        'target_ratio': 0.5,
                        'estimated_effort': 'low'
                    })
        
        return recommendations
    
    def _suggest_sources_for_topic(self, topic: str) -> List[str]:
        """Suggest potential sources for under-covered topics."""
        source_suggestions = {
            'quantum': ['ArXiv quantum computing papers', 'Nature Quantum Information', 'Physical Review Quantum'],
            'robotics': ['IEEE Robotics journals', 'International Journal of Robotics Research'],
            'computer_vision': ['CVPR papers', 'ICCV proceedings', 'IEEE PAMI'],
            'natural_language_processing': ['ACL papers', 'EMNLP proceedings', 'NAACL papers'],
            'reinforcement_learning': ['ICML RL papers', 'NeurIPS RL track', 'JMLR RL papers'],
        }
        
        return source_suggestions.get(topic, ['Related academic papers', 'Conference proceedings', 'Journal articles'])
    
    def export_analysis_results(self):
        """Export analysis results to files."""
        logger.info("Exporting analysis results...")
        
        results_dir = Path("data/graph_rag_analysis")
        results_dir.mkdir(exist_ok=True)
        
        # Document statistics
        doc_stats = {
            'total_documents': len(self.documents),
            'total_chunks': len(self.chunks),
            'avg_chunks_per_document': len(self.chunks) / len(self.documents) if self.documents else 0,
            'topic_distribution': dict(Counter([topic for doc in self.documents.values() for topic in doc['topics']])),
            'entity_distribution': dict(Counter([entity for doc in self.documents.values() for entity in doc['entities']])),
            'quality_distribution': {
                'high_quality': len([d for d in self.documents.values() if d['quality_score'] > 0.8]),
                'medium_quality': len([d for d in self.documents.values() if 0.6 < d['quality_score'] <= 0.8]),
                'low_quality': len([d for d in self.documents.values() if d['quality_score'] <= 0.6]),
            },
            'credibility_distribution': {
                'high_credibility': len([d for d in self.documents.values() if d['credibility_score'] > 0.8]),
                'medium_credibility': len([d for d in self.documents.values() if 0.6 < d['credibility_score'] <= 0.8]),
                'low_credibility': len([d for d in self.documents.values() if d['credibility_score'] <= 0.6]),
            }
        }
        
        with open(results_dir / "document_statistics.json", 'w') as f:
            json.dump(doc_stats, f, indent=2)
        
        # Knowledge gaps
        gap_analysis = {
            'total_gaps_identified': len(self.knowledge_gaps),
            'gaps_by_severity': Counter([gap.get('severity', 'unknown') for gap in self.knowledge_gaps]),
            'gaps_by_type': Counter([gap['type'] for gap in self.knowledge_gaps]),
            'detailed_gaps': self.knowledge_gaps
        }
        
        with open(results_dir / "knowledge_gaps.json", 'w') as f:
            json.dump(gap_analysis, f, indent=2)
        
        # Repair recommendations
        recommendations = self.generate_repair_recommendations()
        repair_analysis = {
            'total_recommendations': len(recommendations),
            'recommendations_by_priority': Counter([r['priority'] for r in recommendations]),
            'recommendations_by_action': Counter([r['action'] for r in recommendations]),
            'detailed_recommendations': recommendations
        }
        
        with open(results_dir / "repair_recommendations.json", 'w') as f:
            json.dump(repair_analysis, f, indent=2)
        
        # Sample chunks for verification
        sample_chunks = {}
        for i, (chunk_id, chunk) in enumerate(list(self.chunks.items())[:20]):
            sample_chunks[chunk_id] = {
                'text': chunk['text'][:200] + '...' if len(chunk['text']) > 200 else chunk['text'],
                'topics': chunk['topics'],
                'entities': chunk['entities'],
                'word_count': chunk['word_count']
            }
        
        with open(results_dir / "sample_chunks.json", 'w') as f:
            json.dump(sample_chunks, f, indent=2)
        
        # Summary report
        summary = {
            'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'processing_summary': {
                'documents_processed': len(self.documents),
                'chunks_created': len(self.chunks),
                'topics_identified': len(set([topic for doc in self.documents.values() for topic in doc['topics']])),
                'entities_extracted': len(set([entity for doc in self.documents.values() for entity in doc['entities']])),
            },
            'gap_analysis_summary': {
                'total_gaps': len(self.knowledge_gaps),
                'high_severity_gaps': len([g for g in self.knowledge_gaps if g.get('severity') == 'high']),
                'medium_severity_gaps': len([g for g in self.knowledge_gaps if g.get('severity') == 'medium']),
                'low_severity_gaps': len([g for g in self.knowledge_gaps if g.get('severity') == 'low']),
            },
            'repair_recommendations_summary': {
                'total_recommendations': len(recommendations),
                'high_priority': len([r for r in recommendations if r['priority'] == 'high']),
                'medium_priority': len([r for r in recommendations if r['priority'] == 'medium']),
                'low_priority': len([r for r in recommendations if r['priority'] == 'low']),
            }
        }
        
        with open(results_dir / "analysis_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Analysis results exported to {results_dir}")
        return results_dir, summary


def main():
    """Main analysis function."""
    logger.info("Starting Graph RAG analysis...")
    
    # Initialize analyzer
    analyzer = GraphRagAnalyzer()
    
    # Load and analyze documents
    analyzer.load_and_analyze_documents()
    
    if not analyzer.documents:
        logger.error("No documents loaded. Check the ingested_papers directory.")
        return
    
    # Analyze knowledge gaps
    analyzer.analyze_knowledge_gaps()
    
    # Export results
    results_dir, summary = analyzer.export_analysis_results()
    
    # Print summary
    print("\n" + "="*70)
    print("GRAPH RAG ANALYSIS COMPLETE")
    print("="*70)
    print(f"Documents processed: {summary['processing_summary']['documents_processed']}")
    print(f"Chunks created: {summary['processing_summary']['chunks_created']}")
    print(f"Topics identified: {summary['processing_summary']['topics_identified']}")
    print(f"Entities extracted: {summary['processing_summary']['entities_extracted']}")
    print("\nKnowledge Gap Analysis:")
    print(f"  Total gaps identified: {summary['gap_analysis_summary']['total_gaps']}")
    print(f"  High severity: {summary['gap_analysis_summary']['high_severity_gaps']}")
    print(f"  Medium severity: {summary['gap_analysis_summary']['medium_severity_gaps']}")
    print(f"  Low severity: {summary['gap_analysis_summary']['low_severity_gaps']}")
    print("\nRepair Recommendations:")
    print(f"  Total recommendations: {summary['repair_recommendations_summary']['total_recommendations']}")
    print(f"  High priority: {summary['repair_recommendations_summary']['high_priority']}")
    print(f"  Medium priority: {summary['repair_recommendations_summary']['medium_priority']}")
    print(f"  Low priority: {summary['repair_recommendations_summary']['low_priority']}")
    print(f"\nResults saved to: {results_dir}")
    print("="*70)


if __name__ == "__main__":
    main()