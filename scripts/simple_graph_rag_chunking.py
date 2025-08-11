#!/usr/bin/env python3
"""Simple Graph RAG Chunking and Integration.

This script processes all ingested PDFs and creates a knowledge graph with:
1. Intelligent chunking of all documents
2. Semantic relationship detection between chunks
3. Bayesian trust weighting based on document credibility
4. Knowledge gap detection and basic repair
5. Export of results for analysis
"""

import asyncio
import json
import logging
import re
import time
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import hashlib
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    import networkx as nx
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    print("Installing required dependencies...")
    import os
    os.system("pip install sentence-transformers networkx")
    try:
        from sentence_transformers import SentenceTransformer
        import networkx as nx
        DEPENDENCIES_AVAILABLE = True
    except ImportError:
        DEPENDENCIES_AVAILABLE = False
        print("Failed to install dependencies. Using fallback implementation.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("graph_rag_chunking.log"),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class SimpleChunk:
    """Simple chunk representation."""
    
    chunk_id: str
    document_id: str
    text: str
    position: int
    start_idx: int = 0
    end_idx: int = 0
    
    # Quality metrics
    coherence_score: float = 0.7
    trust_score: float = 0.7
    quality_score: float = 0.7
    
    # Content analysis
    keywords: List[str] = None
    entities: List[str] = None
    topic_category: str = "general"
    
    # Embeddings
    embedding: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
        if self.entities is None:
            self.entities = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        if self.embedding is not None:
            data['embedding'] = self.embedding.tolist()
        return data


@dataclass
class SimpleRelationship:
    """Simple relationship between chunks."""
    
    source_chunk: str
    target_chunk: str
    relationship_type: str
    confidence: float
    weight: float
    semantic_similarity: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ProcessingStats:
    """Processing statistics."""
    
    total_documents: int = 0
    total_chunks: int = 0
    total_relationships: int = 0
    graph_nodes: int = 0
    graph_edges: int = 0
    processing_time: float = 0.0
    avg_chunk_quality: float = 0.0
    avg_trust_score: float = 0.0
    knowledge_gaps_detected: int = 0
    repairs_applied: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SimpleGraphRAGChunker:
    """Simple Graph RAG chunking system with Bayesian weighting."""
    
    def __init__(
        self,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        similarity_threshold: float = 0.4,
        trust_decay: float = 0.85,
    ):
        """Initialize the chunker."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold
        self.trust_decay = trust_decay
        
        # Initialize components
        if DEPENDENCIES_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
                self.graph = nx.DiGraph()
                logger.info("Initialized with full dependencies")
            except Exception as e:
                logger.warning(f"Failed to initialize sentence transformer: {e}")
                self.embedding_model = None
                self.graph = None
        else:
            self.embedding_model = None
            self.graph = None
            logger.info("Using fallback mode without embeddings")
        
        # Data storage
        self.chunks: Dict[str, SimpleChunk] = {}
        self.relationships: Dict[Tuple[str, str], SimpleRelationship] = {}
        self.document_info: Dict[str, Dict[str, Any]] = {}
        
        # Statistics
        self.stats = ProcessingStats()
    
    async def process_all_documents(self, papers_dir: str = "data/ingested_papers") -> ProcessingStats:
        """Process all ingested documents."""
        
        start_time = time.perf_counter()
        logger.info("Starting document processing...")
        
        papers_path = Path(papers_dir)
        if not papers_path.exists():
            logger.error(f"Papers directory not found: {papers_dir}")
            return self.stats
        
        # Get all full text files
        full_text_files = list(papers_path.glob("*_full.txt"))
        logger.info(f"Found {len(full_text_files)} documents to process")
        
        # Process each document
        for i, file_path in enumerate(full_text_files):
            try:
                await self._process_single_document(file_path)
                
                if (i + 1) % 20 == 0:
                    logger.info(f"Processed {i + 1}/{len(full_text_files)} documents")
                    
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                continue
        
        # Build relationships between chunks
        await self._build_chunk_relationships()
        
        # Apply Bayesian trust propagation
        await self._propagate_trust_scores()
        
        # Detect knowledge gaps
        await self._detect_knowledge_gaps()
        
        # Apply basic repairs
        await self._apply_basic_repairs()
        
        # Update final statistics
        processing_time = time.perf_counter() - start_time
        self.stats.processing_time = processing_time
        
        if self.chunks:
            self.stats.avg_chunk_quality = sum(c.quality_score for c in self.chunks.values()) / len(self.chunks)
            self.stats.avg_trust_score = sum(c.trust_score for c in self.chunks.values()) / len(self.chunks)
        
        logger.info(f"Processing completed in {processing_time:.1f}s")
        return self.stats
    
    async def _process_single_document(self, file_path: Path) -> None:
        """Process a single document."""
        
        try:
            # Read content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse content and metadata
            if "=== CONTENT ===" in content:
                metadata_section, text_content = content.split("=== CONTENT ===", 1)
                
                # Parse metadata
                metadata = {}
                for line in metadata_section.split('\n'):
                    if ':' in line and not line.startswith('==='):
                        key, value = line.split(':', 1)
                        metadata[key.strip()] = value.strip()
            else:
                text_content = content
                metadata = {}
            
            # Document information
            doc_id = file_path.stem.replace('_full', '')
            doc_type = self._classify_document_type(text_content, doc_id)
            base_credibility = self._calculate_base_credibility(metadata, doc_type, text_content)
            
            self.document_info[doc_id] = {
                'title': metadata.get('title', doc_id.replace('_', ' ').title()),
                'doc_type': doc_type,
                'base_credibility': base_credibility,
                'metadata': metadata,
                'file_path': str(file_path),
            }
            
            # Create chunks
            chunks = self._create_intelligent_chunks(text_content, doc_id, base_credibility)
            
            # Store chunks
            for chunk in chunks:
                self.chunks[chunk.chunk_id] = chunk
            
            self.stats.total_documents += 1
            self.stats.total_chunks += len(chunks)
            
            logger.debug(f"Processed {doc_id}: {len(chunks)} chunks created")
            
        except Exception as e:
            logger.error(f"Failed to process document {file_path}: {e}")
            raise
    
    def _create_intelligent_chunks(
        self, 
        text: str, 
        doc_id: str, 
        base_credibility: float
    ) -> List[SimpleChunk]:
        """Create intelligent chunks from text."""
        
        chunks = []
        
        # Simple sentence-aware chunking
        sentences = self._split_into_sentences(text)
        
        current_chunk_text = ""
        current_chunk_sentences = []
        chunk_position = 0
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk_text) + len(sentence) > self.chunk_size and current_chunk_text:
                # Create chunk from accumulated sentences
                chunk = self._create_chunk_from_text(
                    current_chunk_text,
                    doc_id,
                    chunk_position,
                    base_credibility
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk_sentences[-2:] if len(current_chunk_sentences) > 2 else current_chunk_sentences
                current_chunk_text = " ".join(overlap_sentences)
                current_chunk_sentences = overlap_sentences[:]
                chunk_position += 1
            
            # Add sentence to current chunk
            current_chunk_text += " " + sentence if current_chunk_text else sentence
            current_chunk_sentences.append(sentence)
        
        # Create final chunk if there's remaining text
        if current_chunk_text.strip():
            chunk = self._create_chunk_from_text(
                current_chunk_text,
                doc_id,
                chunk_position,
                base_credibility
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk_from_text(
        self, 
        text: str, 
        doc_id: str, 
        position: int, 
        base_credibility: float
    ) -> SimpleChunk:
        """Create a chunk from text with analysis."""
        
        # Generate unique chunk ID
        chunk_id = f"{doc_id}_chunk_{position}_{hash(text) % 10000}"
        
        # Extract keywords and entities
        keywords = self._extract_keywords(text)
        entities = self._extract_entities(text)
        topic_category = self._classify_topic(text, keywords)
        
        # Calculate quality scores
        coherence_score = self._calculate_coherence_score(text, keywords)
        quality_score = self._calculate_quality_score(text, coherence_score)
        
        # Create embedding if available
        embedding = None
        if self.embedding_model:
            try:
                embedding = self.embedding_model.encode(text[:512])  # Limit text length
            except Exception as e:
                logger.warning(f"Failed to create embedding for chunk {chunk_id}: {e}")
        
        chunk = SimpleChunk(
            chunk_id=chunk_id,
            document_id=doc_id,
            text=text,
            position=position,
            coherence_score=coherence_score,
            trust_score=base_credibility,
            quality_score=quality_score,
            keywords=keywords,
            entities=entities,
            topic_category=topic_category,
            embedding=embedding,
        )
        
        return chunk
    
    async def _build_chunk_relationships(self) -> None:
        """Build relationships between chunks."""
        
        logger.info("Building relationships between chunks...")
        
        chunk_list = list(self.chunks.values())
        relationships_created = 0
        
        # Process in batches to avoid memory issues with large datasets
        batch_size = 100
        total_comparisons = 0
        
        for i in range(0, len(chunk_list), batch_size):
            batch = chunk_list[i:i + batch_size]
            
            for j, chunk1 in enumerate(batch):
                # Compare with chunks from same document (sequential relationships)
                for chunk2 in chunk_list:
                    if chunk1.chunk_id == chunk2.chunk_id:
                        continue
                    
                    # Check if relationship should be created
                    relationship = await self._analyze_chunk_relationship(chunk1, chunk2)
                    if relationship:
                        self.relationships[(chunk1.chunk_id, chunk2.chunk_id)] = relationship
                        relationships_created += 1
                        
                        # Add to graph if available
                        if self.graph:
                            self.graph.add_edge(
                                chunk1.chunk_id, 
                                chunk2.chunk_id, 
                                **relationship.to_dict()
                            )
                
                total_comparisons += 1
                
                if total_comparisons % 100 == 0:
                    logger.debug(f"Processed {total_comparisons} chunks for relationships")
        
        self.stats.total_relationships = relationships_created
        logger.info(f"Created {relationships_created} relationships between chunks")
    
    async def _analyze_chunk_relationship(
        self, 
        chunk1: SimpleChunk, 
        chunk2: SimpleChunk
    ) -> Optional[SimpleRelationship]:
        """Analyze potential relationship between two chunks."""
        
        # Sequential relationship (same document, adjacent positions)
        if (chunk1.document_id == chunk2.document_id and 
            abs(chunk1.position - chunk2.position) == 1):
            
            return SimpleRelationship(
                source_chunk=chunk1.chunk_id,
                target_chunk=chunk2.chunk_id,
                relationship_type="continues",
                confidence=0.9,
                weight=0.8,
                semantic_similarity=0.7,
            )
        
        # Semantic similarity relationship
        if chunk1.embedding is not None and chunk2.embedding is not None:
            similarity = self._calculate_cosine_similarity(chunk1.embedding, chunk2.embedding)
            
            if similarity > self.similarity_threshold:
                # Classify relationship type based on content
                rel_type = self._classify_relationship_type(chunk1, chunk2, similarity)
                
                return SimpleRelationship(
                    source_chunk=chunk1.chunk_id,
                    target_chunk=chunk2.chunk_id,
                    relationship_type=rel_type,
                    confidence=similarity,
                    weight=similarity * 0.8,
                    semantic_similarity=similarity,
                )
        
        # Keyword/topic overlap relationship
        if chunk1.keywords and chunk2.keywords:
            keyword_overlap = len(set(chunk1.keywords) & set(chunk2.keywords))
            total_keywords = len(set(chunk1.keywords) | set(chunk2.keywords))
            
            if total_keywords > 0:
                overlap_ratio = keyword_overlap / total_keywords
                
                if overlap_ratio > 0.3:  # 30% keyword overlap
                    return SimpleRelationship(
                        source_chunk=chunk1.chunk_id,
                        target_chunk=chunk2.chunk_id,
                        relationship_type="references",
                        confidence=overlap_ratio,
                        weight=overlap_ratio * 0.6,
                        semantic_similarity=overlap_ratio,
                    )
        
        return None
    
    async def _propagate_trust_scores(self) -> None:
        """Propagate trust scores through the relationship graph using Bayesian methods."""
        
        logger.info("Propagating trust scores through the graph...")
        
        if not self.graph:
            logger.warning("Graph not available, using simple trust propagation")
            return
        
        # Iterative trust propagation
        max_iterations = 10
        convergence_threshold = 0.001
        
        for iteration in range(max_iterations):
            previous_scores = {chunk_id: chunk.trust_score for chunk_id, chunk in self.chunks.items()}
            
            # Update trust scores based on relationships
            for chunk_id, chunk in self.chunks.items():
                if chunk_id not in self.graph:
                    continue
                
                # Get incoming trust from related chunks
                incoming_trust = 0.0
                total_weight = 0.0
                
                for predecessor in self.graph.predecessors(chunk_id):
                    if predecessor in self.chunks:
                        pred_chunk = self.chunks[predecessor]
                        edge_data = self.graph.get_edge_data(predecessor, chunk_id)
                        
                        weight = edge_data.get('weight', 0.5)
                        transferred_trust = pred_chunk.trust_score * weight * self.trust_decay
                        
                        incoming_trust += transferred_trust
                        total_weight += weight
                
                # Bayesian combination of base trust and propagated trust
                if total_weight > 0:
                    evidence_weight = min(total_weight, 1.0)
                    propagated_trust = incoming_trust / total_weight
                    
                    # Update trust score
                    original_trust = self.document_info[chunk.document_id]['base_credibility']
                    chunk.trust_score = (
                        original_trust * (1 - evidence_weight) +
                        propagated_trust * evidence_weight
                    )
            
            # Check for convergence
            max_change = max(
                abs(chunk.trust_score - previous_scores[chunk_id])
                for chunk_id, chunk in self.chunks.items()
            )
            
            if max_change < convergence_threshold:
                logger.info(f"Trust propagation converged after {iteration + 1} iterations")
                break
        
        logger.info("Trust propagation completed")
    
    async def _detect_knowledge_gaps(self) -> None:
        """Detect knowledge gaps in the graph."""
        
        logger.info("Detecting knowledge gaps...")
        
        gaps_detected = 0
        
        # Gap 1: Isolated chunks (no relationships)
        isolated_chunks = []
        for chunk_id, chunk in self.chunks.items():
            if chunk_id not in [rel.source_chunk for rel in self.relationships.values()] and \
               chunk_id not in [rel.target_chunk for rel in self.relationships.values()]:
                isolated_chunks.append(chunk_id)
        
        if isolated_chunks:
            gaps_detected += 1
            logger.info(f"Found {len(isolated_chunks)} isolated chunks")
        
        # Gap 2: Low coverage topics
        topic_coverage = defaultdict(list)
        for chunk in self.chunks.values():
            topic_coverage[chunk.topic_category].append(chunk.chunk_id)
        
        low_coverage_topics = [topic for topic, chunks in topic_coverage.items() if len(chunks) < 3]
        if low_coverage_topics:
            gaps_detected += len(low_coverage_topics)
            logger.info(f"Found {len(low_coverage_topics)} low coverage topics: {low_coverage_topics}")
        
        # Gap 3: Trust score distribution issues
        trust_scores = [chunk.trust_score for chunk in self.chunks.values()]
        if trust_scores:
            trust_std = np.std(trust_scores)
            if trust_std > 0.3:
                gaps_detected += 1
                logger.info(f"High trust score variance detected: {trust_std:.3f}")
        
        self.stats.knowledge_gaps_detected = gaps_detected
    
    async def _apply_basic_repairs(self) -> None:
        """Apply basic repairs to address simple knowledge gaps."""
        
        logger.info("Applying basic repairs...")
        
        repairs_applied = 0
        
        # Repair 1: Connect isolated chunks to most similar chunks
        isolated_chunks = []
        for chunk_id, chunk in self.chunks.items():
            has_relationships = any(
                chunk_id == rel.source_chunk or chunk_id == rel.target_chunk 
                for rel in self.relationships.values()
            )
            if not has_relationships:
                isolated_chunks.append(chunk)
        
        for isolated_chunk in isolated_chunks[:10]:  # Limit to avoid performance issues
            # Find most similar chunk
            best_similarity = 0.0
            best_match = None
            
            for other_chunk in self.chunks.values():
                if (isolated_chunk.chunk_id == other_chunk.chunk_id or
                    isolated_chunk.embedding is None or 
                    other_chunk.embedding is None):
                    continue
                
                similarity = self._calculate_cosine_similarity(
                    isolated_chunk.embedding, 
                    other_chunk.embedding
                )
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = other_chunk
            
            # Create relationship if good match found
            if best_match and best_similarity > 0.3:
                relationship = SimpleRelationship(
                    source_chunk=isolated_chunk.chunk_id,
                    target_chunk=best_match.chunk_id,
                    relationship_type="references",
                    confidence=best_similarity,
                    weight=best_similarity * 0.5,
                    semantic_similarity=best_similarity,
                )
                
                self.relationships[(isolated_chunk.chunk_id, best_match.chunk_id)] = relationship
                repairs_applied += 1
        
        # Repair 2: Boost quality scores for low coverage topics
        topic_coverage = defaultdict(list)
        for chunk in self.chunks.values():
            topic_coverage[chunk.topic_category].append(chunk)
        
        for topic, chunks in topic_coverage.items():
            if len(chunks) < 3:  # Low coverage topic
                for chunk in chunks:
                    chunk.quality_score = min(1.0, chunk.quality_score + 0.1)
                    chunk.trust_score = min(1.0, chunk.trust_score + 0.05)
                repairs_applied += 1
        
        self.stats.repairs_applied = repairs_applied
        logger.info(f"Applied {repairs_applied} basic repairs")
    
    def export_results(self, output_dir: str = "data/graph_rag_results") -> Dict[str, str]:
        """Export results to files."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        try:
            # Export statistics
            with open(output_path / "processing_stats.json", 'w') as f:
                json.dump(self.stats.to_dict(), f, indent=2)
            exported_files['stats'] = str(output_path / "processing_stats.json")
            
            # Export chunk data (sample)
            chunk_sample = {
                chunk_id: {
                    'document_id': chunk.document_id,
                    'position': chunk.position,
                    'text_length': len(chunk.text),
                    'text_preview': chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                    'keywords': chunk.keywords,
                    'entities': chunk.entities,
                    'topic_category': chunk.topic_category,
                    'coherence_score': chunk.coherence_score,
                    'trust_score': chunk.trust_score,
                    'quality_score': chunk.quality_score,
                }
                for chunk_id, chunk in list(self.chunks.items())[:100]  # First 100 chunks
            }
            
            with open(output_path / "chunk_sample.json", 'w') as f:
                json.dump(chunk_sample, f, indent=2)
            exported_files['chunks'] = str(output_path / "chunk_sample.json")
            
            # Export relationships
            relationships_data = {
                f"{rel.source_chunk}_{rel.target_chunk}": rel.to_dict()
                for rel in list(self.relationships.values())[:500]  # First 500 relationships
            }
            
            with open(output_path / "relationships.json", 'w') as f:
                json.dump(relationships_data, f, indent=2)
            exported_files['relationships'] = str(output_path / "relationships.json")
            
            # Export document summary
            document_summary = {
                'total_documents': len(self.document_info),
                'documents': {
                    doc_id: {
                        'title': info['title'],
                        'doc_type': info['doc_type'],
                        'base_credibility': info['base_credibility'],
                        'chunk_count': len([c for c in self.chunks.values() if c.document_id == doc_id]),
                    }
                    for doc_id, info in self.document_info.items()
                }
            }
            
            with open(output_path / "document_summary.json", 'w') as f:
                json.dump(document_summary, f, indent=2)
            exported_files['documents'] = str(output_path / "document_summary.json")
            
            # Export graph summary if available
            if self.graph:
                graph_summary = {
                    'nodes': len(self.graph.nodes()),
                    'edges': len(self.graph.edges()),
                    'density': nx.density(self.graph),
                    'is_connected': nx.is_weakly_connected(self.graph),
                    'connected_components': nx.number_weakly_connected_components(self.graph),
                }
                
                with open(output_path / "graph_summary.json", 'w') as f:
                    json.dump(graph_summary, f, indent=2)
                exported_files['graph'] = str(output_path / "graph_summary.json")
            
            logger.info(f"Results exported to {output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to export results: {e}")
        
        return exported_files
    
    def generate_report(self) -> str:
        """Generate a comprehensive report."""
        
        report_lines = [
            "SIMPLE GRAPH RAG CHUNKING REPORT",
            "=" * 50,
            "",
            f"Processing completed at: {time.ctime()}",
            f"Total processing time: {self.stats.processing_time:.1f}s",
            "",
            "DOCUMENT PROCESSING",
            "-" * 20,
            f"Documents processed: {self.stats.total_documents}",
            f"Total chunks created: {self.stats.total_chunks}",
            f"Average chunks per document: {self.stats.total_chunks / max(1, self.stats.total_documents):.1f}",
            "",
            "GRAPH CONSTRUCTION",
            "-" * 20,
            f"Relationships created: {self.stats.total_relationships}",
            f"Average relationships per chunk: {self.stats.total_relationships / max(1, self.stats.total_chunks):.2f}",
            "",
            "QUALITY METRICS",
            "-" * 15,
            f"Average chunk quality: {self.stats.avg_chunk_quality:.3f}",
            f"Average trust score: {self.stats.avg_trust_score:.3f}",
            "",
            "GAP ANALYSIS",
            "-" * 12,
            f"Knowledge gaps detected: {self.stats.knowledge_gaps_detected}",
            f"Basic repairs applied: {self.stats.repairs_applied}",
            "",
        ]
        
        # Add document type analysis
        doc_types = defaultdict(int)
        for info in self.document_info.values():
            doc_types[info['doc_type']] += 1
        
        if doc_types:
            report_lines.extend([
                "DOCUMENT TYPE DISTRIBUTION",
                "-" * 25,
            ])
            for doc_type, count in sorted(doc_types.items()):
                report_lines.append(f"{doc_type}: {count}")
            report_lines.append("")
        
        # Add topic analysis
        topic_dist = defaultdict(int)
        for chunk in self.chunks.values():
            topic_dist[chunk.topic_category] += 1
        
        if topic_dist:
            report_lines.extend([
                "TOPIC DISTRIBUTION",
                "-" * 18,
            ])
            for topic, count in sorted(topic_dist.items(), key=lambda x: x[1], reverse=True):
                report_lines.append(f"{topic}: {count} chunks")
            report_lines.append("")
        
        report_lines.extend([
            "=" * 50,
            "Processing completed successfully!",
        ])
        
        return "\n".join(report_lines)
    
    # Utility methods
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from text."""
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Remove common stop words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        }
        
        filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Get most frequent words
        word_counts = Counter(filtered_words)
        keywords = [word for word, count in word_counts.most_common(max_keywords)]
        
        return keywords
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract simple named entities from text."""
        # Very simple entity extraction - look for capitalized words
        entities = re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', text)
        
        # Remove common false positives
        false_positives = {'The', 'This', 'That', 'These', 'Those', 'And', 'But', 'Or', 'If', 'When'}
        entities = [e for e in entities if e not in false_positives]
        
        # Remove duplicates and limit
        return list(dict.fromkeys(entities))[:10]
    
    def _classify_topic(self, text: str, keywords: List[str]) -> str:
        """Classify the topic category of text."""
        text_lower = text.lower()
        
        # Topic classification based on keywords and content
        if any(term in text_lower for term in ['neural', 'network', 'model', 'training', 'learning']):
            return 'machine_learning'
        elif any(term in text_lower for term in ['calculus', 'mathematical', 'equation', 'theorem']):
            return 'mathematics'
        elif any(term in text_lower for term in ['agent', 'multi-agent', 'autonomous', 'behavior']):
            return 'agents'
        elif any(term in text_lower for term in ['graph', 'knowledge', 'semantic', 'ontology']):
            return 'knowledge_systems'
        elif any(term in text_lower for term in ['language', 'nlp', 'text', 'linguistic']):
            return 'natural_language'
        elif any(term in text_lower for term in ['compression', 'quantization', 'optimization']):
            return 'optimization'
        else:
            return 'general'
    
    def _calculate_coherence_score(self, text: str, keywords: List[str]) -> float:
        """Calculate coherence score for text."""
        if not keywords or not text:
            return 0.5
        
        # Simple coherence based on keyword density and text structure
        keyword_density = len(keywords) / max(len(text.split()), 1)
        
        # Check for structural elements
        has_structure = any(marker in text.lower() for marker in [
            'introduction', 'method', 'result', 'conclusion', 'abstract',
            'first', 'second', 'third', 'finally', 'however', 'therefore'
        ])
        
        structure_bonus = 0.1 if has_structure else 0.0
        
        # Base coherence with keyword density and structure
        coherence = min(1.0, 0.5 + keyword_density * 2 + structure_bonus)
        
        return coherence
    
    def _calculate_quality_score(self, text: str, coherence_score: float) -> float:
        """Calculate overall quality score."""
        if not text:
            return 0.0
        
        # Quality factors
        length_score = min(1.0, len(text) / 500)  # Prefer longer chunks up to 500 chars
        
        # Check for academic/technical indicators
        quality_indicators = [
            'research', 'study', 'analysis', 'method', 'result', 'conclusion',
            'algorithm', 'system', 'approach', 'technique', 'framework'
        ]
        
        indicator_count = sum(1 for indicator in quality_indicators if indicator in text.lower())
        indicator_score = min(1.0, indicator_count / 3)
        
        # Combine scores
        quality_score = (coherence_score * 0.4 + length_score * 0.3 + indicator_score * 0.3)
        
        return quality_score
    
    def _classify_document_type(self, content: str, doc_id: str) -> str:
        """Classify document type."""
        content_lower = content.lower()
        doc_id_lower = doc_id.lower()
        
        # Check for academic indicators
        if any(term in content_lower for term in ['abstract', 'introduction', 'methodology', 'arxiv']):
            return 'academic'
        
        # Check for mathematical content
        if 'grossman' in doc_id_lower or any(term in content_lower for term in ['calculus', 'theorem', 'mathematical']):
            return 'mathematical'
        
        # Check for technical content
        if any(term in content_lower for term in ['system', 'algorithm', 'implementation', 'framework']):
            return 'technical'
        
        return 'general'
    
    def _calculate_base_credibility(self, metadata: Dict[str, Any], doc_type: str, content: str) -> float:
        """Calculate base credibility score."""
        base_score = 0.7
        
        # Type-based adjustment
        type_scores = {
            'academic': 0.85,
            'mathematical': 0.9,
            'technical': 0.8,
            'general': 0.7,
        }
        
        base_score = type_scores.get(doc_type, base_score)
        
        # Metadata adjustments
        if metadata.get('author'):
            base_score += 0.05
        
        if metadata.get('title'):
            base_score += 0.03
        
        # Content quality indicators
        if any(term in content.lower() for term in ['peer-reviewed', 'published', 'journal']):
            base_score += 0.05
        
        return max(0.1, min(1.0, base_score))
    
    def _classify_relationship_type(
        self, 
        chunk1: SimpleChunk, 
        chunk2: SimpleChunk, 
        similarity: float
    ) -> str:
        """Classify the type of relationship between chunks."""
        
        # Same document relationships
        if chunk1.document_id == chunk2.document_id:
            if abs(chunk1.position - chunk2.position) == 1:
                return "continues"
            elif abs(chunk1.position - chunk2.position) <= 3:
                return "elaborates"
            else:
                return "references"
        
        # Cross-document relationships
        if chunk1.topic_category == chunk2.topic_category:
            return "related_topic"
        
        # High similarity suggests strong connection
        if similarity > 0.7:
            return "similar_content"
        
        return "references"
    
    def _calculate_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings."""
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        dot_product = np.dot(embedding1, embedding2)
        magnitude1 = np.linalg.norm(embedding1)
        magnitude2 = np.linalg.norm(embedding2)
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)


async def main():
    """Main function."""
    print("SIMPLE GRAPH RAG CHUNKING WITH BAYESIAN WEIGHTING")
    print("=" * 60)
    print("This will:")
    print("1. Process all ingested PDFs with intelligent chunking")
    print("2. Create semantic relationships between chunks")
    print("3. Apply Bayesian trust weighting") 
    print("4. Detect and repair knowledge gaps")
    print("5. Export comprehensive results")
    print()
    
    try:
        # Initialize chunker
        chunker = SimpleGraphRAGChunker(
            chunk_size=2000,
            chunk_overlap=200,
            similarity_threshold=0.4,
            trust_decay=0.85,
        )
        
        # Process all documents
        stats = await chunker.process_all_documents("data/ingested_papers")
        
        # Export results
        exported_files = chunker.export_results("data/graph_rag_results")
        
        # Generate report
        report = chunker.generate_report()
        
        # Save report
        report_file = Path("data/graph_rag_results/processing_report.txt")
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Display results
        print(report)
        
        print(f"\n✅ PROCESSING COMPLETED SUCCESSFULLY!")
        print(f"📊 Processed {stats.total_documents} documents")
        print(f"🧩 Created {stats.total_chunks} intelligent chunks")
        print(f"🔗 Built {stats.total_relationships} semantic relationships")
        print(f"🔍 Detected {stats.knowledge_gaps_detected} knowledge gaps")
        print(f"🔧 Applied {stats.repairs_applied} repairs")
        print(f"📁 Results exported: {exported_files}")
        print(f"📄 Report saved: {report_file}")
        
        return True
        
    except Exception as e:
        logger.exception(f"Processing failed: {e}")
        print(f"\n❌ Processing failed: {e}")
        return False


if __name__ == "__main__":
    asyncio.run(main())