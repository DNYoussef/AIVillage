"""
Semantic Chunker - Intelligent Document Segmentation

Advanced chunking strategies with semantic understanding,
overlap handling, and hierarchical structure preservation.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple
import uuid

logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    """Available chunking strategies."""
    
    SEMANTIC = "semantic"
    PARAGRAPH = "paragraph" 
    SENTENCE = "sentence"
    FIXED_SIZE = "fixed_size"
    HIERARCHICAL = "hierarchical"


@dataclass
class SemanticChunk:
    """A semantically coherent chunk of content."""
    
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    start_index: int = 0
    end_index: int = 0
    
    # Semantic properties
    topic_coherence: float = 1.0
    structural_level: int = 0  # 0=top level, 1=section, 2=paragraph, etc.
    
    # Chunk metadata
    document_id: str = ""
    chunk_index: int = 0
    total_chunks: int = 1
    
    # Overlap information
    overlaps_previous: bool = False
    overlaps_next: bool = False
    overlap_size: int = 0


class SemanticChunker:
    """
    Advanced Semantic Chunker
    
    Intelligent document segmentation that preserves semantic coherence
    while maintaining optimal chunk sizes for retrieval and processing.
    
    Features:
    - Multiple chunking strategies
    - Semantic boundary detection
    - Overlap handling for context preservation
    - Hierarchical structure awareness
    - Topic coherence scoring
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 50,
        strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1024
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strategy = strategy
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        # Semantic analysis patterns
        self.sentence_endings = re.compile(r'[.!?]+\s+')
        self.paragraph_breaks = re.compile(r'\n\s*\n')
        self.section_headers = re.compile(r'^#+\s+|^[A-Z][A-Za-z\s]*:?\s*$', re.MULTILINE)
        
        # Statistics
        self.stats = {
            "chunks_created": 0,
            "documents_processed": 0,
            "avg_chunk_size": 0,
            "semantic_boundaries_detected": 0
        }
        
        self.initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the semantic chunker."""
        try:
            logger.info("🧩 Initializing Semantic Chunker...")
            self.initialized = True
            logger.info("✅ Semantic Chunker ready")
            return True
        except Exception as e:
            logger.error(f"❌ Semantic Chunker initialization failed: {e}")
            return False
    
    async def chunk_document(self, content: str, document_id: str) -> List[SemanticChunk]:
        """Chunk document using the configured strategy."""
        if not self.initialized:
            raise RuntimeError("SemanticChunker not initialized")
        
        logger.debug(f"Chunking document {document_id} using {self.strategy.value} strategy")
        
        try:
            if self.strategy == ChunkingStrategy.SEMANTIC:
                chunks = await self._semantic_chunking(content, document_id)
            elif self.strategy == ChunkingStrategy.PARAGRAPH:
                chunks = await self._paragraph_chunking(content, document_id)
            elif self.strategy == ChunkingStrategy.SENTENCE:
                chunks = await self._sentence_chunking(content, document_id)
            elif self.strategy == ChunkingStrategy.HIERARCHICAL:
                chunks = await self._hierarchical_chunking(content, document_id)
            else:  # FIXED_SIZE
                chunks = await self._fixed_size_chunking(content, document_id)
            
            # Post-process chunks
            chunks = await self._post_process_chunks(chunks)
            
            # Update statistics
            self.stats["chunks_created"] += len(chunks)
            self.stats["documents_processed"] += 1
            if chunks:
                avg_size = sum(len(chunk.content) for chunk in chunks) / len(chunks)
                self.stats["avg_chunk_size"] = avg_size
            
            logger.debug(f"Created {len(chunks)} chunks for document {document_id}")
            return chunks
            
        except Exception as e:
            logger.error(f"Chunking failed for document {document_id}: {e}")
            # Fallback to simple chunking
            return await self._fixed_size_chunking(content, document_id)
    
    async def _semantic_chunking(self, content: str, document_id: str) -> List[SemanticChunk]:
        """Advanced semantic chunking with boundary detection."""
        chunks = []
        
        # Find semantic boundaries
        boundaries = await self._detect_semantic_boundaries(content)
        boundaries = [0] + boundaries + [len(content)]
        
        current_chunk = ""
        start_idx = 0
        chunk_idx = 0
        
        for i in range(1, len(boundaries)):
            boundary_start = boundaries[i-1]
            boundary_end = boundaries[i]
            
            segment = content[boundary_start:boundary_end].strip()
            
            # Check if adding this segment would exceed chunk size
            potential_chunk = current_chunk + "\n" + segment if current_chunk else segment
            
            if len(potential_chunk) <= self.chunk_size or not current_chunk:
                current_chunk = potential_chunk
            else:
                # Create chunk from current content
                if current_chunk:
                    chunk = SemanticChunk(
                        content=current_chunk,
                        start_index=start_idx,
                        end_index=start_idx + len(current_chunk),
                        document_id=document_id,
                        chunk_index=chunk_idx,
                        structural_level=self._determine_structural_level(current_chunk)
                    )
                    chunks.append(chunk)
                    chunk_idx += 1
                
                # Start new chunk with overlap
                overlap_text = self._extract_overlap(current_chunk, self.overlap)
                current_chunk = overlap_text + "\n" + segment if overlap_text else segment
                start_idx = boundary_start - len(overlap_text) if overlap_text else boundary_start
        
        # Add final chunk
        if current_chunk:
            chunk = SemanticChunk(
                content=current_chunk,
                start_index=start_idx,
                end_index=len(content),
                document_id=document_id,
                chunk_index=chunk_idx,
                structural_level=self._determine_structural_level(current_chunk)
            )
            chunks.append(chunk)
        
        self.stats["semantic_boundaries_detected"] += len(boundaries) - 2
        return chunks
    
    async def _detect_semantic_boundaries(self, content: str) -> List[int]:
        """Detect semantic boundaries in content."""
        boundaries = []
        
        # Find paragraph boundaries
        for match in self.paragraph_breaks.finditer(content):
            boundaries.append(match.start())
        
        # Find section header boundaries
        for match in self.section_headers.finditer(content):
            boundaries.append(match.start())
        
        # Find sentence boundaries in long paragraphs
        paragraphs = self.paragraph_breaks.split(content)
        current_pos = 0
        
        for paragraph in paragraphs:
            if len(paragraph) > self.chunk_size:
                # Find sentence boundaries within long paragraphs
                sentences = self.sentence_endings.split(paragraph)
                sentence_pos = current_pos
                
                for sentence in sentences[:-1]:  # Skip last empty split
                    sentence_pos += len(sentence)
                    boundaries.append(sentence_pos)
            
            current_pos += len(paragraph) + 2  # +2 for paragraph break
        
        # Remove duplicates and sort
        boundaries = sorted(list(set(boundaries)))
        return boundaries
    
    async def _paragraph_chunking(self, content: str, document_id: str) -> List[SemanticChunk]:
        """Paragraph-based chunking."""
        paragraphs = self.paragraph_breaks.split(content)
        chunks = []
        
        current_chunk = ""
        start_idx = 0
        chunk_idx = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            potential_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            
            if len(potential_chunk) <= self.chunk_size or not current_chunk:
                current_chunk = potential_chunk
            else:
                # Create chunk
                if current_chunk:
                    chunk = SemanticChunk(
                        content=current_chunk,
                        start_index=start_idx,
                        end_index=start_idx + len(current_chunk),
                        document_id=document_id,
                        chunk_index=chunk_idx,
                        structural_level=1  # Paragraph level
                    )
                    chunks.append(chunk)
                    chunk_idx += 1
                
                # Start new chunk
                overlap_text = self._extract_overlap(current_chunk, self.overlap)
                current_chunk = overlap_text + "\n\n" + paragraph if overlap_text else paragraph
                start_idx = start_idx + len(current_chunk) - len(paragraph) - len(overlap_text)
        
        # Add final chunk
        if current_chunk:
            chunk = SemanticChunk(
                content=current_chunk,
                start_index=start_idx,
                end_index=len(content),
                document_id=document_id,
                chunk_index=chunk_idx,
                structural_level=1
            )
            chunks.append(chunk)
        
        return chunks
    
    async def _sentence_chunking(self, content: str, document_id: str) -> List[SemanticChunk]:
        """Sentence-based chunking."""
        sentences = self.sentence_endings.split(content)
        chunks = []
        
        current_chunk = ""
        start_idx = 0
        chunk_idx = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) <= self.chunk_size or not current_chunk:
                current_chunk = potential_chunk
            else:
                # Create chunk
                if current_chunk:
                    chunk = SemanticChunk(
                        content=current_chunk,
                        start_index=start_idx,
                        end_index=start_idx + len(current_chunk),
                        document_id=document_id,
                        chunk_index=chunk_idx,
                        structural_level=2  # Sentence level
                    )
                    chunks.append(chunk)
                    chunk_idx += 1
                
                # Start new chunk with overlap
                overlap_text = self._extract_overlap(current_chunk, self.overlap)
                current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                start_idx = start_idx + len(current_chunk) - len(sentence) - len(overlap_text)
        
        # Add final chunk
        if current_chunk:
            chunk = SemanticChunk(
                content=current_chunk,
                start_index=start_idx,
                end_index=len(content),
                document_id=document_id,
                chunk_index=chunk_idx,
                structural_level=2
            )
            chunks.append(chunk)
        
        return chunks
    
    async def _hierarchical_chunking(self, content: str, document_id: str) -> List[SemanticChunk]:
        """Hierarchical chunking respecting document structure."""
        chunks = []
        
        # Find hierarchical boundaries (headers, sections, etc.)
        sections = self._extract_hierarchical_sections(content)
        
        chunk_idx = 0
        for section in sections:
            section_content = section["content"]
            section_level = section["level"]
            
            # Chunk section content if it's too large
            if len(section_content) > self.chunk_size:
                sub_chunks = await self._semantic_chunking(section_content, document_id)
                for sub_chunk in sub_chunks:
                    sub_chunk.structural_level = section_level
                    sub_chunk.chunk_index = chunk_idx
                    chunks.append(sub_chunk)
                    chunk_idx += 1
            else:
                # Single chunk for small sections
                chunk = SemanticChunk(
                    content=section_content,
                    start_index=section["start"],
                    end_index=section["end"],
                    document_id=document_id,
                    chunk_index=chunk_idx,
                    structural_level=section_level
                )
                chunks.append(chunk)
                chunk_idx += 1
        
        return chunks
    
    def _extract_hierarchical_sections(self, content: str) -> List[dict]:
        """Extract hierarchical sections from content."""
        sections = []
        
        # Find headers
        header_matches = list(self.section_headers.finditer(content))
        
        if not header_matches:
            # No headers found, treat as single section
            return [{
                "content": content,
                "start": 0,
                "end": len(content),
                "level": 0,
                "title": "Document"
            }]
        
        # Create sections between headers
        for i, match in enumerate(header_matches):
            start_pos = match.start()
            end_pos = header_matches[i + 1].start() if i + 1 < len(header_matches) else len(content)
            
            section_content = content[start_pos:end_pos].strip()
            header_text = match.group().strip()
            
            # Determine header level
            level = 0
            if header_text.startswith("#"):
                level = len(header_text) - len(header_text.lstrip("#"))
            
            sections.append({
                "content": section_content,
                "start": start_pos,
                "end": end_pos,
                "level": level,
                "title": header_text
            })
        
        return sections
    
    async def _fixed_size_chunking(self, content: str, document_id: str) -> List[SemanticChunk]:
        """Simple fixed-size chunking with overlap."""
        chunks = []
        chunk_idx = 0
        
        i = 0
        while i < len(content):
            end_idx = min(i + self.chunk_size, len(content))
            chunk_content = content[i:end_idx]
            
            chunk = SemanticChunk(
                content=chunk_content,
                start_index=i,
                end_index=end_idx,
                document_id=document_id,
                chunk_index=chunk_idx,
                structural_level=0  # No structural awareness
            )
            chunks.append(chunk)
            
            # Move forward with overlap
            i += self.chunk_size - self.overlap
            chunk_idx += 1
        
        return chunks
    
    def _extract_overlap(self, content: str, overlap_size: int) -> str:
        """Extract overlap text from the end of content."""
        if len(content) <= overlap_size:
            return content
        
        # Try to find a good break point (sentence or word boundary)
        overlap_text = content[-overlap_size:]
        
        # Find the last sentence boundary
        sentence_match = None
        for match in self.sentence_endings.finditer(overlap_text):
            sentence_match = match
        
        if sentence_match:
            # Use text after last sentence boundary
            return overlap_text[sentence_match.end():].strip()
        
        # Fallback to word boundary
        words = overlap_text.split()
        if len(words) > 1:
            return " ".join(words[1:])  # Skip first partial word
        
        return overlap_text
    
    def _determine_structural_level(self, content: str) -> int:
        """Determine the structural level of content."""
        content_stripped = content.strip()
        
        # Check for headers
        if re.match(r'^#+\s+', content_stripped):
            header_match = re.match(r'^(#+)', content_stripped)
            if header_match:
                return len(header_match.group(1))
        
        # Check for section indicators
        if re.match(r'^[A-Z][A-Za-z\s]*:?\s*$', content_stripped.split('\n')[0]):
            return 1
        
        # Default to paragraph level
        return 2
    
    async def _post_process_chunks(self, chunks: List[SemanticChunk]) -> List[SemanticChunk]:
        """Post-process chunks for quality and consistency."""
        if not chunks:
            return chunks
        
        # Set total chunks count
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total_chunks
        
        # Calculate topic coherence (simplified)
        for chunk in chunks:
            chunk.topic_coherence = self._calculate_topic_coherence(chunk.content)
        
        # Set overlap flags
        for i, chunk in enumerate(chunks):
            if i > 0:
                chunk.overlaps_previous = self._has_overlap(chunks[i-1].content, chunk.content)
            if i < len(chunks) - 1:
                chunk.overlaps_next = self._has_overlap(chunk.content, chunks[i+1].content)
        
        # Filter out chunks that are too small
        filtered_chunks = []
        for chunk in chunks:
            if len(chunk.content.strip()) >= self.min_chunk_size:
                filtered_chunks.append(chunk)
            else:
                # Merge with next chunk if possible
                if filtered_chunks:
                    filtered_chunks[-1].content += "\n" + chunk.content
                    filtered_chunks[-1].end_index = chunk.end_index
        
        return filtered_chunks
    
    def _calculate_topic_coherence(self, content: str) -> float:
        """Calculate topic coherence score (simplified)."""
        # Simple coherence based on sentence count and repetition
        sentences = self.sentence_endings.split(content)
        
        if len(sentences) <= 1:
            return 1.0
        
        # Check for repeated words (indicator of coherence)
        words = content.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Skip short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        repeated_words = sum(1 for count in word_freq.values() if count > 1)
        coherence = min(1.0, repeated_words / max(1, len(word_freq)))
        
        return coherence
    
    def _has_overlap(self, content1: str, content2: str) -> bool:
        """Check if two content pieces have overlap."""
        # Simple overlap detection
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        overlap = len(words1.intersection(words2))
        return overlap > 3  # Threshold for meaningful overlap
    
    async def get_chunker_stats(self) -> dict:
        """Get chunker statistics."""
        return {
            "initialized": self.initialized,
            "configuration": {
                "chunk_size": self.chunk_size,
                "overlap": self.overlap,
                "strategy": self.strategy.value,
                "min_chunk_size": self.min_chunk_size,
                "max_chunk_size": self.max_chunk_size
            },
            "statistics": self.stats.copy()
        }
    
    async def close(self):
        """Close the semantic chunker."""
        logger.info("Closing Semantic Chunker...")
        self.initialized = False