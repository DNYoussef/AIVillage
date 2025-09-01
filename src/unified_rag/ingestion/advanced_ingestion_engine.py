"""
Advanced Ingestion Engine - Semantic Processing with MCP Integration

Multi-format document processing with semantic chunking, dual context tagging,
and intelligent content analysis using Markitdown MCP for format conversion.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class DocumentFormat(Enum):
    """Supported document formats."""
    
    TEXT = "text"
    PDF = "pdf"
    DOCX = "docx"
    HTML = "html"
    MARKDOWN = "md"
    JSON = "json"
    XML = "xml"


class ChunkingStrategy(Enum):
    """Chunking strategies for different content types."""
    
    SEMANTIC = "semantic"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    FIXED_SIZE = "fixed_size"
    HIERARCHICAL = "hierarchical"


@dataclass
class ProcessingConfig:
    """Configuration for document processing."""
    
    chunk_size: int = 512
    chunk_overlap: int = 50
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC
    
    # Context tagging
    enable_dual_context: bool = True
    context_levels: List[str] = field(default_factory=lambda: ["book", "chapter"])
    
    # Processing options
    extract_metadata: bool = True
    preserve_structure: bool = True
    language_detection: bool = True
    
    # MCP options
    use_markitdown_mcp: bool = True
    validation_level: str = "comprehensive"


@dataclass
class ContextTag:
    """Context tag for dual context system."""
    
    tag_type: str
    content: str
    level: int = 0
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessedChunk:
    """A processed chunk with all metadata."""
    
    chunk_id: str
    content: str
    start_index: int
    end_index: int
    
    # Context information
    primary_context: Optional[ContextTag] = None
    secondary_context: Optional[ContextTag] = None
    additional_contexts: List[ContextTag] = field(default_factory=list)
    
    # Embeddings
    content_embedding: Optional[np.ndarray] = None
    context_embedding: Optional[np.ndarray] = None
    
    # Quality metrics
    semantic_coherence: float = 1.0
    information_density: float = 1.0
    
    # Metadata
    language: str = "en"
    document_id: str = ""
    chunk_index: int = 0
    total_chunks: int = 1
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessedDocument:
    """Fully processed document with all components."""
    
    document_id: str
    original_content: str
    processed_content: str
    document_format: DocumentFormat
    
    # Processing results
    chunks: List[ProcessedChunk] = field(default_factory=list)
    contexts: List[ContextTag] = field(default_factory=list)
    
    # Document-level metadata
    title: str = ""
    author: str = ""
    language: str = "en"
    word_count: int = 0
    
    # Processing metrics
    processing_time_ms: float = 0.0
    chunk_count: int = 0
    context_count: int = 0
    
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedIngestionEngine:
    """
    Advanced Document Ingestion Engine
    
    Processes documents with semantic understanding, context awareness,
    and multi-format support using MCP integration for format conversion
    and intelligent content analysis.
    
    Features:
    - Multi-format document processing (PDF, DOCX, HTML, etc.)
    - Semantic chunking with overlap handling
    - Dual context tagging (book/chapter summaries)
    - Intelligent metadata extraction
    - Language detection and processing
    - MCP integration for enhanced processing
    """
    
    def __init__(
        self,
        mcp_coordinator=None,
        config: Optional[ProcessingConfig] = None
    ):
        self.mcp_coordinator = mcp_coordinator
        self.config = config or ProcessingConfig()
        
        # Processing components
        self.semantic_chunker = None
        self.context_tagger = None
        self.metadata_extractor = None
        
        # Caching
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.processing_cache: Dict[str, ProcessedDocument] = {}
        
        # Statistics
        self.stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "contexts_extracted": 0,
            "mcp_calls": 0,
            "processing_time_ms": 0.0,
            "format_conversions": 0
        }
        
        self.initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the advanced ingestion engine."""
        try:
            logger.info("🔧 Initializing Advanced Ingestion Engine...")
            
            # Initialize processing components
            from .semantic_chunker import SemanticChunker
            from .context_tagger import DualContextTagger
            
            self.semantic_chunker = SemanticChunker(
                chunk_size=self.config.chunk_size,
                overlap=self.config.chunk_overlap,
                strategy=self.config.chunking_strategy
            )
            
            self.context_tagger = DualContextTagger(
                enable_dual_context=self.config.enable_dual_context,
                context_levels=self.config.context_levels,
                mcp_coordinator=self.mcp_coordinator
            )
            
            await self.semantic_chunker.initialize()
            await self.context_tagger.initialize()
            
            self.initialized = True
            logger.info("✅ Advanced Ingestion Engine initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Advanced Ingestion Engine initialization failed: {e}")
            return False
    
    async def process_document(
        self,
        content: str,
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_format: Optional[DocumentFormat] = None
    ) -> ProcessedDocument:
        """Process a document with full semantic analysis."""
        start_time = time.time()
        
        try:
            logger.info(f"📄 Processing document: {document_id}")
            
            # Auto-detect format if not provided
            if document_format is None:
                document_format = await self._detect_format(content, metadata)
            
            # Convert document format if needed
            processed_content = await self._convert_format(content, document_format)
            
            # Create processed document container
            doc = ProcessedDocument(
                document_id=document_id,
                original_content=content,
                processed_content=processed_content,
                document_format=document_format,
                metadata=metadata or {}
            )
            
            # Extract document-level metadata
            await self._extract_document_metadata(doc)
            
            # Semantic chunking
            chunks = await self.semantic_chunker.chunk_document(
                processed_content, document_id
            )
            
            # Context tagging
            if self.config.enable_dual_context:
                contexts = await self.context_tagger.extract_contexts(
                    processed_content, document_id
                )
                doc.contexts = contexts
                
                # Apply contexts to chunks
                for chunk in chunks:
                    chunk.primary_context = contexts[0] if contexts else None
                    chunk.secondary_context = contexts[1] if len(contexts) > 1 else None
            
            # Generate embeddings for chunks
            await self._generate_chunk_embeddings(chunks)
            
            # Calculate quality metrics
            await self._calculate_quality_metrics(chunks)
            
            # Finalize document
            doc.chunks = chunks
            doc.chunk_count = len(chunks)
            doc.context_count = len(doc.contexts)
            doc.processing_time_ms = (time.time() - start_time) * 1000
            
            # Update statistics
            self.stats["documents_processed"] += 1
            self.stats["chunks_created"] += len(chunks)
            self.stats["contexts_extracted"] += len(doc.contexts)
            self.stats["processing_time_ms"] += doc.processing_time_ms
            
            # Cache processed document
            self.processing_cache[document_id] = doc
            
            logger.info(f"✅ Document processed: {len(chunks)} chunks, "
                       f"{len(doc.contexts)} contexts, "
                       f"{doc.processing_time_ms:.1f}ms")
            
            return doc
            
        except Exception as e:
            logger.error(f"❌ Document processing failed: {e}")
            
            # Return minimal processed document
            return ProcessedDocument(
                document_id=document_id,
                original_content=content,
                processed_content=content,
                document_format=document_format or DocumentFormat.TEXT,
                processing_time_ms=(time.time() - start_time) * 1000,
                metadata={"error": str(e)}
            )
    
    async def process_batch(
        self,
        documents: List[Tuple[str, str, Optional[Dict[str, Any]]]],  # (content, doc_id, metadata)
        max_concurrent: int = 5
    ) -> List[ProcessedDocument]:
        """Process multiple documents in parallel."""
        logger.info(f"📚 Processing batch of {len(documents)} documents")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single(content, doc_id, metadata):
            async with semaphore:
                return await self.process_document(content, doc_id, metadata)
        
        tasks = [
            process_single(content, doc_id, metadata)
            for content, doc_id, metadata in documents
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return successful results
        successful_results = []
        for result in results:
            if not isinstance(result, Exception):
                successful_results.append(result)
            else:
                logger.error(f"Batch processing error: {result}")
        
        logger.info(f"✅ Batch processing complete: {len(successful_results)}/{len(documents)} successful")
        return successful_results
    
    async def _detect_format(self, content: str, metadata: Optional[Dict[str, Any]]) -> DocumentFormat:
        """Auto-detect document format."""
        if metadata and "format" in metadata:
            format_str = metadata["format"].lower()
            for fmt in DocumentFormat:
                if fmt.value == format_str:
                    return fmt
        
        # Simple content-based detection
        content_lower = content.strip().lower()
        
        if content_lower.startswith("<!doctype html") or content_lower.startswith("<html"):
            return DocumentFormat.HTML
        elif content_lower.startswith("{") or content_lower.startswith("["):
            return DocumentFormat.JSON
        elif content_lower.startswith("<?xml"):
            return DocumentFormat.XML
        elif "# " in content or "## " in content:
            return DocumentFormat.MARKDOWN
        else:
            return DocumentFormat.TEXT
    
    async def _convert_format(self, content: str, document_format: DocumentFormat) -> str:
        """Convert document format using MCP if available."""
        if document_format == DocumentFormat.TEXT:
            return content  # No conversion needed
        
        # Use Markitdown MCP for format conversion
        if self.mcp_coordinator and self.config.use_markitdown_mcp:
            try:
                result = await self.mcp_coordinator.process_document(
                    content, document_format.value
                )
                
                self.stats["mcp_calls"] += 1
                self.stats["format_conversions"] += 1
                
                return result.get("processed_content", content)
                
            except Exception as e:
                logger.warning(f"MCP format conversion failed: {e}")
        
        # Fallback format conversion
        return await self._fallback_format_conversion(content, document_format)
    
    async def _fallback_format_conversion(self, content: str, document_format: DocumentFormat) -> str:
        """Fallback format conversion without MCP."""
        if document_format == DocumentFormat.HTML:
            # Simple HTML tag removal
            import re
            clean_content = re.sub(r'<[^>]+>', '', content)
            return clean_content.strip()
        
        elif document_format == DocumentFormat.JSON:
            # Extract text values from JSON
            import json
            try:
                data = json.loads(content)
                text_values = []
                
                def extract_text(obj):
                    if isinstance(obj, str):
                        text_values.append(obj)
                    elif isinstance(obj, dict):
                        for value in obj.values():
                            extract_text(value)
                    elif isinstance(obj, list):
                        for item in obj:
                            extract_text(item)
                
                extract_text(data)
                return "\n".join(text_values)
                
            except json.JSONDecodeError:
                return content
        
        # For other formats, return as-is
        return content
    
    async def _extract_document_metadata(self, doc: ProcessedDocument):
        """Extract document-level metadata."""
        content = doc.processed_content
        
        # Extract title (first heading or first line)
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line.startswith('#') or len(line) < 100):
                doc.title = line.lstrip('#').strip()
                break
        
        # Word count
        doc.word_count = len(content.split())
        
        # Language detection (simplified)
        if self.config.language_detection:
            doc.language = await self._detect_language(content)
        
        # Additional metadata extraction using MCP if available
        if self.mcp_coordinator:
            try:
                # Use systematic breakdown for metadata extraction
                breakdown = await self.mcp_coordinator.systematic_breakdown(
                    f"Extract metadata from: {content[:500]}..."
                )
                
                if breakdown and isinstance(breakdown, dict):
                    doc.metadata.update({
                        "systematic_analysis": breakdown,
                        "complexity_score": len(breakdown.get("main_components", [])),
                        "relationship_types": breakdown.get("relationships", [])
                    })
                
            except Exception as e:
                logger.debug(f"MCP metadata extraction failed: {e}")
    
    async def _detect_language(self, content: str) -> str:
        """Simple language detection."""
        # Basic English detection (would use proper language detection in production)
        english_indicators = ["the", "and", "or", "in", "on", "at", "to", "for", "of", "with"]
        content_lower = content.lower()
        
        english_count = sum(1 for word in english_indicators if word in content_lower)
        
        return "en" if english_count >= 3 else "unknown"
    
    async def _generate_chunk_embeddings(self, chunks: List[ProcessedChunk]):
        """Generate embeddings for chunks using MCP."""
        if not chunks:
            return
        
        try:
            # Prepare texts for embedding
            texts = [chunk.content for chunk in chunks]
            
            # Generate embeddings using MCP
            if self.mcp_coordinator:
                embeddings = await self.mcp_coordinator.generate_embeddings(texts)
                
                # Assign embeddings to chunks
                for i, chunk in enumerate(chunks):
                    if i < len(embeddings):
                        chunk.content_embedding = embeddings[i]
                        
                        # Generate context embedding if contexts exist
                        if chunk.primary_context or chunk.secondary_context:
                            context_text = self._build_context_text(chunk)
                            context_embeddings = await self.mcp_coordinator.generate_embeddings([context_text])
                            if context_embeddings:
                                chunk.context_embedding = context_embeddings[0]
                
                self.stats["mcp_calls"] += 1
            
            else:
                # Fallback embedding generation
                for chunk in chunks:
                    chunk.content_embedding = await self._create_fallback_embedding(chunk.content)
                    
        except Exception as e:
            logger.warning(f"Chunk embedding generation failed: {e}")
            
            # Create fallback embeddings
            for chunk in chunks:
                chunk.content_embedding = await self._create_fallback_embedding(chunk.content)
    
    def _build_context_text(self, chunk: ProcessedChunk) -> str:
        """Build context text from chunk contexts."""
        context_parts = []
        
        if chunk.primary_context:
            context_parts.append(f"[{chunk.primary_context.tag_type}] {chunk.primary_context.content}")
        
        if chunk.secondary_context:
            context_parts.append(f"[{chunk.secondary_context.tag_type}] {chunk.secondary_context.content}")
        
        for context in chunk.additional_contexts:
            context_parts.append(f"[{context.tag_type}] {context.content}")
        
        return " | ".join(context_parts)
    
    async def _create_fallback_embedding(self, text: str) -> np.ndarray:
        """Create fallback embedding using deterministic hashing."""
        # Check cache first
        text_hash = str(hash(text))
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        
        # Create deterministic pseudo-embedding
        import hashlib
        text_bytes = text.encode("utf-8")
        hash_bytes = hashlib.md5(text_bytes, usedforsecurity=False).hexdigest()
        seed = int(hash_bytes[:8], 16)
        
        np.random.seed(seed)
        embedding = np.random.normal(0, 1, 768).astype(np.float32)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        # Cache result
        self.embedding_cache[text_hash] = embedding
        return embedding
    
    async def _calculate_quality_metrics(self, chunks: List[ProcessedChunk]):
        """Calculate quality metrics for chunks."""
        for chunk in chunks:
            # Semantic coherence (simplified)
            sentences = chunk.content.split('.')
            chunk.semantic_coherence = min(1.0, len(sentences) / 10)  # More sentences = potentially more coherent
            
            # Information density (words per character ratio)
            words = len(chunk.content.split())
            chars = len(chunk.content)
            chunk.information_density = min(1.0, words / max(1, chars / 6))  # ~6 chars per word average
    
    async def get_processing_status(self) -> Dict[str, Any]:
        """Get processing engine status."""
        return {
            "initialized": self.initialized,
            "configuration": {
                "chunk_size": self.config.chunk_size,
                "chunking_strategy": self.config.chunking_strategy.value,
                "dual_context_enabled": self.config.enable_dual_context,
                "mcp_integration": self.config.use_markitdown_mcp
            },
            "statistics": self.stats.copy(),
            "cache_size": {
                "embeddings": len(self.embedding_cache),
                "documents": len(self.processing_cache)
            }
        }
    
    async def close(self):
        """Close the ingestion engine and clean up resources."""
        logger.info("Shutting down Advanced Ingestion Engine...")
        
        # Clear caches
        self.embedding_cache.clear()
        self.processing_cache.clear()
        
        # Close components
        if self.semantic_chunker:
            await self.semantic_chunker.close()
        
        if self.context_tagger:
            await self.context_tagger.close()
        
        self.initialized = False
        logger.info("Advanced Ingestion Engine shutdown complete")