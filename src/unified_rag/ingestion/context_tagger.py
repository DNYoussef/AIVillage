"""
Dual Context Tagger - Hierarchical Context Extraction

Advanced context extraction system that creates dual context tags
(book/chapter summaries) for enhanced retrieval precision.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import uuid

logger = logging.getLogger(__name__)


@dataclass
class ContextTag:
    """Context tag providing hierarchical information."""
    
    tag_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tag_type: str = ""  # "book", "chapter", "section", "topic"
    content: str = ""
    level: int = 0  # Hierarchy level (0=highest)
    confidence: float = 1.0
    
    # Source information
    source_text: str = ""
    extraction_method: str = "automatic"
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class DualContextTagger:
    """
    Dual Context Tagging System
    
    Extracts hierarchical context information from documents to create
    dual context tags that provide book-level and chapter-level summaries
    for enhanced retrieval precision and contextual understanding.
    
    Features:
    - Hierarchical context extraction (book/chapter/section)
    - Automatic summary generation
    - MCP integration for enhanced analysis
    - Context confidence scoring
    - Multi-level context relationships
    """
    
    def __init__(
        self,
        enable_dual_context: bool = True,
        context_levels: List[str] = None,
        mcp_coordinator = None,
        summary_length: int = 150
    ):
        self.enable_dual_context = enable_dual_context
        self.context_levels = context_levels or ["book", "chapter", "section"]
        self.mcp_coordinator = mcp_coordinator
        self.summary_length = summary_length
        
        # Context extraction patterns
        self.title_patterns = [
            re.compile(r'^#\s+(.+)$', re.MULTILINE),  # Markdown H1
            re.compile(r'^##\s+(.+)$', re.MULTILINE), # Markdown H2
            re.compile(r'^(.+)\n=+$', re.MULTILINE),  # Underlined titles
            re.compile(r'^(.+)\n-+$', re.MULTILINE),  # Underlined subtitles
            re.compile(r'^([A-Z][A-Za-z\s]*):?\s*$', re.MULTILINE)  # Capitalized lines
        ]
        
        self.section_patterns = [
            re.compile(r'(Chapter\s+\d+[:\.]?\s*[^\n]*)', re.IGNORECASE),
            re.compile(r'(Section\s+\d+[:\.]?\s*[^\n]*)', re.IGNORECASE),
            re.compile(r'(Part\s+\d+[:\.]?\s*[^\n]*)', re.IGNORECASE),
            re.compile(r'(\d+\.\s*[A-Z][^\n]*)', re.IGNORECASE)
        ]
        
        # Statistics
        self.stats = {
            "contexts_extracted": 0,
            "documents_processed": 0,
            "mcp_analyses": 0,
            "summary_generations": 0
        }
        
        self.initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the dual context tagger."""
        try:
            logger.info("🏷️ Initializing Dual Context Tagger...")
            self.initialized = True
            logger.info("✅ Dual Context Tagger ready")
            return True
        except Exception as e:
            logger.error(f"❌ Dual Context Tagger initialization failed: {e}")
            return False
    
    async def extract_contexts(self, content: str, document_id: str) -> List[ContextTag]:
        """Extract hierarchical context tags from document content."""
        if not self.initialized:
            raise RuntimeError("DualContextTagger not initialized")
        
        logger.debug(f"Extracting contexts from document {document_id}")
        
        try:
            contexts = []
            
            # Extract document-level context (book level)
            book_context = await self._extract_book_context(content, document_id)
            if book_context:
                contexts.append(book_context)
            
            # Extract chapter-level contexts
            if self.enable_dual_context:
                chapter_contexts = await self._extract_chapter_contexts(content, document_id)
                contexts.extend(chapter_contexts)
            
            # Extract section-level contexts if requested
            if "section" in self.context_levels:
                section_contexts = await self._extract_section_contexts(content, document_id)
                contexts.extend(section_contexts)
            
            # Enhance contexts with MCP analysis if available
            if self.mcp_coordinator:
                contexts = await self._enhance_contexts_with_mcp(contexts, content)
            
            # Post-process contexts
            contexts = await self._post_process_contexts(contexts)
            
            self.stats["contexts_extracted"] += len(contexts)
            self.stats["documents_processed"] += 1
            
            logger.debug(f"Extracted {len(contexts)} context tags for document {document_id}")
            return contexts
            
        except Exception as e:
            logger.error(f"Context extraction failed for document {document_id}: {e}")
            return []
    
    async def _extract_book_context(self, content: str, document_id: str) -> Optional[ContextTag]:
        """Extract book-level context (document summary)."""
        try:
            # Extract document title
            title = await self._extract_document_title(content)
            
            # Generate document summary
            summary = await self._generate_summary(content, self.summary_length)
            
            # Create book context
            book_context = ContextTag(
                tag_type="book",
                content=f"{title}: {summary}" if title else summary,
                level=0,
                confidence=0.9,
                source_text=content[:500],  # First 500 chars as source
                extraction_method="automatic_summary",
                metadata={
                    "document_id": document_id,
                    "title": title,
                    "summary_length": len(summary),
                    "content_length": len(content)
                }
            )
            
            return book_context
            
        except Exception as e:
            logger.warning(f"Book context extraction failed: {e}")
            return None
    
    async def _extract_chapter_contexts(self, content: str, document_id: str) -> List[ContextTag]:
        """Extract chapter-level contexts."""
        contexts = []
        
        try:
            # Find chapter boundaries
            chapters = await self._identify_chapters(content)
            
            for i, chapter in enumerate(chapters):
                chapter_title = chapter.get("title", f"Chapter {i + 1}")
                chapter_content = chapter.get("content", "")
                
                if len(chapter_content) > 50:  # Minimum content length
                    # Generate chapter summary
                    summary = await self._generate_summary(chapter_content, self.summary_length // 2)
                    
                    context = ContextTag(
                        tag_type="chapter",
                        content=f"{chapter_title}: {summary}",
                        level=1,
                        confidence=0.8,
                        source_text=chapter_content[:300],
                        extraction_method="chapter_detection",
                        metadata={
                            "document_id": document_id,
                            "chapter_index": i,
                            "chapter_title": chapter_title,
                            "chapter_length": len(chapter_content)
                        }
                    )
                    contexts.append(context)
            
            return contexts
            
        except Exception as e:
            logger.warning(f"Chapter context extraction failed: {e}")
            return []
    
    async def _extract_section_contexts(self, content: str, document_id: str) -> List[ContextTag]:
        """Extract section-level contexts."""
        contexts = []
        
        try:
            # Find sections using patterns
            sections = await self._identify_sections(content)
            
            for i, section in enumerate(sections):
                section_title = section.get("title", f"Section {i + 1}")
                section_content = section.get("content", "")
                
                if len(section_content) > 30:  # Minimum content length
                    # Generate section summary
                    summary = await self._generate_summary(section_content, self.summary_length // 3)
                    
                    context = ContextTag(
                        tag_type="section",
                        content=f"{section_title}: {summary}",
                        level=2,
                        confidence=0.7,
                        source_text=section_content[:200],
                        extraction_method="section_detection",
                        metadata={
                            "document_id": document_id,
                            "section_index": i,
                            "section_title": section_title,
                            "section_length": len(section_content)
                        }
                    )
                    contexts.append(context)
            
            return contexts
            
        except Exception as e:
            logger.warning(f"Section context extraction failed: {e}")
            return []
    
    async def _extract_document_title(self, content: str) -> Optional[str]:
        """Extract document title from content."""
        # Try each title pattern
        for pattern in self.title_patterns:
            matches = pattern.findall(content)
            if matches:
                # Return the first match that looks like a title
                for match in matches:
                    title = match.strip()
                    if len(title) > 5 and len(title) < 200:  # Reasonable title length
                        return title
        
        # Fallback: use first line if it's short enough
        first_line = content.split('\n')[0].strip()
        if len(first_line) > 5 and len(first_line) < 100:
            return first_line
        
        return None
    
    async def _identify_chapters(self, content: str) -> List[Dict[str, str]]:
        """Identify chapter boundaries and content."""
        chapters = []
        
        # Look for chapter patterns
        chapter_matches = []
        for pattern in self.section_patterns:
            for match in pattern.finditer(content):
                chapter_matches.append({
                    "title": match.group(1).strip(),
                    "start": match.start(),
                    "end": match.end()
                })
        
        # Sort by position
        chapter_matches.sort(key=lambda x: x["start"])
        
        # Extract chapter content
        for i, chapter_match in enumerate(chapter_matches):
            start_pos = chapter_match["end"]
            end_pos = chapter_matches[i + 1]["start"] if i + 1 < len(chapter_matches) else len(content)
            
            chapter_content = content[start_pos:end_pos].strip()
            
            chapters.append({
                "title": chapter_match["title"],
                "content": chapter_content,
                "start": start_pos,
                "end": end_pos
            })
        
        # If no chapters found, treat entire document as one chapter
        if not chapters:
            title = await self._extract_document_title(content)
            chapters.append({
                "title": title or "Main Content",
                "content": content,
                "start": 0,
                "end": len(content)
            })
        
        return chapters
    
    async def _identify_sections(self, content: str) -> List[Dict[str, str]]:
        """Identify section boundaries and content."""
        sections = []
        
        # Split by paragraph breaks and identify sections
        paragraphs = re.split(r'\n\s*\n', content)
        
        current_section = None
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if this paragraph is a section header
            is_header = False
            for pattern in self.title_patterns[1:]:  # Skip H1, use H2+ for sections
                if pattern.match(paragraph):
                    # End current section and start new one
                    if current_section:
                        sections.append(current_section)
                    
                    current_section = {
                        "title": paragraph,
                        "content": "",
                        "start": content.find(paragraph),
                        "end": 0
                    }
                    is_header = True
                    break
            
            if not is_header:
                if current_section:
                    current_section["content"] += paragraph + "\n\n"
                else:
                    # Start first section without explicit header
                    current_section = {
                        "title": "Introduction",
                        "content": paragraph + "\n\n",
                        "start": content.find(paragraph),
                        "end": 0
                    }
        
        # Add final section
        if current_section:
            current_section["end"] = len(content)
            sections.append(current_section)
        
        return sections
    
    async def _generate_summary(self, content: str, max_length: int) -> str:
        """Generate summary from content."""
        try:
            # Use MCP for enhanced summary generation if available
            if self.mcp_coordinator:
                breakdown = await self.mcp_coordinator.systematic_breakdown(
                    f"Summarize in {max_length} characters: {content[:1000]}"
                )
                
                if breakdown and isinstance(breakdown, dict):
                    summary_components = breakdown.get("main_components", [])
                    if summary_components:
                        summary = " ".join(summary_components[:5])  # Top 5 components
                        self.stats["mcp_analyses"] += 1
                        self.stats["summary_generations"] += 1
                        return summary[:max_length] + "..." if len(summary) > max_length else summary
            
            # Fallback: extractive summary
            return await self._extractive_summary(content, max_length)
            
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")
            return await self._extractive_summary(content, max_length)
    
    async def _extractive_summary(self, content: str, max_length: int) -> str:
        """Create extractive summary by selecting key sentences."""
        sentences = re.split(r'[.!?]+', content)
        
        # Score sentences by length and position (earlier sentences get higher scores)
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) > 10:  # Skip very short sentences
                # Simple scoring: longer sentences + position bonus
                score = len(sentence.split()) + (len(sentences) - i) * 0.1
                sentence_scores.append((sentence, score))
        
        # Sort by score and select top sentences
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        summary_sentences = []
        summary_length = 0
        
        for sentence, score in sentence_scores:
            if summary_length + len(sentence) <= max_length:
                summary_sentences.append(sentence)
                summary_length += len(sentence)
            else:
                break
        
        summary = ". ".join(summary_sentences)
        return summary[:max_length] + "..." if len(summary) > max_length else summary
    
    async def _enhance_contexts_with_mcp(self, contexts: List[ContextTag], content: str) -> List[ContextTag]:
        """Enhance contexts using MCP analysis."""
        try:
            if not self.mcp_coordinator:
                return contexts
            
            # Analyze content for enhanced understanding
            breakdown = await self.mcp_coordinator.systematic_breakdown(
                f"Analyze document structure and themes: {content[:800]}"
            )
            
            if breakdown and isinstance(breakdown, dict):
                # Enhance contexts with systematic analysis
                themes = breakdown.get("main_components", [])
                relationships = breakdown.get("relationships", [])
                
                for context in contexts:
                    context.metadata["systematic_themes"] = themes[:3]  # Top 3 themes
                    context.metadata["relationships"] = relationships[:2]  # Top 2 relationships
                    
                    # Boost confidence if themes align with context
                    theme_overlap = sum(1 for theme in themes if theme.lower() in context.content.lower())
                    if theme_overlap > 0:
                        context.confidence = min(1.0, context.confidence + 0.1 * theme_overlap)
                
                self.stats["mcp_analyses"] += 1
            
            return contexts
            
        except Exception as e:
            logger.warning(f"MCP context enhancement failed: {e}")
            return contexts
    
    async def _post_process_contexts(self, contexts: List[ContextTag]) -> List[ContextTag]:
        """Post-process contexts for quality and consistency."""
        if not contexts:
            return contexts
        
        # Sort by hierarchy level
        contexts.sort(key=lambda x: x.level)
        
        # Remove duplicate or very similar contexts
        unique_contexts = []
        for context in contexts:
            is_duplicate = False
            for existing in unique_contexts:
                if self._contexts_similar(context, existing):
                    # Keep the one with higher confidence
                    if context.confidence > existing.confidence:
                        unique_contexts.remove(existing)
                        unique_contexts.append(context)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_contexts.append(context)
        
        # Ensure minimum content quality
        quality_contexts = []
        for context in unique_contexts:
            if len(context.content.strip()) >= 10 and context.confidence >= 0.3:
                quality_contexts.append(context)
        
        return quality_contexts
    
    def _contexts_similar(self, context1: ContextTag, context2: ContextTag) -> bool:
        """Check if two contexts are similar."""
        # Similar if same type and high content overlap
        if context1.tag_type != context2.tag_type:
            return False
        
        words1 = set(context1.content.lower().split())
        words2 = set(context2.content.lower().split())
        
        if not words1 or not words2:
            return False
        
        overlap = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = overlap / union if union > 0 else 0
        return similarity > 0.7  # 70% similarity threshold
    
    async def get_tagger_status(self) -> Dict[str, Any]:
        """Get context tagger status."""
        return {
            "initialized": self.initialized,
            "configuration": {
                "enable_dual_context": self.enable_dual_context,
                "context_levels": self.context_levels,
                "summary_length": self.summary_length,
                "mcp_integration": self.mcp_coordinator is not None
            },
            "statistics": self.stats.copy()
        }
    
    async def close(self):
        """Close the dual context tagger."""
        logger.info("Closing Dual Context Tagger...")
        self.initialized = False